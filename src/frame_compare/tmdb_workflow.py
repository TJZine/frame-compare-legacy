"""TMDB workflow orchestration helpers."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Tuple, cast

import click
import httpx

from src.datatypes import TMDBConfig
from src.frame_compare.layout_utils import sanitize_console_text
from src.frame_compare.metadata import first_non_empty, parse_year_hint
from src.tmdb import (
    TMDBAmbiguityError,
    TMDBCandidate,
    TMDBResolution,
    TMDBResolutionError,
    parse_manual_id,
    resolve_tmdb,
)

if TYPE_CHECKING:
    from typing import Protocol

    class _ManualPrompt(Protocol):
        def __call__(self, candidates: Sequence[TMDBCandidate]) -> Optional[Tuple[str, str]]: ...

    class _ConfirmationPrompt(Protocol):
        def __call__(self, resolution: TMDBResolution) -> Tuple[bool, Optional[Tuple[str, str]]]: ...

__all__ = [
    "TMDBLookupResult",
    "resolve_blocking",
    "resolve_workflow",
    "render_collection_name",
]


@dataclass
class TMDBLookupResult:
    """Outcome of the TMDB workflow (resolution, manual overrides, or failure)."""

    resolution: TMDBResolution | None
    manual_override: Optional[Tuple[str, str]]
    error_message: Optional[str]
    ambiguous: bool


StyleFunc = Callable[..., str]


def _style(text: str, **kwargs: Any) -> str:
    style_fn = getattr(click, "style", None)
    if not callable(style_fn):
        return text
    style_callable = cast(StyleFunc, style_fn)
    return style_callable(text, **kwargs)


def render_collection_name(template_text: str, context: Mapping[str, Any]) -> str:
    """Render the configured TMDB collection template with *context* values."""
    if "${" not in template_text:
        return template_text
    try:
        template = Template(template_text)
        return template.safe_substitute(context)
    except (TypeError, ValueError, KeyError):
        return template_text


def resolve_blocking(
    *,
    file_name: str,
    tmdb_cfg: TMDBConfig,
    year_hint: Optional[int],
    imdb_id: Optional[str],
    tvdb_id: Optional[str],
    attempts: int = 3,
    transport_retries: int = 2,
) -> TMDBResolution | None:
    """Resolve TMDB metadata even when the caller already owns an event loop."""

    max_attempts = max(1, attempts)
    backoff = 0.75
    for attempt in range(max_attempts):
        transport_cls = getattr(httpx, "AsyncHTTPTransport", None)
        if transport_cls is None:
            raise RuntimeError("httpx.AsyncHTTPTransport is unavailable in this environment")
        transport: Any = transport_cls(retries=max(0, transport_retries))

        async def _make_coro(transport: Any = transport) -> TMDBResolution | None:
            return await resolve_tmdb(
                file_name,
                config=tmdb_cfg,
                year=year_hint,
                imdb_id=imdb_id,
                tvdb_id=tvdb_id,
                unattended=tmdb_cfg.unattended,
                category_preference=tmdb_cfg.category_preference,
                http_transport=transport,
            )

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_make_coro())

            result_holder: list[TMDBResolution | None] = []
            error_holder: list[BaseException] = []

            def _worker(
                result_holder: list[TMDBResolution | None] = result_holder,
                error_holder: list[BaseException] = error_holder,
            ) -> None:
                try:
                    result_holder.append(asyncio.run(_make_coro()))
                except BaseException as exc:  # noqa: BLE001
                    # Captured to bubble up when joined
                    error_holder.append(exc)

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            thread.join()
            if error_holder:
                raise error_holder[0]
            return result_holder[0] if result_holder else None
        except TMDBResolutionError as exc:
            message = str(exc)
            if attempt + 1 >= max_attempts or not _should_retry_tmdb_error(message):
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 4.0)
        finally:
            close_fn = getattr(transport, "close", None)
            if callable(close_fn):
                close_fn()
    return None


def resolve_workflow(
    *,
    files: Sequence[Path],
    metadata: Sequence[Mapping[str, str]],
    tmdb_cfg: TMDBConfig,
    year_hint_raw: Optional[str] = None,
) -> TMDBLookupResult:
    """Resolve TMDB metadata for the current comparison set, prompting when needed."""

    if not files or not tmdb_cfg.api_key.strip():
        return TMDBLookupResult(
            resolution=None,
            manual_override=None,
            error_message=None,
            ambiguous=False,
        )

    base_file = files[0]
    imdb_hint_raw = first_non_empty(metadata, "imdb_id")
    imdb_hint = imdb_hint_raw.lower() if imdb_hint_raw else None
    tvdb_hint = first_non_empty(metadata, "tvdb_id") or None
    effective_year_hint = year_hint_raw or first_non_empty(metadata, "year")
    year_hint = parse_year_hint(effective_year_hint)

    resolution: TMDBResolution | None = None
    manual_tmdb: Optional[Tuple[str, str]] = None
    error_message: Optional[str] = None
    ambiguous = False

    try:
        resolution = resolve_blocking(
            file_name=base_file.name,
            tmdb_cfg=tmdb_cfg,
            year_hint=year_hint,
            imdb_id=imdb_hint,
            tvdb_id=tvdb_hint,
        )
    except TMDBAmbiguityError as exc:
        ambiguous = True
        if tmdb_cfg.unattended:
            error_message = (
                "TMDB returned multiple matches but unattended mode prevented prompts."
            )
        else:
            manual_tmdb = _prompt_manual_tmdb(exc.candidates)
    except TMDBResolutionError as exc:
        error_message = str(exc)
    else:
        if resolution is not None and tmdb_cfg.confirm_matches and not tmdb_cfg.unattended:
            accepted, override = _prompt_tmdb_confirmation(resolution)
            if override:
                manual_tmdb = override
                resolution = None
            elif not accepted:
                resolution = None

    return TMDBLookupResult(
        resolution=resolution,
        manual_override=manual_tmdb,
        error_message=error_message,
        ambiguous=ambiguous,
    )


def _should_retry_tmdb_error(message: str) -> bool:
    """Return True when *message* indicates a transient TMDB/HTTP failure."""

    lowered = message.lower()
    transient_markers = (
        "request failed",
        "timeout",
        "temporarily",
        "connection",
        "503",
        "502",
        "504",
        "429",
    )
    return any(marker in lowered for marker in transient_markers)


def _prompt_manual_tmdb(candidates: Sequence[TMDBCandidate]) -> Optional[Tuple[str, str]]:
    """Prompt the user to choose a TMDB candidate when multiple matches exist."""
    click.echo(_style("TMDB search returned multiple plausible matches:", fg="yellow"))
    for cand in candidates:
        year = cand.year or "????"
        category = _style(f"{cand.category.lower()}/{cand.tmdb_id}", fg="cyan")
        title = cand.title or "(unknown title)"
        safe_title = sanitize_console_text(title)
        click.echo(f"  • {category} {safe_title} ({year}) score={cand.score:0.3f}")
    while True:
        response = click.prompt(
            "Enter TMDB id (movie/##### or tv/#####) or leave blank to skip",
            default="",
            show_default=False,
        ).strip()
        if not response:
            return None
        try:
            return parse_manual_id(response)
        except TMDBResolutionError as exc:
            click.echo(f"{_style('Invalid TMDB identifier:', fg='red')} {exc}")


def _prompt_tmdb_confirmation(
    resolution: TMDBResolution,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """Ask the user to confirm the TMDB result or supply a manual override."""
    title = resolution.title or resolution.original_title or "(unknown title)"
    year = resolution.year or "????"
    category = resolution.category.lower()
    link = f"https://www.themoviedb.org/{category}/{resolution.tmdb_id}"
    header = _style("TMDB match found:", fg="cyan")
    link_text = _style(link, underline=True)
    safe_title = sanitize_console_text(title)
    click.echo(f"{header} {safe_title} ({year}) -> {link_text}")
    while True:
        response = click.prompt(
            "Confirm TMDB match? [Y/n or enter movie/#####]",
            default="y",
            show_default=False,
        ).strip()
        if not response or response.lower() in {"y", "yes"}:
            return True, None
        if response.lower() in {"n", "no"}:
            return False, None
        try:
            manual = parse_manual_id(response)
        except TMDBResolutionError as exc:
            click.echo(f"{_style('Invalid TMDB identifier:', fg='red')} {exc}")
        else:
            return True, manual


if TYPE_CHECKING:
    _manual_prompt: _ManualPrompt = _prompt_manual_tmdb
    _confirmation_prompt: _ConfirmationPrompt = _prompt_tmdb_confirmation
