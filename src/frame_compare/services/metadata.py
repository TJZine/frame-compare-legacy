"""MetadataResolver service that encapsulates TMDB + plan preparation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, MutableMapping, Protocol, Sequence, cast

from rich.markup import escape

import src.frame_compare.metadata as metadata_utils
import src.frame_compare.tmdb_workflow as tmdb_workflow
from src.datatypes import AppConfig, RuntimeConfig
from src.frame_compare.cli_runtime import (
    CLIAppError,
    CliOutputManagerProtocol,
    ClipPlan,
    JsonTail,
    SlowpicsTitleInputs,
)
from src.frame_compare.layout_utils import (
    color_text as _color_text,
)
from src.frame_compare.layout_utils import (
    format_kv as _format_kv,
)
from src.frame_compare.layout_utils import (
    sanitize_console_text as _sanitize_console_text,
)
from src.frame_compare.render.naming import (
    SAFE_LABEL_META_KEY,
)
from src.frame_compare.render.naming import (
    sanitise_label as _render_sanitise_label,
)
from src.frame_compare.tmdb_workflow import TMDBLookupResult
from src.frame_compare.vs import ClipInitError
from src.tmdb import TMDBResolution

logger = logging.getLogger(__name__)

__all__ = [
    "CliPromptProtocol",
    "FilesystemProbeProtocol",
    "MetadataResolveRequest",
    "MetadataResolveResult",
    "MetadataResolver",
    "PlanBuilder",
    "TMDBClientProtocol",
]


def _new_str_list() -> list[str]:
    return []


PlanBuilder = Callable[[Sequence[Path], Sequence[dict[str, str]], AppConfig], Sequence[ClipPlan]]


class AnalyzePicker(Protocol):
    def __call__(
        self,
        files: Sequence[Path],
        metadata: Sequence[dict[str, str]],
        target: str | None,
        *,
        cache_dir: Path | None = None,
    ) -> Path: ...


class CliPromptProtocol(CliOutputManagerProtocol, Protocol):
    """Reporter interface subset used by the metadata service."""

    def confirm(self, text: str, *, default: bool = True) -> bool: ...


class TMDBClientProtocol(Protocol):
    """Protocol describing TMDB lookup behaviour."""

    def resolve(
        self,
        *,
        files: Sequence[Path],
        metadata: Sequence[dict[str, str]],
        tmdb_cfg: Any,
        year_hint_raw: str | None,
    ) -> TMDBLookupResult: ...


class FilesystemProbeProtocol(Protocol):
    """Protocol for clip-metadata probing."""

    def probe(
        self,
        plans: Sequence[ClipPlan],
        runtime_cfg: RuntimeConfig,
        cache_dir: Path,
        *,
        reporter: CliPromptProtocol | None = None,
    ) -> None: ...


@dataclass(slots=True)
class MetadataResolveRequest:
    """Inputs required to resolve TMDB metadata and build clip plans."""

    cfg: AppConfig
    root: Path
    files: Sequence[Path]
    reporter: CliPromptProtocol
    json_tail: JsonTail
    layout_data: MutableMapping[str, Any]
    collected_warnings: list[str]


@dataclass(slots=True)
class MetadataResolveResult:
    """Outputs returned by :class:`MetadataResolver`."""

    plans: list[ClipPlan]
    metadata: list[dict[str, str]]
    metadata_title: str | None
    analyze_path: Path
    slowpics_title_inputs: SlowpicsTitleInputs
    slowpics_final_title: str
    slowpics_resolved_base: str | None
    slowpics_tmdb_disclosure_line: str | None
    slowpics_verbose_tmdb_tag: str | None
    tmdb_notes: list[str] = field(default_factory=_new_str_list)


class MetadataResolver:
    """Resolve TMDB context, build plans, and probe clip metadata."""

    def __init__(
        self,
        *,
        tmdb_client: TMDBClientProtocol,
        plan_builder: PlanBuilder,
        analyze_picker: AnalyzePicker,
        clip_probe: FilesystemProbeProtocol,
    ) -> None:
        self._tmdb_client = tmdb_client
        self._plan_builder = plan_builder
        self._analyze_picker = analyze_picker
        self._clip_probe = clip_probe

    def resolve(self, request: MetadataResolveRequest) -> MetadataResolveResult:
        """Resolve TMDB context and prepare clip plans."""

        if not request.files:
            raise CLIAppError("MetadataResolver requires at least one input file.")

        metadata = list(metadata_utils.parse_metadata(request.files, request.cfg.naming))
        metadata_title = (
            metadata_utils.first_non_empty(metadata, "title")
            or metadata_utils.first_non_empty(metadata, "anime_title")
        )
        year_hint_raw = metadata_utils.first_non_empty(metadata, "year")
        tmdb_api_key_present = bool(request.cfg.tmdb.api_key.strip())
        tmdb_notes = _new_str_list()
        tmdb_resolution: TMDBResolution | None = None
        manual_tmdb: tuple[str, str] | None = None
        tmdb_error_message: str | None = None
        tmdb_ambiguous = False
        tmdb_category: str | None = None
        tmdb_id_value: str | None = None
        tmdb_language: str | None = None
        if tmdb_api_key_present:
            lookup = self._tmdb_client.resolve(
                files=request.files,
                metadata=metadata,
                tmdb_cfg=request.cfg.tmdb,
                year_hint_raw=year_hint_raw,
            )
            tmdb_resolution = lookup.resolution
            manual_tmdb = lookup.manual_override
            tmdb_error_message = lookup.error_message
            tmdb_ambiguous = lookup.ambiguous

        if tmdb_resolution is not None:
            tmdb_category = tmdb_resolution.category
            tmdb_id_value = tmdb_resolution.tmdb_id
            tmdb_language = tmdb_resolution.original_language

        if manual_tmdb:
            tmdb_category, tmdb_id_value = manual_tmdb
            tmdb_language = None
            tmdb_resolution = None
            logger.info("TMDB manual override selected: %s/%s", tmdb_category, tmdb_id_value)

        if tmdb_error_message and tmdb_api_key_present:
            logger.warning("TMDB lookup failed for %s: %s", request.files[0].name, tmdb_error_message)

        tmdb_context = self._build_tmdb_context(
            request,
            metadata,
            metadata_title,
            year_hint_raw,
            tmdb_resolution,
            tmdb_category,
            tmdb_id_value,
            tmdb_language,
        )
        if tmdb_resolution is not None:
            self._apply_tmdb_resolution(
                request,
                tmdb_resolution,
                tmdb_context,
                tmdb_category,
                tmdb_id_value,
            )
        elif manual_tmdb:
            self._apply_manual_tmdb(
                request,
                tmdb_context,
                tmdb_category,
                tmdb_id_value,
                tmdb_language,
            )
        elif tmdb_api_key_present:
            self._log_tmdb_failure_notes(
                request,
                tmdb_error_message,
                tmdb_ambiguous,
                tmdb_notes,
            )
        elif not (request.cfg.slowpics.tmdb_id or "").strip():
            self._record_tmdb_note(
                request,
                "TMDB disabled: set [tmdb].api_key in config.toml to enable automatic matching.",
                tmdb_notes,
            )

        if tmdb_id_value and not (request.cfg.slowpics.tmdb_id or "").strip():
            request.cfg.slowpics.tmdb_id = str(tmdb_id_value)
        if tmdb_category and not (getattr(request.cfg.slowpics, "tmdb_category", "") or "").strip():
            request.cfg.slowpics.tmdb_category = tmdb_category

        (
            slowpics_title_inputs,
            slowpics_final_title,
            slowpics_resolved_base,
        ) = self._update_slowpics_blocks(request, tmdb_context, metadata_title)
        slowpics_tmdb_disclosure_line, slowpics_verbose_tmdb_tag = self._render_tmdb_disclosures(
            tmdb_resolution,
            tmdb_context,
            request,
            tmdb_category,
            tmdb_id_value,
            slowpics_resolved_base,
        )

        plans = list(self._plan_builder(request.files, metadata, request.cfg))
        self._assign_unique_safe_labels(plans)

        analyze_path = self._analyze_picker(
            request.files,
            metadata,
            request.cfg.analysis.analyze_clip,
            cache_dir=request.root,
        )

        try:
            self._clip_probe.probe(
                plans,
                request.cfg.runtime,
                request.root,
                reporter=request.reporter,
            )
        except ClipInitError as exc:
            raise CLIAppError(
                f"Failed to open clip: {exc}",
                rich_message=f"[red]Failed to open clip:[/red] {exc}",
            ) from exc

        return MetadataResolveResult(
            plans=plans,
            metadata=metadata,
            metadata_title=metadata_title,
            analyze_path=analyze_path,
            slowpics_title_inputs=slowpics_title_inputs,
            slowpics_final_title=slowpics_final_title,
            slowpics_resolved_base=slowpics_resolved_base,
            slowpics_tmdb_disclosure_line=slowpics_tmdb_disclosure_line,
            slowpics_verbose_tmdb_tag=slowpics_verbose_tmdb_tag,
            tmdb_notes=tmdb_notes,
        )

    def _build_tmdb_context(
        self,
        request: MetadataResolveRequest,
        metadata: Sequence[dict[str, str]],
        metadata_title: str | None,
        year_hint_raw: str | None,
        tmdb_resolution: TMDBResolution | None,
        tmdb_category: str | None,
        tmdb_id_value: str | None,
        tmdb_language: str | None,
    ) -> dict[str, str]:
        files = request.files
        tmdb_context: dict[str, str] = {
            "Title": metadata_title or ((metadata[0].get("label") or "") if metadata else ""),
            "OriginalTitle": "",
            "Year": year_hint_raw or "",
            "TMDBId": tmdb_id_value or "",
            "TMDBCategory": tmdb_category or "",
            "OriginalLanguage": tmdb_language or "",
            "Filename": files[0].stem,
            "FileName": files[0].name,
            "Label": (metadata[0].get("label") or files[0].name) if metadata else files[0].name,
        }
        if tmdb_resolution is not None:
            if tmdb_resolution.title:
                tmdb_context["Title"] = tmdb_resolution.title
            if tmdb_resolution.original_title:
                tmdb_context["OriginalTitle"] = tmdb_resolution.original_title
            if tmdb_resolution.year is not None:
                tmdb_context["Year"] = str(tmdb_resolution.year)
            if tmdb_resolution.original_language:
                tmdb_context["OriginalLanguage"] = tmdb_resolution.original_language
        return tmdb_context

    def _apply_tmdb_resolution(
        self,
        request: MetadataResolveRequest,
        tmdb_resolution: TMDBResolution,
        tmdb_context: dict[str, str],
        tmdb_category: str | None,
        tmdb_id_value: str | None,
    ) -> None:
        reporter = request.reporter
        match_title = tmdb_resolution.title or tmdb_context.get("Title") or request.files[0].stem
        year_display = tmdb_context.get("Year") or ""
        lang_text = tmdb_resolution.original_language or "und"
        tmdb_identifier = f"{tmdb_resolution.category}/{tmdb_resolution.tmdb_id}"
        safe_match_title = _sanitize_console_text(match_title)
        safe_year_display = _sanitize_console_text(year_display)
        safe_lang_text = _sanitize_console_text(lang_text)
        title_segment = _color_text(
            escape(f'"{safe_match_title} ({safe_year_display})"'),
            "bright_white",
        )
        lang_segment = _format_kv("lang", safe_lang_text, label_style="dim cyan", value_style="blue")
        reporter.verbose_line(
            "  ".join(
                [
                    _format_kv("TMDB", tmdb_identifier, label_style="cyan", value_style="bright_white"),
                    title_segment,
                    lang_segment,
                ]
            )
        )
        heuristic = (tmdb_resolution.candidate.reason or "match").replace("_", " ").replace("-", " ")
        source = "filename" if tmdb_resolution.candidate.used_filename_search else "external id"
        reporter.verbose_line(
            f"TMDB match heuristics: source={source} heuristic={heuristic.strip()}"
        )
        layout_tmdb = request.layout_data.get("tmdb")
        if isinstance(layout_tmdb, MutableMapping):
            tmdb_mapping = cast(MutableMapping[str, Any], layout_tmdb)
            tmdb_mapping.update(
                {
                    "category": tmdb_resolution.category,
                    "id": tmdb_resolution.tmdb_id,
                    "title": match_title,
                    "year": year_display,
                    "lang": lang_text,
                }
            )
        reporter.set_flag("tmdb_resolved", True)

    def _apply_manual_tmdb(
        self,
        request: MetadataResolveRequest,
        tmdb_context: dict[str, str],
        tmdb_category: str | None,
        tmdb_id_value: str | None,
        tmdb_language: str | None,
    ) -> None:
        reporter = request.reporter
        files = request.files
        display_title = tmdb_context.get("Title") or files[0].stem
        category_display = tmdb_category or request.cfg.slowpics.tmdb_category or ""
        id_display = tmdb_id_value or request.cfg.slowpics.tmdb_id or ""
        lang_text = tmdb_language or tmdb_context.get("OriginalLanguage") or "und"
        identifier = f"{category_display}/{id_display}".strip("/")
        safe_display_title = _sanitize_console_text(display_title)
        safe_year_component = _sanitize_console_text(tmdb_context.get("Year") or "")
        title_segment = _color_text(
            escape(f'"{safe_display_title} ({safe_year_component})"'),
            "bright_white",
        )
        safe_lang_text = _sanitize_console_text(lang_text)
        lang_segment = _format_kv("lang", safe_lang_text, label_style="dim cyan", value_style="blue")
        reporter.verbose_line(
            "  ".join(
                [
                    _format_kv("TMDB", identifier, label_style="cyan", value_style="bright_white"),
                    title_segment,
                    lang_segment,
                ]
            )
        )
        layout_tmdb = request.layout_data.get("tmdb")
        if isinstance(layout_tmdb, MutableMapping):
            tmdb_mapping = cast(MutableMapping[str, Any], layout_tmdb)
            tmdb_mapping.update(
                {
                    "category": category_display,
                    "id": id_display,
                    "title": display_title,
                    "year": tmdb_context.get("Year") or "",
                    "lang": lang_text,
                }
            )
        reporter.set_flag("tmdb_resolved", True)

    def _log_tmdb_failure_notes(
        self,
        request: MetadataResolveRequest,
        tmdb_error_message: str | None,
        tmdb_ambiguous: bool,
        tmdb_notes: list[str],
    ) -> None:
        if tmdb_error_message:
            self._record_tmdb_note(
                request,
                f"TMDB lookup failed: {tmdb_error_message}",
                tmdb_notes,
            )
        elif tmdb_ambiguous:
            self._record_tmdb_note(
                request,
                f"TMDB ambiguous results for {request.files[0].name}; continuing without metadata.",
                tmdb_notes,
            )
        else:
            self._record_tmdb_note(
                request,
                f"TMDB could not find a confident match for {request.files[0].name}.",
                tmdb_notes,
            )

    def _record_tmdb_note(
        self,
        request: MetadataResolveRequest,
        message: str,
        tmdb_notes: list[str],
    ) -> None:
        safe_message = _sanitize_console_text(message)
        request.collected_warnings.append(safe_message)
        tmdb_notes.append(safe_message)

    def _update_slowpics_blocks(
        self,
        request: MetadataResolveRequest,
        tmdb_context: dict[str, str],
        metadata_title: str | None,
    ) -> tuple[SlowpicsTitleInputs, str, str | None]:
        cfg = request.cfg
        files = request.files
        suffix_literal = getattr(cfg.slowpics, "collection_suffix", "") or ""
        suffix = suffix_literal.strip()
        template_raw = cfg.slowpics.collection_name or ""
        collection_template = template_raw.strip()
        resolved_title_value = (tmdb_context.get("Title") or "").strip()
        resolved_year_value = (tmdb_context.get("Year") or "").strip()
        resolved_base_title: str | None = None
        if resolved_title_value:
            resolved_base_title = resolved_title_value
            if resolved_year_value:
                resolved_base_title = f"{resolved_title_value} ({resolved_year_value})"
        if collection_template:
            rendered_collection = tmdb_workflow.render_collection_name(
                collection_template,
                tmdb_context,
            ).strip()
            final_collection_name = rendered_collection or "Frame Comparison"
        else:
            derived_title = resolved_title_value or metadata_title or files[0].stem
            derived_year = resolved_year_value
            base_collection = (derived_title or "").strip()
            if base_collection and derived_year:
                base_collection = f"{base_collection} ({derived_year})"
            final_collection_name = base_collection or "Frame Comparison"
            if suffix:
                final_collection_name = f"{final_collection_name} {suffix}" if final_collection_name else suffix
        cfg.slowpics.collection_name = final_collection_name
        slowpics_title_inputs: SlowpicsTitleInputs = {
            "resolved_base": resolved_base_title,
            "collection_name": cfg.slowpics.collection_name,
            "collection_suffix": suffix_literal,
        }
        slowpics_block = request.json_tail["slowpics"]
        title_inputs = slowpics_block["title"]["inputs"]
        title_inputs["resolved_base"] = slowpics_title_inputs["resolved_base"]
        title_inputs["collection_name"] = slowpics_title_inputs["collection_name"]
        title_inputs["collection_suffix"] = slowpics_title_inputs["collection_suffix"]
        slowpics_block["title"]["final"] = final_collection_name
        slowpics_layout_view = request.layout_data.get("slowpics")
        if isinstance(slowpics_layout_view, MutableMapping):
            slowpics_mapping = cast(MutableMapping[str, Any], slowpics_layout_view)
            slowpics_mapping["collection_name"] = final_collection_name
            slowpics_mapping["auto_upload"] = bool(cfg.slowpics.auto_upload)
            slowpics_mapping.setdefault(
                "status",
                "pending" if cfg.slowpics.auto_upload else "disabled",
            )
            request.layout_data["slowpics"] = slowpics_mapping
        request.reporter.update_values(request.layout_data)
        return slowpics_title_inputs, final_collection_name, resolved_base_title

    def _render_tmdb_disclosures(
        self,
        tmdb_resolution: TMDBResolution | None,
        tmdb_context: dict[str, str],
        request: MetadataResolveRequest,
        tmdb_category: str | None,
        tmdb_id_value: str | None,
        slowpics_resolved_base: str | None,
    ) -> tuple[str | None, str | None]:
        cfg = request.cfg
        files = request.files
        suffix_literal = getattr(cfg.slowpics, "collection_suffix", "") or ""
        slowpics_tmdb_disclosure_line: str | None = None
        slowpics_verbose_tmdb_tag: str | None = None
        if tmdb_resolution is not None:
            match_title = tmdb_resolution.title or tmdb_context.get("Title") or files[0].stem
            year_display = tmdb_context.get("Year") or ""
            if slowpics_resolved_base:
                base_display = slowpics_resolved_base
            elif match_title and year_display:
                base_display = f"{match_title} ({year_display})"
            else:
                base_display = match_title or "(n/a)"
            safe_base_display = _sanitize_console_text(base_display)
            slowpics_tmdb_disclosure_line = (
                f'slow.pics title inputs: base="{escape(safe_base_display)}"  '
                f'collection_suffix="{escape(str(suffix_literal))}"'
            )
            if tmdb_category and tmdb_id_value:
                slowpics_verbose_tmdb_tag = f"TMDB={tmdb_category}_{tmdb_id_value}"
        elif tmdb_id_value or tmdb_category:
            display_title = tmdb_context.get("Title") or files[0].stem
            if slowpics_resolved_base:
                base_display = slowpics_resolved_base
            else:
                year_component = tmdb_context.get("Year") or ""
                if display_title and year_component:
                    base_display = f"{display_title} ({year_component})"
                else:
                    base_display = display_title or "(n/a)"
            safe_base_display = _sanitize_console_text(base_display)
            slowpics_tmdb_disclosure_line = (
                f'slow.pics title inputs: base="{escape(safe_base_display)}"  '
                f'collection_suffix="{escape(str(suffix_literal))}"'
            )
            if tmdb_category and tmdb_id_value:
                slowpics_verbose_tmdb_tag = f"TMDB={tmdb_category}_{tmdb_id_value}"
        return slowpics_tmdb_disclosure_line, slowpics_verbose_tmdb_tag

    def _assign_unique_safe_labels(self, plans: Sequence[ClipPlan]) -> None:
        counts: dict[str, int] = {}
        for plan in plans:
            raw_label = plan.metadata.get("label") or plan.path.stem
            base = _render_sanitise_label(raw_label)
            occurrence = counts.get(base, 0) + 1
            counts[base] = occurrence
            safe_label = base if occurrence == 1 else f"{base}_{occurrence}"
            plan.metadata[SAFE_LABEL_META_KEY] = safe_label
