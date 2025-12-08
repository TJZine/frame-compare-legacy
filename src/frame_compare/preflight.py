"""Workspace-resolution helpers extracted per docs/refactor/mod_refactor.md Phase 1.1."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, Mapping

import click
from rich.markup import escape

from src.config_loader import ConfigError, load_config
from src.datatypes import (
    AnalysisConfig,
    AppConfig,
    AudioAlignmentConfig,
    CLIConfig,
    ColorConfig,
    DiagnosticsConfig,
    NamingConfig,
    OverridesConfig,
    PathsConfig,
    ReportConfig,
    RunnerConfig,
    RuntimeConfig,
    ScreenshotConfig,
    SlowpicsConfig,
    SourceConfig,
    TMDBConfig,
)
from src.frame_compare.cli_runtime import CLIAppError
from src.frame_compare.config_template import copy_default_config

from .config_helpers import env_flag_enabled

CONFIG_ENV_VAR: Final[str] = "FRAME_COMPARE_CONFIG"
ROOT_ENV_VAR: Final[str] = "FRAME_COMPARE_ROOT"
ROOT_SENTINELS: Final[tuple[str, ...]] = ("pyproject.toml", ".git", "comparison_videos")
NO_WIZARD_ENV_VAR: Final[str] = "FRAME_COMPARE_NO_WIZARD"
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PACKAGED_TEMPLATE_PATH: Final[Path] = (
    PROJECT_ROOT / "src" / "data" / "config.toml.template"
).resolve()

__all__ = [
    "CONFIG_ENV_VAR",
    "ROOT_ENV_VAR",
    "ROOT_SENTINELS",
    "NO_WIZARD_ENV_VAR",
    "PROJECT_ROOT",
    "PACKAGED_TEMPLATE_PATH",
    "PreflightResult",
    "resolve_workspace_root",
    "resolve_subdir",
    "collect_path_diagnostics",
    "prepare_preflight",
    "_path_is_within_root",
    "_path_contains_site_packages",
    "_is_writable_path",
    "_abort_if_site_packages",
    "is_writable_path",
    "abort_if_site_packages",
    "_seed_default_config",
    "_fresh_app_config",
    "fresh_app_config",
]


def resolve_subdir(
    root: Path, relative: str, *, purpose: str, allow_absolute: bool = False
) -> Path:
    """Return a normalised path under *root* for user-managed directories."""

    try:
        root_resolved = root.resolve()
    except OSError as exc:
        raise CLIAppError(
            f"Unable to resolve workspace root '{root}': {exc}",
            rich_message=f"[red]Unable to resolve workspace root:[/red] {exc}",
        ) from exc

    candidate = Path(str(relative))
    if candidate.is_absolute():
        if not allow_absolute:
            message = (
                f"Configured {purpose} must be relative to the input directory, got '{relative}'"
            )
            raise CLIAppError(message, rich_message=f"[red]{message}[/red]")
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise CLIAppError(
                f"Unable to resolve configured {purpose} '{relative}': {exc}",
                rich_message=f"[red]Unable to resolve configured {purpose}:[/red] {exc}",
            ) from exc
        return resolved

    resolved = (root_resolved / candidate).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        message = (
            f"Configured {purpose} escapes the input directory: '{relative}' -> {resolved}"
        )
        raise CLIAppError(message, rich_message=f"[red]{message}[/red]") from exc

    return resolved


_resolve_workspace_subdir = resolve_subdir


def _path_is_within_root(root: Path, candidate: Path) -> bool:
    """Return True when *candidate* resides under *root* after resolution."""

    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False

    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        return False
    return True


_SITE_PACKAGES_MARKERS: Final[set[str]] = {"site-packages", "dist-packages"}


def _path_contains_site_packages(path: Path) -> bool:
    """Return True when *path* (or any ancestor) lives under site/dist-packages."""

    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for part in resolved.parts:
        if part.lower() in _SITE_PACKAGES_MARKERS:
            return True
    return False


def _nearest_existing_dir(path: Path) -> Path:
    """Return the nearest existing directory for *path* (itself or ancestor)."""

    candidate = path
    if candidate.is_file():
        candidate = candidate.parent

    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def _is_writable_path(path: Path, *, for_file: bool) -> bool:
    """Return True when the given path (or its nearest parent) is writable."""

    target = path.parent if for_file else path
    try:
        target = target.resolve(strict=False)
    except OSError:
        pass
    probe = _nearest_existing_dir(target)
    try:
        probe = probe.resolve(strict=False)
    except OSError:
        pass
    return os.access(probe, os.W_OK)


is_writable_path = _is_writable_path


def _abort_if_site_packages(path_map: Mapping[str, Path]) -> None:
    """Abort execution when any mapped path falls under site/dist-packages."""

    for label, candidate in path_map.items():
        if _path_contains_site_packages(candidate):
            message = (
                f"{label} path '{candidate}' resolves inside a site-packages/dist-packages "
                "directory; refuse to continue. Use --root or FRAME_COMPARE_ROOT to "
                "select a writable workspace."
            )
            raise CLIAppError(
                message,
                code=2,
                rich_message=f"[red]{escape(message)}[/red]",
            )


abort_if_site_packages = _abort_if_site_packages


@dataclass
class PreflightResult:
    """Resolved configuration and workspace paths used during startup."""

    workspace_root: Path
    media_root: Path
    config_path: Path
    config: AppConfig
    warnings: tuple[str, ...] = ()
    legacy_config: bool = False


_PathPreflightResult = PreflightResult


def resolve_workspace_root(cli_root: str | None) -> Path:
    """Resolve the workspace root using CLI flag, env var, or sentinel search."""

    if cli_root:
        candidate = Path(cli_root).expanduser()
    else:
        env_root = os.environ.get(ROOT_ENV_VAR)
        if env_root:
            candidate = Path(env_root).expanduser()
        else:
            start = Path.cwd()
            current = start
            sentinel_root: Path | None = None
            while True:
                if any((current / marker).exists() for marker in ROOT_SENTINELS):
                    sentinel_root = current
                    break
                if current.parent == current:
                    break
                current = current.parent
            candidate = sentinel_root or start

    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise CLIAppError(
            f"Failed to resolve workspace root '{candidate}': {exc}",
            code=2,
            rich_message=f"[red]Failed to resolve workspace root:[/red] {exc}",
        ) from exc

    if _path_contains_site_packages(resolved):
        message = (
            f"Workspace root '{resolved}' is inside site-packages/dist-packages; "
            "choose a writable directory via --root or FRAME_COMPARE_ROOT."
        )
        raise CLIAppError(message, code=2, rich_message=f"[red]{escape(message)}[/red]")

    return resolved


_discover_workspace_root = resolve_workspace_root


def _seed_default_config(path: Path) -> None:
    """Atomically seed config.toml at *path* from the packaged template."""

    try:
        copy_default_config(path)
    except FileExistsError:
        return
    except OSError as exc:
        message = f"Unable to create default config at {path}: {exc}"
        raise CLIAppError(
            message,
            code=2,
            rich_message=(
                "[red]Unable to create default config:[/red] "
                f"{exc}. Set --root/FRAME_COMPARE_ROOT to a writable directory."
            ),
        ) from exc


def _fresh_app_config() -> AppConfig:
    """Return an AppConfig populated with built-in defaults."""

    return AppConfig(
        analysis=AnalysisConfig(),
        screenshots=ScreenshotConfig(),
        cli=CLIConfig(),
        runner=RunnerConfig(),
        slowpics=SlowpicsConfig(),
        tmdb=TMDBConfig(),
        naming=NamingConfig(),
        paths=PathsConfig(),
        runtime=RuntimeConfig(),
        overrides=OverridesConfig(),
        color=ColorConfig(),
        source=SourceConfig(),
        audio_alignment=AudioAlignmentConfig(),
        report=ReportConfig(),
        diagnostics=DiagnosticsConfig(),
    )


fresh_app_config = _fresh_app_config


def prepare_preflight(
    *,
    cli_root: str | None,
    config_override: str | None,
    input_override: str | None,
    ensure_config: bool,
    create_dirs: bool,
    create_media_dir: bool,
    allow_auto_wizard: bool = False,
    skip_auto_wizard: bool = False,
    ) -> PreflightResult:
    """Resolve workspace root, configuration, and media directories."""

    workspace_root = resolve_workspace_root(cli_root)
    warnings: list[str] = []
    skip_auto_wizard = skip_auto_wizard or env_flag_enabled(os.environ.get(NO_WIZARD_ENV_VAR))

    if create_dirs:
        try:
            workspace_root.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            detail = exc.strerror or "Permission denied"
            message = f"Unable to create workspace root '{workspace_root}': {detail}"
            raise CLIAppError(
                message,
                code=2,
                rich_message=(
                    "[red]Unable to create workspace root:[/red] "
                    f"{escape(str(workspace_root))} ({escape(detail)})"
                ),
            ) from exc
    elif not workspace_root.exists():
        parent = workspace_root.parent
        if not parent.exists() or not os.access(parent, os.W_OK):
            warnings.append(
                f"Workspace root {workspace_root} may be unwritable; parent directory is inaccessible."
            )
    if workspace_root.exists() and not os.access(workspace_root, os.W_OK):
        if create_dirs:
            raise CLIAppError(
                f"Workspace root '{workspace_root}' is not writable.",
                code=2,
                rich_message=f"[red]Workspace root is not writable:[/red] {workspace_root}",
            )
        warnings.append(f"Workspace root {workspace_root} is not writable.")

    config_path: Path
    legacy = False

    if config_override:
        config_path = Path(config_override).expanduser()
    else:
        env_override = os.environ.get(CONFIG_ENV_VAR)
        if env_override:
            config_path = Path(env_override).expanduser()
        else:
            config_dir = workspace_root / "config"
            config_path = config_dir / "config.toml"
            legacy_path = workspace_root / "config.toml"

            if config_path.exists():
                pass
            elif legacy_path.exists():
                config_path = legacy_path
                legacy = True
                warnings.append(
                    f"Using legacy config at {legacy_path}. Move it to {config_dir / 'config.toml'}."
                )
            elif ensure_config:
                interactive = sys.stdin.isatty()
                cli_module = None
                auto_wizard_base_allowed = (
                    allow_auto_wizard
                    and not skip_auto_wizard
                    and env_override is None
                    and config_override is None
                )
                if auto_wizard_base_allowed:
                    try:
                        cli_module = importlib.import_module("frame_compare")
                    except (ImportError, AttributeError, TypeError):
                        cli_module = None
                    if not interactive and cli_module is not None:
                        proxy_sys = getattr(cli_module, "sys", None)
                        proxy_stdin = getattr(proxy_sys, "stdin", None) if proxy_sys is not None else None
                        if proxy_stdin is not None:
                            try:
                                interactive = bool(proxy_stdin.isatty())
                            except (AttributeError, ValueError, OSError):
                                pass
                auto_wizard_allowed = bool(cli_module and auto_wizard_base_allowed and interactive)
                if auto_wizard_allowed and cli_module:
                    try:
                        execute_wizard = cli_module._execute_wizard_session
                        new_root, new_config_path = execute_wizard(
                            root_override=str(workspace_root),
                            config_override=None,
                            input_override=input_override,
                            preset_name=None,
                            auto_launch=True,
                        )
                    except click.exceptions.Exit as exc:
                        raise exc
                    return prepare_preflight(
                        cli_root=str(new_root),
                        config_override=str(new_config_path),
                        input_override=input_override,
                        ensure_config=ensure_config,
                        create_dirs=create_dirs,
                        create_media_dir=create_media_dir,
                        allow_auto_wizard=False,
                        skip_auto_wizard=skip_auto_wizard,
                    )
                try:
                    config_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError as exc:
                    detail = exc.strerror or "Permission denied"
                    message = f"Unable to create config directory '{config_dir}': {detail}"
                    raise CLIAppError(
                        message,
                        code=2,
                        rich_message=(
                            "[red]Unable to create config directory:[/red] "
                            f"{escape(str(config_dir))} ({escape(detail)})"
                        ),
                    ) from exc
                _seed_default_config(config_path)
                if allow_auto_wizard and (skip_auto_wizard or not interactive):
                    click.echo("Seeded default config. Run 'frame-compare wizard' to customise settings.")

    if _path_contains_site_packages(config_path):
        message = (
            f"Config path '{config_path}' resides inside site-packages/dist-packages; "
            "choose a writable location via --root or --config."
        )
        raise CLIAppError(message, code=2, rich_message=f"[red]{escape(message)}[/red]")

    cfg: AppConfig
    try:
        cfg = load_config(str(config_path))
    except FileNotFoundError:
        if ensure_config:
            raise CLIAppError(
                f"Config file not found: {config_path}",
                code=2,
                rich_message=f"[red]Config file not found:[/red] {config_path}",
            ) from None
        cfg = _fresh_app_config()
        warnings.append(f"Config file not found; using defaults at {config_path}")
    except PermissionError as exc:
        raise CLIAppError(
            f"Config file is not readable: {config_path}",
            code=2,
            rich_message=f"[red]Config file is not readable:[/red] {config_path}",
        ) from exc
    except OSError as exc:
        raise CLIAppError(
            f"Failed to read config file: {exc}",
            code=2,
            rich_message=f"[red]Failed to read config file:[/red] {exc}",
        ) from exc
    except ConfigError as exc:
        raise CLIAppError(
            f"Invalid configuration: {exc}",
            code=2,
            rich_message=f"[red]Invalid configuration:[/red] {exc}",
        ) from exc

    if legacy:
        warnings.append(
            f"Legacy config detected at {config_path}. "
            "Consider moving it to config/config.toml for future releases."
        )

    media_relative = input_override or cfg.paths.input_dir
    media_root = resolve_subdir(
        workspace_root,
        media_relative,
        purpose="[paths].input_dir",
        allow_absolute=bool(input_override),
    )
    if create_media_dir:
        try:
            media_root.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            detail = exc.strerror or "Permission denied"
            raise CLIAppError(
                f"Unable to create input workspace '{media_root}': {detail}",
                code=2,
                rich_message=(
                    "[red]Unable to create input workspace:[/red] "
                    f"{escape(str(media_root))} ({escape(detail)})"
                ),
            ) from exc
    elif not media_root.exists():
        parent = media_root.parent
        if not parent.exists() or not os.access(parent, os.W_OK):
            warnings.append(
                f"Input workspace {media_root} may be unwritable; parent directory is inaccessible."
            )
    if media_root.exists() and not os.access(media_root, os.W_OK):
        if create_media_dir:
            raise CLIAppError(
                f"Input workspace '{media_root}' is not writable.",
                code=2,
                rich_message=f"[red]Input workspace is not writable:[/red] {media_root}",
            )
        warnings.append(f"Input workspace {media_root} is not writable.")

    return PreflightResult(
        workspace_root=workspace_root,
        media_root=media_root,
        config_path=config_path,
        config=cfg,
        warnings=tuple(warnings),
        legacy_config=legacy,
    )


_prepare_preflight = prepare_preflight


def collect_path_diagnostics(
    *,
    cli_root: str | None,
    config_override: str | None,
    input_override: str | None,
) -> Dict[str, Any]:
    """Return a JSON-serialisable mapping describing key runtime paths."""

    preflight = prepare_preflight(
        cli_root=cli_root,
        config_override=config_override,
        input_override=input_override,
        ensure_config=False,  # diagnostics should not create or require a config file
        create_dirs=False,
        create_media_dir=False,
    )
    cfg = preflight.config
    workspace_root = preflight.workspace_root
    media_root = preflight.media_root
    config_path = preflight.config_path

    screens_dir = resolve_subdir(
        media_root,
        cfg.screenshots.directory_name,
        purpose="screenshots.directory_name",
    )
    analysis_cache = resolve_subdir(
        media_root,
        cfg.analysis.frame_data_filename,
        purpose="analysis.frame_data_filename",
    )
    offsets_path = resolve_subdir(
        media_root,
        cfg.audio_alignment.offsets_filename,
        purpose="audio_alignment.offsets_filename",
    )

    under_site_packages = any(
        _path_contains_site_packages(path) for path in (workspace_root, media_root, config_path)
    )

    diagnostics: Dict[str, Any] = {
        "workspace_root": str(workspace_root),
        "media_root": str(media_root),
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "legacy_config": preflight.legacy_config,
        "screens_dir": str(screens_dir),
        "analysis_cache": str(analysis_cache),
        "audio_offsets": str(offsets_path),
        "under_site_packages": under_site_packages,
        "writable": {
            "workspace_root": _is_writable_path(workspace_root, for_file=False),
            "media_root": _is_writable_path(media_root, for_file=False),
            "config_dir": _is_writable_path(config_path, for_file=True),
            "screens_dir": _is_writable_path(screens_dir, for_file=False),
        },
        "warnings": list(preflight.warnings),
    }
    return diagnostics


_collect_path_diagnostics = collect_path_diagnostics
