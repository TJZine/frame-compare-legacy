from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from click.core import Group
from src.datatypes import TMDBConfig
from src.frame_compare import vs as vs_core
from src.frame_compare.core import CLIAppError as CLIAppError
from src.frame_compare.doctor import DoctorCheck as DoctorCheck
from src.frame_compare.doctor import emit_results as emit_doctor_results
from src.frame_compare.doctor import collect_checks as collect_doctor_checks
from src.frame_compare.preflight import (
    PreflightResult as PreflightResult,
    collect_path_diagnostics as collect_path_diagnostics,
    prepare_preflight as prepare_preflight,
    resolve_workspace_root as resolve_workspace_root,
)
from src.frame_compare.render.errors import ScreenshotError as ScreenshotError
from src.frame_compare.runner import RunRequest as RunRequest
from src.frame_compare.runner import RunResult as RunResult
from src.frame_compare.tmdb_workflow import TMDBLookupResult as TMDBLookupResult
from src.frame_compare.tmdb_workflow import render_collection_name as render_collection_name

__all__: tuple[str, ...]


def run_cli(
    config_path: Optional[str],
    input_dir: Optional[str] = ...,
    *,
    root_override: Optional[str] = ...,
    audio_track_overrides: Optional[Iterable[str]] = ...,
    quiet: bool = ...,
    verbose: bool = ...,
    no_color: bool = ...,
    report_enable_override: Optional[bool] = ...,
    skip_wizard: bool = ...,
    debug_color: bool = ...,
    tonemap_overrides: Optional[Mapping[str, Any]] = ...,
    from_cache_only: bool = ...,
    force_cache_refresh: bool = ...,
    show_partial_sections: bool = ...,
    show_missing_sections: bool = ...,
    service_mode_override: Optional[bool] = ...,
    diagnostic_frame_metrics: Optional[bool] = ...,
    dependencies: Any = ...,
) -> RunResult: ...


main: Group


def resolve_tmdb_workflow(
    *,
    files: Sequence[Path],
    metadata: Sequence[Mapping[str, str]],
    tmdb_cfg: TMDBConfig,
    year_hint_raw: Optional[str] = ...,
) -> TMDBLookupResult: ...


def render_collection_name(template_text: str, context: Mapping[str, Any]) -> str: ...


__all__ = (
    "run_cli",
    "main",
    "RunRequest",
    "RunResult",
    "CLIAppError",
    "ScreenshotError",
    "resolve_tmdb_workflow",
    "TMDBLookupResult",
    "render_collection_name",
    "prepare_preflight",
    "resolve_workspace_root",
    "PreflightResult",
    "collect_path_diagnostics",
    "collect_doctor_checks",
    "emit_doctor_results",
    "DoctorCheck",
    "vs_core",
)
