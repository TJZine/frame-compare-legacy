"""Public shim exposing the frame_compare CLI and library surface."""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Iterable, Optional, cast

import src.frame_compare.cli_entry as _cli_entry
import src.frame_compare.compat as _compat
import src.frame_compare.core as _core
import src.frame_compare.doctor as doctor_module
import src.frame_compare.preflight as _preflight
import src.frame_compare.tmdb_workflow as tmdb_workflow
from src.frame_compare import runner
from src.frame_compare import vs as _vs_core
from src.frame_compare.render.errors import ScreenshotError

collect_doctor_checks = doctor_module.collect_checks
collect_path_diagnostics = _preflight.collect_path_diagnostics
prepare_preflight = _preflight.prepare_preflight
resolve_workspace_root = _preflight.resolve_workspace_root
PreflightResult = _preflight.PreflightResult
CLIAppError = _core.CLIAppError

resolve_tmdb_workflow = tmdb_workflow.resolve_workflow
TMDBLookupResult = tmdb_workflow.TMDBLookupResult
render_collection_name = tmdb_workflow.render_collection_name
emit_doctor_results = doctor_module.emit_results
DoctorCheck = doctor_module.DoctorCheck
vs_core = _vs_core

RunResult = runner.RunResult
RunRequest = runner.RunRequest

# Preserve legacy compatibility mapping for callers that still access it directly.
_COMPAT_EXPORTS = _compat.COMPAT_EXPORTS
# Populate legacy compatibility surface while keeping the shim thin.
_compat.apply_compat_exports(globals())

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


def run_cli(
    config_path: str | None,
    input_dir: str | None = None,
    *,
    root_override: str | None = None,
    audio_track_overrides: Iterable[str] | None = None,
    quiet: bool = False,
    verbose: bool = False,
    no_color: bool = False,
    report_enable_override: Optional[bool] = None,
    skip_wizard: bool = False,
    debug_color: bool = False,
    tonemap_overrides: Optional[Dict[str, Any]] = None,
    from_cache_only: bool = False,
    force_cache_refresh: bool = False,
    show_partial_sections: bool = False,
    show_missing_sections: bool = True,
    service_mode_override: bool | None = None,
    diagnostic_frame_metrics: bool | None = None,
    dependencies: runner.RunDependencies | None = None,
) -> RunResult:
    """Delegate to the shared runner module."""
    request = RunRequest(
        config_path=config_path,
        input_dir=input_dir,
        root_override=root_override,
        audio_track_overrides=audio_track_overrides,
        quiet=quiet,
        verbose=verbose,
        no_color=no_color,
        report_enable_override=report_enable_override,
        skip_wizard=skip_wizard,
        debug_color=debug_color,
        tonemap_overrides=tonemap_overrides,
        impl_module=sys.modules.get(__name__),
        from_cache_only=from_cache_only,
        force_cache_refresh=force_cache_refresh,
        show_partial_sections=show_partial_sections,
        show_missing_sections=show_missing_sections,
        service_mode_override=service_mode_override,
        diagnostic_frame_metrics=diagnostic_frame_metrics,
    )
    return runner.run(request, dependencies=dependencies)


main = _cli_entry.main
cli = getattr(_cli_entry, "cli", main)


if __name__ == "__main__":
    _entry_point = cast(Callable[[], None], main)
    _entry_point()
