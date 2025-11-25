from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from rich.console import Console

from src.datatypes import AppConfig
from src.frame_compare.cli_runtime import (
    CliOutputManager,
    CliOutputManagerProtocol,
    JsonTail,
    NullCliOutputManager,
)
from src.frame_compare.orchestration.state import RunRequest, RunResult
from src.frame_compare.result_snapshot import SectionAvailability, SectionState

logger = logging.getLogger('frame_compare')


def create_reporter(
    request: RunRequest,
    layout_path: Path,
) -> CliOutputManagerProtocol:
    """Instantiate the CLI output manager based on request configuration."""
    impl = request.impl_module or importlib.import_module("frame_compare")
    console_cls = getattr(impl, 'Console', Console)
    reporter_console: Console | None = request.console
    reporter: CliOutputManagerProtocol | None = request.reporter

    if reporter is None:
        if reporter_console is None:
            reporter_console = console_cls(no_color=request.no_color, highlight=False)
        assert reporter_console is not None
        if request.reporter_factory is not None:
            reporter = request.reporter_factory(request, layout_path, reporter_console)
        else:
            reporter_cls = getattr(impl, 'CliOutputManager', CliOutputManager)
            null_cls = getattr(impl, 'NullCliOutputManager', NullCliOutputManager)
            if request.quiet:
                reporter = null_cls(
                    quiet=True,
                    verbose=request.verbose,
                    no_color=request.no_color,
                    layout_path=layout_path,
                    console=reporter_console,
                )
            else:
                reporter = reporter_cls(
                    quiet=request.quiet,
                    verbose=request.verbose,
                    no_color=request.no_color,
                    layout_path=layout_path,
                    console=reporter_console,
                )
    assert reporter is not None
    return reporter


def apply_section_availability_overrides(
    section_states: Mapping[str, SectionState],
    mark_section: Callable[[str, SectionAvailability, str | None], None],
    *,
    layout_data: Mapping[str, object],
    result: RunResult,
) -> None:
    """Annotate optional sections so cached renders can hide missing content by default."""

    def _mapping_for(key: str) -> dict[str, Any]:
        raw_value = layout_data.get(key)
        if isinstance(raw_value, MappingABC):
            typed_value = cast(Mapping[str, Any], raw_value)
            return {str(item_key): item_value for item_key, item_value in typed_value.items()}
        return {}

    if "render" in section_states and not result.image_paths:
        mark_section(
            "render",
            SectionAvailability.PARTIAL,
            "Screenshots unavailable; rerun without cache to capture fresh images.",
        )

    slowpics_view: dict[str, Any] = _mapping_for("slowpics")
    slowpics_status = str(slowpics_view.get("status") or "")
    slowpics_auto_upload = bool(slowpics_view.get("auto_upload"))
    if "publish" in section_states:
        # Publish tracks the slow.pics upload lifecycle.
        if slowpics_status == "failed":
            mark_section(
                "publish",
                SectionAvailability.PARTIAL,
                "slow.pics upload failed during the live run.",
            )
        elif not slowpics_auto_upload and not result.slowpics_url:
            mark_section(
                "publish",
                SectionAvailability.MISSING,
                "slow.pics upload disabled for this configuration.",
            )

    report_block: dict[str, Any] = _mapping_for("report")
    viewer_block: dict[str, Any] = _mapping_for("viewer")
    audio_layout: dict[str, Any] = _mapping_for("audio_alignment")
    vspreview_block: dict[str, Any] = _mapping_for("vspreview")

    # Viewer/report sections only render when we have a destination (HTML report or slow.pics URL).
    report_enabled = bool(report_block.get("enabled"))
    report_path = result.report_path
    if report_path is None:
        report_path_value = report_block.get("path")
        if isinstance(report_path_value, Path):
            report_path = report_path_value
        elif isinstance(report_path_value, str) and report_path_value:
            report_path = Path(report_path_value)
    if "report" in section_states:
        if not report_enabled:
            mark_section(
                "report",
                SectionAvailability.MISSING,
                "HTML report disabled for this run.",
            )
        elif report_path is None:
            mark_section(
                "report",
                SectionAvailability.PARTIAL,
                "Report metadata exists but the generated files are unavailable from cache.",
            )
    viewer_mode = str(viewer_block.get("mode") or "")
    viewer_destination = viewer_block.get("destination")
    if "viewer" in section_states:
        if viewer_mode == "none":
            mark_section(
                "viewer",
                SectionAvailability.MISSING,
                "Viewer output unavailable; rerun to generate an HTML report or slow.pics link.",
            )
        elif not viewer_destination:
            mark_section(
                "viewer",
                SectionAvailability.PARTIAL,
                "Viewer metadata present but destination details were not captured in the cache.",
            )

    # Audio alignment output depends on whether the feature is enabled and whether offsets were captured.
    audio_enabled = bool(audio_layout.get("enabled"))
    audio_offsets_present = bool(audio_layout.get("offsets_sec")) or bool(audio_layout.get("measurements"))
    if "audio_align" in section_states:
        if not audio_enabled:
            mark_section(
                "audio_align",
                SectionAvailability.MISSING,
                "Audio alignment disabled; enable it to populate this section.",
            )
        elif not audio_offsets_present:
            mark_section(
                "audio_align",
                SectionAvailability.PARTIAL,
                "Audio alignment metadata missing from cache; rerun to recompute offsets.",
            )

    # VSPreview info relies on the integration being enabled and available.
    use_vspreview = bool(audio_layout.get("use_vspreview"))
    missing_block_value = vspreview_block.get("missing")
    missing_active = False
    if isinstance(missing_block_value, MappingABC):
        typed_missing = cast(Mapping[str, Any], missing_block_value)
        missing_active = bool(typed_missing.get("active"))
    if "vspreview_info" in section_states:
        if not use_vspreview:
            mark_section(
                "vspreview_info",
                SectionAvailability.MISSING,
                "VSPreview integration disabled for this configuration.",
            )
        elif missing_active:
            mark_section(
                "vspreview_info",
                SectionAvailability.PARTIAL,
                "VSPreview dependencies missing; install them to surface preview guidance.",
            )
    if "vspreview_missing" in section_states and not missing_active:
        # The dependency warning section should only render when VSPreview is actually missing.
        mark_section(
            "vspreview_missing",
            SectionAvailability.MISSING,
            "VSPreview dependencies detected; hide the dependency warning by default.",
        )


def create_initial_json_tail(
    cfg: AppConfig,
    workspace_root: Path,
    media_root: Path,
    config_path: Path,
    legacy_config: bool,
    offsets_path: Path,
    vspreview_mode_value: str,
    report_enabled: bool,
) -> JsonTail:
    """Build the initial JSON tail dictionary."""
    return {
        "clips": [],
        "trims": {"per_clip": {}},
        "window": {},
        "alignment": {"manual_start_s": 0.0, "manual_end_s": "unchanged"},
        "audio_alignment": {
            "enabled": bool(cfg.audio_alignment.enable),
            "reference_stream": None,
            "target_stream": {},
            "offsets_sec": {},
            "offsets_frames": {},
            "preview_paths": [],
            "confirmed": None,
            "offsets_filename": str(offsets_path),
            "manual_trim_summary": [],
            "suggestion_mode": False,
            "suggested_frames": {},
            "manual_trim_starts": {},
            "use_vspreview": bool(cfg.audio_alignment.use_vspreview),
            "vspreview_script": None,
            "vspreview_invoked": False,
            "vspreview_exit_code": None,
            "vspreview_manual_offsets": {},
            "vspreview_manual_deltas": {},
            "vspreview_reference_trim": None,
        },
        "analysis": {"output_frame_count": 0, "scanned": 0},
        "render": {},
        "tonemap": {},
        "overlay": {},
        "verify": {
            "count": 0,
            "threshold": float(cfg.color.verify_luma_threshold),
            "delta": {
                "max": None,
                "average": None,
                "frame": None,
                "file": None,
                "auto_selected": None,
            },
            "entries": [],
        },
        "cache": {},
        "workspace": {
            "root": str(workspace_root),
            "media_root": str(media_root),
            "config_path": str(config_path),
            "legacy_config": legacy_config,
        },
        "slowpics": {
            "enabled": bool(cfg.slowpics.auto_upload),
            "title": {
                "inputs": {
                    "resolved_base": None,
                    "collection_name": None,
                    "collection_suffix": getattr(cfg.slowpics, "collection_suffix", ""),
                },
                "final": None,
            },
            "url": None,
            "shortcut_path": None,
            "shortcut_written": False,
            "shortcut_error": None,
            "deleted_screens_dir": False,
            "is_public": bool(cfg.slowpics.is_public),
            "is_hentai": bool(cfg.slowpics.is_hentai),
            "remove_after_days": int(cfg.slowpics.remove_after_days),
        },
        "report": {
            "enabled": report_enabled,
            "path": None,
            "output_dir": cfg.report.output_dir,
            "open_after_generate": bool(getattr(cfg.report, "open_after_generate", True)),
            "mode": cfg.report.default_mode,
        },
        "viewer": {
            "mode": "none",
            "mode_display": "None",
            "destination": None,
            "destination_label": "",
        },
        "warnings": [],
        "vspreview_mode": vspreview_mode_value,
        "suggested_frames": None,
        "suggested_seconds": 0.0,
        "vspreview_offer": None,
    }
