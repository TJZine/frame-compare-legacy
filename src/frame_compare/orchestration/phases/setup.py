from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, cast

from src.frame_compare import vs as vs_core
from src.frame_compare.cli_runtime import CLIAppError, JsonTail
from src.frame_compare.orchestration import reporting
from src.frame_compare.orchestration.phases.base import Phase
from src.frame_compare.orchestration.state import CoordinatorContext, RunResult
from src.frame_compare.result_snapshot import (
    RenderOptions,
    ResultSource,
    load_snapshot,
    render_run_result,
)
from src.frame_compare.vspreview import VSPREVIEW_POSIX_INSTALL, VSPREVIEW_WINDOWS_INSTALL

logger = logging.getLogger('frame_compare')


class SetupPhase(Phase):
    def execute(self, context: CoordinatorContext) -> None:
        setup_service = context.dependencies.setup_service
        env = setup_service.prepare_run_environment(context.request)
        context.env = env

        cfg = env.cfg
        reporter = env.reporter
        request = context.request

        reporter.set_flag("progress_style", "fill")
        if hasattr(cfg, "cli"):
            cli_cfg = cfg.cli
            reporter.set_flag("emit_json_tail", bool(getattr(cli_cfg, "emit_json_tail", True)))
            if hasattr(cli_cfg, "progress"):
                style_value = getattr(cli_cfg.progress, "style", "fill")
                progress_style = str(style_value).strip().lower()
                if progress_style not in {"fill", "dot"}:
                    logger.warning(
                        "Invalid progress style '%s', falling back to 'fill'", style_value
                    )
                    progress_style = "fill"
                reporter.set_flag("progress_style", progress_style)

        reporter.set_flag("service_mode_enabled", env.service_mode_enabled)
        publishing_mode = "publisher services"
        if env.legacy_requested:
            publishing_mode += " (legacy runner path retired; legacy override ignored)"
        logger.info("Publishing mode: %s", publishing_mode)
        reporter.verbose_line(f"[runner] Publishing mode: {publishing_mode}")

        raw_layout_sections_obj = getattr(getattr(reporter, "layout", None), "sections", [])
        layout_sections: list[Mapping[str, Any]] = []
        if isinstance(raw_layout_sections_obj, list):
            layout_sources = cast(list[Any], raw_layout_sections_obj)
            for section_obj in layout_sources:
                if isinstance(section_obj, Mapping):
                    layout_sections.append(cast(Mapping[str, Any], section_obj))

        render_options = RenderOptions(
            show_partial=request.show_partial_sections,
            show_missing_sections=request.show_missing_sections,
        )

        # Handle from_cache_only early exit
        if request.from_cache_only:
            result_snapshot_path = env.result_snapshot_path
            cached_snapshot = load_snapshot(result_snapshot_path)
            if cached_snapshot is None:
                raise CLIAppError(
                    f"No cached run result found at {result_snapshot_path}",
                    rich_message=(
                        "[red]Cached run unavailable.[/red] "
                        f"Run without --from-cache-only or delete {result_snapshot_path}."
                    ),
                )
            cached_snapshot.source = ResultSource.CACHE
            render_run_result(
                snapshot=cached_snapshot,
                reporter=reporter,
                layout_sections=layout_sections,
                options=render_options,
            )
            cached_tail = cast(JsonTail, cached_snapshot.json_tail) if cached_snapshot.json_tail else None
            cached_report_path = (
                Path(cached_snapshot.report_path) if cached_snapshot.report_path else None
            )

            # Populate result in context to signal completion
            context.result = RunResult(
                files=[Path(path) for path in cached_snapshot.files],
                frames=list(cached_snapshot.frames),
                out_dir=env.out_dir,
                out_dir_created=False,
                out_dir_created_path=None,
                root=env.root,
                config=cfg,
                image_paths=list(cached_snapshot.image_paths),
                slowpics_url=cached_snapshot.slowpics_url,
                json_tail=cached_tail,
                report_path=cached_report_path,
                snapshot=cached_snapshot,
                snapshot_path=result_snapshot_path,
            )
            return

        # Initialize JSON tail and layout data
        preflight = env.preflight
        workspace_root = preflight.workspace_root
        config_location = preflight.config_path
        offsets_path = env.offsets_path
        vspreview_mode_value = env.vspreview_mode_value
        report_enabled = env.report_enabled
        root = env.root

        json_tail = reporting.create_initial_json_tail(
            cfg=cfg,
            workspace_root=workspace_root,
            media_root=root,
            config_path=config_location,
            legacy_config=bool(preflight.legacy_config),
            offsets_path=offsets_path,
            vspreview_mode_value=vspreview_mode_value,
            report_enabled=report_enabled,
        )
        context.json_tail = json_tail

        vspreview_mode_display = (
            "baseline (0f applied to both clips)"
            if vspreview_mode_value == "baseline"
            else "seeded (suggested offsets applied before preview)"
        )

        layout_data: Dict[str, Any] = {
            "clips": {
                "count": 0,
                "items": [],
                "ref": {},
                "tgt": {},
            },
            "vspreview": {
                "mode": vspreview_mode_value,
                "mode_display": vspreview_mode_display,
                "suggested_frames": None,
                "suggested_seconds": 0.0,
                "script_path": None,
                "script_command": "",
                "missing": {
                    "active": False,
                    "windows_install": VSPREVIEW_WINDOWS_INSTALL,
                    "posix_install": VSPREVIEW_POSIX_INSTALL,
                    "command": "",
                    "reason": "",
                },
                "clips": {
                    "ref": {"label": ""},
                    "tgt": {"label": ""},
                },
            },
            "trims": {},
            "window": json_tail["window"],
            "alignment": json_tail["alignment"],
            "audio_alignment": json_tail["audio_alignment"],
            "analysis": json_tail["analysis"],
            "render": json_tail.get("render", {}),
            "tonemap": json_tail.get("tonemap", {}),
            "overlay": json_tail.get("overlay", {}),
            "verify": json_tail.get("verify", {}),
            "cache": json_tail["cache"],
            "slowpics": json_tail["slowpics"],
            "report": json_tail["report"],
            "tmdb": {
                "category": None,
                "id": None,
                "title": None,
                "year": None,
                "lang": None,
            },
            "overrides": {
                "change_fps": "change_fps" if cfg.overrides.change_fps else "none",
            },
            "viewer": json_tail["viewer"],
            "warnings": [],
        }

        context.layout_data = layout_data
        reporter.update_values(layout_data)
        reporter.set_flag("upload_enabled", bool(cfg.slowpics.auto_upload))
        reporter.set_flag("tmdb_resolved", False)

        vs_core.configure(
            search_paths=cfg.runtime.vapoursynth_python_paths,
            source_preference=cfg.source.preferred,
        )
