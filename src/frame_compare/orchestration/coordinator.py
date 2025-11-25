from __future__ import annotations

import copy
import logging
import math
import time
import traceback
from collections import Counter
from collections.abc import Mapping as MappingABC
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Union, cast

from rich.markup import escape

import src.frame_compare.cache as cache_utils
import src.frame_compare.media as media_utils
import src.frame_compare.metadata as metadata_utils
import src.frame_compare.preflight as preflight_utils
import src.frame_compare.runtime_utils as runtime_utils
import src.frame_compare.selection as selection_utils
import src.frame_compare.vspreview as vspreview_utils
from src.datatypes import AppConfig
from src.frame_compare import vs as vs_core
from src.frame_compare.alignment_helpers import derive_frame_hint
from src.frame_compare.analysis import (
    CacheLoadResult,
    SelectionDetail,
    export_selection_metadata,
    probe_cached_metrics,
    select_frames,
    selection_details_to_json,
    selection_hash_for_config,
    write_selection_cache_file,
)
from src.frame_compare.cli_runtime import (
    CLIAppError,
    CliOutputManagerProtocol,
    ClipPlan,
    ClipRecord,
    JsonTail,
    TrimClipEntry,
    TrimSummary,
    coerce_str_mapping,
)
from src.frame_compare.diagnostics import (
    build_frame_metric_entry,
    classify_color_range,
    extract_dovi_metadata,
    extract_hdr_metadata,
)
from src.frame_compare.layout_utils import plan_label as _plan_label
from src.frame_compare.orchestration import reporting, setup
from src.frame_compare.orchestration.setup import emit_dovi_debug
from src.frame_compare.orchestration.state import (
    RunContext,
    RunDependencies,
    RunRequest,
    RunResult,
)
from src.frame_compare.result_snapshot import (
    RenderOptions,
    ResultSource,
    SectionAvailability,
    SectionState,
    build_snapshot,
    load_snapshot,
    render_run_result,
    resolve_cli_version,
    write_snapshot,
)
from src.frame_compare.services.alignment import AlignmentRequest, AlignmentWorkflow
from src.frame_compare.services.factory import (
    build_alignment_workflow,
    build_metadata_resolver,
    build_report_publisher,
    build_slowpics_publisher,
)
from src.frame_compare.services.metadata import MetadataResolver, MetadataResolveRequest
from src.frame_compare.services.publishers import (
    ReportPublisher,
    ReportPublisherRequest,
    SlowpicsPublisher,
    SlowpicsPublisherRequest,
)
from src.frame_compare.vs import ClipInitError, ClipProcessError
from src.screenshot import ScreenshotError, generate_screenshots

logger = logging.getLogger('frame_compare')


def default_run_dependencies(
    *,
    cfg: AppConfig | None = None,
    reporter: CliOutputManagerProtocol | None = None,
    cache_dir: Path | None = None,
    metadata_resolver: MetadataResolver | None = None,
    alignment_workflow: AlignmentWorkflow | None = None,
    report_publisher: ReportPublisher | None = None,
    slowpics_publisher: SlowpicsPublisher | None = None,
) -> RunDependencies:
    """
    Build the default service bundle used by :func:`run`.

    The cfg/reporter/cache_dir parameters are accepted for future adapter wiring;
    callers may omit them when defaults suffice but tests can inject overrides.
    """

    del cfg, reporter, cache_dir  # Reserved for future dependency wiring.
    return RunDependencies(
        metadata_resolver=metadata_resolver or build_metadata_resolver(),
        alignment_workflow=alignment_workflow or build_alignment_workflow(),
        report_publisher=report_publisher or build_report_publisher(),
        slowpics_publisher=slowpics_publisher or build_slowpics_publisher(),
    )


class WorkflowCoordinator:
    def __init__(self, dependencies: RunDependencies | None = None):
        self.dependencies = dependencies

    def execute(self, request: RunRequest) -> RunResult:
        """Orchestrate the CLI workflow."""
        env = setup.prepare_run_environment(request)

        cfg = env.cfg
        root = env.root
        out_dir = env.out_dir
        reporter = env.reporter
        result_snapshot_path = env.result_snapshot_path
        collected_warnings = env.collected_warnings
        report_enabled = env.report_enabled
        preflight = env.preflight
        workspace_root = preflight.workspace_root
        config_location = preflight.config_path
        offsets_path = env.offsets_path
        vspreview_mode_value = env.vspreview_mode_value
        analysis_cache_path = env.analysis_cache_path
        service_mode_enabled = env.service_mode_enabled
        legacy_requested = env.legacy_requested
        created_out_dir = env.out_dir_created
        created_out_dir_path = env.out_dir_created_path
        debug_color_enabled = bool(getattr(cfg.color, "debug_color", False))
        entrypoint_name = (
            getattr(request.impl_module, "__name__", "runner") if request.impl_module else "runner"
        )

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

        reporter.set_flag("service_mode_enabled", service_mode_enabled)
        publishing_mode = "publisher services"
        if legacy_requested:
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

        if request.from_cache_only:
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
            return RunResult(
                files=[Path(path) for path in cached_snapshot.files],
                frames=list(cached_snapshot.frames),
                out_dir=out_dir,
                out_dir_created=False,
                out_dir_created_path=None,
                root=root,
                config=cfg,
                image_paths=list(cached_snapshot.image_paths),
                slowpics_url=cached_snapshot.slowpics_url,
                json_tail=cached_tail,
                report_path=cached_report_path,
                snapshot=cached_snapshot,
                snapshot_path=result_snapshot_path,
            )

        service_deps = self.dependencies or default_run_dependencies(
            cfg=cfg,
            reporter=reporter,
            cache_dir=root,
        )
        metadata_resolver = service_deps.metadata_resolver
        alignment_workflow = service_deps.alignment_workflow
        report_publisher = service_deps.report_publisher
        slowpics_publisher = service_deps.slowpics_publisher

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

        audio_track_overrides = request.audio_track_overrides
        audio_track_override_map = metadata_utils.parse_audio_track_overrides(audio_track_overrides or [])

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
                    "windows_install": vspreview_utils.VSPREVIEW_WINDOWS_INSTALL,
                    "posix_install": vspreview_utils.VSPREVIEW_POSIX_INSTALL,
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

        reporter.update_values(layout_data)
        reporter.set_flag("upload_enabled", bool(cfg.slowpics.auto_upload))
        reporter.set_flag("tmdb_resolved", False)

        vs_core.configure(
            search_paths=cfg.runtime.vapoursynth_python_paths,
            source_preference=cfg.source.preferred,
        )

        try:
            files = media_utils.discover_media(root)
        except OSError as exc:
            raise CLIAppError(
                f"Failed to list input directory: {exc}",
                rich_message=f"[red]Failed to list input directory:[/red] {exc}",
            ) from exc

        if len(files) < 2:
            raise CLIAppError(
                "Need at least two video files to compare.",
                rich_message="[red]Need at least two video files to compare.[/red]",
            )

        metadata_request = MetadataResolveRequest(
            cfg=cfg,
            root=root,
            files=files,
            reporter=reporter,
            json_tail=json_tail,
            layout_data=layout_data,
            collected_warnings=collected_warnings,
        )
        metadata_result = metadata_resolver.resolve(metadata_request)
        context = RunContext(
            plans=list(metadata_result.plans),
            metadata=list(metadata_result.metadata),
            json_tail=json_tail,
            layout_data=layout_data,
            metadata_title=metadata_result.metadata_title,
            analyze_path=metadata_result.analyze_path,
            slowpics_title_inputs=metadata_result.slowpics_title_inputs,
            slowpics_final_title=metadata_result.slowpics_final_title,
            slowpics_resolved_base=metadata_result.slowpics_resolved_base,
            slowpics_tmdb_disclosure_line=metadata_result.slowpics_tmdb_disclosure_line,
            slowpics_verbose_tmdb_tag=metadata_result.slowpics_verbose_tmdb_tag,
            tmdb_notes=list(metadata_result.tmdb_notes),
        )
        analyze_path = context.analyze_path
        plans = context.plans

        alignment_request = AlignmentRequest(
            plans=plans,
            cfg=cfg,
            root=root,
            analyze_path=context.analyze_path,
            audio_track_overrides=audio_track_override_map,
            reporter=reporter,
            json_tail=json_tail,
            vspreview_mode=vspreview_mode_value,
            collected_warnings=collected_warnings,
        )
        alignment_result = alignment_workflow.run(alignment_request)
        context.update_alignment(alignment_result)
        plans = context.plans
        alignment_summary = context.alignment_summary
        alignment_display = context.alignment_display

        try:
            selection_utils.init_clips(plans, cfg.runtime, root, reporter=reporter)
        except ClipInitError as exc:
            raise CLIAppError(
                f"Failed to open clip: {exc}", rich_message=f"[red]Failed to open clip:[/red] {exc}"
            ) from exc

        clips = [plan.clip for plan in plans]
        if any(clip is None for clip in clips):
            raise CLIAppError("Clip initialisation failed")
        stored_props_seq = [
            dict(plan.source_frame_props) if plan.source_frame_props is not None else None
            for plan in plans
        ]

        clip_records: List[ClipRecord] = []
        trim_details: List[TrimSummary] = []
        for idx, plan in enumerate(plans):
            label = (plan.metadata.get("label") or plan.path.name).strip()
            frames_total = int(plan.source_num_frames or getattr(plan.clip, "num_frames", 0) or 0)
            width = int(plan.source_width or getattr(plan.clip, "width", 0) or 0)
            height = int(plan.source_height or getattr(plan.clip, "height", 0) or 0)
            fps_tuple = plan.effective_fps or plan.source_fps or (24000, 1001)
            fps_float = runtime_utils.fps_to_float(fps_tuple)
            duration_seconds = frames_total / fps_float if fps_float > 0 else 0.0
            clip_records.append(
                {
                    "label": label,
                    "width": width,
                    "height": height,
                    "fps": fps_float,
                    "frames": frames_total,
                    "duration": duration_seconds,
                    "duration_tc": runtime_utils.format_seconds(duration_seconds),
                    "path": str(plan.path),
                }
            )
            clip_entry: dict[str, Any] = {
                "label": label,
                "width": width,
                "height": height,
                "fps": fps_float,
                "frames": frames_total,
                "duration_s": duration_seconds,
                "duration_tc": runtime_utils.format_seconds(duration_seconds),
                "path": str(plan.path),
            }
            props_snapshot = stored_props_seq[idx] or {}
            hdr_meta = extract_hdr_metadata(props_snapshot)
            if any(value is not None for value in hdr_meta.values()):
                clip_entry["hdr_metadata"] = {k: v for k, v in hdr_meta.items() if v is not None}
            range_label = classify_color_range(props_snapshot)
            if range_label:
                clip_entry["dynamic_range"] = {"label": range_label, "source": "frame_props"}
            json_tail["clips"].append(clip_entry)

            lead_frames = max(0, int(plan.trim_start))
            lead_seconds = lead_frames / fps_float if fps_float > 0 else 0.0
            trail_frames = 0
            if plan.trim_end is not None and plan.trim_end != 0:
                if plan.trim_end < 0:
                    trail_frames = abs(int(plan.trim_end))
                else:
                    trail_frames = 0
            trail_seconds = trail_frames / fps_float if fps_float > 0 else 0.0
            trim_details.append(
                {
                    "label": label,
                    "lead_frames": lead_frames,
                    "lead_seconds": lead_seconds,
                    "trail_frames": trail_frames,
                    "trail_seconds": trail_seconds,
                }
            )
            clip_trim: TrimClipEntry = {
                "lead_f": lead_frames,
                "trail_f": trail_frames,
                "lead_s": lead_seconds,
                "trail_s": trail_seconds,
            }
            json_tail["trims"]["per_clip"][label] = clip_trim

        analyze_index = [plan.path for plan in plans].index(analyze_path)
        analyze_clip = plans[analyze_index].clip
        if analyze_clip is None:
            raise CLIAppError("Missing clip for analysis")

        selection_specs, frame_window, windows_collapsed = selection_utils.resolve_selection_windows(
            plans, cfg.analysis
        )
        analyze_fps_num, analyze_fps_den = plans[analyze_index].effective_fps or selection_utils.extract_clip_fps(
            analyze_clip
        )
        analyze_fps = analyze_fps_num / analyze_fps_den if analyze_fps_den else 0.0
        manual_start_frame, manual_end_frame = frame_window
        analyze_total_frames = clip_records[analyze_index]["frames"]
        manual_start_seconds_value = manual_start_frame / analyze_fps if analyze_fps > 0 else float(manual_start_frame)
        manual_end_seconds_value = manual_end_frame / analyze_fps if analyze_fps > 0 else float(manual_end_frame)
        manual_end_changed = manual_end_frame < analyze_total_frames
        json_tail["alignment"] = {
            "manual_start_s": manual_start_seconds_value,
            "manual_end_s": manual_end_seconds_value if manual_end_changed else None,
        }
        layout_data["alignment"] = json_tail["alignment"]

        json_tail["window"] = {
            "ignore_lead_seconds": float(cfg.analysis.ignore_lead_seconds),
            "ignore_trail_seconds": float(cfg.analysis.ignore_trail_seconds),
            "min_window_seconds": float(cfg.analysis.min_window_seconds),
        }
        layout_data["window"] = json_tail["window"]

        for plan, spec in zip(plans, selection_specs):
            if not spec.warnings:
                continue
            label = plan.metadata.get("label") or plan.path.name
            for warning in spec.warnings:
                message = f"Window warning for {label}: {warning}"
                collected_warnings.append(message)
        if windows_collapsed:
            message = "Ignore lead/trail settings did not overlap across all sources; using fallback range."
            collected_warnings.append(message)

        cache_info = cache_utils.build_cache_info(root, plans, cfg, analyze_index)

        cache_filename = cfg.analysis.frame_data_filename
        cache_status = "disabled"
        cache_reason = None
        cache_progress_message: Optional[str] = None
        cache_probe: CacheLoadResult | None = None

        if not cfg.analysis.save_frames_data:
            cache_status = "disabled"
            cache_reason = "save_frames_data=false"
        elif cache_info is None:
            cache_status = "disabled"
            cache_reason = "no_cache_info"
        else:
            cache_path = cache_info.path
            if request.force_cache_refresh:
                cache_status = "recomputed"
                cache_reason = "forced_refresh"
                cache_probe = CacheLoadResult(metrics=None, status="missing", reason="forced_refresh")
                reporter.line("[yellow]Ignoring cached frame metrics (--no-cache requested).[/]")
                cache_progress_message = "Recomputing frame metrics (forced refresh)."
            elif cache_path.exists():
                probe_result = probe_cached_metrics(cache_info, cfg.analysis)
                cache_probe = probe_result
                if probe_result.status == "reused":
                    cache_status = "reused"
                    cache_progress_message = (
                        f"Loading cached frame metrics from {cache_path.name}…"
                    )
                    reporter.line(
                        f"[green]Reused cached frame metrics from {escape(cache_path.name)}[/]"
                    )
                else:
                    cache_status = "recomputed"
                    reason_code = probe_result.reason or probe_result.status or "unknown"
                    cache_reason = reason_code
                    human_reason = reason_code.replace("_", " ").strip() or "unknown"
                    reporter.line(
                        f"[yellow]Frame metrics cache {probe_result.status} "
                        f"({escape(human_reason)}); recomputing…[/]"
                    )
                    cache_progress_message = f"Recomputing frame metrics ({human_reason})…"
            else:
                cache_status = "recomputed"
                cache_reason = "missing"
                cache_probe = CacheLoadResult(metrics=None, status="missing", reason="missing")
                reporter.line("[yellow]Frame metrics cache missing; recomputing…[/]")
                cache_progress_message = "Recomputing frame metrics (missing)…"

        json_tail["cache"] = {
            "file": cache_filename,
            "status": cache_status,
        }
        if cache_reason:
            json_tail["cache"]["reason"] = cache_reason
        layout_data["cache"] = json_tail["cache"]

        step_size = max(1, int(cfg.analysis.step))
        total_frames = getattr(analyze_clip, 'num_frames', 0)
        sample_count = 0
        if isinstance(total_frames, int) and total_frames > 0:
            sample_count = (total_frames + step_size - 1) // step_size

        analyze_label_raw = plans[analyze_index].metadata.get('label') or analyze_path.name

        cache_ready = cache_probe is not None and cache_probe.status == "reused"
        if cache_ready and cache_info is not None:
            reporter.verbose_line(f"Using cached frame metrics: {cache_info.path.name}")
        elif cache_status == "recomputed" and cache_info is not None:
            reporter.verbose_line(f"Frame metrics cache will be refreshed: {cache_info.path.name}")
        overrides_text = "change_fps" if cfg.overrides.change_fps else "none"
        layout_data["overrides"]["change_fps"] = overrides_text

        analysis_method = "absdiff" if cfg.analysis.motion_use_absdiff else "edge"
        thresholds_cfg = cfg.analysis.thresholds
        threshold_mode_value = str(getattr(thresholds_cfg.mode, "value", thresholds_cfg.mode))
        threshold_mode_lower = threshold_mode_value.lower()
        threshold_payload: Dict[str, float | str] = {"mode": threshold_mode_value}
        if threshold_mode_lower == "quantile":
            threshold_payload.update(
                {
                    "dark_quantile": float(thresholds_cfg.dark_quantile),
                    "bright_quantile": float(thresholds_cfg.bright_quantile),
                }
            )
        else:
            threshold_payload.update(
                {
                    "dark_luma_min": float(thresholds_cfg.dark_luma_min),
                    "dark_luma_max": float(thresholds_cfg.dark_luma_max),
                    "bright_luma_min": float(thresholds_cfg.bright_luma_min),
                    "bright_luma_max": float(thresholds_cfg.bright_luma_max),
                }
            )

        json_tail["analysis"] = {
            "step": int(cfg.analysis.step),
            "downscale_height": int(cfg.analysis.downscale_height),
            "motion_method": analysis_method,
            "motion_scenecut_quantile": float(cfg.analysis.motion_scenecut_quantile),
            "motion_diff_radius": int(cfg.analysis.motion_diff_radius),
            "output_frame_count": 0,
            "counts": {
                "dark": int(cfg.analysis.frame_count_dark),
                "bright": int(cfg.analysis.frame_count_bright),
                "motion": int(cfg.analysis.frame_count_motion),
                "random": int(cfg.analysis.random_frames),
                "user": len(cfg.analysis.user_frames),
            },
            "screen_separation_sec": float(cfg.analysis.screen_separation_sec),
            "random_seed": int(cfg.analysis.random_seed),
            "thresholds": threshold_payload,
        }

        json_tail["analysis"]["cache_reused"] = bool(cache_ready)
        if cache_progress_message:
            json_tail["analysis"]["cache_progress_message"] = cache_progress_message

        layout_data["analysis"] = dict(json_tail["analysis"])
        emit_dovi_debug(
            {
                "phase": "analysis_cache",
                "entrypoint": entrypoint_name,
                "cache_status": cache_status,
                "cache_reason": cache_reason,
                "cache_ready": bool(cache_ready),
                "cache_probe_status": getattr(cache_probe, "status", None) if cache_probe else None,
                "cache_probe_reason": getattr(cache_probe, "reason", None) if cache_probe else None,
                "cache_file": cache_info.path if cache_info is not None else None,
                "analysis_cache_path": analysis_cache_path,
                "json_cache_reused": json_tail["analysis"]["cache_reused"],
            }
        )
        layout_data["clips"]["count"] = len(clip_records)
        layout_data["clips"]["items"] = clip_records
        layout_data["clips"]["ref"] = clip_records[0] if clip_records else {}
        layout_data["clips"]["tgt"] = clip_records[1] if len(clip_records) > 1 else {}

        reference_label = ""
        if alignment_summary is not None:
            reference_label = _plan_label(alignment_summary.reference_plan)
        elif clip_records:
            reference_label = clip_records[0]["label"]

        vspreview_target_plan: ClipPlan | None = None
        tail_suggested_frames: object | None = json_tail.get("suggested_frames")
        tail_suggested_seconds: object | None = json_tail.get("suggested_seconds")

        vspreview_suggested_frames_value: int | None = None
        if tail_suggested_frames is not None:
            try:
                vspreview_suggested_frames_value = int(cast(Union[int, float, str], tail_suggested_frames))
            except (TypeError, ValueError):
                vspreview_suggested_frames_value = None
        vspreview_suggested_seconds_value = 0.0
        tail_seconds_provided = True
        try:
            vspreview_suggested_seconds_value = float(cast(Union[int, float, str], tail_suggested_seconds))
        except (TypeError, ValueError):
            tail_seconds_provided = False
            vspreview_suggested_seconds_value = 0.0

        if alignment_summary is not None:
            for plan in plans:
                if plan is alignment_summary.reference_plan:
                    continue
                vspreview_target_plan = plan
                clip_key = plan.path.name
                if vspreview_suggested_frames_value is None:
                    vspreview_suggested_frames_value = derive_frame_hint(alignment_summary, clip_key)
                if not tail_seconds_provided:
                    detail = alignment_summary.measured_offsets.get(clip_key)
                    if detail and detail.offset_seconds is not None:
                        vspreview_suggested_seconds_value = float(detail.offset_seconds)
                    else:
                        measurement_lookup = {
                            measurement.file.name: measurement
                            for measurement in alignment_summary.measurements
                        }
                        measurement = measurement_lookup.get(clip_key)
                        if measurement and measurement.offset_seconds is not None:
                            vspreview_suggested_seconds_value = float(measurement.offset_seconds)
                break

        target_label = ""
        if vspreview_target_plan is not None:
            target_label = _plan_label(vspreview_target_plan)
        elif len(clip_records) > 1:
            target_label = clip_records[1]["label"]

        vspreview_block = coerce_str_mapping(layout_data.get("vspreview"))
        clips_block = coerce_str_mapping(vspreview_block.get("clips"))
        clips_block["ref"] = {"label": reference_label}
        clips_block["tgt"] = {"label": target_label}
        vspreview_block["clips"] = clips_block
        vspreview_block["mode"] = vspreview_mode_value
        vspreview_block["mode_display"] = vspreview_mode_display
        vspreview_block["suggested_frames"] = vspreview_suggested_frames_value
        vspreview_block["suggested_seconds"] = vspreview_suggested_seconds_value
        existing_vspreview_obj = reporter.values.get("vspreview")
        if isinstance(existing_vspreview_obj, MappingABC):
            existing_vspreview = coerce_str_mapping(cast(Mapping[str, object], existing_vspreview_obj))
            missing_existing_block = coerce_str_mapping(existing_vspreview.get("missing"))
            if missing_existing_block:
                missing_layout_block = coerce_str_mapping(vspreview_block.get("missing"))
                merged_missing_block = missing_layout_block.copy()
                merged_missing_block.update(missing_existing_block)
                vspreview_block["missing"] = merged_missing_block
            script_path_value = existing_vspreview.get("script_path")
            if isinstance(script_path_value, str) and script_path_value:
                vspreview_block["script_path"] = script_path_value
            script_command_value = existing_vspreview.get("script_command")
            if isinstance(script_command_value, str) and script_command_value:
                vspreview_block["script_command"] = script_command_value
            for key, value in existing_vspreview.items():
                if key in {"clips", "missing", "script_path", "script_command"}:
                    continue
                if key not in vspreview_block:
                    vspreview_block[key] = value
        layout_data["vspreview"] = vspreview_block

        trims_per_clip = json_tail["trims"]["per_clip"]
        trim_lookup: dict[str, TrimSummary] = {detail["label"]: detail for detail in trim_details}

        def _trim_entry(label: str) -> TrimClipEntry:
            """
            Build a normalized trim entry for a clip label containing frame and second offsets.

            Parameters:
                label (str): Clip label used to look up trim and detailed timing information.

            Returns:
                dict: Mapping with keys:
                    - "lead_f": number of leading frames trimmed (int, default 0)
                    - "trail_f": number of trailing frames trimmed (int, default 0)
                    - "lead_s": leading trim in seconds (float, default 0.0)
                    - "trail_s": trailing trim in seconds (float, default 0.0)
            """
            trim = trims_per_clip.get(label)
            detail = trim_lookup.get(label)
            return {
                "lead_f": trim["lead_f"] if trim else 0,
                "trail_f": trim["trail_f"] if trim else 0,
                "lead_s": detail["lead_seconds"] if detail else 0.0,
                "trail_s": detail["trail_seconds"] if detail else 0.0,
            }

        layout_data["trims"] = {}
        if clip_records:
            layout_data["trims"]["ref"] = _trim_entry(clip_records[0]["label"])
        if len(clip_records) > 1:
            layout_data["trims"]["tgt"] = _trim_entry(clip_records[1]["label"])

        reporter.update_values(layout_data)
        if context.tmdb_notes:
            for note in context.tmdb_notes:
                reporter.verbose_line(note)
        if context.slowpics_tmdb_disclosure_line:
            reporter.verbose_line(context.slowpics_tmdb_disclosure_line)

        if alignment_display is not None:
            json_tail["audio_alignment"]["preview_paths"] = alignment_display.preview_paths
            confirmation_value = alignment_display.confirmation
            if confirmation_value is None and alignment_summary is not None:
                confirmation_value = "auto"
            json_tail["audio_alignment"]["confirmed"] = confirmation_value
        audio_alignment_view = copy.deepcopy(json_tail["audio_alignment"])
        audio_alignment_layout = cast(dict[str, object], audio_alignment_view)
        offsets_sec_map_obj = coerce_str_mapping(audio_alignment_layout.get("offsets_sec"))
        offsets_frames_map_obj = coerce_str_mapping(audio_alignment_layout.get("offsets_frames"))
        if alignment_display is not None:
            correlations_map: dict[str, float] = dict(alignment_display.correlations)
        else:
            correlations_map = {}
        primary_label: str | None = None
        if offsets_sec_map_obj:
            primary_label = sorted(offsets_sec_map_obj.keys())[0]
        offsets_sec_value_obj = offsets_sec_map_obj.get(primary_label) if primary_label else None
        offsets_sec_value = (
            float(offsets_sec_value_obj)
            if isinstance(offsets_sec_value_obj, (int, float))
            else 0.0
        )
        offsets_frames_value_obj = offsets_frames_map_obj.get(primary_label) if primary_label else None
        offsets_frames_value = (
            int(offsets_frames_value_obj)
            if isinstance(offsets_frames_value_obj, (int, float))
            else 0
        )
        corr_value_obj = correlations_map.get(primary_label) if primary_label else None
        corr_value = (
            float(corr_value_obj)
            if isinstance(corr_value_obj, (int, float))
            else 0.0
        )
        if math.isnan(corr_value):
            corr_value = 0.0
        threshold_value = float(getattr(alignment_display, "threshold", cfg.audio_alignment.correlation_threshold))
        audio_alignment_layout.update(
            {
                "offsets_sec": offsets_sec_value,
                "offsets_frames": offsets_frames_value,
                "corr": corr_value,
                "threshold": threshold_value,
            }
        )
        audio_alignment_layout["measurements"] = json_tail["audio_alignment"].get("measurements", {})
        layout_data["audio_alignment"] = audio_alignment_layout

        using_frame_total = isinstance(total_frames, int) and total_frames > 0
        progress_total = int(total_frames) if using_frame_total else int(sample_count)

        def _run_selection(
            progress_callback: Callable[[int], None] | None = None,
        ) -> tuple[list[int], dict[int, str], Dict[int, SelectionDetail]]:
            try:
                result = select_frames(
                    analyze_clip,
                    cfg.analysis,
                    [plan.path.name for plan in plans],
                    analyze_path.name,
                    cache_info=cache_info,
                    progress=progress_callback,
                    frame_window=frame_window,
                    return_metadata=True,
                    color_cfg=cfg.color,
                    cache_probe=cache_probe,
                )
            except TypeError as exc:
                if "return_metadata" not in str(exc):
                    raise
                result = select_frames(
                    analyze_clip,
                    cfg.analysis,
                    [plan.path.name for plan in plans],
                    analyze_path.name,
                    cache_info=cache_info,
                    progress=progress_callback,
                    frame_window=frame_window,
                    color_cfg=cfg.color,
                    cache_probe=cache_probe,
                )
            if isinstance(result, tuple):
                if len(result) == 3:
                    return result
                if len(result) == 2:
                    frames_only, categories = cast(
                        tuple[list[int], dict[int, str]], result
                    )
                    return frames_only, categories, {}
                frames_iter = cast(Iterable[int], result)
                frames_only = list(frames_iter)
                return frames_only, {frame: "Auto" for frame in frames_only}, {}
            frames_only = list(result)
            return frames_only, {frame: "Auto" for frame in frames_only}, {}

        selection_details: Dict[int, SelectionDetail] = {}

        try:
            if sample_count > 0 and not cache_ready:
                start_time = time.perf_counter()
                samples_done = 0
                reporter.update_progress_state(
                    "analyze_bar",
                    fps="0.00 fps",
                    eta_tc="--:--",
                    elapsed_tc="00:00",
                )
                with reporter.create_progress("analyze_bar", transient=False) as analysis_progress:
                    task_id = analysis_progress.add_task(
                        analyze_label_raw,
                        total=max(1, progress_total),
                    )

                    def _advance_samples(count: int) -> None:
                        """Advance sample counter, update stats, and refresh progress displays."""
                        nonlocal samples_done
                        samples_done += count
                        if progress_total <= 0:
                            return
                        elapsed = max(time.perf_counter() - start_time, 1e-6)
                        frames_processed = samples_done * step_size
                        completed = (
                            min(progress_total, frames_processed)
                            if using_frame_total
                            else min(progress_total, samples_done)
                        )
                        fps_val = frames_processed / elapsed
                        remaining = max(progress_total - completed, 0)
                        eta_seconds = (remaining / fps_val) if fps_val > 0 else None
                        reporter.update_progress_state(
                            "analyze_bar",
                            fps=f"{fps_val:7.2f} fps",
                            eta_tc=runtime_utils.format_clock(eta_seconds),
                            elapsed_tc=runtime_utils.format_clock(elapsed),
                        )
                        analysis_progress.update(task_id, completed=completed)

                    frames, frame_categories, selection_details = _run_selection(_advance_samples)
                    final_completed = progress_total if progress_total > 0 else analysis_progress.tasks[task_id].completed
                    analysis_progress.update(task_id, completed=final_completed)
            else:
                frames, frame_categories, selection_details = _run_selection()

        except Exception as exc:
            tb = traceback.format_exc()
            reporter.console.print("[red]Frame selection trace:[/red]")
            reporter.console.print(tb)
            logger.error("Frame selection trace:\\n%s", tb)
            raise CLIAppError(
                f"Frame selection failed: {exc}",
                rich_message=f"[red]Frame selection failed:[/red] {exc}",
            ) from exc

        if not frames:
            raise CLIAppError(
                "No frames were selected; cannot continue.",
                rich_message="[red]No frames were selected; cannot continue.[/red]",
            )

        selection_hash_value = selection_hash_for_config(cfg.analysis)
        clip_paths = [plan.path for plan in plans]
        selection_sidecar_dir = cache_info.path.parent if cache_info is not None else root
        selection_sidecar_path = selection_sidecar_dir / "generated.selection.v1.json"
        selection_overlay_details: dict[int, dict[str, Any]] = {
            frame: {
                "label": detail.label,
                "timecode": detail.timecode,
                "source": detail.source,
                "score": detail.score,
                "notes": detail.notes,
            }
            for frame, detail in selection_details.items()
        }
        if cache_info is None or not cfg.analysis.save_frames_data:
            export_selection_metadata(
                selection_sidecar_path,
                analyzed_file=analyze_path.name,
                clip_paths=clip_paths,
                cfg=cfg.analysis,
                selection_hash=selection_hash_value,
                selection_frames=frames,
                selection_details=selection_details,
            )
        if not cfg.analysis.save_frames_data:
            compframes_path = preflight_utils.resolve_subdir(
                root,
                cfg.analysis.frame_data_filename,
                purpose="analysis.frame_data_filename",
            )
            write_selection_cache_file(
                compframes_path,
                analyzed_file=analyze_path.name,
                clip_paths=clip_paths,
                cfg=cfg.analysis,
                selection_hash=selection_hash_value,
                selection_frames=frames,
                selection_details=selection_details,
                selection_categories=frame_categories,
            )
        kept_count = len(frames)
        scanned_count = progress_total if progress_total > 0 else max(sample_count, kept_count)
        selection_counts = Counter(detail.label or "Auto" for detail in selection_details.values())
        json_tail["analysis"]["selection_counts"] = dict(selection_counts)
        json_tail["analysis"]["selection_hash"] = selection_hash_value
        json_tail["analysis"]["selection_sidecar"] = str(selection_sidecar_path)
        json_tail["analysis"]["selection_details"] = selection_details_to_json(
            selection_details
        )
        cache_summary_label = "reused" if cache_status == "reused" else ("new" if cache_status == "recomputed" else cache_status)
        json_tail["analysis"]["kept"] = kept_count
        json_tail["analysis"]["scanned"] = scanned_count
        layout_data["analysis"]["kept"] = kept_count
        layout_data["analysis"]["scanned"] = scanned_count
        layout_data["analysis"]["cache_summary_label"] = cache_summary_label
        layout_data["analysis"]["selection_counts"] = dict(selection_counts)
        layout_data["analysis"]["selection_hash"] = selection_hash_value
        layout_data["analysis"]["selection_sidecar"] = str(selection_sidecar_path)
        layout_data["analysis"]["selection_details"] = json_tail["analysis"]["selection_details"]
        reporter.update_values(layout_data)

        preview_rule: dict[str, Any] = {}
        layout_obj = getattr(reporter, "layout", None)
        folding_rules_obj = getattr(layout_obj, "folding", None)
        if isinstance(folding_rules_obj, Mapping):
            folding_rules_map = cast(Mapping[str, object], folding_rules_obj)
            frames_preview_obj = folding_rules_map.get("frames_preview")
            if isinstance(frames_preview_obj, Mapping):
                preview_rule = dict(cast(Mapping[str, Any], frames_preview_obj))
        head_raw: Any = preview_rule["head"] if "head" in preview_rule else None
        tail_raw: Any = preview_rule["tail"] if "tail" in preview_rule else None
        when_raw: Any = preview_rule["when"] if "when" in preview_rule else None
        head = int(head_raw) if isinstance(head_raw, (int, float)) else 4
        tail = int(tail_raw) if isinstance(tail_raw, (int, float)) else 4
        joiner = str(preview_rule["joiner"] if "joiner" in preview_rule else ", ")
        when_text = str(when_raw) if isinstance(when_raw, str) and when_raw else None
        fold_enabled = runtime_utils.evaluate_rule_condition(when_text, flags=reporter.flags)
        preview_text = runtime_utils.fold_sequence(frames, head=head, tail=tail, joiner=joiner, enabled=fold_enabled)

        json_tail["analysis"]["output_frame_count"] = kept_count
        json_tail["analysis"]["output_frames"] = list(frames)
        json_tail["analysis"]["output_frames_preview"] = preview_text
        layout_data["analysis"]["output_frame_count"] = kept_count
        layout_data["analysis"]["output_frames_preview"] = preview_text
        emit_json_tail_flag = reporter.flags.get("emit_json_tail", True)
        if not emit_json_tail_flag:
            full_list_text = ", ".join(str(frame) for frame in frames)
            layout_data["analysis"]["output_frames_full"] = (
                f"[{full_list_text}]" if full_list_text else "[]"
            )
        reporter.update_values(layout_data)

        total_screens = len(frames) * len(plans)

        writer_name = "ffmpeg" if (cfg.screenshots.use_ffmpeg and not debug_color_enabled) else "vs"
        overlay_mode_value = getattr(cfg.color, "overlay_mode", "minimal")
        auto_letterbox_raw = getattr(cfg.screenshots, "auto_letterbox_crop", "off")
        if isinstance(auto_letterbox_raw, Enum):
            auto_letterbox_value = auto_letterbox_raw.value
        else:
            auto_letterbox_value = str(auto_letterbox_raw)

        json_tail["render"] = {
            "writer": writer_name,
            "out_dir": str(out_dir),
            "add_frame_info": bool(cfg.screenshots.add_frame_info),
            "single_res": int(cfg.screenshots.single_res),
            "upscale": bool(cfg.screenshots.upscale),
            "mod_crop": int(cfg.screenshots.mod_crop),
            "letterbox_pillarbox_aware": bool(cfg.screenshots.letterbox_pillarbox_aware),
            "auto_letterbox_mode": auto_letterbox_value,
            "pad_to_canvas": cfg.screenshots.pad_to_canvas,
            "center_pad": bool(cfg.screenshots.center_pad),
            "letterbox_px_tolerance": int(cfg.screenshots.letterbox_px_tolerance),
            "compression": int(cfg.screenshots.compression_level),
            "ffmpeg_timeout_seconds": float(cfg.screenshots.ffmpeg_timeout_seconds),
        }
        layout_data["render"] = json_tail["render"]
        emit_dovi_debug(
            {
                "phase": "render_setup",
                "entrypoint": entrypoint_name,
                "writer": writer_name,
                "total_screens": total_screens,
                "analysis_cache_reused": json_tail.get("analysis", {}).get("cache_reused"),
                "target_nits_cfg": float(getattr(cfg.color, "target_nits", 0.0)),
                "contrast_recovery_cfg": float(getattr(cfg.color, "contrast_recovery", 0.0)),
                "post_gamma_cfg": float(getattr(cfg.color, "post_gamma", 1.0)),
                "post_gamma_enabled_cfg": bool(getattr(cfg.color, "post_gamma_enable", False)),
                "use_dovi_cfg": getattr(cfg.color, "use_dovi", None),
            }
        )
        effective_tonemap_props = plans[0].source_frame_props if plans else None
        effective_tonemap = vs_core.resolve_effective_tonemap(cfg.color, props=effective_tonemap_props)
        json_tail["tonemap"] = {
            "preset": effective_tonemap.get("preset", cfg.color.preset),
            "tone_curve": effective_tonemap.get("tone_curve", cfg.color.tone_curve),
            "dynamic_peak_detection": bool(effective_tonemap.get("dynamic_peak_detection", cfg.color.dynamic_peak_detection)),
            "dpd": bool(effective_tonemap.get("dynamic_peak_detection", cfg.color.dynamic_peak_detection)),
            "target_nits": float(effective_tonemap.get("target_nits", cfg.color.target_nits)),
            "dst_min_nits": float(effective_tonemap.get("dst_min_nits", cfg.color.dst_min_nits)),
            "knee_offset": float(effective_tonemap.get("knee_offset", getattr(cfg.color, "knee_offset", 0.5))),
            "dpd_preset": effective_tonemap.get("dpd_preset", getattr(cfg.color, "dpd_preset", "")),
            "dpd_black_cutoff": float(effective_tonemap.get("dpd_black_cutoff", getattr(cfg.color, "dpd_black_cutoff", 0.0))),
            "verify_luma_threshold": float(cfg.color.verify_luma_threshold),
            "overlay_enabled": bool(cfg.color.overlay_enabled),
            "overlay_mode": overlay_mode_value,
            "post_gamma": float(getattr(cfg.color, "post_gamma", 1.0)),
            "post_gamma_enabled": bool(getattr(cfg.color, "post_gamma_enable", False)),
            "smoothing_period": float(effective_tonemap.get("smoothing_period", getattr(cfg.color, "smoothing_period", 45.0))),
            "scene_threshold_low": float(effective_tonemap.get("scene_threshold_low", getattr(cfg.color, "scene_threshold_low", 0.8))),
            "scene_threshold_high": float(effective_tonemap.get("scene_threshold_high", getattr(cfg.color, "scene_threshold_high", 2.4))),
            "percentile": float(effective_tonemap.get("percentile", getattr(cfg.color, "percentile", 99.995))),
            "contrast_recovery": float(effective_tonemap.get("contrast_recovery", getattr(cfg.color, "contrast_recovery", 0.3))),
            "metadata": effective_tonemap.get("metadata", getattr(cfg.color, "metadata", "auto")),
            "use_dovi": effective_tonemap.get("use_dovi", getattr(cfg.color, "use_dovi", None)),
            "visualize_lut": bool(effective_tonemap.get("visualize_lut", getattr(cfg.color, "visualize_lut", False))),
            "show_clipping": bool(effective_tonemap.get("show_clipping", getattr(cfg.color, "show_clipping", False))),
        }
        emit_dovi_debug(
            {
                "phase": "build_json_tonemap",
                "entrypoint": entrypoint_name,
                "cfg_use_dovi": getattr(cfg.color, "use_dovi", None),
                "effective_use_dovi": effective_tonemap.get("use_dovi"),
                "json_use_dovi": json_tail["tonemap"]["use_dovi"],
                "target_nits": json_tail["tonemap"]["target_nits"],
                "contrast_recovery": json_tail["tonemap"]["contrast_recovery"],
                "post_gamma": json_tail["tonemap"]["post_gamma"],
            }
        )
        metadata_code = json_tail["tonemap"]["metadata"]
        metadata_label_map = {
            0: "auto",
            1: "none",
            2: "hdr10",
            3: "hdr10+",
            4: "luminance",
        }
        if isinstance(metadata_code, int):
            metadata_label = metadata_label_map.get(metadata_code, "auto")
        elif isinstance(metadata_code, str):
            metadata_label = metadata_code
        else:
            metadata_label = "auto"
        json_tail["tonemap"]["metadata_label"] = metadata_label
        use_dovi_value = json_tail["tonemap"]["use_dovi"]
        if isinstance(use_dovi_value, str):
            lowered = use_dovi_value.strip().lower()
            if lowered in {"auto", ""}:
                use_dovi_label = "auto"
            elif lowered in {"true", "1", "yes", "on"}:
                use_dovi_label = "on"
            elif lowered in {"false", "0", "no", "off"}:
                use_dovi_label = "off"
            else:
                use_dovi_label = lowered or "auto"
        elif use_dovi_value is None:
            use_dovi_label = "auto"
        else:
            use_dovi_label = "on" if use_dovi_value else "off"
        json_tail["tonemap"]["use_dovi_label"] = use_dovi_label
        emit_dovi_debug(
            {
                "phase": "final_tonemap_labels",
                "entrypoint": entrypoint_name,
                "json_use_dovi": json_tail["tonemap"]["use_dovi"],
                "json_use_dovi_label": json_tail["tonemap"]["use_dovi_label"],
                "json_metadata": json_tail["tonemap"]["metadata"],
                "json_metadata_label": json_tail["tonemap"]["metadata_label"],
            }
        )
        layout_data["tonemap"] = json_tail["tonemap"]
        json_tail["overlay"] = {
            "enabled": bool(cfg.color.overlay_enabled),
            "template": cfg.color.overlay_text_template,
            "mode": overlay_mode_value,
        }
        layout_data["overlay"] = json_tail["overlay"]

        diagnostics_cfg = getattr(cfg, "diagnostics", None)
        per_frame_config_enabled = bool(getattr(diagnostics_cfg, "per_frame_nits", False))
        per_frame_override = request.diagnostic_frame_metrics
        per_frame_requested = per_frame_override if per_frame_override is not None else per_frame_config_enabled
        per_frame_enabled = bool(per_frame_requested and overlay_mode_value == "diagnostic")

        tonemap_block = json_tail["tonemap"]
        tonemap_target_value = tonemap_block.get("target_nits")
        if isinstance(tonemap_target_value, (int, float)):
            tonemap_target = float(tonemap_target_value)
        else:
            tonemap_target = float(getattr(cfg.color, "target_nits", 0.0))
        per_frame_metrics: dict[int, dict[str, Any]] = {}
        if per_frame_enabled:
            for frame_idx, detail in selection_details.items():
                entry = build_frame_metric_entry(
                    frame_idx,
                    detail.score,
                    detail.label,
                    target_nits=tonemap_target,
                )
                if entry is not None:
                    per_frame_metrics[frame_idx] = entry

        for frame_idx, overlay_detail in selection_overlay_details.items():
            metric_entry = per_frame_metrics.get(frame_idx)
            if metric_entry is None:
                continue
            diagnostics_block = overlay_detail.setdefault("diagnostics", {})
            diagnostics_block["frame_metrics"] = metric_entry

        analyze_props = stored_props_seq[analyze_index] or {}
        dovi_meta = extract_dovi_metadata(analyze_props)
        hdr_meta = extract_hdr_metadata(analyze_props)
        range_label = classify_color_range(analyze_props)
        frame_metrics_json = {str(frame): entry for frame, entry in per_frame_metrics.items()}
        dv_summary = {k: v for k, v in dovi_meta.items() if v is not None}
        metadata_present = bool(dv_summary)
        has_l1_stats = any(key in dv_summary for key in ("l1_average", "l1_maximum"))
        dv_block: dict[str, Any] = {
            "label": json_tail["tonemap"].get("use_dovi_label"),
            "enabled": json_tail["tonemap"].get("use_dovi"),
            "metadata_present": metadata_present,
            "has_l1_stats": has_l1_stats,
        }
        if dv_summary:
            dv_block["l2_summary"] = dv_summary
        hdr_summary = {k: v for k, v in hdr_meta.items() if v is not None}
        overlay_diag: dict[str, Any] = {
            "dv": dv_block,
            "frame_metrics": {
                "enabled": per_frame_enabled,
                "per_frame": frame_metrics_json,
                "gating": {
                    "config": per_frame_config_enabled,
                    "cli_override": per_frame_override,
                    "overlay_mode": overlay_mode_value,
                },
            },
        }
        if hdr_summary:
            overlay_diag["hdr"] = hdr_summary
        if range_label:
            overlay_diag["dynamic_range"] = {"label": range_label, "source": "frame_props"}
        json_tail["overlay"]["diagnostics"] = overlay_diag

        reporter.update_values(layout_data)

        verification_records: List[Dict[str, Any]] = []

        try:
            seen_pivot_messages: set[str] = set()

            def _notify_pivot(message: str) -> None:
                if message in seen_pivot_messages:
                    return
                seen_pivot_messages.add(message)
                reporter.console.log(message, markup=False)

            if total_screens > 0:
                start_time = time.perf_counter()
                processed = 0
                clip_labels = [_plan_label(plan) for plan in plans]
                clip_total_frames = len(frames)
                clip_count = len(clip_labels)
                clip_progress_enabled = clip_count > 0 and clip_total_frames > 0
                if clip_progress_enabled:
                    reporter.update_progress_state(
                        "render_clip_bar",
                        label=clip_labels[0],
                        clip_index=1,
                        clip_total=clip_count,
                        current=0,
                        total=clip_total_frames,
                    )

                reporter.update_progress_state(
                    "render_bar",
                    fps=0.0,
                    eta_tc="--:--",
                    elapsed_tc="00:00",
                    current=0,
                    total=total_screens,
                )

                clip_progress = None
                clip_task_id: Optional[int] = None
                clip_index = 0
                clip_completed = 0

                def _clip_description(idx: int) -> str:
                    if not clip_labels:
                        return "Rendering clip"
                    bounded = max(0, min(idx, len(clip_labels) - 1))
                    return f"{clip_labels[bounded]} ({bounded + 1}/{clip_count})"

                with ExitStack() as progress_stack:
                    render_progress = progress_stack.enter_context(
                        reporter.create_progress("render_bar", transient=False)
                    )
                    task_id = render_progress.add_task(
                        "Rendering outputs",
                        total=total_screens,
                    )
                    if clip_progress_enabled:
                        clip_progress = progress_stack.enter_context(
                            reporter.create_progress("render_clip_bar", transient=False)
                        )
                        clip_task_id = clip_progress.add_task(
                            _clip_description(clip_index),
                            total=clip_total_frames,
                        )

                    def advance_render(count: int) -> None:
                        """Update rendering progress metrics and visible bars."""
                        nonlocal processed
                        nonlocal clip_index
                        nonlocal clip_completed
                        processed += count
                        elapsed = max(time.perf_counter() - start_time, 1e-6)
                        fps_val = processed / elapsed
                        remaining = max(total_screens - processed, 0)
                        eta_seconds = (remaining / fps_val) if fps_val > 0 else None
                        reporter.update_progress_state(
                            "render_bar",
                            fps=fps_val,
                            eta_tc=runtime_utils.format_clock(eta_seconds),
                            elapsed_tc=runtime_utils.format_clock(elapsed),
                            current=min(processed, total_screens),
                            total=total_screens,
                        )
                        render_progress.update(task_id, completed=min(total_screens, processed))
                        if clip_progress_enabled and clip_progress is not None and clip_task_id is not None:
                            clip_completed += count
                            clip_completed = min(clip_completed, clip_total_frames)
                            clip_progress.update(
                                clip_task_id,
                                completed=clip_completed,
                                description=_clip_description(clip_index),
                            )
                            reporter.update_progress_state(
                                "render_clip_bar",
                                current=clip_completed,
                                total=clip_total_frames,
                                clip_index=clip_index + 1,
                                clip_total=clip_count,
                                label=clip_labels[clip_index],
                            )
                            if clip_completed >= clip_total_frames and clip_index + 1 < clip_count:
                                clip_index += 1
                                clip_completed = 0
                                cast(Any, clip_progress).reset(
                                    clip_task_id,
                                    total=clip_total_frames,
                                )
                                clip_progress.update(
                                    clip_task_id,
                                    completed=clip_completed,
                                    description=_clip_description(clip_index),
                                )
                                reporter.update_progress_state(
                                    "render_clip_bar",
                                    current=clip_completed,
                                    total=clip_total_frames,
                                    clip_index=clip_index + 1,
                                    clip_total=clip_count,
                                    label=clip_labels[clip_index],
                                )

                    image_paths = generate_screenshots(
                        clips,
                        frames,
                        [str(plan.path) for plan in plans],
                        [plan.metadata for plan in plans],
                        out_dir,
                        cfg.screenshots,
                        cfg.color,
                        trim_offsets=[plan.trim_start for plan in plans],
                        progress_callback=advance_render,
                        frame_labels=frame_categories,
                        selection_details=selection_overlay_details,
                        warnings_sink=collected_warnings,
                        verification_sink=verification_records,
                        pivot_notifier=_notify_pivot,
                        debug_color=debug_color_enabled,
                        source_frame_props=stored_props_seq,
                    )

                    if processed < total_screens:
                        elapsed = max(time.perf_counter() - start_time, 1e-6)
                        fps_val = processed / elapsed
                        reporter.update_progress_state(
                            "render_bar",
                            fps=fps_val,
                            eta_tc=runtime_utils.format_clock(0.0),
                            elapsed_tc=runtime_utils.format_clock(elapsed),
                            current=total_screens,
                            total=total_screens,
                        )
                        render_progress.update(task_id, completed=total_screens)
                    if clip_progress_enabled and clip_progress is not None and clip_task_id is not None:
                        clip_progress.update(
                            clip_task_id,
                            completed=clip_total_frames,
                            description=_clip_description(min(clip_index, clip_count - 1)),
                        )
                        reporter.update_progress_state(
                            "render_clip_bar",
                            current=clip_total_frames,
                            total=clip_total_frames,
                            clip_index=min(clip_index, clip_count - 1) + 1,
                            clip_total=clip_count,
                            label=clip_labels[min(clip_index, clip_count - 1)],
                        )
            else:
                image_paths = generate_screenshots(
                    clips,
                    frames,
                    [str(plan.path) for plan in plans],
                    [plan.metadata for plan in plans],
                    out_dir,
                    cfg.screenshots,
                    cfg.color,
                    trim_offsets=[plan.trim_start for plan in plans],
                    frame_labels=frame_categories,
                    selection_details=selection_overlay_details,
                    warnings_sink=collected_warnings,
                    verification_sink=verification_records,
                    pivot_notifier=_notify_pivot,
                    debug_color=debug_color_enabled,
                    source_frame_props=stored_props_seq,
                )
        except ClipProcessError as exc:
            hint = "Run 'frame-compare doctor' for dependency diagnostics."
            raise CLIAppError(
                f"Screenshot generation failed: {exc}\nHint: {hint}",
                rich_message=f"[red]Screenshot generation failed:[/red] {exc}\n[yellow]Hint:[/yellow] {hint}",
            ) from exc
        except ScreenshotError as exc:
            raise CLIAppError(
                f"Screenshot generation failed: {exc}",
                rich_message=f"[red]Screenshot generation failed:[/red] {exc}",
            ) from exc

        verify_threshold = float(cfg.color.verify_luma_threshold)
        if verification_records:
            max_entry = max(verification_records, key=lambda item: item["maximum"])
            delta_summary: dict[str, object] = {
                "max": float(max_entry["maximum"]),
                "average": float(max_entry["average"]),
                "frame": int(max_entry["frame"]),
                "file": str(max_entry["file"]),
                "auto_selected": bool(max_entry["auto_selected"]),
            }
            verify_summary = cast(
                dict[str, object],
                {
                    "count": len(verification_records),
                    "threshold": verify_threshold,
                    "delta": delta_summary,
                    "entries": [dict(entry) for entry in verification_records],
                },
            )
        else:
            verify_summary = cast(
                dict[str, object],
                {
                    "count": 0,
                    "threshold": verify_threshold,
                    "delta": {
                        "max": None,
                        "average": None,
                        "frame": None,
                        "file": None,
                        "auto_selected": None,
                    },
                    "entries": [],
                },
            )

        json_tail["verify"] = verify_summary
        layout_data["verify"] = verify_summary

        slowpics_url, report_index_path = self._publish_results(
            context=context,
            reporter=reporter,
            cfg=cfg,
            layout_data=layout_data,
            json_tail=json_tail,
            image_paths=image_paths,
            out_dir=out_dir,
            collected_warnings=collected_warnings,
            report_enabled=report_enabled,
            root=root,
            plans=plans,
            frames=frames,
            selection_details=selection_details,
            report_publisher=report_publisher,
            slowpics_publisher=slowpics_publisher,
        )

        report_block = json_tail["report"]
        viewer_block = json_tail.get("viewer", {})
        viewer_mode = "slow_pics" if slowpics_url else "local_report" if report_block.get("enabled") and report_block.get("path") else "none"
        viewer_destination: Optional[str]
        viewer_label: str
        if viewer_mode == "slow_pics":
            viewer_destination = slowpics_url
        elif viewer_mode == "local_report":
            raw_path = report_block.get("path")
            viewer_destination = str(raw_path) if raw_path is not None else None
        else:
            viewer_destination = None
        viewer_label = viewer_destination or ""
        if viewer_mode == "local_report" and viewer_destination:
            try:
                viewer_label = str(Path(viewer_destination).resolve().relative_to(root.resolve()))
            except ValueError:
                viewer_label = viewer_destination
        viewer_mode_display = {
            "slow_pics": "slow.pics",
            "local_report": "Local report",
            "none": "None",
        }.get(viewer_mode, viewer_mode.title())
        viewer_block.update(
            {
                "mode": viewer_mode,
                "mode_display": viewer_mode_display,
                "destination": viewer_destination,
                "destination_label": viewer_label,
            }
        )
        json_tail["viewer"] = viewer_block
        layout_data["viewer"] = viewer_block
        reporter.update_values(layout_data)

        result = RunResult(
            files=[plan.path for plan in plans],
            frames=list(frames),
            out_dir=out_dir,
            out_dir_created=created_out_dir,
            out_dir_created_path=created_out_dir_path,
            root=root,
            config=cfg,
            image_paths=list(image_paths),
            slowpics_url=slowpics_url,
            json_tail=json_tail,
            report_path=report_index_path,
        )

        for warning in collected_warnings:
            reporter.warn(warning)

        warnings_list = list(dict.fromkeys(reporter.iter_warnings()))
        json_tail["warnings"] = warnings_list
        warnings_section: dict[str, object] | None = None
        for section_map in layout_sections:
            if section_map.get("id") == "warnings":
                warnings_section = dict(section_map)
                break
        fold_config_source = warnings_section.get("fold_labels") if warnings_section is not None else None
        if isinstance(fold_config_source, Mapping):
            fold_config = coerce_str_mapping(cast(Mapping[str, object], fold_config_source))
        else:
            fold_config = {}
        fold_head = fold_config.get("head")
        fold_tail = fold_config.get("tail")
        fold_when = fold_config.get("when")
        head = int(fold_head) if isinstance(fold_head, (int, float)) else 2
        tail = int(fold_tail) if isinstance(fold_tail, (int, float)) else 1
        joiner = str(fold_config.get("joiner", ", "))
        fold_when_text = str(fold_when) if isinstance(fold_when, str) and fold_when else None
        fold_enabled = runtime_utils.evaluate_rule_condition(fold_when_text, flags=reporter.flags)

        warnings_data: List[Dict[str, object]] = []
        if warnings_list:
            labels_text = runtime_utils.fold_sequence(warnings_list, head=head, tail=tail, joiner=joiner, enabled=fold_enabled)
            warnings_data.append(
                {
                    "warning.type": "general",
                    "warning.count": len(warnings_list),
                    "warning.labels": labels_text,
                }
            )
        else:
            warnings_data.append(
                {
                    "warning.type": "general",
                    "warning.count": 0,
                    "warning.labels": "none",
                }
            )

        layout_data["warnings"] = warnings_data
        reporter.update_values(layout_data)

        section_states: Dict[str, SectionState] = {}
        for section in layout_sections:
            section_id_raw = section.get("id")
            if not section_id_raw:
                continue
            section_id = str(section_id_raw).strip()
            if not section_id:
                continue
            section_states[section_id] = SectionState(
                availability=SectionAvailability.MISSING,
                note=None,
            )

        def _mark_section(section_id: str, availability: SectionAvailability, note: str | None = None) -> None:
            if section_id not in section_states:
                return
            section_states[section_id] = SectionState(availability=availability, note=note)

        # Default every known section to FULL once layout data is populated so cache renders
        # receive the same footprint as the live run. Sections with optional artifacts override
        # this baseline below.
        for section_id in list(section_states):
            _mark_section(section_id, SectionAvailability.FULL)

        reporting.apply_section_availability_overrides(
            section_states,
            _mark_section,
            layout_data=layout_data,
            result=result,
        )
        snapshot = build_snapshot(
            values=reporter.values,
            flags=reporter.flags,
            layout_sections=layout_sections,
            section_states=section_states,
            files=result.files,
            frames=result.frames,
            image_paths=result.image_paths,
            slowpics_url=result.slowpics_url,
            report_path=result.report_path,
            warnings=warnings_list,
            json_tail=result.json_tail,
            source=ResultSource.LIVE,
            cli_version=resolve_cli_version(),
        )
        result.snapshot = snapshot
        result.snapshot_path = result_snapshot_path
        try:
            write_snapshot(result_snapshot_path, snapshot)
        except OSError:
            logger.warning("Failed to persist run snapshot to %s", result_snapshot_path, exc_info=True)
        render_run_result(
            snapshot=snapshot,
            reporter=reporter,
            layout_sections=layout_sections,
            options=render_options,
        )

        return result


    def _publish_results(
        self,
        *,
        context: RunContext,
        reporter: CliOutputManagerProtocol,
        cfg: AppConfig,
        layout_data: MutableMapping[str, Any],
        json_tail: JsonTail,
        image_paths: List[str],
        out_dir: Path,
        collected_warnings: List[str],
        report_enabled: bool,
        root: Path,
        plans: List[ClipPlan],
        frames: List[int],
        selection_details: Mapping[int, SelectionDetail],
        report_publisher: ReportPublisher,
        slowpics_publisher: SlowpicsPublisher,
    ) -> tuple[Optional[str], Optional[Path]]:
        """Publish run artifacts via service-mode publishers."""

        slowpics_request = SlowpicsPublisherRequest(
            reporter=reporter,
            json_tail=json_tail,
            layout_data=layout_data,
            title_inputs=context.slowpics_title_inputs,
            final_title=context.slowpics_final_title,
            resolved_base=context.slowpics_resolved_base,
            tmdb_disclosure_line=context.slowpics_tmdb_disclosure_line,
            verbose_tmdb_tag=context.slowpics_verbose_tmdb_tag,
            image_paths=list(image_paths),
            out_dir=out_dir,
            config=cfg.slowpics,
        )
        slowpics_result = slowpics_publisher.publish(slowpics_request)
        slowpics_url = slowpics_result.url
        report_request = ReportPublisherRequest(
            reporter=reporter,
            json_tail=json_tail,
            layout_data=layout_data,
            report_enabled=report_enabled,
            root=root,
            plans=plans,
            frames=list(frames),
            selection_details=selection_details,
            image_paths=list(image_paths),
            metadata_title=context.metadata_title,
            slowpics_url=slowpics_url,
            config=cfg.report,
            collected_warnings=collected_warnings,
        )
        report_result = report_publisher.publish(report_request)
        return slowpics_url, report_result.report_path
