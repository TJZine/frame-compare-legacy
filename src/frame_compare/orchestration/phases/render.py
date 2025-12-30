from __future__ import annotations

import time
from contextlib import ExitStack
from enum import Enum
from typing import Any, Dict, List, Optional, Set, cast

from src.frame_compare import runtime_utils
from src.frame_compare import vs as vs_core
from src.frame_compare.cli_runtime import CLIAppError
from src.frame_compare.diagnostics import (
    build_frame_metric_entry,
    classify_color_range,
    extract_dovi_metadata,
    extract_hdr_metadata,
)
from src.frame_compare.layout_utils import plan_label as _plan_label
from src.frame_compare.orchestration.phases.base import Phase
from src.frame_compare.orchestration.setup import emit_dovi_debug
from src.frame_compare.orchestration.state import CoordinatorContext
from src.frame_compare.render.errors import ScreenshotError
from src.frame_compare.screenshot.orchestrator import generate_screenshots
from src.frame_compare.vs import ClipProcessError


class RenderPhase(Phase):
    def execute(self, context: CoordinatorContext) -> None:
        cfg = context.env.cfg
        json_tail = context.json_tail
        layout_data = context.layout_data
        plans = context.plans
        reporter = context.env.reporter
        frames = context.frames
        selection_details = context.selection_details
        stored_props_seq = context.stored_props_seq
        analyze_path = context.analyze_path
        out_dir = context.env.out_dir

        debug_color_enabled = bool(getattr(cfg.color, "debug_color", False))
        entrypoint_name = "runner" # TODO: get from request

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

        # Tonemap Logic
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

        layout_data["tonemap"] = json_tail["tonemap"]
        json_tail["overlay"] = {
            "enabled": bool(cfg.color.overlay_enabled),
            "template": cfg.color.overlay_text_template,
            "mode": overlay_mode_value,
        }
        layout_data["overlay"] = json_tail["overlay"]

        # Diagnostics & Frame Metrics
        diagnostics_cfg = getattr(cfg, "diagnostics", None)
        per_frame_config_enabled = bool(getattr(diagnostics_cfg, "per_frame_nits", False))
        per_frame_override = context.request.diagnostic_frame_metrics
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

        # Build selection_overlay_details (used for rendering)
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

        for frame_idx, overlay_detail in selection_overlay_details.items():
            metric_entry = per_frame_metrics.get(frame_idx)
            if metric_entry is None:
                continue
            diagnostics_block = overlay_detail.setdefault("diagnostics", {})
            diagnostics_block["frame_metrics"] = metric_entry

        if analyze_path is None:
            raise CLIAppError("analyze_path not set")
        analyze_index = [plan.path for plan in plans].index(analyze_path)
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

        # Rendering
        verification_records: List[Dict[str, Any]] = []
        collected_warnings = context.env.collected_warnings
        clips = [plan.clip for plan in plans]
        frame_categories = {frame: detail.label or "Auto" for frame, detail in selection_details.items()}

        try:
            seen_pivot_messages: Set[str] = set()

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

        context.image_paths = image_paths
        context.verification_records = verification_records

        # Verify Summary
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
