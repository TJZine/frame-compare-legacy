from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, List, Union, cast

from src.frame_compare import runtime_utils
from src.frame_compare import selection as selection_utils
from src.frame_compare.alignment_helpers import derive_frame_hint
from src.frame_compare.cli_runtime import (
    CLIAppError,
    ClipPlan,
    ClipRecord,
    TrimClipEntry,
    TrimSummary,
    coerce_str_mapping,
)
from src.frame_compare.diagnostics import classify_color_range, extract_hdr_metadata
from src.frame_compare.layout_utils import plan_label as _plan_label
from src.frame_compare.orchestration.phases.base import Phase
from src.frame_compare.orchestration.state import CoordinatorContext
from src.frame_compare.vs import ClipInitError


class ClipLoaderPhase(Phase):
    def execute(self, context: CoordinatorContext) -> None:
        plans = context.plans
        cfg = context.env.cfg
        root = context.env.root
        reporter = context.env.reporter
        json_tail = context.json_tail
        layout_data = context.layout_data

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
        context.stored_props_seq = stored_props_seq

        clip_records: List[ClipRecord] = []
        trim_details: List[TrimSummary] = []

        # Build Clip Records
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
            json_tail["trims"].setdefault("per_clip", {})[label] = clip_trim

        context.clip_records = clip_records
        context.trim_details = trim_details

        # Layout: Alignment (using analyze_clip)
        analyze_path = context.analyze_path
        if analyze_path is None:
             raise RuntimeError("analyze_path not set")

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

        # Layout: Window
        json_tail["window"] = {
            "ignore_lead_seconds": float(cfg.analysis.ignore_lead_seconds),
            "ignore_trail_seconds": float(cfg.analysis.ignore_trail_seconds),
            "min_window_seconds": float(cfg.analysis.min_window_seconds),
        }
        layout_data["window"] = json_tail["window"]

        # Warnings about windows
        for plan, spec in zip(plans, selection_specs, strict=False):
            if not spec.warnings:
                continue
            label = plan.metadata.get("label") or plan.path.name
            for warning in spec.warnings:
                message = f"Window warning for {label}: {warning}"
                context.env.collected_warnings.append(message)
        if windows_collapsed:
            message = "Ignore lead/trail settings did not overlap across all sources; using fallback range."
            context.env.collected_warnings.append(message)

        # Layout: Clips
        layout_data["clips"]["count"] = len(clip_records)
        layout_data["clips"]["items"] = clip_records
        layout_data["clips"]["ref"] = clip_records[0] if clip_records else {}
        layout_data["clips"]["tgt"] = clip_records[1] if len(clip_records) > 1 else {}

        # Layout: VSPreview
        self._build_vspreview_block(context, clip_records)

        # Layout: Trims
        self._build_trims_block(context, clip_records)

        # Layout: Audio Alignment
        self._build_audio_alignment_block(context)

        reporter.update_values(layout_data)
        if context.tmdb_notes:
            for note in context.tmdb_notes:
                reporter.verbose_line(note)
        if context.slowpics_tmdb_disclosure_line:
            reporter.verbose_line(context.slowpics_tmdb_disclosure_line)

    def _build_vspreview_block(self, context: CoordinatorContext, clip_records: List[ClipRecord]) -> None:
        json_tail = context.json_tail
        layout_data = context.layout_data
        alignment_summary = context.alignment_summary
        plans = context.plans
        reporter = context.env.reporter

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

        vspreview_mode_value = context.env.vspreview_mode_value
        vspreview_mode_display = (
            "baseline (0f applied to both clips)"
            if vspreview_mode_value == "baseline"
            else "seeded (suggested offsets applied before preview)"
        )

        vspreview_block = coerce_str_mapping(layout_data.get("vspreview"))
        clips_block = coerce_str_mapping(vspreview_block.get("clips"))
        clips_block["ref"] = {"label": reference_label}
        clips_block["tgt"] = {"label": target_label}
        vspreview_block["clips"] = clips_block
        vspreview_block["mode"] = vspreview_mode_value
        vspreview_block["mode_display"] = vspreview_mode_display
        vspreview_block["suggested_frames"] = vspreview_suggested_frames_value
        vspreview_block["suggested_seconds"] = vspreview_suggested_seconds_value

        # Merge existing if reporter has it (less relevant in new arch but keeping for safety)
        existing_vspreview_obj = reporter.values.get("vspreview")
        if isinstance(existing_vspreview_obj, Mapping):
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

    def _build_trims_block(self, context: CoordinatorContext, clip_records: List[ClipRecord]) -> None:
        json_tail = context.json_tail
        layout_data = context.layout_data
        trim_details = context.trim_details

        trims_per_clip = json_tail["trims"].get("per_clip", {})
        trim_lookup: dict[str, TrimSummary] = {detail["label"]: detail for detail in trim_details}

        def _trim_entry(label: str) -> TrimClipEntry:
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

    def _build_audio_alignment_block(self, context: CoordinatorContext) -> None:
        json_tail = context.json_tail
        layout_data = context.layout_data
        alignment_display = context.alignment_display
        alignment_summary = context.alignment_summary
        cfg = context.env.cfg

        if alignment_display is not None:
            json_tail["audio_alignment"]["preview_paths"] = alignment_display.preview_paths
            confirmation_value = alignment_display.confirmation
            if confirmation_value is None and alignment_summary is not None:
                confirmation_value = "auto"
            json_tail["audio_alignment"]["confirmed"] = bool(confirmation_value)

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
