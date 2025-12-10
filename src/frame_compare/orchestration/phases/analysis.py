from __future__ import annotations

import time
import traceback
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Dict, Optional, cast

from rich.markup import escape

import src.frame_compare.cache as cache_utils
import src.frame_compare.preflight as preflight_utils
import src.frame_compare.runtime_utils as runtime_utils
from src.frame_compare import selection as selection_utils
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
from src.frame_compare.cli_runtime import CLIAppError
from src.frame_compare.orchestration.phases.base import Phase
from src.frame_compare.orchestration.setup import emit_dovi_debug
from src.frame_compare.orchestration.state import CoordinatorContext


class AnalysisPhase(Phase):
    def execute(self, context: CoordinatorContext) -> None:
        root = context.env.root
        plans = context.plans
        cfg = context.env.cfg
        reporter = context.env.reporter
        json_tail = context.json_tail
        layout_data = context.layout_data
        analyze_path = context.analyze_path
        request = context.request

        if analyze_path is None:
             raise RuntimeError("analyze_path not set")

        analyze_index = [plan.path for plan in plans].index(analyze_path)
        analyze_clip = plans[analyze_index].clip
        if analyze_clip is None:
            raise CLIAppError("Missing clip for analysis")

        # Cache Logic
        cache_info = cache_utils.build_cache_info(root, plans, cfg, analyze_index)
        context.cache_info = cache_info

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

        # Frame Selection Logic
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

        # Prepare for selection
        _, frame_window, _ = selection_utils.resolve_selection_windows(plans, cfg.analysis)
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
        frames: list[int] = []
        frame_categories: dict[int, str] = {}

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
            raise CLIAppError(
                f"Frame selection failed: {exc}",
                rich_message=f"[red]Frame selection failed:[/red] {exc}",
            ) from exc

        if not frames:
            raise CLIAppError(
                "No frames were selected; cannot continue.",
                rich_message="[red]No frames were selected; cannot continue.[/red]",
            )

        context.frames = frames
        context.selection_details = selection_details

        # Export & Finalize
        selection_hash_value = selection_hash_for_config(cfg.analysis)
        clip_paths = [plan.path for plan in plans]
        selection_sidecar_dir = cache_info.path.parent if cache_info is not None else root
        selection_sidecar_path = selection_sidecar_dir / "generated.selection.v1.json"

        # NOTE: logic in coordinator uses `selection_overlay_details` which is created LATER in coordinator
        # but here we are just exporting. `export_selection_metadata` uses `selection_details` directly.
        # Wait, `write_selection_cache_file` uses `frame_categories`.

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

        # Preview Rule Logic
        preview_rule: dict[str, Any] = {}
        layout_obj = getattr(reporter, "layout", None)
        folding_rules_obj = getattr(layout_obj, "folding", None)
        if isinstance(folding_rules_obj, Mapping):
            folding_rules_map = cast(Mapping[str, object], folding_rules_obj)
            frames_preview_obj = folding_rules_map.get("frames_preview")
            if isinstance(frames_preview_obj, Mapping):
                preview_rule = dict(cast(Mapping[str, Any], frames_preview_obj))
        head_raw: Any = preview_rule.get("head")
        tail_raw: Any = preview_rule.get("tail")
        when_raw: Any = preview_rule.get("when")
        head = int(head_raw) if isinstance(head_raw, (int, float)) else 4
        tail = int(tail_raw) if isinstance(tail_raw, (int, float)) else 4
        joiner = str(preview_rule.get("joiner", ", "))
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

        emit_dovi_debug(
            {
                "phase": "analysis_cache",
                "entrypoint": "runner", # TODO: get from request.impl_module? or context
                "cache_status": cache_status,
                "cache_reason": cache_reason,
                "cache_ready": bool(cache_ready),
                "cache_probe_status": getattr(cache_probe, "status", None) if cache_probe else None,
                "cache_probe_reason": getattr(cache_probe, "reason", None) if cache_probe else None,
                "cache_file": cache_info.path if cache_info is not None else None,
                "analysis_cache_path": context.env.analysis_cache_path,
                "json_cache_reused": json_tail["analysis"]["cache_reused"],
            }
        )
