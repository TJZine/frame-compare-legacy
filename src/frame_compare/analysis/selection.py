# pyright: standard
"""Selection orchestration helpers for frame comparison analysis."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, TypedDict, cast

from src.datatypes import AnalysisConfig, AnalysisThresholdMode, ColorConfig
from src.frame_compare import vs as vs_core

from . import cache_io, metrics
from .cache_io import (
    _SELECTION_SOURCE_ID,
    CacheLoadResult,
    FrameMetricsCacheInfo,
    coerce_frame_index,
    coerce_optional_float,
    coerce_optional_str,
    infer_clip_role,
    probe_cached_metrics,
    selection_sidecar_path,
)


class MetricsCollector(Protocol):
    def __call__(
        self,
        clip: object,
        cfg: AnalysisConfig,
        indices: Sequence[int],
        progress: Callable[[int], None] | None = ...,
        *,
        color_cfg: ColorConfig | None = ...,
        file_name: str | None = ...,
    ) -> tuple[List[tuple[int, float]], List[tuple[int, float]]]:
        ...


class MetricsFallback(Protocol):
    def __call__(
        self,
        indices: Sequence[int],
        cfg: AnalysisConfig,
        progress: Callable[[int], None] | None = ...,
    ) -> tuple[List[tuple[int, float]], List[tuple[int, float]]]:
        ...


logger = logging.getLogger("src.analysis")


def _resolve_collect_metrics_vapoursynth() -> MetricsCollector:
    try:
        from src.frame_compare import analysis as analysis_mod

        func = analysis_mod._collect_metrics_vapoursynth
    except (ImportError, AttributeError):
        try:
            from src import analysis as analysis_mod  # pragma: no cover - legacy fallback

            func = analysis_mod._collect_metrics_vapoursynth
        except (ImportError, AttributeError):
            func = metrics.collect_metrics_vapoursynth
    return func


def _resolve_generate_metrics_fallback() -> MetricsFallback:
    try:
        from src.frame_compare import analysis as analysis_mod

        func = analysis_mod._generate_metrics_fallback
    except (ImportError, AttributeError):
        try:
            from src import analysis as analysis_mod  # pragma: no cover - legacy fallback

            func = analysis_mod._generate_metrics_fallback
        except (ImportError, AttributeError):
            func = metrics.generate_metrics_fallback
    return func


@dataclass(frozen=True)
class SelectionWindowSpec:
    """Resolved selection window boundaries for a clip."""

    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    applied_lead_seconds: float
    applied_trail_seconds: float
    duration_seconds: float
    warnings: Tuple[str, ...] = ()


@dataclass
class SelectionDetail:
    """Captured metadata describing how and why a frame was selected."""

    frame_index: int
    label: str
    score: Optional[float]
    source: str
    timecode: Optional[str]
    clip_role: Optional[str] = None
    notes: Optional[str] = None


class _SerializedSelectionDetail(TypedDict, total=False):
    frame_index: int | float | str
    type: str
    score: float | int | str | None
    source: str | None
    ts_tc: str | int | float | None
    clip_role: str | None
    notes: str | int | float | None


def _frame_to_timecode(frame_idx: int, fps: float) -> Optional[str]:
    if fps <= 0:
        return None
    seconds = frame_idx / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds - hours * 3600 - minutes * 60
    seconds_whole = int(remainder)
    fractional = remainder - seconds_whole
    milliseconds = int(round(fractional * 1000))
    if milliseconds >= 1000:
        milliseconds -= 1000
        seconds_whole += 1
    if seconds_whole >= 60:
        seconds_whole -= 60
        minutes += 1
    # Handle rollover if minutes reached 60 (e.g., 59.9995s rounding)
    if minutes >= 60:
        hours += minutes // 60
        minutes %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds_whole:02d}.{milliseconds:03d}"


def _serialize_selection_details(details: Mapping[int, SelectionDetail]) -> List[_SerializedSelectionDetail]:
    records: List[_SerializedSelectionDetail] = []
    for frame_idx in sorted(details.keys()):
        detail = details[frame_idx]
        record: _SerializedSelectionDetail = {
            "frame_index": int(detail.frame_index),
            "type": detail.label,
            "score": None if detail.score is None else float(detail.score),
            "source": detail.source,
            "ts_tc": detail.timecode,
            "clip_role": detail.clip_role,
            "notes": detail.notes,
        }
        records.append(record)
    return records


def _deserialize_selection_details(value: object) -> Dict[int, SelectionDetail]:
    if not isinstance(value, list):
        return {}
    results: Dict[int, SelectionDetail] = {}
    for entry in value:
        if not isinstance(entry, dict):
            continue
        record = cast(_SerializedSelectionDetail, entry)
        frame_idx = coerce_frame_index(record.get("frame_index"))
        if frame_idx is None:
            continue
        label_obj: object = record.get("type")
        label = str(label_obj).strip() if isinstance(label_obj, str) and label_obj.strip() else "Auto"
        score = coerce_optional_float(record.get("score"))
        source_obj: object = record.get("source")
        source = (
            str(source_obj).strip()
            if isinstance(source_obj, str) and source_obj.strip()
            else _SELECTION_SOURCE_ID
        )
        timecode = coerce_optional_str(record.get("ts_tc"))
        clip_role = coerce_optional_str(record.get("clip_role"))
        notes = coerce_optional_str(record.get("notes"))
        results[frame_idx] = SelectionDetail(
            frame_index=frame_idx,
            label=label,
            score=score,
            source=source,
            timecode=timecode,
            clip_role=clip_role,
            notes=notes,
        )
    return results


def _format_selection_annotation(detail: SelectionDetail) -> str:
    parts = [f"sel={detail.label}"]
    if detail.score is not None:
        parts.append(f"score={detail.score:.4f}")
    if detail.source:
        parts.append(f"src={detail.source}")
    if detail.timecode:
        parts.append(f"tc={detail.timecode}")
    if detail.notes:
        parts.append(f"note={detail.notes}")
    return ";".join(parts)


def selection_details_to_json(details: Mapping[int, SelectionDetail]) -> Dict[str, Dict[str, object]]:
    """Return JSON-friendly mapping for selection details."""

    serialised: Dict[str, Dict[str, object]] = {}
    for frame, detail in details.items():
        serialised[str(frame)] = {
            "frame_index": int(detail.frame_index),
            "type": detail.label,
            "score": detail.score,
            "source": detail.source,
            "timecode": detail.timecode,
            "clip_role": detail.clip_role,
            "notes": detail.notes,
        }
    return serialised


serialize_selection_details = _serialize_selection_details
deserialize_selection_details = _deserialize_selection_details
format_selection_annotation = _format_selection_annotation


def _coerce_seconds(value: object, label: str) -> float:
    """
    Convert ``value`` into a floating-point seconds value with validation.

    Parameters:
        value (object): Raw value to convert; accepts numbers or numeric strings.
        label (str): Configuration label used when raising validation errors.

    Returns:
        float: The coerced seconds value.

    Raises:
        TypeError: If ``value`` cannot be interpreted as a number.
    """
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"{label} must be numeric (got {type(value).__name__})")


def compute_selection_window(
    num_frames: int,
    fps: float,
    ignore_lead_seconds: float,
    ignore_trail_seconds: float,
    min_window_seconds: float,
) -> SelectionWindowSpec:
    """Resolve a trimmed time/frame window respecting configured ignores."""

    if num_frames <= 0:
        return SelectionWindowSpec(
            start_frame=0,
            end_frame=0,
            start_seconds=0.0,
            end_seconds=0.0,
            applied_lead_seconds=0.0,
            applied_trail_seconds=0.0,
            duration_seconds=0.0,
            warnings=(),
        )

    fps_val = float(fps) if isinstance(fps, (int, float)) else 0.0
    if not math.isfinite(fps_val) or fps_val <= 0:
        fps_val = 24000 / 1001

    duration = num_frames / fps_val if fps_val > 0 else 0.0
    lead = max(0.0, _coerce_seconds(ignore_lead_seconds, "analysis.ignore_lead_seconds"))
    trail = max(0.0, _coerce_seconds(ignore_trail_seconds, "analysis.ignore_trail_seconds"))
    min_window = max(0.0, _coerce_seconds(min_window_seconds, "analysis.min_window_seconds"))

    start_sec = min(lead, max(0.0, duration))
    end_sec = max(start_sec, max(0.0, duration - trail))
    warnings: List[str] = []

    span = end_sec - start_sec
    if min_window > 0 and duration > 0 and span < min_window:
        if min_window >= duration:
            start_sec = 0.0
            end_sec = duration
        else:
            needed = min_window - span
            available_end = max(0.0, duration - end_sec)
            extend_end = min(needed, available_end)
            end_sec += extend_end
            needed -= extend_end
            if needed > 0:
                available_start = start_sec
                shift_start = min(needed, available_start)
                start_sec -= shift_start
                needed -= shift_start
            # final clamp inside clip bounds
            start_sec = max(0.0, start_sec)
            end_sec = min(duration, end_sec)
            if end_sec - start_sec < min_window:
                # anchor to trailing edge if needed
                if duration >= min_window:
                    start_sec = max(0.0, duration - min_window)
                    end_sec = duration
                else:
                    start_sec = 0.0
                    end_sec = duration
        warnings.append(
            "Selection window shorter than minimum; expanded within clip bounds."
        )

    start_sec = max(0.0, min(start_sec, duration))
    end_sec = max(start_sec, min(end_sec, duration))

    applied_lead = start_sec
    applied_trail = max(0.0, duration - end_sec)

    epsilon = 1e-9
    start_frame = int(math.ceil(start_sec * fps_val - epsilon))
    end_frame = int(math.ceil(end_sec * fps_val - epsilon))

    start_frame = max(0, min(start_frame, num_frames))
    end_frame = max(start_frame, min(end_frame, num_frames))
    if end_frame == start_frame and start_frame < num_frames:
        end_frame = min(num_frames, start_frame + 1)

    return SelectionWindowSpec(
        start_frame=start_frame,
        end_frame=end_frame,
        start_seconds=start_sec,
        end_seconds=end_sec,
        applied_lead_seconds=applied_lead,
        applied_trail_seconds=applied_trail,
        duration_seconds=duration,
        warnings=tuple(warnings),
    )


def _selection_fingerprint(cfg: AnalysisConfig) -> str:
    """
    Return a hash of configuration fields relevant to frame selection.

    Parameters:
        cfg (AnalysisConfig): Analysis configuration whose selection-related fields should be fingerprinted.

    Returns:
        str: Hex digest capturing the selection-relevant configuration values.
    """
    relevant = {
        "frame_count_dark": cfg.frame_count_dark,
        "frame_count_bright": cfg.frame_count_bright,
        "frame_count_motion": cfg.frame_count_motion,
        "random_frames": cfg.random_frames,
        "random_seed": cfg.random_seed,
        "user_frames": [int(frame) for frame in cfg.user_frames],
        "motion_use_absdiff": cfg.motion_use_absdiff,
        "motion_scenecut_quantile": cfg.motion_scenecut_quantile,
        "screen_separation_sec": cfg.screen_separation_sec,
        "motion_diff_radius": cfg.motion_diff_radius,
        "ignore_lead_seconds": cfg.ignore_lead_seconds,
        "ignore_trail_seconds": cfg.ignore_trail_seconds,
        "min_window_seconds": cfg.min_window_seconds,
        "thresholds": cache_io.threshold_snapshot(cfg.thresholds),
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def selection_hash_for_config(cfg: AnalysisConfig) -> str:
    """Public helper exposing the stable selection fingerprint."""

    return _selection_fingerprint(cfg)


def dedupe(frames: Sequence[int], min_separation_sec: float, fps: float) -> List[int]:
    """
    Remove frames closer than ``min_separation_sec`` seconds apart while preserving order.

    Parameters:
        frames (Sequence[int]): Candidate frame indices.
        min_separation_sec (float): Minimum allowed spacing between kept frames, expressed in seconds.
        fps (float): Clip frame rate used to convert seconds to frame distances.

    Returns:
        List[int]: Filtered frame indices respecting the minimum separation constraint.
    """

    min_gap = 0 if fps <= 0 else int(round(max(0.0, min_separation_sec) * fps))
    result: List[int] = []
    seen: set[int] = set()
    for frame in frames:
        candidate = int(frame)
        if candidate in seen:
            continue
        if min_gap > 0:
            too_close = any(abs(candidate - kept) < min_gap for kept in result)
            if too_close:
                continue
        result.append(candidate)
        seen.add(candidate)
    return result


def _clamp_frame(frame: int, total: int) -> int:
    """
    Clamp ``frame`` to the valid index range for a clip with ``total`` frames.

    Parameters:
        frame (int): Candidate frame index.
        total (int): Total number of frames available.

    Returns:
        int: Frame index restricted to ``[0, total - 1]`` (or ``0`` when ``total`` is non-positive).
    """
    if total <= 0:
        return 0
    return max(0, min(total - 1, int(frame)))


def select_frames(
    clip: object,
    cfg: AnalysisConfig,
    files: List[str],
    file_under_analysis: str,
    cache_info: Optional[FrameMetricsCacheInfo] = None,
    progress: Callable[[int], None] | None = None,
    *,
    frame_window: tuple[int, int] | None = None,
    return_metadata: bool = False,
    color_cfg: Optional[ColorConfig] = None,
    cache_probe: CacheLoadResult | None = None,
) -> List[int] | Tuple[List[int], Dict[int, str], Dict[int, SelectionDetail]]:
    """Select frame indices for comparison using quantiles and motion heuristics.

    Parameters:
        clip: VapourSynth clip or compatible object providing frame accessors.
        cfg (AnalysisConfig): Analysis options controlling quantiles, step size, and filtering.
        files (List[str]): Ordered clip filenames participating in the run.
        file_under_analysis (str): Name of the clip used for analysis heuristics.
        cache_info (FrameMetricsCacheInfo | None): Resolved cache metadata for reuse, if available.
        progress (Callable[[int], None] | None): Optional callback invoked as metric samples are gathered.
        frame_window (tuple[int, int] | None): Optional inclusive/exclusive frame bounds to restrict sampling.
        return_metadata (bool): When ``True`` return frame categories and selection details alongside frames.
        color_cfg (ColorConfig | None): Color configuration required when SDR analysis must tonemap HDR sources.
        cache_probe (CacheLoadResult | None): Optional preloaded cache probe to avoid re-reading disk state when
            the caller already validated cache reuse.

    Returns:
        Either a list of selected frames or a tuple containing frames, category map, and selection details when
        ``return_metadata`` is truthy.
    """

    num_frames = int(getattr(clip, "num_frames", 0))
    if num_frames <= 0:
        return []

    window_start = 0
    window_end = num_frames
    if frame_window is not None:
        try:
            candidate_start, candidate_end = frame_window
        except (ValueError, TypeError, IndexError):
            candidate_start, candidate_end = (0, num_frames)
        else:
            try:
                candidate_start = int(candidate_start)
            except (TypeError, ValueError):
                candidate_start = 0
            try:
                candidate_end = int(candidate_end)
            except (TypeError, ValueError):
                candidate_end = num_frames
        candidate_start = max(0, candidate_start)
        candidate_end = max(candidate_start, candidate_end)
        window_start = min(candidate_start, num_frames)
        window_end = min(candidate_end, num_frames)
        if window_end <= window_start:
            if num_frames > 0:
                logger.warning(
                    "Frame window collapsed for %s; falling back to full clip", file_under_analysis
                )
                window_start = 0
                window_end = num_frames

    if window_end <= window_start:
        return []

    window_span = window_end - window_start

    fps = metrics.frame_rate(clip)
    rng = random.Random(cfg.random_seed)
    min_sep_frames = (
        0
        if cfg.screen_separation_sec <= 0
        else int(round(cfg.screen_separation_sec * fps))
    )
    skip_head_cutoff = window_start
    skip_tail_limit = window_end

    analysis_clip = clip
    if cfg.analyze_in_sdr:
        if metrics.is_hdr_source(clip):
            if color_cfg is None:
                raise ValueError("color_cfg must be provided when analyze_in_sdr is enabled")
            result = vs_core.process_clip_for_screenshot(
                clip,
                file_under_analysis,
                color_cfg,
                enable_overlay=False,
                enable_verification=False,
                logger_override=logger,
            )
            analysis_clip = result.clip
        else:
            logger.info("[ANALYSIS] Source detected as SDR; skipping SDR tonemap path")

    step = max(1, int(cfg.step))
    indices = list(range(window_start, window_end, step))

    selection_hash = _selection_fingerprint(cfg)
    selection_details: Dict[int, SelectionDetail] = {}
    collect_metrics_fn = _resolve_collect_metrics_vapoursynth()
    generate_metrics_fallback_fn = _resolve_generate_metrics_fallback()

    needs_dark = cfg.frame_count_dark > 0
    needs_bright = cfg.frame_count_bright > 0
    needs_motion = cfg.frame_count_motion > 0
    needs_metrics = needs_dark or needs_bright or needs_motion
    thresholds_cfg = cfg.thresholds
    threshold_mode = thresholds_cfg.mode
    if isinstance(threshold_mode, str):
        try:
            threshold_mode = AnalysisThresholdMode(threshold_mode.lower())
        except ValueError:
            threshold_mode = AnalysisThresholdMode.QUANTILE
    use_quantiles = threshold_mode == AnalysisThresholdMode.QUANTILE

    def _ensure_detail(
        frame_idx: int,
        *,
        label: Optional[str] = None,
        score: Optional[float] = None,
        note: Optional[str] = None,
    ) -> SelectionDetail:
        existing = selection_details.get(frame_idx)
        if existing is None:
            detail = SelectionDetail(
                frame_index=frame_idx,
                label=label or "Auto",
                score=score,
                source=_SELECTION_SOURCE_ID,
                timecode=_frame_to_timecode(frame_idx, fps),
                clip_role=None,
                notes=note,
            )
            selection_details[frame_idx] = detail
            existing = detail
        else:
            if label and not existing.label:
                existing.label = label
            if score is not None and existing.score is None:
                existing.score = score
            if note and not existing.notes:
                existing.notes = note
            if existing.timecode is None:
                existing.timecode = _frame_to_timecode(frame_idx, fps)
        if existing.clip_role is None:
            existing.clip_role = selection_clip_role
        return selection_details[frame_idx]

    selection_clip_role = "analyze"
    analyze_index_guess = 0
    if files:
        try:
            analyze_index_guess = files.index(file_under_analysis)
        except ValueError:
            analyze_index_guess = 0
        selection_clip_role = infer_clip_role(
            analyze_index_guess, files[analyze_index_guess], file_under_analysis, len(files)
        )
    for detail in selection_details.values():
        if detail.clip_role is None:
            detail.clip_role = selection_clip_role

    brightness: List[tuple[int, float]] = []
    motion: List[tuple[int, float]] = []

    if cache_info is not None:
        sidecar_result = cache_io.load_selection_sidecar(cache_info, cfg, selection_hash)
        if sidecar_result is not None:
            sidecar_frames, sidecar_details = sidecar_result
            selection_details.update(sidecar_details)
            frames_sorted = sorted(
                dict.fromkeys(
                    int(frame)
                    for frame in sidecar_frames
                    if window_start <= int(frame) < window_end
                )
            )
            filtered_details = {
                frame: detail for frame, detail in selection_details.items() if frame in frames_sorted
            }
            selection_details.clear()
            selection_details.update(filtered_details)
            if return_metadata:
                label_map: Dict[int, str] = {}
                for frame in frames_sorted:
                    detail = selection_details.get(frame)
                    label = detail.label if detail else "Cached"
                    detail = _ensure_detail(frame, label=label)
                    label_map[frame] = detail.label
                return frames_sorted, label_map, selection_details
                return frames_sorted
        logger.info(
            "[ANALYSIS] selection sidecar unavailable for %s (%s); recomputing selections",
            file_under_analysis,
            selection_sidecar_path(cache_info),
        )

    cache_probe_local = cache_probe
    if cache_probe_local is None and cache_info is not None:
        cache_probe_local = probe_cached_metrics(cache_info, cfg)
    cached_metrics = (
        cache_probe_local.metrics
        if cache_probe_local is not None and cache_probe_local.status == "reused"
        else None
    )
    if cache_probe_local is not None and cache_probe_local.status != "reused":
        cache_identifier = cache_info.path if cache_info is not None else "unknown cache"
        logger.info(
            "[ANALYSIS] metrics cache miss for %s (%s): %s",
            file_under_analysis,
            cache_identifier,
            (cache_probe_local.reason or cache_probe_local.status),
        )

    cached_selection: Optional[List[int]] = None
    cached_categories: Optional[Dict[int, str]] = None
    cached_details: Optional[Dict[int, SelectionDetail]] = None

    if cached_metrics is not None:
        if needs_metrics:
            brightness = [
                (idx, val)
                for idx, val in cached_metrics.brightness
                if window_start <= idx < window_end
            ]
            motion = [
                (idx, val)
                for idx, val in cached_metrics.motion
                if window_start <= idx < window_end
            ]
            if progress is not None:
                progress(len(brightness))
            logger.info(
                "[ANALYSIS] using cached metrics (brightness=%d, motion=%d)",
                len(brightness),
                len(motion),
            )
        else:
            brightness = []
            motion = []
            logger.info(
                "[ANALYSIS] skipping cached metric reuse (dark/bright/motion disabled)"
            )
        if cached_metrics.selection_hash == selection_hash:
            if cached_metrics.selection_frames is not None:
                cached_selection = [
                    frame
                    for frame in cached_metrics.selection_frames
                    if window_start <= int(frame) < window_end
                ]
            else:
                cached_selection = None
            cached_categories = cached_metrics.selection_categories
            cached_details = cached_metrics.selection_details
    else:
        if needs_metrics:
            logger.info(
                "[ANALYSIS] collecting metrics (indices=%d, step=%d, analyze_in_sdr=%s)",
                len(indices),
                step,
                cfg.analyze_in_sdr,
            )
            start_metrics = time.perf_counter()
            try:
                brightness, motion = collect_metrics_fn(
                    analysis_clip,
                    cfg,
                    indices,
                    progress,
                    color_cfg=color_cfg,
                    file_name=file_under_analysis,
                )
                logger.info(
                    "[ANALYSIS] metrics collected via VapourSynth in %.2fs (brightness=%d, motion=%d)",
                    time.perf_counter() - start_metrics,
                    len(brightness),
                    len(motion),
                )
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "[ANALYSIS] VapourSynth metrics collection failed (%s); "
                    "falling back to synthetic metrics",
                    exc,
                )
                brightness, motion = generate_metrics_fallback_fn(indices, cfg, progress)
                logger.info(
                    "[ANALYSIS] synthetic metrics generated in %.2fs",
                    time.perf_counter() - start_metrics,
                )
        else:
            logger.info(
                "[ANALYSIS] skipping brightness/motion analysis (dark/bright/motion counts are zero)"
            )

    if cached_selection is not None:
        frames_sorted = sorted(dict.fromkeys(int(frame) for frame in cached_selection))
        if cached_details:
            selection_details.update({frame: detail for frame, detail in cached_details.items() if frame in frames_sorted})
        if return_metadata:
            categories = cached_categories or {}
            label_map: Dict[int, str] = {}
            for frame in frames_sorted:
                label = categories.get(frame, "Cached")
                detail = _ensure_detail(frame, label=label)
                label_map[frame] = detail.label
            return frames_sorted, label_map, selection_details
        return frames_sorted

    brightness_values = [val for _, val in brightness]

    selected: List[int] = []
    selected_set: set[int] = set()
    frame_categories: Dict[int, str] = {}

    def try_add(
        frame: int,
        enforce_gap: bool = True,
        gap_frames: Optional[int] = None,
        allow_edges: bool = False,
        category: Optional[str] = None,
        score: Optional[float] = None,
        note: Optional[str] = None,
    ) -> bool:
        frame_idx = _clamp_frame(frame, num_frames)
        if frame_idx in selected_set:
            return False
        if frame_idx < window_start or frame_idx >= window_end:
            return False
        if not allow_edges:
            if frame_idx < skip_head_cutoff:
                return False
            if frame_idx >= skip_tail_limit:
                return False
        effective_gap = min_sep_frames if gap_frames is None else max(0, int(gap_frames))
        if enforce_gap and effective_gap > 0:
            for existing in selected:
                if abs(existing - frame_idx) < effective_gap:
                    return False
        selected.append(frame_idx)
        selected_set.add(frame_idx)
        if category and frame_idx not in frame_categories:
            frame_categories[frame_idx] = category
        _ensure_detail(frame_idx, label=category, score=score, note=note)
        return True

    dropped_user_frames: List[int] = []
    for frame in cfg.user_frames:
        try:
            frame_int = int(frame)
        except (TypeError, ValueError):
            continue
        if window_start <= frame_int < window_end:
            try_add(
                frame_int,
                enforce_gap=False,
                allow_edges=True,
                category="User",
                note="user_frame",
            )
        else:
            dropped_user_frames.append(frame_int)

    if dropped_user_frames:
        preview = ", ".join(str(val) for val in dropped_user_frames[:5])
        if len(dropped_user_frames) > 5:
            preview += ", …"
        logger.warning(
            "Dropped %d pinned frame(s) outside trimmed window for %s: %s",
            len(dropped_user_frames),
            file_under_analysis,
            preview,
        )

    def pick_from_candidates(
        candidates: List[tuple[int, float]],
        count: int,
        mode: str,
        gap_seconds_override: Optional[float] = None,
    ) -> None:
        if count <= 0 or not candidates:
            return
        unique_indices: List[int] = []
        seen_local: set[int] = set()
        if mode == "motion":
            ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
            for idx, _ in ordered:
                if idx in seen_local:
                    continue
                seen_local.add(idx)
                unique_indices.append(idx)
        elif mode in {"dark", "bright"}:
            for idx, _ in candidates:
                if idx in seen_local:
                    continue
                seen_local.add(idx)
                unique_indices.append(idx)
            rng.shuffle(unique_indices)
        else:
            raise ValueError(f"Unknown candidate mode: {mode}")
        separation = cfg.screen_separation_sec
        if gap_seconds_override is not None:
            separation = gap_seconds_override
        filtered_indices = dedupe(unique_indices, separation, fps)
        gap_frames = (
            None
            if gap_seconds_override is None
            else int(round(max(0.0, gap_seconds_override) * fps))
        )
        score_lookup: Dict[int, float] = {}
        for idx, val in candidates:
            score_lookup.setdefault(int(idx), float(val))
        added = 0
        category_label = "Motion" if mode == "motion" else mode.capitalize()
        for frame_idx in filtered_indices:
            if try_add(
                frame_idx,
                enforce_gap=True,
                gap_frames=gap_frames,
                category=category_label,
                score=score_lookup.get(frame_idx),
                note=mode,
            ):
                added += 1
            if added >= count:
                break

    dark_candidates: List[tuple[int, float]] = []
    if cfg.frame_count_dark > 0 and brightness_values:
        if use_quantiles:
            threshold = metrics.quantile(brightness_values, float(thresholds_cfg.dark_quantile))
            dark_candidates = [(idx, val) for idx, val in brightness if val <= threshold]
        else:
            dark_min = float(thresholds_cfg.dark_luma_min)
            dark_max = float(thresholds_cfg.dark_luma_max)
            dark_candidates = [(idx, val) for idx, val in brightness if dark_min <= val <= dark_max]
    pick_from_candidates(dark_candidates, cfg.frame_count_dark, mode="dark")

    bright_candidates: List[tuple[int, float]] = []
    if cfg.frame_count_bright > 0 and brightness_values:
        if use_quantiles:
            threshold = metrics.quantile(brightness_values, float(thresholds_cfg.bright_quantile))
            bright_candidates = [(idx, val) for idx, val in brightness if val >= threshold]
        else:
            bright_min = float(thresholds_cfg.bright_luma_min)
            bright_max = float(thresholds_cfg.bright_luma_max)
            bright_candidates = [(idx, val) for idx, val in brightness if bright_min <= val <= bright_max]
    pick_from_candidates(bright_candidates, cfg.frame_count_bright, mode="bright")

    motion_candidates: List[tuple[int, float]] = []
    if cfg.frame_count_motion > 0 and motion:
        smoothed_motion = metrics.smooth_motion(motion, max(0, int(cfg.motion_diff_radius)))
        filtered = smoothed_motion
        if cfg.motion_scenecut_quantile > 0:
            threshold = metrics.quantile([val for _, val in smoothed_motion], cfg.motion_scenecut_quantile)
            filtered = [(idx, val) for idx, val in smoothed_motion if val <= threshold]
        motion_candidates = filtered
    motion_gap = cfg.screen_separation_sec / 4 if cfg.screen_separation_sec > 0 else 0
    pick_from_candidates(
        motion_candidates,
        cfg.frame_count_motion,
        mode="motion",
        gap_seconds_override=motion_gap,
    )

    random_count = max(0, int(cfg.random_frames))
    attempts = 0
    while random_count > 0 and attempts < random_count * 10 and window_span > 0:
        candidate = window_start + rng.randrange(window_span)
        if try_add(candidate, enforce_gap=True, category="Random", note="random"):
            random_count -= 1
        attempts += 1

    final_frames = sorted(selected)

    for frame in final_frames:
        _ensure_detail(frame, label=frame_categories.get(frame, "Auto"))

    if cache_info is not None:
        try:
            cache_io.save_cached_metrics(
                cache_info,
                cfg,
                brightness,
                motion,
                selection_hash=selection_hash,
                selection_frames=final_frames,
                selection_categories=frame_categories,
                selection_details=selection_details,
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            pass

    if return_metadata:
        label_map = {frame: frame_categories.get(frame, "Auto") for frame in final_frames}
        return final_frames, label_map, selection_details
    return final_frames
