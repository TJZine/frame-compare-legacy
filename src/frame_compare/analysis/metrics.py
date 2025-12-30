# pyright: standard
"""Metrics helpers for frame comparison analysis."""

from __future__ import annotations

import math
import numbers
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.datatypes import AnalysisConfig, ColorConfig
from src.frame_compare import vs as vs_core


def _quantile(sequence: Sequence[float], q: float) -> float:
    """Return the *q* quantile of *sequence* using linear interpolation."""

    if not sequence:
        raise ValueError("quantile requires a non-empty sequence")
    if math.isnan(q):
        raise ValueError("quantile fraction must be a real number")
    if q <= 0:
        return min(sequence)
    if q >= 1:
        return max(sequence)

    sorted_vals = sorted(sequence)
    position = q * (len(sorted_vals) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_vals[lower_index]
    fraction = position - lower_index
    return sorted_vals[lower_index] * (1 - fraction) + sorted_vals[upper_index] * fraction


def _frame_rate(clip: Any) -> float:
    """
    Return the best-effort floating-point frame rate for ``clip``.

    Parameters:
        clip: VapourSynth-like clip exposing ``fps_num``/``fps_den`` attributes.

    Returns:
        float: Floating-point frames-per-second value, or ``0.0`` if unavailable.
    """
    num = getattr(clip, "fps_num", None)
    den = getattr(clip, "fps_den", None)
    try:
        if isinstance(num, int) and isinstance(den, int) and den:
            return num / den
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):  # pragma: no cover - defensive
        pass
    return 24000 / 1001


def _ensure_even(value: int) -> int:
    """
    Return ``value`` unchanged when even; otherwise subtract one to make it even.

    Parameters:
        value (int): Integer to normalise.

    Returns:
        int: An even integer not greater than ``value``.
    """
    return value if value % 2 == 0 else value - 1


class _ProgressCoalescer:
    """Batch frequent progress callbacks to reduce Python overhead."""

    __slots__ = ("_cb", "_pending", "_last_flush", "_min_batch", "_min_interval")

    def __init__(
        self,
        callback: Callable[[int], None],
        *,
        min_batch: int = 8,
        min_ms: float = 100.0,
    ) -> None:
        self._cb = callback
        self._pending = 0
        self._last_flush = time.perf_counter()
        self._min_batch = max(1, int(min_batch))
        self._min_interval = max(0.0, float(min_ms)) / 1000.0

    def add(self, count: int = 1) -> None:
        self._pending += int(count)
        now = time.perf_counter()
        if self._pending >= self._min_batch or (now - self._last_flush) >= self._min_interval:
            self.flush(now)

    def flush(self, now: Optional[float] = None) -> None:
        if self._pending <= 0:
            return
        try:
            self._cb(self._pending)
        finally:
            self._pending = 0
            self._last_flush = time.perf_counter() if now is None else now


def _is_hdr_source(clip: Any) -> bool:
    """Return True when the clip's transfer characteristics indicate HDR."""

    try:
        props = vs_core.snapshot_frame_props(clip)
        _, transfer, _, _ = vs_core.resolve_color_metadata(props)
    except (AttributeError, ValueError, TypeError, RuntimeError):
        return False

    if transfer is None:
        return False

    try:
        code = int(transfer)
    except (TypeError, ValueError):
        code = None

    if code in {16, 18}:
        return True

    name = str(transfer).strip().upper()
    return name in {"ST2084", "SMPTE2084", "PQ", "HLG", "ARIB-B67"}


def _collect_metrics_vapoursynth(
    clip: Any,
    cfg: AnalysisConfig,
    indices: Sequence[int],
    progress: Callable[[int], None] | None = None,
    *,
    color_cfg: ColorConfig | None = None,
    file_name: str | None = None,
) -> tuple[List[tuple[int, float]], List[tuple[int, float]]]:
    """
    Measure per-frame brightness and motion metrics using VapourSynth.

    Parameters:
        clip: VapourSynth clip to analyse.
        cfg (AnalysisConfig): Analysis settings controlling scaling, colours, and motion smoothing.
        indices (Sequence[int]): Frame indices to sample.
        progress (Callable[[int], None] | None): Optional callback invoked with the count of processed frames.

    Returns:
        tuple[List[tuple[int, float]], List[tuple[int, float]]]: Brightness and motion metric pairs for each processed frame.

    Raises:
        RuntimeError: If VapourSynth processing fails after retries.
    """
    try:
        import vapoursynth as vs  # type: ignore
    except Exception as exc:  # pragma: no cover - handled by fallback
        raise RuntimeError("VapourSynth is unavailable") from exc

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("Expected a VapourSynth clip")

    props = vs_core.snapshot_frame_props(clip)
    clip, props, color_tuple = vs_core.normalise_color_metadata(
        clip,
        props,
        color_cfg=color_cfg,
        file_name=file_name,
    )
    matrix_in, transfer_in, primaries_in, color_range_in = color_tuple

    def _resize_kwargs_for_source() -> Dict[str, int]:
        """Return color-metadata kwargs describing the source clip."""
        kwargs: Dict[str, int] = {}
        if matrix_in is not None:
            kwargs["matrix_in"] = int(matrix_in)
        else:
            try:
                if clip.format is not None and clip.format.color_family == vs.RGB:
                    kwargs["matrix_in"] = getattr(vs, "MATRIX_RGB", 0)
            except AttributeError:
                pass
        if transfer_in is not None:
            kwargs["transfer_in"] = int(transfer_in)
        if primaries_in is not None:
            kwargs["primaries_in"] = int(primaries_in)
        if color_range_in is not None:
            kwargs["range_in"] = int(color_range_in)
        return kwargs

    processed_indices = [
        int(idx)
        for idx in indices
        if isinstance(idx, numbers.Integral) and 0 <= int(idx) < clip.num_frames
    ]

    if not processed_indices:
        return [], []

    def _detect_uniform_step(values: Sequence[int]) -> Optional[int]:
        """Return a positive step size when frame indices form an arithmetic series."""
        if len(values) <= 1:
            return 1
        step_value = values[1] - values[0]
        if step_value <= 0:
            return None
        for prev, curr in zip(values, values[1:], strict=False):
            if curr - prev != step_value:
                return None
        return step_value

    step_value = _detect_uniform_step(processed_indices)

    sequential = step_value is not None

    resize_kwargs = _resize_kwargs_for_source()

    try:
        if sequential:
            first_idx = processed_indices[0]
            last_idx = processed_indices[-1]
            trimmed = vs.core.std.Trim(clip, first=first_idx, last=last_idx)
            if len(processed_indices) > 1 and step_value and step_value > 1:
                sampled = vs.core.std.SelectEvery(trimmed, cycle=step_value, offsets=[0])
            else:
                sampled = trimmed
        else:
            sampled = clip
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to trim analysis clip: {exc}") from exc

    def _prepare_analysis_clip(node):
        """Resize and convert *node* to a grayscale analysis representation."""
        work = node
        try:
            height_obj = getattr(work, "height", None)
            if (
                cfg.downscale_height > 0
                and isinstance(height_obj, numbers.Real)
                and float(height_obj) > float(cfg.downscale_height)
            ):
                target_h = _ensure_even(max(2, int(cfg.downscale_height)))
                width_obj = getattr(work, "width", None)
                height_value = max(1, int(float(height_obj)))
                aspect = 1.0
                if isinstance(width_obj, numbers.Real) and height_value > 0:
                    aspect = float(width_obj) / float(height_value)
                target_w = _ensure_even(max(2, int(round(target_h * aspect))))
                work = vs.core.resize.Bilinear(
                    work,
                    width=target_w,
                    height=target_h,
                    **resize_kwargs,
                )

            target_format = getattr(vs, "GRAY8", None) or vs.GRAY16
            gray_kwargs: Dict[str, int] = dict(resize_kwargs)
            gray_formats = {
                getattr(vs, "GRAY8", None),
                getattr(vs, "GRAY16", None),
                getattr(vs, "GRAY32", None),
            }
            format_obj = getattr(work, "format", None)
            color_family = getattr(format_obj, "color_family", None)
            rgb_constant = getattr(vs, "RGB", None)
            if format_obj is not None and color_family == rgb_constant:
                matrix_in_val = gray_kwargs.get("matrix_in")
                if matrix_in_val is None:
                    matrix_in_val = getattr(vs, "MATRIX_RGB", 0)
                convert_kwargs: Dict[str, int] = dict(gray_kwargs)
                convert_kwargs.pop("matrix", None)
                convert_kwargs["matrix_in"] = int(matrix_in_val)
                if "matrix" not in convert_kwargs:
                    convert_kwargs["matrix"] = getattr(vs, "MATRIX_BT709", 1)
                yuv = vs.core.resize.Bilinear(
                    work,
                    format=vs.YUV444P16,
                    **convert_kwargs,
                )
                work = vs.core.std.ShufflePlanes(yuv, planes=0, colorfamily=vs.GRAY)
            if target_format not in gray_formats:
                if "matrix" not in gray_kwargs:
                    matrix_in_value = gray_kwargs.get("matrix_in")
                    if matrix_in_value is not None:
                        gray_kwargs["matrix"] = int(matrix_in_value)
                    else:
                        gray_kwargs["matrix"] = getattr(vs, "MATRIX_BT709", 1)
            else:
                gray_kwargs.pop("matrix", None)
            work = vs.core.resize.Bilinear(work, format=target_format, **gray_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to prepare analysis clip: {exc}") from exc
        return work

    prepared = _prepare_analysis_clip(sampled)

    try:
        stats_clip = prepared.std.PlaneStats()
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to prepare metrics pipeline: {exc}") from exc

    motion_stats = None
    if cfg.frame_count_motion > 0 and prepared.num_frames > 1:
        try:
            previous = prepared[:-1]
            current = prepared[1:]
            if cfg.motion_use_absdiff:
                diff_clip = vs.core.std.Expr([previous, current], "x y - abs")
            else:
                diff_clip = vs.core.std.MakeDiff(previous, current)
                diff_clip = vs.core.std.Prewitt(diff_clip)
            motion_stats = diff_clip.std.PlaneStats()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to build motion metrics: {exc}") from exc

    brightness: List[tuple[int, float]] = []
    motion: List[tuple[int, float]] = []

    coalescer = _ProgressCoalescer(progress) if progress is not None else None

    try:
        if sequential:
            for position, idx in enumerate(processed_indices):
                if position >= stats_clip.num_frames:
                    break
                frame = stats_clip.get_frame(position)
                luma = float(frame.props.get("PlaneStatsAverage", 0.0))
                brightness.append((idx, luma))
                del frame

                motion_value = 0.0
                if motion_stats is not None and position > 0:
                    diff_frame = motion_stats.get_frame(position - 1)
                    motion_value = float(diff_frame.props.get("PlaneStatsAverage", 0.0))
                    del diff_frame
                motion.append((idx, motion_value))

                if coalescer is not None:
                    coalescer.add(1)
        else:
            for idx in processed_indices:
                if idx >= stats_clip.num_frames:
                    break
                frame = stats_clip.get_frame(idx)
                luma = float(frame.props.get("PlaneStatsAverage", 0.0))
                brightness.append((idx, luma))
                del frame

                motion_value = 0.0
                if motion_stats is not None and idx > 0:
                    diff_index = min(idx - 1, motion_stats.num_frames - 1)
                    if diff_index >= 0:
                        diff_frame = motion_stats.get_frame(diff_index)
                        motion_value = float(diff_frame.props.get("PlaneStatsAverage", 0.0))
                        del diff_frame
                motion.append((idx, motion_value))

                if coalescer is not None:
                    coalescer.add(1)
    finally:
        if coalescer is not None:
            coalescer.flush()

    return brightness, motion


def _generate_metrics_fallback(
    indices: Sequence[int],
    cfg: AnalysisConfig,
    progress: Callable[[int], None] | None = None,
) -> tuple[List[tuple[int, float]], List[tuple[int, float]]]:
    """
    Synthesize deterministic metrics when VapourSynth processing is unavailable.

    Parameters:
        indices (Sequence[int]): Frame indices to simulate metrics for.
        cfg (AnalysisConfig): Analysis configuration controlling quantiles and smoothing.
        progress (Callable[[int], None] | None): Optional callback invoked with the count of processed frames.

    Returns:
        tuple[List[tuple[int, float]], List[tuple[int, float]]]: Synthetic brightness and motion metric samples.
    """
    brightness: List[tuple[int, float]] = []
    motion: List[tuple[int, float]] = []
    for idx in indices:
        brightness.append((idx, (math.sin(idx * 0.137) + 1.0) / 2.0))
        phase = 0.21 if cfg.motion_use_absdiff else 0.17
        motion.append((idx, (math.cos(idx * phase) + 1.0) / 2.0))
        if progress is not None:
            progress(1)
    return brightness, motion


def _smooth_motion(values: List[tuple[int, float]], radius: int) -> List[tuple[int, float]]:
    """
    Apply a simple moving average of ``radius`` to motion metric samples.

    Parameters:
        values (List[tuple[int, float]]): Motion metric samples as ``(frame, value)`` pairs.
        radius (int): Window radius used for smoothing.

    Returns:
        List[tuple[int, float]]: Smoothed motion metric samples.
    """
    if radius <= 0 or not values:
        return values
    smoothed: List[tuple[int, float]] = []
    prefix = [0.0]
    for _, val in values:
        prefix.append(prefix[-1] + val)
    for i, (idx, _) in enumerate(values):
        start = max(0, i - radius)
        end = min(len(values) - 1, i + radius)
        total = prefix[end + 1] - prefix[start]
        count = (end - start) + 1
        smoothed.append((idx, total / max(1, count)))
    return smoothed


quantile = _quantile
frame_rate = _frame_rate
ensure_even = _ensure_even
is_hdr_source = _is_hdr_source
collect_metrics_vapoursynth = _collect_metrics_vapoursynth
generate_metrics_fallback = _generate_metrics_fallback
smooth_motion = _smooth_motion
