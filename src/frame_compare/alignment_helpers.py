"""Shared helpers for audio-alignment presentation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.audio_alignment import AlignmentMeasurement
    from src.frame_compare.alignment import AudioAlignmentSummary


def _lookup_measurement(
    summary: "AudioAlignmentSummary",
    clip_key: str,
) -> "AlignmentMeasurement | None":
    for measurement in summary.measurements:
        file_obj = getattr(measurement, "file", None)
        if file_obj is None:
            continue
        if getattr(file_obj, "name", None) == clip_key:
            return measurement
    return None


def derive_frame_hint(
    summary: "AudioAlignmentSummary | None",
    clip_key: str,
) -> Optional[int]:
    """
    Return the best-effort frame suggestion for *clip_key*.

    Prefers stored suggestions and falls back to deriving one from the raw
    measurements using reference/target FPS. Returns ``None`` when neither
    source can provide a reliable value.
    """

    if summary is None or not clip_key:
        return None

    stored = summary.suggested_frames.get(clip_key)
    if stored is not None:
        try:
            return int(stored)
        except (TypeError, ValueError):
            return None

    measurement = _lookup_measurement(summary, clip_key)
    if measurement is None:
        return None

    seconds_value = getattr(measurement, "offset_seconds", None)
    if seconds_value is None:
        return None

    fps_value = getattr(measurement, "target_fps", None) or getattr(
        measurement,
        "reference_fps",
        None,
    )
    if not fps_value:
        return None

    try:
        fps_float = float(fps_value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(fps_float) or fps_float == 0.0:
        return None

    return int(round(float(seconds_value) * fps_float))


__all__ = ["derive_frame_hint"]
