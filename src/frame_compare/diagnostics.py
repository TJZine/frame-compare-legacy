"""Helper utilities for extracting and formatting overlay diagnostic metadata."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

logger = logging.getLogger(__name__)


def _normalize_key(key: object) -> str:
    if isinstance(key, bytes):
        try:
            key = key.decode("utf-8", "ignore")
        except Exception:
            key = str(key)
    return str(key or "").strip().lower()


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    try:
        return int(round(number))
    except (TypeError, ValueError):
        return None


def _coerce_luminance_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", "ignore")
        except Exception:
            return []
    if isinstance(value, str):
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
        return [float(match) for match in matches]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        iterable = cast(Sequence[Any], value)
        results: list[float] = []
        for item in iterable:
            results.extend(_coerce_luminance_values(item))
        return results
    return []


def extract_mastering_display_luminance(props: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """
    Extract mastering display luminance metadata (min/max) from frame props.
    """

    min_keys = (
        "_MasteringDisplayMinLumi",
        "_MasteringDisplayMinLuminance",
        "MasteringDisplayMinLumi",
        "MasteringDisplayMinLuminance",
        "MasteringDisplayMinimumLuminance",
    )
    max_keys = (
        "_MasteringDisplayMaxLumi",
        "_MasteringDisplayMaxLuminance",
        "MasteringDisplayMaxLumi",
        "MasteringDisplayMaxLuminance",
        "MasteringDisplayMaximumLuminance",
    )

    min_value: float | None = None
    max_value: float | None = None

    for key in min_keys:
        if key in props:
            values = _coerce_luminance_values(props.get(key))
            if values:
                min_value = values[0]
                break
    for key in max_keys:
        if key in props:
            values = _coerce_luminance_values(props.get(key))
            if values:
                max_value = values[0]
                break

    if min_value is None or max_value is None:
        combined_keys = ("_MasteringDisplayLuminance", "MasteringDisplayLuminance")
        for key in combined_keys:
            values = _coerce_luminance_values(props.get(key))
            if len(values) >= 2:
                if min_value is None:
                    min_value = min(values)
                if max_value is None:
                    max_value = max(values)
                break

    return min_value, max_value


def format_luminance_value(value: float) -> str:
    """Format luminance values with context-aware precision."""

    if value < 1.0:
        text = f"{value:.4f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    return f"{value:.1f}"


def format_mastering_display_line(props: Mapping[str, Any]) -> str:
    """Return a human readable mastering display summary line."""

    min_value, max_value = extract_mastering_display_luminance(props)
    if min_value is None or max_value is None:
        return "MDL: Insufficient data"
    return (
        f"MDL: min: {format_luminance_value(min_value)} cd/m², "
        f"max: {format_luminance_value(max_value)} cd/m²"
    )


def _has_dovi_hint(normalized: str) -> bool:
    return "dolby" in normalized or "dovi" in normalized or "rpu" in normalized or "l1" in normalized


def _matches_all(text: str, needles: tuple[str, ...]) -> bool:
    return all(needle in text for needle in needles)


def extract_dovi_metadata(props: Mapping[str, Any]) -> dict[str, float | int | bool | None]:
    """
    Extract Dolby Vision metadata from frame properties.

    Returns a dict with keys:
        - rpu_present (bool)
        - l1_average (float | None)
        - l1_maximum (float | None)
        - ... other fields ...
    """
    result: dict[str, float | int | bool | None] = {
        "rpu_present": None,
        "l1_average": None,
        "l1_maximum": None,
        "l2_target_nits": None,
        "l5_left": None,
        "l5_right": None,
        "l5_top": None,
        "l5_bottom": None,
        "l6_max_cll": None,
        "l6_max_fall": None,
        "block_index": None,
        "block_total": None,
        "target_nits": None,
    }

    # Debug: Log keys if RPU is present to help diagnose missing L1 stats
    has_rpu_blob = any(k in ("DolbyVisionRPU", "_DolbyVisionRPU", "DolbyVisionRPU_b", "_DolbyVisionRPU_b") for k in props)
    if has_rpu_blob:
        dovi_keys = [k for k in props.keys() if "dolby" in k.lower() or "dovi" in k.lower() or "l1" in k.lower()]
        logger.debug("DoVi props present; keys=%s", dovi_keys)

    for key, value in props.items():
        # Check for raw RPU blob first (exact match or common variants)
        if key in ("DolbyVisionRPU", "_DolbyVisionRPU", "DolbyVisionRPU_b", "_DolbyVisionRPU_b"):
            result["rpu_present"] = True
            continue

        normalized = _normalize_key(key)
        if not normalized or not _has_dovi_hint(normalized):
            continue
        if _matches_all(normalized, ("block", "index")):
            result.setdefault("block_index", None)
            if result["block_index"] is None:
                result["block_index"] = _coerce_int(value)
        elif _matches_all(normalized, ("block", "total")) or _matches_all(normalized, ("block", "count")):
            if result["block_total"] is None:
                result["block_total"] = _coerce_int(value)
        elif "target" in normalized and ("nit" in normalized or "pq" in normalized or "brightness" in normalized):
            if result["target_nits"] is None:
                result["target_nits"] = _coerce_float(value)
        elif "l1" in normalized and ("avg" in normalized or "average" in normalized or "mean" in normalized):
            if result["l1_average"] is None:
                result["l1_average"] = _coerce_float(value)
        elif "l1" in normalized and "max" in normalized:
            if result["l1_maximum"] is None:
                result["l1_maximum"] = _coerce_float(value)
        elif "l2" in normalized and "target" in normalized:
            if result["l2_target_nits"] is None:
                result["l2_target_nits"] = _coerce_float(value)
        elif "l5" in normalized:
            if "left" in normalized:
                val = _coerce_int(value)
                if val is not None and val >= 0:
                    result["l5_left"] = val
            elif "right" in normalized:
                val = _coerce_int(value)
                if val is not None and val >= 0:
                    result["l5_right"] = val
            elif "top" in normalized:
                val = _coerce_int(value)
                if val is not None and val >= 0:
                    result["l5_top"] = val
            elif "bottom" in normalized:
                val = _coerce_int(value)
                if val is not None and val >= 0:
                    result["l5_bottom"] = val
        elif "l6" in normalized:
            if "cll" in normalized:
                result["l6_max_cll"] = _coerce_float(value)
            elif "fall" in normalized:
                result["l6_max_fall"] = _coerce_float(value)
    return result


def extract_hdr_metadata(props: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract HDR mastering metadata (MDL/CLL/FALL) from frame props."""

    min_lum, max_lum = extract_mastering_display_luminance(props)
    cll = None
    fall = None
    for key, value in props.items():
        normalized = _normalize_key(key)
        if "cll" in normalized or _matches_all(normalized, ("contentlightlevel", "max")):
            if cll is None:
                cll = _coerce_float(value)
        if "fall" in normalized or _matches_all(normalized, ("contentlightlevel", "fall")):
            if fall is None:
                fall = _coerce_float(value)
    return {
        "min_luminance": min_lum,
        "max_luminance": max_lum,
        "max_cll": cll,
        "max_fall": fall,
    }


def classify_color_range(props: Mapping[str, Any]) -> str | None:
    """Classify the encoded colour range (limited/full) based on frame props."""

    for key, value in props.items():
        normalized = _normalize_key(key)
        if normalized not in {"_colorrange", "colorrange"}:
            continue
        if value is None:
            continue
        try:
            code = int(value)
        except (TypeError, ValueError):
            continue
        if code == 0:
            return "full"
        if code == 1:
            return "limited"
    return None


def _format_nits(value: float | None) -> str | None:
    if value is None:
        return None
    if abs(value - round(value)) < 1e-3:
        return f"{round(value):.0f}"
    return f"{value:.1f}"


def format_dovi_line(label: str | None, metadata: Mapping[str, Any]) -> str | None:
    """Return a DoVi summary line for diagnostic overlays."""

    parts: list[str] = []
    # Check for L1 metadata presence (excluding the rpu_present flag itself)
    l1_keys = {"block_index", "block_total", "target_nits", "l1_average", "l1_maximum"}
    l1_present = any(value is not None for key, value in metadata.items() if key in l1_keys)
    rpu_present = bool(metadata.get("rpu_present"))

    if label:
        parts.append(f"DoVi: {label}")

    l2_target = metadata.get("l2_target_nits")
    if l2_target:
        l2_label = _format_nits(l2_target if isinstance(l2_target, (int, float)) else None)
        if l2_label:
            parts.append(f"(Target: {l2_label}nits)")

    if l1_present:
        block_index = metadata.get("block_index")
        block_total = metadata.get("block_total")
        if isinstance(block_index, int) and block_index >= 0:
            if isinstance(block_total, int) and block_total > 0:
                parts.append(f"L2 {block_index}/{block_total}")
            else:
                parts.append(f"L2 block {block_index}")
        elif isinstance(block_total, int) and block_total > 0:
            parts.append(f"L2 blocks={block_total}")
        target = metadata.get("target_nits")
        target_label = _format_nits(target if isinstance(target, (int, float)) else None)
        if target_label:
            parts.append(f"target {target_label} nits")

    if not l1_present and label and not rpu_present:
        parts.append("(no DV metadata)")

    if not parts:
        return None
    return " ".join(parts)


def format_dovi_l1_line(metadata: Mapping[str, Any]) -> str | None:
    """Render DV RPU Level 1 per-frame brightness stats when available."""

    average = metadata.get("l1_average")
    maximum = metadata.get("l1_maximum")
    avg_label = _format_nits(float(average)) if isinstance(average, (int, float)) else None
    max_label = _format_nits(float(maximum)) if isinstance(maximum, (int, float)) else None
    if not avg_label and not max_label:
        return None
    parts: list[str] = []
    if max_label:
        parts.append(f"{max_label}nits")
    if avg_label:
        parts.append(f"{avg_label}nits")
    descriptor = "MAX/AVG" if max_label and avg_label else "MAX" if max_label else "AVG"
    return f"DV RPU Level 1 {descriptor}: {' / '.join(parts)}"


def format_dovi_l5_line(metadata: Mapping[str, Any]) -> str | None:
    """Render DV L5 active area offsets if present."""
    left = metadata.get("l5_left")
    right = metadata.get("l5_right")
    top = metadata.get("l5_top")
    bottom = metadata.get("l5_bottom")

    # Only show if any value is present and non-zero
    if all(v == 0 for v in (left, right, top, bottom) if isinstance(v, int)):
        return None

    parts: list[str] = []
    if left is not None:
        parts.append(f"L:{left}")
    if right is not None:
        parts.append(f"R:{right}")
    if top is not None:
        parts.append(f"T:{top}")
    if bottom is not None:
        parts.append(f"B:{bottom}")

    if not parts:
        return None
    return f"DV L5 Active Area: {' '.join(parts)}"


def format_dovi_l6_line(metadata: Mapping[str, Any]) -> str | None:
    """Render DV L6 MaxCLL/MaxFALL if present."""
    cll = metadata.get("l6_max_cll")
    fall = metadata.get("l6_max_fall")

    cll_label = _format_nits(cll if isinstance(cll, (int, float)) else None)
    fall_label = _format_nits(fall if isinstance(fall, (int, float)) else None)

    parts: list[str] = []
    if cll_label:
        parts.append(f"MaxCLL {cll_label}")
    if fall_label:
        parts.append(f"MaxFALL {fall_label}")

    if not parts:
        return None
    return f"DV L6 Metadata: {' / '.join(parts)}"


def format_hdr_line(metadata: Mapping[str, Any]) -> str | None:
    """Return an HDR metadata summary (MaxCLL/MaxFALL) line."""

    segments: list[str] = []
    cll_label = _format_nits(metadata.get("max_cll") if isinstance(metadata.get("max_cll"), (int, float)) else None)
    fall_label = _format_nits(metadata.get("max_fall") if isinstance(metadata.get("max_fall"), (int, float)) else None)
    if cll_label:
        segments.append(f"MaxCLL {cll_label}")
    if fall_label:
        segments.append(f"MaxFALL {fall_label}")
    if not segments:
        return None
    return f"HDR: {' / '.join(segments)}"


def format_dynamic_range_line(range_label: str | None) -> str | None:
    """Return a limited/full range summary if available."""

    if not range_label:
        return None
    text = range_label.strip()
    if not text:
        return None
    return f"Range: {text.capitalize()}"


def build_frame_metric_entry(frame: int, score: float | None, category: str | None, *, target_nits: float) -> dict[str, Any] | None:
    """Convert a selection score into a per-frame brightness estimate."""

    if score is None:
        return None
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return None
    if not (numeric_score == numeric_score):  # NaN check
        return None
    clamped = max(0.0, min(numeric_score, 1.0))
    avg = clamped * max(float(target_nits), 0.0)
    entry = {
        "frame": int(frame),
        "score": numeric_score,
        "avg_nits": avg,
        "max_nits": avg,
        "category": category,
    }
    return entry


def format_frame_metrics_line(entry: Mapping[str, Any] | None) -> str | None:
    """Return a human-readable per-frame nit summary."""

    if not entry:
        return None
    avg = entry.get("avg_nits")
    max_value = entry.get("max_nits")
    if not isinstance(avg, (int, float)) and not isinstance(max_value, (int, float)):
        return None
    avg_label = _format_nits(float(avg)) if isinstance(avg, (int, float)) else None
    max_label = _format_nits(float(max_value)) if isinstance(max_value, (int, float)) else None
    values: list[str] = []
    if max_label:
        values.append(f"{max_label}nits")
    if avg_label:
        values.append(f"{avg_label}nits")
    if not values:
        return None
    suffix = ""
    category = entry.get("category")
    if isinstance(category, str) and category.strip():
        suffix = f" ({category.strip()})"
    descriptor = "MAX/AVG" if max_label and avg_label else "MAX" if max_label else "AVG"
    return f"Measurement {descriptor}: {' / '.join(values)}{suffix}"
