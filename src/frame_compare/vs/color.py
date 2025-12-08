"""Colour metadata inference, overrides, and heuristics."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, cast

from .env import _get_vapoursynth_module  # pyright: ignore[reportPrivateUsage]
from .props import (  # pyright: ignore[reportPrivateUsage]
    _MATRIX_NAME_TO_CODE,  # pyright: ignore[reportPrivateUsage]
    _PRIMARIES_NAME_TO_CODE,  # pyright: ignore[reportPrivateUsage]
    _RANGE_NAME_TO_CODE,  # pyright: ignore[reportPrivateUsage]
    _TRANSFER_NAME_TO_CODE,  # pyright: ignore[reportPrivateUsage]
    _apply_frame_props_dict,  # pyright: ignore[reportPrivateUsage]
    _coerce_prop,  # pyright: ignore[reportPrivateUsage]
    _infer_frame_height,  # pyright: ignore[reportPrivateUsage]
    _props_signal_hdr,  # pyright: ignore[reportPrivateUsage]
    _resolve_color_metadata,  # pyright: ignore[reportPrivateUsage]
)

logger = logging.getLogger("src.frame_compare.vs.color")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...datatypes import ColorConfig

def _resolve_configured_color_defaults(
    color_cfg: "ColorConfig | None",
    *,
    is_sd: bool,
    is_hd: bool,
) -> Dict[str, Optional[int]]:
    resolved: Dict[str, Optional[int]] = {
        "matrix": None,
        "primaries": None,
        "transfer": None,
        "range": None,
    }
    if color_cfg is None:
        return resolved

    def _value(name: str) -> Any:
        return getattr(color_cfg, name, None)

    def _coerce(value: Any, mapping: Mapping[str, int]) -> Optional[int]:
        if value in (None, ""):
            return None
        return _coerce_prop(value, mapping)

    if is_sd:
        resolved["matrix"] = _coerce(_value("default_matrix_sd"), _MATRIX_NAME_TO_CODE)
        resolved["primaries"] = _coerce(_value("default_primaries_sd"), _PRIMARIES_NAME_TO_CODE)
    elif is_hd:
        resolved["matrix"] = _coerce(_value("default_matrix_hd"), _MATRIX_NAME_TO_CODE)
        resolved["primaries"] = _coerce(_value("default_primaries_hd"), _PRIMARIES_NAME_TO_CODE)
    else:
        resolved["matrix"] = _coerce(
            _value("default_matrix_hd") or _value("default_matrix_sd"),
            _MATRIX_NAME_TO_CODE,
        )
        resolved["primaries"] = _coerce(
            _value("default_primaries_hd") or _value("default_primaries_sd"),
            _PRIMARIES_NAME_TO_CODE,
        )

    resolved["transfer"] = _coerce(_value("default_transfer_sdr"), _TRANSFER_NAME_TO_CODE)
    resolved["range"] = _coerce(_value("default_range_sdr"), _RANGE_NAME_TO_CODE)
    return resolved


def _resolve_color_overrides(
    color_cfg: "ColorConfig | None",
    file_name: str | Path | None,
) -> Dict[str, Optional[int]]:
    if color_cfg is None:
        return {}
    overrides: Dict[str, Dict[str, Any]] = getattr(color_cfg, "color_overrides", {})
    if not overrides:
        return {}

    lookup_keys: List[str] = []
    normalized_name = str(file_name) if file_name else ""
    if normalized_name:
        lookup_keys.append(normalized_name)
        lookup_keys.append(Path(normalized_name).name)
    lookup_keys.append("*")

    selected: Dict[str, Any] = {}
    for key in lookup_keys:
        if key in overrides:
            selected = overrides[key]
            break
    if not selected:
        return {}

    resolved: Dict[str, Optional[int]] = {}
    for attr, mapping in (
        ("matrix", _MATRIX_NAME_TO_CODE),
        ("primaries", _PRIMARIES_NAME_TO_CODE),
        ("transfer", _TRANSFER_NAME_TO_CODE),
        ("range", _RANGE_NAME_TO_CODE),
    ):
        value = selected.get(attr)
        if value in (None, ""):
            continue
        coerced = _coerce_prop(value, mapping)
        if coerced is None:
            logger.warning(
                "Ignoring invalid color_overrides value for %s: %r (file=%s)",
                attr,
                value,
                file_name or "",
            )
            continue
        resolved[attr] = coerced
    return resolved


def _guess_default_colourspace(
    clip: Any,
    props: Mapping[str, Any],
    matrix: Optional[int],
    transfer: Optional[int],
    primaries: Optional[int],
    color_range: Optional[int],
    *,
    color_cfg: "ColorConfig | None" = None,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    if _props_signal_hdr(props):
        return matrix, transfer, primaries, color_range

    vs_module = _get_vapoursynth_module()
    fmt = getattr(clip, "format", None)
    color_family = getattr(fmt, "color_family", None) if fmt is not None else None
    yuv_family = getattr(vs_module, "YUV", object())
    if color_family != yuv_family:
        return matrix, transfer, primaries, color_range

    height = _infer_frame_height(clip, props)
    is_sd = bool(height is not None and height <= 576)
    is_hd = bool(height is not None and height >= 720)
    configured = _resolve_configured_color_defaults(
        color_cfg,
        is_sd=is_sd,
        is_hd=is_hd,
    )

    if matrix is None:
        matrix = configured.get("matrix")
        if matrix is None:
            matrix = int(
                getattr(
                    vs_module,
                    "MATRIX_SMPTE170M" if is_sd else "MATRIX_BT709",
                    6 if is_sd else 1,
                )
            )
    if primaries is None:
        primaries = configured.get("primaries")
        if primaries is None:
            primaries = int(
                getattr(
                    vs_module,
                    "PRIMARIES_SMPTE170M" if is_sd else "PRIMARIES_BT709",
                    6 if is_sd else 1,
                )
            )
    if transfer is None:
        transfer = configured.get("transfer")
        if transfer is None:
            transfer = int(
                getattr(
                    vs_module,
                    "TRANSFER_SMPTE170M" if is_sd else "TRANSFER_BT709",
                    6 if is_sd else 1,
                )
            )
    if color_range is None:
        color_range = configured.get("range")
        if color_range is None:
            color_range = int(getattr(vs_module, "RANGE_LIMITED", 1))

    return matrix, transfer, primaries, color_range


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalise_to_8bit(
    sample_type: Any,
    bits_per_sample: Optional[int],
    value: Optional[float],
) -> Optional[float]:
    if value is None:
        return None
    try:
        sample_type_int = int(sample_type)
    except (ValueError, TypeError):
        sample_type_int = None
    if sample_type_int == 1:  # FLOAT
        return float(value) * 255.0
    if bits_per_sample is None or bits_per_sample <= 0:
        return None
    scale = float((1 << bits_per_sample) - 1)
    if scale <= 0.0:
        return None
    return float(value) * 255.0 / scale


def _classify_rgb_range_from_stats(
    sample_type: Any,
    bits_per_sample: Optional[int],
    y_min: Optional[float],
    y_max: Optional[float],
) -> Optional[str]:
    min_8 = _normalise_to_8bit(sample_type, bits_per_sample, y_min)
    max_8 = _normalise_to_8bit(sample_type, bits_per_sample, y_max)
    if min_8 is None or max_8 is None:
        return None

    limited_low = 16.0
    limited_high = 235.0
    limited_tolerance = 12.0
    limited_by_max = max_8 <= limited_high + limited_tolerance
    limited_by_band = (max_8 - min_8) <= 6.0 and min_8 <= limited_low + limited_tolerance
    if limited_by_max or limited_by_band:
        return "limited"

    full_margin_high = 6.0
    full_margin_low = 6.0
    full_high = max_8 >= limited_high + full_margin_high
    full_low = min_8 <= limited_low - full_margin_low
    if full_high and full_low:
        return "full"
    return None


def _detect_rgb_color_range(
    core: Any,
    clip: Any,
    *,
    log: logging.Logger,
    label: str,
    max_samples: int = 6,
) -> tuple[Optional[int], Optional[str]]:
    vs_module = _get_vapoursynth_module()
    std_ns = getattr(core, "std", None)
    plane_stats = getattr(std_ns, "PlaneStats", None) if std_ns is not None else None
    if not callable(plane_stats):
        return (None, None)

    fmt = getattr(clip, "format", None)
    if fmt is None:
        return (None, None)
    if getattr(fmt, "color_family", None) != getattr(vs_module, "RGB", object()):
        return (None, None)

    sample_type = getattr(fmt, "sample_type", None)
    bits_per_sample = getattr(fmt, "bits_per_sample", None)

    try:
        stats_clip = cast(Any, plane_stats(clip))
    except (ValueError, TypeError, RuntimeError) as exc:
        log.debug("[TM RANGE] %s failed to create PlaneStats node: %s", label, exc)
        return (None, None)

    total_frames = getattr(stats_clip, "num_frames", None)
    candidate_indices: set[int] = {0}
    if isinstance(total_frames, int) and total_frames > 0:
        candidate_indices.update(
            {
                max(0, total_frames - 1),
                total_frames // 2,
                total_frames // 4,
                (3 * total_frames) // 4,
            }
        )
    indices = sorted(idx for idx in candidate_indices if idx >= 0)
    indices = indices[:max_samples] if max_samples > 0 else indices

    limited_hits = 0
    full_hits = 0

    for idx in indices:
        try:
            frame = stats_clip.get_frame(idx)
        except (ValueError, RuntimeError) as exc:
            log.debug("[TM RANGE] %s failed to sample frame %s: %s", label, idx, exc)
            continue
        props = getattr(frame, "props", {})
        classification = _classify_rgb_range_from_stats(
            sample_type,
            bits_per_sample,
            _coerce_float(props.get("PlaneStatsMin")),
            _coerce_float(props.get("PlaneStatsMax")),
        )
        if classification == "limited":
            limited_hits += 1
        elif classification == "full":
            full_hits += 1

    limited_code = int(getattr(vs_module, "RANGE_LIMITED", 1))
    full_code = int(getattr(vs_module, "RANGE_FULL", 0))

    if limited_hits and not full_hits:
        log.info("[TM RANGE] %s detected limited RGB (samples=%d)", label, limited_hits)
        return (limited_code, "plane_stats")
    if full_hits and not limited_hits:
        log.info("[TM RANGE] %s detected full-range RGB (samples=%d)", label, full_hits)
        return (full_code, "plane_stats")
    if limited_hits and full_hits:
        log.warning(
            "[TM RANGE] %s samples span both limited and full ranges (limited=%d full=%d)",
            label,
            limited_hits,
            full_hits,
        )
    else:
        log.debug(
            "[TM RANGE] %s range detection inconclusive (limited=%d full=%d)",
            label,
            limited_hits,
            full_hits,
        )
    return (None, None)


def _compute_luma_bounds(clip: Any) -> tuple[Optional[float], Optional[float]]:
    core = getattr(clip, "core", None)
    std_ns = getattr(core, "std", None)
    plane_stats = getattr(std_ns, "PlaneStats", None) if std_ns is not None else None
    if not callable(plane_stats):
        return (None, None)
    try:
        stats_clip = cast(Any, plane_stats(clip))
    except (ValueError, TypeError, RuntimeError):
        return (None, None)

    total_frames = getattr(clip, "num_frames", 0) or 0
    if total_frames <= 0:
        indices = [0]
    else:
        indices = list(range(min(total_frames, 3)))

    mins: List[float] = []
    maxs: List[float] = []

    for idx in indices:
        try:
            frame = stats_clip.get_frame(idx)
        except (ValueError, RuntimeError):
            break
        props = getattr(frame, "props", {})
        y_min = _coerce_float(props.get("PlaneStatsMin"))
        y_max = _coerce_float(props.get("PlaneStatsMax"))
        if y_min is not None:
            mins.append(y_min)
        if y_max is not None:
            maxs.append(y_max)

    if not mins or not maxs:
        return (None, None)
    return (min(mins), max(maxs))


def _adjust_color_range_from_signal(
    clip: Any,
    *,
    color_range: Optional[int],
    warning_sink: Optional[List[str]],
    file_name: str | Path | None,
    range_inferred: bool,
    range_from_override: bool,
) -> Optional[int]:
    vs_module = _get_vapoursynth_module()
    limited_code = int(getattr(vs_module, "RANGE_LIMITED", 1))
    full_code = int(getattr(vs_module, "RANGE_FULL", 0))

    # If already limited and appears consistent, keep as-is but warn when signal contradicts metadata.
    y_min, y_max = _compute_luma_bounds(clip)
    if y_min is None or y_max is None:
        if range_inferred or (not range_from_override and color_range in (None, full_code)):
            message = (
                f"[COLOR] {file_name or 'clip'} lacks colour-range metadata and "
                "signal sampling is unavailable; defaulting to limited range."
            )
            logger.warning(message)
            if warning_sink is not None:
                warning_sink.append(message)
            return limited_code
        return color_range

    label = file_name or "clip"

    if range_inferred or color_range in (None, full_code):
        if 12.0 <= y_min <= 20.0 and y_max <= 245.0:
            message = (
                f"[COLOR] {label} lacks reliable colour-range metadata; "
                f"treating as limited (sample min={y_min:.1f}, max={y_max:.1f})."
            )
            logger.warning(message)
            if warning_sink is not None:
                warning_sink.append(message)
            return limited_code
    if color_range == limited_code and (y_min < 4.0 or y_max > 251.0):
        message = (
            f"[COLOR] {label} is tagged limited but sampled values span full range "
            f"(min={y_min:.1f}, max={y_max:.1f}); verify source metadata."
        )
        logger.warning(message)
        if warning_sink is not None:
            warning_sink.append(message)
    return color_range


def normalise_color_metadata(
    clip: Any,
    source_props: Mapping[str, Any] | None,
    *,
    color_cfg: "ColorConfig | None" = None,
    file_name: str | Path | None = None,
    warning_sink: Optional[List[str]] = None,
) -> tuple[
    Any,
    Mapping[str, Any],
    tuple[Optional[int], Optional[int], Optional[int], Optional[int]],
]:
    """Ensure colour metadata is usable, applying heuristics and overrides when needed."""

    props = dict(source_props or {})
    matrix, transfer, primaries, color_range = _resolve_color_metadata(props)

    overrides = _resolve_color_overrides(color_cfg, file_name)
    if "matrix" in overrides:
        matrix = overrides["matrix"]
        if matrix is not None:
            props["_Matrix"] = int(matrix)
    if "transfer" in overrides:
        transfer = overrides["transfer"]
        if transfer is not None:
            props["_Transfer"] = int(transfer)
    if "primaries" in overrides:
        primaries = overrides["primaries"]
        if primaries is not None:
            props["_Primaries"] = int(primaries)
    if "range" in overrides:
        color_range = overrides["range"]
        if color_range is not None:
            props["_ColorRange"] = int(color_range)
    range_from_override = "range" in overrides
    range_inferred = color_range is None and "range" not in overrides

    hdr_detected = _props_signal_hdr(props)

    matrix, transfer, primaries, color_range = _guess_default_colourspace(
        clip,
        props,
        matrix,
        transfer,
        primaries,
        color_range,
        color_cfg=color_cfg,
    )

    color_range = _adjust_color_range_from_signal(
        clip,
        color_range=color_range,
        warning_sink=warning_sink,
        file_name=file_name,
        range_inferred=range_inferred,
        range_from_override=range_from_override,
    )

    if hdr_detected:
        vs_module = _get_vapoursynth_module()

        def _coerce_hd_default(name: str, mapping: Mapping[str, int]) -> Optional[int]:
            if color_cfg is None:
                return None
            value = getattr(color_cfg, name, None)
            if value in (None, ""):
                return None
            return _coerce_prop(value, mapping)

        if matrix is None:
            matrix = _coerce_hd_default("default_matrix_hd", _MATRIX_NAME_TO_CODE)
            if matrix is None:
                matrix = int(
                    getattr(
                        vs_module,
                        "MATRIX_BT2020_CL",
                        getattr(vs_module, "MATRIX_BT2020_NCL", 9),
                    )
                )
        if primaries is None:
            primaries = _coerce_hd_default("default_primaries_hd", _PRIMARIES_NAME_TO_CODE)
            if primaries is None:
                primaries = int(getattr(vs_module, "PRIMARIES_BT2020", 9))
        if transfer is None:
            transfer = int(getattr(vs_module, "TRANSFER_ST2084", 16))

    update_props: Dict[str, int] = {}
    if matrix is not None:
        update_props["_Matrix"] = int(matrix)
        props["_Matrix"] = int(matrix)
    if transfer is not None:
        update_props["_Transfer"] = int(transfer)
        props["_Transfer"] = int(transfer)
    if primaries is not None:
        update_props["_Primaries"] = int(primaries)
        props["_Primaries"] = int(primaries)
    if color_range is not None:
        update_props["_ColorRange"] = int(color_range)
        props["_ColorRange"] = int(color_range)

    clip_with_props = _apply_frame_props_dict(clip, update_props)
    return clip_with_props, props, (matrix, transfer, primaries, color_range)

__all__ = [
    "_resolve_configured_color_defaults",
    "_resolve_color_overrides",
    "_guess_default_colourspace",
    "_coerce_float",
    "_normalise_to_8bit",
    "_classify_rgb_range_from_stats",
    "_detect_rgb_color_range",
    "_compute_luma_bounds",
    "_adjust_color_range_from_signal",
    "normalise_color_metadata",
]
