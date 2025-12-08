"""Frame property helpers and colour metadata utilities."""
from __future__ import annotations

from typing import Any, Mapping, Optional, cast

_HDR_PRIMARIES_NAMES = {"bt2020", "bt.2020", "2020"}


_HDR_PRIMARIES_CODES = {9}


_HDR_TRANSFER_NAMES = {"st2084", "pq", "smpte2084", "hlg", "arib-b67"}


_HDR_TRANSFER_CODES = {16, 18}


_MATRIX_NAME_TO_CODE = {
    "rgb": 0,
    "0": 0,
    "bt709": 1,
    "bt.709": 1,
    "709": 1,
    "bt470bg": 5,
    "470bg": 5,
    "smpte170m": 6,
    "170m": 6,
    "bt601": 6,
    "601": 6,
    "bt2020": 9,
    "bt.2020": 9,
    "2020": 9,
    "2020ncl": 9,
}


_PRIMARIES_NAME_TO_CODE = {
    "bt709": 1,
    "bt.709": 1,
    "709": 1,
    "bt470bg": 5,
    "470bg": 5,
    "smpte170m": 6,
    "170m": 6,
    "bt601": 6,
    "601": 6,
    "bt2020": 9,
    "bt.2020": 9,
    "2020": 9,
}


_TRANSFER_NAME_TO_CODE = {
    "bt709": 1,
    "709": 1,
    "bt1886": 1,
    "gamma2.2": 1,
    "st2084": 16,
    "smpte2084": 16,
    "pq": 16,
    "hlg": 18,
    "arib-b67": 18,
    "smpte170m": 6,
    "170m": 6,
    "bt601": 6,
    "601": 6,
}


_RANGE_NAME_TO_CODE = {
    "limited": 1,
    "tv": 1,
    "full": 0,
    "pc": 0,
    "jpeg": 0,
}


_MATRIX_CODE_LABELS = {
    0: "rgb",
    1: "bt709",
    5: "bt470bg",
    6: "smpte170m",
    9: "bt2020",
}


_PRIMARIES_CODE_LABELS = {
    1: "bt709",
    5: "bt470bg",
    6: "smpte170m",
    9: "bt2020",
}


_TRANSFER_CODE_LABELS = {
    1: "bt1886",
    6: "smpte170m",
    16: "st2084",
    18: "hlg",
}


_RANGE_CODE_LABELS = {
    0: "full",
    1: "limited",
}


def _describe_code(value: Optional[int], mapping: Mapping[int, str], default: str = "auto") -> str:
    """
    Convert a numeric code to its human-readable label using a provided mapping.

    Parameters:
        value (Optional[int]): The code to describe; if `None`, `default` is returned.
        mapping (Mapping[int, str]): Mapping from integer codes to their human-readable labels.
        default (str): Value to return when `value` is `None`. Defaults to "auto".

    Returns:
        str: The label from `mapping` for `int(value)` if present; otherwise `default` when `value` is `None`, or `str(value)` if no mapping entry exists.
    """
    if value is None:
        return default
    try:
        return mapping[int(value)]
    except (ValueError, TypeError, KeyError):
        return str(value)


def _ensure_std_namespace(clip: Any, error: RuntimeError) -> Any:
    std = getattr(clip, "std", None)
    if std is None:
        raise error
    return std


def _call_set_frame_prop(set_prop: Any, clip: Any, **kwargs: Any) -> Any:
    try:
        return set_prop(clip, **kwargs)
    except TypeError as exc_first:
        try:
            return set_prop(**kwargs)
        except TypeError:
            raise exc_first from None


def _normalise_property_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _value_matches(value: Any, names: set[str], codes: set[int]) -> bool:
    if isinstance(value, int):
        return value in codes
    value = _normalise_property_value(value)
    if isinstance(value, str):
        return value in names
    return False


def _extract_frame_props(clip: Any) -> Mapping[str, Any]:
    getter = getattr(clip, "get_frame_props", None)
    if callable(getter):
        props = getter()
        if isinstance(props, Mapping):
            return cast(Mapping[str, Any], props)
    frame_props = getattr(clip, "frame_props", None)
    if isinstance(frame_props, Mapping):
        return cast(Mapping[str, Any], frame_props)
    return cast(Mapping[str, Any], {})


def _snapshot_frame_props(clip: Any) -> Mapping[str, Any]:
    try:
        frame = clip.get_frame(0)
    except (RuntimeError, ValueError, KeyError):
        return dict(_extract_frame_props(clip))
    props = getattr(frame, "props", None)
    if props is None:
        return dict(_extract_frame_props(clip))
    return dict(props)


def _props_signal_hdr(props: Mapping[str, Any]) -> bool:
    primaries = props.get("_Primaries") or props.get("Primaries")
    transfer = props.get("_Transfer") or props.get("Transfer")

    has_hdr_primaries = _value_matches(primaries, _HDR_PRIMARIES_NAMES, _HDR_PRIMARIES_CODES)
    has_hdr_transfer = _value_matches(transfer, _HDR_TRANSFER_NAMES, _HDR_TRANSFER_CODES)

    has_mastering_metadata = False
    for key in props:
        normalized = str(key).lstrip("_").lower()
        if normalized.startswith("masteringdisplay") or normalized.startswith("contentlightlevel"):
            has_mastering_metadata = True
            break

    if has_hdr_primaries and has_hdr_transfer:
        return True

    # Treat any HDR hint (single primaries/transfer flag or mastering metadata) as a partial signal.
    # Previously both flags had to be present, which muted clips that only carried MDL/CLL stats.
    if has_hdr_primaries or has_hdr_transfer or has_mastering_metadata:
        return True
    return False


def _coerce_prop(value: Any, mapping: Mapping[str, int] | None = None) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, (bytes, str)):
        normalized = _normalise_property_value(value)
        if isinstance(normalized, str):
            if mapping and normalized in mapping:
                return mapping[normalized]
            try:
                return int(normalized)
            except ValueError:
                return None
    return None


def _first_present(props: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in props:
            return props[key]
    return None


def _normalise_resolved_code(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    if code == 2:
        return None
    return code


def _resolve_color_metadata(
    props: Mapping[str, Any],
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    matrix = _coerce_prop(
        _first_present(props, "_Matrix", "Matrix"),
        _MATRIX_NAME_TO_CODE,
    )
    primaries = _coerce_prop(
        _first_present(props, "_Primaries", "Primaries"),
        _PRIMARIES_NAME_TO_CODE,
    )
    transfer = _coerce_prop(
        _first_present(props, "_Transfer", "Transfer"),
        _TRANSFER_NAME_TO_CODE,
    )
    color_range = _coerce_prop(
        _first_present(props, "_ColorRange", "ColorRange"),
        _RANGE_NAME_TO_CODE,
    )
    return (
        _normalise_resolved_code(matrix),
        _normalise_resolved_code(transfer),
        _normalise_resolved_code(primaries),
        _normalise_resolved_code(color_range),
    )


def _infer_frame_height(clip: Any, props: Mapping[str, Any]) -> Optional[int]:
    height = getattr(clip, "height", None)
    try:
        if height is not None:
            return int(height)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    for key in ("_Height", "Height"):
        candidate = props.get(key)
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _apply_frame_props_dict(clip: Any, props: Mapping[str, Any]) -> Any:
    if not props:
        return clip
    std_ns = getattr(clip, "std", None)
    if std_ns is None:
        return clip
    set_props = getattr(std_ns, "SetFrameProps", None)
    if not callable(set_props):  # pragma: no cover - depends on VapourSynth build
        return clip
    try:
        return _call_set_frame_prop(set_props, clip, **props)
    except (TypeError, ValueError, RuntimeError):
        return clip

__all__ = [
    "_HDR_PRIMARIES_NAMES",
    "_HDR_PRIMARIES_CODES",
    "_HDR_TRANSFER_NAMES",
    "_HDR_TRANSFER_CODES",
    "_describe_code",
    "_normalise_property_value",
    "_value_matches",
    "_extract_frame_props",
    "_snapshot_frame_props",
    "_props_signal_hdr",
    "_coerce_prop",
    "_first_present",
    "_normalise_resolved_code",
    "_resolve_color_metadata",
    "_infer_frame_height",
    "_MATRIX_NAME_TO_CODE",
    "_PRIMARIES_NAME_TO_CODE",
    "_TRANSFER_NAME_TO_CODE",
    "_RANGE_NAME_TO_CODE",
    "_MATRIX_CODE_LABELS",
    "_PRIMARIES_CODE_LABELS",
    "_TRANSFER_CODE_LABELS",
    "_RANGE_CODE_LABELS",
    "_ensure_std_namespace",
    "_call_set_frame_prop",
    "_apply_frame_props_dict",
]
