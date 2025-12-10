from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    cast,
)

from src.datatypes import RGBDither
from src.frame_compare import vs as vs_core
from src.frame_compare.render import encoders as _enc
from src.frame_compare.render import geometry as _geo
from src.frame_compare.render import overlay as _overlay
from src.frame_compare.render.errors import ScreenshotWriterError

logger = logging.getLogger(__name__)


def sanitize_for_log(value: object) -> str:
    """Return an ASCII-safe representation of *value* for logging."""

    text = str(value)
    try:
        return text.encode("ascii", "replace").decode("ascii")
    except (UnicodeError, AttributeError):
        return repr(value)


def range_constants() -> tuple[int, int]:
    """Return the VapourSynth range constant values (full, limited)."""

    try:
        import vapoursynth as vs  # type: ignore
    except ImportError:
        return (0, 1)
    return int(getattr(vs, "RANGE_FULL", 0)), int(getattr(vs, "RANGE_LIMITED", 1))


def set_clip_range(core: Any, clip: Any, color_range: int | None, *, context: str) -> Any:
    """Stamp `_ColorRange` on *clip* when ``SetFrameProps`` is available."""

    if color_range not in (0, 1, None):
        return clip
    std_ns = getattr(core, "std", None)
    set_props = getattr(std_ns, "SetFrameProps", None) if std_ns is not None else None
    if not callable(set_props):
        return clip
    try:
        if color_range is None:
            return clip
        return set_props(clip, _ColorRange=int(color_range))
    except (RuntimeError, ValueError) as exc:
        logger.debug("Failed to set _ColorRange during %s: %s", context, exc)
        return clip


def format_dimensions(width: int, height: int) -> str:
    """
    Format width and height as "W × H" using integer values.

    Returns:
        str: Formatted dimensions string, e.g. "1920 × 1080".
    """
    return _geo.format_dimensions(width, height)


def extract_mastering_display_luminance(props: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """
    Extract the mastering display minimum and maximum luminance from a properties mapping.

    Checks multiple common property keys for separate min/max entries first; if either is missing, looks for combined mastering display luminance entries that contain two values and uses their min and max as needed.

    Parameters:
        props (Mapping[str, Any]): Source properties that may contain mastering display luminance metadata under several possible keys.

    Returns:
        (min_luminance, max_luminance) (tuple[Optional[float], Optional[float]]): Tuple containing the extracted minimum and maximum mastering display luminance in nits, or `None` for any value that could not be determined.
    """
    return _overlay.extract_mastering_display_luminance(props)


def format_luminance_value(value: float) -> str:
    """
    Format a luminance value for display with sensible precision for small and large values.

    Parameters:
        value (float): Luminance in nits.

    Returns:
        str: Formatted luminance: values less than 1.0 are shown with up to four decimal places (trailing zeros and a trailing decimal point are removed), with "0" used if the result would be empty; values greater than or equal to 1.0 are shown with one decimal place.
    """
    return _overlay.format_luminance_value(value)


def format_mastering_display_line(props: Mapping[str, Any]) -> str:
    """
    Format a single-line Mastering Display Luminance (MDL) summary suitable for overlays or logs.

    Parameters:
        props (Mapping[str, Any]): Source metadata that may contain mastering display luminance information.

    Returns:
        str: A one-line MDL string. If both min and max luminance are available, returns
        "MDL: min: <min> cd/m², max: <max> cd/m²"; otherwise returns "MDL: Insufficient data".
    """
    return _overlay.format_mastering_display_line(props)


def normalize_selection_label(label: Optional[str]) -> str:
    """
    Normalize a selection label into a user-facing display name.

    Parameters:
        label (Optional[str]): Raw selection label (may be None or empty) typically from metadata.

    Returns:
        str: The cleaned display name for the selection; returns `"(unknown)"` if the input is missing or empty. Known internal labels are mapped to their canonical display names.
    """
    return _overlay.normalize_selection_label(label)


def format_selection_line(selection_label: Optional[str]) -> str:
    """
    Format the "Frame Selection Type" line for overlays and metadata.
    """
    return _overlay.format_selection_line(selection_label)


def resolve_resize_color_kwargs(props: Mapping[str, Any]) -> Dict[str, int]:
    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)
    kwargs: Dict[str, int] = {}
    if matrix is not None:
        kwargs["matrix_in"] = int(matrix)
    if transfer is not None:
        kwargs["transfer_in"] = int(transfer)
    if primaries is not None:
        kwargs["primaries_in"] = int(primaries)
    if color_range is not None:
        kwargs["range_in"] = int(color_range)
    return kwargs


def normalize_rgb_dither(value: RGBDither | str) -> RGBDither:
    """Normalise a value into an ``RGBDither`` enum with logging for invalid input."""

    try:
        return RGBDither(value)
    except (ValueError, TypeError):
        logger.debug(
            "Invalid rgb_dither value %r; defaulting to ERROR_DIFFUSION",
            value,
        )
        return RGBDither.ERROR_DIFFUSION


def copy_frame_props(core: Any, target: Any, source: Any, *, context: str) -> Any:
    """
    Best-effort propagation of frame properties from ``source`` to ``target``.
    """

    std_ns = getattr(core, "std", None)
    copy_props = getattr(std_ns, "CopyFrameProps", None) if std_ns is not None else None
    if not callable(copy_props):
        return target
    try:
        return copy_props(target, source)
    except (RuntimeError, ValueError) as exc:
        logger.debug("CopyFrameProps failed during %s: %s", context, exc)
        return target


def expand_limited_rgb(core: Any, clip: Any) -> Any:
    """Scale limited-range RGB to full range when possible."""

    std_ns = getattr(core, "std", None)
    levels = getattr(std_ns, "Levels", None) if std_ns is not None else None
    if not callable(levels):
        return clip

    fmt = getattr(clip, "format", None)
    bits = getattr(fmt, "bits_per_sample", None)
    if not isinstance(bits, int) or bits <= 0:
        return clip

    max_code = (1 << bits) - 1
    scale = 1 << (bits - 8) if bits >= 8 else 1

    min_in = 16 * scale
    max_in = 235 * scale

    try:
        return levels(
            clip,
            min_in=min_in,
            max_in=max_in,
            min_out=0,
            max_out=max_code,
            planes=[0, 1, 2],
        )
    except (RuntimeError, ValueError) as exc:
        logger.debug("Failed to expand limited RGB range: %s", exc)
        return clip


def finalize_existing_rgb24(
    core: Any,
    clip: Any,
    *,
    source_props: Mapping[str, Any] | None,
    target_range: int | None,
    expand_to_full: bool,
    range_full: int,
    range_limited: int,
) -> Any:
    """Adjust metadata and optional range expansion for clips already in RGB24."""

    props: Dict[str, Any] = dict(source_props or {})
    if not props:
        try:
            props = dict(vs_core.snapshot_frame_props(clip))
        except (RuntimeError, ValueError, KeyError):
            props = {}

    tonemapped_flag = props.get("_Tonemapped")
    is_tonemapped = tonemapped_flag is not None

    _, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)
    current_range = range_limited if color_range is None else int(color_range)
    if current_range not in (range_full, range_limited):
        current_range = range_limited

    output_range = current_range
    source_range_meta: int | None = None

    if expand_to_full:
        if current_range == range_limited and not is_tonemapped:
            clip = expand_limited_rgb(core, clip)
            output_range = range_full
            source_range_meta = range_limited
        elif current_range in (range_full, range_limited):
            output_range = current_range
        else:
            output_range = range_full
    elif target_range in (range_full, range_limited):
        output_range = int(target_range)

    std_ns = getattr(core, "std", None)
    set_props = getattr(std_ns, "SetFrameProps", None) if std_ns is not None else None
    if callable(set_props):
        prop_kwargs: Dict[str, int] = {"_Matrix": 0, "_ColorRange": int(output_range)}
        if primaries is not None:
            prop_kwargs["_Primaries"] = int(primaries)
        elif isinstance(props.get("_Primaries"), int):
            prop_kwargs["_Primaries"] = int(props["_Primaries"])
        if transfer is not None:
            prop_kwargs["_Transfer"] = int(transfer)
        elif isinstance(props.get("_Transfer"), int):
            prop_kwargs["_Transfer"] = int(props["_Transfer"])
        if source_range_meta is not None and source_range_meta != output_range:
            prop_kwargs["_SourceColorRange"] = int(source_range_meta)
        elif (
            not expand_to_full
            and isinstance(props.get("_SourceColorRange"), int)
        ):
            prop_kwargs["_SourceColorRange"] = int(props["_SourceColorRange"])
        try:
            clip = cast(Any, set_props(clip, **prop_kwargs))
        except (RuntimeError, ValueError) as exc:
            logger.debug("Failed to set RGB frame props: %s", exc)
    return clip


def ensure_rgb24(
    core: Any,
    clip: Any,
    frame_idx: int,
    *,
    source_props: Mapping[str, Any] | None = None,
    rgb_dither: RGBDither | str = RGBDither.ERROR_DIFFUSION,
    target_range: int | None = None,
    expand_to_full: bool = False,
) -> Any:
    """
    Ensure the given VapourSynth frame is in 8-bit RGB24 color format.

    Parameters:
        core (Any): VapourSynth core instance used for conversions.
        clip (Any): VapourSynth clip or frame to validate/convert.
        frame_idx (int): Index of the frame being processed (used in error messages).

    Returns:
        Any: A clip in RGB24 using the requested range; returns the original clip if it already is 8-bit RGB24.

    Raises:
        ScreenshotWriterError: If VapourSynth is unavailable, the core lacks the resize namespace or Point, or the conversion fails.
    """
    try:
        import vapoursynth as vs  # type: ignore
    except ImportError as exc:
        raise ScreenshotWriterError("VapourSynth is required for screenshot export") from exc

    range_full = int(getattr(vs, "RANGE_FULL", 0))
    range_limited = int(getattr(vs, "RANGE_LIMITED", 1))

    fmt = getattr(clip, "format", None)
    color_family = getattr(fmt, "color_family", None) if fmt is not None else None
    bits = getattr(fmt, "bits_per_sample", None) if fmt is not None else None
    if color_family == getattr(vs, "RGB", object()) and bits == 8:
        return finalize_existing_rgb24(
            core,
            clip,
            source_props=source_props,
            target_range=target_range,
            expand_to_full=expand_to_full,
            range_full=range_full,
            range_limited=range_limited,
        )

    resize_ns = getattr(core, "resize", None)
    if resize_ns is None:
        raise ScreenshotWriterError("VapourSynth core is missing resize namespace")
    point = getattr(resize_ns, "Point", None)
    if not callable(point):
        raise ScreenshotWriterError("VapourSynth resize.Point is unavailable")

    dither = normalize_rgb_dither(rgb_dither).value
    props = dict(source_props or {})
    if not props:
        props = dict(vs_core.snapshot_frame_props(clip))
    resize_kwargs = resolve_resize_color_kwargs(props)
    _, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)
    resize_kwargs.pop("transfer_in", None)
    resize_kwargs.pop("primaries_in", None)

    yuv_constant = getattr(vs, "YUV", object())
    if color_family == yuv_constant:
        defaults: Dict[str, int] = {}
        if "matrix_in" not in resize_kwargs:
            defaults["matrix_in"] = int(getattr(vs, "MATRIX_BT709", 1))
        if "transfer_in" not in resize_kwargs:
            defaults["transfer_in"] = int(getattr(vs, "TRANSFER_BT709", 1))
        if "primaries_in" not in resize_kwargs:
            defaults["primaries_in"] = int(getattr(vs, "PRIMARIES_BT709", 1))
        if "range_in" not in resize_kwargs:
            defaults["range_in"] = int(getattr(vs, "RANGE_LIMITED", 1))
        if defaults:
            resize_kwargs.update(defaults)
            logger.debug(
                "Colour metadata missing for frame %s; applying Rec.709 limited defaults",
                frame_idx,
            )

    source_range = range_limited if color_range is None else int(color_range)
    if source_range not in (range_full, range_limited):
        source_range = range_full

    desired_range = None
    if target_range in (range_full, range_limited):
        desired_range = int(target_range)
    else:
        range_hint = resize_kwargs.get("range_in")
        if range_hint in (range_full, range_limited):
            desired_range = int(range_hint)
    if desired_range is None:
        desired_range = source_range

    output_range = range_full if expand_to_full else desired_range

    try:
        converted = cast(
            Any,
            point(
                clip,
                format=vs.RGB24,
                range=output_range,
                dither_type=dither,
                **resize_kwargs,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        raise ScreenshotWriterError(f"Failed to convert frame {frame_idx} to RGB24: {exc}") from exc

    converted = copy_frame_props(core, converted, clip, context="RGB24 conversion")
    if expand_to_full and source_range == range_limited:
        converted = expand_limited_rgb(core, converted)
        converted = copy_frame_props(core, converted, clip, context="RGB24 expansion")

    try:
        prop_kwargs: Dict[str, int] = {"_Matrix": 0, "_ColorRange": int(output_range)}
        if primaries is not None:
            prop_kwargs["_Primaries"] = int(primaries)
        elif isinstance(props.get("_Primaries"), int):
            prop_kwargs["_Primaries"] = int(props["_Primaries"])
        if transfer is not None:
            prop_kwargs["_Transfer"] = int(transfer)
        elif isinstance(props.get("_Transfer"), int):
            prop_kwargs["_Transfer"] = int(props["_Transfer"])
        if expand_to_full and source_range != output_range:
            prop_kwargs["_SourceColorRange"] = int(source_range)
        converted = converted.std.SetFrameProps(**prop_kwargs)
    except (RuntimeError, ValueError) as exc:
        logger.debug("Failed to set RGB frame props: %s", exc)
    return converted


def restore_color_props(
    core: Any,
    clip: Any,
    props: Mapping[str, Any],
    *,
    context: str,
    include_color_range: bool = True,
) -> Any:
    """Reapply colour metadata to *clip* based on *props*."""

    std_ns = getattr(core, "std", None)
    set_props = getattr(std_ns, "SetFrameProps", None) if std_ns is not None else None
    if not callable(set_props):
        return clip

    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)
    prop_kwargs: Dict[str, int] = {}
    if matrix is not None:
        prop_kwargs["_Matrix"] = int(matrix)
    if transfer is not None:
        prop_kwargs["_Transfer"] = int(transfer)
    if primaries is not None:
        prop_kwargs["_Primaries"] = int(primaries)
    if include_color_range and color_range is not None:
        prop_kwargs["_ColorRange"] = int(color_range)
    if not prop_kwargs:
        return clip
    try:
        return set_props(clip, **prop_kwargs)
    except (RuntimeError, ValueError) as exc:
        logger.debug("Failed to restore colour props during %s: %s", context, exc)
        return clip


def map_fpng_compression(level: int) -> int:
    return _enc.map_fpng_compression(level)
