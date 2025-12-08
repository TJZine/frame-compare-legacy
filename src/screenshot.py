"""Screenshot planning and export utilities."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)

from src.frame_compare import subproc as _subproc
from src.frame_compare import vs as vs_core
from src.frame_compare.render import encoders as _enc
from src.frame_compare.render import geometry as _geo
from src.frame_compare.render import naming as _naming
from src.frame_compare.render import overlay as _overlay
from src.frame_compare.render.errors import (
    ScreenshotError,
    ScreenshotGeometryError,
    ScreenshotWriterError,
)

from .datatypes import (
    AutoLetterboxCropMode,
    ColorConfig,
    ExportRange,
    OddGeometryPolicy,
    RGBDither,
    ScreenshotConfig,
)

logger = logging.getLogger(__name__)

OverlayStateValue = _overlay.OverlayStateValue
OverlayState = _overlay.OverlayState

FRAME_INFO_STYLE = _overlay.FRAME_INFO_STYLE
OVERLAY_STYLE = _overlay.OVERLAY_STYLE


class FrameEvalFunc(Protocol):
    def __call__(
        self,
        clip: Any,
        func: Callable[[int, Any], Any],
        *,
        prop_src: Any | None = None,
    ) -> Any:
        ...


class SubtitleFunc(Protocol):
    def __call__(
        self,
        clip: Any,
        *,
        text: Sequence[str] | None = None,
        style: Any | None = None,
    ) -> Any:
        ...


def _new_overlay_state() -> OverlayState:
    """Create a mutable overlay state container."""
    return _overlay.new_overlay_state()


def _range_constants() -> tuple[int, int]:
    """Return the VapourSynth range constant values (full, limited)."""

    try:
        import vapoursynth as vs  # type: ignore
    except ImportError:
        return (0, 1)
    return int(getattr(vs, "RANGE_FULL", 0)), int(getattr(vs, "RANGE_LIMITED", 1))


def _should_expand_to_full(export_range: ExportRange | str | None) -> bool:
    """Return True when final RGB PNGs should be exported in full range."""

    if isinstance(export_range, ExportRange):
        return export_range is ExportRange.FULL
    if isinstance(export_range, str):
        return export_range.strip().lower() == ExportRange.FULL.value
    return False


def _sanitize_for_log(value: object) -> str:
    """Return an ASCII-safe representation of *value* for logging."""

    text = str(value)
    try:
        return text.encode("ascii", "replace").decode("ascii")
    except (UnicodeError, AttributeError):
        return repr(value)


def _set_clip_range(core: Any, clip: Any, color_range: int | None, *, context: str) -> Any:
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


def _append_overlay_warning(state: OverlayState, message: str) -> None:
    """
    Append a formatted overlay warning to the state's warning list in a type-safe manner.
    """
    _overlay.append_overlay_warning(state, message)


def _get_overlay_warnings(state: OverlayState) -> List[str]:
    """
    Retrieve overlay warning messages from state, returning an empty list when absent.
    """
    return _overlay.get_overlay_warnings(state)


def _format_dimensions(width: int, height: int) -> str:
    """
    Format width and height as "W × H" using integer values.

    Returns:
        str: Formatted dimensions string, e.g. "1920 × 1080".
    """
    return _geo.format_dimensions(width, height)


def _extract_mastering_display_luminance(props: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """
    Extract the mastering display minimum and maximum luminance from a properties mapping.

    Checks multiple common property keys for separate min/max entries first; if either is missing, looks for combined mastering display luminance entries that contain two values and uses their min and max as needed.

    Parameters:
        props (Mapping[str, Any]): Source properties that may contain mastering display luminance metadata under several possible keys.

    Returns:
        (min_luminance, max_luminance) (tuple[Optional[float], Optional[float]]): Tuple containing the extracted minimum and maximum mastering display luminance in nits, or `None` for any value that could not be determined.
    """
    return _overlay.extract_mastering_display_luminance(props)


def _format_luminance_value(value: float) -> str:
    """
    Format a luminance value for display with sensible precision for small and large values.

    Parameters:
        value (float): Luminance in nits.

    Returns:
        str: Formatted luminance: values less than 1.0 are shown with up to four decimal places (trailing zeros and a trailing decimal point are removed), with "0" used if the result would be empty; values greater than or equal to 1.0 are shown with one decimal place.
    """
    return _overlay.format_luminance_value(value)


def _format_mastering_display_line(props: Mapping[str, Any]) -> str:
    """
    Format a single-line Mastering Display Luminance (MDL) summary suitable for overlays or logs.

    Parameters:
        props (Mapping[str, Any]): Source metadata that may contain mastering display luminance information.

    Returns:
        str: A one-line MDL string. If both min and max luminance are available, returns
        "MDL: min: <min> cd/m², max: <max> cd/m²"; otherwise returns "MDL: Insufficient data".
    """
    return _overlay.format_mastering_display_line(props)


def _normalize_selection_label(label: Optional[str]) -> str:
    """
    Normalize a selection label into a user-facing display name.

    Parameters:
        label (Optional[str]): Raw selection label (may be None or empty) typically from metadata.

    Returns:
        str: The cleaned display name for the selection; returns `"(unknown)"` if the input is missing or empty. Known internal labels are mapped to their canonical display names.
    """
    return _overlay.normalize_selection_label(label)


def _format_selection_line(selection_label: Optional[str]) -> str:
    """
    Format the "Frame Selection Type" line for overlays and metadata.
    """
    return _overlay.format_selection_line(selection_label)


def _compose_overlay_text(
    base_text: Optional[str],
    color_cfg: ColorConfig,
    plan: GeometryPlan,
    selection_label: Optional[str],
    source_props: Mapping[str, Any],
    *,
    tonemap_info: Optional[vs_core.TonemapInfo],
    selection_detail: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """
    Compose overlay text for a frame when overlays are enabled.
    """
    return _overlay.compose_overlay_text(
        base_text,
        color_cfg,
        plan,
        selection_label,
        source_props,
        tonemap_info=tonemap_info,
        selection_detail=selection_detail,
    )




def _resolve_resize_color_kwargs(props: Mapping[str, Any]) -> Dict[str, int]:
    """Build resize arguments describing the source clip's colour space."""

    matrix, transfer, primaries, color_range = vs_core._resolve_color_metadata(props)

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


def _normalize_rgb_dither(value: RGBDither | str) -> RGBDither:
    """Normalise a value into an ``RGBDither`` enum with logging for invalid input."""

    try:
        return RGBDither(value)
    except (ValueError, TypeError):
        logger.debug(
            "Invalid rgb_dither value %r; defaulting to ERROR_DIFFUSION",
            value,
        )
        return RGBDither.ERROR_DIFFUSION


class _ColorDebugState:
    """Collects colour debug logs and intermediate PNGs for a single clip."""

    def __init__(
        self,
        *,
        enabled: bool,
        base_dir: Path,
        label: str,
        core: Any | None,
        compression_level: int,
        rgb_dither: RGBDither,
        logger_obj: logging.Logger,
        artifacts: Optional[vs_core.ColorDebugArtifacts],
    ) -> None:
        self.enabled = bool(enabled and core is not None)
        self.base_dir = base_dir
        self.label = label
        self.core = core
        self._logger = logger_obj
        self._compression_value = _map_fpng_compression(compression_level)
        self._rgb_dither = rgb_dither
        self._writer: Optional[Callable[..., Any]] = None
        self._warned_writer = False
        self.normalized_clip = artifacts.normalized_clip if artifacts is not None else None
        self.normalized_props = (
            dict(artifacts.normalized_props)
            if artifacts is not None and artifacts.normalized_props is not None
            else None
        )
        self.original_props = (
            dict(artifacts.original_props)
            if artifacts is not None and artifacts.original_props is not None
            else None
        )
        self.color_tuple = artifacts.color_tuple if artifacts is not None else None

    def capture_stage(
        self,
        stage: str,
        frame_idx: int,
        clip: Any | None,
        props: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled or clip is None:
            return
        try:
            props_map = dict(props or vs_core._snapshot_frame_props(clip))
        except (KeyError, ValueError, RuntimeError, AttributeError):
            props_map = {}
        try:
            y_min, y_max = self._measure_plane_bounds(clip, frame_idx)
        except (RuntimeError, ValueError, KeyError) as exc:
            self._logger.debug(
                "Colour debug PlaneStats failed for %s stage=%s frame=%s: %s",
                self.label,
                stage,
                frame_idx,
                exc,
            )
            y_min = y_max = float("nan")
        self._logger.info(
            "[DEBUG-COLOR] %s stage=%s frame=%d Matrix=%s Transfer=%s Primaries=%s Range=%s Ymin=%.2f Ymax=%.2f",
            self.label,
            stage,
            frame_idx,
            props_map.get("_Matrix"),
            props_map.get("_Transfer"),
            props_map.get("_Primaries"),
            props_map.get("_ColorRange"),
            y_min,
            y_max,
        )
        try:
            self._write_png(stage, frame_idx, clip, props_map)
        except (RuntimeError, ValueError, KeyError) as exc:
            self._logger.debug(
                "Colour debug PNG write failed for %s stage=%s frame=%s: %s",
                self.label,
                stage,
                frame_idx,
                exc,
            )

    def _measure_plane_bounds(self, clip: Any, frame_idx: int) -> tuple[float, float]:
        std_ns = getattr(self.core, "std", None) if self.core is not None else None
        plane_stats = getattr(std_ns, "PlaneStats", None) if std_ns is not None else None
        if not callable(plane_stats):
            raise RuntimeError("VapourSynth std.PlaneStats unavailable for colour debug")
        stats_clip = cast(Any, plane_stats(clip))
        frame = stats_clip.get_frame(frame_idx)
        props = cast(Mapping[str, Any], getattr(frame, "props", {}))
        min_value = float(props.get("PlaneStatsMin", 0.0))
        max_value = float(props.get("PlaneStatsMax", 0.0))
        return (min_value, max_value)

    def _write_png(
        self,
        stage: str,
        frame_idx: int,
        clip: Any,
        props: Mapping[str, Any],
    ) -> None:
        path = self.base_dir / f"{frame_idx:06d}_{stage}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._writer is None:
            fpng_ns = getattr(self.core, "fpng", None) if self.core is not None else None
            writer = getattr(fpng_ns, "Write", None) if fpng_ns is not None else None
            if not callable(writer):
                if not self._warned_writer:
                    self._logger.warning(
                        "Colour debug unable to emit PNGs for %s (fpng.Write unavailable)",
                        self.label,
                    )
                    self._warned_writer = True
                return
            self._writer = writer
        _, _, _, debug_range = vs_core._resolve_color_metadata(props)
        rgb_clip = _ensure_rgb24(
            self.core,
            clip,
            frame_idx,
            source_props=props,
            rgb_dither=self._rgb_dither,
            target_range=int(debug_range) if debug_range is not None else None,
        )
        job = self._writer(rgb_clip, str(path), compression=self._compression_value, overwrite=True)
        job.get_frame(frame_idx)


def _legacy_rgb24_from_clip(
    core: Any,
    clip: Any,
    color_tuple: tuple[Optional[int], Optional[int], Optional[int], Optional[int]] | None,
    *,
    expand_to_full: bool = False,
) -> Any | None:
    if color_tuple is None:
        return None
    try:
        import vapoursynth as vs  # type: ignore
    except ImportError:
        return None

    resize_ns = getattr(core, "resize", None)
    lanczos = getattr(resize_ns, "Lanczos", None) if resize_ns is not None else None
    if not callable(lanczos):
        return None

    matrix, transfer, primaries, color_range = color_tuple
    kwargs: Dict[str, int] = {}
    if matrix is not None:
        kwargs["matrix_in"] = int(matrix)
    # Legacy pipeline historically relied on matrix only; avoid injecting
    # transfer/primaries hints to mirror original conversion behaviour.
    range_limited = int(getattr(vs, "RANGE_LIMITED", 1))
    range_full = int(getattr(vs, "RANGE_FULL", 0))
    resolved_range_raw = range_limited if color_range is None else int(color_range)
    source_range = resolved_range_raw if resolved_range_raw in {range_limited, range_full} else range_full
    kwargs["range_in"] = source_range
    target_range = range_full if expand_to_full else source_range
    try:
        legacy_rgb = lanczos(
            clip,
            format=vs.RGB24,
            dither_type="error_diffusion",
            range=target_range,
            **kwargs,
        )
    except (RuntimeError, ValueError):
        return None
    legacy_rgb = _copy_frame_props(core, legacy_rgb, clip, context="legacy RGB24 conversion")
    if expand_to_full and source_range == range_limited:
        legacy_rgb = _expand_limited_rgb(core, legacy_rgb)
        legacy_rgb = _copy_frame_props(core, legacy_rgb, clip, context="legacy RGB24 expansion")
    try:
        prop_kwargs: Dict[str, int] = {"_Matrix": 0, "_ColorRange": int(target_range)}
        if primaries is not None:
            prop_kwargs["_Primaries"] = int(primaries)
        if transfer is not None:
            prop_kwargs["_Transfer"] = int(transfer)
        if expand_to_full and source_range != target_range:
            prop_kwargs["_SourceColorRange"] = int(source_range)
        legacy_rgb = cast(Any, legacy_rgb.std.SetFrameProps(**prop_kwargs))
    except (RuntimeError, ValueError) as exc:
        logger.debug("Failed to set legacy RGB frame props: %s", exc)
    return legacy_rgb


def _finalize_existing_rgb24(
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
            props = dict(vs_core._snapshot_frame_props(clip))
        except (RuntimeError, ValueError, KeyError):
            props = {}

    tonemapped_flag = props.get("_Tonemapped")
    is_tonemapped = tonemapped_flag is not None

    matrix, transfer, primaries, color_range = vs_core._resolve_color_metadata(props)
    current_range = range_limited if color_range is None else int(color_range)
    if current_range not in (range_full, range_limited):
        current_range = range_limited

    output_range = current_range
    source_range_meta: int | None = None

    if expand_to_full:
        if current_range == range_limited and not is_tonemapped:
            clip = _expand_limited_rgb(core, clip)
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


def _ensure_rgb24(
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
        return _finalize_existing_rgb24(
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

    dither = _normalize_rgb_dither(rgb_dither).value
    props = dict(source_props or {})
    if not props:
        props = dict(vs_core._snapshot_frame_props(clip))
    resize_kwargs = _resolve_resize_color_kwargs(props)
    matrix, transfer, primaries, color_range = vs_core._resolve_color_metadata(props)
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

    converted = _copy_frame_props(core, converted, clip, context="RGB24 conversion")
    if expand_to_full and source_range == range_limited:
        converted = _expand_limited_rgb(core, converted)
        converted = _copy_frame_props(core, converted, clip, context="RGB24 expansion")

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
        converted = cast(Any, converted.std.SetFrameProps(**prop_kwargs))
    except (RuntimeError, ValueError) as exc:
        logger.debug("Failed to set RGB frame props: %s", exc)
    return converted


def _clamp_frame_index(clip: Any, frame_idx: int) -> tuple[int, bool]:
    """
    Clamp ``frame_idx`` to the clip's valid range and flag when adjustment occurred.

    Parameters:
        clip (Any): Clip providing a ``num_frames`` attribute describing valid indices.
        frame_idx (int): Desired frame index.

    Returns:
        tuple[int, bool]: Tuple of the clamped frame index and ``True`` when the value was adjusted.
    """
    total_frames = getattr(clip, "num_frames", None)
    if not isinstance(total_frames, int) or total_frames <= 0:
        return max(0, int(frame_idx)), False
    max_index = max(0, total_frames - 1)
    clamped = max(0, min(int(frame_idx), max_index))
    return clamped, clamped != frame_idx

def _apply_frame_info_overlay(
    core: Any,
    clip: Any,
    title: str,
    requested_frame: int | None,
    selection_label: str | None,
) -> Any:
    """
    Add a per-frame information overlay to a VapourSynth clip.

    Attempts to draw a small text block containing the frame index and picture type onto the provided clip.
    If the required VapourSynth namespaces or overlay functions are unavailable, or an error occurs while applying
    the overlay, the original clip is returned unchanged.

    Parameters:
        core: VapourSynth core object used to access std and sub namespaces.
        clip: VapourSynth clip to annotate.
        title: Title text shown above the per-frame info; falls back to "Clip" when empty.
        requested_frame: Frame index to display in the overlay instead of the intrinsic evaluation index; pass None to use the evaluation index.
        selection_label: Optional selection label (not used by this function).

    Returns:
        The annotated clip if overlay application succeeded, otherwise the original clip.
    """
    std_ns = getattr(core, "std", None)
    sub_ns = getattr(core, "sub", None)
    if std_ns is None or sub_ns is None:
        logger.debug('VapourSynth core missing std/sub namespaces; skipping frame overlay')
        return clip

    frame_eval_obj = getattr(std_ns, 'FrameEval', None)
    subtitle_obj = getattr(sub_ns, 'Subtitle', None)
    if not callable(frame_eval_obj) or not callable(subtitle_obj):
        logger.debug('Required VapourSynth overlay functions unavailable; skipping frame overlay')
        return clip
    frame_eval = cast(FrameEvalFunc, frame_eval_obj)
    subtitle = cast(SubtitleFunc, subtitle_obj)

    label = title.strip() if isinstance(title, str) else ''
    if not label:
        label = 'Clip'

    padding_title = " " + ("\n" * 3)

    def _draw_info(n: int, f: Any, clip_ref: Any) -> Any:
        """
        Create a subtitle node containing per-frame information (frame index and picture type) for overlay.

        Parameters:
            n (int): Evaluation frame index provided by the frame evaluation callback.
            f: Frame object whose properties (e.g., `_PictType`) will be read.
            clip_ref: Source clip reference used to construct the subtitle node.

        Returns:
            A subtitle clip node containing the formatted frame information text.
        """
        pict = f.props.get('_PictType')
        if isinstance(pict, bytes):
            pict_text = pict.decode('utf-8', 'ignore')
        elif isinstance(pict, str):
            pict_text = pict
        else:
            pict_text = 'N/A'
        display_idx = requested_frame if requested_frame is not None else n
        lines: List[str] = [
            f"Frame {display_idx} of {clip_ref.num_frames}",
            f"Picture type: {pict_text}",
        ]
        info = "\n".join(lines)
        return subtitle(clip_ref, text=[info], style=FRAME_INFO_STYLE)

    def _frame_info_callback(
        n: int,
        f: Any,
        clip_ref: Any = clip,
        **kwargs: Any,
    ) -> Any:
        """
        Wrapper passed to std.FrameEval that tolerates keyword arguments supplied by VapourSynth.
        """

        clip_override = kwargs.get("clip")
        if clip_override is not None:
            clip_ref = clip_override
        return _draw_info(n, f, clip_ref)

    try:
        info_clip = frame_eval(clip, _frame_info_callback, prop_src=clip)
        result = subtitle(info_clip, text=[padding_title + label], style=FRAME_INFO_STYLE)
        result = _copy_frame_props(core, result, clip, context="frame info overlay")
        return result
    except (RuntimeError, ValueError) as exc:
        logger.debug('Applying frame overlay failed: %s', exc)
        return clip


def _apply_overlay_text(
    core: Any,
    clip: Any,
    text: Optional[str],
    *,
    strict: bool,
    state: OverlayState,
    file_label: str,
) -> object:
    """
    Apply diagnostic text as an overlay to a VapourSynth clip and update overlay state.

    Attempts to render `text` on `clip` using available overlay filters from `core`. If `text` is falsy or the overlay status in `state` is "error", the original `clip` is returned unchanged. On successful application the returned clip contains the rendered overlay and `state["overlay_status"]` is set to `"ok"`. On failure the function records an error message into `state["warnings"]` (prefixed with "[OVERLAY]") and sets `state["overlay_status"]` to `"error"`; when `strict` is True a ScreenshotWriterError is raised instead of returning the original clip.

    Parameters:
        core: The VapourSynth core object providing overlay/filter namespaces.
        clip: The VapourSynth clip to receive the overlay.
        text: The overlay text to apply; if None or empty, no action is taken.
        strict (bool): If True, raise ScreenshotWriterError on overlay unavailability or failure.
        state (OverlayState): Mutable state mapping used and updated by this function. Recognized keys:
            - "overlay_status": read and updated to "ok" or "error".
            - "warnings": a list that will be appended with overlay-related warning strings when failures occur.
        file_label (str): Human-readable identifier for logging and warning messages.

    Returns:
        The clip with the overlay applied, or the original `clip` if no overlay was applied.

    Raises:
        ScreenshotWriterError: If an overlay cannot be applied and `strict` is True.
    """
    if not text:
        return clip
    status = state.get("overlay_status")
    if status == "error":
        return clip
    sub_ns = getattr(core, "sub", None)
    subtitle = getattr(sub_ns, "Subtitle", None) if sub_ns is not None else None
    if callable(subtitle):
        try:
            result = subtitle(clip, text=[text], style=OVERLAY_STYLE)
        except (RuntimeError, ValueError) as exc:
            logger.debug('Subtitle overlay failed, falling back: %s', exc)
        else:
            result = _copy_frame_props(core, result, clip, context="overlay preservation")
            if status != "ok":
                logger.info('[OVERLAY] %s applied', file_label)
                state["overlay_status"] = "ok"
            return result

    text_ns = getattr(core, "text", None)
    draw = getattr(text_ns, "Text", None) if text_ns is not None else None
    if not callable(draw):
        message = f"Overlay filter unavailable for {file_label}"
        logger.error('[OVERLAY] %s', message)
        state["overlay_status"] = "error"
        _append_overlay_warning(state, f"[OVERLAY] {message}")
        if strict:
            raise ScreenshotWriterError(message)
        return clip
    try:
        result = draw(clip, text, alignment=9)
    except (RuntimeError, ValueError) as exc:
        message = f"Overlay failed for {file_label}: {exc}"
        logger.error('[OVERLAY] %s', message)
        state["overlay_status"] = "error"
        _append_overlay_warning(state, f"[OVERLAY] {message}")
        if strict:
            raise ScreenshotWriterError(message) from exc
        return clip
    result = _copy_frame_props(core, result, clip, context="overlay preservation")
    if status != "ok":
        logger.info('[OVERLAY] %s applied', file_label)
        state["overlay_status"] = "ok"
    return result


def _copy_frame_props(core: Any, target: Any, source: Any, *, context: str) -> Any:
    """
    Best-effort propagation of frame properties from ``source`` to ``target``.

    Parameters:
        core: VapourSynth core providing std.CopyFrameProps.
        target: Clip that should inherit metadata.
        source: Clip supplying the metadata.
        context: Debug label describing the copy site.
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


def _expand_limited_rgb(core: Any, clip: Any) -> Any:
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


def _restore_color_props(
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

    matrix, transfer, primaries, color_range = vs_core._resolve_color_metadata(props)
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

class GeometryPlan(TypedDict):
    """
    Resolved crop/pad/scale plan for rendering a screenshot.

    Attributes:
        width (int): Source clip width.
        height (int): Source clip height.
        crop (tuple[int, int, int, int]): Cropping values for left, top, right, bottom.
        cropped_w (int): Width after cropping.
        cropped_h (int): Height after cropping.
        scaled (tuple[int, int]): Dimensions after scaling.
        pad (tuple[int, int, int, int]): Padding applied around the scaled frame.
        final (tuple[int, int]): Final output dimensions.
        requires_full_chroma (bool): Whether geometry requires a 4:4:4 pivot.
        promotion_axes (str): Subsampling-aware axis label describing which geometry axis
            triggered promotion, or ``"none"`` when no promotion is required.
    """
    width: int
    height: int
    crop: tuple[int, int, int, int]
    cropped_w: int
    cropped_h: int
    scaled: tuple[int, int]
    pad: tuple[int, int, int, int]
    final: tuple[int, int]
    requires_full_chroma: bool
    promotion_axes: str


def _normalise_geometry_policy(value: OddGeometryPolicy | str) -> OddGeometryPolicy:
    return _geo.normalise_geometry_policy(value)


def _get_subsampling(fmt: Any, attr: str) -> int:
    return _geo.get_subsampling(fmt, attr)


def _axis_has_odd(values: Sequence[int]) -> bool:
    return _geo.axis_has_odd(values)


def _describe_plan_axes(plan: GeometryPlan | None) -> str:
    """Return a concise axis label for plans that include odd-pixel geometry."""

    return _geo.describe_plan_axes(plan)


def _safe_pivot_notify(pivot_notifier: Callable[[str], None] | None, note: str) -> None:
    """Invoke *pivot_notifier* without letting exceptions escape."""

    if pivot_notifier is None:
        return
    try:
        pivot_notifier(note)
    except (TypeError, ValueError, RuntimeError):
        logger.debug("pivot_notifier failed", exc_info=True)


def _resolve_source_props(
    clip: Any,
    source_props: Mapping[str, Any] | None,
    *,
    color_cfg: "ColorConfig | None" = None,
    file_name: str | None = None,
    warning_sink: Optional[List[str]] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Return a clip and colour metadata ensuring defaults/overrides are applied."""

    props = dict(source_props or {})
    if props.get("_ColorRange") is not None:
        return clip, props

    normalised_clip, resolved_props, _ = vs_core.normalise_color_metadata(
        clip,
        props if props else None,
        color_cfg=color_cfg,
        file_name=file_name,
        warning_sink=warning_sink,
    )
    return normalised_clip, dict(resolved_props)


def _describe_vs_format(fmt: Any) -> str:
    name = getattr(fmt, "name", None)
    if isinstance(name, str) and name:
        return name
    identifier = getattr(fmt, "id", None)
    if isinstance(identifier, int):
        return f"id={identifier}"
    return repr(fmt)


def _resolve_promotion_axes(
    fmt: Any,
    crop: tuple[int, int, int, int],
    pad: tuple[int, int, int, int],
) -> tuple[bool, str]:
    subsampling_w = _get_subsampling(fmt, "subsampling_w")
    subsampling_h = _get_subsampling(fmt, "subsampling_h")

    axes: List[str] = []
    if subsampling_h > 0 and _axis_has_odd((crop[1], crop[3], pad[1], pad[3])):
        axes.append("vertical")
    if subsampling_w > 0 and _axis_has_odd((crop[0], crop[2], pad[0], pad[2])):
        axes.append("horizontal")

    if not axes:
        return (False, "none")
    return (True, "+".join(axes))


def _is_sdr_pipeline(
    tonemap_info: "vs_core.TonemapInfo | None",
    source_props: Mapping[str, Any],
) -> bool:
    if tonemap_info is not None and tonemap_info.applied:
        return False
    try:
        is_hdr = vs_core._props_signal_hdr(source_props)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.debug("HDR detection failed; defaulting to SDR: %s", exc)
        is_hdr = False
    return not bool(is_hdr)


def _resolve_output_color_range(
    source_props: Mapping[str, Any],
    tonemap_info: "vs_core.TonemapInfo | None",
) -> int | None:
    tonemap_applied = bool(tonemap_info and tonemap_info.applied)
    range_full, range_limited = _range_constants()
    if tonemap_info and tonemap_info.output_color_range in (range_full, range_limited):
        if tonemap_applied:
            return int(tonemap_info.output_color_range)
        return int(tonemap_info.output_color_range)
    matrix, transfer, primaries, color_range = vs_core._resolve_color_metadata(source_props)
    if color_range is None:
        try:
            is_hdr = vs_core._props_signal_hdr(source_props)
        except (TypeError, ValueError, AttributeError) as exc:
            logger.debug("HDR detection failed; defaulting to SDR: %s", exc)
            is_hdr = False
        return range_full if is_hdr else range_limited
    resolved = int(color_range)
    if resolved in (range_full, range_limited):
        return resolved
    return range_full


def _promote_to_yuv444p16(
    core: Any,
    clip: Any,
    *,
    frame_idx: int,
    source_props: Mapping[str, Any],
) -> Any:
    try:
        import vapoursynth as vs  # type: ignore
    except ImportError as exc:
        raise ScreenshotWriterError("VapourSynth is required for screenshot export") from exc

    resize_ns = getattr(core, "resize", None)
    if resize_ns is None:
        raise ScreenshotWriterError("VapourSynth core is missing resize namespace")
    point = getattr(resize_ns, "Point", None)
    if not callable(point):
        raise ScreenshotWriterError("VapourSynth resize.Point is unavailable")

    resize_kwargs = _resolve_resize_color_kwargs(source_props)

    fmt = getattr(clip, "format", None)
    yuv_constant = getattr(vs, "YUV", object())
    if getattr(fmt, "color_family", None) == yuv_constant:
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
                "Colour metadata missing for frame %s during 4:4:4 promotion; applying Rec.709 limited defaults",
                frame_idx,
            )

    try:
        promoted = cast(
            Any,
            point(
                clip,
                format=vs.YUV444P16,
                dither_type="none",
                **resize_kwargs,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        raise ScreenshotWriterError(f"Failed to promote frame {frame_idx} to YUV444P16: {exc}") from exc

    std_ns = getattr(core, "std", None)
    set_props = getattr(std_ns, "SetFrameProps", None) if std_ns is not None else None
    if callable(set_props):
        prop_kwargs: Dict[str, int] = {}
        matrix_in = resize_kwargs.get("matrix_in")
        transfer_in = resize_kwargs.get("transfer_in")
        primaries_in = resize_kwargs.get("primaries_in")
        range_in = resize_kwargs.get("range_in")
        if matrix_in is not None:
            prop_kwargs["_Matrix"] = int(matrix_in)
        if transfer_in is not None:
            prop_kwargs["_Transfer"] = int(transfer_in)
        if primaries_in is not None:
            prop_kwargs["_Primaries"] = int(primaries_in)
        if range_in is not None:
            prop_kwargs["_ColorRange"] = int(range_in)
        if prop_kwargs:
            try:
                promoted = cast(Any, set_props(promoted, **prop_kwargs))
            except (RuntimeError, ValueError) as exc:
                logger.debug("Failed to set frame props after promotion: %s", exc)

    promoted = _copy_frame_props(core, promoted, clip, context="4:4:4 promotion")

    return promoted


def _rebalance_axis_even(first: int, second: int) -> tuple[int, int]:
    left = max(0, int(first))
    right = max(0, int(second))
    removed = 0

    if left % 2 != 0:
        left -= 1
        removed += 1
    if right % 2 != 0:
        right -= 1
        removed += 1

    while removed >= 2:
        right += 2
        removed -= 2

    return left, right


def _compute_requires_full_chroma(
    fmt: Any,
    crop: tuple[int, int, int, int],
    pad: tuple[int, int, int, int],
    policy: OddGeometryPolicy,
) -> bool:
    return _geo.compute_requires_full_chroma(fmt, crop, pad, policy)


def plan_mod_crop(
    width: int,
    height: int,
    mod: int,
    letterbox_pillarbox_aware: bool,
) -> Tuple[int, int, int, int]:
    """Plan left/top/right/bottom croppings so dimensions align to *mod*."""

    return _geo.plan_mod_crop(width, height, mod, letterbox_pillarbox_aware)


def _align_letterbox_pillarbox(plans: List[GeometryPlan]) -> None:
    _geo.align_letterbox_pillarbox(plans)


def _plan_letterbox_offsets(
    plans: Sequence[GeometryPlan],
    *,
    mod: int,
    tolerance: float = _geo.LETTERBOX_RATIO_TOLERANCE,
    max_target_height: int | None = None,
) -> List[tuple[int, int]]:
    return _geo.plan_letterbox_offsets(
        plans,
        mod=mod,
        tolerance=tolerance,
        max_target_height=max_target_height,
    )


def _resolve_auto_letterbox_mode(value: object) -> str:
    """Return the canonical auto-letterbox mode label."""

    if isinstance(value, AutoLetterboxCropMode):
        return value.value
    if isinstance(value, bool):
        return AutoLetterboxCropMode.STRICT.value if value else AutoLetterboxCropMode.OFF.value
    raw = "" if value is None else str(value).strip().lower()
    if raw in {"", "off", "false"}:
        return AutoLetterboxCropMode.OFF.value
    if raw == AutoLetterboxCropMode.BASIC.value:
        return AutoLetterboxCropMode.BASIC.value
    if raw in {AutoLetterboxCropMode.STRICT.value, "true"}:
        return AutoLetterboxCropMode.STRICT.value
    return AutoLetterboxCropMode.OFF.value


def _apply_letterbox_crop_strict(plans: list[GeometryPlan], cfg: ScreenshotConfig) -> None:
    """Apply the legacy auto letterbox heuristic to the supplied plans."""

    # Clamp auto-letterbox targets to the shortest active height across clips.
    # This keeps all clips from retaining more vertical content than the smallest
    # member of the set and matches the legacy behaviour before multi-mode support.
    try:
        max_target_height = min(int(plan["cropped_h"]) for plan in plans)
    except ValueError:
        max_target_height = None
    offsets = _plan_letterbox_offsets(
        plans,
        mod=cfg.mod_crop,
        max_target_height=max_target_height,
    )
    for plan, (extra_top, extra_bottom) in zip(plans, offsets, strict=True):
        if not (extra_top or extra_bottom):
            continue
        left, top, right, bottom = plan["crop"]
        top += int(extra_top)
        bottom += int(extra_bottom)
        new_height = int(plan["height"]) - top - bottom
        if new_height <= 0:
            raise ScreenshotGeometryError("Letterbox detection removed all pixels")
        plan["crop"] = (left, top, right, bottom)
        plan["cropped_h"] = new_height
        logger.info(
            "[LETTERBOX] Cropping %s px top / %s px bottom for width=%s height=%s",
            extra_top,
            extra_bottom,
            plan["width"],
            plan["height"],
        )


def _apply_letterbox_crop_basic(plans: list[GeometryPlan], cfg: ScreenshotConfig) -> None:
    """Apply the conservative cropped-geometry heuristic."""

    synthetic: list[GeometryPlan] = []
    for plan in plans:
        cropped_w = int(plan["cropped_w"])
        cropped_h = int(plan["cropped_h"])
        synthetic_plan = cast(GeometryPlan, dict(plan))
        synthetic_plan["width"] = cropped_w
        synthetic_plan["height"] = cropped_h
        synthetic_plan["crop"] = (0, 0, 0, 0)
        synthetic_plan["cropped_w"] = cropped_w
        synthetic_plan["cropped_h"] = cropped_h
        synthetic_plan["scaled"] = (cropped_w, cropped_h)
        synthetic_plan["pad"] = (0, 0, 0, 0)
        synthetic_plan["final"] = (cropped_w, cropped_h)
        synthetic_plan["requires_full_chroma"] = False
        synthetic_plan["promotion_axes"] = "none"
        synthetic.append(synthetic_plan)

    if cfg.letterbox_pillarbox_aware:
        _align_letterbox_pillarbox(synthetic)

    for plan in synthetic:
        plan["width"] = int(plan["cropped_w"])
        plan["height"] = int(plan["cropped_h"])

    # BASIC mode also clamps to the shortest effective (cropped) height so we
    # don't over-normalise taller clips beyond what the smallest one can support.
    try:
        max_target_height = min(int(entry["cropped_h"]) for entry in synthetic)
    except ValueError:
        max_target_height = None

    offsets = _plan_letterbox_offsets(
        synthetic,
        mod=cfg.mod_crop,
        max_target_height=max_target_height,
    )

    if not offsets:
        return
    if not any(extra_top or extra_bottom for extra_top, extra_bottom in offsets):
        return
    if all(extra_top or extra_bottom for extra_top, extra_bottom in offsets):
        return

    for plan, (extra_top, extra_bottom) in zip(plans, offsets, strict=True):
        if not (extra_top or extra_bottom):
            continue
        left, top, right, bottom = plan["crop"]
        top += int(extra_top)
        bottom += int(extra_bottom)
        new_height = int(plan["height"]) - top - bottom
        if new_height <= 0:
            raise ScreenshotGeometryError("Letterbox detection removed all pixels")
        plan["crop"] = (left, top, right, bottom)
        plan["cropped_h"] = new_height
        logger.info(
            "[LETTERBOX] (basic) Cropping %s px top / %s px bottom for width=%s height=%s",
            extra_top,
            extra_bottom,
            plan["width"],
            plan["height"],
        )


def _split_padding(total: int, center: bool) -> tuple[int, int]:
    return _geo.split_padding(total, center)


def _align_padding_mod(
    width: int,
    height: int,
    pad_left: int,
    pad_top: int,
    pad_right: int,
    pad_bottom: int,
    mod: int,
    center: bool,
) -> tuple[int, int, int, int]:
    return _geo.align_padding_mod(
        width,
        height,
        pad_left,
        pad_top,
        pad_right,
        pad_bottom,
        mod,
        center,
    )


def _compute_scaled_dimensions(
    width: int,
    height: int,
    crop: Tuple[int, int, int, int],
    target_height: int,
) -> Tuple[int, int]:
    return _geo.compute_scaled_dimensions(width, height, crop, target_height)


def _plan_geometry(clips: Sequence[Any], cfg: ScreenshotConfig) -> List[GeometryPlan]:
    policy = _normalise_geometry_policy(cfg.odd_geometry_policy)
    clip_formats: List[Any] = []
    plans: List[GeometryPlan] = []
    for clip in clips:
        width = getattr(clip, "width", None)
        height = getattr(clip, "height", None)
        if not isinstance(width, int) or not isinstance(height, int):
            raise ScreenshotGeometryError("Clip missing width/height metadata")

        clip_formats.append(getattr(clip, "format", None))
        crop = plan_mod_crop(width, height, cfg.mod_crop, cfg.letterbox_pillarbox_aware)
        cropped_w = width - crop[0] - crop[2]
        cropped_h = height - crop[1] - crop[3]
        if cropped_w <= 0 or cropped_h <= 0:
            raise ScreenshotGeometryError("Invalid crop results")

        plans.append(
            GeometryPlan(
                width=width,
                height=height,
                crop=crop,
                cropped_w=cropped_w,
                cropped_h=cropped_h,
                scaled=(cropped_w, cropped_h),
                pad=(0, 0, 0, 0),
                final=(cropped_w, cropped_h),
                requires_full_chroma=False,
                promotion_axes="none",
            )
        )

    mode = _resolve_auto_letterbox_mode(getattr(cfg, "auto_letterbox_crop", "off"))
    if mode == AutoLetterboxCropMode.STRICT.value:
        _apply_letterbox_crop_strict(plans, cfg)
    elif mode == AutoLetterboxCropMode.BASIC.value:
        _apply_letterbox_crop_basic(plans, cfg)

    if cfg.letterbox_pillarbox_aware:
        _align_letterbox_pillarbox(plans)

    if policy is OddGeometryPolicy.SUBSAMP_SAFE:
        for plan, fmt in zip(plans, clip_formats, strict=True):
            subsampling_w = _get_subsampling(fmt, "subsampling_w")
            subsampling_h = _get_subsampling(fmt, "subsampling_h")
            left, top, right, bottom = plan["crop"]

            if subsampling_h > 0:
                new_top, new_bottom = _rebalance_axis_even(top, bottom)
            else:
                new_top, new_bottom = top, bottom

            if subsampling_w > 0:
                new_left, new_right = _rebalance_axis_even(left, right)
            else:
                new_left, new_right = left, right

            changed_vertical = (new_top, new_bottom) != (top, bottom)
            changed_horizontal = (new_left, new_right) != (left, right)

            if changed_vertical:
                logger.warning(
                    "[GEOMETRY] Rebalanced vertical crop from %s/%s to %s/%s for mod-2 safety; content may shift by 1px",
                    top,
                    bottom,
                    new_top,
                    new_bottom,
                )
            if changed_horizontal:
                logger.warning(
                    "[GEOMETRY] Rebalanced horizontal crop from %s/%s to %s/%s for mod-2 safety; content may shift by 1px",
                    left,
                    right,
                    new_left,
                    new_right,
                )

            if changed_vertical or changed_horizontal:
                plan["crop"] = (new_left, new_top, new_right, new_bottom)
                new_cropped_w = int(plan["width"]) - new_left - new_right
                new_cropped_h = int(plan["height"]) - new_top - new_bottom
                if new_cropped_w <= 0 or new_cropped_h <= 0:
                    raise ScreenshotGeometryError("Rebalanced crop removed all pixels")
                plan["cropped_w"] = new_cropped_w
                plan["cropped_h"] = new_cropped_h
                plan["scaled"] = (new_cropped_w, new_cropped_h)

    single_res_target = int(cfg.single_res) if cfg.single_res > 0 else None
    if single_res_target is not None:
        desired_height = max(1, single_res_target)
        global_target = None
    else:
        desired_height = None
        global_target = (
            max((plan["cropped_h"] for plan in plans), default=None)
            if cfg.upscale
            else None
        )

    max_source_width = max((int(plan["width"]) for plan in plans), default=0)

    pad_mode = str(getattr(cfg, "pad_to_canvas", "off")).strip().lower()
    pad_enabled = pad_mode in {"on", "auto"}
    pad_force = pad_mode == "on"
    pad_tolerance = max(0, int(getattr(cfg, "letterbox_px_tolerance", 0)))

    target_heights: List[int] = []

    for plan in plans:
        cropped_w = int(plan["cropped_w"])
        cropped_h = int(plan["cropped_h"])
        if desired_height is not None:
            target_h = desired_height
            if not cfg.upscale and target_h > cropped_h:
                target_h = cropped_h
        elif global_target is not None:
            target_h = max(cropped_h, int(global_target))
        else:
            target_h = cropped_h

        target_heights.append(target_h)

        pad_left = pad_top = pad_right = pad_bottom = 0
        scaled_w = cropped_w
        scaled_h = cropped_h

        if target_h != cropped_h:
            if target_h > cropped_h and cfg.upscale:
                scaled_w, scaled_h = _compute_scaled_dimensions(
                    int(plan["width"]),
                    int(plan["height"]),
                    plan["crop"],
                    target_h,
                )
            elif target_h < cropped_h:
                scaled_w, scaled_h = _compute_scaled_dimensions(
                    int(plan["width"]),
                    int(plan["height"]),
                    plan["crop"],
                    target_h,
                )
            elif pad_enabled and target_h > cropped_h:
                diff = target_h - cropped_h
                if pad_force or diff <= pad_tolerance:
                    add_top, add_bottom = _split_padding(diff, True)
                    pad_top += add_top
                    pad_bottom += add_bottom

        if cfg.upscale and max_source_width > 0 and scaled_w > max_source_width:
            base_w = int(plan["cropped_w"])
            if base_w > 0:
                scale = max_source_width / float(base_w)
                adjusted_h = int(round(int(plan["cropped_h"]) * scale))
                scaled_w = max_source_width
                scaled_h = max(1, adjusted_h)

        plan["scaled"] = (scaled_w, scaled_h)
        plan["pad"] = (pad_left, pad_top, pad_right, pad_bottom)

    canvas_height = None
    if desired_height is not None:
        canvas_height = desired_height
    elif global_target is not None:
        try:
            canvas_height = max(int(value) for value in target_heights)
        except ValueError:
            canvas_height = None

    canvas_width = None
    if pad_enabled:
        if single_res_target is not None and max_source_width > 0:
            canvas_width = max_source_width
        else:
            try:
                canvas_width = max(int(plan["scaled"][0]) for plan in plans)
            except ValueError:
                canvas_width = None

    for plan in plans:
        scaled_w, scaled_h = plan["scaled"]
        pad_left, pad_top, pad_right, pad_bottom = plan["pad"]

        if canvas_height is not None and pad_enabled:
            target_h = canvas_height
            current_h = scaled_h + pad_top + pad_bottom
            diff_h = target_h - current_h
            if diff_h > 0:
                if pad_force or diff_h <= pad_tolerance:
                    add_top, add_bottom = _split_padding(diff_h, True)
                    pad_top += add_top
                    pad_bottom += add_bottom
                else:
                    logger.debug(
                        "Skipping vertical padding (%s px) for width=%s due to tolerance",
                        diff_h,
                        plan["width"],
                    )

        if canvas_width is not None and pad_enabled:
            current_w = scaled_w + pad_left + pad_right
            diff_w = canvas_width - current_w
            if diff_w > 0:
                if pad_force or diff_w <= pad_tolerance:
                    add_left, add_right = _split_padding(diff_w, True)
                    pad_left += add_left
                    pad_right += add_right
                else:
                    logger.debug(
                        "Skipping horizontal padding (%s px) for width=%s due to tolerance",
                        diff_w,
                        plan["width"],
                    )

        pad_left, pad_top, pad_right, pad_bottom = _align_padding_mod(
            scaled_w,
            scaled_h,
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
            cfg.mod_crop,
            True,
        )

        plan["pad"] = (pad_left, pad_top, pad_right, pad_bottom)
        plan["final"] = (
            scaled_w + pad_left + pad_right,
            scaled_h + pad_top + pad_bottom,
        )

    for plan, fmt in zip(plans, clip_formats, strict=True):
        if policy is OddGeometryPolicy.SUBSAMP_SAFE:
            subsampling_w = _get_subsampling(fmt, "subsampling_w")
            subsampling_h = _get_subsampling(fmt, "subsampling_h")
            pad_left, pad_top, pad_right, pad_bottom = plan["pad"]
            scaled_w, scaled_h = plan["scaled"]

            new_pad_top, new_pad_bottom = (pad_top, pad_bottom)
            new_pad_left, new_pad_right = (pad_left, pad_right)

            if subsampling_h > 0:
                new_pad_top, new_pad_bottom = _rebalance_axis_even(pad_top, pad_bottom)
                if (new_pad_top, new_pad_bottom) != (pad_top, pad_bottom):
                    logger.warning(
                        "[GEOMETRY] Rebalanced vertical padding from %s/%s to %s/%s for mod-2 safety; content may shift by 1px",
                        pad_top,
                        pad_bottom,
                        new_pad_top,
                        new_pad_bottom,
                    )
            if subsampling_w > 0:
                new_pad_left, new_pad_right = _rebalance_axis_even(pad_left, pad_right)
                if (new_pad_left, new_pad_right) != (pad_left, pad_right):
                    logger.warning(
                        "[GEOMETRY] Rebalanced horizontal padding from %s/%s to %s/%s for mod-2 safety; content may shift by 1px",
                        pad_left,
                        pad_right,
                        new_pad_left,
                        new_pad_right,
                    )

            if (
                (new_pad_top, new_pad_bottom) != (pad_top, pad_bottom)
                or (new_pad_left, new_pad_right) != (pad_left, pad_right)
            ):
                plan["pad"] = (new_pad_left, new_pad_top, new_pad_right, new_pad_bottom)
                plan["final"] = (
                    scaled_w + new_pad_left + new_pad_right,
                    scaled_h + new_pad_top + new_pad_bottom,
                )

                aligned_pad_left, aligned_pad_top, aligned_pad_right, aligned_pad_bottom = _align_padding_mod(
                    scaled_w,
                    scaled_h,
                    new_pad_left,
                    new_pad_top,
                    new_pad_right,
                    new_pad_bottom,
                    cfg.mod_crop,
                    True,
                )

                if (
                    aligned_pad_left,
                    aligned_pad_top,
                    aligned_pad_right,
                    aligned_pad_bottom,
                ) != plan["pad"]:
                    plan["pad"] = (
                        aligned_pad_left,
                        aligned_pad_top,
                        aligned_pad_right,
                        aligned_pad_bottom,
                    )
                    plan["final"] = (
                        scaled_w + aligned_pad_left + aligned_pad_right,
                        scaled_h + aligned_pad_top + aligned_pad_bottom,
                    )

        plan["requires_full_chroma"] = _compute_requires_full_chroma(
            fmt,
            plan["crop"],
            plan["pad"],
            policy,
        )

        if plan["requires_full_chroma"]:
            needs_promotion, promotion_axes = _resolve_promotion_axes(
                fmt,
                plan["crop"],
                plan["pad"],
            )
            plan["promotion_axes"] = promotion_axes if needs_promotion else "none"
        else:
            plan["promotion_axes"] = "none"

    _maybe_log_geometry_debug(plans, cfg)
    return plans


def _maybe_log_geometry_debug(plans: Sequence[GeometryPlan], cfg: ScreenshotConfig) -> None:
    """Emit per-plan geometry diagnostics when FRAME_COMPARE_DEBUG_GEOMETRY is set."""

    if not os.environ.get("FRAME_COMPARE_DEBUG_GEOMETRY"):
        return

    try:
        pad_mode = str(getattr(cfg, "pad_to_canvas", "off")).strip().lower()
    except (AttributeError, ValueError):
        pad_mode = "?"

    logger.info(
        "[GEOMETRY DEBUG] clips=%s upscale=%s single_res=%s pad=%s "
        "auto_letterbox=%s pillarbox_aware=%s mod_crop=%s",
        len(plans),
        bool(getattr(cfg, "upscale", False)),
        getattr(cfg, "single_res", 0),
        pad_mode,
        _resolve_auto_letterbox_mode(getattr(cfg, "auto_letterbox_crop", "off")),
        bool(getattr(cfg, "letterbox_pillarbox_aware", False)),
        getattr(cfg, "mod_crop", 0),
    )

    for idx, plan in enumerate(plans):
        try:
            width = int(plan["width"])
            height = int(plan["height"])
            crop = tuple(plan["crop"])
            cropped_w = int(plan["cropped_w"])
            cropped_h = int(plan["cropped_h"])
            scaled_w, scaled_h = plan["scaled"]
            pad = tuple(plan["pad"])
            final_w, final_h = plan["final"]
        except (ValueError, TypeError, KeyError) as exc:
            logger.info(
                "[GEOMETRY DEBUG] idx=%s error=%s plan=%r",
                idx,
                exc,
                plan,
            )
            continue

        logger.info(
            "[GEOMETRY DEBUG] idx=%s src=%dx%d crop=%s cropped=%dx%d scaled=%dx%d pad=%s final=%dx%d",
            idx,
            width,
            height,
            crop,
            cropped_w,
            cropped_h,
            int(scaled_w),
            int(scaled_h),
            pad,
            int(final_w),
            int(final_h),
        )


def _sanitise_label(label: str) -> str:
    return _naming.sanitise_label(label)


def _derive_labels(source: str, metadata: Mapping[str, str]) -> tuple[str, str]:
    return _naming.derive_labels(source, metadata)


def _prepare_filename(frame: int, label: str) -> str:
    return _naming.prepare_filename(frame, label)


def _normalise_compression_level(level: int) -> int:
    return _enc.normalise_compression_level(level)


def _map_fpng_compression(level: int) -> int:
    return _enc.map_fpng_compression(level)


def _map_png_compression_level(level: int) -> int:
    """Translate the user configured level into a PNG compress level."""

    return _enc.map_png_compression_level(level)


def _save_frame_with_fpng(
    clip: Any,
    frame_idx: int,
    crop: Tuple[int, int, int, int],
    scaled: Tuple[int, int],
    pad: Tuple[int, int, int, int],
    path: Path,
    cfg: ScreenshotConfig,
    label: str,
    requested_frame: int,
    selection_label: str | None = None,
    *,
    overlay_text: Optional[str] = None,
    overlay_state: Optional[OverlayState] = None,
    strict_overlay: bool = False,
    source_props: Mapping[str, Any] | None = None,
    geometry_plan: GeometryPlan | None = None,
    tonemap_info: "vs_core.TonemapInfo | None" = None,
    pivot_notifier: Callable[[str], None] | None = None,
    color_cfg: "ColorConfig | None" = None,
    file_name: str | None = None,
    warning_sink: Optional[List[str]] = None,
    debug_state: Optional[_ColorDebugState] = None,
    frame_info_allowed: bool = True,
    overlays_allowed: bool = True,
    expand_to_full: bool = False,
) -> None:
    try:
        import vapoursynth as vs  # type: ignore
    except ImportError as exc:
        raise ScreenshotWriterError("VapourSynth is required for screenshot export") from exc

    if not isinstance(clip, vs.VideoNode):
        raise ScreenshotWriterError("Expected a VapourSynth clip for rendering")

    resolved_policy = _normalise_geometry_policy(cfg.odd_geometry_policy)
    rgb_dither = _normalize_rgb_dither(cfg.rgb_dither)
    expand_to_full = bool(expand_to_full) or _should_expand_to_full(getattr(cfg, "export_range", None))
    clip, source_props_map = _resolve_source_props(
        clip,
        source_props,
        color_cfg=color_cfg,
        file_name=file_name,
        warning_sink=warning_sink,
    )
    tonemap_applied = bool(tonemap_info and tonemap_info.applied)
    if tonemap_applied:
        try:
            tonemapped_props = vs_core._snapshot_frame_props(clip)
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "Falling back to source props for tonemapped clip %s: %s",
                file_name or "<unknown>",
                exc,
            )
        else:
            source_props_map = dict(tonemapped_props)
    requires_full_chroma = bool(geometry_plan and geometry_plan.get("requires_full_chroma"))
    fmt = getattr(clip, "format", None)
    has_axis, axis_label = _resolve_promotion_axes(fmt, crop, pad)
    yuv_constant = getattr(vs, "YUV", object())
    color_family = getattr(fmt, "color_family", None)
    is_sdr = _is_sdr_pipeline(tonemap_info, source_props_map)
    output_color_range = _resolve_output_color_range(source_props_map, tonemap_info)
    range_full, range_limited = _range_constants()
    if expand_to_full:
        output_color_range = range_full
    include_color_range = bool(
        (tonemap_info and tonemap_info.output_color_range is not None) or not tonemap_applied
    )
    should_promote = (
        requires_full_chroma
        and has_axis
        and is_sdr
        and color_family == yuv_constant
    )
    format_label = _describe_vs_format(fmt)

    if should_promote:
        logger.info(
            "Odd-geometry on subsampled SDR \u2192 promoting to YUV444P16 (policy=%s, axis=%s, fmt=%s)",
            resolved_policy.value,
            axis_label,
            format_label,
        )
        logger.debug(
            "Promotion details frame=%s src_format=%s dst_format=YUV444P16 dither=%s",
            frame_idx,
            format_label,
            rgb_dither.value,
        )
        if pivot_notifier is not None:
            note = (
                "Full-chroma pivot active (axis={axis}, policy={policy}, backend=fpng, fmt={fmt})"
            ).format(axis=axis_label, policy=resolved_policy.value, fmt=format_label)
            _safe_pivot_notify(pivot_notifier, note)

    core = getattr(clip, "core", None) or getattr(vs, "core", None)
    fpng_ns = getattr(core, "fpng", None) if core is not None else None
    writer = getattr(fpng_ns, "Write", None) if fpng_ns is not None else None
    if not callable(writer):
        raise ScreenshotWriterError("VapourSynth fpng.Write plugin is unavailable")

    work = clip
    if should_promote:
        work = _promote_to_yuv444p16(
            core,
            work,
            frame_idx=frame_idx,
            source_props=source_props_map,
        )
    try:
        left, top, right, bottom = crop
        if any(crop):
            pre_crop = work
            work = work.std.CropRel(left=left, right=right, top=top, bottom=bottom)
            work = _copy_frame_props(core, work, pre_crop, context="geometry cropping")
            work = _restore_color_props(
                core,
                work,
                source_props_map,
                context="geometry cropping",
                include_color_range=include_color_range,
            )
        target_w, target_h = scaled
        if work.width != target_w or work.height != target_h:
            resize_ns = getattr(core, "resize", None)
            if resize_ns is None:
                raise ScreenshotWriterError("VapourSynth core is missing resize namespace")
            resampler = getattr(resize_ns, "Spline36", None)
            if not callable(resampler):
                raise ScreenshotWriterError("VapourSynth resize.Spline36 is unavailable")
            pre_resize = work
            work = resampler(work, width=target_w, height=target_h)
            work = _copy_frame_props(core, work, pre_resize, context="geometry resize")
            work = _restore_color_props(
                core,
                work,
                source_props_map,
                context="geometry resize",
                include_color_range=include_color_range,
            )

        pad_left, pad_top, pad_right, pad_bottom = pad
        if pad_left or pad_top or pad_right or pad_bottom:
            std_ns = getattr(core, "std", None)
            add_borders = getattr(std_ns, "AddBorders", None) if std_ns is not None else None
            if not callable(add_borders):
                raise ScreenshotWriterError("VapourSynth std.AddBorders is unavailable")
            pre_pad = work
            work = add_borders(
                work,
                left=max(0, pad_left),
                right=max(0, pad_right),
                top=max(0, pad_top),
                bottom=max(0, pad_bottom),
            )
            work = _copy_frame_props(core, work, pre_pad, context="geometry padding")
            work = _restore_color_props(
                core,
                work,
                source_props_map,
                context="geometry padding",
                include_color_range=include_color_range,
            )
    except (RuntimeError, ValueError) as exc:
        raise ScreenshotWriterError(f"Failed to prepare frame {frame_idx}: {exc}") from exc

    work = _restore_color_props(
        core,
        work,
        source_props_map,
        context="geometry final",
        include_color_range=include_color_range,
    )

    render_clip = work
    overlay_range: Optional[int] = (
        output_color_range
        if output_color_range in (range_full, range_limited)
        else range_full
    )

    if debug_state is not None:
        try:
            post_geom_props = vs_core._snapshot_frame_props(work)
        except (ValueError, TypeError, RuntimeError):
            post_geom_props = {}
        debug_state.capture_stage(
            "post_geometry",
            frame_idx,
            work,
            post_geom_props,
        )
        legacy_clip = _legacy_rgb24_from_clip(
            core,
            work,
            getattr(debug_state, "color_tuple", None),
            expand_to_full=expand_to_full,
        )
        if legacy_clip is not None:
            try:
                legacy_props = vs_core._snapshot_frame_props(legacy_clip)
            except (ValueError, TypeError, RuntimeError):
                legacy_props = {}
            debug_state.capture_stage("legacy_rgb24", frame_idx, legacy_clip, legacy_props)
    overlay_input_range = overlay_range
    clip_format = getattr(render_clip, "format", None)
    overlay_resize_kwargs: Dict[str, Any] = {}
    overlay_original_format = getattr(clip_format, "id", None)
    overlay_rgb_format = None
    try:
        import vapoursynth as vs  # type: ignore
    except ImportError:
        vs = None  # type: ignore[assignment]
    if vs is not None:
        overlay_rgb_format = getattr(vs, "RGB24", None)
    log_overlay = bool(os.getenv("FRAME_COMPARE_LOG_OVERLAY_RANGE"))
    if log_overlay:
        fmt_name = getattr(clip_format, "name", None)
        logger.info(
            "[OVERLAY DEBUG] clip=%s frame=%s stage=pre-overlay range=%s props_range=%s fmt=%s",
            _sanitize_for_log(label),
            frame_idx,
            overlay_input_range,
            source_props_map.get("_ColorRange"),
            _sanitize_for_log(fmt_name),
        )
    converted_for_overlay = False
    if frame_info_allowed or (overlays_allowed and overlay_text):
        resize_ns = getattr(core, "resize", None)
        point = getattr(resize_ns, "Point", None) if resize_ns is not None else None
        if callable(point):
            try:
                point_kwargs: Dict[str, Any] = {"dither_type": "none"}
                if overlay_rgb_format is not None:
                    point_kwargs["format"] = overlay_rgb_format
                if overlay_input_range in (range_full, range_limited):
                    point_kwargs["range"] = int(overlay_input_range)
                point_kwargs.update(overlay_resize_kwargs)
                render_clip = point(render_clip, **point_kwargs)
                converted_for_overlay = True
                render_clip = _set_clip_range(
                    core,
                    render_clip,
                    overlay_input_range,
                    context="overlay range normalisation",
                )
                if log_overlay:
                    fmt_name = getattr(getattr(render_clip, "format", None), "name", None)
                    logger.info(
                        "[OVERLAY DEBUG] clip=%s frame=%s stage=normalized range=%s fmt=%s",
                        _sanitize_for_log(label),
                        frame_idx,
                        overlay_input_range,
                        _sanitize_for_log(fmt_name),
                    )
            except (RuntimeError, ValueError) as exc:
                logger.debug("Failed to normalize overlay range for frame %s: %s", frame_idx, exc)
        else:
            logger.debug("VapourSynth resize.Point unavailable; skipping overlay range normalization")

    if frame_info_allowed:
        render_clip = _apply_frame_info_overlay(
            core,
            render_clip,
            label,
            requested_frame,
            selection_label,
        )

    overlay_state = overlay_state or _new_overlay_state()
    if overlays_allowed and overlay_text:
        render_clip = _apply_overlay_text(
            core,
            render_clip,
            overlay_text,
            strict=strict_overlay,
            state=overlay_state,
            file_label=label,
        )

    if (
        converted_for_overlay
        and overlay_input_range != output_color_range
        and output_color_range in (range_full, range_limited)
    ):
        resize_ns = getattr(core, "resize", None)
        point = getattr(resize_ns, "Point", None) if resize_ns is not None else None
        if callable(point):
            try:
                point_kwargs: Dict[str, Any] = {"dither_type": "none"}
                if isinstance(overlay_original_format, int):
                    point_kwargs["format"] = overlay_original_format
                if output_color_range in (range_full, range_limited):
                    point_kwargs["range"] = int(output_color_range)
                point_kwargs.update(overlay_resize_kwargs)
                render_clip = point(render_clip, **point_kwargs)
                render_clip = _set_clip_range(
                    core,
                    render_clip,
                    int(output_color_range),
                    context="overlay range restore",
                )
                if log_overlay:
                    fmt_name = getattr(getattr(render_clip, "format", None), "name", None)
                    logger.info(
                        "[OVERLAY DEBUG] clip=%s frame=%s stage=restored range=%s fmt=%s",
                        _sanitize_for_log(label),
                        frame_idx,
                        output_color_range,
                        _sanitize_for_log(fmt_name),
                    )
            except (RuntimeError, ValueError) as exc:
                logger.debug(
                    "Failed to restore target range after overlay for frame %s: %s",
                    frame_idx,
                    exc,
                )
        else:
            logger.debug("VapourSynth resize.Point unavailable; skipping overlay range restore")

    render_clip = _ensure_rgb24(
        core,
        render_clip,
        frame_idx,
        source_props=source_props_map,
        rgb_dither=rgb_dither,
        target_range=output_color_range,
        expand_to_full=expand_to_full,
    )
    if debug_state is not None:
        try:
            rgb_props = vs_core._snapshot_frame_props(render_clip)
        except (ValueError, TypeError, RuntimeError):
            rgb_props = {}
        debug_state.capture_stage("post_rgb24", frame_idx, render_clip, rgb_props)
    logger.debug(
        "RGB24 conversion for frame %s used dither=%s (policy=%s)",
        frame_idx,
        rgb_dither.value,
        resolved_policy.value,
    )

    compression = _map_fpng_compression(cfg.compression_level)
    try:
        job: Any = writer(render_clip, str(path), compression=compression, overwrite=True)
        job.get_frame(frame_idx)
    except (RuntimeError, ValueError) as exc:
        raise ScreenshotWriterError(f"fpng failed for frame {frame_idx}: {exc}") from exc


def _map_ffmpeg_compression(level: int) -> int:
    """Map config compression level to ffmpeg's PNG compression scale."""

    return _enc.map_ffmpeg_compression(level)


def _escape_drawtext(text: str) -> str:
    return _enc.escape_drawtext(text)


def _resolve_source_frame_index(frame_idx: int, trim_start: int) -> int | None:
    if trim_start == 0:
        return frame_idx
    if trim_start > 0:
        return frame_idx + trim_start
    blank = abs(int(trim_start))
    if frame_idx < blank:
        return None
    return frame_idx - blank


def _save_frame_with_ffmpeg(
    source: str,
    frame_idx: int,
    crop: Tuple[int, int, int, int],
    scaled: Tuple[int, int],
    pad: Tuple[int, int, int, int],
    path: Path,
    cfg: ScreenshotConfig,
    width: int,
    height: int,
    selection_label: str | None,
    *,
    overlay_text: Optional[str] = None,
    geometry_plan: GeometryPlan | None = None,
    is_sdr: bool = True,
    pivot_notifier: Callable[[str], None] | None = None,
    frame_info_allowed: bool = True,
    overlays_allowed: bool = True,
    target_range: int | None = None,
    expand_to_full: bool = False,
    source_color_range: int | None = None,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise ScreenshotWriterError("FFmpeg executable not found in PATH")

    cropped_w = max(1, width - crop[0] - crop[2])
    cropped_h = max(1, height - crop[1] - crop[3])

    requires_full_chroma = bool(geometry_plan and geometry_plan.get("requires_full_chroma"))
    promotion_axes_value = (
        geometry_plan.get("promotion_axes", "") if geometry_plan is not None else ""
    )
    axis_label = str(promotion_axes_value).strip() if promotion_axes_value is not None else ""
    if not axis_label:
        axis_label = _describe_plan_axes(geometry_plan)
    filters = [f"select=eq(n\\,{int(frame_idx)})"]
    should_apply_full_chroma = requires_full_chroma and is_sdr
    if should_apply_full_chroma:
        filters.append("format=yuv444p16")
    if any(crop):
        filters.append(
            "crop={w}:{h}:{x}:{y}".format(
                w=max(1, cropped_w),
                h=max(1, cropped_h),
                x=max(0, crop[0]),
                y=max(0, crop[1]),
            )
        )
    if scaled != (cropped_w, cropped_h):
        filters.append(f"scale={max(1, scaled[0])}:{max(1, scaled[1])}:flags=lanczos")
    pad_left, pad_top, pad_right, pad_bottom = pad
    final_w = max(1, scaled[0] + pad_left + pad_right)
    final_h = max(1, scaled[1] + pad_top + pad_bottom)
    if pad_left or pad_top or pad_right or pad_bottom:
        filters.append(
            "pad={w}:{h}:{x}:{y}".format(
                w=final_w,
                h=final_h,
                x=max(0, pad_left),
                y=max(0, pad_top),
            )
        )
    if frame_info_allowed:
        text_lines = [f"Frame\\ {int(frame_idx)}"]
        if selection_label:
            text_lines.append(f"Content Type\\: {selection_label}")
        text = "\\\\n".join(text_lines)
        drawtext = (
            "drawtext=text={text}:fontcolor=white:borderw=2:bordercolor=black:"
            "box=0:shadowx=1:shadowy=1:shadowcolor=black:x=10:y=10"
        ).format(text=_escape_drawtext(text))
        filters.append(drawtext)
    if overlays_allowed and overlay_text:
        overlay_cmd = (
            "drawtext=text={text}:fontcolor=white:borderw=2:bordercolor=black:"
            "box=0:shadowx=1:shadowy=1:shadowcolor=black:x=10:y=80"
        ).format(text=_escape_drawtext(overlay_text))
        filters.append(overlay_cmd)

    range_full, range_limited = _range_constants()
    limited_source = source_color_range == range_limited
    if source_color_range is None and target_range in (range_full, range_limited):
        limited_source = int(target_range) == range_limited
    resolved_target_range = target_range
    if expand_to_full:
        resolved_target_range = range_full
        if limited_source:
            filters.append("scale=in_range=tv:out_range=pc")
    if should_apply_full_chroma:
        configured = _normalize_rgb_dither(cfg.rgb_dither)
        ffmpeg_dither = "ordered"
        if configured is RGBDither.NONE:
            ffmpeg_dither = "none"
        elif configured is RGBDither.ORDERED:
            ffmpeg_dither = "ordered"
        else:
            logger.debug(
                "FFmpeg RGB24 conversion forcing deterministic dither=ordered (configured=%s)",
                configured.value,
            )
        filters.append(f"format=rgb24:dither={ffmpeg_dither}")
        if pivot_notifier is not None:
            resolved_policy = _normalise_geometry_policy(cfg.odd_geometry_policy)
            note = (
                "Full-chroma pivot active (axis={axis}, policy={policy}, backend=ffmpeg)"
            ).format(axis=axis_label, policy=resolved_policy.value)
            _safe_pivot_notify(pivot_notifier, note)
    elif requires_full_chroma and not is_sdr:
        logger.debug(
            "Skipping full-chroma pivot for HDR content (axis=%s)",
            axis_label or "none",
        )

    filter_chain = ",".join(filters)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source,
        "-vf",
        filter_chain,
        "-frames:v",
        "1",
        "-vsync",
        "0",
        "-compression_level",
        str(_map_ffmpeg_compression(cfg.compression_level)),
    ]

    if resolved_target_range is not None:
        range_value = int(resolved_target_range)
        color_range_flag = "pc" if range_value == range_full else "tv"
        cmd.extend(["-color_range", color_range_flag])

    cmd.append(str(path))

    timeout_value = getattr(cfg, "ffmpeg_timeout_seconds", None)
    timeout_seconds_raw: float | None
    try:
        timeout_seconds_raw = float(timeout_value) if timeout_value is not None else None
    except (TypeError, ValueError):
        timeout_seconds_raw = None

    if timeout_seconds_raw is not None and timeout_seconds_raw <= 0:
        timeout_seconds: float | None = None
    else:
        timeout_seconds = timeout_seconds_raw

    try:
        process = _subproc.run_checked(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            text=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = timeout_seconds if timeout_seconds is not None else 0.0
        raise ScreenshotWriterError(
            f"FFmpeg timed out after {duration:.1f}s for frame {frame_idx}"
        ) from exc
    if process.returncode != 0:
        try:
            stderr = process.stderr.decode("utf-8", "ignore").strip()
        except (AttributeError, ValueError):
            stderr = ""
        message = stderr or "unknown error"
        raise ScreenshotWriterError(
            f"FFmpeg failed for frame {frame_idx}: {message}"
        )



def _save_frame_placeholder(path: Path) -> None:
    path.write_bytes(b"placeholder\n")


def generate_screenshots(
    clips: Sequence[Any],
    frames: Sequence[int],
    files: Sequence[str],
    metadata: Sequence[Mapping[str, str]],
    out_dir: Path,
    cfg: ScreenshotConfig,
    color_cfg: ColorConfig,
    *,
    trim_offsets: Sequence[int] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    frame_labels: Mapping[int, str] | None = None,
    selection_details: Mapping[int, Mapping[str, Any]] | None = None,
    alignment_maps: Sequence[Any] | None = None,
    warnings_sink: List[str] | None = None,
    verification_sink: List[Dict[str, Any]] | None = None,
    pivot_notifier: Callable[[str], None] | None = None,
    debug_color: bool = False,
    source_frame_props: Sequence[Mapping[str, Any] | None] | None = None,
) -> List[str]:
    """
    Render and save screenshots for the given frames from each input clip using the configured writers.

    Render each requested frame for every clip, applying geometry planning, optional overlays, alignment mapping, and the selected writer backend (fpng or ffmpeg). Created files are written into out_dir and their paths are returned in the order they were produced.

    Parameters:
        clips: Sequence of clip objects prepared for rendering.
        frames: Sequence of frame indices to render for each clip.
        files: Sequence of source file paths corresponding to clips; must match length of clips.
        metadata: Sequence of metadata mappings (one per file); must match length of files.
        out_dir: Destination directory for written screenshot files.
        cfg: ScreenshotConfig controlling writer selection, geometry and format options.
        color_cfg: ColorConfig controlling overlays, tonemapping and related color options.
        trim_offsets: Optional per-file trim start offsets; if None, treated as zeros. Must match length of files.
        source_frame_props: Optional sequence mirroring ``files`` containing cached source frame props for each clip.
        progress_callback: Optional callable invoked with 1 for each saved file to indicate progress.
        frame_labels: Optional mapping from frame index to a user-visible selection label used in overlays and filenames.
        alignment_maps: Optional sequence of alignment mappers (one per clip) used to map source frame indices.
        warnings_sink: Optional list to which non-fatal warning messages will be appended.
        verification_sink: Optional list to which per-clip verification records will be appended; each record contains keys: file, frame, average, maximum, auto_selected.
        pivot_notifier: Optional callable invoked with a short text note whenever a full-chroma pivot is applied.
        debug_color: Enable detailed colour debugging (logs, intermediate PNGs, legacy conversions) when True.

    Returns:
        List[str]: Ordered list of file paths for all created screenshot files.
    """

    if len(clips) != len(files):
        raise ScreenshotError("clips and files must have matching lengths")
    if len(metadata) != len(files):
        raise ScreenshotError("metadata and files must have matching lengths")
    if not frames:
        return []

    if trim_offsets is None:
        trim_offsets = [0] * len(files)
    if len(trim_offsets) != len(files):
        raise ScreenshotError("trim_offsets and files must have matching lengths")
    if source_frame_props is not None and len(source_frame_props) != len(files):
        raise ScreenshotError("source_frame_props and files must have matching lengths")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise ScreenshotError(
            "Unable to create screenshot directory "
            f"'{out_dir}': {exc.strerror or exc}"
        ) from exc
    except OSError as exc:
        raise ScreenshotError(
            f"Unable to prepare screenshot directory '{out_dir}': {exc}"
        ) from exc
    created: List[str] = []

    processed_results: List[vs_core.ClipProcessResult] = []
    overlay_states: List[OverlayState] = []
    debug_enabled = bool(debug_color or getattr(color_cfg, "debug_color", False))
    debug_root = out_dir / "debug" if debug_enabled else None
    debug_dither = _normalize_rgb_dither(cfg.rgb_dither)
    use_ffmpeg_runtime = bool(cfg.use_ffmpeg and not debug_enabled)
    frame_info_allowed_default = bool(cfg.add_frame_info and not debug_enabled)
    overlays_allowed_default = not debug_enabled

    for index, (clip, file_path) in enumerate(zip(clips, files, strict=True)):
        stored_props = None
        if source_frame_props is not None and index < len(source_frame_props):
            stored_props = source_frame_props[index]
        result = vs_core.process_clip_for_screenshot(
            clip,
            file_path,
            color_cfg,
            enable_overlay=True,
            enable_verification=True,
            logger_override=logger,
            warning_sink=warnings_sink,
            debug_color=debug_enabled,
            stored_source_props=stored_props,
        )
        processed_results.append(result)
        overlay_states.append(_new_overlay_state())
        if result.verification is not None:
            logger.info(
                "[VERIFY] %s frame=%d avg=%.4f max=%.4f",
                file_path,
                result.verification.frame,
                result.verification.average,
                result.verification.maximum,
            )
            if verification_sink is not None:
                verification_sink.append(
                    {
                        "file": str(file_path),
                        "frame": int(result.verification.frame),
                        "average": float(result.verification.average),
                        "maximum": float(result.verification.maximum),
                        "auto_selected": bool(result.verification.auto_selected),
                    }
                )

    geometry = _plan_geometry([result.clip for result in processed_results], cfg)

    for clip_index, (result, file_path, meta, plan, trim_start, source_clip) in enumerate(
        zip(processed_results, files, metadata, geometry, trim_offsets, clips, strict=True)
    ):
        mapper = None
        if alignment_maps is not None and clip_index < len(alignment_maps):
            mapper = alignment_maps[clip_index]
        if frame_labels:
            logger.debug('frame_labels keys: %s', list(frame_labels.keys()))
        crop = plan["crop"]
        scaled = plan["scaled"]
        pad = plan["pad"]
        width = int(plan["width"])
        height = int(plan["height"])
        trim_start = int(trim_start)
        raw_label, safe_label = _derive_labels(file_path, meta)

        debug_state: Optional[_ColorDebugState] = None
        if debug_enabled and debug_root is not None:
            artifacts = getattr(result, "debug", None)
            core = getattr(result.clip, "core", None)
            if core is None and artifacts is not None and artifacts.normalized_clip is not None:
                core = getattr(artifacts.normalized_clip, "core", None)
            if artifacts is None:
                fallback_props = {}
                if hasattr(result, "source_props"):
                    try:
                        fallback_props = dict(getattr(result, "source_props", {}) or {})
                    except (TypeError, ValueError):
                        fallback_props = {}
                fallback_clip = result.clip if result.clip is not None else None
                if fallback_clip is not None and core is None:
                    core = getattr(fallback_clip, "core", None)
                logger.debug(
                    "Colour debug artifacts unavailable for %s; falling back to processed clip",
                    file_path,
                )
                artifacts = vs_core.ColorDebugArtifacts(
                    normalized_clip=fallback_clip,
                    normalized_props=fallback_props,
                    original_props=fallback_props,
                    color_tuple=(None, None, None, None),
                )
            if core is None:
                try:
                    import vapoursynth as vs  # type: ignore

                    core = getattr(vs, "core", None)
                except ImportError:
                    core = None
            debug_state = _ColorDebugState(
                enabled=core is not None,
                base_dir=(debug_root / safe_label),
                label=safe_label,
                core=core,
                compression_level=int(getattr(cfg, "compression_level", 1)),
                rgb_dither=debug_dither,
                logger_obj=logger,
                artifacts=artifacts,
            )

        overlay_state = overlay_states[clip_index]
        base_overlay_text = getattr(result, "overlay_text", None)
        source_props_raw = getattr(result, "source_props", {})
        resolved_clip, resolved_source_props = _resolve_source_props(
            result.clip,
            source_props_raw,
            color_cfg=color_cfg,
            file_name=str(file_path),
            warning_sink=warnings_sink,
        )
        source_props = resolved_source_props
        is_sdr_pipeline = _is_sdr_pipeline(result.tonemap, resolved_source_props)
        range_full, range_limited = _range_constants()
        export_range = getattr(cfg, "export_range", ExportRange.FULL)
        expand_to_full = _should_expand_to_full(export_range)
        clip_color_range = _resolve_output_color_range(resolved_source_props, result.tonemap)
        original_color_range = clip_color_range
        if expand_to_full:
            clip_color_range = range_full
        _, _, _, source_color_meta = vs_core._resolve_color_metadata(resolved_source_props)
        if source_color_meta not in (range_full, range_limited):
            source_color_hint = original_color_range
        else:
            source_color_hint = int(source_color_meta)

        for frame in frames:
            frame_idx = int(frame)
            mapped_idx = frame_idx
            if mapper is not None:
                try:
                    mapped_idx, _, clamped = mapper.map_frame(frame_idx)
                except (IndexError, ValueError, LookupError) as exc:
                    logger.warning(
                        "Failed to map frame %s for %s via alignment: %s",
                        frame_idx,
                        file_path,
                        exc,
                    )
                    clamped = False
                else:
                    if clamped:
                        logger.debug(
                            "Alignment clamped frame %s→%s for %s",
                            frame_idx,
                            mapped_idx,
                            file_path,
                        )
            detail_info = selection_details.get(frame_idx) if selection_details else None
            selection_label = frame_labels.get(frame_idx) if frame_labels else None
            if selection_label is None and detail_info is not None:
                derived_label = detail_info.get("label") or detail_info.get("type")
                if derived_label:
                    selection_label = str(derived_label)
            if selection_label is not None:
                logger.debug('Selection label for frame %s: %s', frame_idx, selection_label)
            actual_idx, was_clamped = _clamp_frame_index(resolved_clip, mapped_idx)
            if was_clamped:
                logger.debug(
                    "Frame %s exceeds available frames (%s) in %s; using %s",
                    mapped_idx,
                    getattr(resolved_clip, 'num_frames', 'unknown'),
                    file_path,
                    actual_idx,
                )
            if debug_state and debug_state.normalized_clip is not None:
                debug_state.capture_stage(
                    "post_normalisation",
                    actual_idx,
                    debug_state.normalized_clip,
                    debug_state.normalized_props,
                )

            # Fetch dynamic props from the source clip for this frame
            current_props = dict(source_props)
            try:
                # Use source_clip to ensure we get original metadata (e.g. DoVi RPU stats)
                # that might be lost or static in the processed clip.
                # We use actual_idx which is clamped and mapped.
                source_frame_idx = _resolve_source_frame_index(actual_idx, trim_start)

                if source_frame_idx is not None:
                    frame_ref = source_clip.get_frame(source_frame_idx)
                    dynamic_props = dict(frame_ref.props)
                    current_props.update(dynamic_props)
                else:
                    logger.debug("Frame %s falls within padding; skipping dynamic props", actual_idx)
            except (RuntimeError, ValueError, KeyError) as exc:
                logger.debug("Failed to fetch dynamic props for frame %s: %s", actual_idx, exc)

            if debug_enabled:
                overlay_text = None
            else:
                overlay_text = _compose_overlay_text(
                    base_overlay_text,
                    color_cfg,
                    plan,
                    selection_label,
                    current_props,
                    tonemap_info=result.tonemap,
                    selection_detail=detail_info,
                )
            file_name = _prepare_filename(frame_idx, safe_label)
            target_path = out_dir / file_name

            try:
                resolved_frame = _resolve_source_frame_index(actual_idx, trim_start)
                use_ffmpeg = use_ffmpeg_runtime and resolved_frame is not None
                if use_ffmpeg_runtime and resolved_frame is None:
                    logger.debug(
                        "Frame %s for %s falls within synthetic trim padding; "
                        "using VapourSynth writer",
                        frame_idx,
                        file_path,
                    )
                if use_ffmpeg:
                    assert resolved_frame is not None
                    if overlay_text and overlay_state.get("overlay_status") != "ok":
                        logger.info("[OVERLAY] %s applied (ffmpeg)", file_path)
                        overlay_state["overlay_status"] = "ok"
                    _save_frame_with_ffmpeg(
                        file_path,
                        resolved_frame,
                        crop,
                        scaled,
                        pad,
                        target_path,
                        cfg,
                        width,
                        height,
                        selection_label,
                        overlay_text=overlay_text,
                        geometry_plan=plan,
                        is_sdr=is_sdr_pipeline,
                        pivot_notifier=pivot_notifier,
                        frame_info_allowed=frame_info_allowed_default,
                        overlays_allowed=overlays_allowed_default and bool(overlay_text),
                        target_range=clip_color_range,
                        expand_to_full=expand_to_full,
                        source_color_range=source_color_hint,
                    )
                else:
                    _save_frame_with_fpng(
                        resolved_clip,
                        actual_idx,
                        crop,
                        scaled,
                        pad,
                        target_path,
                        cfg,
                        raw_label,
                        frame_idx,
                        selection_label,
                        overlay_text=overlay_text,
                        overlay_state=overlay_state,
                        strict_overlay=bool(getattr(color_cfg, "strict", False)),
                        source_props=resolved_source_props,
                        geometry_plan=plan,
                        tonemap_info=result.tonemap,
                        pivot_notifier=pivot_notifier,
                        color_cfg=color_cfg,
                        file_name=str(file_path),
                        warning_sink=warnings_sink,
                        debug_state=debug_state,
                        frame_info_allowed=frame_info_allowed_default,
                        overlays_allowed=overlays_allowed_default,
                        expand_to_full=expand_to_full,
                    )
            except ScreenshotWriterError:
                raise
            except Exception as exc:
                message = (
                    f"[RENDER] Falling back to placeholder for frame {frame_idx} of {file_path}: {exc}"
                )
                logger.exception(message)
                if warnings_sink is not None:
                    warnings_sink.append(message)
                _save_frame_placeholder(target_path)

            created.append(str(target_path))
            if progress_callback is not None:
                progress_callback(1)

        if warnings_sink is not None:
            warnings_sink.extend(_get_overlay_warnings(overlay_state))

    return created
