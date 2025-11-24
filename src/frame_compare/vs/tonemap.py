"""Tonemap processing, verification, and screenshot preparation."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, cast

from src.frame_compare.env_flags import env_flag_enabled

from .color import _detect_rgb_color_range, normalise_color_metadata  # pyright: ignore[reportPrivateUsage]
from .env import _get_vapoursynth_module  # pyright: ignore[reportPrivateUsage]
from .props import (  # pyright: ignore[reportPrivateUsage]
    _apply_frame_props_dict,  # pyright: ignore[reportPrivateUsage]
    _call_set_frame_prop,  # pyright: ignore[reportPrivateUsage]
    _ensure_std_namespace,  # pyright: ignore[reportPrivateUsage]
    _props_signal_hdr,  # pyright: ignore[reportPrivateUsage]
    _snapshot_frame_props,  # pyright: ignore[reportPrivateUsage]
)

logger = logging.getLogger("src.frame_compare.vs.tonemap")
DOVI_DEBUG_ENV_FLAG = "FRAME_COMPARE_DOVI_DEBUG"


def _emit_vs_dovi_debug(payload: Mapping[str, Any]) -> None:
    if not env_flag_enabled(os.environ.get(DOVI_DEBUG_ENV_FLAG)):
        return
    try:
        message = json.dumps(dict(payload), default=str)
    except Exception:
        logger.debug("Unable to serialize VS tonemap debug payload", exc_info=True)
        return
    print("[DOVI_DEBUG]", message, file=sys.stderr)

class ClipProcessError(RuntimeError):
    """Raised when screenshot preparation fails."""


@dataclass(frozen=True)
class TonemapInfo:
    """Metadata describing how a clip was tonemapped."""

    applied: bool
    tone_curve: Optional[str]
    dpd: int
    target_nits: float
    dst_min_nits: float
    src_csp_hint: Optional[int]
    reason: Optional[str] = None
    output_color_range: Optional[int] = None
    range_detection: Optional[str] = None
    knee_offset: Optional[float] = None
    dpd_preset: Optional[str] = None
    dpd_black_cutoff: Optional[float] = None
    post_gamma: Optional[float] = None
    post_gamma_enabled: bool = False
    smoothing_period: Optional[float] = None
    scene_threshold_low: Optional[float] = None
    scene_threshold_high: Optional[float] = None
    percentile: Optional[float] = None
    contrast_recovery: Optional[float] = None
    metadata: Optional[int] = None
    use_dovi: Optional[bool] = None
    visualize_lut: bool = False
    show_clipping: bool = False


@dataclass(frozen=True)
class VerificationResult:
    """Result of comparing a tonemapped clip against a naive SDR convert."""

    frame: int
    average: float
    maximum: float
    auto_selected: bool


@dataclass(frozen=True)
class ColorDebugArtifacts:
    """Clips and metadata captured for colour debugging."""

    normalized_clip: Any | None
    normalized_props: Mapping[str, Any] | None
    original_props: Mapping[str, Any] | None
    color_tuple: tuple[Optional[int], Optional[int], Optional[int], Optional[int]] | None


@dataclass(frozen=True)
class TonemapSettings:
    """Resolved tonemap configuration used for the current clip."""

    preset: str
    tone_curve: str
    target_nits: float
    dynamic_peak_detection: bool
    dst_min_nits: float
    knee_offset: float
    dpd_preset: str
    dpd_black_cutoff: float
    smoothing_period: float
    scene_threshold_low: float
    scene_threshold_high: float
    percentile: float
    contrast_recovery: float
    metadata: Optional[int]
    use_dovi: Optional[bool]
    visualize_lut: bool
    show_clipping: bool


@dataclass(frozen=True)
class ClipProcessResult:
    """Container for processed clip and metadata."""

    clip: Any
    tonemap: TonemapInfo
    overlay_text: Optional[str]
    verification: Optional[VerificationResult]
    source_props: Mapping[str, Any]
    debug: Optional[ColorDebugArtifacts] = None


def _apply_set_frame_prop(clip: Any, **kwargs: Any) -> Any:
    std_ns = _ensure_std_namespace(
        clip,
        ClipProcessError("clip.std namespace missing for SetFrameProp"),
    )
    set_prop = getattr(std_ns, "SetFrameProp", None)
    if not callable(set_prop):  # pragma: no cover - defensive
        raise ClipProcessError("clip.std.SetFrameProp is unavailable")
    return _call_set_frame_prop(set_prop, clip, **kwargs)


def _normalize_rgb_props(clip: Any, transfer: Optional[int], primaries: Optional[int]) -> Any:
    work = _apply_set_frame_prop(clip, prop="_Matrix", intval=0)
    work = _apply_set_frame_prop(work, prop="_ColorRange", intval=0)
    if transfer is not None:
        work = _apply_set_frame_prop(work, prop="_Transfer", intval=int(transfer))
    if primaries is not None:
        work = _apply_set_frame_prop(work, prop="_Primaries", intval=int(primaries))
    return work


def _deduce_src_csp_hint(transfer: Optional[int], primaries: Optional[int]) -> Optional[int]:
    if transfer == 16 and primaries == 9:
        return 1
    if transfer == 18 and primaries == 9:
        return 2
    return None


_TONEMAP_UNSUPPORTED_KWARGS: set[str] = set()


def _parse_unexpected_kwarg(exc: BaseException) -> tuple[str, ...]:
    """
    Extract unexpected keyword argument names from TypeError/vapoursynth errors.

    Returns:
        tuple[str, ...]: Names of any kwargs rejected by the downstream Tonemap call.
    """

    message = str(exc)
    match = re.search(r"unexpected keyword argument '([^']+)'", message)
    if match:
        return (match.group(1),)

    match = re.search(
        r"does not take argument\(s\) named ([^:]+)",
        message,
        re.IGNORECASE,
    )
    if match:
        raw = match.group(1)
        sanitized = raw.replace(" and ", ",")
        names = [
            part.strip().strip("'\"")
            for part in sanitized.split(",")
        ]
        filtered = tuple(name for name in names if name)
        if filtered:
            return filtered

    return ()


def _call_tonemap_function(
    func: Callable[..., Any],
    clip: Any,
    call_kwargs: Dict[str, Any],
    *,
    file_name: str,
) -> Any:
    usable_kwargs = {
        key: value for key, value in call_kwargs.items() if key not in _TONEMAP_UNSUPPORTED_KWARGS
    }
    max_attempts = len(usable_kwargs) + 1
    attempts = 0
    while True:
        attempts += 1
        try:
            return func(clip, **usable_kwargs)
        except Exception as exc:  # pragma: no cover - vapoursynth raises custom errors
            missing_names = _parse_unexpected_kwarg(exc)
            handled = False
            for missing in missing_names:
                if missing in usable_kwargs:
                    if missing not in _TONEMAP_UNSUPPORTED_KWARGS:
                        _TONEMAP_UNSUPPORTED_KWARGS.add(missing)
                        logger.warning(
                            "[Tonemap compat] %s missing support for '%s'; retrying without it",
                            file_name,
                            missing,
                        )
                    usable_kwargs.pop(missing, None)
                    handled = True
            if handled and attempts < max_attempts:
                continue
            raise


def _apply_post_gamma_levels(
    core: Any,
    clip: Any,
    *,
    gamma: float,
    file_name: str,
    log: logging.Logger,
) -> tuple[Any, bool]:
    if abs(gamma - 1.0) < 1e-4:
        return clip, False

    def _resolve_level_bounds() -> tuple[float | int, float | int, float | int, float | int]:
        fmt = getattr(clip, "format", None)
        if fmt is None:
            return 16, 235, 16, 235
        sample_type = getattr(fmt, "sample_type", None)
        bits = getattr(fmt, "bits_per_sample", None)
        sample_type_val: Optional[int] = None
        if sample_type is not None:
            try:
                sample_type_val = int(sample_type)
            except Exception:
                name = str(getattr(sample_type, "name", "")).upper()
                if name == "INTEGER":
                    sample_type_val = 0
                elif name == "FLOAT":
                    sample_type_val = 1
        min_ratio = 16.0 / 255.0
        max_ratio = 235.0 / 255.0
        if sample_type_val == 1:
            return min_ratio, max_ratio, min_ratio, max_ratio
        if sample_type_val == 0 and isinstance(bits, int) and bits > 0:
            full_scale = float((1 << bits) - 1)
            min_value = round(min_ratio * full_scale)
            max_value = round(max_ratio * full_scale)
            return int(min_value), int(max_value), int(min_value), int(max_value)
        return 16, 235, 16, 235

    min_in, max_in, min_out, max_out = _resolve_level_bounds()
    std_ns = getattr(core, "std", None)
    levels = getattr(std_ns, "Levels", None) if std_ns is not None else None
    if not callable(levels):
        log.warning("[TM GAMMA] %s requested post-gamma but std.Levels is unavailable", file_name)
        return clip, False
    try:
        adjusted = levels(
            clip,
            min_in=min_in,
            max_in=max_in,
            min_out=min_out,
            max_out=max_out,
            gamma=float(gamma),
        )
        log.info("[TM GAMMA] %s applied gamma=%.3f", file_name, gamma)
        return adjusted, True
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("[TM GAMMA] %s failed to apply gamma: %s", file_name, exc)
        return clip, False


def _tonemap_with_retries(
    core: Any,
    rgb_clip: Any,
    *,
    tone_curve: str,
    target_nits: float,
    dst_min: float,
    dpd: int,
    knee_offset: float,
    dpd_preset: str,
    dpd_black_cutoff: float,
    smoothing_period: float,
    scene_threshold_low: float,
    scene_threshold_high: float,
    percentile: float,
    contrast_recovery: float,
    metadata: Optional[int],
    use_dovi: Optional[bool],
    visualize_lut: bool,
    show_clipping: bool,
    src_hint: Optional[int],
    file_name: str,
) -> Any:
    libplacebo = getattr(core, "libplacebo", None)
    tonemap = getattr(libplacebo, "Tonemap", None) if libplacebo is not None else None
    if not callable(tonemap):
        libplacebo = getattr(core, "placebo", None)
        tonemap = getattr(libplacebo, "Tonemap", None) if libplacebo is not None else None
    if not callable(tonemap):
        raise ClipProcessError("libplacebo.Tonemap is unavailable")

    kwargs = dict(
        dst_csp=0,
        dst_prim=1,
        dst_max=float(target_nits),
        dst_min=float(dst_min),
        dynamic_peak_detection=int(dpd),
        smoothing_period=float(smoothing_period),
        scene_threshold_low=float(scene_threshold_low),
        scene_threshold_high=float(scene_threshold_high),
        percentile=float(percentile),
        gamut_mapping=1,
        tone_mapping_function_s=tone_curve,
        tone_mapping_param=float(knee_offset),
        peak_detection_preset=str(dpd_preset),
        black_cutoff=float(dpd_black_cutoff),
        contrast_recovery=float(contrast_recovery),
        visualize_lut=bool(visualize_lut),
        show_clipping=bool(show_clipping),
        log_level=2,
    )
    if metadata is not None:
        kwargs["metadata"] = int(metadata)
    if use_dovi is not None:
        kwargs["use_dovi"] = bool(use_dovi)

    def _attempt(**extra_kwargs: Any) -> Any:
        combined = dict(kwargs)
        combined.update(extra_kwargs)
        return _call_tonemap_function(tonemap, rgb_clip, combined, file_name=file_name)

    if src_hint is not None:
        try:
            return _attempt(src_csp=src_hint)
        except Exception as exc:
            logger.warning("[Tonemap attempt A failed] %s src_csp=%s: %s", file_name, src_hint, exc)
    try:
        return _attempt()
    except Exception as exc:
        logger.warning("[Tonemap attempt B failed] %s infer-from-props: %s", file_name, exc)
    try:
        return _attempt(src_csp=1)
    except Exception as exc:
        raise ClipProcessError(
            f"libplacebo.Tonemap final fallback failed for '{file_name}': {exc}"
        ) from exc


_TONEMAP_PRESETS: Dict[str, Dict[str, float | str | bool]] = {
    # NOTE: Keep these defaults in sync with the preset matrix in src/data/config.toml.template.
    "reference": {
        "tone_curve": "bt.2390",
        "target_nits": 100.0,
        "dynamic_peak_detection": True,
        "knee_offset": 0.50,
        "dst_min_nits": 0.18,
        "dpd_preset": "high_quality",
        "dpd_black_cutoff": 0.01,
        "smoothing_period": 45.0,
        "scene_threshold_low": 0.8,
        "scene_threshold_high": 2.4,
        "percentile": 99.995,
        "contrast_recovery": 0.30,
        "metadata": "auto",
        "use_dovi": True,
        "visualize_lut": False,
        "show_clipping": False,
    },
    "bt2390_spec": {
        "tone_curve": "bt.2390",
        "target_nits": 100.0,
        "dynamic_peak_detection": True,
        "knee_offset": 0.50,
        "dst_min_nits": 0.18,
        "dpd_preset": "high_quality",
        "dpd_black_cutoff": 0.0,
        "smoothing_period": 25.0,
        "scene_threshold_low": 0.9,
        "scene_threshold_high": 3.0,
        "percentile": 100.0,
        "contrast_recovery": 0.05,
        "metadata": "auto",
        "use_dovi": True,
        "visualize_lut": False,
        "show_clipping": False,
    },
    "filmic": {
        "tone_curve": "bt.2446a",
        "target_nits": 100.0,
        "dynamic_peak_detection": True,
        "dst_min_nits": 0.16,
        "dpd_preset": "high_quality",
        "dpd_black_cutoff": 0.008,
        "knee_offset": 0.58,
        "smoothing_period": 55.0,
        "scene_threshold_low": 0.7,
        "scene_threshold_high": 2.0,
        "percentile": 99.9,
        "contrast_recovery": 0.20,
        "metadata": "auto",
        "use_dovi": True,
        "visualize_lut": False,
        "show_clipping": False,
    },
    "spline": {
        "tone_curve": "spline",
        "target_nits": 105.0,
        "dynamic_peak_detection": True,
        "dst_min_nits": 0.17,
        "dpd_preset": "high_quality",
        "dpd_black_cutoff": 0.009,
        "knee_offset": 0.52,
        "smoothing_period": 35.0,
        "scene_threshold_low": 0.8,
        "scene_threshold_high": 2.2,
        "percentile": 99.98,
        "contrast_recovery": 0.25,
        "metadata": "auto",
        "use_dovi": True,
        "visualize_lut": False,
        "show_clipping": False,
    },
    "contrast": {
        "tone_curve": "bt.2390",
        "target_nits": 110.0,
        "dynamic_peak_detection": True,
        "dst_min_nits": 0.15,
        "dpd_preset": "high_quality",
        "dpd_black_cutoff": 0.008,
        "knee_offset": 0.42,
        "smoothing_period": 30.0,
        "scene_threshold_low": 0.8,
        "scene_threshold_high": 2.2,
        "percentile": 99.99,
        "contrast_recovery": 0.45,
        "metadata": "auto",
        "use_dovi": True,
        "visualize_lut": False,
        "show_clipping": False,
    },
    "bright_lift": {
        "tone_curve": "bt.2390",
        "target_nits": 130.0,
        "dynamic_peak_detection": True,
        "dst_min_nits": 0.22,
        "dpd_preset": "high_quality",
        "dpd_black_cutoff": 0.012,
        "knee_offset": 0.46,
        "smoothing_period": 35.0,
        "scene_threshold_low": 0.8,
        "scene_threshold_high": 2.0,
        "percentile": 99.99,
        "contrast_recovery": 0.50,
        "metadata": "auto",
        "use_dovi": True,
        "visualize_lut": False,
        "show_clipping": False,
    },
    "highlight_guard": {
        "tone_curve": "bt.2390",
        "target_nits": 90.0,
        "dynamic_peak_detection": True,
        "dst_min_nits": 0.16,
        "dpd_preset": "high_quality",
        "dpd_black_cutoff": 0.008,
        "knee_offset": 0.55,
        "smoothing_period": 50.0,
        "scene_threshold_low": 0.9,
        "scene_threshold_high": 3.0,
        "percentile": 99.9,
        "contrast_recovery": 0.15,
        "metadata": "auto",
        "use_dovi": True,
        "visualize_lut": False,
        "show_clipping": False,
    },
}


_METADATA_NAME_TO_CODE = {
    "auto": 0,
    "none": 1,
    "hdr10": 2,
    "hdr10+": 3,
    "hdr10plus": 3,
    "luminance": 4,
    "ciey": 4,
    "cie_y": 4,
}


def _resolve_tonemap_settings(cfg: Any, props: Mapping[str, Any] | None = None) -> TonemapSettings:
    preset = str(getattr(cfg, "preset", "") or "").strip().lower()
    tone_curve = str(getattr(cfg, "tone_curve", "bt.2390") or "bt.2390")
    provided_raw = getattr(cfg, "_provided_keys", None)
    provided: set[str]
    if isinstance(provided_raw, set):
        provided = {str(field) for field in cast(set[Any], provided_raw)}
    else:
        provided = set()
    if preset and preset != "custom":
        preset_vals = _TONEMAP_PRESETS.get(preset) or {}
    else:
        preset_vals = {}

    def _resolve_value(field: str, default: Any) -> Any:
        if preset_vals and field in preset_vals and field not in provided:
            return preset_vals[field]
        return getattr(cfg, field, preset_vals.get(field, default))

    tone_curve = str(_resolve_value("tone_curve", tone_curve))
    target_nits = float(_resolve_value("target_nits", 100.0))
    dpd_flag = bool(_resolve_value("dynamic_peak_detection", True))
    dst_min = float(_resolve_value("dst_min_nits", 0.18))
    knee_offset = float(_resolve_value("knee_offset", 0.5))
    dpd_preset_value = str(_resolve_value("dpd_preset", "high_quality") or "").strip().lower()
    dpd_black_cutoff = float(_resolve_value("dpd_black_cutoff", 0.01))
    smoothing_period = float(_resolve_value("smoothing_period", 45.0))
    scene_threshold_low = float(_resolve_value("scene_threshold_low", 0.8))
    scene_threshold_high = float(_resolve_value("scene_threshold_high", 2.4))
    percentile = float(_resolve_value("percentile", 99.995))
    contrast_recovery = float(_resolve_value("contrast_recovery", 0.3))
    metadata_value = _resolve_value("metadata", "auto")
    use_dovi_value = _resolve_value("use_dovi", None)
    visualize_lut = bool(_resolve_value("visualize_lut", False))
    show_clipping = bool(_resolve_value("show_clipping", False))

    if not dpd_flag:
        dpd_preset_value = "off"
        dpd_black_cutoff = 0.0

    if dpd_preset_value not in {"off", "fast", "balanced", "high_quality"}:
        dpd_preset_value = "off" if not dpd_flag else "high_quality"

    metadata_code: Optional[int]
    if metadata_value is None:
        metadata_code = None
    elif isinstance(metadata_value, (int, float)):
        metadata_code = int(metadata_value)
    else:
        metadata_key = str(metadata_value).strip().lower().replace(" ", "")
        if metadata_key in _METADATA_NAME_TO_CODE:
            metadata_code = _METADATA_NAME_TO_CODE[metadata_key]
        else:
            try:
                metadata_code = int(metadata_key)
            except ValueError:
                metadata_code = 0
            else:
                metadata_code = max(0, min(4, metadata_code))

    if metadata_code is not None and metadata_code < 0:
        metadata_code = 0

    if isinstance(use_dovi_value, str):
        lowered = use_dovi_value.strip().lower()
        if lowered in {"auto", ""}:
            use_dovi_value = None
        elif lowered in {"true", "1", "yes", "on"}:
            use_dovi_value = True
        elif lowered in {"false", "0", "no", "off"}:
            use_dovi_value = False
        else:
            use_dovi_value = None
    elif use_dovi_value is not None:
        use_dovi_value = bool(use_dovi_value)

    # Auto-enable DoVi if RPU blob is present in props and setting is auto (None)
    if use_dovi_value is None and props:
        for key in props:
            if key in ("DolbyVisionRPU", "_DolbyVisionRPU", "DolbyVisionRPU_b", "_DolbyVisionRPU_b"):
                use_dovi_value = True
                break

    return TonemapSettings(
        preset=preset or "custom",
        tone_curve=tone_curve,
        target_nits=float(target_nits),
        dynamic_peak_detection=bool(dpd_flag),
        dst_min_nits=float(dst_min),
        knee_offset=float(knee_offset),
        dpd_preset=dpd_preset_value or ("off" if not dpd_flag else "high_quality"),
        dpd_black_cutoff=float(dpd_black_cutoff),
        smoothing_period=float(smoothing_period),
        scene_threshold_low=float(scene_threshold_low),
        scene_threshold_high=float(scene_threshold_high),
        percentile=float(percentile),
        contrast_recovery=float(contrast_recovery),
        metadata=metadata_code,
        use_dovi=use_dovi_value if isinstance(use_dovi_value, (bool, type(None))) else None,
        visualize_lut=bool(visualize_lut),
        show_clipping=bool(show_clipping),
    )







def resolve_effective_tonemap(cfg: Any, props: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Resolve the effective tonemap preset, curve, and luminance for ``cfg``."""

    settings = _resolve_tonemap_settings(cfg, props=props)
    resolved = {
        "preset": settings.preset,
        "tone_curve": settings.tone_curve,
        "target_nits": float(settings.target_nits),
        "dynamic_peak_detection": bool(settings.dynamic_peak_detection),
        "dst_min_nits": float(settings.dst_min_nits),
        "knee_offset": float(settings.knee_offset),
        "dpd_preset": settings.dpd_preset,
        "dpd_black_cutoff": float(settings.dpd_black_cutoff),
        "smoothing_period": float(settings.smoothing_period),
        "scene_threshold_low": float(settings.scene_threshold_low),
        "scene_threshold_high": float(settings.scene_threshold_high),
        "percentile": float(settings.percentile),
        "contrast_recovery": float(settings.contrast_recovery),
        "metadata": settings.metadata,
        "use_dovi": settings.use_dovi,
        "visualize_lut": bool(settings.visualize_lut),
        "show_clipping": bool(settings.show_clipping),
    }
    _emit_vs_dovi_debug(
        {
            "phase": "vs_resolve_effective_tonemap",
            "entrypoint": "vs_core",
            "cfg_preset": getattr(cfg, "preset", None),
            "cfg_tone_curve": getattr(cfg, "tone_curve", None),
            "cfg_target_nits": getattr(cfg, "target_nits", None),
            "cfg_use_dovi": getattr(cfg, "use_dovi", None),
            "cfg_metadata": getattr(cfg, "metadata", None),
            "resolved_use_dovi": resolved.get("use_dovi"),
            "resolved_metadata": resolved.get("metadata"),
            "resolved_target_nits": resolved.get("target_nits"),
        }
    )
    return resolved


def _format_overlay_text(
    template: str,
    *,
    tone_curve: str,
    dpd: int,
    target_nits: float,
    preset: str,
    dst_min_nits: float,
    knee_offset: float,
    dpd_preset: str,
    dpd_black_cutoff: float,
    post_gamma: float,
    post_gamma_enabled: bool,
    smoothing_period: float,
    scene_threshold_low: float,
    scene_threshold_high: float,
    percentile: float,
    contrast_recovery: float,
    metadata: Optional[int],
    use_dovi: Optional[bool],
    visualize_lut: bool,
    show_clipping: bool,
    reason: Optional[str] = None,
) -> str:
    """
    Format an overlay text template with tonemapping parameters.

    Parameters:
        template (str): A format string that may reference the following keys: `tone_curve`, `curve` (alias),
        `dynamic_peak_detection`, `dpd` (numeric), `dynamic_peak_detection_bool`, `dpd_bool` (boolean),
        `target_nits` (int when whole number, otherwise float), `target_nits_float` (always float),
        `dst_min_nits`, `knee_offset`, `dpd_preset`, `dpd_black_cutoff`, `post_gamma`,
        `post_gamma_enabled`, `preset`, and `reason`.
        tone_curve (str): Name of the tone curve to show.
        dpd (int): Dynamic peak detection flag (0 or 1); boolean aliases are provided in the template values.
        target_nits (float): Target display luminance in nits.
        preset (str): Tonemap preset name.
        reason (Optional[str]): Optional explanatory text included as `reason` in the template.

    Returns:
        Formatted overlay string using the provided template and values; returns `template` unchanged if formatting fails.
    """
    values = {
        "tone_curve": tone_curve,
        "curve": tone_curve,
        "dynamic_peak_detection": dpd,
        "dpd": dpd,
        "dynamic_peak_detection_bool": bool(dpd),
        "dpd_bool": bool(dpd),
        "target_nits": (
            int(target_nits)
            if abs(target_nits - round(target_nits)) < 1e-6
            else target_nits
        ),
        "target_nits_float": target_nits,
        "preset": preset,
        "reason": reason or "",
        "dst_min_nits": dst_min_nits,
        "knee_offset": knee_offset,
        "dpd_preset": dpd_preset,
        "dpd_black_cutoff": dpd_black_cutoff,
        "post_gamma": post_gamma,
        "post_gamma_enabled": post_gamma_enabled,
        "smoothing_period": smoothing_period,
        "scene_threshold_low": scene_threshold_low,
        "scene_threshold_high": scene_threshold_high,
        "percentile": percentile,
        "contrast_recovery": contrast_recovery,
        "metadata": metadata,
        "use_dovi": use_dovi,
        "visualize_lut": visualize_lut,
        "show_clipping": show_clipping,
    }
    try:
        return template.format(**values)
    except Exception:
        return template


def _pick_verify_frame(
    clip: Any,
    cfg: Any,
    *,
    fps: float,
    file_name: str,
    warning_sink: Optional[List[str]] = None,
) -> tuple[int, bool]:
    """
    Select a frame index to use for verification, optionally using an automatic brightness-based sampling.

    Parameters:
        clip (Any): VapourSynth clip to inspect; must expose `num_frames` and support `std.PlaneStats()`.
        cfg (Any): Configuration object with optional attributes:
            - verify_frame (int): explicit frame index to use.
            - verify_auto (bool): enable automatic sampling when not set.
            - verify_start_seconds (float): sampling start time in seconds.
            - verify_step_seconds (float): sampling step in seconds.
            - verify_max_seconds (float): maximum sampling time in seconds.
            - verify_luma_threshold (float): PlaneStatsAverage threshold for selection.
        fps (float): Frames-per-second used to convert seconds to frame indices.
        file_name (str): File name used in log and warning messages.
        warning_sink (Optional[List[str]]): Optional list to append human-readable warning strings.

    Returns:
        tuple[int, bool]: Selected frame index and a flag that is `true` if the frame was chosen by automatic sampling, `false` otherwise.
    """
    num_frames = getattr(clip, "num_frames", 0) or 0
    if num_frames <= 0:
        message = f"[VERIFY] {file_name} has no frames; using frame 0"
        logger.warning(message)
        if warning_sink is not None:
            warning_sink.append(message)
        return 0, False

    manual = getattr(cfg, "verify_frame", None)
    if isinstance(manual, int):
        idx = max(0, min(num_frames - 1, manual))
        logger.info("[VERIFY] %s using configured frame %d", file_name, idx)
        return idx, False

    if not bool(getattr(cfg, "verify_auto", True)):
        idx = max(0, min(num_frames - 1, num_frames // 2))
        logger.info("[VERIFY] %s auto disabled; using middle frame %d", file_name, idx)
        return idx, False

    start_seconds = float(getattr(cfg, "verify_start_seconds", 10.0))
    step_seconds = float(getattr(cfg, "verify_step_seconds", 10.0))
    max_seconds = float(getattr(cfg, "verify_max_seconds", 90.0))
    threshold = float(getattr(cfg, "verify_luma_threshold", 0.10))

    step_frames = (
        max(1, int(round(step_seconds * fps)))
        if fps > 0
        else max(1, int(step_seconds) or 1)
    )
    start_frame = int(round(start_seconds * fps)) if fps > 0 else int(start_seconds)
    start_frame = max(0, min(num_frames - 1, start_frame))
    max_frame = int(round(max_seconds * fps)) if fps > 0 else int(max_seconds)
    max_frame = max(
        start_frame,
        min(num_frames - 1, max_frame if max_frame > 0 else num_frames - 1),
    )

    stats_clip = None
    try:
        stats_clip = clip.std.PlaneStats()
    except Exception as exc:
        message = f"[VERIFY] {file_name} unable to create PlaneStats: {exc}"
        logger.warning(message)
        if warning_sink is not None:
            warning_sink.append(message)
        middle = max(0, min(num_frames - 1, num_frames // 2))
        return middle, False

    best_idx: Optional[int] = None
    best_avg = -1.0
    for idx in range(start_frame or 1, max_frame + 1, step_frames):
        try:
            frame = stats_clip.get_frame(idx)
        except Exception:
            continue
        avg = float(frame.props.get("PlaneStatsAverage", 0.0))
        if avg >= threshold:
            logger.info(
                "[VERIFY] %s auto-picked frame %d (avg=%.4f) start=%d step=%d",
                file_name,
                idx,
                avg,
                start_frame,
                step_frames,
            )
            return idx, True
        if avg > best_avg:
            best_idx, best_avg = idx, avg

    if best_idx is not None:
        logger.info(
            "[VERIFY] %s brightest sampled frame %d (avg=%.4f) threshold %.3f",
            file_name,
            best_idx,
            best_avg,
            threshold,
        )
        return best_idx, True

    middle = max(0, min(num_frames - 1, num_frames // 2))
    logger.info("[VERIFY] %s fallback to middle frame %d", file_name, middle)
    return middle, False


def _compute_verification(
    core: Any,
    tonemapped: Any,
    naive: Any,
    frame_idx: int,
    *,
    auto_selected: bool,
) -> VerificationResult:
    expr = core.std.Expr([tonemapped, naive], "x y - abs")
    stats = core.std.PlaneStats(expr)
    props = stats.get_frame(frame_idx).props
    average = float(props.get("PlaneStatsAverage", 0.0))
    maximum = float(props.get("PlaneStatsMax", 0.0))
    fmt = getattr(expr, "format", None)
    bits = getattr(fmt, "bits_per_sample", None) if fmt is not None else None
    sample_type = getattr(fmt, "sample_type", None) if fmt is not None else None

    is_integer_format = False
    if sample_type is not None:
        name = getattr(sample_type, "name", None)
        if isinstance(name, str):
            is_integer_format = name.upper() == "INTEGER"
        else:
            try:
                is_integer_format = int(sample_type) == 0
            except Exception:
                is_integer_format = False

    if is_integer_format and isinstance(bits, int) and bits > 0:
        peak = float((1 << bits) - 1)
        if peak > 0.0:
            average /= peak
            maximum /= peak
    return VerificationResult(
        frame=frame_idx,
        average=average,
        maximum=maximum,
        auto_selected=auto_selected,
    )


def process_clip_for_screenshot(
    clip: Any,
    file_name: str,
    cfg: Any,
    *,
    enable_overlay: bool = True,
    enable_verification: bool = True,
    logger_override: Optional[logging.Logger] = None,
    warning_sink: Optional[List[str]] = None,
    debug_color: bool = False,
    stored_source_props: Optional[Mapping[str, Any]] = None,
) -> ClipProcessResult:
    """
    Prepare a VapourSynth clip for screenshot export by applying HDR->SDR tonemapping, optional overlay text, and optional verification against a naive SDR conversion.

    Parameters:
        clip: VapourSynth clip to process.
        file_name (str): Source filename used in log messages.
        cfg: Configuration object supplying tonemap and verification settings (e.g., enable_tonemap, overlay_text_template, overlay_enabled, verify_enabled, strict, tonemap preset/parameters).
        enable_overlay (bool): Runtime override to enable or disable overlay generation.
        enable_verification (bool): Runtime override to enable or disable verification.
        logger_override (Optional[logging.Logger]): Logger to use instead of the module logger.
        warning_sink (Optional[List[str]]): Optional list to which the function will append human-readable warning messages produced during frame selection/verification.
        stored_source_props (Optional[Mapping[str, Any]]): Cached pre-trim frame props from the source clip used to rehydrate HDR metadata before normalization.

    Returns:
        ClipProcessResult: Container with the processed clip, tonemap metadata (TonemapInfo), optional overlay text, optional verification results (VerificationResult), and a snapshot of source frame properties.

    Raises:
        ClipProcessError: If VapourSynth core/resize namespaces or required resize methods are missing, if clip has no associated core, or if verification fails in strict mode; also used for other processing failures.
    """

    log = logger_override or logger
    base_props = _snapshot_frame_props(clip)
    merged_props = dict(base_props)
    if stored_source_props:
        merged_props = dict(stored_source_props)
        merged_props.update(base_props)
        clip = _apply_frame_props_dict(clip, merged_props)
    source_props = dict(merged_props)
    original_props = dict(source_props)
    clip, source_props, color_tuple = normalise_color_metadata(
        clip,
        source_props,
        color_cfg=cfg,
        file_name=file_name,
        warning_sink=warning_sink,
    )
    debug_artifacts: Optional[ColorDebugArtifacts] = None
    if debug_color:
        debug_artifacts = ColorDebugArtifacts(
            normalized_clip=clip,
            normalized_props=dict(source_props),
            original_props=original_props,
            color_tuple=color_tuple,
        )
    vs_module = _get_vapoursynth_module()
    core = getattr(clip, "core", None)
    if core is None:
        core = getattr(vs_module, "core", None)
    if core is None:
        raise ClipProcessError("Clip has no associated VapourSynth core")

    tonemap_settings = _resolve_tonemap_settings(cfg, props=source_props)
    preset = tonemap_settings.preset
    tone_curve = tonemap_settings.tone_curve
    target_nits = tonemap_settings.target_nits
    dpd = int(tonemap_settings.dynamic_peak_detection)
    dst_min = tonemap_settings.dst_min_nits
    post_gamma_cfg_enabled = bool(getattr(cfg, "post_gamma_enable", False))
    post_gamma_value = float(getattr(cfg, "post_gamma", 1.0))
    overlay_enabled = enable_overlay and bool(getattr(cfg, "overlay_enabled", True)) and not debug_color
    verify_enabled = enable_verification and bool(getattr(cfg, "verify_enabled", True))
    strict = bool(getattr(cfg, "strict", False))

    matrix_in, transfer_in, primaries_in, color_range_in = color_tuple
    tonemap_enabled = bool(getattr(cfg, "enable_tonemap", True))
    is_hdr_source = _props_signal_hdr(source_props)
    is_hdr = tonemap_enabled and is_hdr_source

    range_limited = getattr(vs_module, "RANGE_LIMITED", 1)
    range_full = getattr(vs_module, "RANGE_FULL", 0)

    tonemap_reason = None
    if not is_hdr:
        if not tonemap_enabled and is_hdr_source:
            tonemap_reason = "Tonemap disabled"
        elif not is_hdr_source:
            tonemap_reason = "SDR source"
        else:
            tonemap_reason = "Tonemap bypass"

    source_range_value: Optional[int]
    try:
        source_range_value = int(color_range_in) if color_range_in is not None else None
    except Exception:
        source_range_value = None
    if source_range_value not in (range_full, range_limited):
        source_range_value = None

    tonemap_info = TonemapInfo(
        applied=False,
        tone_curve=None,
        dpd=dpd,
        target_nits=target_nits,
        dst_min_nits=dst_min,
        src_csp_hint=None,
        reason=tonemap_reason,
        output_color_range=source_range_value,
        range_detection="source_props" if source_range_value is not None else None,
        knee_offset=tonemap_settings.knee_offset,
        dpd_preset=tonemap_settings.dpd_preset,
        dpd_black_cutoff=tonemap_settings.dpd_black_cutoff if dpd else 0.0,
        post_gamma=post_gamma_value if post_gamma_cfg_enabled else 1.0,
        post_gamma_enabled=False,
        smoothing_period=tonemap_settings.smoothing_period,
        scene_threshold_low=tonemap_settings.scene_threshold_low,
        scene_threshold_high=tonemap_settings.scene_threshold_high,
        percentile=tonemap_settings.percentile,
        contrast_recovery=tonemap_settings.contrast_recovery,
        metadata=tonemap_settings.metadata,
        use_dovi=tonemap_settings.use_dovi,
        visualize_lut=tonemap_settings.visualize_lut,
        show_clipping=tonemap_settings.show_clipping,
    )
    overlay_text = None
    verification: Optional[VerificationResult] = None

    if not is_hdr:
        log.info(
            "[TM BYPASS] %s reason=%s Matrix=%s Transfer=%s Primaries=%s Range=%s",
            file_name,
            tonemap_reason,
            matrix_in,
            transfer_in,
            primaries_in,
            color_range_in,
        )
        return ClipProcessResult(
            clip=clip,
            tonemap=tonemap_info,
            overlay_text=overlay_text,
            verification=None,
            source_props=source_props,
            debug=debug_artifacts,
        )

    resize_ns = getattr(core, "resize", None)
    if resize_ns is None:
        raise ClipProcessError("VapourSynth core missing resize namespace")
    spline36 = getattr(resize_ns, "Spline36", None)
    if not callable(spline36):
        raise ClipProcessError("VapourSynth resize.Spline36 is unavailable")

    log.info(
        "[TM INPUT] %s Matrix=%s Transfer=%s Primaries=%s Range=%s",
        file_name,
        matrix_in,
        transfer_in,
        primaries_in,
        color_range_in,
    )

    rgb16 = spline36(
        clip,
        format=getattr(vs_module, "RGB48"),
        matrix_in=matrix_in if matrix_in is not None else 1,
        transfer_in=transfer_in if transfer_in is not None else None,
        primaries_in=primaries_in if primaries_in is not None else None,
        range_in=color_range_in if color_range_in is not None else range_limited,
        dither_type="error_diffusion",
    )
    rgb16 = _normalize_rgb_props(rgb16, transfer_in, primaries_in)

    src_hint = _deduce_src_csp_hint(transfer_in, primaries_in)
    tonemapped = _tonemap_with_retries(
        core,
        rgb16,
        tone_curve=tone_curve,
        target_nits=target_nits,
        dst_min=dst_min,
        dpd=dpd,
        knee_offset=tonemap_settings.knee_offset,
        dpd_preset=tonemap_settings.dpd_preset,
        dpd_black_cutoff=tonemap_settings.dpd_black_cutoff,
        smoothing_period=tonemap_settings.smoothing_period,
        scene_threshold_low=tonemap_settings.scene_threshold_low,
        scene_threshold_high=tonemap_settings.scene_threshold_high,
        percentile=tonemap_settings.percentile,
        contrast_recovery=tonemap_settings.contrast_recovery,
        metadata=tonemap_settings.metadata,
        use_dovi=tonemap_settings.use_dovi,
        visualize_lut=tonemap_settings.visualize_lut,
        show_clipping=tonemap_settings.show_clipping,
        src_hint=src_hint,
        file_name=file_name,
    )

    tonemapped = _apply_set_frame_prop(
        tonemapped,
        prop="_Tonemapped",
        data=f"placebo:{tone_curve},dpd={dpd},dst_max={target_nits}",
    )
    tonemapped = _normalize_rgb_props(tonemapped, transfer=1, primaries=1)
    applied_post_gamma = False
    if post_gamma_cfg_enabled:
        tonemapped, applied_post_gamma = _apply_post_gamma_levels(
            core,
            tonemapped,
            gamma=post_gamma_value,
            file_name=file_name,
            log=log,
        )

    detected_range, detection_source = _detect_rgb_color_range(
        core,
        tonemapped,
        log=log,
        label=file_name,
    )

    effective_range: Optional[int] = detected_range
    fallback_source = detection_source
    if (
        effective_range is None
        and color_range_in is not None
        and color_range_in in (range_full, range_limited)
    ):
        try:
            effective_range = int(color_range_in)
        except Exception:
            effective_range = None
        else:
            fallback_source = fallback_source or "source_props"

    source_range_int: Optional[int] = None
    if color_range_in is not None and color_range_in in (range_full, range_limited):
        try:
            source_range_int = int(color_range_in)
        except Exception:
            source_range_int = None

    if effective_range is not None:
        range_value = int(effective_range)
        changed_from_source = (
            source_range_int is not None and source_range_int != range_value
        )
        if (
            changed_from_source
            and source_range_int == range_limited
            and range_value == range_full
        ):
            log.info(
                "[TM RANGE] %s plane-stats suggested full range; retaining limited metadata",
                file_name,
            )
            fallback_source = (fallback_source or "plane_stats") + "_conflict"
            assert source_range_int is not None
            effective_range = source_range_int
            range_value = int(source_range_int)
            changed_from_source = False
        tonemapped = _apply_set_frame_prop(
            tonemapped,
            prop="_ColorRange",
            intval=range_value,
        )
        if changed_from_source:
            log.info(
                "[TM RANGE] %s remapping colour range %s\u2192%s",
                file_name,
                color_range_in,
                effective_range,
            )
            if source_range_int is not None:
                tonemapped = _apply_set_frame_prop(
                    tonemapped,
                    prop="_SourceColorRange",
                    intval=source_range_int,
                )

    tonemap_info = TonemapInfo(
        applied=True,
        tone_curve=tone_curve,
        dpd=dpd,
        target_nits=target_nits,
        dst_min_nits=dst_min,
        src_csp_hint=src_hint,
        reason=None,
        output_color_range=effective_range,
        range_detection=fallback_source,
        knee_offset=tonemap_settings.knee_offset,
        dpd_preset=tonemap_settings.dpd_preset,
        dpd_black_cutoff=tonemap_settings.dpd_black_cutoff if dpd else 0.0,
        post_gamma=post_gamma_value if applied_post_gamma else 1.0,
        post_gamma_enabled=applied_post_gamma,
        smoothing_period=tonemap_settings.smoothing_period,
        scene_threshold_low=tonemap_settings.scene_threshold_low,
        scene_threshold_high=tonemap_settings.scene_threshold_high,
        percentile=tonemap_settings.percentile,
        contrast_recovery=tonemap_settings.contrast_recovery,
        metadata=tonemap_settings.metadata,
        use_dovi=tonemap_settings.use_dovi,
        visualize_lut=tonemap_settings.visualize_lut,
        show_clipping=tonemap_settings.show_clipping,
    )

    overlay_template = str(
        getattr(
            cfg,
            "overlay_text_template",
            "Tonemapping Algorithm: {tone_curve} dpd = {dynamic_peak_detection} dst = {target_nits} nits",
        )
    )
    if overlay_enabled:
        overlay_text = _format_overlay_text(
            overlay_template,
            tone_curve=tone_curve,
            dpd=dpd,
            target_nits=target_nits,
            preset=preset,
            dst_min_nits=dst_min,
            knee_offset=tonemap_settings.knee_offset,
            dpd_preset=tonemap_settings.dpd_preset,
            dpd_black_cutoff=tonemap_settings.dpd_black_cutoff if dpd else 0.0,
            post_gamma=post_gamma_value,
            post_gamma_enabled=applied_post_gamma,
            smoothing_period=tonemap_settings.smoothing_period,
            scene_threshold_low=tonemap_settings.scene_threshold_low,
            scene_threshold_high=tonemap_settings.scene_threshold_high,
            percentile=tonemap_settings.percentile,
            contrast_recovery=tonemap_settings.contrast_recovery,
            metadata=tonemap_settings.metadata,
            use_dovi=tonemap_settings.use_dovi,
            visualize_lut=tonemap_settings.visualize_lut,
            show_clipping=tonemap_settings.show_clipping,
            reason="HDR",
        )
        log.info("[OVERLAY] %s using text '%s'", file_name, overlay_text)

    log.info(
        "[TM APPLIED] %s curve=%s dpd=%d dst_max=%.2f hint=%s",
        file_name,
        tone_curve,
        dpd,
        target_nits,
        src_hint,
    )

    if verify_enabled:
        fps_num = getattr(tonemapped, "fps_num", None)
        fps_den = getattr(tonemapped, "fps_den", None)
        fps = (
            (fps_num / fps_den)
            if isinstance(fps_num, int)
            and isinstance(fps_den, int)
            and fps_den
            else 0.0
        )
        frame_idx, auto = _pick_verify_frame(
            tonemapped,
            cfg,
            fps=fps,
            file_name=file_name,
            warning_sink=warning_sink,
        )
        try:
            naive = spline36(
                clip,
                format=getattr(vs_module, "RGB24"),
                matrix_in=matrix_in if matrix_in is not None else 1,
                transfer_in=transfer_in if transfer_in is not None else None,
                primaries_in=primaries_in if primaries_in is not None else None,
                range_in=color_range_in if color_range_in is not None else range_limited,
                transfer=1,
                primaries=1,
                range=range_full,
                dither_type="error_diffusion",
            )
            point = getattr(resize_ns, "Point", None)
            if not callable(point):
                raise ClipProcessError("VapourSynth resize.Point is unavailable")
            tm_rgb24 = point(
                tonemapped,
                format=getattr(vs_module, "RGB24"),
                range=range_full,
                dither_type="error_diffusion",
            )
            verification = _compute_verification(
                core,
                tm_rgb24,
                naive,
                frame_idx,
                auto_selected=auto,
            )
            log.info(
                "[VERIFY] %s frame=%d avg=%.4f max=%.4f vs naive SDR",
                file_name,
                verification.frame,
                verification.average,
                verification.maximum,
            )
        except Exception as exc:
            message = f"Verification failed for '{file_name}': {exc}"
            log.error("[VERIFY] %s", message)
            if strict:
                raise ClipProcessError(message) from exc

    return ClipProcessResult(
        clip=tonemapped,
        tonemap=tonemap_info,
        overlay_text=overlay_text,
        verification=verification,
        source_props=source_props,
        debug=debug_artifacts,
    )

__all__ = [
    "TonemapInfo",
    "VerificationResult",
    "ColorDebugArtifacts",
    "TonemapSettings",
    "ClipProcessResult",
    "ClipProcessError",
    "_TONEMAP_UNSUPPORTED_KWARGS",
    "_parse_unexpected_kwarg",
    "_call_tonemap_function",
    "_apply_post_gamma_levels",
    "_tonemap_with_retries",
    "_TONEMAP_PRESETS",
    "_METADATA_NAME_TO_CODE",
    "_resolve_tonemap_settings",
    "resolve_effective_tonemap",
    "_format_overlay_text",
    "_pick_verify_frame",
    "_compute_verification",
    "process_clip_for_screenshot",
    "_normalize_rgb_props",
    "_deduce_src_csp_hint",
    "_apply_set_frame_prop",
]
