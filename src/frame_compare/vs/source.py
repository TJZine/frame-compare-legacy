"""Source plugin discovery and clip initialisation helpers."""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .env import _SOURCE_PREFERENCE, ClipInitError, _get_vapoursynth_module  # pyright: ignore[reportPrivateUsage]
from .props import (  # pyright: ignore[reportPrivateUsage]
    _apply_frame_props_dict,  # pyright: ignore[reportPrivateUsage]
    _ensure_std_namespace,  # pyright: ignore[reportPrivateUsage]
    snapshot_frame_props,  # pyright: ignore[reportPrivateUsage]
)

logger = logging.getLogger("src.frame_compare.vs.source")
_SOURCE_PLUGIN_FUNCS = {"lsmas": "LWLibavSource", "ffms2": "Source"}


_CACHE_SUFFIX = {"lsmas": ".lwi", "ffms2": ".ffindex"}
_HDR_PROP_BASE_NAMES = {"matrix", "primaries", "transfer", "colorrange"}


class VSPluginError(ClipInitError):
    """Base class for VapourSynth plugin discovery failures."""

    def __init__(self, plugin: str, message: str) -> None:
        super().__init__(message)
        self.plugin = plugin


class VSPluginMissingError(VSPluginError):
    """Raised when a required VapourSynth plugin is absent."""


class VSPluginWrongArchError(VSPluginError):
    """Raised when a plugin binary targets the wrong CPU architecture."""


class VSPluginDepMissingError(VSPluginError):
    """Raised when a plugin has unresolved shared library dependencies."""

    def __init__(self, plugin: str, dependency: str | None, message: str) -> None:
        super().__init__(plugin, message)
        self.dependency = dependency


class VSPluginBadBinaryError(VSPluginError):
    """Raised when a plugin binary is malformed or lacks an entry point."""


class VSSourceUnavailableError(ClipInitError):
    """Raised when no usable source plugin is available."""

    def __init__(self, message: str, *, errors: Mapping[str, VSPluginError] | None = None) -> None:
        super().__init__(message)
        self.errors: Mapping[str, VSPluginError] = dict(errors or {})


def _resolve_core(core: Optional[Any]) -> Any:
    if core is not None:
        return core
    vs_module = _get_vapoursynth_module()
    module_core = getattr(vs_module, "core", None)
    if callable(module_core):
        try:
            resolved = module_core()
            if resolved is not None:
                return resolved
        except (TypeError, RuntimeError):
            pass

    if module_core is not None and not callable(module_core):
        return module_core
    get_core = getattr(vs_module, "get_core", None)
    if callable(get_core):
        resolved = get_core()
        if resolved is not None:
            return resolved
    fallback_core = getattr(vs_module, "core", None)
    if fallback_core is None:
        raise ClipInitError("VapourSynth core is not available on this interpreter")
    return fallback_core


def _build_source_order() -> list[str]:
    """Return the ordered list of source plugins to try."""

    if _SOURCE_PREFERENCE == "ffms2":
        return ["ffms2", "lsmas"]
    return ["lsmas", "ffms2"]


def _build_plugin_missing_message(plugin: str) -> str:
    base = f"VapourSynth plugin '{plugin}' is not available on the current core."
    if plugin == "lsmas":
        return (
            base
            + " Install L-SMASH-Works (LWLibavSource) built for this architecture and place"
            " it in ~/Library/VapourSynth/plugins or /opt/homebrew/lib/vapoursynth."
        )
    if plugin == "ffms2":
        return (
            base
            + " Install FFMS2 and ensure the plugin dylib resides in a VapourSynth plugin"
            " directory (e.g. via 'brew install vapoursynth-ffms2')."
        )
    return base


def _resolve_source_callable(core: Any, plugin: str) -> Callable[..., Any]:
    namespace = getattr(core, plugin, None)
    if namespace is None:
        raise VSPluginMissingError(plugin, _build_plugin_missing_message(plugin))
    func_name = _SOURCE_PLUGIN_FUNCS.get(plugin)
    if not func_name:
        raise VSPluginBadBinaryError(plugin, f"No loader defined for plugin '{plugin}'")
    source = getattr(namespace, func_name, None)
    if not callable(source):
        raise VSPluginBadBinaryError(
            plugin,
            f"{plugin}.{func_name} is not callable. The plugin may have failed to load or"
            " is not a VapourSynth binary compatible with this release.",
        )
    return source


def _cache_path_for(cache_root: Path, base_name: str, plugin: str) -> Path:
    suffix = _CACHE_SUFFIX.get(plugin, ".lwi")
    return cache_root / f"{base_name}{suffix}"


_DEPENDENCY_PATTERN = re.compile(r"Library not loaded: (?P<path>\S+)")


_ENTRY_POINT_PATTERN = re.compile(r"(vapoursynthplugininit|entry point)", re.IGNORECASE)


_WRONG_ARCH_PATTERN = re.compile(r"wrong architecture", re.IGNORECASE)


def _extract_major_version(library_name: str) -> str | None:
    match = re.search(r"\.(\d+)(?:\.dylib)?$", library_name)
    if match:
        return match.group(1)
    return None


def _build_dependency_hint(plugin: str, dependency: str | None, details: str) -> str:
    parts = [
        f"{plugin} plugin could not be initialised because a dependency was missing.",
        details,
    ]
    if dependency:
        dep_name = Path(dependency).name
        parts.append(f"Missing library: {dep_name} ({dependency}).")
        lowered = dep_name.lower()
        major = _extract_major_version(dep_name)
        if "libav" in lowered:
            if major:
                parts.append(
                    f"Install an FFmpeg build that provides {dep_name} (major {major})"
                    " or adjust the plugin's install_name to point at /opt/homebrew/lib."
                )
            else:
                parts.append(
                    "Install a matching FFmpeg build (e.g. via Homebrew) and ensure the"
                    " dylibs are discoverable."
                )
        if "liblsmash" in lowered:
            parts.append(
                "Install liblsmash (brew install l-smash) or ensure DYLD_LIBRARY_PATH"
                " includes the directory that provides it."
            )
    else:
        parts.append(
            "Check that the plugin binary and its dependencies are located in a"
            " VapourSynth plugin directory."
        )
    return " ".join(parts)


def _build_wrong_arch_message(plugin: str, details: str) -> str:
    return (
        f"{plugin} plugin failed to load because the binary targets a different CPU"
        f" architecture. {details} Install an arm64-compatible build or run under"
        " Rosetta with matching x86_64 dependencies."
    )


def _classify_plugin_exception(plugin: str, exc: Exception) -> VSPluginError | None:
    message = str(exc)
    lower = message.lower()
    if _WRONG_ARCH_PATTERN.search(lower):
        return VSPluginWrongArchError(plugin, _build_wrong_arch_message(plugin, message))
    match = _DEPENDENCY_PATTERN.search(message)
    if match:
        dependency = match.group("path")
        return VSPluginDepMissingError(
            plugin,
            dependency,
            _build_dependency_hint(plugin, dependency, message),
        )
    if "image not found" in lower and "dlopen" in lower:
        return VSPluginDepMissingError(
            plugin,
            None,
            _build_dependency_hint(plugin, None, message),
        )
    if _ENTRY_POINT_PATTERN.search(lower) or "no entry point" in lower:
        return VSPluginBadBinaryError(
            plugin,
            f"{plugin} plugin appears to be an incompatible binary. {message} Ensure"
            " it exports VapourSynthPluginInit2 and matches this VapourSynth release.",
        )
    return None


def _open_clip_with_sources(
    core: Any,
    path: str,
    cache_root: Path,
    *,
    indexing_notifier: Optional[Callable[[str], None]] = None,
) -> Any:
    order = _build_source_order()
    errors: dict[str, VSPluginError] = {}
    base_name = Path(path).name
    for plugin in order:
        try:
            source = _resolve_source_callable(core, plugin)
        except VSPluginError as plugin_error:
            logger.warning(
                "VapourSynth plugin '%s' unavailable: %s", plugin, plugin_error
            )
            errors[plugin] = plugin_error
            continue

        cache_path = _cache_path_for(cache_root, base_name, plugin)
        try:
            if not cache_path.exists():
                logger.info("[CACHE] Indexing %s via %s", base_name, plugin)
                if indexing_notifier is not None:
                    indexing_notifier(base_name)
            return source(path, cachefile=str(cache_path))
        except (ClipInitError, VSPluginError, RuntimeError, ValueError) as exc:
            classified = _classify_plugin_exception(plugin, exc)
            if classified:
                errors[plugin] = classified
            else:
                errors[plugin] = VSPluginError(plugin, f"Error initializing {plugin}: {exc}")

    if errors:
        detail = "; ".join(f"{name}: {err}" for name, err in errors.items())
        raise VSSourceUnavailableError(
            (
                f"No usable VapourSynth source plugin was able to open '{path}'. Tried"
                f" {', '.join(order)}. Details: {detail}"
            ),
            errors=errors,
        )
    raise VSSourceUnavailableError(
        f"No VapourSynth source plugins were available to open '{path}'.",
        errors=errors,
    )


def _slice_clip(clip: Any, *, start: Optional[int] = None, end: Optional[int] = None) -> Any:
    try:
        if start is not None and end is not None:
            return clip[start:end]
        if start is not None:
            return clip[start:]
        if end is not None:
            return clip[:end]
    except (IndexError, ValueError, TypeError) as exc:
        raise ClipInitError("Failed to apply trim to clip") from exc
    return clip


def _extend_with_blank(
    clip: Any,
    core: Any,
    length: int,
    *,
    frame_props: Mapping[str, Any] | None = None,
) -> Any:
    std_ns = getattr(core, "std", None)
    if std_ns is None:
        raise ClipInitError("VapourSynth core missing std namespace for BlankClip")
    blank_clip = getattr(std_ns, "BlankClip", None)
    if not callable(blank_clip):
        raise ClipInitError("std.BlankClip is unavailable on the VapourSynth core")
    try:
        extension = blank_clip(clip, length=length)
        if frame_props:
            extension = _apply_frame_props_dict(extension, frame_props)
        return extension + clip
    except (ValueError, TypeError) as exc:
        raise ClipInitError("Failed to prepend blank frames to clip") from exc


def _apply_fps_map(clip: Any, fps_map: Tuple[int, int]) -> Any:
    std = _ensure_std_namespace(clip, ClipInitError("Clip is missing std namespace for AssumeFPS"))
    num, den = fps_map
    if den <= 0:
        raise ClipInitError("fps_map denominator must be positive")
    try:
        return std.AssumeFPS(num=num, den=den)
    except (ValueError, TypeError) as exc:
        raise ClipInitError("Failed to apply FPS mapping to clip") from exc


def init_clip(
    path: str,
    *,
    trim_start: int = 0,
    trim_end: Optional[int] = None,
    fps_map: Tuple[int, int] | None = None,
    cache_dir: Optional[str | Path] = None,
    core: Optional[Any] = None,
    indexing_notifier: Optional[Callable[[str], None]] = None,
    frame_props_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    source_frame_props_hint: Mapping[str, Any] | None = None,
) -> Any:
    """
    Initialise a VapourSynth clip for subsequent processing and optionally snapshot source frame props.

    When ``frame_props_sink`` is provided it will be invoked exactly once with a dictionary of frame
    properties captured before any trims or padding are applied so callers can persist HDR metadata.
    ``source_frame_props_hint`` allows callers to reuse previously captured props (for example from an
    earlier metadata probe) to avoid repeated frame snapshots.
    """

    resolved_core = _resolve_core(core)

    path_obj = Path(path)
    cache_root = Path(cache_dir) if cache_dir is not None else path_obj.parent
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ClipInitError(f"Failed to prepare cache directory '{cache_root}': {exc}") from exc

    try:
        clip = _open_clip_with_sources(
            resolved_core,
            str(path_obj),
            cache_root,
            indexing_notifier=indexing_notifier,
        )
    except ClipInitError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ClipInitError(f"Failed to open clip '{path}': {exc}") from exc

    try:
        if source_frame_props_hint:
            source_frame_props = dict(source_frame_props_hint)
        else:
            source_frame_props = dict(snapshot_frame_props(clip))
    except (ValueError, TypeError, KeyError):
        source_frame_props = {}
    if frame_props_sink is not None:
        try:
            frame_props_sink(dict(source_frame_props))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to record source frame props for %s", path)

    if trim_start < 0:
        padding_props = _collect_blank_extension_props(source_frame_props)
        clip = _extend_with_blank(
            clip,
            resolved_core,
            abs(int(trim_start)),
            frame_props=padding_props if padding_props else None,
        )
    elif trim_start > 0:
        clip = _slice_clip(clip, start=int(trim_start))

    if trim_end is not None and trim_end != 0:
        clip = _slice_clip(clip, end=int(trim_end))
    if fps_map is not None:
        clip = _apply_fps_map(clip, fps_map)

    clip = _maybe_inject_dovi_metadata(clip, resolved_core, str(path_obj), trim_start)
    return clip


def _collect_blank_extension_props(frame_props: Mapping[str, Any]) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    for key, value in frame_props.items():
        if _is_hdr_prop(key):
            extracted[key] = value
    return extracted


def _is_hdr_prop(key: str) -> bool:
    normalized = key.lstrip("_").lower()
    if normalized in _HDR_PROP_BASE_NAMES:
        return True
    return normalized.startswith("masteringdisplay") or normalized.startswith("contentlightlevel")


def _maybe_inject_dovi_metadata(clip: Any, core: Any, file_path: str, trim_start: int) -> Any:
    """
    Attempt to inject Dolby Vision metadata using dovi_tool.
    """
    from src.frame_compare.services.dovi_tool import dovi_tool

    if not dovi_tool.is_available():
        return clip

    try:
        print("Extracting Dolby Vision metadata... this may take a minute", file=sys.stderr)
        metadata = dovi_tool.extract_rpu_metadata(Path(file_path))
        if not metadata:
            return clip

        logger.info("Injecting DoVi metadata from dovi_tool (%d frames)", len(metadata))

        def _inject_props(n: int, f: Any) -> Any:
            # Calculate original source frame index
            if trim_start > 0:
                source_idx = n + trim_start
            elif trim_start < 0:
                source_idx = n - abs(trim_start)
            else:
                source_idx = n

            if 0 <= source_idx < len(metadata):
                data = metadata[source_idx]
                fout = f.copy()

                if "l1_avg_nits" in data:
                    fout.props["DolbyVision_L1_Average"] = float(data["l1_avg_nits"])
                if "l1_max_nits" in data:
                    fout.props["DolbyVision_L1_Maximum"] = float(data["l1_max_nits"])

                if "l2_target_nits" in data:
                    fout.props["DolbyVision_L2_TargetNits"] = float(data["l2_target_nits"])

                if "l5_left" in data:
                    fout.props["DolbyVision_L5_Left"] = int(data["l5_left"])
                if "l5_right" in data:
                    fout.props["DolbyVision_L5_Right"] = int(data["l5_right"])
                if "l5_top" in data:
                    fout.props["DolbyVision_L5_Top"] = int(data["l5_top"])
                if "l5_bottom" in data:
                    fout.props["DolbyVision_L5_Bottom"] = int(data["l5_bottom"])

                if "l6_max_cll" in data:
                    fout.props["DolbyVision_L6_MaxCLL"] = float(data["l6_max_cll"])
                if "l6_max_fall" in data:
                    fout.props["DolbyVision_L6_MaxFALL"] = float(data["l6_max_fall"])
                # Also inject RPU present flag if not already there, to trigger overlay
                if "DolbyVisionRPU" not in fout.props:
                     fout.props["DolbyVisionRPU"] = b"1" # Dummy blob to signal presence
                return fout
            return f

        return core.std.ModifyFrame(clip, clip, _inject_props)

    except (OSError, RuntimeError, ValueError, KeyError, AttributeError) as exc:
        logger.warning("Failed to inject DoVi metadata: %s", exc)
        return clip


__all__ = [
    "VSPluginError",
    "VSPluginMissingError",
    "VSPluginWrongArchError",
    "VSPluginDepMissingError",
    "VSPluginBadBinaryError",
    "VSSourceUnavailableError",
    "_SOURCE_PLUGIN_FUNCS",
    "_CACHE_SUFFIX",
    "_build_source_order",
    "_build_plugin_missing_message",
    "_resolve_source_callable",
    "_cache_path_for",
    "_DEPENDENCY_PATTERN",
    "_ENTRY_POINT_PATTERN",
    "_WRONG_ARCH_PATTERN",
    "_extract_major_version",
    "_build_dependency_hint",
    "_build_wrong_arch_message",
    "_classify_plugin_exception",
    "_open_clip_with_sources",
    "_slice_clip",
    "_extend_with_blank",
    "_apply_fps_map",
    "_resolve_core",
    "init_clip",
]
