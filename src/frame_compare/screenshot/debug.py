from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    cast,
)

from src.datatypes import RGBDither
from src.frame_compare import vs as vs_core
from src.frame_compare.screenshot.helpers import ensure_rgb24, map_fpng_compression


class ColorDebugState:
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
        self._compression_value = map_fpng_compression(compression_level)
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
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ColorDebug")

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
            props_map = dict(props or vs_core.snapshot_frame_props(clip))
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
        
        def _execute_write() -> None:
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
            
            try:
                _, _, _, debug_range = vs_core.resolve_color_metadata(props)
                rgb_clip = ensure_rgb24(
                    self.core,
                    clip,
                    frame_idx,
                    source_props=props,
                    rgb_dither=self._rgb_dither,
                    target_range=int(debug_range) if debug_range is not None else None,
                )
                job = self._writer(rgb_clip, str(path), compression=self._compression_value, overwrite=True)
                job.get_frame(frame_idx)
            except Exception as e:
                self._logger.warning(f"Failed to write debug frame: {e}")

        self._executor.submit(_execute_write)

    def close(self) -> None:
        """Shutdown the internal executor."""
        self._executor.shutdown(wait=False)
