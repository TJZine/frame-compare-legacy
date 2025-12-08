"""Helpers for picking which clip should drive downstream analysis."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Mapping, Sequence

from src.frame_compare import vs as vs_core

logger = logging.getLogger(__name__)

__all__ = ["pick_analyze_file"]


def _estimate_analysis_time(file: Path, cache_dir: Path | None) -> float:
    """Estimate the time to read two windows of frames via VapourSynth."""
    try:
        clip = vs_core.init_clip(str(file), cache_dir=str(cache_dir) if cache_dir else None)
    except (RuntimeError, ValueError, OSError, TypeError):
        return float("inf")

    try:
        total = getattr(clip, "num_frames", 0)
        if not isinstance(total, int) or total <= 1:
            return float("inf")
        read_len = 15
        while (total // 3) + 1 < read_len and read_len > 1:
            read_len -= 1

        stats = clip.std.PlaneStats()

        def _read_window(base: int) -> float:
            start = max(0, min(base, max(0, total - 1)))
            t0 = time.perf_counter()
            for j in range(read_len):
                idx = min(start + j, max(0, total - 1))
                frame = stats.get_frame(idx)
                del frame
            return time.perf_counter() - t0

        t1 = _read_window(total // 3)
        t2 = _read_window((2 * total) // 3)
        return (t1 + t2) / 2.0
    except (RuntimeError, ValueError, ZeroDivisionError, AttributeError):
        return float("inf")


def pick_analyze_file(
    files: Sequence[Path],
    metadata: Sequence[Mapping[str, str]],
    target: str | None,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve the clip selected for analysis, honouring user targets when provided."""
    if not files:
        raise ValueError("No files to analyze")
    target = (target or "").strip()
    if not target:
        logger.info("Determining which file to analyze...")
        times = [(_estimate_analysis_time(file, cache_dir), idx) for idx, file in enumerate(files)]
        times.sort(key=lambda x: x[0])
        fastest_idx = times[0][1] if times else 0
        return files[fastest_idx]

    target_lower = target.lower()

    if target.isdigit():
        idx = int(target)
        if 0 <= idx < len(files):
            return files[idx]

    metadata_len = len(metadata)
    for idx, file in enumerate(files):
        if file.name.lower() == target_lower or file.stem.lower() == target_lower:
            return file
        meta = metadata[idx] if idx < metadata_len else None
        if meta:
            for key in ("label", "release_group", "anime_title", "file_name"):
                value = str(meta.get(key) or "")
                if value and value.lower() == target_lower:
                    return file
        if target_lower == str(idx):
            return file

    return files[0]
