"""Persistence logic for audio alignment."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, cast

from src import audio_alignment
from src.frame_compare.cli_runtime import CLIAppError

if TYPE_CHECKING:
    from .models import AudioMeasurementDetail


def resolve_subdir(root: Path, relative: str, *, purpose: str, allow_absolute: bool = False) -> Path:
    """Delegate to preflight.resolve_subdir without creating an import cycle."""
    from src.frame_compare import preflight as _preflight

    return _preflight.resolve_subdir(root, relative, purpose=purpose, allow_absolute=allow_absolute)


def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        float_value = float(value)
        if math.isnan(float_value):
            return None
        return float_value
    return None


def _safe_int(value: object) -> int | None:
    if isinstance(value, (int, float)):
        float_value = float(value)
        if math.isnan(float_value):
            return None
        return int(float_value)
    return None


def load_existing_entries(offsets_path: Path) -> tuple[str | None, Dict[str, Mapping[str, Any]]]:
    try:
        reference_name, existing_entries_raw = audio_alignment.load_offsets(offsets_path)
    except audio_alignment.AudioAlignmentError as exc:
        raise CLIAppError(
            f"Failed to read audio offsets file: {exc}",
            rich_message=f"[red]Failed to read audio offsets file:[/red] {exc}",
        ) from exc
    normalized_entries: Dict[str, Mapping[str, Any]] = {}
    if isinstance(existing_entries_raw, MappingABC):
        raw_entries = cast(Mapping[Any, Any], existing_entries_raw)
        for key_obj, value in raw_entries.items():
            if not isinstance(value, MappingABC):
                continue
            entry_mapping = cast(Mapping[Any, Any], value)
            normalized_entries[str(key_obj)] = {
                str(sub_key): sub_value for sub_key, sub_value in entry_mapping.items()
            }

    return (
        reference_name,
        normalized_entries,
    )


def extract_suggestion_hints(
    entries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, tuple[int | None, float | None]]:
    hints: Dict[str, tuple[int | None, float | None]] = {}
    for name, entry in entries.items():
        frames_hint = _safe_int(entry.get("suggested_frames"))
        seconds_hint = _safe_float(entry.get("suggested_seconds"))
        if frames_hint is None and seconds_hint is None:
            continue
        hints[name] = (frames_hint, seconds_hint)
    return hints


def apply_suggestion_hints_to_details(
    detail_map: Dict[str, AudioMeasurementDetail],
    hints: Mapping[str, tuple[int | None, float | None]],
) -> None:
    for clip_name, (frames_hint, seconds_hint) in hints.items():
        detail = detail_map.get(clip_name)
        if detail is None:
            continue
        if frames_hint is not None:
            detail.frames = int(frames_hint)
        if seconds_hint is not None:
            detail.offset_seconds = float(seconds_hint)
