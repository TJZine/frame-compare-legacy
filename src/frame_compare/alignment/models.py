"""Data models for audio alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from src.audio_alignment import AlignmentMeasurement
    from src.frame_compare.cli_runtime import ClipPlan


def str_int_dict_factory() -> dict[str, int]:
    return {}


def str_float_dict_factory() -> dict[str, float]:
    return {}


def measurement_dict_factory() -> dict[str, "AudioMeasurementDetail"]:
    return {}


def str_list_factory() -> list[str]:
    return []


@dataclass
class AudioAlignmentSummary:
    """
    Bundle of audio-alignment details used for reporting and persistence.
    """

    offsets_path: Path
    reference_name: str
    measurements: Sequence["AlignmentMeasurement"]
    applied_frames: dict[str, int]
    baseline_shift: int
    statuses: dict[str, str]
    reference_plan: ClipPlan
    final_adjustments: dict[str, int]
    swap_details: dict[str, str]
    suggested_frames: dict[str, int] = field(default_factory=str_int_dict_factory)
    suggestion_mode: bool = False
    manual_trim_starts: dict[str, int] = field(default_factory=str_int_dict_factory)
    vspreview_manual_offsets: dict[str, int] = field(default_factory=str_int_dict_factory)
    vspreview_manual_deltas: dict[str, int] = field(default_factory=str_int_dict_factory)
    measured_offsets: dict[str, "AudioMeasurementDetail"] = field(default_factory=measurement_dict_factory)


@dataclass
class AudioMeasurementDetail:
    """Snapshot of an audio alignment measurement for CLI/JSON reporting."""

    label: str
    stream: str
    offset_seconds: Optional[float]
    frames: Optional[int]
    correlation: Optional[float]
    status: str
    applied: bool
    note: Optional[str] = None


@dataclass
class AudioAlignmentDisplayData:
    """
    Pre-rendered data used to present audio alignment results in the CLI.
    """

    stream_lines: list[str]
    estimation_line: Optional[str]
    offset_lines: list[str]
    offsets_file_line: str
    json_reference_stream: Optional[str]
    json_target_streams: dict[str, str]
    json_offsets_sec: dict[str, float]
    json_offsets_frames: dict[str, int]
    warnings: list[str]
    preview_paths: list[str] = field(default_factory=str_list_factory)
    inspection_paths: list[str] = field(default_factory=str_list_factory)
    confirmation: Optional[str] = None
    correlations: dict[str, float] = field(default_factory=str_float_dict_factory)
    threshold: float = 0.0
    manual_trim_lines: list[str] = field(default_factory=str_list_factory)
    measurements: dict[str, AudioMeasurementDetail] = field(default_factory=measurement_dict_factory)
