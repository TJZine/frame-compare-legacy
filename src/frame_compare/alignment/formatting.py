"""Formatting and output logic for audio alignment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, cast, TYPE_CHECKING

from src.frame_compare import vspreview
from src.frame_compare.alignment_helpers import derive_frame_hint
from src.frame_compare.cli_runtime import CLIAppError, ensure_audio_alignment_block
from src.frame_compare.config_helpers import coerce_config_flag as _coerce_config_flag

if TYPE_CHECKING:
    from src.datatypes import AppConfig
    from src.frame_compare.cli_runtime import CliOutputManagerProtocol, ClipPlan, JsonTail
    from .models import AudioAlignmentDisplayData, AudioAlignmentSummary

logger = logging.getLogger(__name__)

_launch_vspreview = vspreview.launch


def format_alignment_output(
    plans: Sequence[ClipPlan],
    summary: AudioAlignmentSummary | None,
    display: AudioAlignmentDisplayData | None,
    *,
    cfg: AppConfig,
    root: Path,
    reporter: CliOutputManagerProtocol,
    json_tail: JsonTail,
    vspreview_mode: str,
    collected_warnings: List[str] | None = None,
) -> None:
    """
    Populate json_tail/audio layout data and optionally launch VSPreview.
    """

    audio_block = ensure_audio_alignment_block(json_tail)

    vspreview_target_plan: ClipPlan | None = None
    vspreview_suggested_frames_value: int | None = None
    vspreview_suggested_seconds_value = 0.0
    if summary is not None:
        for plan in plans:
            if plan is summary.reference_plan:
                continue
            vspreview_target_plan = plan
            break
        if vspreview_target_plan is not None:
            clip_key = vspreview_target_plan.path.name
            vspreview_suggested_frames_value = derive_frame_hint(summary, clip_key)
            measurement_seconds: Optional[float] = None
            if summary.measured_offsets:
                detail = summary.measured_offsets.get(clip_key)
                if detail and detail.offset_seconds is not None:
                    measurement_seconds = float(detail.offset_seconds)
            if measurement_seconds is None and summary.measurements:
                measurement_lookup = {
                    measurement.file.name: measurement for measurement in summary.measurements
                }
                measurement = measurement_lookup.get(clip_key)
                if measurement is not None and measurement.offset_seconds is not None:
                    measurement_seconds = float(measurement.offset_seconds)
            if measurement_seconds is not None:
                vspreview_suggested_seconds_value = measurement_seconds

    json_tail["vspreview_mode"] = vspreview_mode
    json_tail["suggested_frames"] = (
        int(vspreview_suggested_frames_value)
        if vspreview_suggested_frames_value is not None
        else None
    )
    json_tail["suggested_seconds"] = float(round(vspreview_suggested_seconds_value, 6))

    vspreview_enabled_for_session = _coerce_config_flag(cfg.audio_alignment.use_vspreview)
    if vspreview_enabled_for_session and summary is not None and summary.suggestion_mode:
        try:
            _launch_vspreview(
                plans,
                summary,
                display,
                cfg,
                root,
                reporter,
                json_tail,
            )
        except CLIAppError:
            raise
        except Exception as exc:
            logger.warning(
                "VSPreview launch failed: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            reporter.warn(f"VSPreview launch failed: {exc}")

    if display is not None:
        audio_block["offsets_filename"] = display.offsets_file_line.split(": ", 1)[-1]
        audio_block["reference_stream"] = display.json_reference_stream
        target_streams: dict[str, object] = dict(display.json_target_streams.items())
        audio_block["target_stream"] = target_streams
        offsets_sec_source = dict(display.json_offsets_sec)
        offsets_frames_source = dict(display.json_offsets_frames)
        if (
            not offsets_sec_source
            and summary is not None
            and summary.measured_offsets
        ):
            offsets_sec_source = {}
            offsets_frames_source = {}
            for detail in summary.measured_offsets.values():
                if detail.offset_seconds is not None:
                    offsets_sec_source[detail.label] = float(detail.offset_seconds)
                if detail.frames is not None:
                    offsets_frames_source[detail.label] = int(detail.frames)
        audio_block["offsets_sec"] = {key: float(value) for key, value in offsets_sec_source.items()}
        audio_block["offsets_frames"] = {
            key: int(value) for key, value in offsets_frames_source.items()
        }
        stream_lines_output = list(display.stream_lines)
        if display.estimation_line:
            stream_lines_output.append(display.estimation_line)
        audio_block["stream_lines"] = stream_lines_output
        audio_block["stream_lines_text"] = "\n".join(stream_lines_output) if stream_lines_output else ""
        offset_lines_output = list(display.offset_lines)
        audio_block["offset_lines"] = offset_lines_output
        audio_block["offset_lines_text"] = "\n".join(offset_lines_output) if offset_lines_output else ""
        measurement_source = dict(display.measurements)
        if (
            not measurement_source
            and summary is not None
            and summary.measured_offsets
        ):
            measurement_source = {
                detail.label: detail for detail in summary.measured_offsets.values()
            }
        measurements_output: dict[str, dict[str, object]] = {}
        for label, detail in measurement_source.items():
            measurements_output[label] = {
                "stream": detail.stream,
                "seconds": detail.offset_seconds,
                "frames": detail.frames,
                "correlation": detail.correlation,
                "status": detail.status,
                "applied": detail.applied,
                "note": detail.note,
            }
        audio_block["measurements"] = measurements_output
        if display.manual_trim_lines:
            audio_block["manual_trim_summary"] = list(display.manual_trim_lines)
        else:
            audio_block["manual_trim_summary"] = []
        if display.warnings and collected_warnings is not None:
            collected_warnings.extend(display.warnings)
    else:
        audio_block["reference_stream"] = None
        audio_block["target_stream"] = cast(dict[str, object], {})
        audio_block["offsets_sec"] = cast(dict[str, object], {})
        audio_block["offsets_frames"] = cast(dict[str, object], {})
        audio_block["manual_trim_summary"] = []
        audio_block["stream_lines"] = []
        audio_block["stream_lines_text"] = ""
        audio_block["offset_lines"] = []
        audio_block["offset_lines_text"] = ""
        audio_block["measurements"] = {}

    audio_block["enabled"] = bool(cfg.audio_alignment.enable)
    audio_block["suggestion_mode"] = bool(summary.suggestion_mode if summary else False)
    audio_block["suggested_frames"] = dict(summary.suggested_frames) if summary else {}
    audio_block["manual_trim_starts"] = dict(summary.manual_trim_starts) if summary else {}
    audio_block["vspreview_manual_offsets"] = (
        dict(summary.vspreview_manual_offsets) if summary else {}
    )
    audio_block["vspreview_manual_deltas"] = (
        dict(summary.vspreview_manual_deltas) if summary else {}
    )
    if (
        summary is not None
        and summary.reference_plan.path.name in summary.vspreview_manual_offsets
    ):
        audio_block["vspreview_reference_trim"] = int(
            summary.vspreview_manual_offsets[summary.reference_plan.path.name]
        )
