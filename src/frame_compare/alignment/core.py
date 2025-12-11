"""Core orchestration logic for audio alignment."""

from __future__ import annotations

import logging
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ContextManager,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    cast,
)

import click

from src import audio_alignment
from src.datatypes import AppConfig
from src.frame_compare.cli_runtime import CLIAppError, CliOutputManagerProtocol, ClipPlan
from src.frame_compare.config_helpers import coerce_config_flag as _coerce_config_flag
from src.frame_compare.layout_utils import plan_label as _plan_label
from src.frame_compare.metadata import match_override as _match_override

from .models import (
    AudioAlignmentDisplayData,
    AudioAlignmentSummary,
    AudioMeasurementDetail,
)
from .persistence import (
    apply_suggestion_hints_to_details,
    extract_suggestion_hints,
    load_existing_entries,
    resolve_subdir,
)

if TYPE_CHECKING:
    from src.audio_alignment import AlignmentMeasurement, AudioStreamInfo

logger = logging.getLogger(__name__)


def _fps_to_float(value: tuple[int, int] | None) -> float:
    """Convert an FPS tuple to a float, guarding against missing/zero denominators."""

    if value is None:
        return 0.0
    numerator, denominator = value
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _plan_fps_map(plans: Sequence[ClipPlan]) -> dict[Path, tuple[int, int]]:
    """Build a lookup with preferred FPS tuples for each plan."""

    fps_map: dict[Path, tuple[int, int]] = {}
    for plan in plans:
        for candidate in (
            plan.effective_fps,
            plan.applied_fps,
            plan.source_fps,
            plan.fps_override,
        ):
            if candidate is None:
                continue
            numerator, denominator = candidate
            if numerator <= 0 or denominator <= 0:
                continue
            fps_map[plan.path] = candidate
            break
    return fps_map


def _resolve_alignment_reference(
    plans: Sequence[ClipPlan],
    analyze_path: Path,
    reference_hint: str,
) -> ClipPlan:
    """Choose the audio alignment reference plan using optional hint and fallbacks."""
    if not plans:
        raise CLIAppError("No clips available for alignment")

    hint = (reference_hint or "").strip().lower()
    if hint:
        if hint.isdigit():
            idx = int(hint)
            if 0 <= idx < len(plans):
                return plans[idx]
        for plan in plans:
            candidates = {
                plan.path.name.lower(),
                plan.path.stem.lower(),
                (plan.metadata.get("label") or "").lower(),
            }
            if hint in candidates and hint:
                return plan

    for plan in plans:
        if plan.path == analyze_path:
            return plan
    return plans[0]


def resolve_alignment_reference(
    plans: Sequence[ClipPlan],
    analyze_path: Path,
    reference_hint: str,
) -> ClipPlan:
    return _resolve_alignment_reference(plans, analyze_path, reference_hint)


def _estimate_frames_from_seconds(
    measurement: "AlignmentMeasurement",
    plan_lookup: Mapping[str, ClipPlan],
) -> Optional[int]:
    """Derive a frame delta from the available FPS hints when frames are missing."""

    if measurement.frames is not None:
        return int(measurement.frames)

    offset_seconds = float(measurement.offset_seconds or 0.0)
    if math.isclose(offset_seconds, 0.0, abs_tol=1e-9):
        return 0

    def _from_float(fps_value: Optional[float]) -> Optional[int]:
        if fps_value and fps_value > 0:
            return int(round(offset_seconds * float(fps_value)))
        return None

    for candidate in (measurement.target_fps, measurement.reference_fps):
        derived = _from_float(candidate)
        if derived is not None:
            return derived

    plan = plan_lookup.get(measurement.file.name)
    if plan is not None:
        fps_candidates = (
            plan.effective_fps,
            plan.applied_fps,
            plan.fps_override,
            plan.source_fps,
        )
        for fps_tuple in fps_candidates:
            fps_float = _fps_to_float(fps_tuple)
            derived = _from_float(fps_float)
            if derived is not None:
                return derived
    return None


def apply_manual_offsets_logic(
    plans: Sequence[ClipPlan],
    vspreview_reuse: dict[str, int],
    display_data: AudioAlignmentDisplayData,
    plan_labels: dict[Path, str],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Apply manual offsets from VSPreview history, normalizing to avoid negative trims (padding).
    Extracted for testability.

    Returns:
        tuple[dict[str, int], dict[str, int]]: (delta_map, manual_trim_starts)
    """
    if display_data.manual_trim_lines:
        display_data.manual_trim_lines.clear()

    baseline_map = {plan.path.name: int(plan.trim_start) for plan in plans}
    delta_map: dict[str, int] = {}
    manual_trim_starts: dict[str, int] = {}

    # 1. Calculate proposed trims for all plans
    proposed_trims: dict[str, int] = {}
    for plan in plans:
        key = plan.path.name
        baseline = baseline_map.get(key, int(plan.trim_start))
        delta = int(vspreview_reuse.get(key, 0))
        proposed_trims[key] = baseline + delta
        if key in vspreview_reuse:
            delta_map[key] = delta

    logger.info(f"[ALIGN DEBUG] Raw manual offsets (vspreview_reuse): {vspreview_reuse}")
    logger.info(f"[ALIGN DEBUG] Baseline trims: {baseline_map}")
    logger.info(f"[ALIGN DEBUG] Proposed trims before normalization: {proposed_trims}")

    # 2. Normalize to ensure no negative trims (avoid padding)
    # Normalize by shifting all clips up so the minimum trim becomes 0
    # Example: offsets [-361, 0] → shift +361 → trims [0, 361]
    min_offset = min(proposed_trims.values()) if proposed_trims else 0
    shift = -min_offset if min_offset < 0 else 0

    if shift > 0:
        logger.info(f"[ALIGN DEBUG] Found negative proposed trim {min_offset}, applying global shift of {shift}f.")
    else:
        logger.info("[ALIGN DEBUG] No negative proposed trims found, no global shift applied.")

    logger.info(f"[ALIGN DEBUG] Normalizing with shift={shift}")

    # 3. Apply normalized trims
    for plan in plans:
        key = plan.path.name
        if key in proposed_trims:
            raw_offset = proposed_trims[key]
            # Calculate normalized trim
            normalized_trim = raw_offset + shift

            # Apply to plan
            # Note: We SET the new absolute trim_start (normalized_trim includes the baseline)
            original_trim = int(plan.trim_start)
            if normalized_trim != original_trim:
                plan.trim_start = normalized_trim
                plan.source_num_frames = None
                plan.has_trim_start_override = True

            manual_trim_starts[key] = int(plan.trim_start)

            baseline = baseline_map.get(key, 0)
            effective_delta = normalized_trim - baseline
            delta_map[key] = effective_delta

            label = plan_labels.get(plan.path, key)
            input_delta = proposed_trims[key] - baseline

            line = (
                f"VSPreview manual offset applied: {label} baseline {baseline}f "
                f"{input_delta:+d}f (shift {shift:+d}f) → {normalized_trim}f"
            )
            display_data.manual_trim_lines.append(line)

            logger.info(f"[ALIGN DEBUG] Applied trim to {key}: raw={raw_offset} + shift={shift} -> {normalized_trim}. Final trim_start={plan.trim_start}")

    return delta_map, manual_trim_starts


def _reuse_vspreview_manual_offsets_if_available(
    plans: Sequence[ClipPlan],
    reference: ClipPlan | None,
    vspreview_enabled: bool,
    audio_cfg: Any,
    display_data: AudioAlignmentDisplayData,
    plan_labels: Dict[Path, str],
    offsets_path: Path,
) -> AudioAlignmentSummary | None:
    if not (vspreview_enabled and reference and plans):
        return None

    try:
        _, existing_entries = load_existing_entries(offsets_path)
    except CLIAppError:
        if not audio_cfg.enable:
            return None
        raise

    vspreview_reuse: Dict[str, int] = {}
    allowed_keys = {plan.path.name for plan in plans}
    for key, value in existing_entries.items():
        entry = value
        status_obj = entry.get("status")
        note_obj = entry.get("note")
        frames_obj = entry.get("frames")
        if not isinstance(status_obj, str) or not isinstance(frames_obj, (int, float)):
            continue
        if status_obj.strip().lower() != "manual":
            continue
        note_text = str(note_obj or "").strip().lower()
        if "vspreview" not in note_text:
            continue
        if key in allowed_keys:
            vspreview_reuse[key] = int(frames_obj)

    if not vspreview_reuse:
        return None

    # Logic extracted to apply_manual_offsets_logic for testability
    delta_map, manual_trim_starts = apply_manual_offsets_logic(
        plans=plans,
        vspreview_reuse=vspreview_reuse,
        display_data=display_data,
        plan_labels=plan_labels,
    )

    if not display_data.manual_trim_lines:
        return None

    display_data.offset_lines = ["Audio offsets: VSPreview manual offsets applied"]
    if display_data.manual_trim_lines:
        display_data.offset_lines.extend(display_data.manual_trim_lines)

    label_map = {plan.path.name: plan_labels.get(plan.path, plan.path.name) for plan in plans}
    display_data.json_offsets_frames = {
        label_map.get(key, key): int(value)
        for key, value in delta_map.items()
    }
    statuses_map = {key: "manual" for key in delta_map}
    return AudioAlignmentSummary(
        offsets_path=offsets_path,
        reference_name=reference.path.name,
        measurements=(),
        applied_frames=dict(delta_map),
        baseline_shift=0,
        statuses=statuses_map,
        reference_plan=reference,
        final_adjustments=dict(manual_trim_starts),
        swap_details={},
        suggested_frames={},
        suggestion_mode=False,
        manual_trim_starts=manual_trim_starts,
        vspreview_manual_offsets=dict(manual_trim_starts),
        vspreview_manual_deltas=dict(delta_map),
    )


def apply_audio_alignment(
    plans: Sequence[ClipPlan],
    cfg: AppConfig,
    analyze_path: Path,
    root: Path,
    audio_track_overrides: Mapping[str, int],
    reporter: CliOutputManagerProtocol | None = None,
) -> tuple[AudioAlignmentSummary | None, AudioAlignmentDisplayData | None]:
    """Apply audio alignment when enabled, returning summary and display data."""
    audio_cfg = cfg.audio_alignment
    prompt_reuse_offsets = _coerce_config_flag(audio_cfg.prompt_reuse_offsets)
    offsets_path = resolve_subdir(
        root,
        audio_cfg.offsets_filename,
        purpose="audio_alignment.offsets_filename",
    )
    display_data = AudioAlignmentDisplayData(
        stream_lines=[],
        estimation_line=None,
        offset_lines=[],
        offsets_file_line=f"Offsets file: {offsets_path}",
        json_reference_stream=None,
        json_target_streams={},
        json_offsets_sec={},
        json_offsets_frames={},
        warnings=[],
        correlations={},
        threshold=float(audio_cfg.correlation_threshold),
    )

    def _warn(message: str) -> None:
        display_data.warnings.append(f"[AUDIO] {message}")

    vspreview_enabled = _coerce_config_flag(audio_cfg.use_vspreview)

    reference_plan: ClipPlan | None = None
    if plans:
        reference_plan = _resolve_alignment_reference(plans, analyze_path, audio_cfg.reference)

    plan_labels: Dict[Path, str] = {plan.path: _plan_label(plan) for plan in plans}
    plan_lookup: Dict[str, ClipPlan] = {plan.path.name: plan for plan in plans}
    name_to_label: Dict[str, str] = {plan.path.name: plan_labels[plan.path] for plan in plans}
    plan_fps_map = _plan_fps_map(plans)

    reused_summary = _reuse_vspreview_manual_offsets_if_available(
        plans,
        reference_plan,
        vspreview_enabled,
        audio_cfg,
        display_data,
        plan_labels,
        offsets_path,
    )
    if reused_summary is not None:
        if not audio_cfg.enable:
            display_data.warnings.append(
                "[AUDIO] VSPreview manual alignment enabled — audio alignment disabled."
            )
        return reused_summary, display_data

    if not audio_cfg.enable:
        if vspreview_enabled and plans and reference_plan is not None:
            manual_trim_starts = {
                plan.path.name: int(plan.trim_start)
                for plan in plans
                if plan.trim_start != 0
            }
            if manual_trim_starts:
                for plan in plans:
                    trim = manual_trim_starts.get(plan.path.name)
                    if trim:
                        display_data.manual_trim_lines.append(
                            f"Existing manual trim: {plan_labels[plan.path]} → {trim}f"
                        )
            display_data.offset_lines = ["Audio offsets: not computed (manual alignment only)"]
            display_data.offset_lines.extend(display_data.manual_trim_lines)
            display_data.warnings.append(
                "[AUDIO] VSPreview manual alignment enabled — audio alignment disabled."
            )
            summary = AudioAlignmentSummary(
                offsets_path=offsets_path,
                reference_name=reference_plan.path.name,
                measurements=(),
                applied_frames=dict(manual_trim_starts),
                baseline_shift=0,
                statuses={},
                reference_plan=reference_plan,
                final_adjustments={},
                swap_details={},
                suggested_frames={},
                suggestion_mode=True,
                manual_trim_starts=manual_trim_starts,
            )
            return summary, display_data
        return None, display_data
    if len(plans) < 2:
        _warn("Audio alignment skipped: need at least two clips.")
        return None, display_data

    assert reference_plan is not None
    targets = [plan for plan in plans if plan is not reference_plan]
    if not targets:
        _warn("Audio alignment skipped: no secondary clips to compare.")
        return None, display_data

    measurement_order = [plan.path.name for plan in plans]
    negative_override_notes: Dict[str, str] = {}

    def _maybe_reuse_cached_offsets(
        reference: ClipPlan,
        candidate_targets: Sequence[ClipPlan],
    ) -> AudioAlignmentSummary | None:
        if not prompt_reuse_offsets:
            return None
        if not sys.stdin.isatty():
            return None
        try:
            cached_reference, existing_entries = load_existing_entries(offsets_path)
        except CLIAppError:
            return None
        if not existing_entries:
            return None
        if cached_reference is not None and cached_reference != reference.path.name:
            return None

        required_names = [plan.path.name for plan in candidate_targets]
        if any(name not in existing_entries for name in required_names):
            return None

        if click.confirm(
            "Recompute audio offsets using current clips?",
            default=True,
            show_default=True,
        ):
            return None

        display_data.estimation_line = (
            f"Audio offsets reused from existing file ({offsets_path.name})."
        )

        plan_map = plan_lookup
        reference_entry = existing_entries.get(reference.path.name)
        reference_manual_frames: int | None = None
        reference_manual_seconds: float | None = None
        if reference_entry is not None:
            entry = reference_entry
            status_obj = entry.get("status")
            if isinstance(status_obj, str) and status_obj.strip().lower() == "manual":
                reference_manual_frames = _safe_int(entry.get("frames"))
                reference_manual_seconds = _safe_float(entry.get("seconds"))
                if (
                    reference_manual_seconds is None
                    and reference_manual_frames is not None
                ):
                    fps_guess = _safe_float(reference_entry.get("target_fps")) or _safe_float(
                        reference_entry.get("reference_fps")
                    )
                    if fps_guess and fps_guess > 0:
                        reference_manual_seconds = reference_manual_frames / fps_guess

        measurements: list["AlignmentMeasurement"] = []
        swap_details: Dict[str, str] = {}
        negative_offsets: Dict[str, bool] = {}

        def _build_measurement(name: str, entry: Mapping[str, Any]) -> "AlignmentMeasurement":
            plan = plan_map[name]
            frames_val = _safe_int(entry.get("frames")) if entry else None
            seconds_val = _safe_float(entry.get("seconds")) if entry else None
            target_fps = _safe_float(entry.get("target_fps")) if entry else None
            reference_fps = _safe_float(entry.get("reference_fps")) if entry else None
            status_obj = entry.get("status") if entry else None
            is_manual = isinstance(status_obj, str) and status_obj.strip().lower() == "manual"
            if seconds_val is None and frames_val is not None:
                fps_val = target_fps if target_fps and target_fps > 0 else reference_fps
                if fps_val and fps_val > 0:
                    seconds_val = frames_val / fps_val
            if is_manual and reference_manual_frames is not None:
                if frames_val is not None:
                    frames_val -= reference_manual_frames
                if seconds_val is not None and reference_manual_seconds is not None:
                    seconds_val -= reference_manual_seconds
                elif seconds_val is None and frames_val is not None:
                    fps_val = target_fps if target_fps and target_fps > 0 else reference_fps
                    if fps_val and fps_val > 0:
                        seconds_val = frames_val / fps_val
            correlation_val = _safe_float(entry.get("correlation")) if entry else None
            error_obj = entry.get("error") if entry else None
            error_val = str(error_obj).strip() if isinstance(error_obj, str) and error_obj.strip() else None
            measurement = audio_alignment.AlignmentMeasurement(
                file=plan.path,
                offset_seconds=seconds_val if seconds_val is not None else 0.0,
                frames=frames_val,
                correlation=correlation_val if correlation_val is not None else 0.0,
                reference_fps=reference_fps,
                target_fps=target_fps,
                error=error_val,
            )
            note_obj = entry.get("note") if entry else None
            if isinstance(note_obj, str) and note_obj.strip():
                note_text = note_obj.strip()
                swap_details[name] = note_text
                if "opposite clip" in note_text.lower():
                    negative_offsets[name] = True
            return measurement

        for target_plan in candidate_targets:
            entry = existing_entries.get(target_plan.path.name)
            if entry is None:
                return None
            measurements.append(_build_measurement(target_plan.path.name, entry))

        reference_entry = existing_entries.get(reference.path.name)
        if reference_entry is not None:
            measurements.append(_build_measurement(reference.path.name, reference_entry))

        for measurement in measurements:
            if measurement.frames is None:
                derived_frames = _estimate_frames_from_seconds(measurement, plan_map)
                if derived_frames is not None:
                    measurement.frames = derived_frames

        raw_warning_messages: List[str] = []
        for measurement in measurements:
            reasons: List[str] = []
            if measurement.error:
                reasons.append(measurement.error)
            if measurement.offset_seconds is not None and abs(measurement.offset_seconds) > audio_cfg.max_offset_seconds:
                reasons.append(
                    f"offset {measurement.offset_seconds:.3f}s exceeds limit {audio_cfg.max_offset_seconds:.3f}s"
                )
            if measurement.correlation < audio_cfg.correlation_threshold:
                reasons.append(
                    f"correlation {measurement.correlation:.2f} below threshold {audio_cfg.correlation_threshold:.2f}"
                )
            if measurement.frames is None:
                reasons.append("unable to derive frame offset (missing fps)")

            if reasons:
                measurement.frames = None
                measurement.error = "; ".join(reasons)
                file_key = measurement.file.name
                negative_offsets.pop(file_key, None)
                label = name_to_label.get(file_key, file_key)
                raw_warning_messages.append(f"{label}: {measurement.error}")

        for warning_message in dict.fromkeys(raw_warning_messages):
            _warn(warning_message)

        offset_lines: List[str] = []
        offsets_sec: Dict[str, float] = {}
        offsets_frames: Dict[str, int] = {}

        for measurement in measurements:
            clip_name = measurement.file.name
            if clip_name == reference.path.name and len(measurements) > 1:
                continue
            label = name_to_label.get(clip_name, clip_name)
            if measurement.offset_seconds is not None:
                offsets_sec[label] = float(measurement.offset_seconds)
            frames_value = measurement.frames
            if frames_value is None:
                frames_value = _estimate_frames_from_seconds(measurement, plan_map)
                if frames_value is not None:
                    measurement.frames = frames_value
            if frames_value is not None:
                offsets_frames[label] = int(frames_value)
            display_data.correlations[label] = float(measurement.correlation)

            if measurement.error:
                offset_lines.append(
                    f"Audio offsets: {label}: manual edit required ({measurement.error})"
                )
                continue

            fps_value = 0.0
            if measurement.target_fps and measurement.target_fps > 0:
                fps_value = float(measurement.target_fps)
            elif measurement.reference_fps and measurement.reference_fps > 0:
                fps_value = float(measurement.reference_fps)

            frames_text = "n/a"
            if frames_value is not None:
                frames_text = f"{frames_value:+d}f"
            fps_text = f"{fps_value:.3f}" if fps_value > 0 else "0.000"
            suffix = ""
            if clip_name in negative_offsets:
                suffix = " (reference advanced; trimming target)"
            offset_lines.append(
                f"Audio offsets: {label}: {measurement.offset_seconds:+.3f}s ({frames_text} @ {fps_text}){suffix}"
            )
            detail = swap_details.get(clip_name)
            if detail:
                offset_lines.append(f"  note: {detail}")

        if not offset_lines:
            offset_lines.append("Audio offsets: none detected")

        display_data.offset_lines = offset_lines
        display_data.json_offsets_sec = offsets_sec
        display_data.json_offsets_frames = offsets_frames

        suggested_frames: Dict[str, int] = {}
        for measurement in measurements:
            frames_value = measurement.frames
            if frames_value is None:
                frames_value = _estimate_frames_from_seconds(measurement, plan_map)
                if frames_value is not None:
                    measurement.frames = frames_value
            if frames_value is not None:
                suggested_frames[measurement.file.name] = int(frames_value)

        applied_frames: Dict[str, int] = {}
        statuses: Dict[str, str] = {}
        for name, entry in existing_entries.items():
            if name not in plan_map:
                continue
            frames_val = _safe_int(entry.get("frames")) if entry else None
            if frames_val is not None:
                applied_frames[name] = frames_val
            status_obj = entry.get("status") if entry else None
            if isinstance(status_obj, str):
                statuses[name] = status_obj

        final_map: Dict[str, int] = {reference.path.name: 0}
        for name, frames in applied_frames.items():
            final_map[name] = frames

        baseline = min(final_map.values()) if final_map else 0
        baseline_shift = int(-baseline) if baseline < 0 else 0

        final_adjustments: Dict[str, int] = {}
        for plan in plans:
            desired = final_map.get(plan.path.name)
            if desired is None:
                continue
            adjustment = int(desired - baseline)
            if adjustment:
                plan.trim_start = plan.trim_start + adjustment
                plan.source_num_frames = None
                plan.alignment_frames = adjustment
                plan.alignment_status = statuses.get(plan.path.name, "auto")
            else:
                plan.alignment_frames = 0
                if plan.path.name in statuses:
                    plan.alignment_status = statuses.get(plan.path.name, "auto")
                else:
                    plan.alignment_status = ""
            final_adjustments[plan.path.name] = adjustment

        if baseline_shift:
            for plan in plans:
                if plan is reference:
                    plan.alignment_status = "baseline"

        summary = AudioAlignmentSummary(
            offsets_path=offsets_path,
            reference_name=reference.path.name,
            measurements=measurements,
            applied_frames=applied_frames,
            baseline_shift=baseline_shift,
            statuses=statuses,
            reference_plan=reference,
            final_adjustments=final_adjustments,
            swap_details=swap_details,
            suggested_frames=suggested_frames,
            suggestion_mode=False,
            manual_trim_starts={},
        )
        detail_map = _compose_measurement_details(
            measurements,
            applied_frames_map=applied_frames,
            statuses_map=statuses,
            suggestion_mode_active=False,
            manual_trims={},
            swap_map=swap_details,
            negative_notes=negative_override_notes,
        )
        summary.measured_offsets = detail_map
        _emit_measurement_lines(
            detail_map,
            measurement_order,
            append_manual=bool(display_data.manual_trim_lines),
        )
        return summary

    stream_infos: Dict[Path, List["AudioStreamInfo"]] = {}
    for plan in plans:
        try:
            infos = audio_alignment.probe_audio_streams(plan.path)
        except audio_alignment.AudioAlignmentError as exc:
            logger.warning("ffprobe audio stream probe failed for %s: %s", plan.path.name, exc)
            infos = []
        stream_infos[plan.path] = infos

    forced_streams: set[Path] = set()

    def _match_audio_override(plan: ClipPlan) -> Optional[int]:
        """Return override index for *plan* when configured, otherwise ``None``."""
        value = _match_override(plans.index(plan), plan.path, plan.metadata, audio_track_overrides)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _pick_default(streams: Sequence["AudioStreamInfo"]) -> int:
        """Return default stream index, falling back to the first entry or zero."""
        if not streams:
            return 0
        for stream in streams:
            if stream.is_default:
                return stream.index
        return streams[0].index

    ref_override = _match_audio_override(reference_plan)
    if ref_override is not None:
        forced_streams.add(reference_plan.path)
    reference_stream_index = ref_override if ref_override is not None else _pick_default(
        stream_infos.get(reference_plan.path, [])
    )

    reference_stream_info = None
    for candidate in stream_infos.get(reference_plan.path, []):
        if candidate.index == reference_stream_index:
            reference_stream_info = candidate
            break

    def _score_candidate(candidate: "AudioStreamInfo") -> float:
        """
        Compute a heuristic quality score for an audio stream candidate relative to the (closure) reference stream.

        Parameters:
            candidate (audio_alignment.AudioStreamInfo): Audio stream metadata to evaluate.

        Returns:
            score (float): Higher values indicate a better match to the reference stream based on language, codec, channels, sample rate, bitrate, and flags (`is_default`, `is_forced`); used for ranking candidate streams.
        """
        base = 0.0
        if reference_stream_info is not None:
            if reference_stream_info.language and candidate.language == reference_stream_info.language:
                base += 100.0
            elif reference_stream_info.language and not candidate.language:
                base += 10.0
            if candidate.codec_name == reference_stream_info.codec_name:
                base += 30.0
            elif candidate.codec_name.split(".")[0] == reference_stream_info.codec_name.split(".")[0]:
                base += 20.0
            if candidate.channels == reference_stream_info.channels:
                base += 10.0
            if reference_stream_info.channel_layout and candidate.channel_layout == reference_stream_info.channel_layout:
                base += 5.0
            if reference_stream_info.sample_rate and candidate.sample_rate == reference_stream_info.sample_rate:
                base += 10.0
            elif reference_stream_info.sample_rate and candidate.sample_rate:
                base -= abs(candidate.sample_rate - reference_stream_info.sample_rate) / 1000.0
            if reference_stream_info.bitrate and candidate.bitrate:
                base -= abs(candidate.bitrate - reference_stream_info.bitrate) / 10000.0
        base += 3.0 if candidate.is_default else 0.0
        base += 1.0 if candidate.is_forced else 0.0
        if candidate.bitrate:
            base += candidate.bitrate / 1e5
        return base

    target_stream_indices: Dict[Path, int] = {}
    for target in targets:
        override_idx = _match_audio_override(target)
        if override_idx is not None:
            target_stream_indices[target.path] = override_idx
            forced_streams.add(target.path)
            continue
        infos = stream_infos.get(target.path, [])
        if not infos:
            target_stream_indices[target.path] = 0
            continue
        best = max(infos, key=_score_candidate)
        target_stream_indices[target.path] = best.index

    def _describe_stream(plan: ClipPlan, stream_idx: int) -> tuple[str, str]:
        """
        Builds a human-readable label and a concise descriptor for the chosen audio stream of a clip.

        Parameters:
            plan (ClipPlan): Clip plan whose path and label are used in the returned label.
            stream_idx (int): Index of the audio stream to describe.

        Returns:
            tuple[str, str]: A pair (display_label, descriptor) where `display_label` is formatted as
            "<clip_label>-><codec>/<language>/<layout>" with " (forced)" appended if the stream is marked forced,
            and `descriptor` is the "<codec>/<language>/<layout>" string.
        """
        infos = stream_infos.get(plan.path, [])
        picked = next((info for info in infos if info.index == stream_idx), None)
        codec = (picked.codec_name if picked and picked.codec_name else "unknown").strip() or "unknown"
        language = (picked.language if picked and picked.language else "und").strip() or "und"
        if picked and picked.channel_layout:
            layout = picked.channel_layout.strip()
        elif picked and picked.channels:
            layout = f"{picked.channels}ch"
        else:
            layout = "?"
        descriptor = f"{codec}/{language}/{layout}"
        forced_suffix = " (forced)" if plan.path in forced_streams else ""
        label = plan_labels[plan.path]
        return f"{label}->{descriptor}{forced_suffix}", descriptor

    reference_stream_text, reference_descriptor = _describe_stream(reference_plan, reference_stream_index)
    display_data.json_reference_stream = reference_stream_text
    stream_descriptors: Dict[str, str] = {reference_plan.path.name: reference_descriptor}

    for idx, target in enumerate(targets):
        stream_idx = target_stream_indices.get(target.path, 0)
        target_stream_text, target_descriptor = _describe_stream(target, stream_idx)
        display_data.json_target_streams[plan_labels[target.path]] = target_descriptor
        stream_descriptors[target.path.name] = target_descriptor
        if idx == 0:
            display_data.stream_lines.append(
                f"Audio streams: ref={reference_stream_text}  target={target_stream_text}"
            )
        else:
            display_data.stream_lines.append(f"Audio streams: target={target_stream_text}")

    def _format_measurement_line(detail: AudioMeasurementDetail) -> str:
        stream_text = detail.stream or "?"
        seconds_text = (
            f"{detail.offset_seconds:+.3f}s"
            if detail.offset_seconds is not None
            else "n/a"
        )
        frames_text = (
            f"{detail.frames:+d}f" if detail.frames is not None else "n/a"
        )
        corr_text = (
            f"{detail.correlation:.2f}"
            if detail.correlation is not None and not math.isnan(detail.correlation)
            else "n/a"
        )
        applied_text = "applied" if detail.applied else "suggested"
        status_bits: List[str] = []
        if detail.status:
            status_bits.append(detail.status)
        status_bits.append(applied_text)
        status_text = "/".join(status_bits)
        return (
            f"Audio offsets: {detail.label}: [{stream_text}] "
            f"{seconds_text} ({frames_text}) corr={corr_text} status={status_text}"
        )

    def _emit_measurement_lines(
        detail_map: Dict[str, AudioMeasurementDetail],
        order: Sequence[str],
        *,
        append_manual: bool = False,
    ) -> None:
        offsets_sec: Dict[str, float] = {}
        offsets_frames: Dict[str, int] = {}
        offset_lines: List[str] = []
        for name in order:
            detail = detail_map.get(name)
            if detail is None:
                continue
            if (
                name == reference_plan.path.name
                and len(detail_map) > 1
            ):
                continue
            if detail.offset_seconds is not None:
                offsets_sec[detail.label] = float(detail.offset_seconds)
            if detail.frames is not None:
                offsets_frames[detail.label] = int(detail.frames)
            offset_lines.append(_format_measurement_line(detail))
            if detail.note:
                offset_lines.append(f"  note: {detail.note}")
        if not offset_lines:
            offset_lines.append("Audio offsets: none detected")
        if append_manual and display_data.manual_trim_lines:
            offset_lines.extend(display_data.manual_trim_lines)
        display_data.offset_lines = offset_lines
        display_data.json_offsets_sec = offsets_sec
        display_data.json_offsets_frames = offsets_frames
        display_data.measurements = {
            detail.label: detail for detail in detail_map.values()
        }
        display_data.correlations = {
            detail.label: detail.correlation
            for detail in detail_map.values()
            if detail.correlation is not None
        }

    fps_lookup: Dict[str, float] = {}
    for plan in plans:
        fps_tuple = plan.effective_fps or plan.source_fps or plan.fps_override
        fps_lookup[plan.path.name] = _fps_to_float(fps_tuple)

    def _compose_measurement_details(
        measurement_seq: Sequence["AlignmentMeasurement"],
        *,
        applied_frames_map: Mapping[str, int] | None,
        statuses_map: Mapping[str, str] | None,
        suggestion_mode_active: bool,
        manual_trims: Mapping[str, int],
        swap_map: Mapping[str, str],
        negative_notes: Mapping[str, str],
    ) -> Dict[str, AudioMeasurementDetail]:
        """
        Convert raw measurement objects into detail records used for CLI + JSON reporting.

        Parameters:
            measurement_seq: Measurements returned by the alignment pipeline.
            applied_frames_map: Mapping of clip names to frame adjustments actually applied.
            statuses_map: Mapping of clip names to status labels ("auto", "manual", etc.).
            suggestion_mode_active: True when offsets are suggestions only (VSPreview flow).
            manual_trims: Existing manual trims discovered earlier in the run.
            swap_map: Swap/notes per clip (e.g., "reference advanced" notes).
            negative_notes: Notes produced when negative offsets were redirected.

        Returns:
            Dict[str, AudioMeasurementDetail]: Mapping keyed by clip filename.
        """

        detail_map: Dict[str, AudioMeasurementDetail] = {}
        for measurement in measurement_seq:
            clip_name = measurement.file.name
            label = name_to_label.get(clip_name, clip_name)
            descriptor = stream_descriptors.get(clip_name, "")
            seconds_value: Optional[float]
            if measurement.offset_seconds is None:
                seconds_value = None
            else:
                seconds_value = float(measurement.offset_seconds)
            frames_value = int(measurement.frames) if measurement.frames is not None else None
            correlation_value: Optional[float]
            if math.isnan(measurement.correlation):
                correlation_value = None
            else:
                correlation_value = float(measurement.correlation)
            status_text = ""
            if statuses_map and clip_name in statuses_map:
                status_text = statuses_map[clip_name]
            applied_flag = False
            if not suggestion_mode_active and applied_frames_map and clip_name in applied_frames_map:
                applied_flag = True
            note_parts: List[str] = []
            swap_note = swap_map.get(clip_name)
            if swap_note:
                note_parts.append(swap_note)
            negative_note = negative_notes.get(clip_name)
            if negative_note:
                note_parts.append(negative_note)
            if measurement.error:
                note_parts.append(measurement.error)
                if not status_text:
                    status_text = "error"
                applied_flag = False
            note_value = " ".join(note_parts) if note_parts else None
            detail_map[clip_name] = AudioMeasurementDetail(
                label=label,
                stream=descriptor,
                offset_seconds=seconds_value,
                frames=frames_value,
                correlation=correlation_value,
                status=status_text,
                applied=applied_flag,
                note=note_value,
            )

        for clip_name, trim_frames in manual_trims.items():
            if clip_name in detail_map:
                continue
            label = name_to_label.get(clip_name, clip_name)
            descriptor = stream_descriptors.get(clip_name, "")
            fps_value = fps_lookup.get(clip_name, 0.0)
            seconds_value = (trim_frames / fps_value) if fps_value else None
            detail_map[clip_name] = AudioMeasurementDetail(
                label=label,
                stream=descriptor,
                offset_seconds=seconds_value,
                frames=int(trim_frames),
                correlation=None,
                status="manual",
                applied=not suggestion_mode_active,
                note=None,
            )
        return detail_map

    reused_cached = _maybe_reuse_cached_offsets(reference_plan, targets)
    if reused_cached is not None:
        summary = reused_cached
        try:
            _, existing_entries = load_existing_entries(offsets_path)
        except CLIAppError:
            existing_entries = {}
        suggestion_hints = extract_suggestion_hints(existing_entries)
        detail_map: Dict[str, AudioMeasurementDetail] = {}
        for plan in plans:
            key = plan.path.name
            frames_val = summary.applied_frames.get(key)
            seconds_val: Optional[float]
            fps_val = fps_lookup.get(key, 0.0)
            if frames_val is None or not fps_val:
                seconds_val = None
            else:
                seconds_val = frames_val / fps_val if fps_val else None
            descriptor = stream_descriptors.get(key, "")
            detail_map[key] = AudioMeasurementDetail(
                label=name_to_label.get(key, key),
                stream=descriptor,
                offset_seconds=seconds_val,
                frames=frames_val,
                correlation=None,
                status=summary.statuses.get(key, "manual"),
                applied=True,
            )
        apply_suggestion_hints_to_details(detail_map, suggestion_hints)
        for name, (frames_hint, _seconds_hint) in suggestion_hints.items():
            if frames_hint is None:
                continue
            summary.suggested_frames[name] = int(frames_hint)
        summary.measured_offsets = detail_map
        display_data.measurements = {
            detail.label: detail for detail in detail_map.values()
        }
        _emit_measurement_lines(
            detail_map,
            measurement_order,
            append_manual=bool(display_data.manual_trim_lines),
        )
        return summary, display_data

    reference_fps_tuple = reference_plan.effective_fps or reference_plan.source_fps
    reference_fps = _fps_to_float(reference_fps_tuple)
    max_offset = float(audio_cfg.max_offset_seconds)
    raw_duration = audio_cfg.duration_seconds if audio_cfg.duration_seconds is not None else None
    duration_seconds = float(raw_duration) if raw_duration is not None else None
    start_seconds = float(audio_cfg.start_seconds or 0.0)
    search_text = f"±{max_offset:.2f}s"
    window_text = f"{duration_seconds:.2f}s" if duration_seconds is not None else "auto"
    start_text = f"{start_seconds:.2f}s"
    display_data.estimation_line = (
        f"Estimating audio offsets … fps={reference_fps:.3f} "
        f"search={search_text} start={start_text} window={window_text}"
    )

    try:
        base_start = float(audio_cfg.start_seconds or 0.0)
        base_duration_param: Optional[float]
        if audio_cfg.duration_seconds is None:
            base_duration_param = None
        else:
            base_duration_param = float(audio_cfg.duration_seconds)
        hop_length = max(1, min(audio_cfg.hop_length, max(1, audio_cfg.sample_rate // 100)))

        measurements: List["AlignmentMeasurement"]
        negative_offsets: Dict[str, bool] = {}

        spinner_context: ContextManager[object]
        status_factory = None
        if reporter is not None and not getattr(reporter, "quiet", False):
            status_factory = getattr(reporter.console, "status", None)
        if callable(status_factory):
            spinner_context = cast(
                ContextManager[object],
                status_factory("[cyan]Estimating audio offsets…[/cyan]", spinner="dots"),
            )
        else:
            spinner_context = nullcontext()
        processed = 0
        start_time = time.perf_counter()
        total_targets = len(targets)

        with spinner_context as status:
            def _advance_audio(count: int) -> None:
                """
                Advance the audio-alignment progress by a given number of processed pairs.

                Parameters:
                    count (int): Number of audio pair measurements to add to the processed total.
                """
                nonlocal processed
                processed += count
                if status is None or total_targets <= 0:
                    return
                status_update = getattr(status, "update", None)
                if callable(status_update):
                    elapsed = time.perf_counter() - start_time
                    rate_val = processed / elapsed if elapsed > 0 else 0.0
                    status_update(
                        f"[cyan]Estimating audio offsets… {processed}/{total_targets} ({rate_val:0.2f} pairs/s)[/cyan]"
                    )

            measurements = audio_alignment.measure_offsets(
                reference_plan.path,
                [plan.path for plan in targets],
                sample_rate=audio_cfg.sample_rate,
                hop_length=hop_length,
                start_seconds=base_start,
                duration_seconds=base_duration_param,
                reference_stream=reference_stream_index,
                target_streams=target_stream_indices,
                progress_callback=_advance_audio,
                fps_hints=plan_fps_map,
            )

        for measurement in measurements:
            if measurement.frames is None:
                if measurement.file not in plan_fps_map:
                    label = name_to_label.get(measurement.file.name, measurement.file.name)
                    logger.debug(
                        "Audio alignment missing cached FPS for %s (%s); deriving frames from seconds fallback",
                        measurement.file.name,
                        label,
                    )
                derived_frames = _estimate_frames_from_seconds(measurement, plan_lookup)
                if derived_frames is not None:
                    measurement.frames = derived_frames

        suggested_frames: Dict[str, int] = {}
        for measurement in measurements:
            frames_value = measurement.frames
            if frames_value is None:
                frames_value = _estimate_frames_from_seconds(measurement, plan_lookup)
                if frames_value is not None:
                    measurement.frames = frames_value
            if frames_value is not None:
                suggested_frames[measurement.file.name] = int(frames_value)

        negative_override_notes.clear()
        swap_details: Dict[str, str] = {}
        for measurement in measurements:
            if measurement.frames is not None and measurement.frames < 0:
                file_key = measurement.file.name
                negative_offsets[file_key] = True
                negative_override_notes[file_key] = (
                    "Suggested negative offset applied to the opposite clip for trim-first behaviour."
                )

        raw_warning_messages: List[str] = []
        for measurement in measurements:
            reasons: List[str] = []
            if measurement.error:
                reasons.append(measurement.error)
            if measurement.offset_seconds is not None and abs(measurement.offset_seconds) > audio_cfg.max_offset_seconds:
                reasons.append(
                    f"offset {measurement.offset_seconds:.3f}s exceeds limit {audio_cfg.max_offset_seconds:.3f}s"
                )
            if measurement.correlation < audio_cfg.correlation_threshold:
                reasons.append(
                    f"correlation {measurement.correlation:.2f} below threshold {audio_cfg.correlation_threshold:.2f}"
                )
            if measurement.frames is None:
                reasons.append("unable to derive frame offset (missing fps)")

            if reasons:
                measurement.frames = None
                measurement.error = "; ".join(reasons)
                file_key = measurement.file.name
                negative_offsets.pop(file_key, None)
                label = name_to_label.get(file_key, file_key)
                raw_warning_messages.append(f"{label}: {measurement.error}")

        for warning_message in dict.fromkeys(raw_warning_messages):
            _warn(warning_message)

        manual_trim_starts: Dict[str, int] = {}
        if vspreview_enabled:
            for plan in plans:
                if plan.has_trim_start_override and plan.trim_start != 0:
                    manual_trim_starts[plan.path.name] = int(plan.trim_start)
                    label = plan_labels.get(plan.path, plan.path.name)
                    display_data.manual_trim_lines.append(
                        f"Existing manual trim: {label} → {plan.trim_start}f"
                    )
            display_data.warnings.append(
                "[AUDIO] VSPreview manual alignment enabled — offsets reported for guidance only."
            )
            summary = AudioAlignmentSummary(
                offsets_path=offsets_path,
                reference_name=reference_plan.path.name,
                measurements=measurements,
                applied_frames=dict(manual_trim_starts),
                baseline_shift=0,
                statuses={m.file.name: "suggested" for m in measurements},
                reference_plan=reference_plan,
                final_adjustments={},
                swap_details=swap_details,
                suggested_frames=suggested_frames,
                suggestion_mode=True,
                manual_trim_starts=manual_trim_starts,
            )
            detail_map = _compose_measurement_details(
                measurements,
                applied_frames_map=summary.applied_frames,
                statuses_map=summary.statuses,
                suggestion_mode_active=True,
                manual_trims=manual_trim_starts,
                swap_map=swap_details,
                negative_notes=negative_override_notes,
            )
            summary.measured_offsets = detail_map
            _emit_measurement_lines(
                detail_map,
                measurement_order,
                append_manual=bool(display_data.manual_trim_lines),
            )
            return summary, display_data

        applied_frames, statuses = audio_alignment.update_offsets_file(
            offsets_path,
            reference_plan.path.name,
            measurements,
            load_existing_entries(offsets_path)[1],
            negative_override_notes,
        )

        final_map: Dict[str, int] = {reference_plan.path.name: 0}
        for name, frames in applied_frames.items():
            final_map[name] = frames

        baseline = min(final_map.values()) if final_map else 0
        baseline_shift = int(-baseline) if baseline < 0 else 0

        final_adjustments: Dict[str, int] = {}
        for plan in plans:
            desired = final_map.get(plan.path.name)
            if desired is None:
                continue
            adjustment = int(desired - baseline)

            if adjustment:
                plan.trim_start = plan.trim_start + adjustment
                plan.source_num_frames = None
                plan.alignment_frames = adjustment
                plan.alignment_status = statuses.get(plan.path.name, "auto")
            else:
                plan.alignment_frames = 0
                plan.alignment_status = statuses.get(plan.path.name, "auto") if plan.path.name in statuses else ""
            final_adjustments[plan.path.name] = adjustment

        if baseline_shift:
            for plan in plans:
                if plan is reference_plan:
                    plan.alignment_status = "baseline"

        summary = AudioAlignmentSummary(
            offsets_path=offsets_path,
            reference_name=reference_plan.path.name,
            measurements=measurements,
            applied_frames=applied_frames,
            baseline_shift=baseline_shift,
            statuses=statuses,
            reference_plan=reference_plan,
            final_adjustments=final_adjustments,
            swap_details=swap_details,
            suggested_frames=suggested_frames,
            suggestion_mode=False,
            manual_trim_starts=manual_trim_starts,
        )
        detail_map = _compose_measurement_details(
            measurements,
            applied_frames_map=applied_frames,
            statuses_map=statuses,
            suggestion_mode_active=False,
            manual_trims=manual_trim_starts,
            swap_map=swap_details,
            negative_notes=negative_override_notes,
        )
        summary.measured_offsets = detail_map
        _emit_measurement_lines(
            detail_map,
            measurement_order,
            append_manual=bool(display_data.manual_trim_lines),
        )
        return summary, display_data
    except audio_alignment.AudioAlignmentError as exc:
        raise CLIAppError(
            f"Audio alignment failed: {exc}",
            rich_message=f"[red]Audio alignment failed:[/red] {exc}",
        ) from exc


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
