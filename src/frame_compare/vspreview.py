"""VSPreview script rendering, persistence, and launch helpers."""
from __future__ import annotations

import datetime as _dt
import importlib
import importlib.util
import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Final, List, Mapping, Optional, Tuple, cast

import click
from rich.text import Text

from src import audio_alignment
from src.datatypes import AppConfig, ColorConfig
from src.frame_compare import subproc
from src.frame_compare.alignment_helpers import derive_frame_hint
from src.frame_compare.layout_utils import (
    normalise_vspreview_mode as _normalise_vspreview_mode,
)
from src.frame_compare.layout_utils import (
    plan_label as _plan_label,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def resolve_subdir(root: Path, relative: str, *, purpose: str, allow_absolute: bool = False) -> Path:
    """Delegate to preflight.resolve_subdir without creating an import cycle."""

    from src.frame_compare import preflight as _preflight

    return _preflight.resolve_subdir(root, relative, purpose=purpose, allow_absolute=allow_absolute)

if TYPE_CHECKING:  # pragma: no cover
    from src.audio_alignment import AlignmentMeasurement
    from src.frame_compare.alignment import (
        AudioAlignmentDisplayData,
        AudioAlignmentSummary,
        AudioMeasurementDetail,
    )
    from src.frame_compare.cli_runtime import (
        AudioAlignmentJSON,
        CliOutputManagerProtocol,
        ClipPlan,
        JsonTail,
    )
else:  # pragma: no cover - runtime stubs avoid circular import
    CliOutputManagerProtocol = Any  # type: ignore[misc,assignment]
    JsonTail = Dict[str, Any]  # type: ignore[misc,assignment]
    ClipPlan = Any  # type: ignore[misc,assignment]
    AlignmentMeasurement = Any  # type: ignore[misc,assignment]
    AudioAlignmentJSON = Dict[str, Any]  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

VSPREVIEW_WINDOWS_INSTALL: Final[str] = (
    "uv add frame-compare --extra preview  # fallback: uv add vspreview PySide6"
)
VSPREVIEW_POSIX_INSTALL: Final[str] = (
    "uv add frame-compare --extra preview  # fallback: uv add vspreview PySide6"
)
_VSPREVIEW_MANUAL_COMMAND_TEMPLATE: Final[str] = "{python} -m vspreview {script}"
_MANUAL_OFFSET_FALLBACK_FPS: Final[tuple[int, int]] = (24000, 1001)

ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]


def _fps_to_float(value: tuple[int, int] | None) -> float:
    if value is None:
        return 0.0
    numerator, denominator = value
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _resolve_manual_offset_fps(plan: ClipPlan) -> tuple[int, int]:
    """Return a usable FPS tuple for manual-offset reporting."""

    fps_tuple = plan.effective_fps or plan.source_fps or plan.fps_override
    if fps_tuple is None:
        return _MANUAL_OFFSET_FALLBACK_FPS
    num, den = fps_tuple
    if den == 0:
        return _MANUAL_OFFSET_FALLBACK_FPS
    return int(num), int(den)


def _coerce_str_mapping(mapping: Mapping[str, object] | MappingABC[str, object] | None) -> dict[str, object]:
    from src.frame_compare.cli_runtime import coerce_str_mapping as _impl

    return _impl(mapping)


def _ensure_audio_alignment_block(json_tail: JsonTail) -> "AudioAlignmentJSON":
    from src.frame_compare.cli_runtime import (
        ensure_audio_alignment_block as _impl,
    )

    return _impl(json_tail)


def format_manual_command(script_path: Path) -> str:
    """Return a VSPreview CLI invocation that mirrors the internal subprocess call."""

    python_exe = sys.executable or "python"
    script_arg = str(script_path)
    if os.name == "nt":
        if " " in python_exe and not python_exe.startswith('"'):
            python_exe = f'"{python_exe}"'
        if " " in script_arg and not script_arg.startswith('"'):
            script_arg = f'"{script_arg}"'
    else:
        python_exe = shlex.quote(python_exe)
        script_arg = shlex.quote(script_arg)
    return _VSPREVIEW_MANUAL_COMMAND_TEMPLATE.format(
        python=python_exe,
        script=script_arg,
    )


@dataclass(frozen=True)
class _VSPreviewCommandData:
    """Typed container for VSPreview script metadata shared between helpers."""

    script_path: Path
    manual_command: str
    missing_reason: Optional[str] = None


def _color_config_literal(color_cfg: ColorConfig) -> str:
    color_dict = asdict(color_cfg)
    items = ",\n    ".join(f"{key}={value!r}" for key, value in color_dict.items())
    return f"ColorConfig(\n    {items}\n)"


def render_script(
    plans: Sequence[ClipPlan],
    summary: "AudioAlignmentSummary",
    cfg: AppConfig,
    root: Path,
) -> str:
    """Render the VSPreview Python script for the current audio-alignment session."""

    reference_plan = summary.reference_plan
    targets = [plan for plan in plans if plan is not reference_plan]
    project_root = PROJECT_ROOT

    search_paths = [
        str(Path(path).expanduser())
        for path in getattr(cfg.runtime, "vapoursynth_python_paths", [])
        if path
    ]
    color_literal = _color_config_literal(cfg.color)

    preview_mode_value = _normalise_vspreview_mode(
        getattr(cfg.audio_alignment, "vspreview_mode", "baseline")
    )
    apply_seeded_offsets = preview_mode_value == "seeded"
    show_overlay = bool(getattr(cfg.audio_alignment, "show_suggested_in_preview", True))
    if summary.measured_offsets:
        measurement_lookup: Dict[str, Optional[float]] = {
            name: detail.offset_seconds for name, detail in summary.measured_offsets.items()
        }
    else:
        measurement_lookup = {
            measurement.file.name: measurement.offset_seconds
            for measurement in summary.measurements
        }

    manual_trims = {}
    if summary.manual_trim_starts:
        manual_trims = {
            _plan_label(plan): summary.manual_trim_starts.get(plan.path.name, int(plan.trim_start))
            for plan in plans
        }
    else:
        manual_trims = {
            _plan_label(plan): int(plan.trim_start)
            for plan in plans
            if plan.trim_start != 0
        }

    reference_label = _plan_label(reference_plan)
    reference_trim_end = reference_plan.trim_end if reference_plan.trim_end is not None else None
    reference_info = textwrap.dedent(
        f"""        {{
        'label': {reference_label!r},
        'path': {str(reference_plan.path)!r},
        'trim_start': {int(reference_plan.trim_start)},
        'trim_end': {reference_trim_end!r},
        'fps_override': {tuple(reference_plan.fps_override) if reference_plan.fps_override else None!r},
        }}
        """
    ).strip()

    target_lines: list[str] = []
    offset_lines: list[str] = []
    suggestion_lines: list[str] = []
    if not targets:
        offset_lines.append("    # Add entries like 'Clip Label': 0 once targets are available.")
    for plan in targets:
        label = _plan_label(plan)
        trim_end_value = plan.trim_end if plan.trim_end is not None else None
        fps_override = tuple(plan.fps_override) if plan.fps_override else None
        suggested_frames_value = derive_frame_hint(summary, plan.path.name)
        measurement_seconds = measurement_lookup.get(plan.path.name)
        suggested_seconds_value = 0.0
        if measurement_seconds is not None:
            suggested_seconds_value = float(measurement_seconds)
        manual_trim = manual_trims.get(label, int(plan.trim_start))
        manual_note = (
            f"baseline trim {manual_trim}f"
            if manual_trim
            else "no baseline trim"
        )
        applied_initial = (
            suggested_frames_value if apply_seeded_offsets and suggested_frames_value is not None else 0
        )
        suggestion_comment = (
            f"Suggested delta {suggested_frames_value:+d}f"
            if suggested_frames_value is not None
            else "Suggested delta n/a"
        )
        target_lines.append(
            textwrap.dedent(
                f"""                {label!r}: {{
                    'label': {label!r},
                    'path': {str(plan.path)!r},
                    'trim_start': {int(plan.trim_start)},
                    'trim_end': {trim_end_value!r},
                    'fps_override': {fps_override!r},
                    'manual_trim': {manual_trim},
                    'manual_trim_description': {manual_note!r},
                }},"""
            ).rstrip()
        )
        offset_lines.append(
            f"    {label!r}: {applied_initial},  # {suggestion_comment}"
        )
        frames_literal = "None" if suggested_frames_value is None else str(int(suggested_frames_value))
        suggestion_lines.append(
            f"    {label!r}: ({frames_literal}, {suggested_seconds_value!r}),"
        )

    targets_literal = "\n".join(target_lines) if target_lines else ""
    offsets_literal = "\n".join(offset_lines)
    suggestions_literal = "\n".join(suggestion_lines)

    extra_paths = [
        str(project_root),
        str(project_root / "src"),
        str(root),
    ]
    extra_paths_literal = ", ".join(repr(path) for path in extra_paths)
    search_paths_literal = repr(search_paths)

    script = f"""# Auto-generated by Frame Compare to assist with VSPreview alignment.
import sys
from pathlib import Path

try:
    # Prefer UTF-8 on Windows consoles and avoid crashing on encoding errors.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WORKSPACE_ROOT = Path({str(root)!r})
PROJECT_ROOT = Path({str(project_root)!r})
EXTRA_PATHS = [{extra_paths_literal}]
for candidate in EXTRA_PATHS:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import vapoursynth as vs
from src.frame_compare import vs as vs_core
from src.datatypes import ColorConfig

vs_core.configure(
    search_paths={search_paths_literal},
    source_preference={cfg.source.preferred!r},
)

COLOR_CFG = {color_literal}

REFERENCE = {reference_info}

TARGETS = {{
{targets_literal}
}}

OFFSET_MAP = {{
{offsets_literal}
}}

SUGGESTION_MAP = {{
{suggestions_literal}
}}

PREVIEW_MODE = {preview_mode_value!r}
SHOW_SUGGESTED_OVERLAY = {show_overlay!r}

core = vs.core


def safe_print(msg: str) -> None:
    try:
        print(msg)
    except Exception:
        try:
            print(msg.encode("utf-8", "replace").decode("utf-8", "replace"))
        except Exception:
            print("[log message unavailable due to encoding]")


def _load_clip(info):
    clip = vs_core.init_clip(
        str(Path(info['path'])),
        trim_start=int(info.get('trim_start', 0)),
        trim_end=info.get('trim_end'),
        fps_map=tuple(info['fps_override']) if info.get('fps_override') else None,
    )
    processed = vs_core.process_clip_for_screenshot(
        clip,
        info['label'],
        COLOR_CFG,
        enable_overlay=False,
        enable_verification=False,
    ).clip
    return processed


def _apply_offset(reference_clip, target_clip, offset_frames):
    if offset_frames > 0:
        target_clip = target_clip[offset_frames:]
    elif offset_frames < 0:
        reference_clip = reference_clip[abs(offset_frames):]
    return reference_clip, target_clip


def _extract_fps_tuple(clip):
    num = getattr(clip, "fps_num", None)
    den = getattr(clip, "fps_den", None)
    if isinstance(num, int) and isinstance(den, int) and den:
        return int(num), int(den)
    return None


def _harmonise_fps(reference_clip, target_clip, label):
    reference_fps = _extract_fps_tuple(reference_clip)
    target_fps = _extract_fps_tuple(target_clip)
    if not reference_fps or not target_fps:
        return reference_clip, target_clip
    if reference_fps == target_fps:
        return reference_clip, target_clip
    try:
        target_clip = target_clip.std.AssumeFPS(num=reference_fps[0], den=reference_fps[1])
        safe_print(
            "Adjusted FPS for target '%s' to match reference (%s/%s -> %s/%s)"
            % (
                label,
                target_fps[0],
                target_fps[1],
                reference_fps[0],
                reference_fps[1],
            )
        )
    except Exception as exc:
        safe_print("Warning: Failed to harmonise FPS for target '%s': %s" % (label, exc))
    return reference_clip, target_clip


def _format_overlay_text(label, suggested_frames, suggested_seconds, applied_frames):
    applied_label = "baseline" if applied_frames == 0 else "seeded"
    applied_value = "0" if applied_frames == 0 else f"{{applied_frames:+d}}"
    seconds_value = f"{{suggested_seconds:.3f}}"
    if seconds_value == "-0.000":
        seconds_value = "0.000"
    if suggested_frames is None:
        suggested_value = "n/a"
    else:
        suggested_value = f"{{suggested_frames:+d}}f"
    return (
        "{{label}}: {{suggested}} (~{{seconds}}s) • "
        "Preview applied: {{applied}}f ({{status}}) • "
        "(+ trims target / - trims reference)"
    ).format(
        label=label,
        suggested=suggested_value,
        seconds=seconds_value,
        applied=applied_value,
        status=applied_label,
    )


def _maybe_apply_overlay(clip, label, suggested_frames, suggested_seconds, applied_frames):
    if not SHOW_SUGGESTED_OVERLAY:
        return clip
    try:
        message = _format_overlay_text(label, suggested_frames, suggested_seconds, applied_frames)
    except Exception:
        message = "Suggested offset unavailable"
    try:
        return clip.text.Text(message, alignment=7)
    except Exception as exc:
        safe_print("Warning: Failed to draw overlay text for preview: %s" % (exc,))
        return clip


safe_print("Reference clip: %s" % (REFERENCE['label'],))
safe_print("VSPreview mode: %s" % (PREVIEW_MODE,))
if not TARGETS:
    safe_print("No target clips defined; edit TARGETS and OFFSET_MAP to add entries.")

slot = 0
for label, info in TARGETS.items():
    reference_clip = _load_clip(REFERENCE)
    target_clip = _load_clip(info)
    reference_clip, target_clip = _harmonise_fps(reference_clip, target_clip, label)
    offset_frames = int(OFFSET_MAP.get(label, 0))
    suggested_entry = SUGGESTION_MAP.get(label, (None, 0.0))
    suggested_frames = suggested_entry[0]
    suggested_seconds = float(suggested_entry[1])
    if suggested_frames is not None:
        suggested_frames = int(suggested_frames)
    ref_view, tgt_view = _apply_offset(reference_clip, target_clip, offset_frames)
    ref_view = _maybe_apply_overlay(
        ref_view,
        REFERENCE['label'],
        None,
        0.0,
        0,
    )
    tgt_view = _maybe_apply_overlay(
        tgt_view,
        label,
        suggested_frames,
        suggested_seconds,
        offset_frames,
    )
    ref_view.set_output(slot)
    tgt_view.set_output(slot + 1)
    applied_label = "baseline" if offset_frames == 0 else "seeded"
    suggested_display = (
        f"{{suggested_frames:+d}}f" if suggested_frames is not None else "n/a"
    )
    safe_print(
        "Target '%s': baseline trim=%sf (%s), suggested delta=%s (~%+.3fs), preview applied=%+df (%s mode)"
        % (
            label,
            info.get('manual_trim', 0),
            info.get('manual_trim_description', 'n/a'),
            suggested_display,
            suggested_seconds,
            offset_frames,
            applied_label,
        )
    )
    slot += 2

safe_print("VSPreview outputs: reference on even slots, target on odd slots (0<->1, 2<->3, ...).")
safe_print("Edit OFFSET_MAP values and press Ctrl+R in VSPreview to reload the script.")
"""
    return textwrap.dedent(script)
def persist_script(script_text: str, root: Path) -> Path:
    """Persist the generated VSPreview script under ROOT/vspreview/."""

    # Late import to avoid module import cycles during CLI startup.
    from src.frame_compare import preflight as _preflight

    script_dir = _preflight.resolve_subdir(root, "vspreview", purpose="vspreview workspace")
    script_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    script_path = script_dir / f"vspreview_{timestamp}_{uuid.uuid4().hex[:8]}.py"
    while script_path.exists():
        logger.warning(
            "VSPreview script %s already exists; generating alternate filename to avoid overwriting",
            script_path.name,
        )
        script_path = script_dir / f"vspreview_{timestamp}_{uuid.uuid4().hex[:8]}.py"
    script_path.write_text(script_text, encoding="utf-8")
    return script_path


def write_script(
    plans: Sequence[ClipPlan],
    summary: "AudioAlignmentSummary",
    cfg: AppConfig,
    root: Path,
) -> Path:
    """Render and persist a VSPreview script for the supplied plans."""

    script_text = render_script(plans, summary, cfg, root)
    return persist_script(script_text, root)


def prompt_offsets(
    plans: Sequence[ClipPlan],
    summary: "AudioAlignmentSummary",
    reporter: CliOutputManagerProtocol,
    display: Optional["AudioAlignmentDisplayData"],
) -> dict[str, int] | None:
    """Prompt the user for manual frame deltas after VSPreview inspection."""

    reference_plan = summary.reference_plan
    targets = [plan for plan in plans if plan is not reference_plan]
    if not targets:
        return {}

    baseline_map: Dict[str, int] = {
        plan.path.name: summary.manual_trim_starts.get(plan.path.name, int(plan.trim_start))
        for plan in plans
    }

    reference_label = _plan_label(reference_plan)
    reporter.line(
        "Enter VSPreview frame offsets relative to the reported baselines. "
        "Positive trims the target; negative advances the reference."
    )
    offsets: Dict[str, int] = {}
    for plan in targets:
        label = _plan_label(plan)
        baseline_value = baseline_map.get(plan.path.name, int(plan.trim_start))
        suggested = derive_frame_hint(summary, plan.path.name)
        prompt_parts = [
            f"VSPreview offset for {label} relative to {reference_label}",
            f"baseline {baseline_value}f",
        ]
        if suggested is not None:
            prompt_parts.append(f"suggested {suggested:+d}f")
        else:
            prompt_parts.append("suggested n/a")
        prompt_message = " (".join([prompt_parts[0], ", ".join(prompt_parts[1:])]) + ")"
        try:
            delta = int(
                click.prompt(
                    prompt_message,
                    type=int,
                    default=0,
                    show_default=True,
                )
            )
        except click.exceptions.Abort:
            reporter.warn("VSPreview offset entry aborted; keeping existing trims.")
            return None
        offsets[plan.path.name] = delta
        if display is not None:
            display.manual_trim_lines.append(f"Baseline for {label}: {baseline_value}f")
    return offsets


def _measurement_detail_cls() -> type["AudioMeasurementDetail"]:
    from src.frame_compare import alignment as alignment_package

    return alignment_package.AudioMeasurementDetail


def apply_manual_offsets(
    plans: Sequence[ClipPlan],
    summary: "AudioAlignmentSummary",
    deltas: Mapping[str, int],
    reporter: CliOutputManagerProtocol,
    json_tail: JsonTail,
    display: "AudioAlignmentDisplayData | None",
) -> None:
    reference_plan = summary.reference_plan
    reference_name = reference_plan.path.name
    baseline_map: Dict[str, int] = {
        plan.path.name: summary.manual_trim_starts.get(plan.path.name, int(plan.trim_start))
        for plan in plans
    }
    if reference_name not in baseline_map:
        baseline_map[reference_name] = int(reference_plan.trim_start)

    manual_trim_starts: Dict[str, int] = {}
    delta_map: Dict[str, int] = {}
    manual_lines: List[str] = []

    unknown_deltas = sorted(set(deltas.keys()) - set(baseline_map.keys()))
    if unknown_deltas:
        message = (
            "VSPreview provided offsets for clips that are not part of the current plan: "
            + ", ".join(sorted(unknown_deltas))
        )
        reporter.warn(message)
        logger.warning(message)

    from src.frame_compare.alignment.core import apply_manual_offsets_logic

    # 1. Reuse the shared normalization logic from alignment.core
    # We need to adapt the input 'deltas' to what _apply_manual_offsets_logic expects (vspreview_reuse).
    # The logic there expects: vspreview_reuse = {filename: delta_int}
    # It returns: (delta_map, manual_trim_starts)

    # Create a dummy display object to capture log lines
    class _DummyDisplay:
        def __init__(self) -> None:
            self.manual_trim_lines: list[str] = []

    dummy_display = _DummyDisplay()

    # Call the shared logic
    # cast deltas to dict[str, int] to satisfy type checker (Mapping vs dict)
    delta_map, manual_trim_starts = apply_manual_offsets_logic(
        plans,
        dict(deltas),
        dummy_display, # type: ignore
        {p.path: p.path.name for p in plans}
    )

    # Emit the log lines captured by the shared logic
    for line in dummy_display.manual_trim_lines:
        manual_lines.append(line)
        reporter.line(line)

    # We also need to handle the reference plan specifically if it was adjusted
    # The shared logic updates the plans in-place, so we just need to ensure
    # reference_plan properties are set correctly if not covered by the loop below.
    # Actually, _apply_manual_offsets_logic updates ALL plans in the list.

    # However, the original code had some specific reporting for the reference.
    # Let's see if we need to preserve that.
    # The shared logic logs: "baseline X ... -> Y" for all changed plans.
    # The original code logged separate lines for targets and reference.
    # The shared logic's logging should be sufficient.

    if display is not None:
        display.manual_trim_lines.extend(manual_lines)
        display.offset_lines = ["Audio offsets: VSPreview manual offsets applied"]
        display.offset_lines.extend(display.manual_trim_lines)

    fps_lookup: Dict[str, Tuple[int, int]] = {}
    for plan in plans:
        fps_lookup[plan.path.name] = _resolve_manual_offset_fps(plan)

    measurement_order = [plan.path.name for plan in plans]
    plan_lookup: Dict[str, ClipPlan] = {plan.path.name: plan for plan in plans}

    measurements: List["AlignmentMeasurement"] = []
    existing_override_map: Dict[str, Dict[str, object]] = {}
    notes_map: Dict[str, str] = {}
    for plan in plans:
        key = plan.path.name
        frames_value = int(delta_map.get(key, 0))
        fps_tuple = fps_lookup.get(key)
        fps_float = _fps_to_float(fps_tuple) if fps_tuple else 0.0
        seconds_value = float(frames_value) / fps_float if fps_float else 0.0
        measurements.append(
            audio_alignment.AlignmentMeasurement(
                file=plan.path,
                offset_seconds=seconds_value,
                frames=frames_value,
                correlation=1.0,
                reference_fps=fps_float or None,
                target_fps=fps_float or None,
            )
        )
        existing_override_map[key] = {"frames": frames_value, "status": "manual"}
        notes_map[key] = "VSPreview"

    applied_frames, statuses = audio_alignment.update_offsets_file(
        summary.offsets_path,
        reference_plan.path.name,
        tuple(measurements),
        existing_override_map,
        notes_map,
    )

    summary.applied_frames = dict(applied_frames)
    summary.statuses = dict(statuses)
    summary.final_adjustments = dict(manual_trim_starts)
    summary.manual_trim_starts = dict(manual_trim_starts)
    summary.suggestion_mode = False
    summary.vspreview_manual_offsets = dict(manual_trim_starts)
    summary.vspreview_manual_deltas = dict(delta_map)
    summary.measurements = tuple(measurements)

    existing_details = summary.measured_offsets
    detail_cls = _measurement_detail_cls()
    detail_map: Dict[str, "AudioMeasurementDetail"] = {}
    for measurement in measurements:
        clip_name = measurement.file.name
        prev_detail = existing_details.get(clip_name)
        plan = plan_lookup.get(clip_name)
        label = (
            prev_detail.label
            if prev_detail
            else (_plan_label(plan) if plan is not None else clip_name)
        )
        descriptor = prev_detail.stream if prev_detail else ""
        seconds_value = float(measurement.offset_seconds) if measurement.offset_seconds is not None else None
        frames_value = int(measurement.frames) if measurement.frames is not None else None
        correlation_value = float(measurement.correlation)
        status_text = summary.statuses.get(clip_name, "manual")
        note_text = notes_map.get(clip_name)
        detail_map[clip_name] = detail_cls(
            label=label,
            stream=descriptor,
            offset_seconds=seconds_value,
            frames=frames_value,
            correlation=correlation_value,
            status=status_text,
            applied=True,
            note=note_text,
        )
    summary.measured_offsets = detail_map

    audio_block = json_tail.setdefault("audio_alignment", {})
    audio_block["suggestion_mode"] = False
    audio_block["manual_trim_starts"] = dict(manual_trim_starts)
    audio_block["vspreview_manual_offsets"] = dict(manual_trim_starts)
    audio_block["vspreview_manual_deltas"] = dict(delta_map)
    audio_block["vspreview_reference_trim"] = int(
        manual_trim_starts.get(reference_name, int(reference_plan.trim_start))
    )

    offsets_sec_block: Dict[str, float] = {}
    offsets_frames_block: Dict[str, int] = {}
    for clip_name, detail in detail_map.items():
        if clip_name == reference_name and len(detail_map) > 1:
            continue
        if detail.offset_seconds is not None:
            offsets_sec_block[detail.label] = float(detail.offset_seconds)
        if detail.frames is not None:
            offsets_frames_block[detail.label] = int(detail.frames)
    audio_block["offsets_sec"] = cast(dict[str, object], dict(offsets_sec_block))
    audio_block["offsets_frames"] = cast(
        dict[str, object], dict(offsets_frames_block)
    )

    if display is not None:
        offsets_sec: Dict[str, float] = {}
        offsets_frames: Dict[str, int] = {}
        offset_lines: List[str] = []
        for clip_name in measurement_order:
            detail = detail_map.get(clip_name)
            if detail is None:
                continue
            if clip_name == reference_name and len(detail_map) > 1:
                continue
            stream_text = detail.stream or "?"
            seconds_text = (
                f"{detail.offset_seconds:+.3f}s"
                if detail.offset_seconds is not None
                else "n/a"
            )
            frames_text = f"{detail.frames:+d}f" if detail.frames is not None else "n/a"
            corr_text = (
                f"{detail.correlation:.2f}"
                if detail.correlation is not None and not math.isnan(detail.correlation)
                else "n/a"
            )
            status_text = detail.status or "manual"
            offset_lines.append(
                f"Audio offsets: {detail.label}: [{stream_text}] {seconds_text} ({frames_text}) "
                f"corr={corr_text} status={status_text}"
            )
            if detail.note:
                offset_lines.append(f"  note: {detail.note}")
            if detail.offset_seconds is not None:
                offsets_sec[detail.label] = float(detail.offset_seconds)
            if detail.frames is not None:
                offsets_frames[detail.label] = int(detail.frames)
        if not offset_lines:
            offset_lines.append("Audio offsets: VSPreview manual offsets applied")
        else:
            offset_lines.insert(0, "Audio offsets: VSPreview manual offsets applied")
        if display.manual_trim_lines:
            offset_lines.extend(display.manual_trim_lines)
        display.offset_lines = offset_lines
        display.json_offsets_sec = offsets_sec
        display.json_offsets_frames = offsets_frames
        display.measurements = {
            detail.label: detail for detail in detail_map.values()
        }
        display.correlations = {
            detail.label: detail.correlation
            for detail in detail_map.values()
            if detail.correlation is not None
        }

    reporter.line("VSPreview offsets saved to offsets file with manual status.")



def resolve_command(script_path: Path) -> tuple[list[str] | None, str | None]:
    """Return the VSPreview launch command or a reason string when unavailable."""

    executable = shutil.which("vspreview")
    if executable:
        return [executable, str(script_path)], None
    module_spec = importlib.util.find_spec("vspreview")
    if module_spec is None:
        return None, "vspreview-executable-missing"
    backend_spec = importlib.util.find_spec("PySide6") or importlib.util.find_spec("PyQt5")
    if backend_spec is None:
        return None, "vspreview-backend-missing"
    return [sys.executable, "-m", "vspreview", str(script_path)], None


def _activate_missing_panel(
    reporter: CliOutputManagerProtocol,
    command_data: _VSPreviewCommandData,
) -> None:
    manual_command = command_data.manual_command
    reason = command_data.missing_reason or "vspreview-missing"
    vspreview_values_obj = reporter.values.get("vspreview")
    vspreview_mapping_input = (
        cast(Mapping[str, object], vspreview_values_obj)
        if isinstance(vspreview_values_obj, MappingABC)
        else None
    )
    vspreview_block = _coerce_str_mapping(vspreview_mapping_input)
    missing_block_obj = vspreview_block.get("missing")
    missing_block: dict[str, object]
    if isinstance(missing_block_obj, MappingABC):
        missing_block = _coerce_str_mapping(cast(Mapping[str, object], missing_block_obj))
    else:
        missing_block = {
            "windows_install": VSPREVIEW_WINDOWS_INSTALL,
            "posix_install": VSPREVIEW_POSIX_INSTALL,
        }
    if "windows_install" not in missing_block:
        missing_block["windows_install"] = VSPREVIEW_WINDOWS_INSTALL
    if "posix_install" not in missing_block:
        missing_block["posix_install"] = VSPREVIEW_POSIX_INSTALL
    missing_block["command"] = manual_command
    missing_block["reason"] = reason
    missing_block["active"] = True
    vspreview_block["missing"] = missing_block
    vspreview_block["script_command"] = manual_command
    reporter.update_values({"vspreview": vspreview_block})
    reporter.render_sections(["vspreview_missing"])


def _report_missing(
    reporter: CliOutputManagerProtocol,
    json_tail: JsonTail,
    command_data: _VSPreviewCommandData,
) -> None:
    reason = command_data.missing_reason or "vspreview-missing"
    manual_command = command_data.manual_command
    _activate_missing_panel(reporter, command_data)
    width_lines = [
        "VSPreview dependency missing. Install with:",
        f"  Windows: {VSPREVIEW_WINDOWS_INSTALL}",
        f"  Linux/macOS: {VSPREVIEW_POSIX_INSTALL}",
        f"Then run: {manual_command}",
    ]
    for line in width_lines:
        reporter.console.print(Text(line, no_wrap=True))
    reporter.warn(
        "VSPreview dependencies missing. Install with "
        f"'{VSPREVIEW_WINDOWS_INSTALL}' (Windows) or "
        f"'{VSPREVIEW_POSIX_INSTALL}' (Linux/macOS), then run "
        f"'{manual_command}'."
    )
    json_tail["vspreview_offer"] = {"vspreview_offered": False, "reason": reason}


def launch(
    plans: Sequence[ClipPlan],
    summary: Optional["AudioAlignmentSummary"],
    display: Optional["AudioAlignmentDisplayData"],
    cfg: AppConfig,
    root: Path,
    reporter: CliOutputManagerProtocol,
    json_tail: JsonTail,
    process_runner: ProcessRunner | None = None,
) -> None:
    """Render and launch VSPreview, prompting for manual offsets when available."""

    audio_block = _ensure_audio_alignment_block(json_tail)
    if "vspreview_script" not in audio_block:
        audio_block["vspreview_script"] = None
    if "vspreview_invoked" not in audio_block:
        audio_block["vspreview_invoked"] = False
    if "vspreview_exit_code" not in audio_block:
        audio_block["vspreview_exit_code"] = None

    if summary is None:
        reporter.warn("VSPreview skipped: no alignment summary available.")
        return

    if len(plans) < 2:
        reporter.warn("VSPreview skipped: need at least two clips to compare.")
        return

    script_path = write_script(plans, summary, cfg, root)
    audio_block["vspreview_script"] = str(script_path)
    reporter.console.print(
        f"[cyan]VSPreview script ready:[/cyan] {script_path}\n"
        "Edit the OFFSET_MAP values inside the script and reload VSPreview (Ctrl+R) after changes."
    )

    manual_command = format_manual_command(script_path)
    vspreview_values_obj = reporter.values.get("vspreview")
    vspreview_mapping = (
        cast(Mapping[str, object], vspreview_values_obj)
        if isinstance(vspreview_values_obj, MappingABC)
        else None
    )
    vspreview_block = _coerce_str_mapping(vspreview_mapping)
    vspreview_block["script_path"] = str(script_path)
    vspreview_block["script_command"] = manual_command
    missing_block_obj = vspreview_block.get("missing")
    missing_block: dict[str, object]
    if isinstance(missing_block_obj, MappingABC):
        missing_block = _coerce_str_mapping(cast(Mapping[str, object], missing_block_obj))
    else:
        missing_block = {
            "windows_install": VSPREVIEW_WINDOWS_INSTALL,
            "posix_install": VSPREVIEW_POSIX_INSTALL,
        }
    missing_block["active"] = False
    if "windows_install" not in missing_block:
        missing_block["windows_install"] = VSPREVIEW_WINDOWS_INSTALL
    if "posix_install" not in missing_block:
        missing_block["posix_install"] = VSPREVIEW_POSIX_INSTALL
    missing_block["command"] = manual_command
    if "reason" not in missing_block:
        missing_block["reason"] = ""
    vspreview_block["missing"] = missing_block
    reporter.update_values({"vspreview": vspreview_block})

    if not sys.stdin.isatty():
        reporter.warn(
            "VSPreview launch skipped (non-interactive session). Open the script manually if needed."
        )
        return

    env = dict(os.environ)
    search_paths = getattr(cfg.runtime, "vapoursynth_python_paths", [])
    if search_paths:
        env["VAPOURSYNTH_PYTHONPATH"] = os.pathsep.join(
            str(Path(path).expanduser()) for path in search_paths if path
        )
    else:
        logger.info(
            "VAPOURSYNTH_PYTHONPATH not set; configure [runtime].vapoursynth_python_paths "
            "to help VSPreview discover the VapourSynth site-packages path."
        )

    command, missing_reason = resolve_command(script_path)
    if command is None:
        logger.warning(
            "VSPreview executable unavailable: %s",
            missing_reason or "vspreview-missing",
        )
        missing_data = _VSPreviewCommandData(
            script_path=script_path,
            manual_command=manual_command,
            missing_reason=missing_reason or "vspreview-missing",
        )
        _report_missing(
            reporter,
            json_tail,
            missing_data,
        )
        return

    verbose_requested = bool(reporter.flags.get("verbose")) or bool(
        reporter.flags.get("debug")
    )

    runner: ProcessRunner = process_runner or subproc.run_checked  # type: ignore[assignment]

    try:
        if verbose_requested:
            result = runner(
                command,
                env=env,
                check=False,
                stdin=None,
                stdout=None,
                stderr=None,
                text=True,
            )
        else:
            result = runner(
                command,
                env=env,
                check=False,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
    except FileNotFoundError:
        missing_data = _VSPreviewCommandData(
            script_path=script_path,
            manual_command=manual_command,
            missing_reason="vspreview-missing",
        )
        _report_missing(
            reporter,
            json_tail,
            missing_data,
        )
        return
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        logger.warning(
            "VSPreview launch failed: %s",
            exc,
            exc_info=True,
        )
        reporter.warn(f"VSPreview launch failed: {exc}")
        return
    audio_block["vspreview_invoked"] = True
    audio_block["vspreview_exit_code"] = int(result.returncode)
    captured_stdout = getattr(result, "stdout", None)
    captured_stderr = getattr(result, "stderr", None)
    if not verbose_requested:
        for stream_value, label in ((captured_stdout, "stdout"), (captured_stderr, "stderr")):
            if isinstance(stream_value, str) and stream_value.strip():
                logger.debug("VSPreview %s (suppressed): %s", label, stream_value.strip())
    if result.returncode != 0:
        reporter.warn(
            f"VSPreview exited with code {result.returncode}."
            + (" Re-run with --verbose to inspect VSPreview output." if not verbose_requested else "")
        )
        return

    offsets = prompt_offsets(plans, summary, reporter, display)
    if offsets is None:
        return
    apply_manual_offsets(plans, summary, offsets, reporter, json_tail, display)


# Backwards-compatible aliases for downstream imports.
_render_vspreview_script = render_script
_persist_vspreview_script = persist_script
_write_vspreview_script = write_script
_prompt_vspreview_offsets = prompt_offsets
_apply_vspreview_manual_offsets = apply_manual_offsets
_launch_vspreview = launch
_format_vspreview_manual_command = format_manual_command
_resolve_vspreview_command = resolve_command
_VSPREVIEW_WINDOWS_INSTALL = VSPREVIEW_WINDOWS_INSTALL
_VSPREVIEW_POSIX_INSTALL = VSPREVIEW_POSIX_INSTALL

__all__ = [
    "ProcessRunner",
    "VSPREVIEW_WINDOWS_INSTALL",
    "VSPREVIEW_POSIX_INSTALL",
    "apply_manual_offsets",
    "format_manual_command",
    "launch",
    "persist_script",
    "prompt_offsets",
    "render_script",
    "resolve_command",
    "write_script",
]
