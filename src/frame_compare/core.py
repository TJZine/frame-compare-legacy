"""CLI entry point and orchestration logic for frame comparison runs."""
# pyright: reportUnusedFunction=false

from __future__ import annotations

import datetime as _dt  # noqa: F401  (re-exported via frame_compare)
import importlib as _importlib
import logging
import math
import time as _time
from typing import Any, Final, List, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple, cast

import click
from rich.console import Console as _Console  # noqa: F401
from rich.progress import Progress as _Progress  # noqa: F401
from rich.progress import ProgressColumn as _ProgressColumn

import src.frame_compare.alignment as _alignment_package
import src.frame_compare.alignment_preview as _alignment_preview_module
import src.frame_compare.doctor as _doctor_module
import src.frame_compare.planner as _planner_module
import src.frame_compare.preflight as _preflight_constants
import src.frame_compare.vspreview as _vspreview_module
import src.frame_compare.wizard as _wizard_module
import src.screenshot as _screenshot_module
from src import audio_alignment as _audio_alignment_module
from src.config_loader import load_config as _load_config
from src.frame_compare.analysis import (
    export_selection_metadata as _export_selection_metadata,  # noqa: F401
)
from src.frame_compare.analysis import (
    probe_cached_metrics as _probe_cached_metrics,  # noqa: F401
)
from src.frame_compare.analysis import (
    select_frames as _select_frames,  # noqa: F401  # type: ignore[reportUnknownVariableType]
)
from src.frame_compare.analysis import (
    selection_details_to_json as _selection_details_to_json,  # noqa: F401
)
from src.frame_compare.analysis import (
    selection_hash_for_config as _selection_hash_for_config,  # noqa: F401
)
from src.frame_compare.analysis import (
    write_selection_cache_file as _write_selection_cache_file,  # noqa: F401
)
from src.frame_compare.analyze_target import pick_analyze_file as _pick_analyze_file
from src.frame_compare.cli_runtime import (
    AudioAlignmentJSON as _AudioAlignmentJSON,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    CLIAppError as _CLIAppError,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    ClipPlan,
    coerce_str_mapping,
)
from src.frame_compare.cli_runtime import (
    ClipRecord as _ClipRecord,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    JsonTail as _JsonTail,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    ReportJSON as _ReportJSON,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    SlowpicsJSON as _SlowpicsJSON,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    SlowpicsTitleBlock as _SlowpicsTitleBlock,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    SlowpicsTitleInputs as _SlowpicsTitleInputs,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    TrimClipEntry as _TrimClipEntry,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    TrimsJSON as _TrimsJSON,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    TrimSummary as _TrimSummary,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.cli_runtime import (
    ViewerJSON as _ViewerJSON,  # noqa: F401 - re-exported for compatibility
)
from src.frame_compare.preflight import (
    PACKAGED_TEMPLATE_PATH as _PACKAGED_TEMPLATE_PATH,  # noqa: F401 - compatibility re-export
)
from src.frame_compare.preflight import (
    PROJECT_ROOT as _PROJECT_ROOT,  # noqa: F401 - compatibility re-export
)
from src.frame_compare.preflight import (
    PreflightResult as _PreflightResult,
)
from src.frame_compare.preflight import abort_if_site_packages as _abort_if_site_packages_public
from src.frame_compare.preflight import (
    collect_path_diagnostics as _collect_path_diagnostics,
)
from src.frame_compare.preflight import (
    fresh_app_config as _fresh_app_config_public,
)
from src.frame_compare.preflight import (
    prepare_preflight as _prepare_preflight,
)
from src.frame_compare.slowpics import (
    SlowpicsAPIError as _SlowpicsAPIError,
)  # noqa: F401
from src.frame_compare.slowpics import (
    build_shortcut_filename as _build_shortcut_filename,
)
from src.frame_compare.slowpics import (
    upload_comparison as _upload_comparison,
)
from src.frame_compare.vs import ClipInitError as _ClipInitError
from src.frame_compare.vs import ClipProcessError as _ClipProcessError
from src.tmdb import resolve_tmdb as _resolve_tmdb_public

logger = logging.getLogger(__name__)

CONFIG_ENV_VAR: Final[str] = _preflight_constants.CONFIG_ENV_VAR
NO_WIZARD_ENV_VAR: Final[str] = _preflight_constants.NO_WIZARD_ENV_VAR
ROOT_ENV_VAR: Final[str] = _preflight_constants.ROOT_ENV_VAR
ROOT_SENTINELS: Final[tuple[str, ...]] = _preflight_constants.ROOT_SENTINELS
resolve_workspace_root = _preflight_constants.resolve_workspace_root

ScreenshotError = _screenshot_module.ScreenshotError
generate_screenshots = _screenshot_module.generate_screenshots
fresh_app_config = _fresh_app_config_public
_fresh_app_config = fresh_app_config
confirm_alignment_with_screenshots = _alignment_preview_module.confirm_alignment_with_screenshots
_confirm_alignment_with_screenshots = confirm_alignment_with_screenshots
load_config = _load_config
build_plans = _planner_module.build_plans
_build_plans = _planner_module.build_plans
_ClipPlan = ClipPlan
_coerce_str_mapping = coerce_str_mapping
pick_analyze_file = _pick_analyze_file
datetime = _dt
time = _time
importlib = _importlib
Console = _Console
Progress = _Progress
ProgressColumn = _ProgressColumn
export_selection_metadata = cast(Any, _export_selection_metadata)
probe_cached_metrics = cast(Any, _probe_cached_metrics)
select_frames = cast(Any, _select_frames)
selection_details_to_json = cast(Any, _selection_details_to_json)
selection_hash_for_config = cast(Any, _selection_hash_for_config)
write_selection_cache_file = cast(Any, _write_selection_cache_file)
resolve_tmdb = _resolve_tmdb_public
AudioAlignmentJSON = _AudioAlignmentJSON
CLIAppError = _CLIAppError
ClipRecord = _ClipRecord
JsonTail = _JsonTail
ReportJSON = _ReportJSON
SlowpicsJSON = _SlowpicsJSON
SlowpicsTitleBlock = _SlowpicsTitleBlock
SlowpicsTitleInputs = _SlowpicsTitleInputs
TrimClipEntry = _TrimClipEntry
TrimsJSON = _TrimsJSON
TrimSummary = _TrimSummary
ViewerJSON = _ViewerJSON
PACKAGED_TEMPLATE_PATH = _PACKAGED_TEMPLATE_PATH
PROJECT_ROOT = _PROJECT_ROOT
PreflightResult = _PreflightResult
collect_path_diagnostics = _collect_path_diagnostics
prepare_preflight = _prepare_preflight
SlowpicsAPIError = _SlowpicsAPIError
build_shortcut_filename = _build_shortcut_filename
upload_comparison = _upload_comparison
_PathPreflightResult = PreflightResult
_collect_path_diagnostics = collect_path_diagnostics
_prepare_preflight = prepare_preflight

audio_alignment = _audio_alignment_module
AudioAlignmentSummary = _alignment_package.AudioAlignmentSummary
AudioAlignmentDisplayData = _alignment_package.AudioAlignmentDisplayData
AudioMeasurementDetail = _alignment_package.AudioMeasurementDetail
_AudioAlignmentSummary = AudioAlignmentSummary
_AudioAlignmentDisplayData = AudioAlignmentDisplayData
_AudioMeasurementDetail = AudioMeasurementDetail
apply_audio_alignment = _alignment_package.apply_audio_alignment
format_alignment_output = _alignment_package.format_alignment_output
_maybe_apply_audio_alignment = _alignment_package.apply_audio_alignment
resolve_alignment_reference = _alignment_package.resolve_alignment_reference
_resolve_alignment_reference = resolve_alignment_reference
_prompt_vspreview_offsets = _vspreview_module.prompt_offsets
_apply_vspreview_manual_offsets = _vspreview_module.apply_manual_offsets
_write_vspreview_script = _vspreview_module.write_script
_launch_vspreview = _vspreview_module.launch
_format_vspreview_manual_command = _vspreview_module.format_manual_command
_VSPREVIEW_WINDOWS_INSTALL = _vspreview_module.VSPREVIEW_WINDOWS_INSTALL
_VSPREVIEW_POSIX_INSTALL = _vspreview_module.VSPREVIEW_POSIX_INSTALL
ClipInitError = _ClipInitError
ClipProcessError = _ClipProcessError

_DEFAULT_CONFIG_HELP: Final[str] = (
    "Optional explicit path to config.toml. When omitted, Frame Compare looks for "
    "ROOT/config/config.toml (see --root/FRAME_COMPARE_ROOT)."
)


DoctorStatus = _doctor_module.DoctorStatus
DoctorCheck = _doctor_module.DoctorCheck
resolve_wizard_paths = _wizard_module.resolve_wizard_paths
_resolve_wizard_paths = resolve_wizard_paths
abort_if_site_packages = _abort_if_site_packages_public
_abort_if_site_packages = abort_if_site_packages

def _extract_clip_fps(clip: object) -> Tuple[int, int]:
    """Return (fps_num, fps_den) from *clip*, defaulting to 24000/1001 when missing."""
    num = getattr(clip, "fps_num", None)
    den = getattr(clip, "fps_den", None)
    if isinstance(num, int) and isinstance(den, int) and den:
        return (num, den)
    return (24000, 1001)


def _format_seconds(value: float) -> str:
    """
    Format a time value in seconds as an HH:MM:SS.s string with one decimal place.

    Negative input is treated as zero. The seconds component is rounded to one decimal place and may carry into minutes (and similarly minutes into hours) when rounding produces overflow.

    Parameters:
        value (float): Time in seconds.

    Returns:
        str: Formatted time as "HH:MM:SS.s" with two-digit hours and minutes and one decimal place for seconds.
    """
    total = max(0.0, float(value))
    hours = int(total // 3600)
    minutes = int((total - hours * 3600) // 60)
    seconds = total - hours * 3600 - minutes * 60
    seconds = round(seconds, 1)
    if seconds >= 60.0:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        hours += 1
    return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"


def _fps_to_float(value: Tuple[int, int] | None) -> float:
    """
    Convert an FPS expressed as a (numerator, denominator) tuple into a floating-point frames-per-second value.

    Parameters:
        value ((int, int) | None): A two-integer tuple representing FPS as (numerator, denominator). May be None.

    Returns:
        float: The FPS as a float. Returns 0.0 if `value` is None, the denominator is zero, or the tuple is invalid.
    """
    if not value:
        return 0.0
    num, den = value
    if not den:
        return 0.0
    return float(num) / float(den)


def _fold_sequence(
    values: Sequence[object],
    *,
    head: int,
    tail: int,
    joiner: str,
    enabled: bool,
) -> str:
    """
    Produce a compact string representation of a sequence by optionally folding the middle elements with an ellipsis.

    Parameters:
        values (Sequence[object]): Items to render; each item is stringified.
        head (int): Number of items to keep from the start when folding is enabled.
        tail (int): Number of items to keep from the end when folding is enabled.
        joiner (str): Separator used to join items.
        enabled (bool): If True and the sequence is longer than head + tail, replace the omitted middle with "…".

    Returns:
        str: The joined string containing all items when folding is disabled or not needed, or a string containing the head items, a single "…" token, and the tail items when folding is applied.
    """
    items = [str(item) for item in values]
    if not enabled or len(items) <= head + tail:
        return joiner.join(items)
    head_items = items[: max(0, head)]
    tail_items = items[-max(0, tail) :]
    if not head_items:
        return joiner.join(tail_items)
    if not tail_items:
        return joiner.join(head_items)
    return joiner.join([*head_items, "…", *tail_items])


def _evaluate_rule_condition(condition: Optional[str], *, flags: Mapping[str, Any]) -> bool:
    """
    Evaluate a simple rule condition string against a mapping of flags.

    The condition may be None/empty (treated as satisfied), a flag name (satisfied if the flag is truthy), or a negated flag name prefixed with `!`. Known tokens `verbose` and `upload_enabled` are supported like any other flag name.

    Parameters:
        condition (Optional[str]): The rule expression to evaluate (e.g. "verbose", "!upload_enabled") or None/empty to always satisfy.
        flags (Mapping[str, Any]): Mapping of flag names to values; values are interpreted by their truthiness.

    Returns:
        True if the condition is satisfied given `flags`, False otherwise.
    """
    if not condition:
        return True
    expr = condition.strip()
    if not expr:
        return True
    if expr == "!verbose":
        return not bool(flags.get("verbose"))
    if expr == "verbose":
        return bool(flags.get("verbose"))
    if expr == "upload_enabled":
        return bool(flags.get("upload_enabled"))
    if expr == "!upload_enabled":
        return not bool(flags.get("upload_enabled"))
    return bool(flags.get(expr))


def _build_legacy_summary_lines(values: Mapping[str, Any], *, emit_json_tail: bool) -> List[str]:
    """
    Generate legacy human-readable summary lines from the collected layout values.

    Parameters:
        values (Mapping[str, Any]): Mapping containing layout sections (for example:
            "clips", "window", "analysis", "audio_alignment", "render",
            "tonemap", "cache"). The function reads specific keys from those
            sections to synthesize compact summary lines.

    Returns:
        List[str]: A list of non-empty summary lines suitable for the legacy
        textual summary display.
    """

    def _maybe_number(value: Any) -> float | None:
        """Convert numeric-like input to float, returning ``None`` on failure."""
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _format_number(value: Any, fmt: str, fallback: str) -> str:
        """Format numeric values with ``fmt``; fall back to the provided string."""
        number = _maybe_number(value)
        if number is None:
            return fallback
        return format(number, fmt)

    def _string(value: Any, fallback: str = "n/a") -> str:
        """Return lowercase booleans, fallback for ``None``, else ``str(value)``."""
        if value is None:
            return fallback
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _bool_text(value: Any) -> str:
        """
        Format a value as lowercase boolean text.

        Returns:
            `'true'` if the value evaluates to True, `'false'` otherwise.
        """
        return "true" if bool(value) else "false"

    clips = _coerce_str_mapping(values.get("clips"))
    window = _coerce_str_mapping(values.get("window"))
    analysis = _coerce_str_mapping(values.get("analysis"))
    counts = _coerce_str_mapping(analysis.get("counts")) if analysis else {}
    audio = _coerce_str_mapping(values.get("audio_alignment"))
    render = _coerce_str_mapping(values.get("render"))
    tonemap = _coerce_str_mapping(values.get("tonemap"))
    cache = _coerce_str_mapping(values.get("cache"))

    lines: List[str] = []

    clip_count = _string(clips.get("count"), "0")
    lead_text = _format_number(window.get("ignore_lead_seconds"), ".2f", "0.00")
    trail_text = _format_number(window.get("ignore_trail_seconds"), ".2f", "0.00")
    step_text = _string(analysis.get("step"), "0")
    downscale_text = _string(analysis.get("downscale_height"), "0")
    lines.append(
        f"Clips: {clip_count}  Window: lead={lead_text}s trail={trail_text}s  step={step_text} downscale={downscale_text}px"
    )

    offsets_text = _format_number(audio.get("offsets_sec"), "+.3f", "+0.000")
    offsets_file = _string(audio.get("offsets_filename"), "n/a")
    lines.append(
        f"Align: audio={_bool_text(audio.get('enabled'))}  offsets={offsets_text}s  file={offsets_file}"
    )

    lines.append(
        "Plan: "
        f"Dark={_string(counts.get('dark'), '0')} "
        f"Bright={_string(counts.get('bright'), '0')} "
        f"Motion={_string(counts.get('motion'), '0')} "
        f"Random={_string(counts.get('random'), '0')} "
        f"User={_string(counts.get('user'), '0')}  "
        f"sep={_format_number(analysis.get('screen_separation_sec'), '.1f', '0.0')}s"
    )

    lines.append(
        "Canvas: "
        f"single_res={_string(render.get('single_res'), '0')} "
        f"upscale={_bool_text(render.get('upscale'))} "
        f"crop=mod{_string(render.get('mod_crop'), '0')} "
        f"pad={_bool_text(render.get('center_pad'))}"
    )

    tonemap_curve = _string(tonemap.get("tone_curve"), "n/a")
    tonemap_target = _format_number(tonemap.get("target_nits"), ".0f", "0")
    tonemap_dst_min = _format_number(tonemap.get("dst_min_nits"), ".2f", "0.00")
    tonemap_knee = _format_number(tonemap.get("knee_offset"), ".2f", "0.00")
    tonemap_preset_label = _string(tonemap.get("dpd_preset"), "n/a")
    tonemap_cutoff = _format_number(tonemap.get("dpd_black_cutoff"), ".3f", "0.000")
    tonemap_gamma = _format_number(tonemap.get("post_gamma"), ".2f", "1.00")
    gamma_flag = "*" if bool(tonemap.get("post_gamma_enabled")) else ""
    dpd_enabled = bool(
        tonemap.get("dpd")
        if "dpd" in tonemap
        else tonemap.get("dynamic_peak_detection")
    )
    preset_suffix = f" ({tonemap_preset_label})" if dpd_enabled and tonemap_preset_label.lower() != "n/a" else ""
    lines.append(
        "Tonemap: "
        f"{tonemap_curve}@{tonemap_target}nits "
        f"dst_min={tonemap_dst_min} knee={tonemap_knee} "
        f"dpd={_bool_text(dpd_enabled)}"
        f"{preset_suffix} black_cutoff={tonemap_cutoff}  "
        f"gamma={tonemap_gamma}{gamma_flag}  "
        f"verify≤{_format_number(tonemap.get('verify_luma_threshold'), '.2f', '0.00')}"
    )

    lines.append(
        f"Output: {_string(render.get('out_dir'), 'n/a')}  compression={_string(render.get('compression'), 'n/a')}"
    )

    lines.append(f"Cache: {_string(cache.get('file'), 'n/a')}  {_string(cache.get('status'), 'unknown')}")

    frame_count = _string(analysis.get("output_frame_count"), "0")
    preview = _string(analysis.get("output_frames_preview"), "")
    preview_display = f"[{preview}]" if preview else "[]"
    if emit_json_tail:
        lines.append(
            f"Output frames: {frame_count}  e.g., {preview_display}  (full list in JSON)"
        )
    else:
        full_list = _string(analysis.get("output_frames_full"), "[]")
        lines.append(f"Output frames ({frame_count}): {full_list}")

    return [line for line in lines if line]


def _format_clock(seconds: Optional[float]) -> str:
    """Format seconds as H:MM:SS (or MM:SS) with a placeholder for invalid input."""
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    total = max(0, int(seconds + 0.5))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _validate_tonemap_overrides(overrides: MutableMapping[str, Any]) -> None:
    """Validate CLI-provided tonemap overrides and raise ClickException on invalid values."""

    if not overrides:
        return

    def _bad(message: str) -> NoReturn:
        raise click.ClickException(message)

    parsed_floats: dict[str, float] = {}

    def _get_float(key: str, error_message: str) -> float:
        if key in parsed_floats:
            return parsed_floats[key]
        try:
            value = float(overrides[key])
        except (TypeError, ValueError):
            _bad(error_message)
        if not math.isfinite(value):
            _bad(error_message)
        parsed_floats[key] = value
        return value

    if "knee_offset" in overrides:
        knee_value = _get_float(
            "knee_offset",
            "--tm-knee must be a finite number in [0.0, 1.0]",
        )
        if knee_value < 0.0 or knee_value > 1.0:
            _bad("--tm-knee must be between 0.0 and 1.0")
    if "dst_min_nits" in overrides:
        dst_value = _get_float(
            "dst_min_nits",
            "--tm-dst-min must be a finite, non-negative number",
        )
        if dst_value < 0.0:
            _bad("--tm-dst-min must be >= 0.0")
    if "target_nits" in overrides:
        target_value = _get_float(
            "target_nits",
            "--tm-target must be a finite, positive number",
        )
        if target_value <= 0.0:
            _bad("--tm-target must be > 0")
    if "post_gamma" in overrides:
        gamma_value = _get_float(
            "post_gamma",
            "--tm-gamma must be a finite number between 0.9 and 1.1",
        )
        if gamma_value < 0.9 or gamma_value > 1.1:
            _bad("--tm-gamma must be between 0.9 and 1.1")
    if "dpd_preset" in overrides:
        dpd_value = str(overrides["dpd_preset"]).strip().lower()
        if dpd_value not in {"off", "fast", "balanced", "high_quality"}:
            _bad("--tm-dpd-preset must be one of: off, fast, balanced, high_quality")
    if "dpd_black_cutoff" in overrides:
        cutoff = _get_float(
            "dpd_black_cutoff",
            "--tm-dpd-black-cutoff must be a finite number in [0.0, 0.05]",
        )
        if cutoff < 0.0 or cutoff > 0.05:
            _bad("--tm-dpd-black-cutoff must be between 0.0 and 0.05")
    if "smoothing_period" in overrides:
        smoothing = _get_float(
            "smoothing_period",
            "--tm-smoothing must be a finite, non-negative number",
        )
        if smoothing < 0.0:
            _bad("--tm-smoothing must be >= 0")
    if "scene_threshold_low" in overrides:
        low_value = _get_float(
            "scene_threshold_low",
            "--tm-scene-low must be a finite, non-negative number",
        )
        if low_value < 0.0:
            _bad("--tm-scene-low must be >= 0")
    if "scene_threshold_high" in overrides:
        high_value = _get_float(
            "scene_threshold_high",
            "--tm-scene-high must be a finite, non-negative number",
        )
        if high_value < 0.0:
            _bad("--tm-scene-high must be >= 0")
    if "scene_threshold_low" in overrides and "scene_threshold_high" in overrides:
        high_value = parsed_floats["scene_threshold_high"]
        low_value = parsed_floats["scene_threshold_low"]
        if high_value < low_value:
            _bad("--tm-scene-high must be >= --tm-scene-low")
    if "percentile" in overrides:
        percentile = _get_float(
            "percentile",
            "--tm-percentile must be a finite number between 0 and 100",
        )
        if percentile < 0.0 or percentile > 100.0:
            _bad("--tm-percentile must be between 0 and 100")
    if "contrast_recovery" in overrides:
        contrast = _get_float(
            "contrast_recovery",
            "--tm-contrast must be a finite, non-negative number",
        )
        if contrast < 0.0:
            _bad("--tm-contrast must be >= 0")
    if "metadata" in overrides:
        meta_value = overrides["metadata"]
        if isinstance(meta_value, str):
            lowered = meta_value.strip().lower()
            if lowered in {"auto", ""}:
                overrides["metadata"] = "auto"
            elif lowered in {"none", "hdr10", "hdr10+", "hdr10plus", "luminance"}:
                overrides["metadata"] = lowered
            else:
                try:
                    meta_int = int(lowered)
                except ValueError:
                    _bad("--tm-metadata must be auto, none, hdr10, hdr10+, luminance, or 0-4")
                else:
                    if meta_int < 0 or meta_int > 4:
                        _bad("--tm-metadata integer must be between 0 and 4")
                    overrides["metadata"] = meta_int
        else:
            try:
                meta_int = int(meta_value)
            except (TypeError, ValueError):
                _bad("--tm-metadata must be auto, none, hdr10, hdr10+, luminance, or 0-4")
            else:
                if meta_int < 0 or meta_int > 4:
                    _bad("--tm-metadata integer must be between 0 and 4")
    if "use_dovi" in overrides:
        if overrides["use_dovi"] not in {None, True, False}:
            _bad("--tm-use-dovi/--tm-no-dovi must be specified without a value")
    for boolean_key in ("visualize_lut", "show_clipping"):
        if boolean_key in overrides and not isinstance(overrides[boolean_key], bool):
            _bad(f"--tm-{boolean_key.replace('_', '-')} must be used without a value")


validate_tonemap_overrides = _validate_tonemap_overrides
