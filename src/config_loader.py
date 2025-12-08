"""Configuration loader that parses and validates user-provided TOML."""

from __future__ import annotations

import logging
import math
import tomllib
import warnings
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from .datatypes import (
    AnalysisConfig,
    AnalysisThresholdMode,
    AnalysisThresholds,
    AppConfig,
    AudioAlignmentConfig,
    AutoLetterboxCropMode,
    CLIConfig,
    ColorConfig,
    DiagnosticsConfig,
    NamingConfig,
    OverridesConfig,
    PathsConfig,
    ReportConfig,
    RunnerConfig,
    RuntimeConfig,
    ScreenshotConfig,
    SlowpicsConfig,
    SourceConfig,
    TMDBConfig,
)


class ConfigError(ValueError):
    """Raised when the configuration file is malformed or fails validation."""


_FFMPEG_TIMEOUT_NOT_NUMBER_MSG = "screenshots.ffmpeg_timeout_seconds must be a number"
_FFMPEG_TIMEOUT_NOT_FINITE_MSG = (
    "screenshots.ffmpeg_timeout_seconds must be a finite number"
)
_FFMPEG_TIMEOUT_NEGATIVE_MSG = "screenshots.ffmpeg_timeout_seconds must be >= 0"


def _coerce_bool(value: Any, dotted_key: str) -> bool:
    """Return a bool, coercing simple 0/1 representations when necessary."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "1"}:
            return normalized == "1"
        if normalized in {"true", "false"}:
            return normalized == "true"
    raise ConfigError(f"{dotted_key} must be a boolean (use true/false).")


def _coerce_enum(value: Any, dotted_key: str, enum_type: type[Enum]) -> Enum:
    """Return an enum member, coercing string values case-insensitively."""

    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for member in enum_type:
            member_value = str(member.value).lower()
            if normalized == member_value:
                return member
    raise ConfigError(
        f"{dotted_key} must be one of: {', '.join(str(member.value) for member in enum_type)}"
    )


def _normalise_auto_letterbox_mode(value: object, dotted_key: str) -> str:
    """Return a canonical auto letterbox mode label."""

    if isinstance(value, AutoLetterboxCropMode):
        resolved = value.value
    elif isinstance(value, bool):
        return AutoLetterboxCropMode.STRICT.value if value else AutoLetterboxCropMode.OFF.value
    elif value is None:
        return AutoLetterboxCropMode.OFF.value
    else:
        resolved = str(value).strip().lower()

    if resolved in {"", "off", "false"}:
        return AutoLetterboxCropMode.OFF.value
    if resolved == AutoLetterboxCropMode.BASIC.value:
        return AutoLetterboxCropMode.BASIC.value
    if resolved in {AutoLetterboxCropMode.STRICT.value, "true"}:
        return AutoLetterboxCropMode.STRICT.value
    raise ConfigError(
        f"{dotted_key} must be 'off', 'basic', 'strict', true/false, or omitted."
    )


def _sanitize_section(raw: dict[str, Any], name: str, cls):
    """
    Coerce a raw TOML table into an instance of ``cls`` with cleaned booleans.

    Parameters:
        raw (dict[str, Any]): Raw TOML section data.
        name (str): Section name used when reporting validation errors.
        cls: Dataclass type used to construct the section object.

    Returns:
        Any: Instantiated dataclass populated with values from ``raw``.

    Raises:
        ConfigError: If the section is not a table or contains invalid keys or values.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}] must be a table")
    cleaned: Dict[str, Any] = {}
    cls_fields = {field.name: field for field in fields(cls)}
    bool_fields = {name for name, field in cls_fields.items() if field.type is bool}
    enum_fields = {
        field_name: field.type
        for field_name, field in cls_fields.items()
        if isinstance(field.type, type) and issubclass(field.type, Enum)
    }
    nested_fields = {
        name: field.type
        for name, field in cls_fields.items()
        if is_dataclass(field.type)
    }
    raw_keys = set(raw.keys())
    for key, value in raw.items():
        if key in bool_fields:
            cleaned[key] = _coerce_bool(value, f"{name}.{key}")
        elif key in enum_fields:
            cleaned[key] = _coerce_enum(value, f"{name}.{key}", enum_fields[key])
        elif key in nested_fields:
            if not isinstance(value, dict):
                raise ConfigError(f"[{name}.{key}] must be a table")
            cleaned[key] = _sanitize_section(value, f"{name}.{key}", nested_fields[key])
        else:
            cleaned[key] = value
    try:
        instance = cls(**cleaned)
    except TypeError as exc:
        raise ConfigError(f"Invalid keys in [{name}]: {exc}") from exc
    effective_provided: set[str]
    if not raw_keys:
        effective_provided = set()
    else:
        try:
            default_instance = cls()
        except TypeError:
            # Some dataclasses expect arguments; keep legacy raw-key semantics.
            effective_provided = set(raw_keys)
        else:
            effective_provided = set()
            for field in fields(cls):
                field_name = field.name
                if field_name not in raw_keys:
                    continue
                try:
                    current_value = getattr(instance, field_name)
                    default_value = getattr(default_instance, field_name)
                except (AttributeError, TypeError, ValueError):
                    effective_provided.add(field_name)
                    continue
                if not _values_match(current_value, default_value):
                    effective_provided.add(field_name)
    try:
        instance._provided_keys = effective_provided
    except (AttributeError, TypeError):
        pass
    return instance


def _values_match(first: Any, second: Any) -> bool:
    """Best-effort comparison that tolerates string/number pairs."""

    if first == second:
        return True
    try:
        return float(first) == float(second)
    except (TypeError, ValueError):
        return False


def _migrate_skip_field(
    section: Dict[str, Any],
    *,
    legacy_key: str,
    target_key: str,
) -> None:
    """Map legacy skip_* fields onto ignore_* while warning users."""

    if legacy_key not in section:
        return
    value = section.pop(legacy_key)
    warnings.warn(
        f"{legacy_key} is deprecated; set {target_key} instead.",
        UserWarning,
        stacklevel=2,
    )
    existing = section.get(target_key)
    if existing is None:
        section[target_key] = value
        return
    if not _values_match(existing, value):
        raise ConfigError(
            f"{legacy_key} conflicts with {target_key}. Remove the legacy key or "
            f"ensure both values match."
        )


def _migrate_threshold_fields(section: Dict[str, Any]) -> None:
    """Backfill the structured thresholds table from legacy keys."""

    thresholds = section.get("thresholds")
    if thresholds is not None and not isinstance(thresholds, dict):
        raise ConfigError("[analysis.thresholds] must be a table")

    def _ensure_thresholds_table() -> Dict[str, Any]:
        nonlocal thresholds
        if thresholds is None:
            thresholds = {}
            section["thresholds"] = thresholds
        return thresholds

    for key in ("dark_quantile", "bright_quantile"):
        if key in section:
            warnings.warn(
                f"analysis.{key} is deprecated; set analysis.thresholds.{key} instead.",
                UserWarning,
                stacklevel=2,
            )
            _ensure_thresholds_table().setdefault(key, section.pop(key))

    use_quantiles_value = section.pop("use_quantiles", None)
    if use_quantiles_value is not None:
        mode_value = (
            AnalysisThresholdMode.QUANTILE.value
            if _coerce_bool(use_quantiles_value, "analysis.use_quantiles")
            else AnalysisThresholdMode.FIXED_RANGE.value
        )
        warnings.warn(
            "analysis.use_quantiles is deprecated; configure analysis.thresholds.mode instead.",
            UserWarning,
            stacklevel=2,
        )
        _ensure_thresholds_table().setdefault("mode", mode_value)

    if thresholds is not None:
        section["thresholds"] = thresholds


def _migrate_analysis_section(section: Dict[str, Any]) -> None:
    """Apply backwards-compatible rewrites for legacy analysis keys."""

    _migrate_skip_field(
        section,
        legacy_key="skip_head_seconds",
        target_key="ignore_lead_seconds",
    )
    _migrate_skip_field(
        section,
        legacy_key="skip_tail_seconds",
        target_key="ignore_trail_seconds",
    )
    _migrate_threshold_fields(section)


def _normalize_float(value: Any, dotted_key: str) -> float:
    """Return ``value`` as a finite float, raising ConfigError otherwise."""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{dotted_key} must be a number") from exc
    if not math.isfinite(numeric):
        raise ConfigError(f"{dotted_key} must be a finite number")
    return numeric


def _normalize_fraction(value: Any, dotted_key: str) -> float:
    """Return a float in the [0, 1] range."""

    numeric = _normalize_float(value, dotted_key)
    if numeric < 0 or numeric > 1:
        raise ConfigError(f"{dotted_key} must be between 0 and 1")
    return numeric


def _validate_thresholds(thresholds: AnalysisThresholds) -> None:
    """Validate and normalize the analysis threshold configuration."""

    mode = thresholds.mode
    if isinstance(mode, str):
        try:
            mode = AnalysisThresholdMode(mode.lower())
        except ValueError as exc:  # pragma: no cover - defensive
            raise ConfigError("analysis.thresholds.mode must be 'quantile' or 'fixed_range'") from exc
        thresholds.mode = mode

    if mode == AnalysisThresholdMode.QUANTILE:
        dark = _normalize_fraction(thresholds.dark_quantile, "analysis.thresholds.dark_quantile")
        bright = _normalize_fraction(thresholds.bright_quantile, "analysis.thresholds.bright_quantile")
        if bright <= dark:
            raise ConfigError("analysis.thresholds.bright_quantile must be > analysis.thresholds.dark_quantile")
        thresholds.dark_quantile = dark
        thresholds.bright_quantile = bright
    elif mode == AnalysisThresholdMode.FIXED_RANGE:
        dark_min = _normalize_fraction(thresholds.dark_luma_min, "analysis.thresholds.dark_luma_min")
        dark_max = _normalize_fraction(thresholds.dark_luma_max, "analysis.thresholds.dark_luma_max")
        if dark_max <= dark_min:
            raise ConfigError("analysis.thresholds.dark_luma_max must be > dark_luma_min")
        bright_min = _normalize_fraction(thresholds.bright_luma_min, "analysis.thresholds.bright_luma_min")
        bright_max = _normalize_fraction(thresholds.bright_luma_max, "analysis.thresholds.bright_luma_max")
        if bright_max <= bright_min:
            raise ConfigError("analysis.thresholds.bright_luma_max must be > bright_luma_min")
        thresholds.dark_luma_min = dark_min
        thresholds.dark_luma_max = dark_max
        thresholds.bright_luma_min = bright_min
        thresholds.bright_luma_max = bright_max
    else:  # pragma: no cover - future-proof
        raise ConfigError(
            "analysis.thresholds.mode must be one of: "
            f"{', '.join(m.value for m in AnalysisThresholdMode)}"
        )

def _validate_trim(mapping: Dict[str, Any], label: str) -> None:
    """
    Ensure all trim overrides map to integer frame counts.

    Parameters:
        mapping (Dict[str, Any]): Raw trim override mapping.
        label (str): Configuration label used in error messages.

    Raises:
        ConfigError: If any trim override is not an integer.
    """
    for key, value in mapping.items():
        if not isinstance(value, int):
            raise ConfigError(f"{label} entry '{key}' must map to an integer")


def _validate_change_fps(change_fps: Dict[str, Any]) -> None:
    """
    Validate ``change_fps`` overrides as ``"set"`` or two positive integers.

    Parameters:
        change_fps (Dict[str, Any]): Mapping from clip identifiers to override values.

    Raises:
        ConfigError: If any override is not ``"set"`` or a two-integer list of positive numbers.
    """
    for key, value in change_fps.items():
        if isinstance(value, str):
            if value != "set":
                raise ConfigError(f"change_fps entry '{key}' must be a [num, den] pair or \"set\"")
        elif isinstance(value, list):
            if len(value) != 2 or not all(isinstance(v, int) and v > 0 for v in value):
                raise ConfigError(f"change_fps entry '{key}' must contain two positive integers")
        else:
            raise ConfigError(f"change_fps entry '{key}' must be a list or \"set\"")


def _validate_color_overrides(overrides: Dict[str, Any]) -> None:
    """Validate per-clip colour override tables."""

    if not isinstance(overrides, dict):
        raise ConfigError("[color].color_overrides must be a table of clip overrides")

    valid_keys = {"matrix", "transfer", "primaries", "range"}
    for clip_name, table in overrides.items():
        if not isinstance(table, dict):
            raise ConfigError(
                f"[color].color_overrides.{clip_name} must be a table of colour properties"
            )
        for key, value in table.items():
            if key not in valid_keys:
                raise ConfigError(
                    "[color].color_overrides entries may only set matrix, transfer, primaries, or range"
                )
            if not isinstance(value, (str, int)):
                raise ConfigError(
                    f"[color].color_overrides.{clip_name}.{key} must be a string or integer"
                )


def load_config(path: str) -> AppConfig:
    """
    Load and validate an application configuration from a TOML file.

    Reads the file at `path`, parses it as UTF-8 TOML (BOM is accepted), coerces and validates all top-level sections, normalizes a few fields (for example pad/overlay/source/category preferences), and returns a fully populated AppConfig ready for use by the application.

    Returns:
        AppConfig: The validated and normalized application configuration.

    Raises:
        ConfigError: If the file is not UTF-8, TOML parsing fails, required values are missing, or any validation rule is violated.
    """

    with open(path, "rb") as handle:
        raw_bytes = handle.read()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
    try:
        raw = tomllib.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError("Configuration file must be UTF-8 encoded") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Failed to parse TOML: {exc}") from exc

    analysis_section = raw.get("analysis", {})
    if isinstance(analysis_section, dict):
        _migrate_analysis_section(analysis_section)
    app = AppConfig(
        analysis=_sanitize_section(analysis_section, "analysis", AnalysisConfig),
        screenshots=_sanitize_section(raw.get("screenshots", {}), "screenshots", ScreenshotConfig),
        cli=_sanitize_section(raw.get("cli", {}), "cli", CLIConfig),
        runner=_sanitize_section(raw.get("runner", {}), "runner", RunnerConfig),
        slowpics=_sanitize_section(raw.get("slowpics", {}), "slowpics", SlowpicsConfig),
        tmdb=_sanitize_section(raw.get("tmdb", {}), "tmdb", TMDBConfig),
        naming=_sanitize_section(raw.get("naming", {}), "naming", NamingConfig),
        paths=_sanitize_section(raw.get("paths", {}), "paths", PathsConfig),
        runtime=_sanitize_section(raw.get("runtime", {}), "runtime", RuntimeConfig),
        overrides=_sanitize_section(raw.get("overrides", {}), "overrides", OverridesConfig),
        color=_sanitize_section(raw.get("color", {}), "color", ColorConfig),
        source=_sanitize_section(raw.get("source", {}), "source", SourceConfig),
        audio_alignment=_sanitize_section(
            raw.get("audio_alignment", {}), "audio_alignment", AudioAlignmentConfig
        ),
        report=_sanitize_section(raw.get("report", {}), "report", ReportConfig),
        diagnostics=_sanitize_section(raw.get("diagnostics", {}), "diagnostics", DiagnosticsConfig),
    )

    normalized_style = str(app.cli.progress.style).strip().lower()
    if normalized_style not in {"fill", "dot"}:
        raise ConfigError("cli.progress.style must be 'fill' or 'dot'")
    app.cli.progress.style = normalized_style

    if app.analysis.step < 1:
        raise ConfigError("analysis.step must be >= 1")
    if app.analysis.downscale_height < 0:
        raise ConfigError("analysis.downscale_height must be >= 0")
    if 0 < app.analysis.downscale_height < 64:
        raise ConfigError("analysis.downscale_height must be 0 or >= 64")
    if app.analysis.random_seed < 0:
        raise ConfigError("analysis.random_seed must be >= 0")
    if not app.analysis.frame_data_filename:
        raise ConfigError("analysis.frame_data_filename must be set")
    if app.analysis.ignore_lead_seconds < 0:
        raise ConfigError("analysis.ignore_lead_seconds must be >= 0")
    if app.analysis.ignore_trail_seconds < 0:
        raise ConfigError("analysis.ignore_trail_seconds must be >= 0")
    if app.analysis.min_window_seconds < 0:
        raise ConfigError("analysis.min_window_seconds must be >= 0")
    _validate_thresholds(app.analysis.thresholds)

    _validate_color_overrides(app.color.color_overrides)

    if app.screenshots.compression_level not in (0, 1, 2):
        raise ConfigError("screenshots.compression_level must be 0, 1, or 2")
    if app.screenshots.mod_crop < 0:
        raise ConfigError("screenshots.mod_crop must be >= 0")
    if not isinstance(app.screenshots.letterbox_px_tolerance, int):
        raise ConfigError("screenshots.letterbox_px_tolerance must be an integer")
    if app.screenshots.letterbox_px_tolerance < 0:
        raise ConfigError("screenshots.letterbox_px_tolerance must be >= 0")
    try:
        timeout_value = float(app.screenshots.ffmpeg_timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ConfigError(_FFMPEG_TIMEOUT_NOT_NUMBER_MSG) from exc
    if not math.isfinite(timeout_value):
        raise ConfigError(_FFMPEG_TIMEOUT_NOT_FINITE_MSG)
    if timeout_value < 0:
        raise ConfigError(_FFMPEG_TIMEOUT_NEGATIVE_MSG)
    app.screenshots.ffmpeg_timeout_seconds = timeout_value
    pad_mode = str(app.screenshots.pad_to_canvas).strip().lower()
    if pad_mode not in {"off", "on", "auto"}:
        raise ConfigError("screenshots.pad_to_canvas must be 'off', 'on', or 'auto'")
    app.screenshots.pad_to_canvas = pad_mode
    app.screenshots.auto_letterbox_crop = _normalise_auto_letterbox_mode(
        getattr(app.screenshots, "auto_letterbox_crop", AutoLetterboxCropMode.OFF.value),
        "[screenshots].auto_letterbox_crop",
    )

    if hasattr(app.screenshots, "center_pad") and app.screenshots.center_pad is False:
        logging.warning(
            "Config: [screenshots].center_pad is deprecated and ignored; padding is always centered."
        )
        app.screenshots.center_pad = True

    progress_style = str(app.cli.progress.style).strip().lower()
    if progress_style not in {"fill", "dot"}:
        raise ConfigError("cli.progress.style must be 'fill' or 'dot'")
    app.cli.progress.style = progress_style

    if app.slowpics.remove_after_days < 0:
        raise ConfigError("slowpics.remove_after_days must be >= 0")
    if app.slowpics.image_upload_timeout_seconds <= 0:
        raise ConfigError("slowpics.image_upload_timeout_seconds must be > 0")

    if app.tmdb.year_tolerance < 0:
        raise ConfigError("tmdb.year_tolerance must be >= 0")
    if app.tmdb.cache_ttl_seconds < 0:
        raise ConfigError("tmdb.cache_ttl_seconds must be >= 0")
    if app.tmdb.cache_max_entries < 0:
        raise ConfigError("tmdb.cache_max_entries must be >= 0")
    if app.tmdb.category_preference is not None:
        preference = app.tmdb.category_preference.strip().upper()
        if preference not in {"", "MOVIE", "TV"}:
            raise ConfigError("tmdb.category_preference must be MOVIE, TV, or omitted")
        app.tmdb.category_preference = preference or None

    if app.runtime.ram_limit_mb <= 0:
        raise ConfigError("runtime.ram_limit_mb must be > 0")

    if app.color.target_nits <= 0:
        raise ConfigError("color.target_nits must be > 0")
    if app.color.dst_min_nits < 0:
        raise ConfigError("color.dst_min_nits must be >= 0")
    if app.color.knee_offset < 0 or app.color.knee_offset > 1:
        raise ConfigError("color.knee_offset must be between 0 and 1")
    dpd_preset_value = str(getattr(app.color, "dpd_preset", "") or "").strip().lower() or "off"
    valid_dpd_presets = {"off", "fast", "balanced", "high_quality"}
    if dpd_preset_value not in valid_dpd_presets:
        raise ConfigError("color.dpd_preset must be one of off, fast, balanced, or high_quality")
    if not app.color.dynamic_peak_detection:
        dpd_preset_value = "off"
        app.color.dpd_black_cutoff = 0.0
    else:
        try:
            cutoff_value = float(app.color.dpd_black_cutoff)
        except (TypeError, ValueError) as exc:
            raise ConfigError("color.dpd_black_cutoff must be a number") from exc
        if cutoff_value < 0 or cutoff_value > 0.05:
            raise ConfigError("color.dpd_black_cutoff must be between 0 and 0.05")
        app.color.dpd_black_cutoff = cutoff_value
    app.color.dpd_preset = dpd_preset_value
    try:
        post_gamma_value = float(app.color.post_gamma)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ConfigError("color.post_gamma must be a number") from exc
    if post_gamma_value < 0.9 or post_gamma_value > 1.1:
        raise ConfigError("color.post_gamma must be between 0.9 and 1.1")
    app.color.post_gamma = post_gamma_value
    if app.color.verify_luma_threshold < 0 or app.color.verify_luma_threshold > 1:
        raise ConfigError("color.verify_luma_threshold must be between 0 and 1")
    if app.color.verify_start_seconds < 0:
        raise ConfigError("color.verify_start_seconds must be >= 0")
    if app.color.verify_step_seconds <= 0:
        raise ConfigError("color.verify_step_seconds must be > 0")
    if app.color.verify_max_seconds < 0:
        raise ConfigError("color.verify_max_seconds must be >= 0")
    try:
        smoothing_period = float(app.color.smoothing_period)
    except (TypeError, ValueError) as exc:
        raise ConfigError("color.smoothing_period must be a number") from exc
    if smoothing_period < 0:
        raise ConfigError("color.smoothing_period must be >= 0")
    app.color.smoothing_period = smoothing_period
    try:
        scene_low = float(app.color.scene_threshold_low)
        scene_high = float(app.color.scene_threshold_high)
    except (TypeError, ValueError) as exc:
        raise ConfigError("color.scene_threshold_low/high must be numbers") from exc
    if scene_low < 0:
        raise ConfigError("color.scene_threshold_low must be >= 0")
    if scene_high < 0:
        raise ConfigError("color.scene_threshold_high must be >= 0")
    if scene_high < scene_low:
        raise ConfigError("color.scene_threshold_high must be >= color.scene_threshold_low")
    app.color.scene_threshold_low = scene_low
    app.color.scene_threshold_high = scene_high
    try:
        percentile_value = float(app.color.percentile)
    except (TypeError, ValueError) as exc:
        raise ConfigError("color.percentile must be a number") from exc
    if percentile_value < 0 or percentile_value > 100:
        raise ConfigError("color.percentile must be between 0 and 100")
    app.color.percentile = percentile_value
    try:
        contrast_recovery = float(app.color.contrast_recovery)
    except (TypeError, ValueError) as exc:
        raise ConfigError("color.contrast_recovery must be a number") from exc
    if contrast_recovery < 0:
        raise ConfigError("color.contrast_recovery must be >= 0")
    app.color.contrast_recovery = contrast_recovery
    metadata_value = getattr(app.color, "metadata", "auto")
    metadata_options = {"none", "hdr10", "hdr10+", "hdr10plus", "luminance"}
    if isinstance(metadata_value, str):
        lowered = metadata_value.strip().lower()
        if lowered in {"", "auto"}:
            app.color.metadata = "auto"
        elif lowered in metadata_options:
            app.color.metadata = lowered
        else:
            try:
                numeric_metadata = int(lowered)
            except ValueError as exc:
                raise ConfigError("color.metadata must be one of auto, none, hdr10, hdr10+, luminance, or an integer 0-4") from exc
            if numeric_metadata < 0 or numeric_metadata > 4:
                raise ConfigError("color.metadata integer must be between 0 and 4")
            app.color.metadata = numeric_metadata
    else:
        if type(metadata_value) is not int:
            raise ConfigError(
                "color.metadata must be one of auto, none, hdr10, hdr10+, luminance, or an integer 0-4"
            )
        numeric_metadata = metadata_value
        if numeric_metadata < 0 or numeric_metadata > 4:
            raise ConfigError("color.metadata integer must be between 0 and 4")
        app.color.metadata = numeric_metadata
    use_dovi_value = getattr(app.color, "use_dovi", None)
    if isinstance(use_dovi_value, str):
        lowered = use_dovi_value.strip().lower()
        if lowered in {"auto", ""}:
            app.color.use_dovi = None
        elif lowered in {"true", "1", "yes", "on"}:
            app.color.use_dovi = True
        elif lowered in {"false", "0", "no", "off"}:
            app.color.use_dovi = False
        else:
            raise ConfigError("color.use_dovi must be true, false, or auto")
    elif use_dovi_value is not None:
        if isinstance(use_dovi_value, bool):
            app.color.use_dovi = use_dovi_value
        elif isinstance(use_dovi_value, int) and use_dovi_value in (0, 1):
            app.color.use_dovi = bool(use_dovi_value)
        else:
            raise ConfigError("color.use_dovi must be true, false, or auto")
    overlay_mode = str(getattr(app.color, "overlay_mode", "minimal")).strip().lower()
    if overlay_mode not in {"minimal", "diagnostic"}:
        raise ConfigError("color.overlay_mode must be 'minimal' or 'diagnostic'")
    app.color.overlay_mode = overlay_mode

    preferred = app.source.preferred.strip().lower()
    if preferred not in {"lsmas", "ffms2"}:
        raise ConfigError("source.preferred must be either 'lsmas' or 'ffms2'")
    app.source.preferred = preferred

    audio_cfg = app.audio_alignment
    if audio_cfg.sample_rate <= 0:
        raise ConfigError("audio_alignment.sample_rate must be > 0")
    if audio_cfg.hop_length <= 0:
        raise ConfigError("audio_alignment.hop_length must be > 0")
    if audio_cfg.start_seconds is not None and audio_cfg.start_seconds < 0:
        raise ConfigError("audio_alignment.start_seconds must be >= 0")
    if audio_cfg.duration_seconds is not None and audio_cfg.duration_seconds <= 0:
        raise ConfigError("audio_alignment.duration_seconds must be > 0")
    if audio_cfg.correlation_threshold < 0 or audio_cfg.correlation_threshold > 1:
        raise ConfigError("audio_alignment.correlation_threshold must be between 0 and 1")
    if audio_cfg.max_offset_seconds <= 0:
        raise ConfigError("audio_alignment.max_offset_seconds must be > 0")
    if not audio_cfg.offsets_filename.strip():
        raise ConfigError("audio_alignment.offsets_filename must be set")
    if audio_cfg.random_seed < 0:
        raise ConfigError("audio_alignment.random_seed must be >= 0")

    report_cfg = app.report
    report_output_dir = report_cfg.output_dir.strip()
    if not report_output_dir:
        raise ConfigError("report.output_dir must be set")
    output_path = Path(report_output_dir)
    if output_path.is_absolute():
        raise ConfigError("report.output_dir must be relative to the workspace root")
    if ".." in output_path.parts:
        raise ConfigError("report.output_dir may not contain '..' segments")
    report_cfg.output_dir = report_output_dir
    include_mode = str(report_cfg.include_metadata).strip().lower()
    if include_mode not in {"minimal", "full"}:
        raise ConfigError("report.include_metadata must be 'minimal' or 'full'")
    report_cfg.include_metadata = include_mode
    default_mode = str(getattr(report_cfg, "default_mode", "slider")).strip().lower()
    if default_mode not in {"slider", "overlay"}:
        raise ConfigError("report.default_mode must be 'slider' or 'overlay'")
    report_cfg.default_mode = default_mode
    if isinstance(report_cfg.title, str):
        stripped_title = report_cfg.title.strip()
        report_cfg.title = stripped_title or None
    else:
        report_cfg.title = None
    if isinstance(report_cfg.default_left_label, str):
        stripped_left = report_cfg.default_left_label.strip()
        report_cfg.default_left_label = stripped_left or None
    if isinstance(report_cfg.default_right_label, str):
        stripped_right = report_cfg.default_right_label.strip()
        report_cfg.default_right_label = stripped_right or None
    if report_cfg.thumb_height < 0:
        raise ConfigError("report.thumb_height must be >= 0")

    _validate_trim(app.overrides.trim, "overrides.trim")
    _validate_trim(app.overrides.trim_end, "overrides.trim_end")
    _validate_change_fps(app.overrides.change_fps)

    return app
