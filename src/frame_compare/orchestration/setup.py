from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, cast

from src.datatypes import AppConfig
from src.frame_compare import core
from src.frame_compare import preflight as preflight_utils
from src.frame_compare.cli_runtime import CLIAppError
from src.frame_compare.env_flags import env_flag_enabled
from src.frame_compare.layout_utils import normalise_vspreview_mode as _normalise_vspreview_mode
from src.frame_compare.orchestration import reporting
from src.frame_compare.orchestration.state import RunEnvironment, RunRequest
from src.frame_compare.result_snapshot import snapshot_path

logger = logging.getLogger('frame_compare')
DOVI_DEBUG_ENV_FLAG = "FRAME_COMPARE_DOVI_DEBUG"


def _safe_debug_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def emit_dovi_debug(payload: Mapping[str, Any]) -> None:
    if not env_flag_enabled(os.environ.get(DOVI_DEBUG_ENV_FLAG)):
        return
    try:
        message = json.dumps(dict(payload), default=_safe_debug_default)
    except (TypeError, ValueError):
        logging.getLogger(__name__).debug("Unable to serialize DOVI debug payload", exc_info=True)
        return
    print("[DOVI_DEBUG]", message, file=sys.stderr)


def _apply_cli_tonemap_overrides(color_cfg: Any, overrides: Mapping[str, Any]) -> None:
    if color_cfg is None or not overrides:
        return
    provided_raw = getattr(color_cfg, "_provided_keys", None)
    provided: set[str]
    if isinstance(provided_raw, set):
        provided = set(cast(set[Any], provided_raw))
    else:
        provided = set()
    updated: set[str] = set()

    def _assign(field: str, converter: Callable[[Any], Any]) -> None:
        if field in overrides:
            setattr(color_cfg, field, converter(overrides[field]))
            updated.add(field)

    _assign("preset", lambda value: str(value))
    _assign("tone_curve", lambda value: str(value))
    _assign("target_nits", lambda value: float(value))
    _assign("dst_min_nits", lambda value: float(value))
    _assign("knee_offset", lambda value: float(value))
    _assign("dpd_preset", lambda value: str(value))
    _assign("dpd_black_cutoff", lambda value: float(value))
    _assign("post_gamma", lambda value: float(value))
    _assign("post_gamma_enable", lambda value: bool(value))
    _assign("smoothing_period", lambda value: float(value))
    _assign("scene_threshold_low", lambda value: float(value))
    _assign("scene_threshold_high", lambda value: float(value))
    _assign("percentile", lambda value: float(value))
    _assign("contrast_recovery", lambda value: float(value))
    if "metadata" in overrides:
        color_cfg.metadata = overrides["metadata"]
        updated.add("metadata")
    if "use_dovi" in overrides:
        color_cfg.use_dovi = overrides["use_dovi"]
        updated.add("use_dovi")
    _assign("visualize_lut", lambda value: bool(value))
    _assign("show_clipping", lambda value: bool(value))

    if updated:
        provided.update(updated)
        try:
            color_cfg._provided_keys = provided
        except (AttributeError, TypeError, ValueError):
            pass


def prepare_run_environment(request: RunRequest) -> RunEnvironment:
    """
    Perform preflight checks, configure environment, and prepare dependencies.
    """
    config_path = request.config_path
    input_dir = request.input_dir
    root_override = request.root_override
    skip_wizard = request.skip_wizard
    debug_color = request.debug_color
    tonemap_overrides = request.tonemap_overrides
    impl = request.impl_module or importlib.import_module("frame_compare")
    module_file = Path(getattr(impl, '__file__', Path(__file__)))
    entrypoint_name = (
        getattr(request.impl_module, "__name__", "runner") if request.impl_module else "runner"
    )

    preflight = preflight_utils.prepare_preflight(
        cli_root=root_override,
        config_override=config_path,
        input_override=input_dir,
        ensure_config=True,
        create_dirs=True,
        create_media_dir=input_dir is None,
        allow_auto_wizard=True,
        skip_auto_wizard=skip_wizard,
    )
    cfg: AppConfig = preflight.config
    color_cfg = getattr(cfg, "color", None)
    cfg_use_dovi_before = getattr(color_cfg, "use_dovi", None)

    if tonemap_overrides:
        core.validate_tonemap_overrides(tonemap_overrides)
        _apply_cli_tonemap_overrides(color_cfg, tonemap_overrides)
    if debug_color:
        try:
            cfg.color.debug_color = True
        except AttributeError:
            pass

    report_enable_override = request.report_enable_override
    report_enabled = (
        bool(report_enable_override)
        if report_enable_override is not None
        else bool(getattr(cfg.report, "enable", False))
    )
    runner_cfg = getattr(cfg, "runner", None)
    legacy_requested = (
        request.service_mode_override is False
        or (runner_cfg is not None and bool(getattr(runner_cfg, "enable_service_mode", True)) is False)
    )
    service_mode_enabled = True
    workspace_root = preflight.workspace_root
    root = preflight.media_root
    config_location = preflight.config_path
    cfg_use_dovi_after = getattr(color_cfg, "use_dovi", None)
    tonemap_override_keys = sorted(tonemap_overrides.keys()) if tonemap_overrides else []
    emit_dovi_debug(
        {
            "phase": "pre_vs_effective_tonemap",
            "entrypoint": entrypoint_name,
            "config_path": config_location,
            "workspace_root": workspace_root,
            "media_root": root,
            "root_override": root_override,
            "cfg_use_dovi_before": cfg_use_dovi_before,
            "cfg_use_dovi_after": cfg_use_dovi_after,
            "tonemap_override_keys": tonemap_override_keys,
            "tonemap_override_has_use_dovi": bool(
                tonemap_overrides and "use_dovi" in tonemap_overrides
            ),
            "tonemap_overrides": tonemap_overrides or {},
        }
    )

    if not root.exists():
        raise CLIAppError(
            f"Input directory not found: {root}",
            rich_message=f"[red]Input directory not found:[/red] {root}",
        )

    out_dir = preflight_utils.resolve_subdir(
        root,
        cfg.screenshots.directory_name,
        purpose="screenshots.directory_name",
    )
    out_dir_preexisting = out_dir.exists()
    created_out_dir = False
    created_out_dir_path: Optional[Path] = None
    if not out_dir_preexisting:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CLIAppError(
                f"Unable to create screenshots directory '{out_dir}': {exc}",
                rich_message=(
                    "[red]Unable to create screenshots directory.[/red] "
                    f"Adjust [screenshots].directory_name or choose a writable --root. ({exc})"
                ),
            ) from exc
        created_out_dir = True
        try:
            created_out_dir_path = out_dir.resolve()
        except OSError:
            created_out_dir_path = out_dir
    result_snapshot_path = snapshot_path(out_dir)
    analysis_cache_path = preflight_utils.resolve_subdir(
        root,
        cfg.analysis.frame_data_filename,
        purpose="analysis.frame_data_filename",
    )
    offsets_path = preflight_utils.resolve_subdir(
        root,
        cfg.audio_alignment.offsets_filename,
        purpose="audio_alignment.offsets_filename",
    )
    preflight_utils.abort_if_site_packages(
        {
            "config": config_location,
            "workspace_root": workspace_root,
            "root": root,
            "screenshots": out_dir,
            "analysis_cache": analysis_cache_path,
            "audio_offsets": offsets_path,
        }
    )

    vspreview_mode_value = _normalise_vspreview_mode(
        getattr(cfg.audio_alignment, "vspreview_mode", "baseline")
    )

    layout_path = module_file.with_name("cli_layout.v1.json")

    reporter = reporting.create_reporter(request, layout_path)

    collected_warnings: list[str] = []
    if legacy_requested:
        legacy_warning = (
            "Legacy runner path has been retired; using the service-mode pipeline even when legacy was requested."
        )
        collected_warnings.append(legacy_warning)
        reporter.warn(legacy_warning)
        logger.warning(legacy_warning)

    if bool(getattr(cfg.slowpics, "auto_upload", False)):
        auto_upload_warning = (
            "slow.pics auto-upload is enabled; confirm you trust the destination or disable "
            "[slowpics].auto_upload to keep screenshots local."
        )
        reporter.warn(auto_upload_warning)
        logger.warning(auto_upload_warning)
        collected_warnings.append(auto_upload_warning)
    for note in preflight.warnings:
        reporter.warn(note)
        collected_warnings.append(note)

    return RunEnvironment(
        preflight=preflight,
        cfg=cfg,
        root=root,
        out_dir=out_dir,
        out_dir_created=created_out_dir,
        out_dir_created_path=created_out_dir_path,
        result_snapshot_path=result_snapshot_path,
        analysis_cache_path=analysis_cache_path,
        offsets_path=offsets_path,
        vspreview_mode_value=vspreview_mode_value,
        layout_path=layout_path,
        reporter=reporter,
        service_mode_enabled=service_mode_enabled,
        legacy_requested=legacy_requested,
        collected_warnings=collected_warnings,
        report_enabled=report_enabled,
    )

