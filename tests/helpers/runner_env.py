from __future__ import annotations

import os
import shutil
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Callable, Dict, Final, Optional, cast

import pytest
from rich.console import Console

import frame_compare
import src.audio_alignment as audio_alignment_module
import src.frame_compare.alignment as alignment_package
import src.frame_compare.alignment.core as alignment_runner_module
import src.frame_compare.alignment_preview as alignment_preview_module
import src.frame_compare.cache as cache_module
import src.frame_compare.config_helpers as config_helpers_module
import src.frame_compare.core as core_module
import src.frame_compare.media as media_module
import src.frame_compare.metadata as metadata_module
import src.frame_compare.planner as planner_module
import src.frame_compare.preflight as preflight_module
import src.frame_compare.tmdb_workflow as tmdb_workflow_module
import src.frame_compare.vspreview as vspreview_module
from src.datatypes import (
    AnalysisConfig,
    AppConfig,
    AudioAlignmentConfig,
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
from src.frame_compare import runner as runner_module
from src.frame_compare import vs as vs_core_module
from src.frame_compare.analysis import SelectionDetail
from src.frame_compare.cli_runtime import (
    AudioAlignmentJSON,
    CliOutputManager,
    JsonTail,
    _AudioAlignmentDisplayData,
)
from src.frame_compare.orchestration import coordinator as coordinator_module
from src.frame_compare.orchestration import reporting as reporting_module
from src.frame_compare.orchestration import setup as setup_module
from src.frame_compare.orchestration.state import RunEnvironment, RunRequest

__all__ = [
    "_CliRunnerEnv",
    "_CliRunnerEnvState",
    "_RecordingOutputManager",
    "_make_cli_preflight",
    "_patch_core_helper",
    "_patch_runner_module",
    "_patch_vs_core",
    "_patch_audio_alignment",
    "_make_json_tail_stub",
    "_make_display_stub",
    "_make_config",
    "_make_runner_preflight",
    "DummyProgress",
    "_expect_mapping",
    "_patch_load_config",
    "_selection_details_to_json",
    "install_vs_core_stub",
    "install_dummy_progress",
    "install_tty_stdin",
    "FakeTTY",
    "_format_vspreview_manual_command",
    "_VSPREVIEW_WINDOWS_INSTALL",
    "_VSPREVIEW_POSIX_INSTALL",
    "VSPreviewPatch",
    "install_vspreview_presence",
    "install_which_map",
    "MockSetupService",
]


class FakeTTY:
    """Stub for sys.stdin that reports as a TTY.

    Used by tests that verify interactive prompt behavior (e.g., audio alignment
    offset confirmation) without requiring an actual terminal.
    """

    def isatty(self) -> bool:
        return True


def install_tty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sys.stdin appear as a TTY for interactive prompt tests."""
    monkeypatch.setattr(sys, "stdin", FakeTTY())


class MockSetupService:
    """A setup service that returns a pre-configured environment."""

    def __init__(self, env: RunEnvironment) -> None:
        self.env = env

    def prepare_run_environment(self, request: RunRequest) -> RunEnvironment:
        return self.env


@dataclass
class _CliRunnerEnvState:
    """Tracked state for the CLI harness."""

    workspace_root: Path
    media_root: Path
    config_path: Path
    cfg: AppConfig


def _make_cli_preflight(
    base_dir: Path,
    cfg: AppConfig,
    *,
    workspace_name: str | None = None,
) -> core_module.PreflightResult:
    """Build a PreflightResult anchored in a temporary workspace."""

    workspace_root = base_dir / (workspace_name or "workspace")
    workspace_root.mkdir(parents=True, exist_ok=True)

    config_dir = workspace_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    if not config_path.exists():
        config_path.write_text("config = true\n", encoding="utf-8")

    input_dir = Path(cfg.paths.input_dir)
    media_root = input_dir if input_dir.is_absolute() else workspace_root / input_dir
    media_root.mkdir(parents=True, exist_ok=True)

    return core_module.PreflightResult(
        workspace_root=workspace_root,
        media_root=media_root,
        config_path=config_path,
        config=cfg,
        warnings=(),
        legacy_config=False,
    )


class _CliRunnerEnv:
    """Factory that installs deterministic CLI preflight results for tests."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, base_dir: Path) -> None:
        self._monkeypatch = monkeypatch
        self._base_dir = base_dir
        self.cfg = core_module._fresh_app_config()
        self.state: _CliRunnerEnvState | None = None
        self.reinstall()

    def reinstall(
        self,
        cfg: AppConfig | None = None,
        *,
        workspace_name: str | None = None,
    ) -> _CliRunnerEnvState:
        """Rebuild the harness with the provided config and workspace label."""

        if cfg is not None:
            self.cfg = cfg
        preflight = _make_cli_preflight(
            self._base_dir,
            self.cfg,
            workspace_name=workspace_name,
        )

        def _fake_preflight(**_kwargs: object) -> core_module.PreflightResult:
            return preflight

        module_targets = (
            core_module,
            frame_compare,
            preflight_module,
            getattr(runner_module, "preflight_utils", None),
            setup_module,
            coordinator_module,
        )
        for module in module_targets:
            if module is None:
                continue
            for attr_name in ("prepare_preflight", "_prepare_preflight"):
                self._monkeypatch.setattr(module, attr_name, _fake_preflight, raising=False)

        self.state = _CliRunnerEnvState(
            workspace_root=preflight.workspace_root,
            media_root=preflight.media_root,
            config_path=preflight.config_path,
            cfg=self.cfg,
        )
        return self.state

    @property
    def workspace_root(self) -> Path:
        assert self.state is not None
        return self.state.workspace_root

    @property
    def media_root(self) -> Path:
        assert self.state is not None
        return self.state.media_root

    @property
    def config_path(self) -> Path:
        assert self.state is not None
        return self.state.config_path


def _patch_core_helper(monkeypatch: pytest.MonkeyPatch, attr: str, value: object) -> None:
    """Patch both frame_compare.* and runner_module.core.* helpers simultaneously."""

    alias_map: dict[str, tuple[str, ...]] = {
        "prepare_preflight": ("_prepare_preflight",),
        "_prepare_preflight": ("prepare_preflight",),
        "collect_path_diagnostics": ("_collect_path_diagnostics",),
        "_collect_path_diagnostics": ("collect_path_diagnostics",),
        "_parse_metadata": ("parse_metadata",),
        "parse_metadata": ("_parse_metadata",),
        "_build_plans": ("build_plans",),
        "build_plans": ("_build_plans",),
        "apply_audio_alignment": ("_maybe_apply_audio_alignment",),
        "_maybe_apply_audio_alignment": ("apply_audio_alignment",),
        "_apply_vspreview_manual_offsets": ("apply_manual_offsets",),
        "apply_manual_offsets": ("_apply_vspreview_manual_offsets",),
        "write_script": ("_write_vspreview_script",),
        "_write_vspreview_script": ("write_script",),
        "_persist_script": ("persist_script",),
        "persist_script": ("_persist_script",),
        "_resolve_vspreview_command": ("resolve_command",),
        "resolve_command": ("_resolve_vspreview_command",),
        "_resolve_vspreview_subdir": ("resolve_subdir",),
        "resolve_subdir": ("_resolve_vspreview_subdir",),
        "_launch_vspreview": ("launch",),
        "launch": ("_launch_vspreview",),
        "_prompt_offsets": ("prompt_offsets",),
        "_prompt_vspreview_offsets": ("prompt_offsets",),
        "prompt_offsets": ("_prompt_offsets", "_prompt_vspreview_offsets"),
        "_confirm_alignment_with_screenshots": ("confirm_alignment_with_screenshots",),
        "confirm_alignment_with_screenshots": ("_confirm_alignment_with_screenshots",),
    }
    attrs_to_patch = (attr,) + alias_map.get(attr, tuple())

    targets = [
        frame_compare,
        core_module,
        alignment_runner_module,
        alignment_package,
        vspreview_module,
        getattr(runner_module, "vspreview", None),
        preflight_module,
        media_module,
        cache_module,
        alignment_preview_module,
        config_helpers_module,
        getattr(runner_module, "preflight_utils", None),
        getattr(runner_module, "media_utils", None),
        getattr(runner_module, "cache_utils", None),
        getattr(runner_module, "alignment_preview_utils", None),
        getattr(runner_module, "alignment_runner", None),
        getattr(runner_module, "config_helpers", None),
        metadata_module,
        getattr(runner_module, "metadata_utils", None),
        planner_module,
        getattr(runner_module, "planner_utils", None),
        tmdb_workflow_module,
        getattr(runner_module, "tmdb_workflow", None),
        coordinator_module,
        setup_module,
        reporting_module,
    ]
    for target in targets:
        if target is None:
            continue
        for attr_name in attrs_to_patch:
            if hasattr(target, attr_name):
                monkeypatch.setattr(target, attr_name, value, raising=False)


def _patch_vs_core(monkeypatch: pytest.MonkeyPatch, attr: str, value: object) -> None:
    """Patch VapourSynth helpers in both the shim module and the runner module."""

    monkeypatch.setattr(vs_core_module, attr, value, raising=False)


def _patch_runner_module(monkeypatch: pytest.MonkeyPatch, attr: str, value: object) -> None:
    """Patch shared runner dependencies exposed at the runner module level."""

    import src.frame_compare.cli_runtime as cli_runtime_module
    from src.frame_compare.orchestration.phases import (
        alignment as alignment_phase,
    )
    from src.frame_compare.orchestration.phases import (
        analysis as analysis_phase,
    )
    from src.frame_compare.orchestration.phases import (
        discovery as discovery_phase,
    )
    from src.frame_compare.orchestration.phases import (
        loader as loader_phase,
    )
    from src.frame_compare.orchestration.phases import (
        publish as publish_phase,
    )
    from src.frame_compare.orchestration.phases import (
        render as render_phase,
    )
    from src.frame_compare.orchestration.phases import (
        result as result_phase,
    )
    from src.frame_compare.orchestration.phases import (
        setup as setup_phase,
    )

    targets = [
        frame_compare,
        core_module,
        runner_module,
        getattr(runner_module, "preflight_utils", None),
        getattr(runner_module, "media_utils", None),
        getattr(runner_module, "cache_utils", None),
        getattr(runner_module, "alignment_preview_utils", None),
        getattr(runner_module, "config_helpers", None),
        alignment_preview_module,
        tmdb_workflow_module,
        getattr(runner_module, "tmdb_workflow", None),
        coordinator_module,
        setup_module,
        reporting_module,
        cli_runtime_module,
        setup_phase,
        discovery_phase,
        alignment_phase,
        loader_phase,
        analysis_phase,
        render_phase,
        publish_phase,
        result_phase,
    ]
    for target in targets:
        if target is None:
            continue
        if hasattr(target, attr):
            monkeypatch.setattr(target, attr, value, raising=False)


def _patch_audio_alignment(monkeypatch: pytest.MonkeyPatch, attr: str, value: object) -> None:
    """Patch audio alignment helpers in multiple namespaces simultaneously."""

    target = getattr(frame_compare, "audio_alignment", None)
    if target is not None:
        monkeypatch.setattr(target, attr, value, raising=False)
    monkeypatch.setattr(core_module.audio_alignment, attr, value, raising=False)
    monkeypatch.setattr(alignment_runner_module.audio_alignment, attr, value, raising=False)
    monkeypatch.setattr(vspreview_module.audio_alignment, attr, value, raising=False)
    runner_alignment = getattr(runner_module, "alignment_runner", None)
    if runner_alignment is not None:
        monkeypatch.setattr(runner_alignment.audio_alignment, attr, value, raising=False)
    monkeypatch.setattr(audio_alignment_module, attr, value, raising=False)


def _make_json_tail_stub() -> JsonTail:
    """Replicate the full JsonTail payload used by CLI telemetry assertions."""

    audio_block: AudioAlignmentJSON = {
        "enabled": False,
        "offsets_filename": "offsets.toml",
        "offsets_sec": {},
        "offsets_frames": {},
        "suggestion_mode": True,
        "vspreview_mode": "baseline",
        "vspreview_script": None,
        "vspreview_invoked": False,
        "vspreview_exit_code": None,
        "manual_trim_starts": {},
        "vspreview_manual_offsets": {},
        "vspreview_manual_deltas": {},
        "vspreview_reference_trim": 0,
        "preview_paths": [],
        "confirmed": False,
        "suggested_frames": {},
        "reference_stream": None,
        "target_stream": {},
        "stream_lines": [],
        "stream_lines_text": "",
        "offset_lines": [],
        "offset_lines_text": "",
        "measurements": {},
        "manual_trim_summary": [],
    }
    tail: JsonTail = {
        "clips": [],
        "trims": {"per_clip": {}},
        "window": {},
        "alignment": {"manual_start_s": 0.0, "manual_end_s": "unchanged"},
        "audio_alignment": audio_block,
        "analysis": {},
        "render": {},
        "tonemap": {},
        "overlay": {
            "enabled": True,
            "template": "",
            "mode": "minimal",
            "diagnostics": {
                "dv": {"enabled": None, "label": "auto"},
                "frame_metrics": {
                    "enabled": False,
                    "per_frame": {},
                    "gating": {
                        "config": False,
                        "cli_override": None,
                        "overlay_mode": "minimal",
                    },
                },
            },
        },
        "verify": {
            "count": 0,
            "threshold": 0.0,
            "delta": {
                "max": None,
                "average": None,
                "frame": None,
                "file": None,
                "auto_selected": None,
            },
            "entries": [],
        },
        "cache": {},
        "slowpics": {
            "enabled": False,
            "title": {
                "inputs": {
                    "resolved_base": None,
                    "collection_name": None,
                    "collection_suffix": "",
                },
                "final": None,
            },
            "url": None,
            "shortcut_path": None,
            "shortcut_written": False,
            "shortcut_error": None,
            "deleted_screens_dir": False,
            "is_public": False,
            "is_hentai": False,
            "remove_after_days": 0,
        },
        "warnings": [],
        "workspace": {
            "root": "",
            "media_root": "",
            "config_path": "",
            "legacy_config": False,
        },
        "report": {
            "enabled": False,
            "path": None,
            "output_dir": "report",
            "open_after_generate": True,
            "opened": False,
            "mode": "slider",
        },
        "viewer": {
            "mode": "none",
            "mode_display": "None",
            "destination": None,
            "destination_label": "",
        },
        "vspreview_mode": None,
        "suggested_frames": None,
        "suggested_seconds": 0.0,
        "vspreview_offer": None,
    }
    return tail


def _make_display_stub() -> _AudioAlignmentDisplayData:
    """Return a reusable AudioAlignmentDisplayData stub."""

    return _AudioAlignmentDisplayData(
        stream_lines=[],
        estimation_line=None,
        offset_lines=[],
        offsets_file_line="Offsets file: offsets.toml",
        json_reference_stream=None,
        json_target_streams={},
        json_offsets_sec={},
        json_offsets_frames={},
        warnings=[],
        manual_trim_lines=[],
    )


def _make_config(input_dir: Path) -> AppConfig:
    """Create an AppConfig with deterministic defaults rooted at input_dir."""

    return AppConfig(
        analysis=AnalysisConfig(
            frame_count_dark=1,
            frame_count_bright=1,
            frame_count_motion=1,
            random_frames=0,
            user_frames=[],
        ),
        screenshots=ScreenshotConfig(directory_name="screens", add_frame_info=False),
        cli=CLIConfig(),
        color=ColorConfig(),
        slowpics=SlowpicsConfig(auto_upload=False),
        tmdb=TMDBConfig(),
        naming=NamingConfig(always_full_filename=False, prefer_guessit=False),
        paths=PathsConfig(input_dir=str(input_dir)),
        runtime=RuntimeConfig(ram_limit_mb=4096),
        overrides=OverridesConfig(
            trim={"0": 5},
            trim_end={"BBB - 01.mkv": -12},
            change_fps={"BBB - 01.mkv": "set"},
        ),
        source=SourceConfig(preferred="lsmas"),
        audio_alignment=AudioAlignmentConfig(enable=False),
        report=ReportConfig(enable=False),
        runner=RunnerConfig(),
        diagnostics=DiagnosticsConfig(),
    )


def _make_runner_preflight(
    workspace_root: Path,
    media_root: Path,
    cfg: AppConfig,
) -> core_module.PreflightResult:
    """Build a PreflightResult pointing at prepared workspace/media roots for runner tests."""

    config_dir = workspace_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    config_path.write_text("config", encoding="utf-8")
    cfg.paths.input_dir = str(media_root)
    return core_module.PreflightResult(
        workspace_root=workspace_root,
        media_root=media_root,
        config_path=config_path,
        config=cfg,
        warnings=(),
        legacy_config=False,
    )


class _RecordingOutputManager(CliOutputManager):
    """CliOutputManager test double that records lines emitted during confirmation flows."""

    def __init__(self) -> None:
        layout_path = Path(frame_compare.__file__).with_name("cli_layout.v1.json")
        super().__init__(
            quiet=False,
            verbose=False,
            no_color=True,
            layout_path=layout_path,
            console=Console(record=True, force_terminal=False),
        )
        self.lines: list[str] = []

    def line(self, text: str = "", *, style: Optional[str] = None) -> None:
        """Record the rendered line while still delegating to the base implementation."""

        self.lines.append(text)
        super().line(text)

    def error(self, text: str) -> None:
        """Record error output."""
        self.lines.append(f"ERROR: {text}")
        super().error(text)


class DummyProgress:
    """No-op progress helper used to stub Rich progress bars in CLI tests."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def __enter__(self) -> DummyProgress:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def add_task(self, *_: object, **__: object) -> int:
        return 1

    def update(self, *_: object, **__: object) -> None:
        return None


def install_dummy_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install DummyProgress as the Progress implementation across runner entry points."""

    _patch_runner_module(monkeypatch, "Progress", DummyProgress)
    _patch_core_helper(monkeypatch, "Progress", DummyProgress)


JsonMapping = Mapping[str, Any]


def _expect_mapping(value: object) -> JsonMapping:
    """Assert the provided value is a mapping and return it with a narrowed type."""

    assert isinstance(value, Mapping)
    return cast(JsonMapping, value)


def _patch_load_config(monkeypatch: pytest.MonkeyPatch, cfg: AppConfig) -> None:
    """Force both the shim and runner modules to load the provided config instance."""

    monkeypatch.setattr(core_module, "load_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(frame_compare, "load_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(preflight_module, "load_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(setup_module, "load_config", lambda *_args, **_kwargs: cfg, raising=False)


def _selection_details_to_json(
    details: Mapping[int, SelectionDetail]
) -> Dict[int, Dict[str, str]]:
    """Convert SelectionDetail mappings into serializable JSON structures."""

    return {frame: {"label": detail.label} for frame, detail in details.items()}


def install_vs_core_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a default VapourSynth stub so CLI tests never import the real module."""

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    def _default_clip(*_args: object, **_kwargs: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            width=1280,
            height=720,
            fps_num=24000,
            fps_den=1001,
            num_frames=120,
        )

    _patch_vs_core(monkeypatch, "configure", _noop)
    _patch_vs_core(monkeypatch, "set_ram_limit", _noop)
    _patch_vs_core(monkeypatch, "init_clip", _default_clip)


def _require_vspreview_constant(attr_name: str) -> str:
    """Fetch a VSPreview constant from the module and raise if it is missing."""

    value = getattr(vspreview_module, attr_name, None)
    if value is None:
        raise RuntimeError(f"{attr_name} is not available on src.frame_compare.vspreview")
    return cast(str, value)


_VSPREVIEW_WINDOWS_INSTALL: Final[str] = _require_vspreview_constant("_VSPREVIEW_WINDOWS_INSTALL")
_VSPREVIEW_POSIX_INSTALL: Final[str] = _require_vspreview_constant("_VSPREVIEW_POSIX_INSTALL")


def _format_vspreview_manual_command(script_path: Path) -> str:
    """Typed shim that returns the VSPreview manual command used by CLI tests."""

    formatter = getattr(vspreview_module, "_format_vspreview_manual_command", None)
    if formatter is None:
        raise RuntimeError("_format_vspreview_manual_command is not available on src.frame_compare.vspreview")
    return formatter(script_path)


_VSPREVIEW_IMPORT_NAMES: Final[frozenset[str]] = frozenset({"vapoursynth", "vspreview", "PySide6"})


class VSPreviewPatch:
    """Context manager for deterministic VSPreview availability (sunsets in Phase 10)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
        self._monkeypatch = monkeypatch
        self._present = present

    def __enter__(self) -> VSPreviewPatch:
        install_vspreview_presence(self._monkeypatch, present=self._present)
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        return False


def install_vspreview_presence(monkeypatch: pytest.MonkeyPatch, *, present: bool) -> None:
    """Toggle VSPreview module/CLI availability (fixtures replace this helper in Phase 10)."""

    original_find_spec = importlib_util.find_spec

    def _patched_find_spec(name: str, package: str | None = None):
        if name in _VSPREVIEW_IMPORT_NAMES and not present:
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib_util, "find_spec", _patched_find_spec, raising=False)

    original_which = shutil.which

    def _patched_which(cmd: str, mode: int = os.F_OK, path: str | None = None) -> str | None:
        if cmd == "vspreview":
            if present:
                return f"/usr/bin/{cmd}"
            return None
        return original_which(cmd, mode=mode, path=path)

    _patch_shutil_which(monkeypatch, _patched_which)


def install_which_map(monkeypatch: pytest.MonkeyPatch, missing: set[str] | None = None) -> None:
    """Map command availability to `/usr/bin/<cmd>` except for requested missing tools (Phase 10 sunset)."""

    missing_set = set(missing or set())

    def _patched_which(cmd: str, mode: int = os.F_OK, path: str | None = None) -> str | None:
        if cmd in missing_set:
            return None
        return f"/usr/bin/{cmd}"

    _patch_shutil_which(monkeypatch, _patched_which)


def _patch_shutil_which(
    monkeypatch: pytest.MonkeyPatch,
    stub: Callable[[str, int, str | None], str | None],
) -> None:
    """Apply the provided which stub across shim/core modules."""

    targets = [shutil, getattr(frame_compare, "shutil", None), getattr(core_module, "shutil", None)]
    for target in targets:
        if target is None:
            continue
        monkeypatch.setattr(target, "which", stub, raising=False)
