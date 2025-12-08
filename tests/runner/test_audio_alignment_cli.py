"""Audio-alignment CLI regression tests covering VSPreview, prompts, and confirmation flows."""

from __future__ import annotations

import datetime as dt
import importlib
import json
import shutil
import subprocess
import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner, Result

import frame_compare
import src.frame_compare.alignment as alignment_package
import src.frame_compare.alignment.core as alignment_core_module
import src.frame_compare.core as core_module
import src.frame_compare.vs as vs_core_module
from src.audio_alignment import AlignmentMeasurement, AudioStreamInfo, FpsHintMap
from src.datatypes import (
    AnalysisConfig,
    AppConfig,
    ColorConfig,
    ScreenshotConfig,
)
from src.frame_compare import runner as runner_module
from src.frame_compare.analysis import CacheLoadResult, FrameMetricsCacheInfo, SelectionDetail
from src.frame_compare.cli_runtime import (
    NullCliOutputManager,
    _AudioAlignmentDisplayData,
    _AudioAlignmentSummary,
    _ClipPlan,
)
from src.frame_compare.orchestration.state import RunEnvironment
from src.frame_compare.preflight import PreflightResult
from src.frame_compare.runner import RunDependencies
from src.frame_compare.services.alignment import AlignmentRequest, AlignmentResult
from src.frame_compare.services.metadata import MetadataResolveResult
from tests.helpers.runner_env import (
    _VSPREVIEW_WINDOWS_INSTALL,
    DummyProgress,
    MockSetupService,
    _CliRunnerEnv,
    _expect_mapping,
    _format_vspreview_manual_command,
    _make_config,
    _make_display_stub,
    _make_json_tail_stub,
    _make_runner_preflight,
    _patch_audio_alignment,
    _patch_core_helper,
    _patch_runner_module,
    _patch_vs_core,
    _RecordingOutputManager,
    _selection_details_to_json,
)

pytestmark = pytest.mark.usefixtures("runner_vs_core_stub", "dummy_progress")  # type: ignore[attr-defined]


def _setup_cli_analysis_environment(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: _CliRunnerEnv,
) -> tuple[Path, Path, Path]:
    """Common runner patches for cache-observability tests."""

    reference_path = cli_runner_env.media_root / "Ref.mkv"
    target_path = cli_runner_env.media_root / "Target.mkv"
    for file_path in (reference_path, target_path):
        file_path.write_bytes(b"data")

    cfg = cli_runner_env.cfg
    cfg.analysis.frame_data_filename = "generated.compframes"
    cfg.analysis.save_frames_data = True
    cfg.audio_alignment.enable = False

    files = [reference_path, target_path]
    metadata = [
        {"label": "Reference", "file_name": reference_path.name},
        {"label": "Target", "file_name": target_path.name},
    ]

    _patch_core_helper(monkeypatch, "_discover_media", lambda _root: list(files))
    _patch_core_helper(monkeypatch, "parse_metadata", lambda *_args: list(metadata))
    _patch_core_helper(
        monkeypatch,
        "_pick_analyze_file",
        lambda _files, _metadata, _target, **_kwargs: reference_path,
    )

    import src.frame_compare.orchestration.coordinator as coordinator_module
    monkeypatch.setattr(coordinator_module, "write_selection_cache_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator_module, "export_selection_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator_module, "generate_screenshots", lambda *args, **kwargs: [])

    cache_file = cli_runner_env.media_root / cfg.analysis.frame_data_filename
    return reference_path, target_path, cache_file


def test_audio_alignment_vspreview_constants_raise_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VSPreview helpers should fail fast when the CLI shim stops exporting constants."""

    import tests.helpers.runner_env as runner_env_module

    monkeypatch.delattr(runner_env_module.vspreview_module, "_VSPREVIEW_WINDOWS_INSTALL", raising=False)
    monkeypatch.delattr(runner_env_module.vspreview_module, "_VSPREVIEW_POSIX_INSTALL", raising=False)

    with pytest.raises(RuntimeError, match="_VSPREVIEW_WINDOWS_INSTALL"):
        importlib.reload(runner_env_module)

    # Undo the temporary removal so other tests reload the shim with real exports.
    monkeypatch.undo()
    importlib.reload(runner_env_module)


def test_audio_alignment_manual_vspreview_handles_existing_trim(
    tmp_path: Path,
) -> None:
    """Manual VSPreview flow reports trims without crashing when alignment is off."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = False
    cfg.audio_alignment.use_vspreview = True

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )
    target_plan.trim_start = 42
    target_plan.has_trim_start_override = True

    summary, display = core_module._maybe_apply_audio_alignment(
        [reference_plan, target_plan],
        cfg,
        reference_path,
        tmp_path,
        {},
        reporter=None,
    )

    assert summary is not None
    assert display is not None
    assert summary.suggestion_mode is True
    assert summary.manual_trim_starts[target_path.name] == 42
    assert any("Existing manual trim" in line for line in display.offset_lines)
    assert any("manual alignment enabled" in warning for warning in display.warnings)

def test_audio_alignment_string_false_vspreview_triggers_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """String config values like "off" should disable VSPreview reuse logic."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True
    cfg.audio_alignment.use_vspreview = "off"  # type: ignore[assignment]

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )

    manual_entry = {
        "status": "manual",
        "note": "VSPreview delta",
        "frames": 7,
    }

    monkeypatch.setattr(
        core_module.audio_alignment,
        "load_offsets",
        lambda _path: (reference_path.name, {target_path.name: manual_entry}),
    )

    class _SentinelError(Exception):
        pass

    calls: list[str] = []

    def boom(*_args: object, **_kwargs: object) -> list[object]:
        calls.append("called")
        raise _SentinelError

    _patch_audio_alignment(monkeypatch, "measure_offsets", boom)
    assert alignment_core_module.audio_alignment.measure_offsets is boom
    monkeypatch.setattr(
        core_module.audio_alignment,
        "update_offsets_file",
        lambda *args, **kwargs: ({}, {}),
    )

    with pytest.raises(_SentinelError):
        core_module._maybe_apply_audio_alignment(
            [reference_plan, target_plan],
            cfg,
            reference_path,
            tmp_path,
            {},
            reporter=None,
        )
    assert calls

def test_audio_alignment_prompt_reuse_decline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When prompted and declined, cached offsets are reused without recomputation."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True
    cfg.audio_alignment.prompt_reuse_offsets = True

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )

    cached_entry = {
        "frames": 6,
        "seconds": 0.25,
        "correlation": 0.95,
        "target_fps": 24.0,
        "status": "auto",
    }

    monkeypatch.setattr(
        core_module.audio_alignment,
        "load_offsets",
        lambda _path: (reference_path.name, {target_path.name: dict(cached_entry)}),
    )
    def _fail_measure(*_args: object, **_kwargs: object) -> list[AlignmentMeasurement]:
        raise AssertionError("measure_offsets should not run")

    def _fail_update(*_args: object, **_kwargs: object) -> tuple[dict[str, int], dict[str, str]]:
        raise AssertionError("update_offsets_file should not run")

    _patch_audio_alignment(monkeypatch, "measure_offsets", _fail_measure)
    _patch_audio_alignment(monkeypatch, "update_offsets_file", _fail_update)

    class _TTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", _TTY())

    confirm_calls: dict[str, int] = {"count": 0}

    def _fake_confirm(*_args: object, **_kwargs: object) -> bool:
        confirm_calls["count"] += 1
        return False

    monkeypatch.setattr(click, "confirm", _fake_confirm)

    summary, display = core_module._maybe_apply_audio_alignment(
        [reference_plan, target_plan],
        cfg,
        reference_path,
        tmp_path,
        {},
        reporter=None,
    )

    assert confirm_calls["count"] == 1
    assert summary is not None
    assert display is not None
    assert summary.suggestion_mode is False
    assert summary.applied_frames[target_path.name] == 6
    assert target_plan.trim_start == 6
    assert summary.final_adjustments[target_path.name] == 6
    assert display.estimation_line and "reused" in display.estimation_line.lower()
    assert any("Audio offsets" in line for line in display.offset_lines)
    first_line = display.offset_lines[0]
    assert "[unknown/und/?]" in first_line
    assert "corr=" in first_line
    assert "status=auto/applied" in first_line


def test_audio_alignment_prompt_reuse_hydrates_suggestions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached runs should hydrate prior suggested offsets into summaries and CLI telemetry."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True
    cfg.audio_alignment.prompt_reuse_offsets = True

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )

    cached_entry = {
        "frames": 0,
        "seconds": 0.0,
        "correlation": 0.97,
        "target_fps": 24.0,
        "status": "auto",
        "suggested_frames": 7,
        "suggested_seconds": 0.291,
    }

    monkeypatch.setattr(
        core_module.audio_alignment,
        "load_offsets",
        lambda _path: (reference_path.name, {target_path.name: dict(cached_entry)}),
    )

    def _fail_measure(*_args: object, **_kwargs: object) -> list[AlignmentMeasurement]:
        raise AssertionError("measure_offsets should not run")

    def _fail_update(*_args: object, **_kwargs: object) -> tuple[dict[str, int], dict[str, str]]:
        raise AssertionError("update_offsets_file should not run")

    _patch_audio_alignment(monkeypatch, "measure_offsets", _fail_measure)
    _patch_audio_alignment(monkeypatch, "update_offsets_file", _fail_update)

    class _TTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", _TTY())

    confirm_calls: dict[str, int] = {"count": 0}

    def _fake_confirm(*_args: object, **_kwargs: object) -> bool:
        confirm_calls["count"] += 1
        return False

    monkeypatch.setattr(click, "confirm", _fake_confirm)

    summary, display = core_module._maybe_apply_audio_alignment(
        [reference_plan, target_plan],
        cfg,
        reference_path,
        tmp_path,
        {},
        reporter=None,
    )

    assert confirm_calls["count"] == 1
    assert summary is not None
    assert display is not None
    assert summary.suggested_frames[target_path.name] == 7
    detail = summary.measured_offsets[target_path.name]
    assert detail.frames == 7
    assert detail.offset_seconds == pytest.approx(0.291, rel=1e-6)

    json_tail = _make_json_tail_stub()
    reporter = NullCliOutputManager(quiet=True, verbose=False, no_color=True)
    alignment_package.format_alignment_output(
        [reference_plan, target_plan],
        summary,
        display,
        cfg=cfg,
        root=tmp_path,
        reporter=reporter,
        json_tail=json_tail,
        vspreview_mode="baseline",
        collected_warnings=[],
    )
    assert json_tail["suggested_frames"] == 7
    assert json_tail["suggested_seconds"] == pytest.approx(0.291, rel=1e-6)

def test_audio_alignment_prompt_reuse_affirm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Affirming the prompt (or skipping it) triggers fresh alignment."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True
    cfg.audio_alignment.prompt_reuse_offsets = True

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )

    monkeypatch.setattr(
        core_module.audio_alignment,
        "load_offsets",
        lambda _path: (reference_path.name, {target_path.name: {"frames": 4, "seconds": 0.2}}),
    )

    measure_calls: dict[str, int] = {"count": 0}

    def _fake_measure(
        _ref: Path,
        targets: Sequence[Path],
        *,
        progress_callback,
        **_kwargs: object,
    ) -> list[AlignmentMeasurement]:
        measure_calls["count"] += 1
        progress_callback(len(targets))
        return [
            AlignmentMeasurement(
                file=targets[0],
                offset_seconds=0.3,
                frames=7,
                correlation=0.9,
                reference_fps=24.0,
                target_fps=24.0,
            )
        ]

    _patch_audio_alignment(monkeypatch, "measure_offsets", _fake_measure)
    _patch_audio_alignment(monkeypatch, "probe_audio_streams", lambda _path: [])

    update_calls: dict[str, int] = {"count": 0}

    def _fake_update(
        _path: Path,
        _reference_name: str,
        measurements: Sequence[AlignmentMeasurement],
        _existing: Mapping[str, Mapping[str, object]],
        _notes: Mapping[str, str],
    ) -> tuple[dict[str, int], dict[str, str]]:
        update_calls["count"] += 1
        applied = {m.file.name: int(m.frames or 0) for m in measurements}
        return applied, {name: "auto" for name in applied}

    _patch_audio_alignment(monkeypatch, "update_offsets_file", _fake_update)

    class _TTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", _TTY())

    def _confirm_true(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(click, "confirm", _confirm_true)

    summary, display = core_module._maybe_apply_audio_alignment(
        [reference_plan, target_plan],
        cfg,
        reference_path,
        tmp_path,
        {},
        reporter=None,
    )

    assert measure_calls["count"] == 1
    assert update_calls["count"] == 1
    assert summary is not None
    assert display is not None
    assert summary.applied_frames[target_path.name] == 7
    assert target_plan.trim_start == 7
    assert summary.suggestion_mode is False

def test_run_cli_reuses_vspreview_manual_offsets_when_alignment_disabled(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: _CliRunnerEnv,
) -> None:
    """Manual VSPreview offsets should be reused during CLI runs when auto alignment is off."""

    reference_path = cli_runner_env.media_root / "Ref.mkv"
    target_path = cli_runner_env.media_root / "Target.mkv"
    for file_path in (reference_path, target_path):
        file_path.write_bytes(b"data")

    cfg = cli_runner_env.cfg
    cfg.analysis.frame_data_filename = "generated.compframes"
    cfg.audio_alignment.enable = False
    cfg.audio_alignment.use_vspreview = True

    files = [reference_path, target_path]
    metadata = [
        {"label": "Reference", "file_name": reference_path.name},
        {"label": "Target", "file_name": target_path.name},
    ]

    _patch_core_helper(monkeypatch, "_discover_media", lambda _root: list(files))
    _patch_core_helper(
        monkeypatch,
        "parse_metadata",
        lambda _files, _naming: list(metadata),
    )
    _patch_core_helper(
        monkeypatch,
        "_pick_analyze_file",
        lambda _files, _metadata, _target, **_kwargs: reference_path,
    )

    cache_file = cli_runner_env.media_root / cfg.analysis.frame_data_filename
    cache_file.write_text("cache", encoding="utf-8")

    init_calls: list[tuple[str, int]] = []

    def fake_init_clip(
        path: str,
        *,
        trim_start: int = 0,
        trim_end: int | None = None,
        fps_map: tuple[int, int] | None = None,
        cache_dir: str | None = None,
        **_kwargs: object,
    ) -> types.SimpleNamespace:
        init_calls.append((path, trim_start))
        return types.SimpleNamespace(
            path=path,
            width=1920,
            height=1080,
            fps_num=24000,
            fps_den=1001,
            num_frames=2400,
        )

    _patch_vs_core(monkeypatch, "init_clip", fake_init_clip)

    import src.frame_compare.orchestration.coordinator as coordinator_module
    monkeypatch.setattr(coordinator_module, "write_selection_cache_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator_module, "export_selection_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator_module, "generate_screenshots", lambda *args, **kwargs: [])

    def fake_select(
        clip: types.SimpleNamespace,
        analysis_cfg: AnalysisConfig,
        files_list: list[str],
        file_under_analysis: str,
        cache_info: FrameMetricsCacheInfo | None = None,
        progress: object = None,
        *,
        frame_window: tuple[int, int] | None = None,
        return_metadata: bool = False,
        color_cfg: ColorConfig | None = None,
        cache_probe: CacheLoadResult | None = None,
    ) -> list[int]:
        assert cache_probe is not None and cache_probe.status == "reused"
        return [10, 20]

    monkeypatch.setattr(coordinator_module, "select_frames", fake_select)

    cache_probes: list[FrameMetricsCacheInfo] = []

    def fake_probe(info: FrameMetricsCacheInfo, _analysis_cfg: AnalysisConfig) -> CacheLoadResult:
        cache_probes.append(info)
        return CacheLoadResult(metrics=None, status="reused", reason=None)

    monkeypatch.setattr(coordinator_module, "probe_cached_metrics", fake_probe)

    reporter = _RecordingOutputManager()
    env = RunEnvironment(
        preflight=PreflightResult(
            workspace_root=cli_runner_env.media_root,
            media_root=cli_runner_env.media_root,
            config_path=Path("mock_config.toml"),
            config=cfg,
            warnings=(),
        ),
        cfg=cfg,
        root=cli_runner_env.media_root,
        out_dir=cli_runner_env.media_root / "output",
        out_dir_created=False,
        out_dir_created_path=None,
        result_snapshot_path=cli_runner_env.media_root / "snapshot.json",
        analysis_cache_path=cli_runner_env.media_root / "analysis.json",
        offsets_path=cli_runner_env.media_root / "offsets.json",
        vspreview_mode_value="slider",
        layout_path=Path("layout.json"),
        reporter=reporter,
        service_mode_enabled=False,
        legacy_requested=False,
        collected_warnings=[],
        report_enabled=False,
    )

    mock_metadata = MagicMock()
    mock_metadata.resolve.return_value = MetadataResolveResult(
        plans=[
            _ClipPlan(path=reference_path, metadata={"label": "Reference"}, use_as_reference=True),
            _ClipPlan(path=target_path, metadata={"label": "Target"}),
        ],
        metadata=[{"label": "Reference"}, {"label": "Target"}],
        metadata_title="Test Title",
        analyze_path=reference_path,
        slowpics_title_inputs={"resolved_base": "Test", "collection_name": "Test", "collection_suffix": ""},
        slowpics_final_title="Test Title",
        slowpics_resolved_base="Test",
        slowpics_tmdb_disclosure_line=None,
        slowpics_verbose_tmdb_tag=None,
        tmdb_notes=[],
    )

    mock_alignment = MagicMock()
    # Create new plans with expected trims applied
    aligned_plans = [
        _ClipPlan(path=reference_path, metadata={"label": "Reference"}, use_as_reference=True, trim_start=0),
        _ClipPlan(path=target_path, metadata={"label": "Target"}, trim_start=11),
    ]

    def _alignment_side_effect(request: AlignmentRequest) -> AlignmentResult:
        request.json_tail["audio_alignment"]["manual_trim_starts"] = {
            reference_path.name: 0,
            target_path.name: 11,
        }
        request.json_tail["audio_alignment"]["offsets_frames"] = {
            "Target": 11,
            "Reference": 0,
        }
        request.json_tail["audio_alignment"]["vspreview_mode"] = "slider"
        return AlignmentResult(
            plans=aligned_plans,
            summary=None,
            display=None,
        )

    mock_alignment.run.side_effect = _alignment_side_effect

    mock_report = MagicMock()
    mock_report.publish.return_value = types.SimpleNamespace(report_path=None)

    mock_slowpics = MagicMock()
    mock_slowpics.publish.return_value = types.SimpleNamespace(url=None)

    deps = RunDependencies(
        metadata_resolver=mock_metadata,
        alignment_workflow=mock_alignment,
        report_publisher=mock_report,
        slowpics_publisher=mock_slowpics,
        setup_service=MockSetupService(env),
    )

    result = frame_compare.run_cli(
        None,
        None,
        dependencies=deps,
    )

    assert init_calls, "Clips should be initialised with trims applied"
    trims_by_path = {Path(path).name: trim for path, trim in init_calls}
    # Normalized trims: min is -3, so shift is +3.
    # Target: 8 + 3 = 11
    # Reference: -3 + 3 = 0
    assert trims_by_path[target_path.name] == 11
    assert trims_by_path[reference_path.name] == 0
    assert cache_probes and cache_probes[0].path == cache_file.resolve()
    assert result.json_tail is not None
    audio_json = _expect_mapping(result.json_tail["audio_alignment"])
    manual_map = cast(dict[str, int], audio_json.get("manual_trim_starts", {}))
    assert manual_map[target_path.name] == 11
    assert manual_map[reference_path.name] == 0
    offsets_frames = _expect_mapping(audio_json.get("offsets_frames", {}))
    assert offsets_frames.get("Target") == 11
    assert offsets_frames.get("Reference") == 0
    cache_json = _expect_mapping(result.json_tail["cache"])
    assert cache_json["status"] == "reused"
    analysis_json = _expect_mapping(result.json_tail["analysis"])
    assert analysis_json["cache_reused"] is True
    assert result.json_tail["vspreview_mode"] == "slider"
    assert result.json_tail["suggested_frames"] is None
    assert result.json_tail["suggested_seconds"] == 0.0


def test_run_cli_surfaces_cache_recompute_reason(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: _CliRunnerEnv,
) -> None:
    _, _, cache_file = _setup_cli_analysis_environment(monkeypatch, cli_runner_env)
    cache_file.write_text("cache", encoding="utf-8")

    reason_code = "config_mismatch"

    def fake_probe(info: FrameMetricsCacheInfo, _analysis_cfg: AnalysisConfig) -> CacheLoadResult:
        assert info.path == cache_file.resolve()
        return CacheLoadResult(metrics=None, status="stale", reason=reason_code)

    import src.frame_compare.orchestration.coordinator as coordinator_module
    monkeypatch.setattr(coordinator_module, "probe_cached_metrics", fake_probe)

    observed_probes: list[CacheLoadResult | None] = []

    def fake_select(
        *_args: object,
        return_metadata: bool = False,
        cache_probe: CacheLoadResult | None = None,
        **_kwargs: object,
    ):
        observed_probes.append(cache_probe)
        frames = [10, 20]
        categories = {10: "Auto", 20: "Auto"}
        if return_metadata:
            return frames, categories, {}
        return frames

    monkeypatch.setattr(coordinator_module, "select_frames", fake_select)

    result = frame_compare.run_cli(None, None)

    assert observed_probes and observed_probes[0] is not None
    assert observed_probes[0].status == "stale"
    assert result.json_tail is not None
    cache_json = _expect_mapping(result.json_tail["cache"])
    assert cache_json["status"] == "recomputed"
    assert cache_json["reason"] == reason_code
    analysis_json = _expect_mapping(result.json_tail["analysis"])
    assert analysis_json["cache_reused"] is False
    progress_message = analysis_json.get("cache_progress_message", "")
    assert "config mismatch" in progress_message.lower()
    assert progress_message.startswith("Recomputing frame metrics")


def test_run_cli_reports_missing_cache_reason(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: _CliRunnerEnv,
) -> None:
    _, _, cache_file = _setup_cli_analysis_environment(monkeypatch, cli_runner_env)

    def _probe_unused(*_args: object, **_kwargs: object) -> CacheLoadResult:
        raise AssertionError("probe_cached_metrics should not be called when cache file is missing")

    import src.frame_compare.orchestration.coordinator as coordinator_module
    monkeypatch.setattr(coordinator_module, "probe_cached_metrics", _probe_unused)

    observed_probes: list[CacheLoadResult | None] = []

    def fake_select(
        *_args: object,
        return_metadata: bool = False,
        cache_probe: CacheLoadResult | None = None,
        **_kwargs: object,
    ):
        observed_probes.append(cache_probe)
        frames = [5, 15]
        categories = {5: "Auto", 15: "Auto"}
        if return_metadata:
            return frames, categories, {}
        return frames

    monkeypatch.setattr(coordinator_module, "select_frames", fake_select)

    result = frame_compare.run_cli(None, None)

    assert observed_probes and observed_probes[0] is not None
    assert observed_probes[0].status == "missing"
    assert result.json_tail is not None
    cache_json = _expect_mapping(result.json_tail["cache"])
    assert cache_json["status"] == "recomputed"
    assert cache_json["reason"] == "missing"
    analysis_json = _expect_mapping(result.json_tail["analysis"])
    assert analysis_json["cache_reused"] is False
    progress_message = analysis_json.get("cache_progress_message", "")
    assert "missing" in progress_message.lower()
    assert progress_message.startswith("Recomputing frame metrics")

def test_audio_alignment_vspreview_suggestion_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VSPreview flow surfaces offsets without mutating trims or writing offsets."""

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True
    cfg.audio_alignment.use_vspreview = True

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
        clip=None,
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
        clip=None,
    )
    target_plan.trim_start = 120
    target_plan.has_trim_start_override = True

    measurement = AlignmentMeasurement(
        file=target_path,
        offset_seconds=0.5,
        frames=12,
        correlation=0.92,
        reference_fps=24.0,
        target_fps=24.0,
    )

    monkeypatch.setattr(
        core_module.audio_alignment,
        "probe_audio_streams",
        lambda _path: [],
    )

    def _fake_measure(
        _ref: Path,
        targets: list[Path],
        *,
        progress_callback,
        **_kwargs: object,
    ):
        progress_callback(len(targets))
        return [measurement]

    monkeypatch.setattr(
        core_module.audio_alignment,
        "measure_offsets",
        _fake_measure,
    )
    monkeypatch.setattr(
        core_module.audio_alignment,
        "load_offsets",
        lambda _path: (None, {}),
    )

    def _fail_update(*_args, **_kwargs):
        raise AssertionError("update_offsets_file should not be called in VSPreview mode")

    monkeypatch.setattr(
        core_module.audio_alignment,
        "update_offsets_file",
        _fail_update,
    )

    summary, display = core_module._maybe_apply_audio_alignment(
        [reference_plan, target_plan],
        cfg,
        reference_path,
        tmp_path,
        {},
        reporter=None,
    )

    assert summary is not None
    assert display is not None
    assert summary.suggestion_mode is True
    assert summary.applied_frames == {target_path.name: 120}
    assert summary.suggested_frames[target_path.name] == 12
    assert summary.manual_trim_starts[target_path.name] == 120
    assert target_plan.trim_start == 120, "Trim should remain unchanged in suggestion mode"
    assert any("VSPreview manual alignment enabled" in warning for warning in display.warnings)
    assert any("Existing manual trim" in line for line in display.offset_lines)

def test_launch_vspreview_generates_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vspreview_env,
) -> None:
    """VSPreview launcher should emit a script and attempt to execute it."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.use_vspreview = True

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )
    target_plan.trim_start = 10
    target_plan.has_trim_start_override = True
    plans = [reference_plan, target_plan]

    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.toml",
        reference_name=reference_path.name,
        measurements=(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={target_path.name: 7},
        suggestion_mode=True,
        manual_trim_starts={target_path.name: 10},
    )

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()
    audio_block = json_tail["audio_alignment"]

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    recorded_command: list[list[str]] = []

    class _Result:
        def __init__(self, returncode: int = 0) -> None:
            self.returncode = returncode

    vspreview_env(True)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: True if name in ("vspreview", "PySide6", "PyQt5") else None,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, env=None, check=False, **kwargs: recorded_command.append(list(cmd)) or _Result(0),
    )

    display = _AudioAlignmentDisplayData(
        stream_lines=[],
        estimation_line=None,
        offset_lines=[],
        offsets_file_line="Offsets file: offsets.toml",
        json_reference_stream=None,
        json_target_streams={},
        json_offsets_sec={},
        json_offsets_frames={},
        warnings=[],
    )

    prompt_calls: list[dict[str, int] | None] = []
    _patch_core_helper(
        monkeypatch,
        "_prompt_vspreview_offsets",
        lambda *args, **kwargs: prompt_calls.append({}) or {},
    )

    apply_calls: list[Mapping[str, int]] = []

    def _record_apply(
        _plans: Sequence[_ClipPlan],
        _summary: _AudioAlignmentSummary,
        offsets: Mapping[str, int],
        *_args: object,
        **_kwargs: object,
    ) -> None:
        apply_calls.append(dict(offsets))

    _patch_core_helper(monkeypatch, "_apply_vspreview_manual_offsets", _record_apply)

    core_module._launch_vspreview(plans, summary, display, cfg, tmp_path, reporter, json_tail)

    script_path_str = audio_block.get("vspreview_script")
    assert script_path_str, "Script path should be recorded in JSON tail"
    script_path = Path(script_path_str)
    assert script_path.exists()
    script_text = script_path.read_text(encoding="utf-8")
    assert "OFFSET_MAP" in script_text
    assert "vs_core.configure" in script_text
    assert "ColorConfig" in script_text
    assert "AssumeFPS" in script_text
    assert "PREVIEW_MODE = 'baseline'" in script_text
    assert "SHOW_SUGGESTED_OVERLAY = True" in script_text
    assert "'Target': 0,  # Suggested delta +7f" in script_text
    assert "SUGGESTION_MAP" in script_text
    assert "'Target': (7, 0.0)" in script_text
    assert "message = _format_overlay_text" in script_text
    assert (
        "def _format_overlay_text(label, suggested_frames, suggested_seconds, applied_frames):"
        in script_text
    )
    assert '"{label}: {suggested} (~{seconds}s) • "' in script_text
    assert "Preview applied: {applied}f ({status})" in script_text
    assert "preview applied=%+df" in script_text
    assert recorded_command, "VSPreview command should be invoked when interactive"
    assert recorded_command[0][0] == sys.executable
    assert recorded_command[0][-1] == str(script_path)
    assert audio_block.get("vspreview_invoked") is True
    assert audio_block.get("vspreview_exit_code") == 0
    assert prompt_calls, "Prompt should be invoked even when returning default offsets"
    assert apply_calls == [{}]

def test_launch_vspreview_baseline_mode_persists_manual_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Baseline preview emits zeroed offsets yet records manual selections."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.use_vspreview = True
    cfg.audio_alignment.vspreview_mode = "baseline"

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )
    target_plan.trim_start = 2
    target_plan.has_trim_start_override = True
    plans = [reference_plan, target_plan]

    measurement = AlignmentMeasurement(
        file=target_path,
        offset_seconds=0.375,
        frames=9,
        correlation=0.91,
        reference_fps=24.0,
        target_fps=24.0,
    )

    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.toml",
        reference_name=reference_path.name,
        measurements=(measurement,),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={target_path.name: 9},
        suggestion_mode=True,
        manual_trim_starts={target_path.name: 2},
    )

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()
    audio_block = json_tail["audio_alignment"]

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    importlib_module = cast(Any, core_module.importlib)
    monkeypatch.setattr(importlib_module.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, env=None, check=False, **kwargs: types.SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    _patch_core_helper(
        monkeypatch,
        "_prompt_vspreview_offsets",
        lambda *args, **kwargs: {target_path.name: 3},
    )
    _patch_audio_alignment(
        monkeypatch,
        "update_offsets_file",
        lambda *_args, **_kwargs: (
            {reference_path.name: 0, target_path.name: 5},
            {reference_path.name: "manual", target_path.name: "manual"},
        ),
    )

    core_module._launch_vspreview(plans, summary, None, cfg, tmp_path, reporter, json_tail)

    script_path_str = audio_block.get("vspreview_script")
    assert script_path_str, "Script path should be recorded in JSON tail"
    script_text = Path(script_path_str).read_text(encoding="utf-8")
    assert "'Target': 0,  # Suggested delta +9f" in script_text
    assert summary.vspreview_manual_offsets[target_path.name] == 5
    assert summary.vspreview_manual_deltas[target_path.name] == 3
    manual_json = cast(dict[str, int], audio_block.get("vspreview_manual_offsets", {}))
    assert manual_json[target_path.name] == 5
    delta_json = cast(dict[str, int], audio_block.get("vspreview_manual_deltas", {}))
    assert delta_json[target_path.name] == 3


def test_vspreview_suggestions_use_measured_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VSPreview hints should reflect measured offsets without adjustment."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True
    cfg.audio_alignment.use_vspreview = True

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )
    plans = [reference_plan, target_plan]

    measurement = AlignmentMeasurement(
        file=target_path,
        offset_seconds=0.0417,
        frames=1,
        correlation=0.92,
        reference_fps=24.0,
        target_fps=24.0,
    )
    stream_info = AudioStreamInfo(
        index=0,
        language="eng",
        codec_name="aac",
        channels=2,
        channel_layout="stereo",
        sample_rate=48000,
        bitrate=128_000,
        is_default=True,
        is_forced=False,
    )

    _patch_audio_alignment(
        monkeypatch,
        "measure_offsets",
        lambda *_args, **_kwargs: [measurement],
    )
    _patch_audio_alignment(monkeypatch, "probe_audio_streams", lambda _path: [stream_info])

    summary, _display = alignment_package.apply_audio_alignment(
        plans,
        cfg,
        analyze_path=tmp_path,
        root=tmp_path,
        audio_track_overrides={},
        reporter=None,
    )

    assert summary is not None, "alignment summary should be available for VSPreview suggestions"

    assert summary.suggested_frames[target_path.name] == 1
    detail = summary.measured_offsets[target_path.name]
    assert detail.frames == 1
    assert detail.offset_seconds == pytest.approx(measurement.offset_seconds, rel=1e-6)


def test_write_vspreview_script_generates_unique_filenames_same_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VSPreview script writes should never clobber same-second launches."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.use_vspreview = True

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )
    plans = [reference_plan, target_plan]

    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.toml",
        reference_name=reference_path.name,
        measurements=(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={target_path.name: 7},
        suggestion_mode=True,
        manual_trim_starts={target_path.name: 10},
    )

    fixed_instant = dt.datetime(2024, 1, 1, 12, 34, 56)

    class _FixedDatetime(dt.datetime):
        @classmethod
        def now(cls, tz: dt.tzinfo | None = None) -> dt.datetime:
            return fixed_instant if tz is None else fixed_instant.replace(tzinfo=tz)

    monkeypatch.setattr(core_module._dt, "datetime", _FixedDatetime)

    first_path = core_module._write_vspreview_script(plans, summary, cfg, tmp_path)
    second_path = core_module._write_vspreview_script(plans, summary, cfg, tmp_path)

    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()
    assert first_path.name != second_path.name

def test_launch_vspreview_warns_when_command_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vspreview_env,
) -> None:
    """VSPreview launcher should fall back cleanly when no executable is available."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.use_vspreview = True

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )
    plans = [reference_plan, target_plan]

    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.toml",
        reference_name=reference_path.name,
        measurements=(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={target_path.name: 4},
        suggestion_mode=True,
        manual_trim_starts={target_path.name: 2},
    )

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()
    audio_block = json_tail["audio_alignment"]

    display = _AudioAlignmentDisplayData(
        stream_lines=[],
        estimation_line=None,
        offset_lines=[],
        offsets_file_line="Offsets file: offsets.toml",
        json_reference_stream=None,
        json_target_streams={},
        json_offsets_sec={},
        json_offsets_frames={},
        warnings=[],
    )

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    vspreview_env(False)

    prompt_called: list[None] = []

    def _fail_prompt(*_args: object, **_kwargs: object) -> dict[str, int]:
        prompt_called.append(None)
        return {}

    _patch_core_helper(monkeypatch, "_prompt_vspreview_offsets", _fail_prompt)

    core_module._launch_vspreview(plans, summary, display, cfg, tmp_path, reporter, json_tail)

    script_path_str = audio_block.get("vspreview_script")
    assert script_path_str, "Script path should still be recorded for manual launches"
    assert Path(script_path_str).exists()
    assert audio_block.get("vspreview_invoked") is False
    assert audio_block.get("vspreview_exit_code") is None
    assert not prompt_called, "Prompt should not run when VSPreview cannot launch"
    warnings = reporter.get_warnings()
    assert any("VSPreview dependencies missing" in warning for warning in warnings)
    layout_state = reporter.values.get("vspreview", {})
    missing_state = cast(dict[str, object], layout_state.get("missing", {}))
    assert missing_state.get("active") is True
    expected_command = _format_vspreview_manual_command(
        Path(script_path_str)
    )
    assert missing_state.get("command") == expected_command
    offer_entry = json_tail.get("vspreview_offer")
    assert offer_entry == {
        "vspreview_offered": False,
        "reason": "vspreview-executable-missing",
    }
    console_output = reporter.console.export_text()
    normalized_output = " ".join(console_output.split())
    assert "VSPreview dependency missing" in normalized_output
    expected_windows_install = " ".join(_VSPREVIEW_WINDOWS_INSTALL.split())
    assert expected_windows_install in normalized_output
    python_executable = sys.executable or "python"
    assert python_executable in normalized_output
    assert "-m vspreview" in normalized_output

def test_format_alignment_output_updates_json_tail(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True

    reference_plan = _ClipPlan(path=tmp_path / "Ref.mkv", metadata={"label": "Reference"})
    target_plan = _ClipPlan(path=tmp_path / "Target.mkv", metadata={"label": "Target"})
    plans = [reference_plan, target_plan]

    offsets_path = tmp_path / "offsets.json"
    summary = _AudioAlignmentSummary(
        offsets_path=offsets_path,
        reference_name=reference_plan.path.name,
        measurements=tuple(),
        applied_frames={reference_plan.path.name: 0, target_plan.path.name: 3},
        baseline_shift=0,
        statuses={target_plan.path.name: "auto"},
        reference_plan=reference_plan,
        final_adjustments={target_plan.path.name: 3},
        swap_details={},
        suggested_frames={target_plan.path.name: 5},
        suggestion_mode=False,
        manual_trim_starts={target_plan.path.name: 12},
    )
    detail = alignment_package.AudioMeasurementDetail(
        label="Target",
        stream="1/2",
        offset_seconds=0.25,
        frames=3,
        correlation=0.95,
        status="auto",
        applied=True,
        note="ok",
    )
    summary.measured_offsets = {target_plan.path.name: detail}

    display = _AudioAlignmentDisplayData(
        stream_lines=["Reference: Audio"],
        estimation_line="Peak correlation 0.95",
        offset_lines=["Audio offsets: Target +3f"],
        offsets_file_line=f"Offsets file: {offsets_path}",
        json_reference_stream="Ref Track",
        json_target_streams={"Target": "Track 1"},
        json_offsets_sec={"Target": 0.25},
        json_offsets_frames={"Target": 3},
        warnings=["[AUDIO] verify offsets"],
    )
    display.measurements = {"Target": detail}

    reporter = NullCliOutputManager(quiet=True, verbose=False, no_color=True)
    json_tail = _make_json_tail_stub()
    collected_warnings: list[str] = []

    alignment_package.format_alignment_output(
        plans,
        summary,
        display,
        cfg=cfg,
        root=tmp_path,
        reporter=reporter,
        json_tail=json_tail,
        vspreview_mode="baseline",
        collected_warnings=collected_warnings,
    )

    assert json_tail["suggested_frames"] == 5
    audio_block = json_tail["audio_alignment"]
    offsets_frames = audio_block.get("offsets_frames", {})
    assert offsets_frames.get("Target") == 3
    assert audio_block.get("suggestion_mode") is False
    assert collected_warnings == display.warnings


def test_format_alignment_output_preserves_negative_manual_trim(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True

    reference_plan = _ClipPlan(path=tmp_path / "Ref.mkv", metadata={"label": "Reference"})
    target_plan = _ClipPlan(path=tmp_path / "Target.mkv", metadata={"label": "Target"})
    plans = [reference_plan, target_plan]

    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.json",
        reference_name=reference_plan.path.name,
        measurements=tuple(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={},
        suggestion_mode=False,
        manual_trim_starts={target_plan.path.name: -6},
    )

    display = _AudioAlignmentDisplayData(
        stream_lines=[],
        estimation_line=None,
        offset_lines=[],
        offsets_file_line=f"Offsets file: {tmp_path / 'offsets.json'}",
        json_reference_stream=None,
        json_target_streams={},
        json_offsets_sec={},
        json_offsets_frames={},
        warnings=[],
        manual_trim_lines=[f"Existing manual trim: {target_plan.metadata['label']} → -6f"],
    )

    reporter = NullCliOutputManager(quiet=True, verbose=False, no_color=True)
    json_tail = _make_json_tail_stub()

    alignment_package.format_alignment_output(
        plans,
        summary,
        display,
        cfg=cfg,
        root=tmp_path,
        reporter=reporter,
        json_tail=json_tail,
        vspreview_mode="baseline",
        collected_warnings=None,
    )

    assert json_tail["suggested_frames"] is None
    audio_block = json_tail["audio_alignment"]
    manual_summary = audio_block.get("manual_trim_summary", [])
    assert manual_summary == display.manual_trim_lines
    manual_map = audio_block.get("manual_trim_starts", {})
    assert manual_map[target_plan.path.name] == -6

def test_vspreview_manual_offsets_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_plan = _ClipPlan(path=reference_path, metadata={"label": "Reference"})
    target_plan = _ClipPlan(path=target_path, metadata={"label": "Target"})
    target_plan.trim_start = 5
    target_plan.has_trim_start_override = True
    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.toml",
        reference_name=reference_path.name,
        measurements=(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={target_path.name: 3},
        suggestion_mode=True,
        manual_trim_starts={target_path.name: 5},
    )

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()
    display = _make_display_stub()

    captured: dict[str, object] = {}

    def fake_update(
        path: Path,
        reference_name: str,
        measurements: Sequence[AlignmentMeasurement],
        existing: Mapping[str, Mapping[str, object]],
        notes: Mapping[str, str],
    ) -> tuple[dict[str, int], dict[str, str]]:
        captured["path"] = path
        captured["reference"] = reference_name
        captured["measurements"] = list(measurements)
        captured["existing"] = dict(existing)
        captured["notes"] = dict(notes)
        applied = {m.file.name: int(m.frames or 0) for m in measurements}
        return applied, {name: "manual" for name in applied}

    _patch_audio_alignment(monkeypatch, "update_offsets_file", fake_update)

    core_module._apply_vspreview_manual_offsets(
        [reference_plan, target_plan],
        summary,
        {target_path.name: 7},
        reporter,
        json_tail,
        display,
    )

    assert target_plan.trim_start == 12
    assert summary.suggestion_mode is False
    assert summary.manual_trim_starts[target_path.name] == 12
    assert summary.vspreview_manual_offsets[target_path.name] == 12
    assert summary.vspreview_manual_deltas[target_path.name] == 7
    audio_block = json_tail["audio_alignment"]
    offsets_map = cast(dict[str, int], audio_block.get("vspreview_manual_offsets", {}))
    deltas_map = cast(dict[str, int], audio_block.get("vspreview_manual_deltas", {}))
    assert offsets_map[target_path.name] == 12
    assert deltas_map[target_path.name] == 7
    measurements_list = cast(list[AlignmentMeasurement], captured["measurements"])
    assert measurements_list
    target_measurements = [
        int(measurement.frames or 0)
        for measurement in measurements_list
        if measurement.file.name == target_path.name
    ]
    assert target_measurements and target_measurements[0] == 7
    notes_map = cast(dict[str, str], captured["notes"])
    existing_map = cast(dict[str, Mapping[str, object]], captured["existing"])
    assert notes_map[target_path.name] == "VSPreview"
    entry = cast(dict[str, object], existing_map[target_path.name])
    assert entry.get("status") == "manual"
    assert int(cast(int | float, entry.get("frames", 0))) == 7
    assert any("VSPreview manual offset applied" in line for line in reporter.lines)

def test_vspreview_manual_offsets_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_plan = _ClipPlan(path=reference_path, metadata={"label": "Reference"})
    target_plan = _ClipPlan(path=target_path, metadata={"label": "Target"})
    target_plan.trim_start = 4
    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.toml",
        reference_name=reference_path.name,
        measurements=(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={target_path.name: 0},
        suggestion_mode=True,
        manual_trim_starts={target_path.name: 4},
    )

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()
    display = _make_display_stub()

    monkeypatch.setattr(
        core_module.audio_alignment,
        "update_offsets_file",
        lambda *_args, **_kwargs: ({target_path.name: 4, reference_path.name: 0}, {target_path.name: "manual", reference_path.name: "manual"}),
    )

    core_module._apply_vspreview_manual_offsets(
        [reference_plan, target_plan],
        summary,
        {target_path.name: 0},
        reporter,
        json_tail,
        display,
    )

    assert target_plan.trim_start == 4
    assert summary.manual_trim_starts[target_path.name] == 4
    assert summary.vspreview_manual_deltas[target_path.name] == 0
    audio_block = json_tail["audio_alignment"]
    offsets_map = cast(dict[str, int], audio_block.get("vspreview_manual_offsets", {}))
    assert offsets_map[target_path.name] == 4

def test_vspreview_manual_offsets_negative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_plan = _ClipPlan(path=reference_path, metadata={"label": "Reference"})
    target_plan = _ClipPlan(path=target_path, metadata={"label": "Target"})
    target_plan.trim_start = 3
    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.toml",
        reference_name=reference_path.name,
        measurements=(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={target_path.name: -5},
        suggestion_mode=True,
        manual_trim_starts={target_path.name: 3},
    )

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()
    display = _make_display_stub()

    monkeypatch.setattr(
        core_module.audio_alignment,
        "update_offsets_file",
        lambda *_args, **_kwargs: (
            {target_path.name: 0, reference_path.name: 4},
            {target_path.name: "manual", reference_path.name: "manual"},
        ),
    )

    core_module._apply_vspreview_manual_offsets(
        [reference_plan, target_plan],
        summary,
        {target_path.name: -7},
        reporter,
        json_tail,
        display,
    )

    assert target_plan.trim_start == 0
    assert reference_plan.trim_start == 4
    assert summary.manual_trim_starts[target_path.name] == 0
    assert summary.vspreview_manual_offsets[reference_path.name] == 4
    assert summary.vspreview_manual_deltas[target_path.name] == -3
    assert summary.vspreview_manual_deltas[reference_path.name] == 4
    audio_block = json_tail["audio_alignment"]
    manual_map = cast(dict[str, int], audio_block.get("manual_trim_starts", {}))
    assert manual_map[target_path.name] == 0
    assert audio_block.get("vspreview_reference_trim") == 4
    assert any("VSPreview manual offset applied" in line and reference_path.name in line for line in reporter.lines)
    assert any("Target" in line and "0f" in line for line in display.manual_trim_lines)

def test_vspreview_manual_offsets_multiple_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_path = tmp_path / "Ref.mkv"
    target_a_path = tmp_path / "A.mkv"
    target_b_path = tmp_path / "B.mkv"
    reference_plan = _ClipPlan(path=reference_path, metadata={"label": "Reference"})
    target_a_plan = _ClipPlan(path=target_a_path, metadata={"label": "A"})
    target_b_plan = _ClipPlan(path=target_b_path, metadata={"label": "B"})
    target_a_plan.trim_start = 5
    target_b_plan.trim_start = 5
    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.toml",
        reference_name=reference_path.name,
        measurements=(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={
            target_a_path.name: -3,
            target_b_path.name: -7,
        },
        suggestion_mode=True,
        manual_trim_starts={
            target_a_path.name: 5,
            target_b_path.name: 5,
        },
    )

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()
    display = _make_display_stub()

    captured: dict[str, object] = {}

    def fake_update(
        path: Path,
        reference_name: str,
        measurements: Sequence[AlignmentMeasurement],
        existing: Mapping[str, Mapping[str, object]],
        notes: Mapping[str, str],
    ) -> tuple[dict[str, int], dict[str, str]]:
        captured["path"] = path
        captured["reference"] = reference_name
        captured["measurements"] = list(measurements)
        captured["existing"] = dict(existing)
        captured["notes"] = dict(notes)
        applied = {m.file.name: int(m.frames or 0) for m in measurements}
        return applied, {name: "manual" for name in applied}

    _patch_audio_alignment(monkeypatch, "update_offsets_file", fake_update)

    core_module._apply_vspreview_manual_offsets(
        [reference_plan, target_a_plan, target_b_plan],
        summary,
        {target_a_path.name: -3, target_b_path.name: -7},
        reporter,
        json_tail,
        display,
    )

    assert target_a_plan.trim_start == 4
    assert target_b_plan.trim_start == 0
    assert reference_plan.trim_start == 2
    assert summary.suggestion_mode is False
    assert summary.manual_trim_starts[target_a_path.name] == 4
    assert summary.manual_trim_starts[target_b_path.name] == 0
    assert summary.vspreview_manual_offsets[target_a_path.name] == 4
    assert summary.vspreview_manual_offsets[target_b_path.name] == 0
    assert summary.vspreview_manual_offsets[reference_path.name] == 2
    assert summary.vspreview_manual_deltas[target_a_path.name] == -1
    assert summary.vspreview_manual_deltas[target_b_path.name] == -5
    assert summary.vspreview_manual_deltas[reference_path.name] == 2

    audio_block = json_tail["audio_alignment"]
    offsets_map = cast(dict[str, int], audio_block.get("vspreview_manual_offsets", {}))
    deltas_map = cast(dict[str, int], audio_block.get("vspreview_manual_deltas", {}))
    assert offsets_map[target_a_path.name] == 4
    assert offsets_map[target_b_path.name] == 0
    assert offsets_map[reference_path.name] == 2
    assert deltas_map[target_a_path.name] == -1
    assert deltas_map[target_b_path.name] == -5
    assert deltas_map[reference_path.name] == 2

    measurements = cast(list[AlignmentMeasurement], captured["measurements"])
    assert {m.file.name for m in measurements} == {
        reference_path.name,
        target_a_path.name,
        target_b_path.name,
    }
    assert any("manual offset applied" in line for line in reporter.lines)

def test_runner_audio_alignment_summary_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    media_root = workspace / "media"
    workspace.mkdir(parents=True, exist_ok=True)
    media_root.mkdir(parents=True, exist_ok=True)
    for name in ("Reference.mkv", "Target.mkv"):
        (media_root / name).write_bytes(b"data")

    cfg = _make_config(media_root)
    cfg.tmdb.api_key = "token"
    cfg.slowpics.auto_upload = False
    cfg.analysis.frame_count_dark = 0
    cfg.analysis.frame_count_bright = 0
    cfg.analysis.frame_count_motion = 0
    cfg.analysis.random_frames = 0
    cfg.analysis.save_frames_data = False

    preflight = _make_runner_preflight(workspace, media_root, cfg)
    _patch_core_helper(monkeypatch, "prepare_preflight", lambda **_: preflight)

    files = [media_root / "Reference.mkv", media_root / "Target.mkv"]
    metadata = [{"label": "Reference"}, {"label": "Target"}]
    plans = [
        core_module._ClipPlan(path=files[0], metadata={"label": "Reference"}),
        core_module._ClipPlan(path=files[1], metadata={"label": "Target"}),
    ]
    plans[0].use_as_reference = True

    _patch_core_helper(monkeypatch, "_discover_media", lambda _root: list(files))
    _patch_core_helper(monkeypatch, "parse_metadata", lambda *_: list(metadata))
    _patch_core_helper(monkeypatch, "_build_plans", lambda *_: list(plans))
    monkeypatch.setattr(core_module, "_pick_analyze_file", lambda *_args, **_kwargs: files[0])

    cache_info = FrameMetricsCacheInfo(
        path=workspace / cfg.analysis.frame_data_filename,
        files=[file.name for file in files],
        analyzed_file=files[0].name,
        release_group="",
        trim_start=0,
        trim_end=None,
        fps_num=24000,
        fps_den=1001,
    )
    _patch_core_helper(monkeypatch, "_build_cache_info", lambda *_: cache_info)

    summary = _AudioAlignmentSummary(
        offsets_path=workspace / "generated.audio_offsets.toml",
        reference_name="Reference",
        measurements=[],
        applied_frames={files[1].name: 8},
        baseline_shift=0,
        statuses={files[1].name: "manual"},
        reference_plan=plans[0],
        final_adjustments={files[1].name: 8},
        swap_details={},
        suggested_frames={},
        suggestion_mode=False,
        manual_trim_starts={files[1].name: 8},
        vspreview_manual_offsets={files[1].name: 8},
        vspreview_manual_deltas={files[1].name: 8},
        measured_offsets={},
    )
    display = _AudioAlignmentDisplayData(
        stream_lines=["Reference stream"],
        estimation_line="Audio offsets reused from VSPreview",
        offset_lines=["VSPreview manual trim reused"],
        offsets_file_line="Offsets file: generated.audio_offsets.toml",
        json_reference_stream="Reference",
        json_target_streams={files[1].name: "0"},
        json_offsets_sec={files[1].name: 0.333},
        json_offsets_frames={files[1].name: 8},
        warnings=[],
        preview_paths=[],
        confirmation=None,
        correlations={files[1].name: 0.98},
        threshold=0.55,
        manual_trim_lines=["Target -> 8f"],
        measurements={},
    )

    _patch_core_helper(
        monkeypatch,
        "_maybe_apply_audio_alignment",
        lambda *args, **kwargs: (summary, display),
    )

    monkeypatch.setattr(vs_core_module, "configure", lambda **_: None)
    monkeypatch.setattr(vs_core_module, "set_ram_limit", lambda *_: None)
    monkeypatch.setattr(vs_core_module, "init_clip", lambda *args, **kwargs: types.SimpleNamespace(num_frames=120, fps_num=24000, fps_den=1001, width=1280, height=720))
    import src.frame_compare.orchestration.coordinator as coordinator_module
    monkeypatch.setattr(
        coordinator_module,
        "probe_cached_metrics",
        lambda *_: CacheLoadResult(metrics=None, status="missing", reason=None),
    )
    monkeypatch.setattr(coordinator_module, "selection_hash_for_config", lambda *_: "selection-hash")
    monkeypatch.setattr(coordinator_module, "write_selection_cache_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator_module, "export_selection_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        coordinator_module,
        "select_frames",
        lambda *args, **kwargs: ([5], {5: "Auto"}, {5: SelectionDetail(frame_index=5, label="Auto", score=None, source="Test", timecode="00:00:05.0")}),
    )
    monkeypatch.setattr(coordinator_module, "selection_details_to_json", _selection_details_to_json)
    monkeypatch.setattr(coordinator_module, "generate_screenshots", lambda *args, **kwargs: [str(media_root / "shot.png")])

    monkeypatch.setattr(runner_module, "impl", frame_compare, raising=False)
    request = runner_module.RunRequest(
        config_path=str(preflight.config_path),
        root_override=str(workspace),
    )
    result = runner_module.run(request)

    assert result.json_tail is not None
    audio_json = _expect_mapping(result.json_tail["audio_alignment"])
    manual_trims = cast(dict[str, int], audio_json.get("manual_trim_starts"))
    assert manual_trims[files[1].name] == 8
    assert any("VSPreview" in line for line in audio_json.get("offset_lines", []))
    assert audio_json.get("vspreview_invoked") is False
    assert audio_json.get("offsets_sec").get(files[1].name) == pytest.approx(0.333)  # type: ignore[arg-type]

def test_audio_alignment_block_and_json(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    cli_runner_env: _CliRunnerEnv,
) -> None:
    reference_path = cli_runner_env.media_root / "ClipA.mkv"
    target_path = cli_runner_env.media_root / "ClipB.mkv"
    for file in (reference_path, target_path):
        file.write_bytes(b"data")

    cfg = _make_config(cli_runner_env.media_root)
    cfg.audio_alignment.enable = True
    cfg.audio_alignment.max_offset_seconds = 5.0
    cfg.audio_alignment.offsets_filename = "alignment.toml"
    cfg.audio_alignment.start_seconds = 0.25
    cfg.audio_alignment.duration_seconds = 1.5
    cfg.color.overlay_mode = "diagnostic"

    cli_runner_env.reinstall(cfg)

    def fake_parse(name: str, **_kwargs: object) -> dict[str, str]:
        """
        Create a minimal fake parse result for a clip name.

        Parameters:
            name (str): Clip identifier or filename used to derive the returned label. Additional keyword arguments are ignored.

        Returns:
            dict: Mapping with keys:
                - "label" (str): "Clip A" if `name` starts with "ClipA", otherwise "Clip B".
                - "file_name" (str): The original `name` value.
        """
        if name.startswith("ClipA"):
            return {"label": "Clip A", "file_name": name}
        return {"label": "Clip B", "file_name": name}

    _patch_core_helper(monkeypatch, "parse_filename_metadata", fake_parse)
    _patch_vs_core(monkeypatch, "set_ram_limit", lambda limit: None)

    init_calls: list[tuple[str, int]] = []

    def fake_init_clip(
        path: str | Path,
        *,
        trim_start: int = 0,
        trim_end: int | None = None,
        fps_map: tuple[int, int] | None = None,
        cache_dir: str | Path | None = None,
        **_kwargs: object,
    ) -> types.SimpleNamespace:
        """
        Create a lightweight fake clip object for tests that resembles the real clip interface.

        Parameters:
            path: Path-like or str specifying the clip file path.
            trim_start (int): Ignored in this fake; present for compatibility with callers.
            trim_end (int | None): Ignored in this fake; present for compatibility with callers.
            fps_map: Ignored in this fake; present for compatibility with callers.
            cache_dir: Ignored in this fake; present for compatibility with callers.

        Returns:
            SimpleNamespace: An object with attributes:
                - path (Path): Resolved Path of the provided `path`.
                - width (int): Horizontal resolution (1920).
                - height (int): Vertical resolution (1080).
                - fps_num (int): Frame rate numerator (24000).
                - fps_den (int): Frame rate denominator (1001).
                - num_frames (int): Total frame count (24000).
        """
        base_frames = 24000
        clip_name = Path(path).name
        init_calls.append((clip_name, int(trim_start)))
        trim_value = int(trim_start)
        if trim_value >= 0:
            frames_after_trim = max(base_frames - trim_value, 0)
        else:
            frames_after_trim = base_frames + abs(trim_value)
        return types.SimpleNamespace(
            path=Path(path),
            width=1920,
            height=1080,
            fps_num=24000,
            fps_den=1001,
            num_frames=frames_after_trim,
        )

    _patch_vs_core(monkeypatch, "init_clip", fake_init_clip)

    _patch_runner_module(monkeypatch, "select_frames", lambda *args, **kwargs: [42])

    def fake_generate(
        clips: list[types.SimpleNamespace],
        frames: list[int],
        files: list[str],
        metadata: list[dict[str, object]],
        out_dir: Path,
        cfg_screens: ScreenshotConfig,
        color_cfg: ColorConfig,
        **kwargs: object,
    ) -> list[str]:
        """
        Create a fake set of screenshot files in out_dir and return their file paths.

        This helper ensures out_dir exists and produces a list of string paths representing generated shot images; the number of returned paths is len(frames) * len(files).

        Parameters:
            out_dir (Path): Directory where fake screenshot files are created.
            frames (Sequence): Sequence of frame descriptors used to determine per-file shot count.
            files (Sequence): Sequence of input files; combined with frames to compute total shots.

        Returns:
            list[str]: Paths to the generated shot image files as strings.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        return [str(out_dir / f"shot_{idx}.png") for idx in range(len(frames) * len(files))]

    _patch_runner_module(monkeypatch, "generate_screenshots", fake_generate)

    def fake_probe(path: Path) -> list[AudioStreamInfo]:
        """
        Create a fake audio probe result for the given file path.

        Parameters:
            path (Path): File path to probe; compared against the module-level `reference_path` to determine which mock stream to return.

        Returns:
            list[AudioStreamInfo]: A single-item list with a mocked audio stream. If `path == reference_path` the stream has `index=0`, `language='eng'`, and `is_default=True`; otherwise the stream has `index=1`, `language='jpn'`, and `is_default=False`.
        """
        if Path(path) == reference_path:
            return [
                AudioStreamInfo(
                    index=0,
                    language="eng",
                    codec_name="aac",
                    channels=2,
                    channel_layout="stereo",
                    sample_rate=48000,
                    bitrate=192000,
                    is_default=True,
                    is_forced=False,
                )
            ]
        return [
            AudioStreamInfo(
                index=1,
                language="jpn",
                codec_name="aac",
                channels=2,
                channel_layout="stereo",
                sample_rate=48000,
                bitrate=192000,
                is_default=False,
                is_forced=False,
            )
        ]

    _patch_audio_alignment(monkeypatch, "probe_audio_streams", fake_probe)

    measurement = AlignmentMeasurement(
        file=target_path,
        offset_seconds=0.1,
        frames=None,
        correlation=0.93,
        reference_fps=None,
        target_fps=None,
    )

    captured_measure_kwargs: dict[str, object] = {}

    def fake_measure(*args: object, **kwargs: object) -> list[AlignmentMeasurement]:
        captured_measure_kwargs.update(kwargs)
        fps_hints_obj = kwargs.get("fps_hints")
        assert isinstance(fps_hints_obj, Mapping)
        fps_hints = cast(FpsHintMap, fps_hints_obj)
        cached_target_fps = fps_hints.get(target_path)
        assert cached_target_fps is not None, "Expected cached FPS hint for target clip"
        if isinstance(cached_target_fps, tuple):
            fps_float = alignment_core_module._fps_to_float(cached_target_fps)
        else:
            fps_float = float(cached_target_fps)
        measurement.target_fps = fps_float
        offset_seconds = float(measurement.offset_seconds or 0.0)
        measurement.frames = int(round(offset_seconds * fps_float))
        cached_ref_fps = fps_hints.get(reference_path)
        if cached_ref_fps is not None:
            measurement.reference_fps = (
                alignment_core_module._fps_to_float(cached_ref_fps)
                if isinstance(cached_ref_fps, tuple)
                else float(cached_ref_fps)
            )
        return [measurement]

    _patch_audio_alignment(monkeypatch, "measure_offsets", fake_measure)
    _patch_audio_alignment(monkeypatch, "load_offsets", lambda *_args, **_kwargs: ({}, {}))

    def fake_update(
        _path: Path,
        reference_name: str,
        measurements: Iterable[AlignmentMeasurement],
        _existing: Mapping[str, int],
        _negative_notes: Mapping[str, str],
    ) -> tuple[dict[str, int], dict[str, str]]:
        """
        Produce applied frame indices and status labels for a set of measurement objects.

        This test helper assigns 0 to the provided reference_name and, for each item in measurements,
        maps the measurement's file name to its frames value or 0 when frames is falsy. It also
        marks every measurement's status as "auto".

        Parameters:
            reference_name (str): Identifier to be added to the applied frames mapping with value 0.
            measurements (Iterable): Iterable of objects with `file.name` and `frames` attributes.

        Returns:
            tuple: A pair (applied_frames, statuses).
            - applied_frames (dict): Mapping of names (reference_name and each measurement.file.name) to integer frame indices.
            - statuses (dict): Mapping of each measurement.file.name to the string `"auto"`.
        """
        applied_frames: dict[str, int] = {reference_name: 0}
        applied_frames.update({m.file.name: m.frames or 0 for m in measurements})
        statuses: dict[str, str] = {m.file.name: "auto" for m in measurements}
        return applied_frames, statuses

    _patch_audio_alignment(monkeypatch, "update_offsets_file", fake_update)

    result: Result = runner.invoke(frame_compare.main, ["--no-color"], catch_exceptions=False)
    assert result.exit_code == 0
    fps_hints = captured_measure_kwargs.get("fps_hints")
    assert isinstance(fps_hints, Mapping)
    assert reference_path in fps_hints
    assert target_path in fps_hints

    output_lines: list[str] = result.output.splitlines()
    assert any("alignment.toml" in line for line in output_lines)
    assert "mode=diagnostic" in result.output

    json_start = result.output.rfind('{"clips":')
    json_payload = result.output[json_start:].replace('\n', '')
    payload: dict[str, Any] = json.loads(json_payload)
    audio_json = _expect_mapping(payload["audio_alignment"])
    ref_label = audio_json["reference_stream"].split("->", 1)[0]
    assert ref_label in {"Clip A", "Reference"}
    tgt_map = _expect_mapping(audio_json["target_stream"])
    assert "Clip B" in tgt_map or "Target" in tgt_map
    tgt_descriptor = tgt_map.get("Clip B") or tgt_map.get("Target")
    assert isinstance(tgt_descriptor, str) and tgt_descriptor.startswith("aac/")
    offsets_sec_map = _expect_mapping(audio_json["offsets_sec"])
    offsets_frames_map = _expect_mapping(audio_json["offsets_frames"])
    clip_key = "Clip B" if "Clip B" in offsets_sec_map else "Target"
    assert offsets_sec_map[clip_key] == pytest.approx(0.1)
    assert offsets_frames_map[clip_key] == 2
    suggested_frames_map = _expect_mapping(audio_json.get("suggested_frames", {}))
    assert suggested_frames_map.get(target_path.name) == 2
    assert audio_json["preview_paths"] == []
    assert audio_json["confirmed"] is True
    offset_lines = audio_json.get("offset_lines")
    assert isinstance(offset_lines, list) and offset_lines, "Expected offset_lines for cached alignment reuse"
    assert any("Clip B" in line for line in offset_lines)
    assert not any("missing fps" in str(line).lower() for line in offset_lines)
    offset_lines_text = audio_json.get("offset_lines_text")
    assert isinstance(offset_lines_text, str) and "Clip B" in offset_lines_text
    stream_lines = audio_json.get("stream_lines")
    assert isinstance(stream_lines, list) and stream_lines, "Expected stream_lines entries"
    assert any("ref=" in line for line in stream_lines)
    assert any("target=" in line for line in stream_lines)
    measurements_obj = audio_json.get("measurements") or {}
    measurements_map = _expect_mapping(measurements_obj)
    measurement_value = measurements_map.get(clip_key)
    assert measurement_value is not None
    measurement_entry = _expect_mapping(measurement_value)
    stream_value = measurement_entry.get("stream")
    assert isinstance(stream_value, str)
    assert stream_value.startswith("aac/")
    seconds_value = measurement_entry.get("seconds")
    assert isinstance(seconds_value, (int, float))
    assert pytest.approx(seconds_value) == 0.1
    frames_value = measurement_entry.get("frames")
    assert isinstance(frames_value, (int, float))
    assert frames_value == 2
    correlation_value = measurement_entry.get("correlation")
    assert isinstance(correlation_value, (int, float))
    assert pytest.approx(correlation_value) == 0.93
    status_value = measurement_entry.get("status")
    assert status_value == "auto"
    applied_value = measurement_entry.get("applied")
    assert applied_value is True
    tonemap_json = payload["tonemap"]
    assert tonemap_json["overlay_mode"] == "diagnostic"
    assert tonemap_json["smoothing_period"] == pytest.approx(45.0)
    assert tonemap_json["scene_threshold_low"] == pytest.approx(0.8)
    assert tonemap_json["scene_threshold_high"] == pytest.approx(2.4)
    assert tonemap_json["percentile"] == pytest.approx(99.995)
    assert tonemap_json["contrast_recovery"] == pytest.approx(0.3)
    assert "metadata_label" in tonemap_json
    assert "use_dovi_label" in tonemap_json

    clips_payload = payload.get("clips")
    assert isinstance(clips_payload, list) and len(clips_payload) >= 2
    clip_map = {entry["label"]: entry for entry in clips_payload if isinstance(entry, dict)}
    final_trim_by_file: dict[str, int] = {}
    for file_name, trim_value in init_calls:
        final_trim_by_file[file_name] = trim_value
    label_to_file = {"Clip A": reference_path.name, "Clip B": target_path.name}
    for label, file_name in label_to_file.items():
        assert file_name in final_trim_by_file, f"Missing trim entry for {file_name}"
        trim_value = final_trim_by_file[file_name]
        expected_frames = max(24000 - trim_value, 0) if trim_value >= 0 else 24000 + abs(trim_value)
        assert clip_map[label]["frames"] == expected_frames

def test_audio_alignment_default_duration_avoids_zero_window(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    cli_runner_env: _CliRunnerEnv,
) -> None:
    """
    Verifies that leaving audio alignment duration unspecified does not pass a zero-length window to the measurement routine.

    Configures audio alignment with start_seconds and duration_seconds set to None, runs the CLI, and asserts that the call to the alignment measurement does not include a `duration_seconds` value of zero (i.e., it remains `None`).
    """
    reference_path = cli_runner_env.media_root / "ClipA.mkv"
    target_path = cli_runner_env.media_root / "ClipB.mkv"
    for file in (reference_path, target_path):
        file.write_bytes(b"data")

    cfg = _make_config(cli_runner_env.media_root)
    cfg.audio_alignment.enable = True
    cfg.audio_alignment.max_offset_seconds = 5.0
    cfg.audio_alignment.start_seconds = None
    cfg.audio_alignment.duration_seconds = None

    cli_runner_env.reinstall(cfg)

    def fake_parse(name: str, **_kwargs: object) -> dict[str, str]:
        """
        Create a minimal fake parse result for a clip name.

        Parameters:
            name (str): Clip identifier or filename used to derive the returned label. Additional keyword arguments are ignored.

        Returns:
            dict: Mapping with keys:
                - "label" (str): "Clip A" if `name` starts with "ClipA", otherwise "Clip B".
                - "file_name" (str): The original `name` value.
        """
        if name.startswith("ClipA"):
            return {"label": "Clip A", "file_name": name}
        return {"label": "Clip B", "file_name": name}

    _patch_core_helper(monkeypatch, "parse_filename_metadata", fake_parse)
    _patch_vs_core(monkeypatch, "set_ram_limit", lambda limit: None)

    def fake_init_clip(
        path: str | Path,
        *,
        trim_start: int = 0,
        trim_end: int | None = None,
        fps_map: tuple[int, int] | None = None,
        cache_dir: str | Path | None = None,
        **_kwargs: object,
    ) -> types.SimpleNamespace:
        """
        Create a lightweight fake clip object for tests that resembles the real clip interface.

        Parameters:
            path: Path-like or str specifying the clip file path.
            trim_start (int): Ignored in this fake; present for compatibility with callers.
            trim_end (int | None): Ignored in this fake; present for compatibility with callers.
            fps_map: Ignored in this fake; present for compatibility with callers.
            cache_dir: Ignored in this fake; present for compatibility with callers.

        Returns:
            SimpleNamespace: An object with attributes:
                - path (Path): Resolved Path of the provided `path`.
                - width (int): Horizontal resolution (1920).
                - height (int): Vertical resolution (1080).
                - fps_num (int): Frame rate numerator (24000).
                - fps_den (int): Frame rate denominator (1001).
                - num_frames (int): Total frame count (24000).
        """
        return types.SimpleNamespace(
            path=Path(path),
            width=1920,
            height=1080,
            fps_num=24000,
            fps_den=1001,
            num_frames=24000,
        )

    _patch_vs_core(monkeypatch, "init_clip", fake_init_clip)

    _patch_runner_module(monkeypatch, "select_frames", lambda *args, **kwargs: [42])

    def fake_generate(
        clips: list[types.SimpleNamespace],
        frames: list[int],
        files: list[str],
        metadata: list[dict[str, object]],
        out_dir: Path,
        cfg_screens: ScreenshotConfig,
        color_cfg: ColorConfig,
        **kwargs: object,
    ) -> list[str]:
        """
        Create the output directory and return a single fake screenshot path.

        Returns:
            list[str]: A one-element list containing the string path to "shot.png" inside `out_dir`.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        return [str(out_dir / "shot.png")]

    _patch_runner_module(monkeypatch, "generate_screenshots", fake_generate)

    def fake_probe(path: Path) -> list[AudioStreamInfo]:
        """
        Create a fake audio probe result for the given file path.

        Parameters:
            path (Path): File path to probe; compared against the module-level `reference_path` to determine which mock stream to return.

        Returns:
            list[AudioStreamInfo]: A single-item list with a mocked audio stream. If `path == reference_path` the stream has `index=0`, `language='eng'`, and `is_default=True`; otherwise the stream has `index=1`, `language='jpn'`, and `is_default=False`.
        """
        if Path(path) == reference_path:
            return [
                AudioStreamInfo(
                    index=0,
                    language="eng",
                    codec_name="aac",
                    channels=2,
                    channel_layout="stereo",
                    sample_rate=48000,
                    bitrate=192000,
                    is_default=True,
                    is_forced=False,
                )
            ]
        return [
            AudioStreamInfo(
                index=1,
                language="jpn",
                codec_name="aac",
                channels=2,
                channel_layout="stereo",
                sample_rate=48000,
                bitrate=192000,
                is_default=False,
                is_forced=False,
            )
        ]

    _patch_audio_alignment(monkeypatch, "probe_audio_streams", fake_probe)

    measurement = AlignmentMeasurement(
        file=target_path,
        offset_seconds=0.1,
        frames=3,
        correlation=0.9,
        reference_fps=24.0,
        target_fps=24.0,
    )

    captured_kwargs: dict[str, object] = {}

    def fake_measure(*args: object, **kwargs: object) -> list[AlignmentMeasurement]:
        """
        Stub measurement function used in tests.

        Records any keyword arguments into the enclosing `captured_kwargs` mapping and returns a single-element list containing the preconstructed `measurement` object.

        Returns:
            list: A list with the `measurement` object as its only element.
        """
        captured_kwargs.update(kwargs)
        return [measurement]

    _patch_audio_alignment(monkeypatch, "measure_offsets", fake_measure)
    _patch_audio_alignment(monkeypatch, "load_offsets", lambda *_args, **_kwargs: ({}, {}))
    _patch_audio_alignment(
        monkeypatch,
        "update_offsets_file",
        lambda *_args, **_kwargs: ({target_path.name: 3}, {target_path.name: "auto"}),
    )

    result: Result = runner.invoke(frame_compare.main, ["--no-color"], catch_exceptions=False)
    assert result.exit_code == 0
    assert captured_kwargs.get("duration_seconds") is None

def _build_alignment_context(
    tmp_path: Path,
) -> tuple[
    AppConfig,
    list[_ClipPlan],
    _AudioAlignmentSummary,
    _AudioAlignmentDisplayData,
]:
    """
    Builds a minimal audio-alignment test context with example clips, plans, and alignment state.

    Parameters:
        tmp_path (Path): Temporary directory used to create sample reference and target clip files.

    Returns:
        tuple: A 4-tuple containing:
            - cfg: AppConfig with audio alignment enabled.
            - plans: List containing the reference and target ClipPlan objects for the two sample clips.
            - summary: AudioAlignmentSummary prepopulated for the reference clip.
            - display: AudioAlignmentDisplayData initialized with empty/placeholder display values.
    """
    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True

    reference_clip = types.SimpleNamespace(
        width=1920,
        height=1080,
        fps_num=24000,
        fps_den=1001,
        num_frames=10,
    )
    target_clip = types.SimpleNamespace(
        width=1920,
        height=1080,
        fps_num=24000,
        fps_den=1001,
        num_frames=10,
    )

    reference_path = tmp_path / "Ref.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.touch()
    target_path.touch()

    reference_plan = _ClipPlan(
        path=reference_path,
        metadata={"label": "Reference Clip"},
        clip=reference_clip,
    )
    target_plan = _ClipPlan(
        path=target_path,
        metadata={"label": "Target Clip"},
        clip=target_clip,
    )

    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "alignment.toml",
        reference_name="Reference Clip",
        measurements=(),
        applied_frames={"Reference Clip": 0},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={},
        suggestion_mode=False,
        manual_trim_starts={},
    )

    display = _AudioAlignmentDisplayData(
        stream_lines=[],
        estimation_line=None,
        offset_lines=[],
        offsets_file_line="",
        json_reference_stream=None,
        json_target_streams={},
        json_offsets_sec={},
        json_offsets_frames={},
        warnings=[],
        manual_trim_lines=[],
    )

    return cfg, [reference_plan, target_plan], summary, display

def test_confirm_alignment_reports_preview_paths(tmp_path: Path) -> None:
    cfg, plans, summary, display = _build_alignment_context(tmp_path)

    core_module._confirm_alignment_with_screenshots(
        plans,
        summary,
        cfg,
        tmp_path,
        _RecordingOutputManager(),
        display,
    )

    assert display.preview_paths == []
    assert display.confirmation == "auto"

def test_run_cli_calls_alignment_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: _CliRunnerEnv,
) -> None:
    """
    Verifies that running the CLI triggers the audio-alignment confirmation hook.

    Sets up a configuration enabling audio alignment, creates two dummy media files, and monkeypatches discovery, metadata parsing, plan building, selection, and alignment application. Replaces the confirmation function with one that records its arguments and raises a sentinel exception so the test can assert the confirmation was invoked with the expected parameters.
    """
    cfg = _make_config(cli_runner_env.media_root)
    cfg.audio_alignment.enable = True

    cli_runner_env.reinstall(cfg)

    files: list[Path] = [
        cli_runner_env.media_root / "Ref.mkv",
        cli_runner_env.media_root / "Tgt.mkv",
    ]
    for file in files:
        file.write_bytes(b"data")

    def fake_discover(_root: Path) -> list[Path]:
        """
        Return a precomputed list of discovered files; the provided `_root` argument is ignored.

        Returns:
            files (list): The predefined list of discovered file paths.
        """
        return files

    def fake_parse_metadata(_files: Sequence[Path], _naming: object) -> list[dict[str, str]]:
        """
        Produce metadata for a reference/target pair using the first two entries of the provided files.

        Parameters:
            _files (Sequence[pathlib.Path|os.PathLike|object]): Iterable where the first two items represent the reference and target files; only their `.name` is used.
            _naming (any): Unused naming parameter kept for signature compatibility.

        Returns:
            list[dict]: Two dictionaries with keys `label`, `file_name`, `year`, `title`, `anime_title`, `imdb_id`, and `tvdb_id`. The `label` values are `"Reference"` and `"Target"`, `file_name` is taken from the corresponding file's `.name`, and the remaining fields are empty strings.
        """
        return [
            {
                "label": "Reference",
                "file_name": files[0].name,
                "year": "",
                "title": "",
                "anime_title": "",
                "imdb_id": "",
                "tvdb_id": "",
            },
            {
                "label": "Target",
                "file_name": files[1].name,
                "year": "",
                "title": "",
                "anime_title": "",
                "imdb_id": "",
                "tvdb_id": "",
            },
        ]

    def fake_build_plans(
        _files: Sequence[Path], metadata: Sequence[dict[str, str]], _cfg: AppConfig
    ) -> list[_ClipPlan]:
        """
        Builds a list of clip plans from input file paths and corresponding metadata, marking the first clip as the reference.

        Parameters:
            _files (Sequence[Path]): Input file paths in the order they should be planned.
            metadata (Sequence): Per-file metadata objects; must have the same length as `_files`.
            _cfg: Configuration object (not used by this fake builder, accepted for signature compatibility).

        Returns:
            list[_ClipPlan]: A list of ClipPlan objects where the first element has `use_as_reference=True` and all others have `use_as_reference=False`.
        """
        plans: list[_ClipPlan] = []
        for idx, path in enumerate(_files):
            plans.append(
                _ClipPlan(
                    path=path,
                    metadata=metadata[idx],
                    use_as_reference=(idx == 0),
                )
            )
        return plans

    def fake_pick_analyze(
        _files: Sequence[Path],
        _metadata: Sequence[object],
        _analyze_clip: object,
        cache_dir: Path | None = None,
    ) -> Path:
        """
        Select the first candidate file for analysis.

        Parameters:
            _files: Sequence of candidate file paths; the first element is selected.
            _metadata: Ignored.
            _analyze_clip: Ignored.
            cache_dir: Ignored.

        Returns:
            The first file from `_files`.
        """
        return files[0]

    offsets_path = cli_runner_env.workspace_root / "alignment.toml"

    def fake_maybe_apply(
        plans: Sequence[_ClipPlan],
        _cfg: AppConfig,
        _analyze_path: Path,
        _root: Path,
        _overrides: object,
        reporter: object | None = None,
    ) -> tuple[_AudioAlignmentSummary, _AudioAlignmentDisplayData]:
        """
        Create and return a synthetic audio-alignment summary and display objects for testing.

        Parameters:
            plans (Sequence): Sequence of clip plan objects; the first plan is used as the reference.
            reporter (optional): Ignored; present for API compatibility.

        Returns:
            tuple: A pair (summary, display) where:
                - summary: a _AudioAlignmentSummary with the first plan as the reference_plan,
                  empty measurements/applied_frames/statuses, and baseline_shift 0.
                - display: a _AudioAlignmentDisplayData containing empty display lines,
                  an offsets file line referencing the module's offsets path, and JSON-ready fields
                  for a single target with zero offset (0.0 seconds, 0 frames).
        """
        summary = _AudioAlignmentSummary(
            offsets_path=offsets_path,
            reference_name="Reference",
            measurements=(),
            applied_frames={},
            baseline_shift=0,
            statuses={},
            reference_plan=plans[0],
            final_adjustments={},
            swap_details={},
            suggested_frames={},
            suggestion_mode=False,
            manual_trim_starts={},
        )
        display = _AudioAlignmentDisplayData(
            stream_lines=[],
            estimation_line=None,
            offset_lines=[],
            offsets_file_line=f"Offsets file: {offsets_path}",
            json_reference_stream="ref",
            json_target_streams={"Target": "tgt"},
            json_offsets_sec={"Target": 0.0},
            json_offsets_frames={"Target": 0},
            warnings=[],
            manual_trim_lines=[],
        )
        return summary, display

    class _DummyReporter:
        def __init__(self, *_, **__):
            """
            Create a no-op progress context used to mock progress handling in tests.

            This initializer accepts and ignores any positional or keyword arguments and configures a
            `console` attribute whose `print` method is a no-op to suppress output during tests.
            """
            self.console = types.SimpleNamespace(print=lambda *args, **kwargs: None)
            self.flags: dict[str, object] = {}
            self.values: dict[str, object] = {}

        def update_values(self, *_args, **_kwargs):
            """
            No-op progress update method used to satisfy a progress interface.

            Accepts arbitrary positional and keyword arguments and performs no action.
            """
            return None

        def set_flag(self, *_args, **_kwargs):
            """
            No-op method that accepts any positional and keyword arguments and does nothing.

            Used as a compatibility stub where a flag-setting method is required but no action is desired.
            """
            return None

        def line(self, *_args, **_kwargs):
            """
            Accepts any positional and keyword arguments and performs no action.

            Used as a no-op placeholder to satisfy progress-reporting interfaces.
            """
            return None

        def verbose_line(self, *_args, **_kwargs):
            """
            A no-op placeholder that accepts any positional or keyword arguments and does nothing.

            This method intentionally ignores all inputs and always returns None; it can be used where a verbose callback is optional or not required.
            """
            return None

        def render_sections(self, *_args, **_kwargs):
            """
            No-op renderer for section content; accepts any positional and keyword arguments and performs no action.

            Parameters:
                *_args: Arbitrary positional arguments that are ignored.
                **_kwargs: Arbitrary keyword arguments that are ignored.

            Returns:
                None: Always returns None.
            """
            return None

        def update_progress_state(self, *_args, **_kwargs):
            """
            No-op progress update method that accepts any arguments and has no effect.

            Used as a placeholder in contexts where progress updates are optional; accepts arbitrary positional
            and keyword arguments and performs no action.
            """
            return None

        def set_status(self, *_args, **_kwargs):
            """
            No-op status handler that ignores all arguments.

            This method accepts any positional and keyword arguments and intentionally performs no action.
            """
            return None

        def create_progress(self, *_args, **_kwargs):
            """
            Create a no-op progress context manager.

            Parameters:
                *_args, **_kwargs: Ignored positional and keyword arguments kept for API compatibility.

            Returns:
                DummyProgress: A progress-like object that performs no operations and can be used as a context manager.
            """
            return DummyProgress()

    class _SentinelError(Exception):
        pass

    called: dict[str, object] = {}

    def fake_confirm(
        plans: Sequence[_ClipPlan],
        summary: _AudioAlignmentSummary,
        cfg_obj: AppConfig,
        root: Path,
        reporter: object,
        display: _AudioAlignmentDisplayData,
    ) -> None:
        """
        Test helper that records its invocation arguments and then raises a sentinel error.

        Parameters:
            plans: The clip plans passed to the confirmation function.
            summary: The summary object produced by analysis or alignment.
            cfg_obj: The application configuration object.
            root: The root path or context used by the caller.
            reporter: The reporter used to emit messages.
            display: The display/preview object provided to the confirmation flow.

        Raises:
            _SentinelError: Always raised to signal that this fake confirmation was invoked.
        """
        called["args"] = (plans, summary, cfg_obj, root, reporter, display)
        raise _SentinelError

    _patch_core_helper(monkeypatch, "_discover_media", fake_discover)
    _patch_core_helper(monkeypatch, "parse_metadata", fake_parse_metadata)
    _patch_core_helper(monkeypatch, "_build_plans", fake_build_plans)
    _patch_core_helper(monkeypatch, "_pick_analyze_file", fake_pick_analyze)
    _patch_core_helper(monkeypatch, "_maybe_apply_audio_alignment", fake_maybe_apply)
    _patch_runner_module(monkeypatch, "CliOutputManager", _DummyReporter)
    _patch_core_helper(monkeypatch, "_confirm_alignment_with_screenshots", fake_confirm)
    _patch_vs_core(monkeypatch, "configure", lambda *args, **kwargs: None)

    with pytest.raises(_SentinelError):
        frame_compare.run_cli(None, None)

    assert "args" in called


def test_apply_audio_alignment_derives_frames_from_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seconds-only measurements should still populate suggested frames using clip FPS."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.enable = True
    cfg.audio_alignment.use_vspreview = False
    cfg.audio_alignment.max_offset_seconds = 100.0

    reference = _ClipPlan(path=tmp_path / "Ref.mkv", metadata={"label": "Reference"})
    target = _ClipPlan(path=tmp_path / "Target.mkv", metadata={"label": "Target"})
    for plan in (reference, target):
        plan.path.parent.mkdir(parents=True, exist_ok=True)
        plan.path.write_bytes(b"\x00")
        plan.effective_fps = (24000, 1001)

    plans = [reference, target]
    analyze_path = reference.path
    reporter = _RecordingOutputManager()

    def _fake_probe(_path: Path) -> list[AudioStreamInfo]:
        return [
            AudioStreamInfo(
                index=0,
                language="eng",
                codec_name="aac",
                channels=2,
                channel_layout="stereo",
                sample_rate=48000,
                bitrate=128000,
                is_default=True,
                is_forced=False,
            )
        ]

    monkeypatch.setattr(alignment_core_module.audio_alignment, "probe_audio_streams", _fake_probe)

    offset_seconds = 47.78
    measurement = AlignmentMeasurement(
        file=target.path,
        offset_seconds=offset_seconds,
        frames=None,
        correlation=0.95,
        reference_fps=None,
        target_fps=None,
    )

    monkeypatch.setattr(
        alignment_core_module.audio_alignment,
        "measure_offsets",
        lambda *args, **kwargs: [measurement],
    )

    def _fake_update(
        _path: Path,
        reference_name: str,
        measurements: Any,
        existing: Any = None,
        negative_override_notes: Any = None,
    ) -> tuple[dict[str, int], dict[str, str]]:
        applied = {}
        statuses = {}
        for item in measurements:
            if item.frames is not None:
                applied[item.file.name] = int(item.frames)
            statuses[item.file.name] = "auto"
        return applied, statuses

    monkeypatch.setattr(alignment_core_module.audio_alignment, "update_offsets_file", _fake_update)

    summary, display = alignment_package.apply_audio_alignment(
        plans,
        cfg,
        analyze_path,
        tmp_path,
        audio_track_overrides={},
        reporter=reporter,
    )

    fps_float = alignment_core_module._fps_to_float(target.effective_fps)
    assert fps_float > 0
    expected_frames = int(round(offset_seconds * fps_float))
    assert summary is not None
    assert display is not None
    assert summary.suggested_frames[target.path.name] == expected_frames
    assert any(f"{expected_frames:+d}f" in line for line in display.offset_lines)


def test_plan_fps_map_prioritizes_available_metadata(tmp_path: Path) -> None:
    """_plan_fps_map() should record the first viable FPS tuple per plan."""

    plan_a = _ClipPlan(path=tmp_path / "A.mkv", metadata={})
    plan_b = _ClipPlan(path=tmp_path / "B.mkv", metadata={})
    plan_c = _ClipPlan(path=tmp_path / "C.mkv", metadata={})
    plan_d = _ClipPlan(path=tmp_path / "D.mkv", metadata={})
    plan_e = _ClipPlan(path=tmp_path / "E.mkv", metadata={})
    plan_f = _ClipPlan(path=tmp_path / "F.mkv", metadata={})

    plan_a.effective_fps = (24000, 1001)
    plan_a.source_fps = (1, 0)  # invalid, should be ignored
    plan_b.source_fps = (30000, 1001)
    plan_b.applied_fps = (24, 1)
    plan_c.applied_fps = (25, 1)
    plan_c.fps_override = (26, 1)
    plan_d.fps_override = (27, 1)
    plan_e.fps_override = (48, 0)  # invalid denominator, should be skipped entirely
    plan_f.fps_override = (0, 1000)  # invalid numerator, should be skipped entirely

    fps_map = alignment_core_module._plan_fps_map([plan_a, plan_b, plan_c, plan_d, plan_e, plan_f])
    assert fps_map[plan_a.path] == (24000, 1001)
    assert fps_map[plan_b.path] == (24, 1)
    assert fps_map[plan_c.path] == (25, 1)
    assert fps_map[plan_d.path] == (27, 1)
    assert plan_e.path not in fps_map
    assert plan_f.path not in fps_map
