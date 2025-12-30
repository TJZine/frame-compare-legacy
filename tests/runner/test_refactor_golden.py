
from __future__ import annotations

import types
from pathlib import Path

import pytest
from rich.console import Console

from src.datatypes import AnalysisConfig, ColorConfig, ScreenshotConfig
from src.frame_compare.analysis import CacheLoadResult, FrameMetricsCacheInfo, SelectionDetail
from src.frame_compare.orchestration.coordinator import WorkflowCoordinator
from src.frame_compare.orchestration.state import RunRequest
from tests.helpers.runner_env import (
    _CliRunnerEnv,
    _make_config,
    _patch_core_helper,
    _patch_vs_core,
)


@pytest.fixture
def cli_runner_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _CliRunnerEnv:
    return _CliRunnerEnv(monkeypatch, tmp_path)

def test_workflow_coordinator_golden_master(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: _CliRunnerEnv,
) -> None:
    """
    Golden Master test for WorkflowCoordinator.execute.
    Ensures that the refactored pipeline produces the exact same output as the monolith.
    """
    # 1. Setup Environment
    first = cli_runner_env.media_root / "AAA - 01.mkv"
    second = cli_runner_env.media_root / "BBB - 01.mkv"
    for file in (first, second):
        file.write_bytes(b"data")

    cfg = _make_config(cli_runner_env.media_root)
    # Ensure some complexity
    cfg.overrides.change_fps = {"BBB - 01.mkv": "set"}
    cfg.overrides.trim = {"0": 5}
    cfg.overrides.trim_end = {"BBB - 01.mkv": -12}

    cli_runner_env.reinstall(cfg)

    # 2. Mock Core Helpers
    def fake_parse(name: str, **kwargs: object) -> dict[str, object]:
        if name.startswith("AAA"):
            return {"label": "AAA Short", "release_group": "AAA", "file_name": name}
        return {"label": "BBB Short", "release_group": "BBB", "file_name": name}

    _patch_core_helper(monkeypatch, "parse_filename_metadata", fake_parse)

    # 3. Mock VapourSynth
    _patch_vs_core(monkeypatch, "set_ram_limit", lambda limit_mb: None)
    _patch_vs_core(monkeypatch, "configure", lambda **_: None)

    def fake_init_clip(
        path: str,
        *,
        trim_start: int = 0,
        trim_end: int | None = None,
        fps_map: tuple[int, int] | None = None,
        cache_dir: str | None = None,
        **_kwargs: object,
    ) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            path=path,
            width=1920,
            height=1080,
            fps_num=24000,
            fps_den=1001,
            num_frames=24000,
        )

    _patch_vs_core(monkeypatch, "init_clip", fake_init_clip)
    _patch_vs_core(monkeypatch, "resolve_effective_tonemap", lambda _cfg, **_kwargs: {})

    # 4. Mock Selection
    def fake_select(
        clip: types.SimpleNamespace,
        analysis_cfg: AnalysisConfig,
        files: list[str],
        file_under_analysis: str,
        cache_info: FrameMetricsCacheInfo | None = None,
        progress: object = None,
        *,
        frame_window: tuple[int, int] | None = None,
        return_metadata: bool = False,
        color_cfg: ColorConfig | None = None,
        cache_probe: CacheLoadResult | None = None,
    ) -> list[int] | tuple[list[int], dict[int, str], dict[int, SelectionDetail]]:
        frames = [10, 20]
        categories = {10: "Auto", 20: "Auto"}
        details = {
             10: SelectionDetail(frame_index=10, label="Auto", score=None, source="auto", timecode="00:00:10.000"),
             20: SelectionDetail(frame_index=20, label="Auto", score=None, source="auto", timecode="00:00:20.000"),
        }
        if return_metadata:
            return frames, categories, details
        return frames

    import src.frame_compare.orchestration.phases.analysis as analysis_phase_module
    monkeypatch.setattr(analysis_phase_module, "select_frames", fake_select)

    # 5. Mock Screenshots
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
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for idx in range(len(frames) * len(clips)):
            p = out_dir / f"shot_{idx}.png"
            p.write_text("data", encoding="utf-8")
            paths.append(str(p))
        return paths

    import src.frame_compare.orchestration.phases.render as render_phase_module
    monkeypatch.setattr(render_phase_module, "generate_screenshots", fake_generate)

    # Mock exports
    monkeypatch.setattr(analysis_phase_module, "export_selection_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis_phase_module, "write_selection_cache_file", lambda *args, **kwargs: None)

    import src.frame_compare.orchestration.phases.result as result_phase_module
    monkeypatch.setattr(result_phase_module, "write_snapshot", lambda *args, **kwargs: None)

    # Mock Reporting
    monkeypatch.setattr(result_phase_module, "render_run_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(result_phase_module, "resolve_cli_version", lambda: "0.0.0-test")


    # 6. Execute
    coordinator = WorkflowCoordinator()
    request = RunRequest(
        config_path=str(cli_runner_env.config_path),
        console=Console(force_terminal=False, width=80),
    )
    result = coordinator.execute(request)

    # 7. Assertions (The "Golden" state)

    # Basic Integrity
    assert result.root == cli_runner_env.media_root
    assert len(result.files) == 2
    assert result.frames == [10, 20]
    assert len(result.image_paths) == 4 # 2 frames * 2 clips

    # JSON Tail Structure
    assert result.json_tail is not None
    tail = result.json_tail

    # Clips
    assert len(tail["clips"]) == 2
    assert tail["clips"][0]["label"] == "AAA Short"
    assert tail["clips"][1]["label"] == "BBB Short"

    # Trims
    assert tail["trims"]["per_clip"]["AAA Short"]["lead_f"] == 5
    assert tail["trims"]["per_clip"]["BBB Short"]["trail_f"] == 12

    # Analysis
    assert tail["analysis"]["output_frame_count"] == 2
    assert tail["analysis"]["output_frames"] == [10, 20]

    # Verify Layout Data (Indirectly via what's in json_tail as they are synced in the coordinator)
    # The actual layout_data is harder to inspect on the result object as it is not returned directly,
    # but the RunResult has json_tail which is populated from it.

    # Verify context propagation (implied by success)
    assert result.out_dir == cli_runner_env.media_root / "screens"
