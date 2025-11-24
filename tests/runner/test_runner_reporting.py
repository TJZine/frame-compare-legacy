from pathlib import Path

import pytest

from src.frame_compare import runner as runner_module


def test_runner_reports_dovi_auto_enabled_simple(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that JSON tail reports use_dovi=True when auto-enabled by RPU presence."""
    from src.frame_compare.cli_runtime import ClipPlan
    from src.frame_compare.services.metadata import MetadataResolveResult
    from tests.helpers.runner_env import _make_config, _make_runner_preflight, _patch_core_helper
    from tests.runner.test_overlay_diagnostics import (
        _build_dependencies,
        _install_analysis_stubs,
        _install_render_stubs,
    )

    workspace = tmp_path / "workspace"
    media_root = workspace / "media"
    media_root.mkdir(parents=True)

    cfg = _make_config(media_root)
    cfg.color.use_dovi = None  # Set to AUTO

    preflight = _make_runner_preflight(workspace, media_root, cfg)
    _patch_core_helper(monkeypatch, "prepare_preflight", lambda **_kwargs: preflight)

    # Mock media discovery
    monkeypatch.setattr(runner_module.media_utils, "discover_media", lambda _root: [media_root / "Alpha.mkv", media_root / "Beta.mkv"])

    # Mock metadata result with RPU
    plans = [
        ClipPlan(
            path=media_root / "Alpha.mkv",
            metadata={"label": "Alpha"},
            trim_start=0,
            trim_end=None,
            source_frame_props={"DolbyVisionRPU": b"blob"},
            source_width=1920,
            source_height=1080,
            source_num_frames=100,
        ),
        ClipPlan(
            path=media_root / "Beta.mkv",
            metadata={"label": "Beta"},
            trim_start=0,
            trim_end=None,
            source_frame_props={"_ColorRange": 1},
            source_width=1920,
            source_height=1080,
            source_num_frames=100,
        )
    ]

    metadata_result = MetadataResolveResult(
        plans=plans,
        metadata=[],
        metadata_title="Test",
        analyze_path=plans[0].path,
        slowpics_title_inputs={
            "resolved_base": None,
            "collection_name": None,
            "collection_suffix": "",
        },
        slowpics_final_title="Test",
        slowpics_resolved_base=None,
        slowpics_tmdb_disclosure_line=None,
        slowpics_verbose_tmdb_tag=None,
    )

    deps, _, _ = _build_dependencies(metadata_result)
    _install_analysis_stubs(monkeypatch, score=0.5)
    _install_render_stubs(monkeypatch, out_dir=media_root / "screens")

    request = runner_module.RunRequest(
        config_path=str(preflight.config_path),
        root_override=str(workspace),
        diagnostic_frame_metrics=None,
        reporter=None,
    )

    result = runner_module.run(request, dependencies=deps)

    assert result.json_tail is not None
    tonemap = result.json_tail["tonemap"]

    # Crucial assertions:
    # 1. use_dovi should be True (resolved from auto + RPU)
    assert tonemap["use_dovi"] is True
    # 2. use_dovi_label should be "on"
    assert tonemap["use_dovi_label"] == "on"
