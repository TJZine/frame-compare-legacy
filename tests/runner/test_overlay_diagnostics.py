"""Runner-level regression tests for overlay diagnostics JSON output."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pytest

from src.datatypes import AppConfig
from src.frame_compare import runner as runner_module
from src.frame_compare.analysis import CacheLoadResult, FrameMetricsCacheInfo, SelectionDetail
from src.frame_compare.cli_runtime import CliOutputManager, ClipPlan, SlowpicsTitleInputs
from src.frame_compare.services.alignment import AlignmentRequest, AlignmentResult, AlignmentWorkflow
from src.frame_compare.services.metadata import (
    MetadataResolver,
    MetadataResolveRequest,
    MetadataResolveResult,
)
from src.frame_compare.services.publishers import ReportPublisher, SlowpicsPublisher
from tests.helpers.runner_env import (
    _make_config,
    _make_runner_preflight,
    _patch_core_helper,
    _selection_details_to_json,
)

pytestmark = pytest.mark.usefixtures("runner_vs_core_stub", "dummy_progress")  # type: ignore[attr-defined]


class _StubMetadataResolver:
    def __init__(self, result: MetadataResolveResult) -> None:
        self._result = result
        self.request: MetadataResolveRequest | None = None

    def resolve(self, request: MetadataResolveRequest) -> MetadataResolveResult:
        self.request = request
        return self._result


class _StubAlignmentWorkflow:
    def __init__(
        self,
        *,
        suggested_frames: Optional[int] = None,
        suggested_seconds: Optional[float] = None,
    ) -> None:
        self.request: AlignmentRequest | None = None
        self._suggested_frames = suggested_frames
        self._suggested_seconds = suggested_seconds

    def run(self, request: AlignmentRequest) -> AlignmentResult:
        self.request = request
        if self._suggested_frames is not None:
            request.json_tail["suggested_frames"] = self._suggested_frames
        if self._suggested_seconds is not None:
            request.json_tail["suggested_seconds"] = self._suggested_seconds
        return AlignmentResult(plans=list(request.plans), summary=None, display=None)


class _NullReportPublisher:
    def publish(self, _request: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(report_path=None)


class _NullSlowpicsPublisher:
    def publish(self, _request: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(url=None)


def _build_dependencies(
    result: MetadataResolveResult,
    *,
    alignment_workflow: _StubAlignmentWorkflow | None = None,
) -> tuple[runner_module.RunDependencies, _StubMetadataResolver, _StubAlignmentWorkflow]:
    resolver = _StubMetadataResolver(result)
    workflow = alignment_workflow or _StubAlignmentWorkflow()
    deps = runner_module.RunDependencies(
        metadata_resolver=cast(MetadataResolver, resolver),
        alignment_workflow=cast(AlignmentWorkflow, workflow),
        report_publisher=cast(ReportPublisher, _NullReportPublisher()),
        slowpics_publisher=cast(SlowpicsPublisher, _NullSlowpicsPublisher()),
    )
    return deps, resolver, workflow


def _seed_media(root: Path) -> List[Path]:
    files = [root / "Alpha.mkv", root / "Beta.mkv"]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"0")
    return files


def _make_plans(files: List[Path]) -> List[ClipPlan]:
    props = {
        "MasteringDisplayLuminance": "0.005 1000",
        "ContentLightLevelMax": "850",
        "ContentLightLevelFall": "350",
        "DolbyVision_Block_Index": 3,
        "DolbyVision_Block_Total": 7,
        "DolbyVision_Target_Nits": "700",
        "DolbyVision_L1_Average": 0.12,
        "DolbyVision_L1_Maximum": 0.52,
        "_colorrange": 0,
    }
    plans = [
        ClipPlan(
            path=files[0],
            metadata={"label": "Alpha"},
            trim_start=0,
            trim_end=None,
            use_as_reference=True,
            source_frame_props=dict(props),
            source_width=3840,
            source_height=2160,
            source_num_frames=120,
        ),
        ClipPlan(
            path=files[1],
            metadata={"label": "Beta"},
            trim_start=0,
            trim_end=None,
            source_frame_props={"_ColorRange": 1},
            source_width=3840,
            source_height=2160,
            source_num_frames=120,
        ),
    ]
    return plans


def _make_metadata_result(files: List[Path]) -> MetadataResolveResult:
    plans = _make_plans(files)
    title_inputs: SlowpicsTitleInputs = {
        "resolved_base": None,
        "collection_name": None,
        "collection_suffix": "",
    }
    return MetadataResolveResult(
        plans=list(plans),
        metadata=[{"label": "Alpha"}, {"label": "Beta"}],
        metadata_title="Alpha vs Beta",
        analyze_path=files[0],
        slowpics_title_inputs=title_inputs,
        slowpics_final_title="Alpha vs Beta",
        slowpics_resolved_base=None,
        slowpics_tmdb_disclosure_line=None,
        slowpics_verbose_tmdb_tag=None,
    )


def _install_analysis_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    score: float | None,
) -> None:
    def fake_select(*_args: object, **_kwargs: object):
        details = {
            10: SelectionDetail(
                frame_index=10,
                label="Auto",
                score=score,
                source="auto",
                timecode="00:00:10.000",
                notes=None,
            )
        }
        return [10], {10: "Auto"}, details

    import src.frame_compare.orchestration.coordinator as coordinator_module

    monkeypatch.setattr(coordinator_module, "select_frames", fake_select)
    monkeypatch.setattr(coordinator_module, "selection_details_to_json", _selection_details_to_json)
    monkeypatch.setattr(
        coordinator_module,
        "probe_cached_metrics",
        lambda *_args, **_kwargs: CacheLoadResult(metrics=None, status="missing", reason=None),
    )
    monkeypatch.setattr(coordinator_module, "selection_hash_for_config", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(coordinator_module, "export_selection_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(coordinator_module, "write_selection_cache_file", lambda *_args, **_kwargs: None)


def _install_render_stubs(monkeypatch: pytest.MonkeyPatch, out_dir: Path) -> None:
    def fake_generate(
        clips: List[object],
        frames: List[int],
        files_for_run: List[Path],
        metadata_list: List[Dict[str, Any]],
        out_dir_param: Path,
        cfg_screens: object,
        color_cfg: object,
        **_kwargs: object,
    ) -> List[str]:
        out_dir_param.mkdir(parents=True, exist_ok=True)
        shot = out_dir_param / "shot.png"
        shot.write_text("img", encoding="utf-8")
        return [str(shot)]

    def fake_init_clips(plans: List[ClipPlan], *_args: object, **_kwargs: object) -> None:
        for plan in plans:
            plan.clip = types.SimpleNamespace(width=plan.source_width, height=plan.source_height, num_frames=plan.source_num_frames or 120)

    import src.frame_compare.cache as cache_utils
    import src.frame_compare.orchestration.coordinator as coordinator_module
    import src.frame_compare.selection as selection_utils
    from src.frame_compare import vs as vs_core

    monkeypatch.setattr(coordinator_module, "generate_screenshots", fake_generate)
    monkeypatch.setattr(selection_utils, "init_clips", fake_init_clips)
    monkeypatch.setattr(
        cache_utils,
        "build_cache_info",
        lambda *_args, **_kwargs: FrameMetricsCacheInfo(
            path=out_dir / "generated.compframes",
            files=["Alpha.mkv", "Beta.mkv"],
            analyzed_file="Alpha.mkv",
            release_group="grp",
            trim_start=0,
            trim_end=None,
            fps_num=24000,
            fps_den=1001,
        ),
    )
    monkeypatch.setattr(
        vs_core,
        "resolve_effective_tonemap",
        lambda color_cfg, props=None: {
            "preset": getattr(color_cfg, "preset", "reference"),
            "tone_curve": getattr(color_cfg, "tone_curve", "bt.2390"),
            "target_nits": getattr(color_cfg, "target_nits", 100.0),
            "metadata": getattr(color_cfg, "metadata", "auto"),
            "use_dovi": (
                True
                if getattr(color_cfg, "use_dovi", None) is None
                and props
                and any(k in props for k in ("DolbyVisionRPU", "_DolbyVisionRPU"))
                else getattr(color_cfg, "use_dovi", None)
            ),
        },
    )


def _prepare_config(workspace: Path, media_root: Path, *, per_frame_nits: bool) -> tuple[AppConfig, Path]:
    cfg = _make_config(media_root)
    cfg.analysis.frame_count_dark = 0
    cfg.analysis.frame_count_bright = 0
    cfg.analysis.frame_count_motion = 0
    cfg.analysis.random_frames = 0
    cfg.analysis.save_frames_data = False
    cfg.color.overlay_mode = "diagnostic"
    cfg.color.overlay_enabled = True
    cfg.color.target_nits = 900.0
    cfg.color.use_dovi = True
    cfg.diagnostics.per_frame_nits = per_frame_nits
    cfg.slowpics.auto_upload = False
    cfg.report.enable = False
    preflight = _make_runner_preflight(workspace, media_root, cfg)
    return cfg, preflight.config_path


def _common_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    per_frame_nits: bool,
    cli_override: bool | None,
    alignment_workflow: _StubAlignmentWorkflow | None = None,
    reporter: CliOutputManager | None = None,
) -> tuple[runner_module.RunRequest, runner_module.RunDependencies, MetadataResolveResult]:
    workspace = tmp_path / "workspace"
    media_root = workspace / "media"
    files = _seed_media(media_root)
    cfg, config_path = _prepare_config(workspace, media_root, per_frame_nits=per_frame_nits)
    preflight = _make_runner_preflight(workspace, media_root, cfg)
    _patch_core_helper(monkeypatch, "prepare_preflight", lambda **_kwargs: preflight)
    import src.frame_compare.media as media_utils
    monkeypatch.setattr(media_utils, "discover_media", lambda _root: list(files))
    metadata_result = _make_metadata_result(files)
    deps, _, _ = _build_dependencies(metadata_result, alignment_workflow=alignment_workflow)
    _install_analysis_stubs(monkeypatch, score=0.6)
    _install_render_stubs(monkeypatch, out_dir=media_root / cfg.screenshots.directory_name)

    request = runner_module.RunRequest(
        config_path=str(config_path),
        root_override=str(workspace),
        diagnostic_frame_metrics=cli_override,
        reporter=reporter,
    )
    return request, deps, metadata_result


def test_runner_emits_overlay_diagnostics_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, deps, _ = _common_setup(tmp_path, monkeypatch, per_frame_nits=True, cli_override=None)

    result = runner_module.run(request, dependencies=deps)

    assert result.json_tail is not None
    tail_overlay = cast(Dict[str, Any], result.json_tail["overlay"])
    overlay_diag = cast(Dict[str, Any], tail_overlay["diagnostics"])
    dv_block = cast(Dict[str, Any], overlay_diag["dv"])
    assert dv_block["enabled"] is True
    assert dv_block["label"] == "on"
    assert dv_block["metadata_present"] is True
    assert dv_block["has_l1_stats"] is True
    assert dv_block["l2_summary"]["block_index"] == 3
    assert dv_block["l2_summary"]["l1_average"] == pytest.approx(0.12)
    assert dv_block["l2_summary"]["l1_maximum"] == pytest.approx(0.52)
    hdr_block = cast(Dict[str, Any], overlay_diag["hdr"])
    assert hdr_block["max_cll"] == 850.0
    assert hdr_block["max_fall"] == 350.0
    range_block = cast(Dict[str, Any], overlay_diag["dynamic_range"])
    assert range_block["label"] == "full"
    frame_metrics = cast(Dict[str, Any], overlay_diag["frame_metrics"])
    assert frame_metrics["enabled"] is True
    assert frame_metrics["gating"]["config"] is True
    assert frame_metrics["gating"]["cli_override"] is None
    per_frame = cast(Dict[str, Any], frame_metrics["per_frame"])
    assert per_frame["10"]["category"] == "Auto"
    assert per_frame["10"]["avg_nits"] == pytest.approx(540.0)
    clips_block = cast(List[Dict[str, Any]], result.json_tail["clips"])
    clip_entry = clips_block[0]
    assert clip_entry["hdr_metadata"]["max_cll"] == 850.0
    assert clip_entry["dynamic_range"]["label"] == "full"


def test_runner_marks_missing_dv_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, deps, metadata_result = _common_setup(tmp_path, monkeypatch, per_frame_nits=True, cli_override=None)
    metadata_result.plans[0].source_frame_props = {}

    result = runner_module.run(request, dependencies=deps)

    assert result.json_tail is not None
    tail_overlay = cast(Dict[str, Any], result.json_tail["overlay"])
    overlay_diag = cast(Dict[str, Any], tail_overlay["diagnostics"])
    dv_block = cast(Dict[str, Any], overlay_diag["dv"])
    assert dv_block["metadata_present"] is False
    assert dv_block["has_l1_stats"] is False
    assert "l2_summary" not in dv_block


def test_cli_override_disables_frame_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, deps, _ = _common_setup(tmp_path, monkeypatch, per_frame_nits=True, cli_override=False)

    result = runner_module.run(request, dependencies=deps)

    assert result.json_tail is not None
    tail_overlay = cast(Dict[str, Any], result.json_tail["overlay"])
    overlay_diag = cast(Dict[str, Any], tail_overlay["diagnostics"])
    frame_metrics = cast(Dict[str, Any], overlay_diag["frame_metrics"])
    assert frame_metrics["enabled"] is False
    assert cast(Dict[str, Any], frame_metrics["per_frame"]) == {}
    assert frame_metrics["gating"]["cli_override"] is False


def test_runner_layout_preserves_vspreview_suggestions_from_json_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recording_output_manager: CliOutputManager,
) -> None:
    workflow = _StubAlignmentWorkflow(suggested_frames=5, suggested_seconds=0.25)
    request, deps, _ = _common_setup(
        tmp_path,
        monkeypatch,
        per_frame_nits=False,
        cli_override=None,
        alignment_workflow=workflow,
        reporter=recording_output_manager,
    )

    result = runner_module.run(request, dependencies=deps)

    assert result.json_tail is not None
    assert result.json_tail["suggested_frames"] == 5
    assert result.json_tail["suggested_seconds"] == pytest.approx(0.25, rel=1e-6)
    vspreview_layout = cast(Dict[str, Any], recording_output_manager.values.get("vspreview"))
    assert vspreview_layout["suggested_frames"] == 5
    assert vspreview_layout["suggested_seconds"] == pytest.approx(0.25, rel=1e-6)
