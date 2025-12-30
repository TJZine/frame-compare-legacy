import json
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import src.frame_compare.analysis as analysis_mod
import src.frame_compare.analysis.cache_io as cache_io
import src.frame_compare.cache as cache_module
from src.datatypes import AnalysisConfig, ColorConfig
from src.frame_compare.analysis import (
    FrameMetricsCacheInfo,
    SelectionDetail,
    _quantile,
    compute_selection_window,
    dedupe,
    probe_cached_metrics,
    select_frames,
    selection_details_to_json,
    selection_hash_for_config,
    write_selection_cache_file,
)
from src.frame_compare.analysis.cache_io import ClipIdentity
from src.frame_compare.cli_runtime import ClipPlan
from tests.helpers.runner_env import _make_config

pytestmark = pytest.mark.usefixtures("mock_vapoursynth")  # type: ignore[reportAttributeAccessIssue,reportUnknownMemberType]


class MockVideoNode:
    def __init__(self) -> None:
        self.props = {}

@pytest.fixture
def mock_vapoursynth(monkeypatch: pytest.MonkeyPatch):
    class MockVS:
        VideoNode = MockVideoNode
        RGB24 = 0
        YUV444P16 = 0
        core = types.SimpleNamespace(
            resize=types.SimpleNamespace(Point=lambda clip, *a,**k: clip),
            std=types.SimpleNamespace(SetFrameProps=lambda clip, *a,**k: clip, Levels=lambda clip, *a,**k: clip)
        )
    monkeypatch.setitem(sys.modules, 'vapoursynth', MockVS())
    monkeypatch.setitem(sys.modules, 'vs', MockVS())


class FakeClip(MockVideoNode):
    def __init__(self, num_frames: int, brightness: Sequence[float], motion: Sequence[float]) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.fps_num = 24
        self.fps_den = 1
        self.analysis_brightness: Sequence[float] = brightness
        self.analysis_motion: Sequence[float] = motion
        self.frame_props: dict[str, int] | None = None

    def get_frame(self, idx: int) -> Any:
        return types.SimpleNamespace(props=self.frame_props or {})


@pytest.mark.usefixtures("mock_vapoursynth")  # type: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
def _select_frames_list(
    clip: FakeClip,
    cfg: AnalysisConfig,
    files: list[str],
    file_under_analysis: str,
    cache_info: FrameMetricsCacheInfo | None = None,
    progress: Callable[[int], None] | None = None,
    *,
    frame_window: tuple[int, int] | None = None,
    return_metadata: bool = False,
    color_cfg: ColorConfig | None = None,
) -> list[int]:
    result = select_frames(
        clip,
        cfg,
        files,
        file_under_analysis,
        cache_info,
        progress,
        frame_window=frame_window,
        return_metadata=return_metadata,
        color_cfg=color_cfg,
    )
    if isinstance(result, tuple):
        return list(result[0])
    return cast(list[int], result)


@pytest.mark.usefixtures("mock_vapoursynth")  # type: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
def _select_frames_with_metadata(
    clip: FakeClip,
    cfg: AnalysisConfig,
    files: list[str],
    file_under_analysis: str,
    cache_info: FrameMetricsCacheInfo | None = None,
    progress: Callable[[int], None] | None = None,
    *,
    frame_window: tuple[int, int] | None = None,
    return_metadata: bool = False,
    color_cfg: ColorConfig | None = None,
):
    result = select_frames(
        clip,
        cfg,
        files,
        file_under_analysis,
        cache_info,
        progress,
        frame_window=frame_window,
        return_metadata=True,
        color_cfg=color_cfg,
    )
    assert isinstance(result, tuple) and len(result) == 3
    return result


def _seed_cached_metrics(tmp_path: Path) -> tuple[FrameMetricsCacheInfo, AnalysisConfig, list[int]]:
    cache_path = tmp_path / "metrics.json"
    cfg = AnalysisConfig(
        frame_count_dark=1,
        frame_count_bright=1,
        frame_count_motion=1,
        random_frames=0,
        user_frames=[],
        downscale_height=0,
        step=1,
        analyze_in_sdr=False,
    )
    cache_info = FrameMetricsCacheInfo(
        path=cache_path,
        files=["sample.mkv"],
        analyzed_file="sample.mkv",
        release_group="",
        trim_start=0,
        trim_end=None,
        fps_num=24,
        fps_den=1,
    )
    brightness = [(idx, float(idx) / 10.0) for idx in range(10)]
    motion = [(idx, float(idx) / 5.0) for idx in range(10)]
    selection_frames = [0, 5, 9]
    selection_hash = selection_hash_for_config(cfg)
    selection_details = {
        frame: SelectionDetail(
            frame_index=frame,
            label="Auto",
            score=None,
            source="unit",
            timecode=None,
        )
        for frame in selection_frames
    }
    analysis_mod._save_cached_metrics(
        cache_info,
        cfg,
        brightness,
        motion,
        selection_hash=selection_hash,
        selection_frames=selection_frames,
        selection_categories={frame: "Auto" for frame in selection_frames},
        selection_details=selection_details,
    )
    return cache_info, cfg, selection_frames


def _make_cache_info(tmp_path: Path, file_name: str) -> FrameMetricsCacheInfo:
    return FrameMetricsCacheInfo(
        path=tmp_path / "metrics.json",
        files=[file_name],
        analyzed_file=file_name,
        release_group="",
        trim_start=0,
        trim_end=None,
        fps_num=24,
        fps_den=1,
    )


def test_build_clip_inputs_does_not_hash_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_name = "clip.mkv"
    clip_path = tmp_path / file_name
    clip_path.write_bytes(b"data")
    info = _make_cache_info(tmp_path, file_name)

    def _fail_compute(_path: Path, *, chunk_size: int = 1024 * 1024) -> str:
        raise AssertionError("hash computation should not run on the fast path")

    monkeypatch.setattr(cache_io, "_compute_file_sha1", _fail_compute)
    clip_inputs = cache_io._build_clip_inputs(info)
    assert clip_inputs[0]["sha1"] is None


def test_build_clip_inputs_hashes_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_name = "clip.mkv"
    clip_path = tmp_path / file_name
    clip_path.write_bytes(b"data")
    info = _make_cache_info(tmp_path, file_name)

    digest_calls: list[Path] = []

    def _capture_compute(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
        digest_calls.append(path)
        return "abc123"

    monkeypatch.setattr(cache_io, "_compute_file_sha1", _capture_compute)
    monkeypatch.setenv("FRAME_COMPARE_CACHE_HASH", "1")
    clip_inputs = cache_io._build_clip_inputs(info)

    assert clip_inputs[0]["sha1"] == "abc123"
    assert digest_calls == [clip_path]


def test_build_clip_inputs_hashes_when_param_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_name = "clip.mkv"
    clip_path = tmp_path / file_name
    clip_path.write_bytes(b"data")
    info = _make_cache_info(tmp_path, file_name)

    digest_calls: list[Path] = []

    def _capture_compute(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
        digest_calls.append(path)
        return "param-digest"

    monkeypatch.setattr(cache_io, "_compute_file_sha1", _capture_compute)
    monkeypatch.delenv("FRAME_COMPARE_CACHE_HASH", raising=False)

    clip_inputs = cache_io._build_clip_inputs(
        info,
        compute_sha1=True,
        env_opt_in=False,
    )

    assert clip_inputs[0]["sha1"] == "param-digest"
    assert digest_calls == [clip_path]


def test_build_clip_inputs_from_paths_hashes_when_param_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip_path = tmp_path / "clip.mkv"
    clip_path.write_bytes(b"data")

    digest_calls: list[Path] = []

    def _capture_compute(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
        digest_calls.append(path)
        return "paths-digest"

    monkeypatch.setattr(cache_io, "_compute_file_sha1", _capture_compute)
    monkeypatch.setenv("FRAME_COMPARE_CACHE_HASH", "0")

    clip_inputs = cache_io.build_clip_inputs_from_paths(
        analyzed_file=clip_path.name,
        clip_paths=[clip_path],
        compute_sha1=True,
        env_opt_in=False,
    )

    assert clip_inputs[0]["sha1"] == "paths-digest"
    assert digest_calls == [clip_path.resolve()]


def test_compare_clip_identities_requires_sha1_when_missing() -> None:
    exp = ClipIdentity(
        role="ref",
        path="/a",
        name="a",
        size=None,
        mtime=None,
        sha1=None,
    )
    obs = ClipIdentity(
        role="ref",
        path="/a",
        name="a",
        size=None,
        mtime=None,
        sha1=None,
    )
    mismatch = cache_io._compare_clip_identities([exp], [obs], require_sha1=True)
    assert mismatch == "inputs_sha1_mismatch"

    exp2 = replace(exp, sha1="abc123")
    obs2 = replace(obs, sha1=None)
    mismatch = cache_io._compare_clip_identities([exp2], [obs2], require_sha1=True)
    assert mismatch == "inputs_sha1_mismatch"


def test_selection_sidecar_cache_key_stable_without_sha1(tmp_path: Path) -> None:
    file_name = "clip.mkv"
    clip_path = tmp_path / file_name
    clip_path.write_bytes(b"data")
    info = _make_cache_info(tmp_path, file_name)
    cfg = AnalysisConfig(
        frame_count_dark=1,
        frame_count_bright=1,
        frame_count_motion=1,
        random_frames=0,
        user_frames=[],
        downscale_height=0,
        step=1,
        analyze_in_sdr=False,
    )

    clip_inputs_1 = cache_io._build_clip_inputs(info)
    clip_inputs_2 = cache_io._build_clip_inputs(info)
    cache_key_1 = cache_io._selection_cache_key(
        clip_inputs=clip_inputs_1,
        cfg=cfg,
        selection_source="select_frames.v1",
    )
    cache_key_2 = cache_io._selection_cache_key(
        clip_inputs=clip_inputs_2,
        cfg=cfg,
        selection_source="select_frames.v1",
    )

    assert cache_key_1 == cache_key_2



def test_build_cache_info_skips_sha1_when_stat_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_config(tmp_path)
    cfg.analysis.save_frames_data = True
    missing_path = tmp_path / "missing.mkv"
    plan = ClipPlan(path=missing_path, metadata={})

    monkeypatch.setattr(cache_module, "cache_hash_env_requested", lambda: True)

    def _fail_hash(_path: Path) -> str:
        raise AssertionError("compute_file_sha1 should not be called when stat fails")

    monkeypatch.setattr(cache_module, "compute_file_sha1", _fail_hash)

    info = cache_module._build_cache_info(tmp_path, [plan], cfg, analyze_index=0)
    assert info is not None
    assert info.clips is not None
    assert info.clips[0].sha1 is None


def test_build_cache_info_handles_sha1_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_config(tmp_path)
    cfg.analysis.save_frames_data = True
    clip_path = tmp_path / "clip.mkv"
    clip_path.write_bytes(b"data")
    plan = ClipPlan(path=clip_path, metadata={})

    monkeypatch.setattr(cache_module, "cache_hash_env_requested", lambda: True)

    def _raise_hash(_path: Path) -> str:
        raise OSError("hash failure")

    monkeypatch.setattr(cache_module, "compute_file_sha1", _raise_hash)

    info = cache_module._build_cache_info(tmp_path, [plan], cfg, analyze_index=0)
    assert info is not None
    assert info.clips is not None
    assert info.clips[0].sha1 is None


def test_quantile_basic():
    data = [0, 1, 2, 3, 4]
    assert _quantile(data, 0.0) == 0
    assert _quantile(data, 1.0) == 4
    assert _quantile(data, 0.5) == 2
    assert pytest.approx(_quantile(data, 0.25)) == 1.0
    with pytest.raises(ValueError):
        _quantile([], 0.5)


def test_dedupe_separation():
    frames = [0, 10, 20, 30, 100]
    deduped = dedupe(frames, min_separation_sec=1.0, fps=24.0)
    assert deduped == [0, 30, 100]


def test_compute_selection_window_basic():
    spec = compute_selection_window(2400, 24.0, ignore_lead_seconds=10.0, ignore_trail_seconds=5.0, min_window_seconds=1.0)
    assert spec.start_frame == 240
    assert spec.end_frame == 2280
    assert pytest.approx(spec.start_seconds, rel=1e-6) == 10.0
    assert pytest.approx(spec.end_seconds, rel=1e-6) == 95.0
    assert pytest.approx(spec.applied_lead_seconds, rel=1e-6) == 10.0
    assert pytest.approx(spec.applied_trail_seconds, rel=1e-6) == 5.0
    assert not spec.warnings


def test_compute_selection_window_clamps_to_clip():
    spec = compute_selection_window(60, 30.0, ignore_lead_seconds=2.0, ignore_trail_seconds=2.0, min_window_seconds=5.0)
    assert spec.start_frame == 0
    assert spec.end_frame == 60
    assert pytest.approx(spec.start_seconds, rel=1e-6) == 0.0
    assert pytest.approx(spec.end_seconds, rel=1e-6) == 2.0
    assert pytest.approx(spec.applied_lead_seconds, rel=1e-6) == 0.0
    assert pytest.approx(spec.applied_trail_seconds, rel=1e-6) == 0.0
    assert spec.warnings


def test_compute_selection_window_invalid_type():
    with pytest.raises(TypeError) as excinfo:
        bad_value = cast(Any, object())
        compute_selection_window(100, 24.0, ignore_lead_seconds=bad_value, ignore_trail_seconds=0.0, min_window_seconds=0.0)
    assert "analysis.ignore_lead_seconds" in str(excinfo.value)


def test_select_frames_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(
        num_frames=300,
        brightness=[i / 300 for i in range(300)],
        motion=[(300 - i) / 300 for i in range(300)],
    )

    calls: list[str] = []

    def fake_process(
        target_clip: FakeClip,
        file_name: str,
        color_cfg: ColorConfig,
        **kwargs: object,
    ) -> types.SimpleNamespace:
        calls.append(file_name)
        return types.SimpleNamespace(clip=target_clip, overlay_text=None, verification=None)

    monkeypatch.setattr(
        analysis_mod.vs_core,
        "process_clip_for_screenshot",
        fake_process,
    )

    cfg = AnalysisConfig(
        frame_count_dark=3,
        frame_count_bright=3,
        frame_count_motion=2,
        random_frames=0,
        user_frames=[],
        downscale_height=0,
        step=10,
        analyze_in_sdr=True,
    )

    files: list[str] = ["a.mkv", "b.mkv"]
    color_cfg = ColorConfig()
    first = _select_frames_list(clip, cfg, files, file_under_analysis="a.mkv", color_cfg=color_cfg)
    second = _select_frames_list(clip, cfg, files, file_under_analysis="a.mkv", color_cfg=color_cfg)

    assert first == second
    assert sorted(first) == first
    assert len(calls) == 0




def test_select_frames_returns_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(
        num_frames=120,
        brightness=[i / 120 for i in range(120)],
        motion=[(120 - i) / 120 for i in range(120)],
    )

    monkeypatch.setattr(
        analysis_mod.vs_core,
        "process_clip_for_screenshot",
        lambda clip, file_name, color_cfg, **kwargs: types.SimpleNamespace(clip=clip, overlay_text=None, verification=None),
    )

    cfg = AnalysisConfig(
        frame_count_dark=1,
        frame_count_bright=1,
        frame_count_motion=1,
        random_frames=0,
        user_frames=[5],
        analyze_in_sdr=False,
    )
    color_cfg = ColorConfig()

    frames, labels, details = _select_frames_with_metadata(clip, cfg, files=["a.mkv"], file_under_analysis="a.mkv", color_cfg=color_cfg)
    assert frames
    assert labels[frames[0]] in {"Dark", "Bright", "Motion", "User", "Auto"}
    assert frames[0] in details
    detail = details[frames[0]]
    assert detail.label == labels[frames[0]]
    assert detail.frame_index == frames[0]

def test_select_frames_hdr_tonemap(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(
        num_frames=300,
        brightness=[i / 300 for i in range(300)],
        motion=[(300 - i) / 300 for i in range(300)],
    )
    clip.frame_props = {"_Transfer": 16, "_Primaries": 9}

    calls: list[str] = []

    def fake_process(
        target_clip: FakeClip,
        file_name: str,
        color_cfg: ColorConfig,
        **kwargs: object,
    ) -> types.SimpleNamespace:
        calls.append(file_name)
        return types.SimpleNamespace(clip=target_clip, overlay_text=None, verification=None)

    monkeypatch.setattr(
        analysis_mod.vs_core,
        "process_clip_for_screenshot",
        fake_process,
    )

    cfg = AnalysisConfig(
        frame_count_dark=3,
        frame_count_bright=3,
        frame_count_motion=2,
        random_frames=0,
        user_frames=[],
        downscale_height=0,
        step=10,
        analyze_in_sdr=True,
    )

    files: list[str] = ["a.mkv", "b.mkv"]
    color_cfg = ColorConfig()
    first = _select_frames_list(clip, cfg, files, file_under_analysis="a.mkv", color_cfg=color_cfg)
    second = _select_frames_list(clip, cfg, files, file_under_analysis="a.mkv", color_cfg=color_cfg)

    assert first == second
    assert sorted(first) == first
    assert len(calls) == 2


def test_user_and_random_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(
        num_frames=200,
        brightness=[(i % 50) / 50 for i in range(200)],
        motion=[(i % 30) / 30 for i in range(200)],
    )

    monkeypatch.setattr(
        analysis_mod.vs_core,
        "process_clip_for_screenshot",
        lambda clip, file_name, color_cfg, **kwargs: types.SimpleNamespace(clip=clip, overlay_text=None, verification=None),
    )

    cfg = AnalysisConfig(
        frame_count_dark=0,
        frame_count_bright=0,
        frame_count_motion=0,
        random_frames=3,
        user_frames=[5, 10, 150],
        screen_separation_sec=0,
        step=5,
        analyze_in_sdr=False,
    )

    color_cfg = ColorConfig()
    frames = _select_frames_list(clip, cfg, files=["x.mkv"], file_under_analysis="x.mkv", color_cfg=color_cfg)
    assert frames == sorted(frames)
    for user_frame in cfg.user_frames:
        assert user_frame in frames
    extras = [f for f in frames if f not in set(cfg.user_frames)]
    assert len(extras) == cfg.random_frames


def test_select_frames_respects_window(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clip = FakeClip(
        num_frames=220,
        brightness=[i / 220 for i in range(220)],
        motion=[(220 - i) / 220 for i in range(220)],
    )

    monkeypatch.setattr(
        analysis_mod.vs_core,
        "process_clip_for_screenshot",
        lambda clip, file_name, color_cfg, **kwargs: types.SimpleNamespace(clip=clip, overlay_text=None, verification=None),
    )

    cfg = AnalysisConfig(
        frame_count_dark=2,
        frame_count_bright=0,
        frame_count_motion=0,
        random_frames=0,
        user_frames=[10, 75, 180],
        screen_separation_sec=0,
        step=1,
    )

    color_cfg = ColorConfig()
    with caplog.at_level("WARNING"):
        frames = _select_frames_list(
            clip,
            cfg,
            files=["clip.mkv"],
            file_under_analysis="clip.mkv",
            frame_window=(50, 150),
            color_cfg=color_cfg,
        )

    assert frames
    assert all(50 <= frame < 150 for frame in frames)
    assert 75 in frames
    assert 10 not in frames
    assert 180 not in frames
    assert any("Dropped" in message for message in caplog.messages)


def test_select_frames_uses_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = FakeClip(
        num_frames=120,
        brightness=[i / 120 for i in range(120)],
        motion=[(120 - i) / 120 for i in range(120)],
    )

    monkeypatch.setattr(
        analysis_mod.vs_core,
        "process_clip_for_screenshot",
        lambda clip, file_name, color_cfg, **kwargs: types.SimpleNamespace(clip=clip, overlay_text=None, verification=None),
    )

    calls: dict[str, int] = {"count": 0}

    def fake_collect(
        analysis_clip: FakeClip,
        cfg: AnalysisConfig,
        indices: Sequence[int],
        progress: object = None,
        *,
        color_cfg: ColorConfig | None = None,
        file_name: str | None = None,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        calls["count"] += 1
        results = [(idx, float(idx)) for idx in indices]
        return (results, results)

    monkeypatch.setattr(analysis_mod, "_collect_metrics_vapoursynth", fake_collect)

    cfg = AnalysisConfig(
        frame_count_dark=1,
        frame_count_bright=1,
        frame_count_motion=1,
        random_frames=0,
        user_frames=[],
        downscale_height=0,
        analyze_in_sdr=False,
    )

    cache_info = FrameMetricsCacheInfo(
        path=tmp_path / "metrics.json",
        files=["a.mkv"],
        analyzed_file="a.mkv",
        release_group="",
        trim_start=0,
        trim_end=None,
        fps_num=24,
        fps_den=1,
    )

    color_cfg = ColorConfig()
    frames_first = _select_frames_list(clip, cfg, ["a.mkv"], "a.mkv", cache_info=cache_info, color_cfg=color_cfg)
    assert cache_info.path.exists()
    assert calls["count"] == 1

    calls["count"] = 0
    frames_second = _select_frames_list(clip, cfg, ["a.mkv"], "a.mkv", cache_info=cache_info, color_cfg=color_cfg)
    assert calls["count"] == 0
    assert frames_first == frames_second


def test_probe_cached_metrics_detects_config_change(tmp_path: Path) -> None:
    cache_info, cfg, selection_frames = _seed_cached_metrics(tmp_path)

    reused = probe_cached_metrics(cache_info, cfg)
    assert reused.status == "reused"
    assert reused.metrics is not None
    assert reused.metrics.selection_frames == selection_frames

    stale = probe_cached_metrics(cache_info, replace(cfg, frame_count_dark=cfg.frame_count_dark + 1))
    assert stale.status == "stale"
    assert stale.reason == "config_mismatch"


def test_select_frames_uses_cache_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_info, cfg, selection_frames = _seed_cached_metrics(tmp_path)
    cache_probe = probe_cached_metrics(cache_info, cfg)
    assert cache_probe.status == "reused"

    def fail_collect(*args: object, **kwargs: object) -> None:
        raise AssertionError("metrics collection should not run when cache is reused")

    monkeypatch.setattr(analysis_mod, "_collect_metrics_vapoursynth", fail_collect)

    clip = FakeClip(num_frames=60, brightness=[0.1] * 60, motion=[0.2] * 60)
    result = select_frames(
        clip,
        cfg,
        list(cache_info.files),
        cache_info.analyzed_file,
        cache_info=cache_info,
        return_metadata=True,
        color_cfg=ColorConfig(),
        cache_probe=cache_probe,
    )
    assert isinstance(result, tuple) and len(result) == 3
    frames, categories, details = result
    assert frames == selection_frames
    assert categories == {frame: "Auto" for frame in selection_frames}
    assert set(details) == set(selection_frames)


def test_motion_quarter_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(
        num_frames=240,
        brightness=[0.5 for _ in range(240)],
        motion=[0.0 for _ in range(240)],
    )

    def fake_collect(
        analysis_clip: FakeClip,
        cfg: AnalysisConfig,
        indices: Sequence[int],
        progress: object = None,
        *,
        color_cfg: ColorConfig | None = None,
        file_name: str | None = None,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        brightness = [(idx, 0.0) for idx in indices]
        motion = [(idx, float(idx)) for idx in indices]
        return brightness, motion

    monkeypatch.setattr(analysis_mod, "_collect_metrics_vapoursynth", fake_collect)
    monkeypatch.setattr(
        analysis_mod.vs_core,
        "process_clip_for_screenshot",
        lambda clip, file_name, color_cfg, **kwargs: types.SimpleNamespace(clip=clip, overlay_text=None, verification=None),
    )

    cfg = AnalysisConfig(
        frame_count_dark=0,
        frame_count_bright=0,
        frame_count_motion=4,
        random_frames=0,
        user_frames=[],
        screen_separation_sec=8,
        motion_diff_radius=0,
        analyze_in_sdr=False,
        step=1,
    )

    color_cfg = ColorConfig()
    frames = _select_frames_list(clip, cfg, files=["file.mkv"], file_under_analysis="file.mkv", color_cfg=color_cfg)
    assert len(frames) == 4
    diffs = [b - a for a, b in zip(frames, frames[1:], strict=False)]
    assert all(diff >= 48 for diff in diffs)
    assert any(diff < 192 for diff in diffs)


def test_select_frames_skips_metric_collection_when_unneeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = FakeClip(
        num_frames=120,
        brightness=[i / 120 for i in range(120)],
        motion=[(120 - i) / 120 for i in range(120)],
    )

    cfg = AnalysisConfig(
        frame_count_dark=0,
        frame_count_bright=0,
        frame_count_motion=0,
        random_frames=2,
        user_frames=[10],
        screen_separation_sec=0,
        step=1,
        analyze_in_sdr=False,
    )

    def _fail_collect(*args: object, **kwargs: object) -> None:
        raise AssertionError("metric collection should be bypassed")

    monkeypatch.setattr(analysis_mod, "_collect_metrics_vapoursynth", _fail_collect)
    monkeypatch.setattr(analysis_mod, "_generate_metrics_fallback", _fail_collect)

    frames, categories, _details = _select_frames_with_metadata(
        clip,
        cfg,
        files=["clip.mkv"],
        file_under_analysis="clip.mkv",
    )

    assert frames
    assert set(categories.values()).issubset({"User", "Random"})

def test_selection_details_to_json_roundtrip():
    detail = SelectionDetail(
        frame_index=10,
        label="Bright",
        score=0.75,
        source="unit-test",
        timecode="00:00:10.000",
        clip_role="analyze",
        notes="sample",
    )
    payload = selection_details_to_json({10: detail})
    assert payload["10"]["type"] == "Bright"
    assert payload["10"]["timecode"] == "00:00:10.000"
    assert payload["10"]["notes"] == "sample"


def test_write_selection_cache_file(tmp_path: Path) -> None:
    cfg = AnalysisConfig()
    detail = SelectionDetail(
        frame_index=12,
        label="Random",
        score=0.5,
        source="unit",
        timecode="00:00:05.000",
        clip_role="analyze",
        notes=None,
    )
    target = tmp_path / "generated.compframes"
    write_selection_cache_file(
        target,
        analyzed_file="clip.mkv",
        clip_paths=[tmp_path / "clip.mkv"],
        cfg=cfg,
        selection_hash="hash123",
        selection_frames=[12],
        selection_details={12: detail},
        selection_categories={12: "Random"},
    )
    data = json.loads(target.read_text())
    assert data["selection_hash"] == "hash123"
    assert data["selection"]["frames"] == [12]
    assert data["selection"]["annotations"]["12"].startswith("sel=Random")
