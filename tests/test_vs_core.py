import logging
import math
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pytest

from src.datatypes import ColorConfig
from src.frame_compare import vs as vs_core
from src.frame_compare.vs import VerificationResult
from src.frame_compare.vs import color as vs_color
from src.frame_compare.vs import source as vs_source
from src.frame_compare.vs import tonemap as vs_tonemap


@pytest.fixture(autouse=True)
def reset_tonemap_kwargs() -> Iterator[None]:
    vs_core._TONEMAP_UNSUPPORTED_KWARGS.clear()
    yield
    vs_core._TONEMAP_UNSUPPORTED_KWARGS.clear()


def test_pick_verify_frame_warns_when_no_frames() -> None:
    clip = types.SimpleNamespace(num_frames=0)
    cfg = types.SimpleNamespace(
        verify_frame=None,
        verify_auto=True,
        verify_start_seconds=10.0,
        verify_step_seconds=10.0,
        verify_max_seconds=90.0,
        verify_luma_threshold=0.10,
    )
    warnings: list[str] = []

    frame_idx, auto_selected = vs_core._pick_verify_frame(
        clip,
        cfg,
        fps=24.0,
        file_name="clip.mkv",
        warning_sink=warnings,
    )

    assert frame_idx == 0
    assert auto_selected is False
    assert warnings == ["[VERIFY] clip.mkv has no frames; using frame 0"]


def test_resolve_effective_tonemap_uses_preset_defaults() -> None:
    cfg = types.SimpleNamespace(
        preset="contrast",
        tone_curve="bt.2390",
        target_nits=100.0,
        dynamic_peak_detection=True,
        dst_min_nits=0.1,
        knee_offset=0.4,
        dpd_preset="high_quality",
        dpd_black_cutoff=0.02,
        _provided_keys={"preset"},
    )

    resolved = vs_core.resolve_effective_tonemap(cfg)

    assert resolved["preset"] == "contrast"
    assert resolved["tone_curve"] == "bt.2390"
    assert resolved["target_nits"] == 110.0
    assert resolved["dynamic_peak_detection"] is True
    assert resolved["dst_min_nits"] == pytest.approx(0.15)
    assert resolved["knee_offset"] == pytest.approx(0.42)
    assert resolved["dpd_preset"] == "high_quality"
    assert resolved["dpd_black_cutoff"] == pytest.approx(0.008)
    assert resolved["smoothing_period"] == pytest.approx(30.0)
    assert resolved["scene_threshold_low"] == pytest.approx(0.8)
    assert resolved["scene_threshold_high"] == pytest.approx(2.2)
    assert resolved["percentile"] == pytest.approx(99.99)
    assert resolved["contrast_recovery"] == pytest.approx(0.45)
    assert resolved["metadata"] == 0
    assert resolved["use_dovi"] is True
    assert resolved["visualize_lut"] is False
    assert resolved["show_clipping"] is False


def test_resolve_effective_tonemap_keeps_manual_overrides() -> None:
    cfg = ColorConfig()
    cfg.preset = "filmic"
    cfg.target_nits = 120.0
    cfg._provided_keys = {"preset", "target_nits"}

    resolved = vs_core.resolve_effective_tonemap(cfg)

    assert resolved["preset"] == "filmic"
    assert resolved["tone_curve"] == "bt.2446a"
    assert resolved["target_nits"] == pytest.approx(120.0)
    assert resolved["knee_offset"] == pytest.approx(0.58)
    assert resolved["dpd_black_cutoff"] == pytest.approx(0.008)
    assert resolved["contrast_recovery"] == pytest.approx(0.20)


def test_resolve_tonemap_settings_uses_color_defaults() -> None:
    cfg = types.SimpleNamespace(
        preset="custom",
        tone_curve="bt.2390",
        target_nits=100.0,
        dynamic_peak_detection=True,
        dst_min_nits=0.18,
        knee_offset=0.5,
        dpd_preset="high_quality",
        dpd_black_cutoff=0.01,
        metadata="auto",
        use_dovi=None,
        visualize_lut=False,
        show_clipping=False,
        _provided_keys=set(),
    )

    settings = vs_core._resolve_tonemap_settings(cfg)

    assert settings.smoothing_period == pytest.approx(45.0)
    assert settings.scene_threshold_low == pytest.approx(0.8)
    assert settings.scene_threshold_high == pytest.approx(2.4)
    assert settings.percentile == pytest.approx(99.995)
    assert settings.contrast_recovery == pytest.approx(0.3)


def testprops_signal_hdr_accepts_transfer_only() -> None:
    props = {"_Transfer": 16}

    assert vs_core.props_signal_hdr(props) is True


def testprops_signal_hdr_accepts_primaries_only() -> None:
    props = {"_Primaries": 9}

    assert vs_core.props_signal_hdr(props) is True


def testprops_signal_hdr_detects_mastering_metadata() -> None:
    props = {"_MasteringDisplayMaxLuminance": 1000}

    assert vs_core.props_signal_hdr(props) is True


class _FakeSampleType:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self._value = value

    def __int__(self) -> int:
        return self._value


@dataclass
class _FakeFormat:
    bits_per_sample: int
    sample_type: _FakeSampleType


class _FakeClip:
    def __init__(self, fmt: _FakeFormat) -> None:
        self.format = fmt


class _FakeFrame:
    def __init__(self, props: Dict[str, float]) -> None:
        self.props = props


class _FakeStatsClip:
    def __init__(self, props: Dict[str, float]) -> None:
        self._props = props

    def get_frame(self, index: int) -> _FakeFrame:
        return _FakeFrame(self._props)


class _DetectStatsFrame:
    def __init__(self, props: Dict[str, float]) -> None:
        self.props = props


class _DetectStatsClip:
    def __init__(self, frames: Sequence[Dict[str, float]]) -> None:
        self._frames = [_DetectStatsFrame(item) for item in frames]
        self.num_frames = len(self._frames)

    def get_frame(self, index: int) -> _DetectStatsFrame:
        if not self._frames:
            raise RuntimeError("no frames available")
        clamped = min(max(index, 0), len(self._frames) - 1)
        return self._frames[clamped]


class _DetectStd:
    def __init__(self, frames: Sequence[Dict[str, float]]) -> None:
        self._stats_clip = _DetectStatsClip(frames)

    def PlaneStats(self, clip: Any) -> _DetectStatsClip:
        return self._stats_clip


class _FakeStd:
    def __init__(self, expr_clip: _FakeClip, stats_clip: _FakeStatsClip) -> None:
        self._expr_clip = expr_clip
        self._stats_clip = stats_clip
        self.expr_calls: List[Sequence[Any]] = []

    def Expr(self, clips: Sequence[Any], expr: str) -> _FakeClip:
        self.expr_calls.append(clips)
        return self._expr_clip

    def PlaneStats(self, clip: _FakeClip) -> _FakeStatsClip:
        return self._stats_clip


class _FakeCore:
    def __init__(self, expr_clip: _FakeClip, stats_clip: _FakeStatsClip) -> None:
        self.std = _FakeStd(expr_clip, stats_clip)


def _run_compute_verification(
    fmt: _FakeFormat, props: Dict[str, float]
) -> VerificationResult:
    expr_clip = _FakeClip(fmt)
    stats_clip = _FakeStatsClip(props)
    core = _FakeCore(expr_clip, stats_clip)
    return vs_core._compute_verification(core, object(), object(), 3, auto_selected=False)


def test_detect_rgb_color_range_identifies_limited(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    frames = [
        {"PlaneStatsMin": 4096.0, "PlaneStatsMax": 50200.0},
        {"PlaneStatsMin": 4200.0, "PlaneStatsMax": 49600.0},
    ]
    std = _DetectStd(frames)
    clip = types.SimpleNamespace(
        format=types.SimpleNamespace(
            color_family=fake_vs.RGB,
            sample_type=_FakeSampleType("INTEGER", 0),
            bits_per_sample=16,
        )
    )
    core = types.SimpleNamespace(std=std)

    detected, source = vs_core._detect_rgb_color_range(
        core,
        clip,
        log=logging.getLogger("test"),
        label="limited",
    )

    assert detected == fake_vs.RANGE_LIMITED
    assert source == "plane_stats"


def test_detect_rgb_color_range_identifies_full(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    frames = [
        {"PlaneStatsMin": 0.0, "PlaneStatsMax": 65535.0},
        {"PlaneStatsMin": 300.0, "PlaneStatsMax": 64000.0},
    ]
    std = _DetectStd(frames)
    clip = types.SimpleNamespace(
        format=types.SimpleNamespace(
            color_family=fake_vs.RGB,
            sample_type=_FakeSampleType("INTEGER", 0),
            bits_per_sample=16,
        )
    )
    core = types.SimpleNamespace(std=std)

    detected, source = vs_core._detect_rgb_color_range(
        core,
        clip,
        log=logging.getLogger("test"),
        label="full",
    )

    assert detected == fake_vs.RANGE_FULL
    assert source == "plane_stats"


def test_detect_rgb_color_range_detects_undershoot(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    frames = [
        {"PlaneStatsMin": 2000.0, "PlaneStatsMax": 42000.0},
        {"PlaneStatsMin": 2100.0, "PlaneStatsMax": 43000.0},
    ]
    std = _DetectStd(frames)
    clip = types.SimpleNamespace(
        format=types.SimpleNamespace(
            color_family=fake_vs.RGB,
            sample_type=_FakeSampleType("INTEGER", 0),
            bits_per_sample=16,
        )
    )
    core = types.SimpleNamespace(std=std)

    detected, source = vs_core._detect_rgb_color_range(
        core,
        clip,
        log=logging.getLogger("test"),
        label="undershoot",
    )

    assert detected == fake_vs.RANGE_LIMITED
    assert source == "plane_stats"


def test_compute_verification_normalizes_integer_clip() -> None:
    fmt = _FakeFormat(bits_per_sample=8, sample_type=_FakeSampleType("INTEGER", 0))
    props = {"PlaneStatsAverage": 25.5, "PlaneStatsMax": 51.0}
    result = _run_compute_verification(fmt, props)
    assert math.isclose(result.average, 0.1)
    assert math.isclose(result.maximum, 0.2)


def test_compute_verification_preserves_float_clip() -> None:
    fmt = _FakeFormat(bits_per_sample=32, sample_type=_FakeSampleType("FLOAT", 1))
    props = {"PlaneStatsAverage": 0.25, "PlaneStatsMax": 0.75}
    result = _run_compute_verification(fmt, props)
    assert math.isclose(result.average, 0.25)
    assert math.isclose(result.maximum, 0.75)


def test_apply_post_gamma_levels_uses_limited_bounds() -> None:
    captured: Dict[str, Any] = {}

    class FakeStd:
        def Levels(self, clip: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return "gamma"

    core = types.SimpleNamespace(std=FakeStd())
    clip = types.SimpleNamespace(
        format=_FakeFormat(bits_per_sample=16, sample_type=_FakeSampleType("INTEGER", 0))
    )
    result, applied = vs_core._apply_post_gamma_levels(
        core,
        clip=clip,
        gamma=0.95,
        file_name="sample",
        log=logging.getLogger("test"),
    )
    assert result == "gamma"
    assert applied is True
    assert captured["min_in"] == 16 * 257
    assert captured["max_in"] == 235 * 257
    assert captured["min_out"] == 16 * 257
    assert captured["max_out"] == 235 * 257
    assert captured["gamma"] == pytest.approx(0.95)


def test_apply_post_gamma_levels_skips_when_unity() -> None:
    core = types.SimpleNamespace(std=None)
    clip = object()
    result, applied = vs_core._apply_post_gamma_levels(
        core,
        clip=clip,
        gamma=1.0,
        file_name="skip",
        log=logging.getLogger("test"),
    )
    assert result is clip
    assert applied is False


def test_apply_post_gamma_levels_scales_float_clip() -> None:
    captured: Dict[str, Any] = {}

    class FakeStd:
        def Levels(self, clip: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return clip

    clip = types.SimpleNamespace(
        format=_FakeFormat(bits_per_sample=32, sample_type=_FakeSampleType("FLOAT", 1))
    )
    core = types.SimpleNamespace(std=FakeStd())
    _, applied = vs_core._apply_post_gamma_levels(
        core,
        clip=clip,
        gamma=1.05,
        file_name="float",
        log=logging.getLogger("test"),
    )
    assert applied is True
    assert captured["min_in"] == pytest.approx(16 / 255)
    assert captured["max_in"] == pytest.approx(235 / 255)
    assert captured["min_out"] == pytest.approx(16 / 255)
    assert captured["max_out"] == pytest.approx(235 / 255)
    assert captured["gamma"] == pytest.approx(1.05)


def test_tonemap_kwargs_include_optional_parameters() -> None:
    captured: Dict[str, Any] = {}

    def fake_tonemap(clip: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return clip

    core = types.SimpleNamespace(libplacebo=types.SimpleNamespace(Tonemap=fake_tonemap))
    vs_core._tonemap_with_retries(
        core,
        rgb_clip=object(),
        tone_curve="bt.2390",
        target_nits=120.0,
        dst_min=0.18,
        dpd=1,
        knee_offset=0.45,
        dpd_preset="high_quality",
        dpd_black_cutoff=0.01,
        smoothing_period=25.0,
        scene_threshold_low=0.8,
        scene_threshold_high=2.5,
        percentile=99.5,
        contrast_recovery=0.3,
        metadata=2,
        use_dovi=True,
        visualize_lut=False,
        show_clipping=False,
        src_hint=None,
        file_name="demo",
    )

    assert captured["tone_mapping_param"] == pytest.approx(0.45)
    assert captured["peak_detection_preset"] == "high_quality"
    assert captured["black_cutoff"] == pytest.approx(0.01)
    assert captured["smoothing_period"] == pytest.approx(25.0)
    assert captured["scene_threshold_low"] == pytest.approx(0.8)
    assert captured["scene_threshold_high"] == pytest.approx(2.5)
    assert captured["percentile"] == pytest.approx(99.5)
    assert captured["contrast_recovery"] == pytest.approx(0.3)
    assert captured["metadata"] == 2
    assert captured["use_dovi"] is True
    assert captured["visualize_lut"] is False
    assert captured["show_clipping"] is False


def test_tonemap_drops_unsupported_kwargs_and_warns() -> None:
    captured: Dict[str, Any] = {}

    class RejectingTonemap:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, clip: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise TypeError("got an unexpected keyword argument 'peak_detection_preset'")
            captured.update(kwargs)
            return clip

    core = types.SimpleNamespace(libplacebo=types.SimpleNamespace(Tonemap=RejectingTonemap()))
    vs_core._tonemap_with_retries(
        core,
        rgb_clip=object(),
        tone_curve="bt.2390",
        target_nits=100.0,
        dst_min=0.18,
        dpd=1,
        knee_offset=0.5,
        dpd_preset="high_quality",
        dpd_black_cutoff=0.01,
        smoothing_period=20.0,
        scene_threshold_low=1.0,
        scene_threshold_high=3.0,
        percentile=100.0,
        contrast_recovery=0.0,
        metadata=None,
        use_dovi=True,
        visualize_lut=False,
        show_clipping=False,
        src_hint=None,
        file_name="compat",
    )

    assert "peak_detection_preset" not in captured
    assert "tone_mapping_param" in captured
    assert "black_cutoff" in captured
    assert captured["smoothing_period"] == pytest.approx(20.0)
    assert captured["scene_threshold_low"] == pytest.approx(1.0)
    assert captured["scene_threshold_high"] == pytest.approx(3.0)
    assert "percentile" in captured
    assert "contrast_recovery" in captured


def test_tonemap_drops_kwargs_when_vapoursynth_error_lists_multiple_names() -> None:
    captured: Dict[str, Any] = {}

    class FakeVSError(Exception):
        pass

    class RejectingTonemap:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, clip: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise FakeVSError(
                    "Tonemap: Function does not take argument(s) named "
                    "peak_detection_preset, black_cutoff"
                )
            captured.update(kwargs)
            return clip

    core = types.SimpleNamespace(libplacebo=types.SimpleNamespace(Tonemap=RejectingTonemap()))
    vs_core._tonemap_with_retries(
        core,
        rgb_clip=object(),
        tone_curve="bt.2390",
        target_nits=100.0,
        dst_min=0.18,
        dpd=1,
        knee_offset=0.5,
        dpd_preset="high_quality",
        dpd_black_cutoff=0.01,
        smoothing_period=20.0,
        scene_threshold_low=1.0,
        scene_threshold_high=3.0,
        percentile=100.0,
        contrast_recovery=0.0,
        metadata=None,
        use_dovi=True,
        visualize_lut=False,
        show_clipping=False,
        src_hint=None,
        file_name="vapoursynth",
    )

    assert "peak_detection_preset" not in captured
    assert "black_cutoff" not in captured
    assert "tone_mapping_param" in captured
    assert captured["smoothing_period"] == pytest.approx(20.0)
    assert captured["scene_threshold_low"] == pytest.approx(1.0)
    assert captured["scene_threshold_high"] == pytest.approx(3.0)


class _DummyStd:
    def __init__(self, clip: "_DummyClip") -> None:
        self._clip = clip
        self.calls: List[Dict[str, int]] = []

    def SetFrameProps(self, clip: Any, **kwargs: int) -> "_DummyClip":
        assert clip is self._clip
        self.calls.append({key: int(value) for key, value in kwargs.items()})
        return self._clip


class _DummyClip:
    def __init__(self, fake_vs: Any, height: int) -> None:
        self.format = types.SimpleNamespace(color_family=getattr(fake_vs, "YUV", object()))
        self.height = height
        self.std = _DummyStd(self)


class _PaddingTestFrame:
    def __init__(self, props: Mapping[str, Any]) -> None:
        self.props = dict(props)


class _PaddingTestClip:
    def __init__(
        self,
        core: "_PaddingCore",
        frames: Sequence[Mapping[str, Any]],
        *,
        height: int = 1080,
        fps_num: int = 24000,
        fps_den: int = 1001,
    ) -> None:
        self.core = core
        self.std = core.std
        self.resize = core.resize
        self.format = types.SimpleNamespace(color_family=getattr(core.vs_module, "YUV", object()))
        self.height = height
        self.fps_num = fps_num
        self.fps_den = fps_den
        self._frames = [_PaddingTestFrame(props) for props in frames]
        self.num_frames = len(self._frames)

    def _copy_frames(self) -> List[Mapping[str, Any]]:
        return [dict(frame.props) for frame in self._frames]

    def __add__(self, other: "_PaddingTestClip") -> "_PaddingTestClip":
        return _PaddingTestClip(
            self.core,
            self._copy_frames() + other._copy_frames(),
            height=self.height,
            fps_num=self.fps_num,
            fps_den=self.fps_den,
        )

    def get_frame(self, index: int) -> _PaddingTestFrame:
        if not self._frames:
            raise RuntimeError("clip has no frames")
        idx = min(max(index, 0), len(self._frames) - 1)
        return self._frames[idx]

    def apply_frame_props(self, props: Mapping[str, Any]) -> None:
        for frame in self._frames:
            frame.props.update(props)


class _PaddingStd:
    def __init__(self, core: "_PaddingCore") -> None:
        self._core = core

    def BlankClip(self, clip: _PaddingTestClip, length: int) -> _PaddingTestClip:
        return _PaddingTestClip(
            self._core,
            [{} for _ in range(length)],
            height=clip.height,
            fps_num=clip.fps_num,
            fps_den=clip.fps_den,
        )

    def SetFrameProp(self, clip: _PaddingTestClip, **kwargs: Any) -> _PaddingTestClip:
        clip.apply_frame_props(kwargs)
        return clip

    def SetFrameProps(self, clip: _PaddingTestClip, **kwargs: Any) -> _PaddingTestClip:
        clip.apply_frame_props(kwargs)
        return clip


class _PaddingResize:
    def Spline36(self, clip: _PaddingTestClip, **kwargs: Any) -> _PaddingTestClip:
        return clip

    def Point(self, clip: _PaddingTestClip, **kwargs: Any) -> _PaddingTestClip:
        return clip


class _PaddingCore:
    def __init__(self, vs_module: Any) -> None:
        self.vs_module = vs_module
        self.std = _PaddingStd(self)
        self.resize = _PaddingResize()


def _install_fake_vs(monkeypatch: Any, **overrides: int) -> Any:
    yuv_family = object()
    attributes = dict(
        YUV=yuv_family,
        RGB=object(),
        MATRIX_BT709=1,
        MATRIX_SMPTE170M=6,
        MATRIX_BT2020_CL=9,
        MATRIX_BT2020_NCL=9,
        PRIMARIES_BT709=1,
        PRIMARIES_SMPTE170M=6,
        PRIMARIES_BT2020=9,
        TRANSFER_BT709=1,
        TRANSFER_SMPTE170M=6,
        TRANSFER_ST2084=16,
        RANGE_LIMITED=1,
        RANGE_FULL=0,
    )
    attributes.update(overrides)
    fake_vs = types.SimpleNamespace(**attributes)
    from src.frame_compare.vs import env as vs_env

    monkeypatch.setitem(sys.modules, "vapoursynth", fake_vs)
    monkeypatch.setattr(vs_env, "_vs_module", fake_vs, raising=False)
    monkeypatch.setattr(vs_core, "_vs_module", fake_vs, raising=False)
    return fake_vs


def test_normalise_color_metadata_infers_hd_defaults(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    clip = _DummyClip(fake_vs, height=1080)

    normalised_clip, props, color_tuple = vs_core.normalise_color_metadata(
        clip,
        {},
        color_cfg=ColorConfig(),
        file_name="clip.mkv",
    )

    assert normalised_clip is clip
    assert color_tuple == (
        int(fake_vs.MATRIX_BT709),
        int(fake_vs.TRANSFER_BT709),
        int(fake_vs.PRIMARIES_BT709),
        int(fake_vs.RANGE_LIMITED),
    )
    assert props["_Matrix"] == int(fake_vs.MATRIX_BT709)
    assert props["_Transfer"] == int(fake_vs.TRANSFER_BT709)
    assert props["_Primaries"] == int(fake_vs.PRIMARIES_BT709)
    assert props["_ColorRange"] == int(fake_vs.RANGE_LIMITED)
    assert clip.std.calls == [
        {
            "_Matrix": int(fake_vs.MATRIX_BT709),
            "_Transfer": int(fake_vs.TRANSFER_BT709),
            "_Primaries": int(fake_vs.PRIMARIES_BT709),
            "_ColorRange": int(fake_vs.RANGE_LIMITED),
        }
    ]


def test_normalise_color_metadata_backfills_hdr_defaults(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    clip = _DummyClip(fake_vs, height=1080)

    _, props, color_tuple = vs_core.normalise_color_metadata(
        clip,
        {"_Transfer": 16},
        color_cfg=ColorConfig(),
        file_name="hdr.mkv",
    )

    expected_matrix = int(
        getattr(fake_vs, "MATRIX_BT2020_CL", getattr(fake_vs, "MATRIX_BT2020_NCL", 9))
    )
    assert color_tuple == (
        expected_matrix,
        16,
        int(fake_vs.PRIMARIES_BT2020),
        int(fake_vs.RANGE_LIMITED),
    )
    assert props["_Matrix"] == expected_matrix
    assert props["_Transfer"] == 16
    assert props["_Primaries"] == int(fake_vs.PRIMARIES_BT2020)
    assert vs_core.props_signal_hdr(props) is True


def test_normalise_color_metadata_infers_sd_defaults(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(
        monkeypatch,
        MATRIX_SMPTE170M=106,
        PRIMARIES_SMPTE170M=206,
        TRANSFER_SMPTE170M=306,
        RANGE_LIMITED=17,
    )
    clip = _DummyClip(fake_vs, height=480)

    normalised_clip, props, color_tuple = vs_core.normalise_color_metadata(
        clip,
        {},
        color_cfg=ColorConfig(),
        file_name="clip_sd.mkv",
    )

    assert normalised_clip is clip
    assert color_tuple == (106, 306, 206, 17)
    assert props["_Matrix"] == 106
    assert props["_Transfer"] == 306
    assert props["_Primaries"] == 206
    assert props["_ColorRange"] == 17


def test_normalise_color_metadata_honours_overrides(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    clip = _DummyClip(fake_vs, height=2160)

    cfg = ColorConfig(
        color_overrides={
            "clip.mkv": {
                "matrix": "bt2020",
                "primaries": "bt2020",
                "transfer": "st2084",
                "range": "full",
            }
        }
    )

    _, props, color_tuple = vs_core.normalise_color_metadata(
        clip,
        {},
        color_cfg=cfg,
        file_name="clip.mkv",
    )

    assert color_tuple == (9, 16, 9, 0)
    assert props["_Matrix"] == 9
    assert props["_Transfer"] == 16
    assert props["_Primaries"] == 9
    assert props["_ColorRange"] == 0


def test_normalise_color_metadata_honours_path_overrides(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    clip = _DummyClip(fake_vs, height=2160)
    override_path = tmp_path / "shots" / "demo.mkv"
    cfg = ColorConfig(
        color_overrides={
            str(override_path): {
                "matrix": "bt2020",
                "primaries": "bt2020",
                "transfer": "st2084",
                "range": "full",
            }
        }
    )

    _, props, color_tuple = vs_core.normalise_color_metadata(
        clip,
        {},
        color_cfg=cfg,
        file_name=override_path,
    )

    assert color_tuple == (9, 16, 9, 0)
    assert props["_Matrix"] == 9
    assert props["_Transfer"] == 16
    assert props["_Primaries"] == 9
    assert props["_ColorRange"] == 0


def test_normalise_color_metadata_infers_limited_via_signal(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    clip = _DummyClip(fake_vs, height=1080)

    monkeypatch.setattr(vs_color, "_compute_luma_bounds", lambda _: (16.0, 234.0))
    monkeypatch.setattr(vs_core, "_compute_luma_bounds", lambda _: (16.0, 234.0))

    warnings: list[str] = []
    _, props, _ = vs_core.normalise_color_metadata(
        clip,
        {},
        color_cfg=ColorConfig(),
        file_name="clip.mkv",
        warning_sink=warnings,
    )

    assert props["_ColorRange"] == int(fake_vs.RANGE_LIMITED)
    assert warnings  # warning emitted for metadata adjustment


def test_normalise_color_metadata_warns_for_mismatched_limited(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    clip = _DummyClip(fake_vs, height=1080)

    monkeypatch.setattr(vs_color, "_compute_luma_bounds", lambda _: (0.0, 255.0))
    monkeypatch.setattr(vs_core, "_compute_luma_bounds", lambda _: (0.0, 255.0))

    warnings: list[str] = []
    _, props, _ = vs_core.normalise_color_metadata(
        clip,
        {"_ColorRange": int(fake_vs.RANGE_LIMITED)},
        color_cfg=ColorConfig(),
        file_name="clip.mkv",
        warning_sink=warnings,
    )

    # Range remains limited but a warning is emitted for the suspicious signal.
    assert props["_ColorRange"] == int(fake_vs.RANGE_LIMITED)
    assert warnings


def test_process_clip_tonemaps_after_negative_trim(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    fake_vs.RGB48 = object()
    fake_vs.RGB24 = object()
    core = _PaddingCore(fake_vs)
    hdr_props = {"_Matrix": 9, "_Primaries": 9, "_Transfer": 16, "_ColorRange": 1}
    clip = _PaddingTestClip(core, [hdr_props])
    padded = vs_source._extend_with_blank(clip, core, length=2, frame_props=hdr_props)

    def _fake_normalise(
        clip_obj: Any,
        props: Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[Any, Mapping[str, Any], tuple[Any, Any, Any, Any]]:
        return clip_obj, props, (
            props.get("_Matrix"),
            props.get("_Transfer"),
            props.get("_Primaries"),
            props.get("_ColorRange"),
        )

    monkeypatch.setattr(vs_tonemap, "normalise_color_metadata", _fake_normalise)
    monkeypatch.setattr(vs_tonemap, "_normalize_rgb_props", lambda clip_obj, *args, **kwargs: clip_obj)
    monkeypatch.setattr(vs_tonemap, "_deduce_src_csp_hint", lambda *args, **kwargs: "hdr_hint")
    monkeypatch.setattr(vs_tonemap, "_tonemap_with_retries", lambda _core, rgb_clip, **kwargs: rgb_clip)
    monkeypatch.setattr(
        vs_tonemap,
        "_detect_rgb_color_range",
        lambda *args, **kwargs: (fake_vs.RANGE_LIMITED, "plane_stats"),
    )
    monkeypatch.setattr(
        vs_tonemap,
        "_apply_post_gamma_levels",
        lambda _core, clip_obj, **kwargs: (clip_obj, False),
    )

    cfg = ColorConfig()
    cfg.enable_tonemap = True
    cfg.overlay_enabled = False
    cfg.verify_enabled = False
    cfg.strict = False

    result = vs_core.process_clip_for_screenshot(
        padded,
        "hdr_pad.mkv",
        cfg,
        enable_overlay=False,
        enable_verification=False,
    )

    assert result.tonemap.applied is True
    assert result.tonemap.reason is None


def test_init_clip_preserves_mastering_metadata_on_padding(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    core = _PaddingCore(fake_vs)
    clip = _PaddingTestClip(core, [{}])

    mastering_props = {"_MasteringDisplayMaxLuminance": 1500}
    monkeypatch.setattr(vs_source, "_open_clip_with_sources", lambda *args, **kwargs: clip)
    monkeypatch.setattr(vs_source, "snapshot_frame_props", lambda _clip: dict(mastering_props))

    padded_clip = vs_source.init_clip(
        str(tmp_path / "sample.mkv"),
        trim_start=-3,
        core=core,
    )

    first_frame = padded_clip.get_frame(0)
    assert (
        first_frame.props["_MasteringDisplayMaxLuminance"]
        == mastering_props["_MasteringDisplayMaxLuminance"]
    )


def test_init_clip_reuses_source_props_hint_for_padding(
    monkeypatch: Any, tmp_path: Path
) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    core = _PaddingCore(fake_vs)
    clip = _PaddingTestClip(core, [{}])

    monkeypatch.setattr(vs_source, "_open_clip_with_sources", lambda *args, **kwargs: clip)
    snapshot_called = False

    def fake_snapshot(_clip: Any) -> Mapping[str, Any]:
        nonlocal snapshot_called
        snapshot_called = True
        return {"_Matrix": 1}

    monkeypatch.setattr(vs_source, "snapshot_frame_props", fake_snapshot)

    collected: list[Mapping[str, Any]] = []

    def fake_collect(frame_props: Mapping[str, Any]) -> Mapping[str, Any]:
        collected.append(dict(frame_props))
        return dict(frame_props)

    monkeypatch.setattr(vs_source, "_collect_blank_extension_props", fake_collect)

    captured_props: list[Mapping[str, Any]] = []

    def capture_sink(props: Mapping[str, Any]) -> None:
        captured_props.append(dict(props))

    props_hint = {"_Matrix": 9, "_Transfer": 16}

    vs_source.init_clip(
        str(tmp_path / "sample.mkv"),
        trim_start=-2,
        core=core,
        frame_props_sink=capture_sink,
        source_frame_props_hint=props_hint,
    )

    assert collected and collected[0] == props_hint
    assert captured_props and captured_props[0] == props_hint
    assert snapshot_called is False


def test_process_clip_tonemaps_with_mastering_metadata_only(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    fake_vs.RGB48 = object()
    fake_vs.RGB24 = object()
    core = _PaddingCore(fake_vs)
    hdr_props = {"_MasteringDisplayMaxLuminance": 2000}
    clip = _PaddingTestClip(core, [dict(hdr_props)])

    padding_props = vs_source._collect_blank_extension_props(hdr_props)
    padded = vs_source._extend_with_blank(clip, core, length=2, frame_props=padding_props)

    def _fake_normalise(
        clip_obj: Any,
        props: Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[Any, Mapping[str, Any], tuple[Any, Any, Any, Any]]:
        normalized_props = dict(props)
        normalized_props.setdefault("_Matrix", 9)
        normalized_props.setdefault("_Transfer", 16)
        normalized_props.setdefault("_Primaries", 9)
        normalized_props.setdefault("_ColorRange", int(fake_vs.RANGE_LIMITED))
        return clip_obj, normalized_props, (
            normalized_props.get("_Matrix"),
            normalized_props.get("_Transfer"),
            normalized_props.get("_Primaries"),
            normalized_props.get("_ColorRange"),
        )

    monkeypatch.setattr(vs_tonemap, "normalise_color_metadata", _fake_normalise)
    monkeypatch.setattr(vs_tonemap, "_normalize_rgb_props", lambda clip_obj, *args, **kwargs: clip_obj)
    monkeypatch.setattr(vs_tonemap, "_deduce_src_csp_hint", lambda *args, **kwargs: "hdr_hint")
    monkeypatch.setattr(vs_tonemap, "_tonemap_with_retries", lambda _core, rgb_clip, **kwargs: rgb_clip)
    monkeypatch.setattr(
        vs_tonemap,
        "_detect_rgb_color_range",
        lambda *args, **kwargs: (fake_vs.RANGE_LIMITED, "plane_stats"),
    )
    monkeypatch.setattr(
        vs_tonemap,
        "_apply_post_gamma_levels",
        lambda _core, clip_obj, **kwargs: (clip_obj, False),
    )

    cfg = ColorConfig()
    cfg.enable_tonemap = True
    cfg.overlay_enabled = False
    cfg.verify_enabled = False
    cfg.strict = False

    result = vs_core.process_clip_for_screenshot(
        padded,
        "hdr_md_only.mkv",
        cfg,
        enable_overlay=False,
        enable_verification=False,
    )

    assert result.tonemap.applied is True
    assert result.tonemap.reason is None


def test_process_clip_rehydrates_mastering_metadata_from_plan(monkeypatch: Any) -> None:
    fake_vs = _install_fake_vs(monkeypatch)
    fake_vs.RGB48 = object()
    fake_vs.RGB24 = object()
    core = _PaddingCore(fake_vs)
    clip = _PaddingTestClip(core, [{}])
    stored_props = {
        "_MasteringDisplayMinLuminance": 0.005,
        "_MasteringDisplayMaxLuminance": 1500,
    }

    def _fake_normalise(
        clip_obj: Any,
        props: Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[Any, Mapping[str, Any], tuple[Any, Any, Any, Any]]:
        normalized_props = dict(props)
        normalized_props.setdefault("_Matrix", 9)
        normalized_props.setdefault("_Transfer", 16)
        normalized_props.setdefault("_Primaries", 9)
        normalized_props.setdefault("_ColorRange", int(fake_vs.RANGE_LIMITED))
        return clip_obj, normalized_props, (
            normalized_props.get("_Matrix"),
            normalized_props.get("_Transfer"),
            normalized_props.get("_Primaries"),
            normalized_props.get("_ColorRange"),
        )

    monkeypatch.setattr(vs_tonemap, "normalise_color_metadata", _fake_normalise)
    monkeypatch.setattr(vs_tonemap, "_normalize_rgb_props", lambda clip_obj, *args, **kwargs: clip_obj)
    monkeypatch.setattr(vs_tonemap, "_deduce_src_csp_hint", lambda *args, **kwargs: "hdr_hint")
    monkeypatch.setattr(vs_tonemap, "_tonemap_with_retries", lambda _core, rgb_clip, **kwargs: rgb_clip)
    monkeypatch.setattr(
        vs_tonemap,
        "_detect_rgb_color_range",
        lambda *args, **kwargs: (fake_vs.RANGE_LIMITED, "plane_stats"),
    )
    monkeypatch.setattr(
        vs_tonemap,
        "_apply_post_gamma_levels",
        lambda _core, clip_obj, **kwargs: (clip_obj, False),
    )

    cfg = ColorConfig()
    cfg.enable_tonemap = True
    cfg.overlay_enabled = False
    cfg.verify_enabled = False
    cfg.strict = False

    result = vs_core.process_clip_for_screenshot(
        clip,
        "rehydrated_hdr.mkv",
        cfg,
        enable_overlay=False,
        enable_verification=False,
        stored_source_props=stored_props,
    )

    assert result.tonemap.applied is True
    assert result.tonemap.reason is None
