import importlib.util
import logging
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, TypedDict, cast

import pytest

from src.datatypes import (
    ColorConfig,
    ExportRange,
    OddGeometryPolicy,
    RGBDither,
    ScreenshotConfig,
)
from src.frame_compare import subproc as fc_subproc
from src.frame_compare import vs as vs_core
from src.frame_compare.render import naming as render_naming
from src.frame_compare.render.errors import ScreenshotError, ScreenshotWriterError
from src.frame_compare.screenshot import orchestrator, render
from src.frame_compare.screenshot.config import GeometryPlan
from src.frame_compare.screenshot.naming import sanitise_label
from src.frame_compare.screenshot.render import (
    compute_requires_full_chroma,
)

_vapoursynth_available = importlib.util.find_spec("vapoursynth") is not None

pytestmark = pytest.mark.skipif(  # type: ignore[attr-defined]
    not _vapoursynth_available,
    reason="VapourSynth not available – skipping screenshot tests",
)


class _CapturedWriterCall(TypedDict):
    crop: tuple[int, int, int, int]
    scaled: tuple[int, int]
    pad: tuple[int, int, int, int]
    label: str
    requested: int
    frame: int
    selection_label: Optional[str]
    source: Optional[str]


class FakeClip:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.num_frames = 0
        self.props = {}

    def get_frame(self, idx: int) -> Any:
        return types.SimpleNamespace(props=self.props) # type: ignore[reportUnknownMemberType]


def _prepare_fake_vapoursynth_clip(
    monkeypatch: pytest.MonkeyPatch,
    *,
    width: int,
    height: int,
    subsampling_w: int,
    subsampling_h: int,
    bits_per_sample: int,
    color_family: str = "YUV",
    format_name: str | None = None,
) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """VapourSynth Test Stub Factory.

    VapourSynth is a C extension that requires system-level installation (Windows/Linux).
    On macOS dev environments without the native runtime, this factory creates a complete
    mock of the vapoursynth module for testing screenshot geometry, crop/pad calculations,
    and color processing without requiring the real VapourSynth installation.

    Architecture:
        - _FakeClip: Simulates vs.VideoNode with width, height, format, props, and std
        - _FakeFormat: Holds color_family, bits_per_sample, subsampling_w/h
        - _FakeStd: Simulates core.std (CropRel, AddBorders, SetFrameProps, Levels)
        - _FakeResize: Simulates core.resize (Point, Spline36)
        - _FakeFpng: Simulates core.fpng.Write for PNG output

    The stub records all calls to enable behavior verification in tests:
        - writer_calls: PNG write operations with kwargs and props
        - resize_calls: Resolution change operations
        - levels_calls: Gamma/levels adjustment operations

    Args:
        monkeypatch: pytest fixture for patching sys.modules
        width: Initial clip width in pixels
        height: Initial clip height in pixels
        subsampling_w: Chroma horizontal subsampling (0=4:4:4, 1=4:2:2/4:2:0)
        subsampling_h: Chroma vertical subsampling (0=4:4:4/4:2:2, 1=4:2:0)
        bits_per_sample: Bit depth (8, 10, 12, 16)
        color_family: "YUV" or "RGB"
        format_name: Optional format string (e.g., "YUV420P8")

    Returns:
        tuple of (clip, fake_vs, writer_calls, resize_calls, levels_calls)

    Usage::

        clip, fake_vs, writer_calls, resize_calls, levels_calls = \\
            _prepare_fake_vapoursynth_clip(
                monkeypatch,
                width=1920,
                height=1080,
                subsampling_w=1,
                subsampling_h=1,
                bits_per_sample=8,
            )
        # ... run code that uses vapoursynth ...
        assert len(writer_calls) == 1
        assert writer_calls[0]["props"]["_ColorRange"] == 1
    """

    writer_calls: list[dict[str, Any]] = []
    resize_calls: list[dict[str, Any]] = []
    levels_calls: list[dict[str, Any]] = []
    overlay_calls: list[dict[str, Any]] = []

    fake_vs = types.SimpleNamespace()
    yuv_constant = object()
    rgb_constant = object()

    class _FakeFormat:
        def __init__(
            self,
            *,
            color_family_obj: object,
            bits_per_sample_val: int,
            subsampling_w_val: int,
            subsampling_h_val: int,
            name_val: str,
        ) -> None:
            self.color_family = color_family_obj
            self.bits_per_sample = bits_per_sample_val
            self.subsampling_w = subsampling_w_val
            self.subsampling_h = subsampling_h_val
            self.name = name_val

    def _std_levels(clip: "_FakeClip", **kwargs: Any) -> "_FakeClip":
        levels_calls.append(dict(kwargs))
        clip.props.setdefault("_levels_calls", []).append(kwargs)
        return clip

    class _FakeSub:
        def __init__(self, core: SimpleNamespace) -> None:
            self._core = core

        def Subtitle(
            self,
            clip_obj: "_FakeClip",
            *,
            text: Sequence[str],
            style: str,
        ) -> "_FakeClip":
            overlay_calls.append({"text": list(text), "style": style})
            new_clip = clip_obj._with_dimensions()
            new_clip.props = dict(getattr(clip_obj, "props", {}))
            return new_clip

    class _FakeText:
        def __init__(self, core: SimpleNamespace) -> None:
            self._core = core

        def Text(self, clip_obj: "_FakeClip", text: str, alignment: int = 9) -> "_FakeClip":
            overlay_calls.append({"text": [text], "alignment": alignment})
            new_clip = clip_obj._with_dimensions()
            new_clip.props = dict(getattr(clip_obj, "props", {}))
            return new_clip

    class _FakeStd:
        def __init__(self, parent: "_FakeClip") -> None:
            self._parent = parent

        def CropRel(
            self,
            *,
            left: int,
            right: int,
            top: int,
            bottom: int,
        ) -> "_FakeClip":
            new_width = self._parent.width - left - right
            new_height = self._parent.height - top - bottom
            return self._parent._with_dimensions(width=new_width, height=new_height)

        def AddBorders(
            self,
            clip: "_FakeClip",
            *,
            left: int,
            right: int,
            top: int,
            bottom: int,
        ) -> "_FakeClip":
            return clip._with_dimensions(
                width=clip.width + left + right,
                height=clip.height + top + bottom,
            )

        def CopyFrameProps(self, clip: "_FakeClip", _src: "_FakeClip") -> "_FakeClip":
            clip.props = dict(getattr(_src, "props", {}))
            return clip

        def SetFrameProps(self, **kwargs: Any) -> "_FakeClip":
            self._parent.props.update(kwargs)
            return self._parent

        def Levels(self, clip: "_FakeClip", **kwargs: Any) -> "_FakeClip":
            return _std_levels(clip, **kwargs)

    class _FakeClip:
        def __init__(
            self,
            width: int,
            height: int,
            format_obj: _FakeFormat,
            core: Any,
            *,
            props: Mapping[str, Any] | None = None,
        ) -> None:
            self.width = width
            self.height = height
            self.format = format_obj
            self.core = core
            self.props: dict[str, Any] = dict(props or {})

        def get_frame(self, idx: int) -> Any:
            return types.SimpleNamespace(props=self.props)

        def _with_dimensions(
            self,
            *,
            width: int | None = None,
            height: int | None = None,
            format_obj: _FakeFormat | None = None,
        ) -> "_FakeClip":
            return _FakeClip(
                width if width is not None else self.width,
                height if height is not None else self.height,
                format_obj if format_obj is not None else self.format,
                self.core,
                props=self.props,
            )

        @property
        def std(self) -> _FakeStd:
            return _FakeStd(self)

    class _FakeResize:
        def Point(self, clip: _FakeClip, **kwargs: Any) -> _FakeClip:
            resize_calls.append(kwargs)
            fmt = kwargs.get("format")
            if fmt == fake_vs.YUV444P16:
                promoted_format = _FakeFormat(
                    color_family_obj=yuv_constant,
                    bits_per_sample_val=16,
                    subsampling_w_val=0,
                    subsampling_h_val=0,
                    name_val="YUV444P16",
                )
                new_clip = clip._with_dimensions(format_obj=promoted_format)
                new_clip.props = {}
                return new_clip
            if fmt == fake_vs.RGB24:
                rgb_format = _FakeFormat(
                    color_family_obj=rgb_constant,
                    bits_per_sample_val=8,
                    subsampling_w_val=0,
                    subsampling_h_val=0,
                    name_val="RGB24",
                )
                new_clip = clip._with_dimensions(format_obj=rgb_format)
                new_clip.props = {}
                return new_clip
            default_clip = clip._with_dimensions()
            default_clip.props = {}
            return default_clip

        def Spline36(self, clip: _FakeClip, **kwargs: Any) -> _FakeClip:
            resized = clip._with_dimensions(
                width=kwargs.get("width", clip.width),
                height=kwargs.get("height", clip.height),
            )
            resized.props = {}
            return resized

    class _FakeFpng:
        def Write(self, clip_obj: _FakeClip, path: str, **kwargs: Any) -> Any:
            recorded_kwargs = dict(kwargs)
            recorded_kwargs["props"] = dict(getattr(clip_obj, "props", {}))
            writer_calls.append(recorded_kwargs)

            class _Job:
                def get_frame(self, _index: int) -> None:
                    return None

            Path(path).write_text("png", encoding="utf-8")
            return _Job()

    def _core_add_borders(
        clip: _FakeClip,
        *,
        left: int,
        right: int,
        top: int,
        bottom: int,
    ) -> _FakeClip:
        bordered = clip._with_dimensions(
            width=clip.width + left + right,
            height=clip.height + top + bottom,
        )
        bordered.props = dict(clip.props)
        return bordered

    def _std_copy_frame_props(clip: _FakeClip, src: _FakeClip) -> _FakeClip:
        clip.props = dict(getattr(src, "props", {}))
        return clip

    def _std_set_frame_props(clip: _FakeClip, **kwargs: Any) -> _FakeClip:
        clip.props.update(kwargs)
        return clip

    fake_core = types.SimpleNamespace(
        resize=_FakeResize(),
        fpng=_FakeFpng(),
        std=types.SimpleNamespace(
            AddBorders=_core_add_borders,
            SetFrameProps=_std_set_frame_props,
            CopyFrameProps=_std_copy_frame_props,
            Levels=_std_levels,
        ),
        sub=None,
        text=None,
        _overlay_calls=overlay_calls,
    )
    fake_core.sub = _FakeSub(fake_core)
    fake_core.text = _FakeText(fake_core)

    fake_vs.VideoNode = _FakeClip
    fake_vs.YUV = yuv_constant
    fake_vs.RGB = rgb_constant
    fake_vs.YUV444P16 = "YUV444P16"
    fake_vs.RGB24 = "RGB24"
    fake_vs.RANGE_FULL = 0
    fake_vs.RANGE_LIMITED = 1
    fake_vs.MATRIX_BT709 = 1
    fake_vs.TRANSFER_BT709 = 1
    fake_vs.PRIMARIES_BT709 = 1
    fake_vs.core = fake_core

    initial_color = yuv_constant if color_family.upper() == "YUV" else rgb_constant
    initial_format = _FakeFormat(
        color_family_obj=initial_color,
        bits_per_sample_val=bits_per_sample,
        subsampling_w_val=subsampling_w,
        subsampling_h_val=subsampling_h,
        name_val=format_name or "SyntheticFormat",
    )

    clip = _FakeClip(width, height, initial_format, fake_core)
    clip.props = {"_Matrix": 1, "_ColorRange": 1}

    monkeypatch.setitem(sys.modules, "vapoursynth", fake_vs)

    return clip, fake_vs, writer_calls, resize_calls, levels_calls

@pytest.fixture(autouse=True)
def _stub_process_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace vs_core.process_clip_for_screenshot with a test stub that simulates a processed clip.

    This fixture patches the target function so it returns a types.SimpleNamespace containing:
    - clip: the passed-in clip
    - overlay_text: None
    - verification: None
    - tonemap: a vs_core.TonemapInfo indicating an untonemapped SDR source (applied=False, target_nits=100.0, dst_min_nits=0.18, reason="SDR source")
    - source_props: an empty dict

    Parameters:
        monkeypatch: pytest's monkeypatch fixture used to apply the patch.
    """
    def _stub(
        clip: FakeClip,
        file_name: str,
        color_cfg: ColorConfig,
        **kwargs: object,
    ) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            clip=clip,
            overlay_text=None,
            verification=None,
            tonemap=vs_core.TonemapInfo(
                applied=False,
                tone_curve=None,
                dpd=0,
                target_nits=100.0,
                dst_min_nits=0.18,
                src_csp_hint=None,
                reason="SDR source",
            ),
            source_props={},
        )

    monkeypatch.setattr(vs_core, "process_clip_for_screenshot", _stub)


def test_resolve_source_props_normalises_missing_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_normalise(clip: Any, source_props: Any, **kwargs: Any) -> tuple[str, dict[str, int], tuple[int, int, int, int]]:
        captured["kwargs"] = kwargs
        return "normalized", {"_ColorRange": 1}, (1, 1, 1, 1)

    monkeypatch.setattr(vs_core, "normalise_color_metadata", fake_normalise)

    clip, props = render.resolve_source_props(
        "clip",
        {},
        color_cfg=ColorConfig(),
        file_name="demo.mkv",
    )

    assert clip == "normalized"
    assert props == {"_ColorRange": 1}
    assert captured["kwargs"]["file_name"] == "demo.mkv"


def test_resolve_source_props_skips_normalisation_when_range_present(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_normalise(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        return "normalized", {"_ColorRange": 1}, (1, 1, 1, 1)

    monkeypatch.setattr(vs_core, "normalise_color_metadata", fake_normalise)

    clip, props = render.resolve_source_props(
        "original",
        {"_ColorRange": 1},
        color_cfg=ColorConfig(),
        file_name="demo.mkv",
    )

    assert clip == "original"
    assert props == {"_ColorRange": 1}
    assert called is False


def test_sanitise_label_replaces_forbidden_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_naming.os, "name", "nt")
    raw = 'Group: Episode? 01*<>"| '
    cleaned = sanitise_label(raw)
    assert cleaned
    for forbidden in ':?*<>"|':
        assert forbidden not in cleaned
    assert not cleaned.endswith((" ", "."))


def test_sanitise_label_falls_back_when_blank() -> None:
    cleaned = sanitise_label("   ")
    assert cleaned == "comparison"


def test_plan_mod_crop_modulus() -> None:
    left, top, right, bottom = render.plan_mod_crop(1919, 1079, mod=4, letterbox_pillarbox_aware=True)
    new_w = 1919 - left - right
    new_h = 1079 - top - bottom
    assert new_w % 4 == 0
    assert new_h % 4 == 0
    assert new_w > 0 and new_h > 0


def test_plan_geometry_letterbox_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(3840, 2160), FakeClip(3840, 1800)]
    cfg = ScreenshotConfig(directory_name="screens", add_frame_info=False)
    color_cfg = ColorConfig()

    captured: list[_CapturedWriterCall] = []

    def fake_writer(
        clip: FakeClip,
        frame_idx: int,
        crop: tuple[int, int, int, int],
        scaled: tuple[int, int],
        pad: tuple[int, int, int, int],
        path: Path,
        cfg: ScreenshotConfig,
        label: str,
        requested_frame: int,
        selection_label: str | None = None,
        **kwargs: object,
    ) -> None:
        captured.append(
            {
                "crop": crop,
                "scaled": scaled,
                "pad": pad,
                "label": str(label),
                "requested": int(requested_frame),
                "frame": int(frame_idx),
                "selection_label": selection_label,
                "source": None,
            }
        )
        Path(path).write_text("data", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_writer)
    monkeypatch.setattr(render, "save_frame_with_ffmpeg", lambda *args, **kwargs: None)

    frames: list[int] = [0]
    files: list[str] = ["clip_a.mkv", "clip_b.mkv"]
    metadata: list[dict[str, str]] = [{"label": "Clip A"}, {"label": "Clip B"}]

    orchestrator.generate_screenshots(
        clips,
        frames,
        files,
        metadata,
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0],
    )

    assert len(captured) == 2
    assert captured[0]["crop"] == (0, 180, 0, 180)
    assert captured[0]["scaled"] == (3840, 1800)
    assert captured[1]["crop"] == (0, 0, 0, 0)
    assert captured[1]["scaled"] == (3840, 1800)


def test_generate_screenshots_debug_color_disables_overlays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip, fake_vs, _, _, _ = _prepare_fake_vapoursynth_clip(
        monkeypatch,
        width=1920,
        height=1080,
        subsampling_w=1,
        subsampling_h=1,
        bits_per_sample=8,
        color_family="YUV",
        format_name="YUV420P8",
    )
    monkeypatch.setitem(sys.modules, "vapoursynth", fake_vs)

    cfg = ScreenshotConfig(directory_name="screens", add_frame_info=True, use_ffmpeg=False)
    color_cfg = ColorConfig()
    color_cfg.debug_color = True

    captured: dict[str, Any] = {}

    def fake_writer(
        clip: Any,
        frame_idx: int,
        crop: tuple[int, int, int, int],
        scaled: tuple[int, int],
        pad: tuple[int, int, int, int],
        path: Path,
        cfg: ScreenshotConfig,
        label: str,
        requested_frame: int,
        selection_label: str | None = None,
        *,
        overlay_text: Optional[str] = None,
        debug_state: Any = None,
        frame_info_allowed: bool = True,
        overlays_allowed: bool = True,
        **kwargs: Any,
    ) -> None:
        captured.setdefault("overlay_text", overlay_text)
        captured.setdefault("frame_info_allowed", frame_info_allowed)
        captured.setdefault("overlays_allowed", overlays_allowed)
        Path(path).write_text("data", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_writer)
    monkeypatch.setattr(render, "save_frame_with_ffmpeg", lambda *args, **kwargs: None)

    artifact = vs_core.ColorDebugArtifacts(
        normalized_clip=clip,
        normalized_props={},
        original_props={},
        color_tuple=(1, 1, 1, 1),
    )
    tonemap_info = vs_core.TonemapInfo(
        applied=False,
        tone_curve=None,
        dpd=0,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
        reason="SDR source",
    )

    def fake_process(
        clip_in: Any,
        file_name: str,
        color_cfg_in: ColorConfig,
        **kwargs: Any,
    ) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            clip=clip_in,
            overlay_text="should be suppressed",
            verification=None,
            tonemap=tonemap_info,
            source_props={},
            debug=artifact,
        )

    monkeypatch.setattr(vs_core, "process_clip_for_screenshot", fake_process)

    orchestrator.generate_screenshots(
        [clip],
        [0],
        ["clip.mkv"],
        [{"label": "Debug Clip"}],
        tmp_path,
        cfg,
        color_cfg,
    )

    assert captured["overlay_text"] is None
    assert captured["frame_info_allowed"] is False
    assert captured["overlays_allowed"] is False


def test_generate_screenshots_rehydrates_hdr_props_from_plans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(1920, 1080)
    clip.num_frames = 10
    stored_props = {
        "_MasteringDisplayMinLuminance": 0.005,
        "_MasteringDisplayMaxLuminance": 1500,
    }
    color_cfg = ColorConfig(overlay_mode="diagnostic")
    cfg = ScreenshotConfig(directory_name="shots", use_ffmpeg=False)
    captured: dict[str, Any] = {}

    def fake_writer(
        _clip: Any,
        frame_idx: int,
        crop: tuple[int, int, int, int],
        scaled: tuple[int, int],
        pad: tuple[int, int, int, int],
        path: Path,
        cfg: ScreenshotConfig,
        label: str,
        requested_frame: int,
        selection_label: str | None = None,
        *,
        overlay_text: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        captured["overlay_text"] = overlay_text
        Path(path).write_text("img", encoding="utf-8")

    def fake_process(
        clip_in: Any,
        file_name: str,
        color_cfg: ColorConfig,
        *,
        stored_source_props: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> types.SimpleNamespace:
        assert stored_source_props == stored_props
        captured["processed_source_props"] = stored_source_props
        tonemap_info = vs_core.TonemapInfo(
            applied=True,
            tone_curve="bt.2390",
            dpd=0,
            target_nits=100.0,
            dst_min_nits=0.18,
            src_csp_hint=None,
            reason=None,
        )
        return types.SimpleNamespace(
            clip=clip_in,
            overlay_text=None,
            verification=None,
            tonemap=tonemap_info,
            source_props=dict(stored_source_props or {}),
        )

    def fake_plan_geometry(clips: Sequence[Any], _cfg: ScreenshotConfig) -> list[GeometryPlan]:
        plans: list[GeometryPlan] = []
        for clip_obj in clips:
            width = int(getattr(clip_obj, "width", clip.width))
            height = int(getattr(clip_obj, "height", clip.height))
            plans.append(
                {
                    "width": width,
                    "height": height,
                    "crop": (0, 0, 0, 0),
                    "cropped_w": width,
                    "cropped_h": height,
                    "scaled": (width, height),
                    "pad": (0, 0, 0, 0),
                    "final": (width, height),
                    "requires_full_chroma": False,
                    "promotion_axes": "none",
                }
            )
        return plans

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_writer)
    monkeypatch.setattr(vs_core, "process_clip_for_screenshot", fake_process)
    monkeypatch.setattr(render, "plan_geometry", fake_plan_geometry)

    created = orchestrator.generate_screenshots(
        [clip],
        [0],
        ["clip.mkv"],
        [{"label": "HDR"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[-2],
        source_frame_props=[stored_props],
    )

    assert created
    assert captured["overlay_text"] is not None
    assert "MDL" in captured["overlay_text"]
    assert captured["processed_source_props"] == stored_props


def test_plan_geometry_subsamp_safe_rebalance_aligns_modulus() -> None:
    class _Format:
        def __init__(self, subsampling_w: int, subsampling_h: int) -> None:
            self.subsampling_w = subsampling_w
            self.subsampling_h = subsampling_h

    class _ClipWithFormat:
        def __init__(self, width: int, height: int, subsampling_w: int, subsampling_h: int) -> None:
            self.width = width
            self.height = height
            self.format = _Format(subsampling_w, subsampling_h)

    cfg = ScreenshotConfig(
        directory_name="screens",
        add_frame_info=False,
        odd_geometry_policy=OddGeometryPolicy.SUBSAMP_SAFE,
        pad_to_canvas="on",
        mod_crop=4,
        letterbox_pillarbox_aware=True,
        center_pad=True,
        upscale=False,
    )

    clips: list[_ClipWithFormat] = [
        _ClipWithFormat(1919, 720, 1, 1),
        _ClipWithFormat(1920, 1080, 1, 1),
    ]

    plans = render.plan_geometry(clips, cfg)

    first_plan = plans[0]
    assert first_plan["final"][0] % cfg.mod_crop == 0
    assert first_plan["final"][1] % cfg.mod_crop == 0
    assert not first_plan["requires_full_chroma"]
    assert first_plan["promotion_axes"] == "none"


@pytest.mark.parametrize(
    "policy,sub_w,sub_h,crop,pad,expected",
    [
        (OddGeometryPolicy.AUTO, 1, 1, (0, 1, 0, 0), (0, 0, 0, 0), True),
        (OddGeometryPolicy.AUTO, 1, 0, (0, 1, 0, 0), (0, 0, 0, 0), False),
        (OddGeometryPolicy.AUTO, 1, 0, (1, 0, 0, 0), (0, 0, 0, 0), True),
        (OddGeometryPolicy.AUTO, 1, 1, (0, 0, 0, 0), (0, 0, 0, 0), False),
        (OddGeometryPolicy.FORCE_FULL_CHROMA, 0, 0, (0, 0, 0, 0), (0, 0, 0, 0), True),
        (OddGeometryPolicy.SUBSAMP_SAFE, 1, 1, (0, 1, 0, 0), (0, 0, 0, 0), False),
    ],
)
def test_compute_requires_full_chroma_policy_matrix(
    policy: OddGeometryPolicy,
    sub_w: int,
    sub_h: int,
    crop: tuple[int, int, int, int],
    pad: tuple[int, int, int, int],
    expected: bool,
) -> None:
    fmt = types.SimpleNamespace(subsampling_w=sub_w, subsampling_h=sub_h)
    result = compute_requires_full_chroma(fmt, crop, pad, policy)
    assert result is expected


def test_plan_geometry_aligns_vertical_odds_require_promotion() -> None:
    class _Format:
        def __init__(self, subsampling_w: int, subsampling_h: int) -> None:
            self.subsampling_w = subsampling_w
            self.subsampling_h = subsampling_h

    class _Clip:
        def __init__(self, width: int, height: int, fmt: _Format) -> None:
            self.width = width
            self.height = height
            self.format = fmt

    cfg = ScreenshotConfig(
        odd_geometry_policy=OddGeometryPolicy.AUTO,
        letterbox_pillarbox_aware=True,
        mod_crop=2,
        upscale=False,
    )

    clip_short = _Clip(1920, 1036, _Format(1, 1))
    clip_tall = _Clip(1920, 1038, _Format(1, 1))

    plans = render.plan_geometry([clip_short, clip_tall], cfg)

    assert len(plans) == 2
    assert plans[0]["final"] == plans[1]["final"] == (1920, 1036)
    assert plans[1]["crop"] == (0, 1, 0, 1)
    assert plans[1]["requires_full_chroma"]
    assert not plans[0]["requires_full_chroma"]
    assert plans[1]["promotion_axes"] == "vertical"
    assert plans[0]["promotion_axes"] == "none"


def test_plan_geometry_even_difference_skips_full_chroma() -> None:
    class _Format:
        def __init__(self, subsampling_w: int, subsampling_h: int) -> None:
            self.subsampling_w = subsampling_w
            self.subsampling_h = subsampling_h

    class _Clip:
        def __init__(self, width: int, height: int, fmt: _Format) -> None:
            self.width = width
            self.height = height
            self.format = fmt

    cfg = ScreenshotConfig(
        odd_geometry_policy=OddGeometryPolicy.AUTO,
        letterbox_pillarbox_aware=True,
        mod_crop=2,
        upscale=False,
    )

    clip_short = _Clip(1920, 1036, _Format(1, 1))
    clip_taller = _Clip(1920, 1040, _Format(1, 1))

    plans = render.plan_geometry([clip_short, clip_taller], cfg)

    assert len(plans) == 2
    assert plans[1]["crop"] == (0, 2, 0, 2)
    assert all(not plan["requires_full_chroma"] for plan in plans)
    assert plans[0]["final"] == plans[1]["final"]
    assert all(plan["promotion_axes"] == "none" for plan in plans)


def test_generate_screenshots_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(1280, 720)
    cfg = ScreenshotConfig(directory_name="screens")
    color_cfg = ColorConfig()

    calls: list[_CapturedWriterCall] = []

    def fake_writer(
        clip: FakeClip,
        frame_idx: int,
        crop: tuple[int, int, int, int],
        scaled: tuple[int, int],
        pad: tuple[int, int, int, int],
        path: Path,
        cfg: ScreenshotConfig,
        label: str,
        requested_frame: int,
        selection_label: str | None = None,
        **kwargs: object,
    ) -> None:
        calls.append(
            {
                "frame": int(frame_idx),
                "crop": crop,
                "scaled": scaled,
                "pad": pad,
                "label": str(label),
                "requested": int(requested_frame),
                "selection_label": selection_label,
                "source": None,
            }
        )
        Path(path).write_text("data", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_writer)

    frames: list[int] = [5, 25]
    files: list[str] = ["example_video.mkv"]
    metadata: list[dict[str, str]] = [{"label": "Example Release"}]
    created = orchestrator.generate_screenshots(
        [clip],
        frames,
        files,
        metadata,
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0],
    )
    assert len(created) == len(frames)
    expected_names = {f"{frame} - Example Release.png" for frame in frames}
    assert {Path(path).name for path in created} == expected_names
    for entry in calls:
        assert entry["label"] == "Example Release"
        assert entry["requested"] == entry["frame"]

    assert len(calls) == len(frames)


def test_generate_screenshots_reports_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips = [FakeClip(1280, 720)]
    frames = [0]
    files = ["clip.mkv"]
    metadata = [{}]
    cfg = ScreenshotConfig(directory_name="screens")
    color_cfg = ColorConfig()
    out_dir = tmp_path / "screens"

    path_type = type(out_dir)
    real_mkdir = path_type.mkdir

    def _deny_mkdir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if self == out_dir:
            raise PermissionError("denied")
        real_mkdir(self, mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(path_type, "mkdir", _deny_mkdir)

    with pytest.raises(ScreenshotError) as excinfo:
        orchestrator.generate_screenshots(
            clips,
            frames,
            files,
            metadata,
            out_dir,
            cfg,
            color_cfg,
        )

    assert "Unable to create screenshot directory" in str(excinfo.value)


def _make_plan(
    *,
    width: int = 1920,
    height: int = 1080,
    crop: tuple[int, int, int, int] = (0, 0, 0, 0),
    cropped_w: int = 1920,
    cropped_h: int = 1080,
    scaled: tuple[int, int] = (1920, 1080),
    pad: tuple[int, int, int, int] = (0, 0, 0, 0),
    final: tuple[int, int] = (1920, 1080),
    requires_full_chroma: bool = False,
    promotion_axes: str = "none",
) -> GeometryPlan:
    """
    Builds a rendering plan dictionary describing dimensions, crop, scaling, padding, and final output size.

    Returns:
        dict: A plan mapping with the following keys:
            - "width": source frame width.
            - "height": source frame height.
            - "crop": 4-tuple (left, top, right, bottom) representing pixel crop offsets.
            - "cropped_w": width after cropping.
            - "cropped_h": height after cropping.
            - "scaled": 2-tuple (width, height) after scaling.
            - "pad": 4-tuple (left, top, right, bottom) of pixels added as padding.
            - "final": 2-tuple (width, height) of the final output frame.
            - "promotion_axes": Label describing which axes triggered 4:4:4 promotion.
    """
    plan = cast(
        GeometryPlan,
        {
            "width": width,
            "height": height,
            "crop": crop,
            "cropped_w": cropped_w,
            "cropped_h": cropped_h,
            "scaled": scaled,
            "pad": pad,
            "final": final,
            "requires_full_chroma": requires_full_chroma,
            "promotion_axes": promotion_axes,
        },
    )
    return plan


def test_compose_overlay_text_minimal_adds_resolution_and_selection() -> None:
    color_cfg = ColorConfig(overlay_mode="minimal")
    plan = _make_plan()
    base_text = "Tonemapping Algorithm: bt.2390 dpd = 1 dst = 100 nits"
    tonemap_info = vs_core.TonemapInfo(
        applied=True,
        tone_curve="bt.2390",
        dpd=1,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
    )

    composed = render.compose_overlay_text(
        base_text,
        color_cfg,
        plan,
        selection_label="Dark",
        source_props={},
        tonemap_info=tonemap_info,
    )

    assert composed is not None
    lines = composed.split("\n")
    assert lines[0] == base_text
    assert lines[1] == "1920 × 1080  (native)"
    assert lines[2] == "Frame Selection Type: Dark"


def test_compose_overlay_text_minimal_handles_missing_base_and_label() -> None:
    color_cfg = ColorConfig(overlay_mode="minimal")
    plan = _make_plan()

    composed = render.compose_overlay_text(
        base_text=None,
        color_cfg=color_cfg,
        plan=plan,
        selection_label=None,
        source_props={},
        tonemap_info=None,
    )

    assert composed is not None
    lines = composed.split("\n")
    assert lines[0] == "1920 × 1080  (native)"
    assert lines[1] == "Frame Selection Type: (unknown)"


def test_compose_overlay_text_minimal_ignores_hdr_details() -> None:
    color_cfg = ColorConfig(overlay_mode="minimal")
    plan = _make_plan()
    props = {
        "_MasteringDisplayMinLuminance": 0.0001,
        "_MasteringDisplayMaxLuminance": 1000.0,
    }
    tonemap_info = vs_core.TonemapInfo(
        applied=True,
        tone_curve="bt.2390",
        dpd=1,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
    )

    composed = render.compose_overlay_text(
        base_text=None,
        color_cfg=color_cfg,
        plan=plan,
        selection_label="Dark",
        source_props=props,
        tonemap_info=tonemap_info,
    )

    assert composed is not None
    assert "MDL:" not in composed
    assert composed.endswith("Frame Selection Type: Dark")


def test_compose_overlay_text_diagnostic_appends_required_lines() -> None:
    color_cfg = ColorConfig(overlay_mode="diagnostic")
    plan = _make_plan(
        scaled=(3840, 2160),
        final=(3840, 2160),
    )
    base_text = "Tonemapping Algorithm: bt.2390 dpd = 1 dst = 100 nits"
    tonemap_info = vs_core.TonemapInfo(
        applied=True,
        tone_curve="bt.2390",
        dpd=1,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
        use_dovi=True,
    )
    props = {
        "_MasteringDisplayMinLuminance": 0.0001,
        "_MasteringDisplayMaxLuminance": 1000.0,
        "DolbyVision_BlockIndex": 2,
        "DolbyVision_BlockCount": 8,
        "DolbyVision_TargetNits": 400.0,
        "ContentLightLevelMax": 1200,
        "MaxFALL": 400,
        "_ColorRange": 1,
    }
    selection_detail = {
        "diagnostics": {
            "frame_metrics": {"avg_nits": 45.0, "max_nits": 50.0, "category": "Bright"}
        }
    }

    composed = render.compose_overlay_text(
        base_text,
        color_cfg,
        plan,
        selection_label="Dark",
        source_props=props,
        tonemap_info=tonemap_info,
        selection_detail=selection_detail,
    )

    assert composed is not None
    lines = composed.split("\n")
    assert lines[0] == base_text
    assert lines[1] == "1920 × 1080 → 3840 × 2160  (original → target)"
    assert lines[2] == "MDL: min: 0.0001 cd/m², max: 1000.0 cd/m²"
    assert lines[3] == "HDR: MaxCLL 1200 / MaxFALL 400"
    assert lines[4] == "DoVi: on L2 2/8 target 400 nits"
    assert lines[5] == "Range: Limited"
    assert lines[6] == "Measurement MAX/AVG: 50nits / 45nits (Bright)"
    assert lines[6] == "Measurement MAX/AVG: 50nits / 45nits (Bright)"


def test_compose_overlay_text_skips_hdr_details_for_sdr() -> None:
    color_cfg = ColorConfig(overlay_mode="diagnostic")
    plan = _make_plan()
    base_text = "Tonemapping Algorithm: bt.2390 dpd = 1 dst = 100 nits"
    tonemap_info = vs_core.TonemapInfo(
        applied=False,
        tone_curve=None,
        dpd=0,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
        reason="SDR source",
    )

    composed = render.compose_overlay_text(
        base_text,
        color_cfg,
        plan,
        selection_label="Cached",
        source_props={},
        tonemap_info=tonemap_info,
    )

    assert composed is not None
    lines = composed.split("\n")
    assert "MDL:" not in composed
    assert "Measurement" not in composed
    assert "Measurement" not in composed


def test_compression_flag_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(1920, 1080)
    cfg = ScreenshotConfig(use_ffmpeg=True, compression_level=2)
    color_cfg = ColorConfig()

    captured: dict[int, int] = {}

    def fake_writer(
        source: Path,
        frame_idx: int,
        crop: tuple[int, int, int, int],
        scaled: tuple[int, int],
        pad: tuple[int, int, int, int],
        path: Path,
        cfg: ScreenshotConfig,
        width: int,
        height: int,
        selection_label: str | None,
        *,
        overlay_text: str | None = None,
        is_sdr: bool = True,
        **_kwargs: Any,
    ) -> None:
        captured[frame_idx] = render.map_ffmpeg_compression(cfg.compression_level)
        path.write_text("ffmpeg", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_ffmpeg", fake_writer)

    orchestrator.generate_screenshots(
        [clip],
        [10],
        ["video.mkv"],
        [{"label": "video"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0],
    )
    assert captured[10] == 9


def test_ffmpeg_respects_trim_offsets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(1920, 1080)
    cfg = ScreenshotConfig(use_ffmpeg=True)
    color_cfg = ColorConfig()

    calls: list[int] = []

    def fake_ffmpeg(
        source,
        frame_idx,
        crop,
        scaled,
        pad,
        path,
        cfg,
        width,
        height,
        selection_label,
        *,
        overlay_text=None,
        is_sdr: bool = True,
        **_kwargs: Any,
    ):
        calls.append(frame_idx)
        Path(path).write_text("ff", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(render, "save_frame_with_fpng", lambda *args, **kwargs: None)

    orchestrator.generate_screenshots(
        [clip],
        [0, 5],
        ["video.mkv"],
        [{"label": "video"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[3],
    )

    assert calls == [3, 8]


def test_global_upscale_coordination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(1280, 720), FakeClip(1920, 1080), FakeClip(640, 480)]
    cfg = ScreenshotConfig(upscale=True, use_ffmpeg=False, add_frame_info=False)
    color_cfg = ColorConfig()

    scaled: list[tuple[int, int]] = []

    def fake_vs_writer(
        clip,
        frame_idx,
        crop,
        scaled_dims,
        pad,
        path,
        cfg,
        label,
        requested_frame,
        selection_label=None,
        **kwargs,
    ):
        scaled.append((scaled_dims, pad))
        Path(path).write_text("vs", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_vs_writer)
    monkeypatch.setattr(render, "save_frame_with_ffmpeg", lambda *args, **kwargs: None)

    metadata: list[dict[str, str]] = [{"label": f"clip{i}"} for i in range(len(clips))]
    orchestrator.generate_screenshots(
        clips,
        [0],
        [f"clip{i}.mkv" for i in range(len(clips))],
        metadata,
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0, 0],
    )

    assert [dims for dims, _ in scaled] == [(1920, 1080), (1920, 1080), (1440, 1080)]
    assert all(pad == (0, 0, 0, 0) for _, pad in scaled)


def test_upscale_clamps_letterbox_width(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(3840, 2160), FakeClip(3832, 1384)]
    cfg = ScreenshotConfig(upscale=True, use_ffmpeg=False, add_frame_info=False)
    color_cfg = ColorConfig()

    recorded: list[tuple[int, int]] = []

    def fake_vs_writer(
        clip,
        frame_idx,
        crop,
        scaled_dims,
        pad,
        path,
        cfg,
        label,
        requested_frame,
        selection_label=None,
        **kwargs,
    ):
        recorded.append((scaled_dims, pad))
        Path(path).write_text("vs", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_vs_writer)

    metadata: list[dict[str, str]] = [{"label": "hdr"}, {"label": "scope"}]
    orchestrator.generate_screenshots(
        clips,
        [0],
        ["hdr.mkv", "scope.mkv"],
        metadata,
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0],
    )

    assert recorded[0][0] == (3840, 2160)
    assert recorded[0][1] == (0, 0, 0, 0)
    expected_height = int(round(1384 * (3840 / 3832)))
    assert recorded[1][0] == (3840, expected_height)
    assert recorded[1][1] == (0, 0, 0, 0)


def test_auto_letterbox_crop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(3840, 2160), FakeClip(3832, 1384)]
    cfg = ScreenshotConfig(
        upscale=False,
        use_ffmpeg=False,
        add_frame_info=False,
        auto_letterbox_crop=True,
    )
    color_cfg = ColorConfig()

    captured: list[_CapturedWriterCall] = []

    def fake_vs_writer(
        clip,
        frame_idx,
        crop,
        scaled_dims,
        pad,
        path,
        cfg,
        label,
        requested_frame,
        selection_label=None,
        **kwargs,
    ):
        captured.append(
            {
                "crop": crop,
                "scaled": scaled_dims,
                "pad": pad,
                "label": str(label),
                "requested": int(requested_frame),
                "frame": int(frame_idx),
                "selection_label": selection_label,
                "source": None,
            }
        )
        Path(path).write_text("vs", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_vs_writer)

    orchestrator.generate_screenshots(
        clips,
        [0],
        ["bars.mkv", "scope.mkv"],
        [{"label": "bars"}, {"label": "scope"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0],
    )

    assert len(captured) == 2
    first_crop = captured[0]["crop"]
    assert isinstance(first_crop, tuple)
    assert first_crop[1] > 0 and first_crop[3] > 0
    assert captured[0]["scaled"] == (3840, 1384)
    assert captured[0]["pad"] == (0, 0, 0, 0)
    assert captured[1]["scaled"] == (3832, 1384)
    assert captured[1]["pad"] == (0, 0, 0, 0)


def test_auto_letterbox_basic_avoids_weird_scope_pair() -> None:
    clips = [FakeClip(3840, 2160), FakeClip(3600, 2160)]
    cfg = ScreenshotConfig(
        upscale=True,
        use_ffmpeg=False,
        add_frame_info=False,
        mod_crop=2,
        letterbox_pillarbox_aware=True,
        auto_letterbox_crop="basic",
    )

    plans = render.plan_geometry(clips, cfg)
    assert len(plans) == 2

    wider, narrower = plans
    assert wider["cropped_w"] == 3600
    assert sum(wider["crop"][::2]) == 240
    assert wider["crop"][1] == 0 and wider["crop"][3] == 0
    assert narrower["crop"][1] == 0 and narrower["crop"][3] == 0
    assert wider["cropped_h"] == 2160
    assert narrower["cropped_h"] == 2160
    assert wider["final"] == narrower["final"]


def test_auto_letterbox_strict_preserves_legacy_scope_behavior() -> None:
    clips = [FakeClip(3840, 2160), FakeClip(3600, 2160)]
    cfg = ScreenshotConfig(
        upscale=True,
        use_ffmpeg=False,
        add_frame_info=False,
        mod_crop=2,
        letterbox_pillarbox_aware=True,
        auto_letterbox_crop="strict",
    )

    plans = render.plan_geometry(clips, cfg)
    assert len(plans) == 2

    wider, narrower = plans
    assert wider["crop"] == (120, 0, 120, 0)
    assert wider["cropped_w"] == 3600
    assert narrower["crop"][1] > 0 or narrower["crop"][3] > 0
    assert narrower["cropped_h"] < 2160


def test_pad_to_canvas_auto_handles_micro_bars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(3840, 2152), FakeClip(1920, 1080)]
    cfg = ScreenshotConfig(
        upscale=True,
        use_ffmpeg=False,
        add_frame_info=False,
        single_res=2160,
        pad_to_canvas="auto",
        letterbox_px_tolerance=8,
    )
    color_cfg = ColorConfig()

    captured: list[_CapturedWriterCall] = []

    def fake_vs_writer(
        clip,
        frame_idx,
        crop,
        scaled_dims,
        pad,
        path,
        cfg,
        label,
        requested_frame,
        selection_label=None,
        **kwargs,
    ):
        captured.append(
            {
                "crop": crop,
                "scaled": scaled_dims,
                "pad": pad,
                "label": str(label),
                "requested": int(requested_frame),
                "frame": int(frame_idx),
                "selection_label": selection_label,
                "source": None,
            }
        )
        Path(path).write_text("vs", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_vs_writer)

    orchestrator.generate_screenshots(
        clips,
        [0],
        ["uhd.mkv", "hd.mkv"],
        [{"label": "UHD"}, {"label": "HD"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0],
    )

    by_label = {entry["label"]: entry for entry in captured}
    assert by_label["UHD"]["scaled"] == (3840, 2152)
    assert by_label["UHD"]["pad"] == (0, 4, 0, 4)
    assert by_label["HD"]["scaled"] == (3840, 2160)
    assert by_label["HD"]["pad"] == (0, 0, 0, 0)


def test_pad_to_canvas_auto_respects_tolerance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(3840, 2048), FakeClip(1920, 1080)]
    cfg = ScreenshotConfig(
        upscale=True,
        use_ffmpeg=False,
        add_frame_info=False,
        single_res=2160,
        pad_to_canvas="auto",
        letterbox_px_tolerance=8,
    )
    color_cfg = ColorConfig()

    captured: list[_CapturedWriterCall] = []

    def fake_vs_writer(
        clip,
        frame_idx,
        crop,
        scaled_dims,
        pad,
        path,
        cfg,
        label,
        requested_frame,
        selection_label=None,
        **kwargs,
    ):
        captured.append(
            {
                "crop": crop,
                "scaled": scaled_dims,
                "pad": pad,
                "label": str(label),
                "requested": int(requested_frame),
                "frame": int(frame_idx),
                "selection_label": selection_label,
                "source": None,
            }
        )
        Path(path).write_text("vs", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_vs_writer)

    orchestrator.generate_screenshots(
        clips,
        [0],
        ["scope.mkv", "hd.mkv"],
        [{"label": "scope"}, {"label": "hd"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0],
    )

    by_label = {entry["label"]: entry for entry in captured}
    assert by_label["scope"]["pad"] == (0, 0, 0, 0)
    assert by_label["scope"]["scaled"][1] == 2048
    assert by_label["hd"]["pad"] == (0, 0, 0, 0)


def test_pad_to_canvas_on_pillarboxes_narrow_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(1920, 1080), FakeClip(1440, 1080)]
    cfg = ScreenshotConfig(
        upscale=False,
        use_ffmpeg=False,
        add_frame_info=False,
        single_res=1080,
        pad_to_canvas="on",
        letterbox_pillarbox_aware=False,
    )
    color_cfg = ColorConfig()

    captured: list[_CapturedWriterCall] = []

    def fake_vs_writer(
        clip,
        frame_idx,
        crop,
        scaled_dims,
        pad,
        path,
        cfg,
        label,
        requested_frame,
        selection_label=None,
        **kwargs,
    ):
        captured.append(
            {
                "crop": crop,
                "scaled": scaled_dims,
                "pad": pad,
                "label": str(label),
                "requested": int(requested_frame),
                "frame": int(frame_idx),
                "selection_label": selection_label,
                "source": None,
            }
        )
        Path(path).write_text("vs", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_vs_writer)

    orchestrator.generate_screenshots(
        clips,
        [0],
        ["widescreen.mkv", "academy.mkv"],
        [{"label": "ws"}, {"label": "academy"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0],
    )

    by_label = {entry["label"]: entry for entry in captured}
    assert by_label["ws"]["pad"] == (0, 0, 0, 0)
    assert by_label["ws"]["scaled"] == (1920, 1080)
    assert by_label["academy"]["scaled"] == (1440, 1080)
    assert by_label["academy"]["pad"] == (240, 0, 240, 0)


def test_pad_to_canvas_on_without_single_res(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(1920, 1080), FakeClip(1440, 1080)]
    cfg = ScreenshotConfig(
        upscale=False,
        use_ffmpeg=False,
        add_frame_info=False,
        single_res=0,
        pad_to_canvas="on",
        letterbox_pillarbox_aware=False,
    )
    color_cfg = ColorConfig()

    captured: list[_CapturedWriterCall] = []

    def fake_vs_writer(
        clip,
        frame_idx,
        crop,
        scaled_dims,
        pad,
        path,
        cfg,
        label,
        requested_frame,
        selection_label=None,
        **kwargs,
    ):
        captured.append(
            {
                "crop": crop,
                "scaled": scaled_dims,
                "pad": pad,
                "label": str(label),
                "requested": int(requested_frame),
                "frame": int(frame_idx),
                "selection_label": selection_label,
                "source": None,
            }
        )
        Path(path).write_text("vs", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_vs_writer)

    orchestrator.generate_screenshots(
        clips,
        [0],
        ["wide.mkv", "narrow.mkv"],
        [{"label": "wide"}, {"label": "narrow"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0],
    )

    by_label = {entry["label"]: entry for entry in captured}
    assert by_label["wide"]["scaled"] == (1920, 1080)
    assert by_label["wide"]["pad"] == (0, 0, 0, 0)
    assert by_label["narrow"]["scaled"] == (1440, 1080)
    assert by_label["narrow"]["pad"] == (240, 0, 240, 0)


def test_pad_to_canvas_auto_zero_tolerance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(3840, 2152), FakeClip(1920, 1080)]
    cfg = ScreenshotConfig(
        upscale=True,
        use_ffmpeg=False,
        add_frame_info=False,
        single_res=2160,
        pad_to_canvas="auto",
        letterbox_px_tolerance=0,
    )
    color_cfg = ColorConfig()

    captured: list[_CapturedWriterCall] = []

    def fake_vs_writer(
        clip,
        frame_idx,
        crop,
        scaled_dims,
        pad,
        path,
        cfg,
        label,
        requested_frame,
        selection_label=None,
        **kwargs,
    ):
        captured.append(
            {
                "crop": crop,
                "scaled": scaled_dims,
                "pad": pad,
                "label": str(label),
                "requested": int(requested_frame),
                "frame": int(frame_idx),
                "selection_label": selection_label,
                "source": None,
            }
        )
        Path(path).write_text("vs", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_fpng", fake_vs_writer)

    orchestrator.generate_screenshots(
        clips,
        [0],
        ["uhd.mkv", "hd.mkv"],
        [{"label": "UHD"}, {"label": "HD"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0],
    )

    by_label = {entry["label"]: entry for entry in captured}
    assert by_label["UHD"]["scaled"] == (3840, 2152)
    assert by_label["UHD"]["pad"] == (0, 0, 0, 0)
    assert by_label["HD"]["scaled"] == (3840, 2160)
    assert by_label["HD"]["pad"] == (0, 0, 0, 0)


def test_ffmpeg_writer_receives_padding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [FakeClip(1920, 1080), FakeClip(1440, 1080)]
    cfg = ScreenshotConfig(
        use_ffmpeg=True,
        add_frame_info=False,
        single_res=1080,
        pad_to_canvas="on",
        letterbox_pillarbox_aware=False,
        mod_crop=2,
    )
    color_cfg = ColorConfig()

    calls: list[_CapturedWriterCall] = []

    def fake_ffmpeg(
        source,
        frame_idx,
        crop,
        scaled,
        pad,
        path,
        cfg,
        width,
        height,
        selection_label,
        *,
        overlay_text=None,
        is_sdr: bool = True,
        **_kwargs: Any,
    ):
        calls.append(
            {
                "crop": crop,
                "scaled": scaled,
                "pad": pad,
                "label": "ffmpeg",
                "requested": int(frame_idx),
                "frame": int(frame_idx),
                "selection_label": selection_label,
                "source": str(source),
            }
        )
        Path(path).write_text("ff", encoding="utf-8")

    monkeypatch.setattr(render, "save_frame_with_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(render, "save_frame_with_fpng", lambda *args, **kwargs: None)

    orchestrator.generate_screenshots(
        clips,
        [0],
        ["wide.mkv", "narrow.mkv"],
        [{"label": "wide"}, {"label": "narrow"}],
        tmp_path,
        cfg,
        color_cfg,
        trim_offsets=[0, 0],
    )

    assert len(calls) == 2
    wide_call = next(call for call in calls if call["source"] is not None and call["source"].endswith("wide.mkv"))
    narrow_call = next(call for call in calls if call["source"] is not None and call["source"].endswith("narrow.mkv"))
    assert wide_call["source"] is not None
    assert narrow_call["source"] is not None
    assert wide_call["scaled"] == (1920, 1080)
    assert wide_call["pad"] == (0, 0, 0, 0)
    assert narrow_call["scaled"] == (1440, 1080)
    assert narrow_call["pad"] == (240, 0, 240, 0)
def test_placeholder_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = FakeClip(1280, 720)
    cfg = ScreenshotConfig(use_ffmpeg=False)
    color_cfg = ColorConfig()

    def failing_writer(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(render, "save_frame_with_fpng", failing_writer)
    monkeypatch.setattr(render, "save_frame_with_ffmpeg", lambda *args, **kwargs: None)

    with caplog.at_level("WARNING"):
        created = orchestrator.generate_screenshots(
            [clip],
            [0],
            ["clip.mkv"],
            [{"label": "clip"}],
            tmp_path,
            cfg,
            color_cfg,
            trim_offsets=[0],
        )

    assert any("Falling back to placeholder" in message for message in caplog.messages)
    placeholder = Path(created[0])
    assert placeholder.read_bytes() == b"placeholder\n"


def test_compose_overlay_text_omits_selection_detail_lines() -> None:
    color_cfg = ColorConfig(overlay_mode="diagnostic")
    plan = _make_plan()
    base_text = "Tonemapping Algorithm: bt.2390 dpd = 1 dst = 100 nits"
    selection_detail = {"timecode": "00:00:05.000", "score": 0.42, "notes": "motion"}
    composed = render.compose_overlay_text(
        base_text,
        color_cfg,
        plan,
        selection_label="Motion",
        source_props={},
        tonemap_info=None,
        selection_detail=selection_detail,
    )
    assert composed is not None
    assert "Selection Timecode" not in composed
    assert "Selection Score" not in composed
    assert "Selection Notes" not in composed
    assert "Selection Notes" not in composed


def test_overlay_state_warning_helpers_roundtrip() -> None:
    state = render.new_overlay_state()
    render.append_overlay_warning(state, "first")
    render.append_overlay_warning(state, "second")
    assert render.get_overlay_warnings(state) == ["first", "second"]


def test_apply_frame_info_overlay_preserves_metadata() -> None:
    clip = types.SimpleNamespace(num_frames=10, props={"_ColorRange": 1})

    class DummyStd:
        def __init__(self) -> None:
            self.copy_invocations = 0

        def FrameEval(self, clip_ref: Any, func: Any, *, prop_src: Any = None) -> Any:
            frame = types.SimpleNamespace(props={"_PictType": b"I"})
            return func(0, frame, clip_ref)

        def CopyFrameProps(self, target: Any, source: Any) -> Any:
            self.copy_invocations += 1
            target.props = dict(getattr(source, "props", {}))
            return target

    class DummySub:
        def Subtitle(self, clip_ref: Any, *, text: Sequence[str], style: str) -> Any:
            return types.SimpleNamespace(props={})

    core = types.SimpleNamespace(std=DummyStd(), sub=DummySub())
    result = render.apply_frame_info_overlay(
        core,
        clip,
        title="Test Clip",
        requested_frame=None,
        selection_label="dark",
    )
    assert getattr(result, "props", {}).get("_ColorRange") == 1
    assert core.std.copy_invocations > 0


def test_apply_frame_info_overlay_includes_selection_label() -> None:
    clip = types.SimpleNamespace(num_frames=10, props={"_ColorRange": 1})

    class DummyStd:
        def FrameEval(self, clip_ref: Any, func: Any, *, prop_src: Any = None) -> Any:
            frame = types.SimpleNamespace(props={"_PictType": b"I"})
            return func(0, frame, clip_ref)

        def CopyFrameProps(self, target: Any, source: Any) -> Any:
            target.props = dict(getattr(source, "props", {}))
            return target

    class DummySub:
        def __init__(self) -> None:
            self.calls: list[Sequence[str]] = []

        def Subtitle(self, clip_ref: Any, *, text: Sequence[str], style: str) -> Any:
            self.calls.append(text)
            return types.SimpleNamespace(props={})

    dummy_sub = DummySub()
    core = types.SimpleNamespace(std=DummyStd(), sub=dummy_sub)
    render.apply_frame_info_overlay(
        core,
        clip,
        title="Test Clip",
        requested_frame=None,
        selection_label="SelectionA",
    )
    assert dummy_sub.calls, "Subtitle should be invoked"
    joined_calls = ["\n".join(call) for call in dummy_sub.calls]
    assert any("Test Clip" in call for call in joined_calls)
    assert any("Selection: SelectionA" in call for call in joined_calls)


def test_apply_overlay_text_subtitle_path_preserves_metadata() -> None:
    clip = types.SimpleNamespace(props={"_ColorRange": 1})

    class DummyStd:
        def __init__(self) -> None:
            self.copy_invocations = 0

        def CopyFrameProps(self, target: Any, source: Any) -> Any:
            self.copy_invocations += 1
            target.props = dict(getattr(source, "props", {}))
            return target

    class DummySub:
        def Subtitle(self, clip_ref: Any, *, text: Sequence[str], style: str) -> Any:
            return types.SimpleNamespace(props={})

    core = types.SimpleNamespace(std=DummyStd(), sub=DummySub())
    state: Dict[str, Any] = {}

    result = render.apply_overlay_text(
        core,
        clip,
        text="Overlay",
        strict=False,
        state=state,
        file_label="clip",
    )
    assert getattr(result, "props", {}).get("_ColorRange") == 1
    assert core.std.copy_invocations > 0


def test_save_frame_with_ffmpeg_honours_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = ScreenshotConfig(ffmpeg_timeout_seconds=37.5)
    recorded: dict[str, object] = {}

    def fake_run(cmd: Sequence[str], **kwargs: Any):  # type: ignore[override]
        recorded.update(kwargs)
        recorded["cmd"] = cmd

        class _Result:
            returncode = 0
            stderr = b""

        return _Result()

    monkeypatch.setattr(render.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(fc_subproc, "run_checked", fake_run)

    render.save_frame_with_ffmpeg(
        source="video.mkv",
        frame_idx=12,
        crop=(0, 0, 0, 0),
        scaled=(1920, 1080),
        pad=(0, 0, 0, 0),
        path=tmp_path / "frame.png",
        cfg=cfg,
        width=1920,
        height=1080,
        selection_label=None,
        frame_info_allowed=True,
        overlays_allowed=True,
    )

    cmd = recorded.get("cmd")
    assert isinstance(cmd, list)
    assert "-nostdin" in cmd
    assert recorded.get("stdin") is subprocess.DEVNULL
    assert recorded.get("stdout") is subprocess.DEVNULL
    assert recorded.get("stderr") is subprocess.PIPE
    assert recorded.get("timeout") == pytest.approx(37.5)


def test_save_frame_with_ffmpeg_disables_timeout_when_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = ScreenshotConfig(ffmpeg_timeout_seconds=0.0)
    recorded: dict[str, object] = {}

    def fake_run(cmd, **kwargs):  # type: ignore[override]
        recorded.update(kwargs)

        class _Result:
            returncode = 0
            stderr = b""

        return _Result()

    monkeypatch.setattr(render.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(fc_subproc, "run_checked", fake_run)

    render.save_frame_with_ffmpeg(
        source="video.mkv",
        frame_idx=3,
        crop=(0, 0, 0, 0),
        scaled=(1920, 1080),
        pad=(0, 0, 0, 0),
        path=tmp_path / "frame.png",
        cfg=cfg,
        width=1920,
        height=1080,
        selection_label=None,
        frame_info_allowed=True,
        overlays_allowed=True,
    )

    assert "timeout" not in recorded or recorded.get("timeout") is None


def test_save_frame_with_ffmpeg_inserts_full_chroma_filters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = ScreenshotConfig(rgb_dither=RGBDither.ERROR_DIFFUSION)
    plan = _make_plan(
        crop=(1, 0, 2, 0),
        pad=(0, 1, 0, 1),
        requires_full_chroma=True,
        promotion_axes="horizontal+vertical",
    )
    recorded_cmd: list[str] = []
    pivot_notes: list[str] = []

    def fake_run(cmd: Sequence[str], **_kwargs: Any):  # type: ignore[override]
        recorded_cmd[:] = list(cmd)

        class _Result:
            returncode = 0
            stderr = b""

        return _Result()

    monkeypatch.setattr(render.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(fc_subproc, "run_checked", fake_run)

    render.save_frame_with_ffmpeg(
        source="video.mkv",
        frame_idx=7,
        crop=(1, 0, 2, 0),
        scaled=(1917, 1080),
        pad=(0, 1, 0, 1),
        path=tmp_path / "frame.png",
        cfg=cfg,
        width=1920,
        height=1080,
        selection_label=None,
        geometry_plan=plan,
        pivot_notifier=pivot_notes.append,
        frame_info_allowed=True,
        overlays_allowed=True,
        target_range=0,
    )

    assert recorded_cmd
    vf_index = recorded_cmd.index("-vf")
    filter_chain = recorded_cmd[vf_index + 1]
    filters = filter_chain.split(",")
    assert "format=yuv444p16" in filters
    assert any(entry.startswith("format=rgb24") for entry in filters)
    assert filters[-1].endswith("dither=ordered")
    assert any("Full-chroma pivot" in note for note in pivot_notes)
    assert "-color_range" in recorded_cmd
    range_flag_index = recorded_cmd.index("-color_range")
    assert recorded_cmd[range_flag_index + 1] == "pc"


def test_ffmpeg_expands_limited_range_when_exporting_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = ScreenshotConfig()
    cfg.export_range = ExportRange.FULL
    plan = _make_plan(
        pad=(0, 0, 0, 0),
        requires_full_chroma=False,
        promotion_axes="none",
    )
    recorded_cmd: list[str] = []

    def fake_run(cmd: Sequence[str], **_kwargs: Any):  # type: ignore[override]
        recorded_cmd[:] = list(cmd)

        class _Result:
            returncode = 0
            stderr = b""

        return _Result()

    monkeypatch.setattr(render.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(fc_subproc, "run_checked", fake_run)

    render.save_frame_with_ffmpeg(
        source="video.mkv",
        frame_idx=11,
        crop=(0, 0, 0, 0),
        scaled=(1920, 1080),
        pad=(0, 0, 0, 0),
        path=tmp_path / "frame.png",
        cfg=cfg,
        width=1920,
        height=1080,
        selection_label=None,
        geometry_plan=plan,
        pivot_notifier=None,
        frame_info_allowed=False,
        overlays_allowed=False,
        target_range=1,
        expand_to_full=True,
        source_color_range=1,
    )

    assert recorded_cmd
    vf_index = recorded_cmd.index("-vf")
    filters = recorded_cmd[vf_index + 1].split(",")
    assert "scale=in_range=tv:out_range=pc" in filters
    assert "-color_range" in recorded_cmd
    flag_idx = recorded_cmd.index("-color_range")
    assert recorded_cmd[flag_idx + 1] == "pc"


def test_save_frame_with_ffmpeg_raises_on_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = ScreenshotConfig(ffmpeg_timeout_seconds=5.0)

    def fake_run(*args: object, **kwargs: Any):  # type: ignore[override]
        timeout_value = float(kwargs.get("timeout", 0.0) or 0.0)
        cmd_arg = cast(Any, args[0])
        raise subprocess.TimeoutExpired(cmd=cmd_arg, timeout=timeout_value)

    monkeypatch.setattr(render.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(fc_subproc, "run_checked", fake_run)

    with pytest.raises(ScreenshotWriterError) as exc_info:
        render.save_frame_with_ffmpeg(
            source="video.mkv",
            frame_idx=99,
            crop=(0, 0, 0, 0),
            scaled=(1280, 720),
            pad=(0, 0, 0, 0),
            path=tmp_path / "frame.png",
            cfg=cfg,
            width=1280,
            height=720,
            selection_label=None,
            frame_info_allowed=True,
            overlays_allowed=True,
        )

    assert "timed out" in str(exc_info.value)


def test_save_frame_with_fpng_promotes_subsampled_sdr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clip, fake_vs, writer_calls, resize_calls, _ = _prepare_fake_vapoursynth_clip(
        monkeypatch,
        width=1920,
        height=1080,
        subsampling_w=1,
        subsampling_h=1,
        bits_per_sample=10,
        format_name="YUV420P10",
    )

    cfg = ScreenshotConfig(add_frame_info=False, rgb_dither=RGBDither.ORDERED)
    tonemap_info = vs_core.TonemapInfo(
        applied=False,
        tone_curve=None,
        dpd=0,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
        reason="SDR source",
    )
    plan = _make_plan(
        pad=(0, 1, 0, 1),
        requires_full_chroma=True,
        promotion_axes="vertical",
    )
    source_props = {"_Matrix": 1, "_Transfer": 1, "_Primaries": 1, "_ColorRange": 1}

    caplog_any = cast(Any, caplog)
    caplog_any.set_level(logging.INFO)

    pivot_notes: list[str] = []

    render.save_frame_with_fpng(
        clip,
        frame_idx=0,
        crop=plan["crop"],
        scaled=plan["scaled"],
        pad=plan["pad"],
        path=tmp_path / "frame.png",
        cfg=cfg,
        label="Clip",
        requested_frame=0,
        selection_label=None,
        source_props=source_props,
        geometry_plan=plan,
        tonemap_info=tonemap_info,
        pivot_notifier=pivot_notes.append,
        debug_state=None,
        frame_info_allowed=False,
        overlays_allowed=False,
    )

    log_records: Sequence[logging.LogRecord] = list(caplog_any.records)
    assert any("promoting to YUV444P16" in record.getMessage() for record in log_records)
    assert writer_calls, "fpng writer should be invoked"
    assert len(resize_calls) >= 2
    first_call, second_call = resize_calls[0], resize_calls[-1]
    assert first_call.get("format") == fake_vs.YUV444P16
    assert first_call.get("dither_type") == "none"
    assert second_call.get("format") == fake_vs.RGB24
    assert second_call.get("dither_type") == RGBDither.ORDERED.value
    assert any("Full-chroma pivot" in note for note in pivot_notes)


def test_save_frame_with_fpng_skips_promotion_on_even_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clip, fake_vs, writer_calls, resize_calls, _ = _prepare_fake_vapoursynth_clip(
        monkeypatch,
        width=1920,
        height=1080,
        subsampling_w=1,
        subsampling_h=1,
        bits_per_sample=10,
        format_name="YUV420P10",
    )

    cfg = ScreenshotConfig(add_frame_info=False, rgb_dither=RGBDither.ORDERED)
    tonemap_info = vs_core.TonemapInfo(
        applied=False,
        tone_curve=None,
        dpd=0,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
        reason="SDR source",
    )
    plan = _make_plan(pad=(0, 2, 0, 2), requires_full_chroma=False)
    source_props = {"_Matrix": 1, "_Transfer": 1, "_Primaries": 1, "_ColorRange": 1}

    render.save_frame_with_fpng(
        clip,
        frame_idx=3,
        crop=plan["crop"],
        scaled=plan["scaled"],
        pad=plan["pad"],
        path=tmp_path / "frame.png",
        cfg=cfg,
        label="Clip",
        requested_frame=3,
        selection_label=None,
        source_props=source_props,
        geometry_plan=plan,
        tonemap_info=tonemap_info,
        debug_state=None,
        frame_info_allowed=False,
        overlays_allowed=False,
    )

    assert writer_calls, "fpng writer should be invoked"
    assert not any(call.get("format") == fake_vs.YUV444P16 for call in resize_calls)
    assert any(call.get("format") == fake_vs.RGB24 for call in resize_calls)


def test_geometry_preserves_colour_props(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clip, _fake_vs, writer_calls, _, _ = _prepare_fake_vapoursynth_clip(
        monkeypatch,
        width=1920,
        height=1080,
        subsampling_w=1,
        subsampling_h=1,
        bits_per_sample=10,
        format_name="YUV420P10",
    )
    clip.props.update({"_Matrix": 1, "_ColorRange": 1})

    captured_source_props: dict[str, Any] = {}
    original_ensure = render.ensure_rgb24

    def _capture_ensure(
        core: Any,
        render_clip: Any,
        frame_idx: int,
        *,
        source_props: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if source_props is not None:
            captured_source_props.update(source_props)
        return original_ensure(core, render_clip, frame_idx, source_props=source_props, **kwargs)

    monkeypatch.setattr(render, "ensure_rgb24", _capture_ensure)

    cfg = ScreenshotConfig(add_frame_info=False)
    cfg.export_range = ExportRange.FULL
    tonemap_info = vs_core.TonemapInfo(
        applied=False,
        tone_curve=None,
        dpd=0,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
        reason="SDR source",
    )
    plan = _make_plan(
        pad=(2, 4, 2, 4),
        requires_full_chroma=True,
        promotion_axes="horizontal+vertical",
    )
    render.save_frame_with_fpng(
        clip,
        frame_idx=5,
        crop=plan["crop"],
        scaled=plan["scaled"],
        pad=plan["pad"],
        path=tmp_path / "frame.png",
        cfg=cfg,
        label="Clip",
        requested_frame=5,
        selection_label=None,
        source_props={"_Matrix": 1, "_Transfer": 1, "_Primaries": 1, "_ColorRange": 1},
        geometry_plan=plan,
        tonemap_info=tonemap_info,
        pivot_notifier=None,
        debug_state=None,
        frame_info_allowed=False,
        overlays_allowed=False,
    )

    assert captured_source_props.get("_ColorRange") == 1
    assert writer_calls, "fpng writer should receive clip"
    final_props = writer_calls[-1]["props"]
    expected_range = 0 if cfg.export_range is ExportRange.FULL else 1
    assert final_props.get("_ColorRange") == expected_range
    assert final_props.get("_Matrix") == 0


def test_fpng_respects_limited_export_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip, _fake_vs, writer_calls, _, _ = _prepare_fake_vapoursynth_clip(
        monkeypatch,
        width=1920,
        height=1080,
        subsampling_w=1,
        subsampling_h=1,
        bits_per_sample=10,
        format_name="YUV420P10",
    )
    clip.props.update({"_Matrix": 1, "_ColorRange": 1})

    cfg = ScreenshotConfig(add_frame_info=False)
    cfg.export_range = ExportRange.LIMITED
    tonemap_info = vs_core.TonemapInfo(
        applied=False,
        tone_curve=None,
        dpd=0,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
        reason="SDR source",
    )
    plan = _make_plan(
        pad=(0, 0, 0, 0),
        requires_full_chroma=False,
        promotion_axes="none",
    )
    render.save_frame_with_fpng(
        clip,
        frame_idx=2,
        crop=plan["crop"],
        scaled=plan["scaled"],
        pad=plan["pad"],
        path=tmp_path / "frame.png",
        cfg=cfg,
        label="Clip",
        requested_frame=2,
        selection_label=None,
        source_props={"_Matrix": 1, "_Transfer": 1, "_Primaries": 1, "_ColorRange": 1},
        geometry_plan=plan,
        tonemap_info=tonemap_info,
        pivot_notifier=None,
        debug_state=None,
        frame_info_allowed=False,
        overlays_allowed=False,
    )

    assert writer_calls, "fpng writer should receive clip"
    final_props = writer_calls[-1]["props"]
    assert final_props.get("_ColorRange") == 1
    assert "_SourceColorRange" not in final_props


def test_overlay_preserves_limited_range_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip, fake_vs, writer_calls, resize_calls, levels_calls = _prepare_fake_vapoursynth_clip(
        monkeypatch,
        width=1280,
        height=720,
        subsampling_w=1,
        subsampling_h=1,
        bits_per_sample=8,
        format_name="YUV420P8",
    )
    cfg = ScreenshotConfig(add_frame_info=False)
    cfg.export_range = ExportRange.LIMITED
    tonemap_info = vs_core.TonemapInfo(
        applied=False,
        tone_curve=None,
        dpd=0,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
        reason="SDR source",
    )
    overlay_state: dict[str, Any] = {}

    render.save_frame_with_fpng(
        clip,
        frame_idx=0,
        crop=(0, 0, 0, 0),
        scaled=(clip.width, clip.height),
        pad=(0, 0, 0, 0),
        path=tmp_path / "limited.png",
        cfg=cfg,
        label="Clip",
        requested_frame=0,
        selection_label=None,
        overlay_text="Demo overlay",
        overlay_state=overlay_state,
        strict_overlay=False,
        source_props={"_Matrix": 1, "_ColorRange": 1, "_Primaries": 1, "_Transfer": 1},
        geometry_plan=cast(GeometryPlan, {"requires_full_chroma": False}),
        tonemap_info=tonemap_info,
        color_cfg=ColorConfig(),
        warning_sink=[],
        frame_info_allowed=False,
        overlays_allowed=True,
        expand_to_full=False,
    )

    assert writer_calls, "fpng writer should have been invoked"
    props = writer_calls[0]["props"]
    assert props.get("_ColorRange") == 1
    assert "_SourceColorRange" not in props
    assert any(call.get("format") == fake_vs.RGB24 for call in resize_calls)
    assert not levels_calls, "Limited export should not expand range"
    assert fake_vs.core._overlay_calls, "Overlay path should be invoked"


def test_overlay_expands_range_when_exporting_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip, fake_vs, writer_calls, resize_calls, levels_calls = _prepare_fake_vapoursynth_clip(
        monkeypatch,
        width=1280,
        height=720,
        subsampling_w=1,
        subsampling_h=1,
        bits_per_sample=8,
        format_name="YUV420P8",
    )
    cfg = ScreenshotConfig(add_frame_info=False)
    tonemap_info = vs_core.TonemapInfo(
        applied=False,
        tone_curve=None,
        dpd=0,
        target_nits=100.0,
        dst_min_nits=0.18,
        src_csp_hint=None,
        reason="SDR source",
    )

    render.save_frame_with_fpng(
        clip,
        frame_idx=0,
        crop=(0, 0, 0, 0),
        scaled=(clip.width, clip.height),
        pad=(0, 0, 0, 0),
        path=tmp_path / "full.png",
        cfg=cfg,
        label="Clip",
        requested_frame=0,
        selection_label=None,
        overlay_text="Demo overlay",
        overlay_state={},
        strict_overlay=False,
        source_props={"_Matrix": 1, "_ColorRange": 1, "_Primaries": 1, "_Transfer": 1},
        geometry_plan=cast(GeometryPlan, {"requires_full_chroma": False}),
        tonemap_info=tonemap_info,
        color_cfg=ColorConfig(),
        warning_sink=[],
        frame_info_allowed=False,
        overlays_allowed=True,
        expand_to_full=True,
    )

    assert writer_calls, "fpng writer should have been invoked"
    props = writer_calls[0]["props"]
    assert props.get("_ColorRange") == 0
    assert props.get("_SourceColorRange") == 1
    assert any(call.get("format") == fake_vs.RGB24 for call in resize_calls)
    assert levels_calls, "Full export should apply limited-to-full expansion"
    assert fake_vs.core._overlay_calls, "Overlay path should be invoked"


def test_ensure_rgb24_skips_tonemapped_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    clip, fake_vs, _, _, levels_calls = _prepare_fake_vapoursynth_clip(
        monkeypatch,
        width=960,
        height=540,
        subsampling_w=0,
        subsampling_h=0,
        bits_per_sample=8,
        color_family="RGB",
        format_name="RGB24",
    )
    clip.props.update({"_ColorRange": 1, "_Tonemapped": "placebo:bt.2390"})

    converted = render.ensure_rgb24(
        fake_vs.core,
        clip,
        frame_idx=0,
        source_props={"_ColorRange": 1, "_Tonemapped": "placebo:bt.2390"},
        expand_to_full=True,
    )

    assert isinstance(converted, type(clip))
    assert levels_calls == []
    assert converted.props.get("_ColorRange") == 1
    assert converted.props.get("_Tonemapped") == "placebo:bt.2390"


def test_ensure_rgb24_applies_rec709_defaults_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    captured_props: dict[str, Any] = {}

    class _DummyStd:
        def __init__(self, parent: "_DummyClip") -> None:
            self._parent = parent

        def SetFrameProps(self, **kwargs: Any) -> "_DummyClip":
            captured_props.update(kwargs)
            return self._parent

    class _DummyClip:
        def __init__(self) -> None:
            self.std = _DummyStd(self)

    def fake_point(clip: Any, **kwargs: Any) -> _DummyClip:
        captured.update(kwargs)
        return _DummyClip()

    fake_core = types.SimpleNamespace(resize=types.SimpleNamespace(Point=fake_point))
    yuv_family = object()
    fake_vs = types.SimpleNamespace(
        RGB24=0,
        RGB=object(),
        YUV=yuv_family,
        RANGE_FULL=0,
        RANGE_LIMITED=1,
        MATRIX_BT709=1,
        TRANSFER_BT709=1,
        PRIMARIES_BT709=1,
    )
    fake_vs.core = fake_core

    class _SourceClip:
        def __init__(self) -> None:
            self.core = fake_core
            self.format = types.SimpleNamespace(color_family=yuv_family, bits_per_sample=8)
            self.height = 1080

        def get_frame(self, idx: int) -> Any:  # type: ignore[override]
            # Return a dummy frame with empty props to satisfy the new strict check
            return types.SimpleNamespace(props={})

    patcher = cast(Any, monkeypatch)
    patcher.setitem(sys.modules, "vapoursynth", fake_vs)

    converted = render.ensure_rgb24(
        fake_core,
        _SourceClip(),
        frame_idx=12,
        source_props={},
        rgb_dither=RGBDither.ERROR_DIFFUSION,
    )
    assert isinstance(converted, _DummyClip)
    assert captured.get("matrix_in") == 1
    assert captured.get("transfer_in") == 1
    assert captured.get("primaries_in") == 1
    assert captured.get("range_in") == 1
    assert captured.get("dither_type") == RGBDither.ERROR_DIFFUSION.value
    expected_range = 1
    assert captured.get("range") == expected_range
    expected_props = {"_Matrix": 0, "_ColorRange": expected_range}
    assert captured_props == expected_props


def test_ensure_rgb24_uses_source_colour_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    captured_props: dict[str, Any] = {}

    class _DummyStd:
        def __init__(self, parent: "_DummyClip") -> None:
            self._parent = parent

        def SetFrameProps(self, **kwargs: Any) -> "_DummyClip":
            captured_props.update(kwargs)
            return self._parent

    class _DummyClip:
        def __init__(self) -> None:
            self.std = _DummyStd(self)

    def fake_point(clip: Any, **kwargs: Any) -> _DummyClip:
        captured.update(kwargs)
        return _DummyClip()

    fake_core = types.SimpleNamespace(resize=types.SimpleNamespace(Point=fake_point))
    yuv_family = object()
    fake_vs = types.SimpleNamespace(
        RGB24=0,
        RGB=object(),
        YUV=yuv_family,
        RANGE_FULL=0,
        RANGE_LIMITED=1,
        MATRIX_BT709=1,
        TRANSFER_BT709=1,
        PRIMARIES_BT709=1,
    )
    fake_vs.core = fake_core

    class _SourceClip:
        def __init__(self) -> None:
            self.core = fake_core
            self.format = types.SimpleNamespace(color_family=yuv_family, bits_per_sample=10)
            self.height = 1080
            self._frame = types.SimpleNamespace(
                props={
                    "_Matrix": 9,
                    "_Transfer": 16,
                    "_Primaries": 9,
                    "_ColorRange": 0,
                }
            )

        def get_frame(self, idx: int) -> Any:  # type: ignore[override]
            raise AssertionError("get_frame should not be called when props are supplied")

    patcher = cast(Any, monkeypatch)
    patcher.setitem(sys.modules, "vapoursynth", fake_vs)

    converted = render.ensure_rgb24(
        fake_core,
        _SourceClip(),
        frame_idx=24,
        source_props={
            "_Matrix": 9,
            "_Transfer": 16,
            "_Primaries": 9,
            "_ColorRange": 0,
        },
        rgb_dither=RGBDither.ORDERED,
    )
    assert isinstance(converted, _DummyClip)
    assert captured.get("matrix_in") == 9
    assert captured.get("range_in") == 0
    assert captured.get("dither_type") == RGBDither.ORDERED.value
    expected_range = 0
    assert captured.get("range") == expected_range
    expected_props = {
        "_Matrix": 0,
        "_ColorRange": expected_range,
        "_Primaries": 9,
        "_Transfer": 16,
    }
    assert captured_props == expected_props
