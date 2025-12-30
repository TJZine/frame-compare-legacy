from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, cast

import pytest

from src.datatypes import OddGeometryPolicy
from src.frame_compare.render import geometry
from src.frame_compare.render.errors import ScreenshotGeometryError

if TYPE_CHECKING:
    from src.frame_compare.screenshot.config import GeometryPlan
else:  # pragma: no cover - typing fallback
    GeometryPlan = Dict[str, Any]


def test_plan_mod_crop_aligns_to_even_mod() -> None:
    left, top, right, bottom = geometry.plan_mod_crop(1923, 1079, 2, False)
    cropped_w = 1923 - left - right
    cropped_h = 1079 - top - bottom
    assert cropped_w % 2 == 0
    assert cropped_h % 2 == 0


def test_plan_mod_crop_rejects_invalid_dimensions() -> None:
    with pytest.raises(ScreenshotGeometryError):
        geometry.plan_mod_crop(0, 1080, 2, False)


def test_align_padding_mod_adjusts_final_dimensions() -> None:
    pad = geometry.align_padding_mod(1920, 1080, 1, 0, 1, 0, mod=4, center=True)
    final_w = 1920 + pad[0] + pad[2]
    assert final_w % 4 == 0


def test_compute_requires_full_chroma_detects_odd_axes() -> None:
    fmt = SimpleNamespace(subsampling_w=1, subsampling_h=1)
    crop = (1, 0, 1, 0)
    pad = (0, 0, 0, 0)
    assert geometry.compute_requires_full_chroma(fmt, crop, pad, OddGeometryPolicy.AUTO)


def test_plan_letterbox_offsets_returns_even_offsets() -> None:
    plan = cast(
        "GeometryPlan",
        {"width": 1280, "height": 720, "cropped_w": 1280, "cropped_h": 720},
    )
    taller_plan = cast(
        "GeometryPlan",
        {"width": 1280, "height": 1080, "cropped_w": 1280, "cropped_h": 1080},
    )
    offsets = geometry.plan_letterbox_offsets([plan, taller_plan], mod=2)
    assert offsets[1][0] >= 0
    assert offsets[1][0] == offsets[1][1]


def test_compute_scaled_dimensions_scales_height() -> None:
    result = geometry.compute_scaled_dimensions(1920, 1080, (0, 0, 0, 0), 540)
    assert result == (960, 540)
