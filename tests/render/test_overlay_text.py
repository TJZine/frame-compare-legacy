from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, cast

from src.datatypes import ColorConfig
from src.frame_compare.render import overlay

if TYPE_CHECKING:
    from src.frame_compare.screenshot.config import GeometryPlan
    from src.frame_compare.vs import TonemapInfo
else:  # pragma: no cover - typing fallback
    GeometryPlan = Dict[str, Any]
    TonemapInfo = Any


def _make_plan() -> "GeometryPlan":
    return cast(
        "GeometryPlan",
        {
            "width": 1920,
            "height": 1080,
            "crop": (0, 0, 0, 0),
            "pad": (0, 0, 0, 0),
            "cropped_w": 1920,
            "cropped_h": 1080,
            "scaled": (1920, 1080),
            "final": (1920, 1080),
        },
    )


def test_compose_overlay_text_minimal_mode_includes_base_and_selection() -> None:
    cfg = ColorConfig()
    cfg.overlay_mode = "minimal"
    text = overlay.compose_overlay_text(
        "Base",
        cfg,
        _make_plan(),
        "dark",
        {},
        tonemap_info=None,
    )
    assert text is not None
    lines = text.splitlines()
    assert lines[0] == "Base"
    assert lines[1].endswith("(native)")
    assert lines[2] == "Frame Selection Type: Dark"


def test_compose_overlay_text_diagnostic_adds_mdl_line() -> None:
    cfg = ColorConfig()
    cfg.overlay_mode = "diagnostic"
    tonemap = cast("TonemapInfo", SimpleNamespace(applied=True))
    props = {"MasteringDisplayLuminance": "0.001 1000"}

    text = overlay.compose_overlay_text(
        None,
        cfg,
        _make_plan(),
        None,
        props,
        tonemap_info=tonemap,
    )
    assert text is not None
    assert "MDL: min:" in text
    assert "Frame Selection Type: (unknown)" in text


def test_mastering_display_extraction_falls_back_to_combined_keys() -> None:
    props = {"MasteringDisplayLuminance": "0.02, 950"}
    min_value, max_value = overlay.extract_mastering_display_luminance(props)
    assert min_value == 0.02
    assert max_value == 950


def test_compose_overlay_text_includes_diagnostics_details() -> None:
    cfg = ColorConfig()
    cfg.overlay_mode = "diagnostic"
    tonemap = cast("TonemapInfo", SimpleNamespace(applied=True, use_dovi=True))
    props = {
        "MasteringDisplayLuminance": "0.005 1000",
        "ContentLightLevelMax": "900",
        "ContentLightLevelFall": "300",
        "DolbyVision_Block_Index": 2,
        "DolbyVision_Block_Total": 10,
        "DolbyVision_Target_Nits": "800",
        "DolbyVision_L1_Average": 12.5,
        "DolbyVision_L1_Maximum": 450,
        "_colorrange": 1,
    }
    selection_detail = {
        "diagnostics": {
            "frame_metrics": {
                "avg_nits": 120.5,
                "max_nits": 180.0,
                "category": "highlight",
            }
        }
    }

    text = overlay.compose_overlay_text(
        "Base",
        cfg,
        _make_plan(),
        "bright",
        props,
        tonemap_info=tonemap,
        selection_detail=selection_detail,
    )
    assert text is not None
    lines = text.splitlines()
    assert any(line.startswith("HDR:") for line in lines)
    assert any(line.startswith("DoVi:") for line in lines)
    assert any("DV RPU Level 1" in line for line in lines)
    assert any("Range: Limited" in line for line in lines)
    assert any("Measurement MAX/AVG:" in line and "highlight" in line for line in lines)


def test_compose_overlay_text_skips_frame_metrics_without_entry() -> None:
    cfg = ColorConfig()
    cfg.overlay_mode = "diagnostic"
    tonemap = cast("TonemapInfo", SimpleNamespace(applied=True, use_dovi=False))
    props = {"MasteringDisplayLuminance": "0.001 1000", "_ColorRange": 0}

    text = overlay.compose_overlay_text(
        None,
        cfg,
        _make_plan(),
        "dark",
        props,
        tonemap_info=tonemap,
        selection_detail={"diagnostics": {}},
    )
    assert text is not None
    assert "Measurement MAX/AVG:" not in text


def test_overlay_warning_helpers_record_messages() -> None:
    state = overlay.new_overlay_state()
    overlay.append_overlay_warning(state, "issue")
    assert overlay.get_overlay_warnings(state) == ["issue"]
