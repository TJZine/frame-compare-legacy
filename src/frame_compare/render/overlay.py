from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, List, Optional, cast

from src.datatypes import ColorConfig
from src.frame_compare import diagnostics as _diagnostics
from src.frame_compare.layout_utils import format_resolution_summary

__all__ = [
    "FRAME_INFO_STYLE",
    "OVERLAY_STYLE",
    "OverlayState",
    "OverlayStateValue",
    "append_overlay_warning",
    "compose_overlay_text",
    "extract_mastering_display_luminance",
    "format_luminance_value",
    "format_mastering_display_line",
    "format_selection_line",
    "get_overlay_warnings",
    "new_overlay_state",
    "normalize_selection_label",
]

if TYPE_CHECKING:
    from src.frame_compare import vs as vs_core
    from src.frame_compare.screenshot.config import GeometryPlan

    TonemapInfo = vs_core.TonemapInfo
else:  # pragma: no cover - runtime type fallback
    GeometryPlan = Mapping[str, Any]  # type: ignore[misc, assignment]
    TonemapInfo = Any


OverlayStateValue = str | List[str]
OverlayState = MutableMapping[str, OverlayStateValue]


SELECTION_LABELS: Mapping[str, str] = {
    "dark": "Dark",
    "bright": "Bright",
    "motion": "Motion",
    "user": "User",
    "random": "Random",
    "auto": "Auto",
    "cached": "Cached",
}


FRAME_INFO_STYLE = (
    'sans-serif,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,'
    '"0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1"'
)
OVERLAY_STYLE = (
    'sans-serif,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,'
    '"0,0,0,0,100,100,0,0,1,2,0,7,10,10,70,1"'
)

extract_mastering_display_luminance = _diagnostics.extract_mastering_display_luminance
format_luminance_value = _diagnostics.format_luminance_value
format_mastering_display_line = _diagnostics.format_mastering_display_line


def new_overlay_state() -> OverlayState:
    """Create a mutable overlay state container."""

    return cast(OverlayState, {})


def append_overlay_warning(state: OverlayState, message: str) -> None:
    """Store a warning message inside *state* preserving existing entries."""

    warnings_value = state.get("warnings")
    if not isinstance(warnings_value, list):
        warnings_value = []
        state["warnings"] = warnings_value
    warnings_value.append(message)


def get_overlay_warnings(state: OverlayState) -> List[str]:
    """Return previously recorded overlay warnings."""

    warnings_value = state.get("warnings")
    if isinstance(warnings_value, list):
        return warnings_value
    return []


def normalize_selection_label(label: Optional[str]) -> str:
    """Normalize a selection label into a user-facing display name."""

    if not label:
        return "(unknown)"
    cleaned = label.strip()
    if not cleaned:
        return "(unknown)"
    normalized = cleaned.lower()
    mapped = SELECTION_LABELS.get(normalized)
    if mapped:
        return mapped
    return cleaned


def format_selection_line(selection_label: Optional[str]) -> str:
    """Return the formatted selection line for overlay text."""

    return f"Frame Selection Type: {normalize_selection_label(selection_label)}"


def compose_overlay_text(
    base_text: Optional[str],
    color_cfg: ColorConfig,
    plan: GeometryPlan,
    selection_label: Optional[str],
    source_props: Mapping[str, Any],
    *,
    tonemap_info: Optional[TonemapInfo],
    selection_detail: Optional[Mapping[str, Any]] = None,  # kept for compatibility
) -> Optional[str]:
    """Compose a user-facing overlay text snippet."""

    if not bool(getattr(color_cfg, "overlay_enabled", True)):
        return None

    mode = str(getattr(color_cfg, "overlay_mode", "minimal")).strip().lower()
    if mode != "diagnostic":
        lines: List[str] = []
        if base_text:
            lines.append(base_text)
        lines.append(format_resolution_summary(plan))
        lines.append(format_selection_line(selection_label))
        return "\n".join(lines)

    lines = []
    if base_text:
        lines.append(base_text)

    lines.append(format_resolution_summary(plan))
    include_hdr_details = bool(tonemap_info and getattr(tonemap_info, "applied", False))
    if include_hdr_details:
        lines.append(format_mastering_display_line(source_props))
    hdr_line = _diagnostics.format_hdr_line(_diagnostics.extract_hdr_metadata(source_props))
    if hdr_line:
        lines.append(hdr_line)
    dovi_metadata = _diagnostics.extract_dovi_metadata(source_props)
    dovi_line = _diagnostics.format_dovi_line(_resolve_dovi_label(tonemap_info), dovi_metadata)
    if dovi_line:
        lines.append(dovi_line)
    dovi_l1_line = _diagnostics.format_dovi_l1_line(dovi_metadata)
    if dovi_l1_line:
        lines.append(dovi_l1_line)
    dovi_l5_line = _diagnostics.format_dovi_l5_line(dovi_metadata)
    if dovi_l5_line:
        lines.append(dovi_l5_line)
    dovi_l6_line = _diagnostics.format_dovi_l6_line(dovi_metadata)
    if dovi_l6_line:
        lines.append(dovi_l6_line)
    range_line = _diagnostics.format_dynamic_range_line(_diagnostics.classify_color_range(source_props))
    if range_line:
        lines.append(range_line)
    frame_metrics_entry = _extract_frame_metrics(selection_detail)
    metrics_line = _diagnostics.format_frame_metrics_line(frame_metrics_entry)
    if metrics_line:
        lines.append(metrics_line)
    lines.append(format_selection_line(selection_label))
    return "\n".join(lines)


def _resolve_dovi_label(tonemap_info: TonemapInfo | None) -> str | None:
    if tonemap_info is None:
        return None
    value = getattr(tonemap_info, "use_dovi", None)
    if isinstance(value, bool):
        return "on" if value else "off"
    if value is None:
        return "auto"
    if isinstance(value, str):
        text = value.strip()
        return text or "auto"
    return None


def _extract_frame_metrics(selection_detail: Optional[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not isinstance(selection_detail, Mapping):
        return None
    diagnostics_obj = selection_detail.get("diagnostics")
    if not isinstance(diagnostics_obj, Mapping):
        return None
    diagnostics_block = cast(Mapping[str, Any], diagnostics_obj)
    entry = diagnostics_block.get("frame_metrics")
    if isinstance(entry, Mapping):
        return cast(Mapping[str, Any], entry)
    return None
