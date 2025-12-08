from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, List, Tuple

from src.datatypes import OddGeometryPolicy
from src.frame_compare.render.errors import ScreenshotGeometryError

__all__ = [
    "LETTERBOX_RATIO_TOLERANCE",
    "align_letterbox_pillarbox",
    "align_padding_mod",
    "axis_has_odd",
    "compute_requires_full_chroma",
    "compute_scaled_dimensions",
    "describe_plan_axes",
    "format_dimensions",
    "get_subsampling",
    "normalise_geometry_policy",
    "plan_letterbox_offsets",
    "plan_mod_crop",
    "split_padding",
]

if TYPE_CHECKING:
    from src.screenshot import GeometryPlan
else:  # pragma: no cover - runtime fallback
    GeometryPlan = MutableMapping[str, Any]  # type: ignore[assignment,misc]


LETTERBOX_RATIO_TOLERANCE = 0.04


def format_dimensions(width: int, height: int) -> str:
    """Return width × height using integer values."""

    return f"{int(width)} \u00D7 {int(height)}"


def get_subsampling(fmt: Any, attr: str) -> int:
    """Safely fetch VapourSynth format subsampling attributes."""

    try:
        raw = getattr(fmt, attr)
    except AttributeError:
        return 0
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 0


def axis_has_odd(values: Sequence[int]) -> bool:
    """Return True when at least one integer in *values* is odd."""

    for value in values:
        try:
            current = int(value)
        except (ValueError, TypeError):
            continue
        if current % 2 != 0:
            return True
    return False


def describe_plan_axes(plan: GeometryPlan | None) -> str:
    """Return a concise axis label for plans that include odd-pixel geometry."""

    if plan is None:
        return "unknown"

    crop_left, crop_top, crop_right, crop_bottom = plan["crop"]
    pad_left, pad_top, pad_right, pad_bottom = plan["pad"]

    axes: list[str] = []
    if axis_has_odd((crop_top, crop_bottom, pad_top, pad_bottom)):
        axes.append("vertical")
    if axis_has_odd((crop_left, crop_right, pad_left, pad_right)):
        axes.append("horizontal")

    if not axes:
        return "none"
    return "+".join(axes)


def normalise_geometry_policy(value: OddGeometryPolicy | str) -> OddGeometryPolicy:
    """Return a canonical OddGeometryPolicy value."""

    if isinstance(value, OddGeometryPolicy):
        return value
    try:
        return OddGeometryPolicy(str(value))
    except ValueError:
        return OddGeometryPolicy.AUTO


def compute_requires_full_chroma(
    fmt: Any,
    crop: Tuple[int, int, int, int],
    pad: Tuple[int, int, int, int],
    policy: OddGeometryPolicy | str,
) -> bool:
    """Determine if full chroma promotion is required for the given plan."""

    resolved_policy = normalise_geometry_policy(policy)
    if resolved_policy is OddGeometryPolicy.FORCE_FULL_CHROMA:
        return True
    if resolved_policy is OddGeometryPolicy.SUBSAMP_SAFE:
        return False

    subsampling_w = get_subsampling(fmt, "subsampling_w")
    subsampling_h = get_subsampling(fmt, "subsampling_h")

    vertical_odd = axis_has_odd((crop[1], crop[3], pad[1], pad[3]))
    horizontal_odd = axis_has_odd((crop[0], crop[2], pad[0], pad[2]))

    return (vertical_odd and subsampling_h > 0) or (horizontal_odd and subsampling_w > 0)


def plan_mod_crop(
    width: int,
    height: int,
    mod: int,
    letterbox_pillarbox_aware: bool,
) -> Tuple[int, int, int, int]:
    """Plan left/top/right/bottom croppings so dimensions align to ``mod``."""

    if width <= 0 or height <= 0:
        raise ScreenshotGeometryError("Clip dimensions must be positive")
    if mod <= 1:
        return (0, 0, 0, 0)

    def _axis_crop(size: int) -> Tuple[int, int]:
        remainder = size % mod
        if remainder == 0:
            return (0, 0)
        before = remainder // 2
        after = remainder - before
        return (before, after)

    left, right = _axis_crop(width)
    top, bottom = _axis_crop(height)

    if letterbox_pillarbox_aware:
        if width > height and (top + bottom) == 0 and (left + right) > 0:
            total = left + right
            left = total // 2
            right = total - left
        elif height >= width and (left + right) == 0 and (top + bottom) > 0:
            total = top + bottom
            top = total // 2
            bottom = total - top

    cropped_w = width - left - right
    cropped_h = height - top - bottom
    if cropped_w <= 0 or cropped_h <= 0:
        raise ScreenshotGeometryError("Cropping removed all pixels")

    return (left, top, right, bottom)


def align_letterbox_pillarbox(plans: List[GeometryPlan]) -> None:
    """Normalize crops for clips that only letterbox or pillarbox."""

    if not plans:
        return

    widths = [int(plan["width"]) for plan in plans]
    heights = [int(plan["height"]) for plan in plans]
    same_w = len({w for w in widths if w > 0}) == 1
    same_h = len({h for h in heights if h > 0}) == 1

    if same_w:
        target_h = min(int(plan["cropped_h"]) for plan in plans)
        for plan in plans:
            current_h = int(plan["cropped_h"])
            diff = current_h - target_h
            if diff <= 0:
                continue
            add_top = diff // 2
            add_bottom = diff - add_top
            left, top, right, bottom = plan["crop"]
            top += add_top
            bottom += add_bottom
            plan["crop"] = (left, top, right, bottom)
            plan["cropped_h"] = plan["height"] - top - bottom
    elif same_h:
        target_w = min(int(plan["cropped_w"]) for plan in plans)
        for plan in plans:
            current_w = int(plan["cropped_w"])
            diff = current_w - target_w
            if diff <= 0:
                continue
            add_left = diff // 2
            add_right = diff - add_left
            left, top, right, bottom = plan["crop"]
            left += add_left
            right += add_right
            plan["crop"] = (left, top, right, bottom)
            plan["cropped_w"] = plan["width"] - left - right


def plan_letterbox_offsets(
    plans: Sequence[GeometryPlan],
    *,
    mod: int,
    tolerance: float = LETTERBOX_RATIO_TOLERANCE,
    max_target_height: int | None = None,
) -> List[tuple[int, int]]:
    """Return top/bottom offsets to align clip ratios when auto letterboxing."""

    ratios: List[float] = []
    for plan in plans:
        try:
            width = float(plan["width"])
            height = float(plan["height"])
        except (ValueError, TypeError, KeyError):
            continue
        if width > 0 and height > 0:
            ratios.append(width / height)

    if not ratios:
        return [(0, 0) for _ in plans]

    target_ratio = max(ratios)
    if target_ratio <= 0:
        return [(0, 0) for _ in plans]

    tolerance = max(0.0, tolerance)
    min_ratio_allowed = target_ratio * (1.0 - tolerance)

    offsets: List[tuple[int, int]] = []
    for plan in plans:
        try:
            width = int(plan["width"])
            height = int(plan["height"])
        except (ValueError, TypeError, KeyError):
            offsets.append((0, 0))
            continue
        if width <= 0 or height <= 0:
            offsets.append((0, 0))
            continue

        ratio = width / height
        if ratio >= min_ratio_allowed:
            offsets.append((0, 0))
            continue

        desired_height = width / target_ratio
        target_height = int(round(desired_height))
        if max_target_height is not None:
            target_height = min(target_height, max_target_height)

        if mod > 1:
            target_height -= target_height % mod
        target_height = max(mod if mod > 0 else 1, target_height)
        if target_height >= height:
            offsets.append((0, 0))
            continue

        crop_total = height - target_height
        if crop_total <= 0:
            offsets.append((0, 0))
            continue

        top_extra = crop_total // 2
        bottom_extra = crop_total - top_extra
        offsets.append((top_extra, bottom_extra))

    return offsets


def split_padding(total: int, center: bool) -> tuple[int, int]:
    """Split padding across both ends or append to the far side."""

    amount = max(0, int(total))
    if amount <= 0:
        return (0, 0)
    if center:
        first = amount // 2
        second = amount - first
        return (first, second)
    return (0, amount)


def align_padding_mod(
    width: int,
    height: int,
    pad_left: int,
    pad_top: int,
    pad_right: int,
    pad_bottom: int,
    mod: int,
    center: bool,
) -> tuple[int, int, int, int]:
    """Expand padding so the final dimensions align with ``mod``."""

    if mod <= 1:
        return (pad_left, pad_top, pad_right, pad_bottom)

    total_pad = pad_left + pad_top + pad_right + pad_bottom
    if total_pad <= 0:
        return (pad_left, pad_top, pad_right, pad_bottom)

    final_w = width + pad_left + pad_right
    final_h = height + pad_top + pad_bottom

    remainder_w = final_w % mod
    if remainder_w:
        extra = mod - remainder_w
        add_left, add_right = split_padding(extra, center)
        pad_left += add_left
        pad_right += add_right
        final_w += extra

    remainder_h = final_h % mod
    if remainder_h:
        extra = mod - remainder_h
        add_top, add_bottom = split_padding(extra, center)
        pad_top += add_top
        pad_bottom += add_bottom

    return (pad_left, pad_top, pad_right, pad_bottom)


def compute_scaled_dimensions(
    width: int,
    height: int,
    crop: Tuple[int, int, int, int],
    target_height: int,
) -> Tuple[int, int]:
    """Return scaled width/height for a given crop + target height."""

    cropped_w = width - crop[0] - crop[2]
    cropped_h = height - crop[1] - crop[3]
    if cropped_w <= 0 or cropped_h <= 0:
        raise ScreenshotGeometryError("Invalid crop results")

    desired_h = max(1, int(round(target_height)))
    scale = desired_h / cropped_h if cropped_h else 1.0
    target_w = int(round(cropped_w * scale)) if scale != 1 else cropped_w
    target_w = max(1, target_w)
    return (target_w, desired_h)
