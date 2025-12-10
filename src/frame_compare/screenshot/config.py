"""Screenshot configuration and data types."""

from __future__ import annotations

from typing import Tuple, TypedDict


class GeometryPlan(TypedDict):
    """
    Resolved crop/pad/scale plan for rendering a screenshot.

    Attributes:
        width (int): Source clip width.
        height (int): Source clip height.
        crop (tuple[int, int, int, int]): Cropping values for left, top, right, bottom.
        cropped_w (int): Width after cropping.
        cropped_h (int): Height after cropping.
        scaled (tuple[int, int]): Dimensions after scaling.
        pad (tuple[int, int, int, int]): Padding applied around the scaled frame.
        final (tuple[int, int]): Final output dimensions.
        requires_full_chroma (bool): Whether geometry requires a 4:4:4 pivot.
        promotion_axes (str): Subsampling-aware axis label describing which geometry axis
            triggered promotion, or ``"none"`` when no promotion is required.
    """

    width: int
    height: int
    crop: Tuple[int, int, int, int]
    cropped_w: int
    cropped_h: int
    scaled: Tuple[int, int]
    pad: Tuple[int, int, int, int]
    final: Tuple[int, int]
    requires_full_chroma: bool
    promotion_axes: str
