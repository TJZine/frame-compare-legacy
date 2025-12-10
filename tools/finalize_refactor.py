import os
import re

def fix_render_py():
    path = "src/frame_compare/screenshot/render.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix missed replacements
    content = content.replace("_ColorDebugState", "ColorDebugState")
    
    # Add __all__
    if "__all__" not in content:
        exports = [
            "normalize_rgb_dither", "new_overlay_state", "plan_geometry", "ColorDebugState",
            "resolve_source_props", "is_sdr_pipeline", "range_constants", "should_expand_to_full",
            "resolve_output_color_range", "clamp_frame_index", "resolve_source_frame_index",
            "compose_overlay_text", "save_frame_with_ffmpeg", "save_frame_with_fpng",
            "save_frame_placeholder", "get_overlay_warnings", "append_overlay_warning",
            "set_clip_range", "apply_overlay_text", "ensure_rgb24", 
            "resolve_resize_color_kwargs", "legacy_rgb24_from_clip", "finalize_existing_rgb24",
            "apply_frame_info_overlay", "copy_frame_props", "expand_limited_rgb", "restore_color_props",
            "normalise_geometry_policy", "get_subsampling", "axis_has_odd", "describe_plan_axes",
            "safe_pivot_notify", "describe_vs_format", "resolve_promotion_axes", "promote_to_yuv444p16",
            "rebalance_axis_even", "compute_requires_full_chroma", "plan_mod_crop",
            "align_letterbox_pillarbox", "plan_letterbox_offsets", "resolve_auto_letterbox_mode",
            "apply_letterbox_crop_strict", "apply_letterbox_crop_basic", "split_padding",
            "align_padding_mod", "compute_scaled_dimensions", "maybe_log_geometry_debug",
            "normalise_compression_level", "map_fpng_compression", "map_png_compression_level",
            "map_ffmpeg_compression",
            "OverlayState", "OverlayStateValue", "FRAME_INFO_STYLE", "OVERLAY_STYLE",
            "FrameEvalFunc", "SubtitleFunc"
        ]
        all_block = f"\n__all__ = {exports!r}\n"
        content += all_block

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed {path}")

def fix_test_screenshot_py():
    path = "tests/test_screenshot.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Add imports
    if "from src.frame_compare.render.errors import" not in content:
        content = content.replace(
            "from src.frame_compare.screenshot.render import (",
            "from src.frame_compare.render.errors import ScreenshotError, ScreenshotWriterError\nfrom src.frame_compare.screenshot.render import ("
        )

    # Fix dict args type errors
    content = content.replace(
        'geometry_plan={"requires_full_chroma": True}', 
        'geometry_plan=cast(GeometryPlan, {"requires_full_chroma": True})'
    )
    content = content.replace(
        'geometry_plan={"requires_full_chroma": False}', 
        'geometry_plan=cast(GeometryPlan, {"requires_full_chroma": False})'
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed {path}")

if __name__ == "__main__":
    fix_render_py()
    fix_test_screenshot_py()
