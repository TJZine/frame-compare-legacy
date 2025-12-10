import os
import re

ROOTS = ["src", "tests"]

REPLACEMENTS = {
    # VS Props (Internal to public)
    r"\b_snapshot_frame_props\b": "snapshot_frame_props",
    r"\b_props_signal_hdr\b": "props_signal_hdr",
    r"\b_resolve_color_metadata\b": "resolve_color_metadata",

    # Render (Internal to public)
    r"\b_normalize_rgb_dither\b": "normalize_rgb_dither",
    r"\b_new_overlay_state\b": "new_overlay_state",
    r"\b_plan_geometry\b": "plan_geometry",
    r"\b_ColorDebugState\b": "ColorDebugState",
    r"\b_resolve_source_props\b": "resolve_source_props",
    r"\b_is_sdr_pipeline\b": "is_sdr_pipeline",
    r"\b_range_constants\b": "range_constants",
    r"\b_should_expand_to_full\b": "should_expand_to_full",
    r"\b_resolve_output_color_range\b": "resolve_output_color_range",
    r"\b_clamp_frame_index\b": "clamp_frame_index",
    r"\b_resolve_source_frame_index\b": "resolve_source_frame_index",
    r"\b_compose_overlay_text\b": "compose_overlay_text",
    r"\b_save_frame_with_ffmpeg\b": "save_frame_with_ffmpeg",
    r"\b_save_frame_with_fpng\b": "save_frame_with_fpng",
    r"\b_save_frame_placeholder\b": "save_frame_placeholder",
    r"\b_get_overlay_warnings\b": "get_overlay_warnings",
    r"\b_append_overlay_warning\b": "append_overlay_warning",
    r"\b_set_clip_range\b": "set_clip_range",
    r"\b_apply_overlay_text\b": "apply_overlay_text",
    r"\b_ensure_rgb24\b": "ensure_rgb24",
    r"\b_format_dimensions\b": "format_dimensions",
    r"\b_extract_mastering_display_luminance\b": "extract_mastering_display_luminance",
    r"\b_format_luminance_value\b": "format_luminance_value",
    r"\b_format_mastering_display_line\b": "format_mastering_display_line",
    r"\b_normalize_selection_label\b": "normalize_selection_label",
    r"\b_format_selection_line\b": "format_selection_line",
    r"\b_resolve_resize_color_kwargs\b": "resolve_resize_color_kwargs",
    r"\b_legacy_rgb24_from_clip\b": "legacy_rgb24_from_clip",
    r"\b_finalize_existing_rgb24\b": "finalize_existing_rgb24",
    r"\b_apply_frame_info_overlay\b": "apply_frame_info_overlay",
    r"\b_copy_frame_props\b": "copy_frame_props",
    r"\b_expand_limited_rgb\b": "expand_limited_rgb",
    r"\b_restore_color_props\b": "restore_color_props",
    r"\b_normalise_geometry_policy\b": "normalise_geometry_policy",
    r"\b_get_subsampling\b": "get_subsampling",
    r"\b_axis_has_odd\b": "axis_has_odd",
    r"\b_describe_plan_axes\b": "describe_plan_axes",
    r"\b_safe_pivot_notify\b": "safe_pivot_notify",
    r"\b_describe_vs_format\b": "describe_vs_format",
    r"\b_resolve_promotion_axes\b": "resolve_promotion_axes",
    r"\b_promote_to_yuv444p16\b": "promote_to_yuv444p16",
    r"\b_rebalance_axis_even\b": "rebalance_axis_even",
    r"\b_compute_requires_full_chroma\b": "compute_requires_full_chroma",
    r"\b_plan_mod_crop\b": "plan_mod_crop",
    r"\b_align_letterbox_pillarbox\b": "align_letterbox_pillarbox",
    r"\b_plan_letterbox_offsets\b": "plan_letterbox_offsets",
    r"\b_resolve_auto_letterbox_mode\b": "resolve_auto_letterbox_mode",
    r"\b_apply_letterbox_crop_strict\b": "apply_letterbox_crop_strict",
    r"\b_apply_letterbox_crop_basic\b": "apply_letterbox_crop_basic",
    r"\b_split_padding\b": "split_padding",
    r"\b_align_padding_mod\b": "align_padding_mod",
    r"\b_compute_scaled_dimensions\b": "compute_scaled_dimensions",
    r"\b_maybe_log_geometry_debug\b": "maybe_log_geometry_debug",
    r"\b_sanitise_label\b": "sanitise_label",
    r"\b_derive_labels\b": "derive_labels",
    r"\b_prepare_filename\b": "prepare_filename",
    r"\b_normalise_compression_level\b": "normalise_compression_level",
    r"\b_map_fpng_compression\b": "map_fpng_compression",
    r"\b_map_png_compression_level\b": "map_png_compression_level",
    r"\b_map_ffmpeg_compression\b": "map_ffmpeg_compression",
    r"\b_escape_drawtext\b": "escape_drawtext",
}

def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    for pattern, replacement in REPLACEMENTS.items():
        content = re.sub(pattern, replacement, content)

    if content != original:
        print(f"Updated {path}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

for root_dir in ROOTS:
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".py"):
                process_file(os.path.join(dirpath, name))
