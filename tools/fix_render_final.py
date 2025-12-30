path = "src/frame_compare/screenshot/render.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # 1. resolve_resize_color_kwargs
    if "kwargs: Dict[str, int] = {}" in line:
        # Check context: resolve_resize_color_kwargs
        # We need to insert before this line, but only if inside resolve_resize_color_kwargs
        pass # Logic handled below by just inserting before specific markers

    # Strategy: Insert before the first usage or distinct marker in each function.

    if "kwargs: Dict[str, int] = {}" in line:
        # For resolve_resize_color_kwargs
        # We can just insert it here if we are sure it's the right place.
        # But this line appears multiple times?
        # resolve_resize_color_kwargs has "matrix, transfer... = ..." before "kwargs = {}" in original.
        # But wait, clean_render.py removed it.
        # I'll check previous lines to confirm context.
        if "resolve_resize_color_kwargs" in "".join(new_lines[-5:]):
             new_lines.append("    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)\n")

    if "current_range = range_limited if color_range is None else int(color_range)" in line:
        # finalize_existing_rgb24
        new_lines.append("    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)\n")

    if 'resize_kwargs.pop("transfer_in", None)' in line:
        # ensure_rgb24
        new_lines.append("    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)\n")

    if "prop_kwargs: Dict[str, int] = {}" in line:
        # restore_color_props
        # Check if we are in restore_color_props
        if "restore_color_props" in "".join(new_lines[-10:]):
             new_lines.append("    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)\n")

    new_lines.append(line)

# Handle the case where I might have duplicate insertions or misscontext?
# Actually, I'll just look for the functions and insert at the start.
# This is safer.

lines = [] # Reset
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Function bodies replacement
# 1. resolve_resize_color_kwargs
if "def resolve_resize_color_kwargs(props: Mapping[str, Any]) -> Dict[str, int]:" in content:
    content = content.replace(
        'def resolve_resize_color_kwargs(props: Mapping[str, Any]) -> Dict[str, int]:\n    """Build resize arguments describing the source clip\'s colour space."""\n\n\n    kwargs: Dict[str, int] = {}',
        'def resolve_resize_color_kwargs(props: Mapping[str, Any]) -> Dict[str, int]:\n    """Build resize arguments describing the source clip\'s colour space."""\n\n    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)\n    kwargs: Dict[str, int] = {}'
    )

# 2. finalize_existing_rgb24
# Search for:
#     tonemapped_flag = props.get("_Tonemapped")
#     is_tonemapped = tonemapped_flag is not None
#
#     current_range = range_limited if color_range is None else int(color_range)
#
# Replace with:
#     tonemapped_flag = props.get("_Tonemapped")
#     is_tonemapped = tonemapped_flag is not None
#
#     matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)
#     current_range = range_limited if color_range is None else int(color_range)

content = content.replace(
    '    tonemapped_flag = props.get("_Tonemapped")\n    is_tonemapped = tonemapped_flag is not None\n\n    current_range = range_limited if color_range is None else int(color_range)',
    '    tonemapped_flag = props.get("_Tonemapped")\n    is_tonemapped = tonemapped_flag is not None\n\n    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)\n    current_range = range_limited if color_range is None else int(color_range)'
)

# 3. ensure_rgb24
# Search for:
#     props = dict(vs_core.snapshot_frame_props(clip))
#     resize_kwargs = resolve_resize_color_kwargs(props)
#     resize_kwargs.pop("transfer_in", None)
#
# Replace with:
#     props = dict(vs_core.snapshot_frame_props(clip))
#     resize_kwargs = resolve_resize_color_kwargs(props)
#     matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)
#     resize_kwargs.pop("transfer_in", None)

content = content.replace(
    '    if not props:\n        props = dict(vs_core.snapshot_frame_props(clip))\n    resize_kwargs = resolve_resize_color_kwargs(props)\n    resize_kwargs.pop("transfer_in", None)',
    '    if not props:\n        props = dict(vs_core.snapshot_frame_props(clip))\n    resize_kwargs = resolve_resize_color_kwargs(props)\n    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)\n    resize_kwargs.pop("transfer_in", None)'
)

# 4. restore_color_props
# Search for:
# def restore_color_props(
# ...
# ) -> Any:
#     """Reapply colour metadata to *clip* based on *props*. """
#
#     std_ns = getattr(core, "std", None)
#     set_props = getattr(std_ns, "SetFrameProps", None) if std_ns is not None else None
#     if not callable(set_props):
#         return clip
#
#     prop_kwargs: Dict[str, int] = {}

# Replace with insertion before prop_kwargs

content = content.replace(
    '    if not callable(set_props):\n        return clip\n\n    prop_kwargs: Dict[str, int] = {}',
    '    if not callable(set_props):\n        return clip\n\n    matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)\n    prop_kwargs: Dict[str, int] = {}'
)

# Also fix the Unnecessary cast at line ~448 (cast(Any, point...))
content = content.replace(
    '        converted = cast(\n            Any,\n            point(',
    '        converted = point('
)
# And the closing paren line. It was:
#             ),\n#         )
# It becomes:
#             ),

# This is tricky with simple replace. I'll rely on "converted = point(" being enough if python syntax allows?
# No, `cast(Any, point(args))` -> `point(args)`.
# `cast(Any, \n point(\n args \n ) \n )`
# I removed `cast(Any, ` part.
# The end `)` of cast needs removal.
# I'll just let pyright complain about unnecessary cast if I can't safely regex it without reading line by line statefully.
# Or I can use my brain.
# `converted = cast(Any, point(..., ...))`
# I will NOT fix the cast with simple replace to avoid syntax errors. I'll try to use regex.

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
