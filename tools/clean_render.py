
path = "src/frame_compare/screenshot/render.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Remove redundant metadata resolution
    if "matrix, transfer, primaries, color_range = vs_core.resolve_color_metadata(props)" in line:
        continue

    # Remove unnecessary cast
    if "converted = cast(" in line and "point(" in line:
        # replace cast(Any, point(...)) with point(...)
        # Regex might be safer
        line = line.replace("cast(", "").replace("Any, ", "")
        # This is brittle. Simple string replace for the specific line seen in error
        # Line 448 (approx): converted = cast(
        #     Any,
        #     point(
        # ...
        # Actually it's likely a multiline statement.
        # I'll skip complex regex editing and just use sed for the simple line removal.
        pass

    new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
