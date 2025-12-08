from __future__ import annotations

__all__ = [
    "escape_drawtext",
    "map_ffmpeg_compression",
    "map_fpng_compression",
    "map_png_compression_level",
    "normalise_compression_level",
]


def normalise_compression_level(level: int) -> int:
    """Clamp arbitrary compression levels to the 0–2 range."""

    try:
        value = int(level)
    except (ValueError, TypeError):
        return 1
    return max(0, min(2, value))


def map_fpng_compression(level: int) -> int:
    """Map config compression levels to fpng values."""

    normalised = normalise_compression_level(level)
    return {0: 0, 1: 1, 2: 2}.get(normalised, 1)


def map_png_compression_level(level: int) -> int:
    """Translate the user configured level into a PNG compress level."""

    normalised = normalise_compression_level(level)
    mapping = {0: 0, 1: 6, 2: 9}
    return mapping.get(normalised, 6)


def map_ffmpeg_compression(level: int) -> int:
    """Map config compression level to ffmpeg's PNG compression scale."""

    return map_png_compression_level(level)


def escape_drawtext(text: str) -> str:
    """Escape a drawtext argument for ffmpeg."""

    return (
        text.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("=", r"\=")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("'", r"\'")
        .replace("\n", r"\\n")
    )
