"""General-purpose utility helpers for frame comparison."""

from __future__ import annotations

import re
from importlib import import_module
from typing import Any, Dict, List, Mapping, Optional

_YEAR_RE = re.compile(r"(19|20)\d{2}")
_IMDB_ID_RE = re.compile(r"(tt\d{7,9})", re.IGNORECASE)
_TVDB_ID_RE = re.compile(r"tvdb\s*(\d+)", re.IGNORECASE)


def _extract_release_group_brackets(file_name: str) -> Optional[str]:
    """Return the leading release-group tag (e.g. "[Group]") without brackets."""

    match = re.match(r"^\[(?P<group>[^\]]+)\]", file_name)
    return match.group("group") if match else None


def _normalize_episode_number(val: Any) -> str:
    """Convert episode identifiers into a stable string representation."""

    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        if not val:
            return ""
        return "-".join(str(item) for item in val)
    return str(val)


def _first_sequence_value(val: Any) -> Any:
    """
    Return the first element from ``val`` if it is a sequence.

    Parameters:
        val (Any): Potential sequence value.

    Returns:
        Any: First element for sequence inputs; otherwise the original value.
    """
    if isinstance(val, (list, tuple)):
        return val[0] if val else None
    return val


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    """
    Return ``value`` as a mapping when possible, otherwise ``None``.

    Parameters:
        value (Any): Candidate mapping.

    Returns:
        Mapping[str, Any] | None: Mapping representation of ``value`` or ``None`` when coercion fails.
    """
    if isinstance(value, Mapping):
        return value
    if isinstance(value, dict):
        return value
    return None


def _call_guessit(file_name: str) -> Mapping[str, Any] | None:
    """
    Invoke GuessIt for ``file_name`` and normalise the result mapping.

    Parameters:
        file_name (str): File name or path to analyse.

    Returns:
        Mapping[str, Any] | None: Normalised GuessIt result when parsing succeeds; otherwise ``None``.
    """
    try:
        module = import_module("guessit")
    except ImportError:
        return None
    parser = getattr(module, "guessit", None)
    if not callable(parser):
        return None
    try:
        result = parser(file_name)
    except (ValueError, TypeError, IndexError, AttributeError, LookupError):
        return None
    return _coerce_mapping(result)


def _call_anitopy(file_name: str) -> Mapping[str, Any]:
    """
    Invoke Anitopy for ``file_name`` and return a mapping of metadata.

    Parameters:
        file_name (str): File name or path to analyse.

    Returns:
        Mapping[str, Any]: Normalised Anitopy metadata mapping (empty when parsing fails).
    """
    try:
        module = import_module("anitopy")
    except ImportError:
        return {}
    parser = getattr(module, "parse", None)
    if not callable(parser):
        return {}
    try:
        result = parser(file_name)
    except (ValueError, TypeError, IndexError, AttributeError, LookupError):
        return {}
    mapped = _coerce_mapping(result)
    return dict(mapped) if mapped else {}


def _episode_designator_for_label(
    episode_value: Any,
    season_value: Any,
    normalized_episode: str,
) -> str:
    """Build a compact episode marker for labels (e.g. S01E03)."""

    season = _first_sequence_value(season_value)
    episode = _first_sequence_value(episode_value)
    try:
        season_int = int(season)
    except (TypeError, ValueError):
        season_int = None
    try:
        episode_int = int(episode)
    except (TypeError, ValueError):
        episode_int = None

    if season_int is not None and episode_int is not None:
        return f"S{season_int:02d}E{episode_int:02d}"
    return normalized_episode


def _build_label(
    file_name: str,
    release_group: str,
    anime_title: str,
    episode_marker: str,
    episode_title: str,
) -> str:
    """
    Compose a descriptive label for an anime episode file.

    Parameters:
        file_name (str): Original file name used as a fallback label.
        release_group (str): Release group tag extracted from the file name.
        anime_title (str): Primary title detected for the series.
        episode_marker (str): Episode designator (for example ``S01E03``).
        episode_title (str): Optional episode title appended to the label.

    Returns:
        str: Formatted label suitable for CLI presentation.
    """
    parts: List[str] = []
    if release_group:
        parts.append(f"[{release_group}]")
    if anime_title:
        parts.append(anime_title)
    if episode_marker:
        parts.append(episode_marker)

    label = " ".join(parts).strip()
    if episode_title:
        label = f"{label} – {episode_title}" if label else episode_title
    return label or file_name


def parse_filename_metadata(
    file_name: str,
    *,
    prefer_guessit: bool | None = None,
    always_full_filename: bool | None = None,
) -> Dict[str, str]:
    """Parse *file_name* metadata using GuessIt and Anitopy heuristics."""

    prefer_guess = True if prefer_guessit is None else bool(prefer_guessit)
    guessit_data = _call_guessit(file_name) if prefer_guess else None

    title = ""
    episode_val: Any = None
    episode_title = ""
    release_group = ""
    label_episode_marker = ""

    year_value: Any = None

    if guessit_data:
        title = str(guessit_data.get("title") or "")
        episode_val = guessit_data.get("episode")
        episode_title = str(guessit_data.get("episode_title") or "")
        release_group = str(guessit_data.get("release_group") or "")
        year_value = guessit_data.get("year")
        normalized_episode = _normalize_episode_number(episode_val)
        label_episode_marker = _episode_designator_for_label(
            episode_val, guessit_data.get("season"), normalized_episode
        )
    else:
        ani_data = _call_anitopy(file_name)
        title = str(ani_data.get("anime_title") or ani_data.get("title") or "")
        episode_val = ani_data.get("episode_number")
        episode_title = str(ani_data.get("episode_title") or "")
        release_group = str(ani_data.get("release_group") or "")
        normalized_episode = _normalize_episode_number(episode_val)
        label_episode_marker = normalized_episode
        year_value = ani_data.get("anime_season") or ani_data.get("year")

    normalized_episode = _normalize_episode_number(episode_val)
    resolved_release_group = release_group or _extract_release_group_brackets(file_name) or ""

    if isinstance(year_value, (list, tuple)):
        year_value = year_value[0] if year_value else None
    year_str = ""
    if isinstance(year_value, (int, float)):
        year_int = int(year_value)
        if 1900 <= year_int <= 2100:
            year_str = str(year_int)
    elif isinstance(year_value, str) and year_value.isdigit():
        year_str = year_value
    if not year_str:
        match = _YEAR_RE.search(file_name)
        if match:
            year_str = match.group(0)

    metadata = {
        "anime_title": title,
        "episode_number": normalized_episode,
        "episode_title": episode_title,
        "release_group": resolved_release_group,
        "file_name": file_name,
        "title": title,
        "year": year_str,
    }

    use_full_name = True if always_full_filename is None else bool(always_full_filename)
    label = file_name if use_full_name else _build_label(
        file_name,
        resolved_release_group,
        title,
        label_episode_marker,
        episode_title,
    )
    metadata["label"] = label

    imdb_match = _IMDB_ID_RE.search(file_name)
    metadata["imdb_id"] = imdb_match.group(1).lower() if imdb_match else ""
    tvdb_match = _TVDB_ID_RE.search(file_name)
    metadata["tvdb_id"] = tvdb_match.group(1) if tvdb_match else ""
    return metadata
