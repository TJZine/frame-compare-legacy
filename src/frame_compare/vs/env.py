"""Environment discovery and configuration helpers for VapourSynth."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

_VS_MODULE_NAME = "vapoursynth"


_ENV_VAR = "VAPOURSYNTH_PYTHONPATH"


_EXTRA_SEARCH_PATHS: list[str] = []


_vs_module: Any | None = None


_SOURCE_PREFERENCE = "lsmas"


_VALID_SOURCE_PLUGINS = {"lsmas", "ffms2"}


class ClipInitError(RuntimeError):
    """Raised when a clip cannot be created via VapourSynth."""


def _normalise_search_path(path: str) -> str:
    """
    Normalize a filesystem search path by expanding a user home and resolving to an absolute path when possible.

    Parameters:
        path (str): The input filesystem path, may contain a leading `~` for the user home.

    Returns:
        normalized_path (str): The expanded and resolved absolute path when resolution succeeds; otherwise the expanded path.
    """
    expanded = Path(path).expanduser()
    try:
        return str(expanded.resolve())
    except (OSError, RuntimeError):
        return str(expanded)


def _add_search_paths(paths: Iterable[str]) -> None:
    for raw in paths:
        if not raw:
            continue
        resolved = _normalise_search_path(raw)
        if resolved in _EXTRA_SEARCH_PATHS:
            continue
        _EXTRA_SEARCH_PATHS.append(resolved)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _set_source_preference(preference: str) -> None:
    """Record the preferred VapourSynth source plugin."""

    global _SOURCE_PREFERENCE
    normalized = preference.strip().lower()
    if normalized not in _VALID_SOURCE_PLUGINS:
        raise ValueError(
            f"Unsupported VapourSynth source preference '{preference}'."
            " Valid options are: " + ", ".join(sorted(_VALID_SOURCE_PLUGINS))
        )
    _SOURCE_PREFERENCE = normalized  # pyright: ignore[reportConstantRedefinition]


def _load_env_paths_from_env() -> None:
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        return
    entries = [entry.strip() for entry in raw.split(os.pathsep)]
    _add_search_paths(entry for entry in entries if entry)


def configure(
    *, search_paths: Sequence[str] | None = None, source_preference: str | None = None
) -> None:
    if search_paths:
        _add_search_paths(search_paths)
    if source_preference is not None:
        _set_source_preference(source_preference)


def _build_missing_vs_message() -> str:
    details: List[str] = []
    if _EXTRA_SEARCH_PATHS:
        details.append("Tried extra search paths: " + ", ".join(_EXTRA_SEARCH_PATHS))
    details.append(
        "Install VapourSynth for this interpreter or expose it via "
        "runtime.vapoursynth_python_paths in config.toml."
    )
    return " ".join(["VapourSynth is not available in this environment."] + details)


def _get_vapoursynth_module() -> Any:
    global _vs_module
    if _vs_module is not None:
        return _vs_module
    try:
        module = importlib.import_module(_VS_MODULE_NAME)
    except Exception as exc:  # pragma: no cover - import failure depends on env
        raise ClipInitError(_build_missing_vs_message()) from exc
    _vs_module = module
    return module


def set_ram_limit(limit_mb: int, *, core: Optional[Any] = None) -> None:
    """Apply a global VapourSynth cache limit based on *limit_mb*."""

    from .source import _resolve_core  # pyright: ignore[reportPrivateUsage]

    if limit_mb <= 0:
        raise ClipInitError("ram_limit_mb must be positive")

    resolved_core = _resolve_core(core)
    try:
        resolved_core.max_cache_size = int(limit_mb)
    except Exception as exc:  # pragma: no cover - defensive
        raise ClipInitError("Failed to apply VapourSynth RAM limit") from exc

_load_env_paths_from_env()

__all__ = [
    "ClipInitError",
    "_VS_MODULE_NAME",
    "_ENV_VAR",
    "_EXTRA_SEARCH_PATHS",
    "_vs_module",
    "_SOURCE_PREFERENCE",
    "_VALID_SOURCE_PLUGINS",
    "_normalise_search_path",
    "_add_search_paths",
    "_set_source_preference",
    "_load_env_paths_from_env",
    "configure",
    "_build_missing_vs_message",
    "_get_vapoursynth_module",
    "set_ram_limit",
]
