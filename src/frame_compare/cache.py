"""Helpers for constructing analysis cache metadata."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, cast

from src.datatypes import AppConfig
from src.frame_compare.analysis import FrameMetricsCacheInfo
from src.frame_compare.analysis.cache_io import (
    ClipIdentity,
    cache_hash_env_requested,
    compute_file_sha1,
    infer_clip_role,
)
from src.frame_compare.cli_runtime import ClipPlan, ClipProbeSnapshot

from .preflight import resolve_subdir

__all__ = [
    "build_cache_info",
    "_build_cache_info",
    "compute_probe_cache_key",
    "load_probe_snapshot",
    "persist_probe_snapshot",
]

_PROBE_CACHE_SCHEMA_VERSION = 1
_PROBE_CACHE_SUBDIR = "probe"


def _build_cache_info(
    root: Path,
    plans: Sequence[ClipPlan],
    cfg: AppConfig,
    analyze_index: int,
) -> Optional[FrameMetricsCacheInfo]:
    """
    Build cache metadata describing frame-metrics that can be saved for reuse.

    Returns ``None`` when frame-data saving is disabled.
    """

    if not cfg.analysis.save_frames_data:
        return None

    analyzed = plans[analyze_index]
    fps_num, fps_den = analyzed.effective_fps or (24000, 1001)
    if fps_den <= 0:
        fps_den = 1

    cache_path = resolve_subdir(
        root,
        cfg.analysis.frame_data_filename,
        purpose="analysis.frame_data_filename",
    )
    clip_identities: List[ClipIdentity] = []
    should_hash = cache_hash_env_requested()
    total = len(plans)
    analyzed_name = analyzed.path.name
    for idx, plan in enumerate(plans):
        resolved = plan.path.resolve()
        stat_ok = True
        try:
            stat_result = resolved.stat()
            size = int(stat_result.st_size)
            mtime = _dt.datetime.fromtimestamp(stat_result.st_mtime, tz=_dt.timezone.utc).isoformat()
        except OSError:
            stat_ok = False
            size = None
            mtime = None
        sha1 = None
        if should_hash and stat_ok:
            try:
                sha1 = compute_file_sha1(resolved)
            except OSError:
                sha1 = None
        clip_identities.append(
            ClipIdentity(
                role=infer_clip_role(idx, plan.path.name, analyzed_name, total),
                path=str(resolved),
                name=plan.path.name,
                size=size,
                mtime=mtime,
                sha1=sha1,
            )
        )
    return FrameMetricsCacheInfo(
        path=cache_path,
        files=[plan.path.name for plan in plans],
        analyzed_file=analyzed.path.name,
        release_group=analyzed.metadata.get("release_group", ""),
        trim_start=analyzed.trim_start,
        trim_end=analyzed.trim_end,
        fps_num=fps_num,
        fps_den=fps_den,
        clips=clip_identities,
    )


def build_cache_info(
    root: Path,
    plans: Sequence[ClipPlan],
    cfg: AppConfig,
    analyze_index: int,
) -> Optional[FrameMetricsCacheInfo]:
    """Public wrapper around the cache metadata builder."""

    return _build_cache_info(root, plans, cfg, analyze_index)


def compute_probe_cache_key(plan: ClipPlan) -> str:
    """
    Build a deterministic cache key for a clip plan using file stats + trim/FPS inputs.

    The key is intentionally stable across processes so metadata snapshots can be rehydrated.
    """

    try:
        resolved = plan.path.resolve()
    except OSError:
        resolved = plan.path
    try:
        stat_result = resolved.stat()
        size = int(stat_result.st_size)
        mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
    except OSError:
        size = None
        mtime_ns = None
    payload = {
        "path": str(resolved),
        "size": size,
        "mtime_ns": mtime_ns,
        "trim_start": int(plan.trim_start),
        "trim_end": int(plan.trim_end) if plan.trim_end is not None else None,
        "fps_override": _tuple_to_list(plan.fps_override),
        "use_as_reference": bool(plan.use_as_reference),
    }
    serialized = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def load_probe_snapshot(cache_root: Path, cache_key: str) -> Optional[ClipProbeSnapshot]:
    """Return the cached probe snapshot for *cache_key*, or ``None`` when missing."""

    cache_path = _resolve_probe_cache_path(cache_root, cache_key)
    try:
        raw = cache_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if data.get("schema_version") != _PROBE_CACHE_SCHEMA_VERSION:
        return None
    try:
        snapshot = ClipProbeSnapshot(
            trim_start=int(data.get("trim_start", 0)),
            trim_end=int(data["trim_end"]) if data.get("trim_end") is not None else None,
            fps_override=_list_to_tuple(data.get("fps_override")),
            applied_fps=_list_to_tuple(data.get("applied_fps")),
            effective_fps=_list_to_tuple(data.get("effective_fps")),
            source_fps=_list_to_tuple(data.get("source_fps")),
            source_num_frames=int(data["source_num_frames"]) if data.get("source_num_frames") is not None else None,
            source_width=int(data["source_width"]) if data.get("source_width") is not None else None,
            source_height=int(data["source_height"]) if data.get("source_height") is not None else None,
            source_frame_props=dict(data.get("source_frame_props") or {}),
            tonemap_prop_keys=tuple(data.get("tonemap_prop_keys") or ()),
            metadata_digest=str(data.get("metadata_digest") or ""),
            cache_key=cache_key,
            cache_path=cache_path,
            cached_at=str(data.get("cached_at") or ""),
        )
    except (ValueError, TypeError, KeyError):
        return None
    return snapshot


def persist_probe_snapshot(cache_root: Path, snapshot: ClipProbeSnapshot) -> tuple[Path, bool]:
    """
    Persist *snapshot* to cache_root/probe/<key>.json.

    Returns ``(path, True)`` when a write occurred or ``(path, False)`` when the on-disk payload
    already matched the in-memory snapshot.
    """

    if snapshot.cache_key is None:
        raise ValueError("Snapshot cache_key must be set before persisting")
    cache_path = _resolve_probe_cache_path(cache_root, snapshot.cache_key)
    payload = _build_snapshot_payload(snapshot)
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = cache_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = None
    except OSError:
        existing = None
    if existing == serialized:
        return cache_path, False
    cache_path.write_text(serialized, encoding="utf-8")
    return cache_path, True


def _resolve_probe_cache_path(cache_root: Path, cache_key: str) -> Path:
    return cache_root / _PROBE_CACHE_SUBDIR / f"{cache_key}.json"


def _tuple_to_list(value: Optional[tuple[int, int]]) -> Optional[list[int]]:
    if value is None:
        return None
    return [int(value[0]), int(value[1])]


def _list_to_tuple(value: Any) -> Optional[tuple[int, int]]:
    if not value:
        return None
    try:
        first, second = value
        return int(first), int(second)
    except (ValueError, TypeError):
        return None


def _metadata_digest(snapshot: ClipProbeSnapshot) -> str:
    payload = {
        "applied_fps": _tuple_to_list(snapshot.applied_fps),
        "effective_fps": _tuple_to_list(snapshot.effective_fps),
        "source_fps": _tuple_to_list(snapshot.source_fps),
        "source_num_frames": snapshot.source_num_frames,
        "source_width": snapshot.source_width,
        "source_height": snapshot.source_height,
        "source_frame_props": snapshot.source_frame_props,
        "tonemap_prop_keys": list(snapshot.tonemap_prop_keys),
    }
    serialized = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _build_snapshot_payload(snapshot: ClipProbeSnapshot) -> Mapping[str, Any]:
    if not snapshot.metadata_digest:
        snapshot.metadata_digest = _metadata_digest(snapshot)
    if not snapshot.cached_at:
        snapshot.cached_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    return {
        "schema_version": _PROBE_CACHE_SCHEMA_VERSION,
        "trim_start": int(snapshot.trim_start),
        "trim_end": int(snapshot.trim_end) if snapshot.trim_end is not None else None,
        "fps_override": _tuple_to_list(snapshot.fps_override),
        "applied_fps": _tuple_to_list(snapshot.applied_fps),
        "effective_fps": _tuple_to_list(snapshot.effective_fps),
        "source_fps": _tuple_to_list(snapshot.source_fps),
        "source_num_frames": snapshot.source_num_frames,
        "source_width": snapshot.source_width,
        "source_height": snapshot.source_height,
        "source_frame_props": snapshot.source_frame_props,
        "tonemap_prop_keys": list(snapshot.tonemap_prop_keys),
        "metadata_digest": snapshot.metadata_digest,
        "cache_key": snapshot.cache_key,
        "cached_at": snapshot.cached_at,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, set):
        typed_set = cast(set[object], value)
        return sorted(str(item) for item in typed_set)
    return value
