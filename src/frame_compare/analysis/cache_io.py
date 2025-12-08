"""Cache I/O helpers for frame comparison analysis."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    SupportsFloat,
    SupportsInt,
    Tuple,
    cast,
)

from src.datatypes import AnalysisConfig, AnalysisThresholds

if TYPE_CHECKING:
    from .selection import SelectionDetail


_SELECTION_METADATA_VERSION = "1"
_SELECTION_SOURCE_ID = "select_frames.v1"
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_CACHE_HASH_ENV_FLAG = "FRAME_COMPARE_CACHE_HASH"
_CACHE_HASH_ENV_FALSEY = {"", "0", "false", "no", "off"}
_METRICS_PAYLOAD_VERSION = 2


def _now_utc_iso() -> str:
    """Return current UTC time formatted with _TIME_FORMAT (Z-suffixed)."""
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime(_TIME_FORMAT)


def _selection_module():
    from . import selection as selection_module

    return selection_module


def _cache_hash_env_requested() -> bool:
    raw = os.environ.get(_CACHE_HASH_ENV_FLAG)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    return normalized not in _CACHE_HASH_ENV_FALSEY


@dataclass(frozen=True)
class ClipIdentity:
    """Snapshot of a clip participating in a cache entry."""

    role: str
    path: str
    name: str
    size: Optional[int]
    mtime: Optional[str]
    sha1: Optional[str]

    def to_payload(self) -> Dict[str, object]:
        """Serialize as a JSON-friendly dictionary."""

        return {
            "role": self.role,
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "mtime": self.mtime,
            "sha1": self.sha1,
        }


@dataclass(frozen=True)
class FrameMetricsCacheInfo:
    """Context needed to load/save cached frame metrics for analysis.

    ``clips`` carries precomputed :class:`ClipIdentity` entries so later cache
    probes can avoid re-stat'ing inputs.
    """

    path: Path
    files: Sequence[str]
    analyzed_file: str
    release_group: str
    trim_start: int
    trim_end: Optional[int]
    fps_num: int
    fps_den: int
    clips: Optional[List[ClipIdentity]] = None


@dataclass
class CachedMetrics:
    """
    Stored brightness and motion metrics captured from previous analyses.

    Attributes:
        brightness (List[tuple[int, float]]): Frame index and brightness pairs.
        motion (List[tuple[int, float]]): Frame index and motion score pairs.
        selection_frames (Optional[List[int]]): Frame indices selected during the cached run.
        selection_hash (Optional[str]): Hash of the selection inputs that produced ``selection_frames``.
        selection_categories (Optional[Dict[int, str]]): Optional per-frame category labels.
    """
    brightness: List[tuple[int, float]]
    motion: List[tuple[int, float]]
    selection_frames: Optional[List[int]]
    selection_hash: Optional[str]
    selection_categories: Optional[Dict[int, str]]
    selection_details: Optional[Dict[int, "SelectionDetail"]]


@dataclass(frozen=True)
class CacheLoadResult:
    """Outcome of probing previously persisted frame metrics for reuse."""

    metrics: Optional[CachedMetrics]
    status: Literal["missing", "reused", "stale", "error"]
    reason: Optional[str] = None


def _coerce_frame_index(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _coerce_optional_float(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if not math.isfinite(parsed):
            return None
        return parsed
    return None


def _coerce_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _coerce_optional_int(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _coerce_str_dict(value: object) -> Dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    result: Dict[str, object] = {}
    source = cast(Dict[Any, object], value)
    for key_obj, entry in source.items():
        if not isinstance(key_obj, str):
            return None
        result[key_obj] = entry
    return result


def _coerce_int_list(value: object) -> List[int] | None:
    if not isinstance(value, list):
        return None
    result: List[int] = []
    entries = cast(List[SupportsInt | str], value)
    for item in entries:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return None
    return result


def _coerce_selection_categories(value: object) -> Optional[Dict[int, str]]:
    if not isinstance(value, list):
        return None
    parsed: Dict[int, str] = {}
    entries = cast(List[List[object]], value)
    for item in entries:
        if len(item) != 2:
            continue
        frame_raw, label_raw = item
        try:
            frame_idx = int(cast(SupportsInt | str, frame_raw))
        except (TypeError, ValueError):
            continue
        parsed[frame_idx] = str(label_raw)
    return parsed or None


def _coerce_metric_series(value: object) -> List[tuple[int, float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("metrics payload must be a list")
    result: List[tuple[int, float]] = []
    entries = cast(List[Sequence[object]], value)
    for entry in entries:
        if len(entry) != 2:
            raise TypeError("invalid metrics entry")
        idx_obj, val_obj = entry
        idx_val = cast(SupportsInt | str, idx_obj)
        val = cast(SupportsFloat | str, val_obj)
        result.append((int(idx_val), float(val)))
    return result


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_handle = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temp_handle = handle.name
        os.replace(temp_handle, path)
    finally:
        if temp_handle and os.path.exists(temp_handle):
            try:
                os.remove(temp_handle)
            except OSError:
                pass


def _compute_file_sha1(path: Path, *, chunk_size: int = 1024 * 1024) -> Optional[str]:
    try:
        with path.open("rb") as handle:
            digest = hashlib.sha1()
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _threshold_snapshot(thresholds: AnalysisThresholds) -> Dict[str, float | str]:
    """Return a JSON-serialisable view of the threshold configuration."""

    mode_value = getattr(thresholds.mode, "value", thresholds.mode)
    snapshot: Dict[str, float | str] = {
        "mode": str(mode_value),
        "dark_quantile": float(thresholds.dark_quantile),
        "bright_quantile": float(thresholds.bright_quantile),
        "dark_luma_min": float(thresholds.dark_luma_min),
        "dark_luma_max": float(thresholds.dark_luma_max),
        "bright_luma_min": float(thresholds.bright_luma_min),
        "bright_luma_max": float(thresholds.bright_luma_max),
    }
    return snapshot


def _config_fingerprint(cfg: AnalysisConfig) -> str:
    """Return a stable hash for config fields that influence metrics generation."""

    relevant = {
        "frame_count_dark": cfg.frame_count_dark,
        "frame_count_bright": cfg.frame_count_bright,
        "frame_count_motion": cfg.frame_count_motion,
        "downscale_height": cfg.downscale_height,
        "step": cfg.step,
        "analyze_in_sdr": cfg.analyze_in_sdr,
        "motion_use_absdiff": cfg.motion_use_absdiff,
        "motion_scenecut_quantile": cfg.motion_scenecut_quantile,
        "screen_separation_sec": cfg.screen_separation_sec,
        "motion_diff_radius": cfg.motion_diff_radius,
        "random_seed": cfg.random_seed,
        "ignore_lead_seconds": cfg.ignore_lead_seconds,
        "ignore_trail_seconds": cfg.ignore_trail_seconds,
        "min_window_seconds": cfg.min_window_seconds,
        "thresholds": _threshold_snapshot(cfg.thresholds),
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def probe_cached_metrics(info: FrameMetricsCacheInfo, cfg: AnalysisConfig) -> CacheLoadResult:
    """Validate and, when possible, load cached frame metrics for reuse."""

    path = info.path
    selection_module = _selection_module()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return CacheLoadResult(metrics=None, status="missing", reason="not_found")
    except OSError as exc:
        return CacheLoadResult(metrics=None, status="error", reason=f"read_error:{exc.errno}")

    try:
        data_raw = json.loads(raw)
    except json.JSONDecodeError:
        return CacheLoadResult(metrics=None, status="error", reason="invalid_json")

    data = _coerce_str_dict(data_raw)
    if data is None:
        return CacheLoadResult(metrics=None, status="error", reason="invalid_payload")

    if data.get("version") != _METRICS_PAYLOAD_VERSION:
        return CacheLoadResult(metrics=None, status="stale", reason="version_mismatch")

    inputs_section = _coerce_str_dict(data.get("inputs"))
    if inputs_section is None:
        return CacheLoadResult(metrics=None, status="stale", reason="inputs_missing")
    clips_raw = inputs_section.get("clips")
    if not isinstance(clips_raw, list):
        return CacheLoadResult(metrics=None, status="stale", reason="inputs_missing")

    observed_clips: List[ClipIdentity] = []
    for entry in cast(List[object], clips_raw):
        clip_identity = _clip_identity_from_payload(entry)
        if clip_identity is None:
            return CacheLoadResult(metrics=None, status="stale", reason="inputs_invalid")
        observed_clips.append(clip_identity)

    expected_clips = _clip_identities_from_info(info)
    mismatch_reason = _compare_clip_identities(
        expected_clips,
        observed_clips,
        require_sha1=_cache_hash_env_requested(),
    )
    if mismatch_reason:
        return CacheLoadResult(metrics=None, status="stale", reason=mismatch_reason)

    if data.get("config_hash") != _config_fingerprint(cfg):
        return CacheLoadResult(metrics=None, status="stale", reason="config_mismatch")
    if data.get("analyzed_file") != info.analyzed_file:
        return CacheLoadResult(metrics=None, status="stale", reason="analyzed_mismatch")

    cached_group = str(data.get("release_group") or "").lower()
    if cached_group != (info.release_group or "").lower():
        return CacheLoadResult(metrics=None, status="stale", reason="release_group_mismatch")

    if data.get("trim_start") != info.trim_start:
        return CacheLoadResult(metrics=None, status="stale", reason="trim_start_mismatch")
    if data.get("trim_end") != info.trim_end:
        return CacheLoadResult(metrics=None, status="stale", reason="trim_end_mismatch")

    fps_obj = data.get("fps")
    fps_values: List[int] = []
    if isinstance(fps_obj, (list, tuple)):
        fps_candidate: List[int] = []
        sequence_obj = cast(Sequence[SupportsInt | str], fps_obj)
        for part in sequence_obj:
            try:
                fps_candidate.append(int(part))
            except (TypeError, ValueError):
                return CacheLoadResult(metrics=None, status="stale", reason="fps_invalid")
        fps_values = fps_candidate
    if fps_values != [info.fps_num, info.fps_den]:
        return CacheLoadResult(metrics=None, status="stale", reason="fps_mismatch")

    try:
        brightness = _coerce_metric_series(data.get("brightness"))
        motion = _coerce_metric_series(data.get("motion"))
    except (TypeError, ValueError):
        return CacheLoadResult(metrics=None, status="error", reason="metric_type_error")

    if not brightness:
        return CacheLoadResult(metrics=None, status="stale", reason="empty_metrics")

    selection_raw: object = data.get("selection") or {}
    selection_frames: Optional[List[int]] = None
    selection_hash: Optional[str] = None
    selection_categories: Optional[Dict[int, str]] = None
    selection_details: Optional[Dict[int, "SelectionDetail"]] = None
    selection_map = _coerce_str_dict(selection_raw)
    if selection_map is not None:
        frames_val = _coerce_int_list(selection_map.get("frames"))
        if frames_val is not None:
            selection_frames = frames_val
        hash_val = selection_map.get("hash")
        if isinstance(hash_val, str):
            selection_hash = hash_val
        categories_val = _coerce_selection_categories(selection_map.get("categories"))
        if categories_val is not None:
            selection_categories = categories_val
        details_val = selection_map.get("details")
        if details_val is not None:
            parsed_details = selection_module.deserialize_selection_details(details_val)
            if parsed_details:
                selection_details = parsed_details

    if selection_details is None:
        annotations_map = _coerce_str_dict(data.get("selection_annotations"))
        if annotations_map:
            parsed_ann: Dict[int, "SelectionDetail"] = {}
            for key, value in annotations_map.items():
                try:
                    frame_idx = int(key)
                except (TypeError, ValueError):
                    continue
                label: Optional[str] = None
                score: Optional[float] = None
                source = _SELECTION_SOURCE_ID
                timecode: Optional[str] = None
                notes: Optional[str] = None
                if isinstance(value, str):
                    label = value.split(";")[0].split("=", 1)[-1] if "=" in value else value
                else:
                    value_dict = _coerce_str_dict(value)
                    if value_dict is None:
                        continue
                    label_obj = value_dict.get("type") or value_dict.get("label")
                    if isinstance(label_obj, str):
                        label = label_obj
                    score = coerce_optional_float(value_dict.get("score"))
                    source_obj = value_dict.get("source")
                    if isinstance(source_obj, str) and source_obj.strip():
                        source = source_obj.strip()
                    timecode = coerce_optional_str(value_dict.get("ts_tc"))
                    notes = coerce_optional_str(value_dict.get("notes"))
                if not label:
                    label = "Auto"
                parsed_ann[frame_idx] = selection_module.SelectionDetail(
                    frame_index=frame_idx,
                    label=label,
                    score=score,
                    source=source,
                    timecode=timecode,
                    notes=notes,
                )
            if parsed_ann:
                selection_details = parsed_ann

    return CacheLoadResult(
        metrics=CachedMetrics(
            brightness,
            motion,
            selection_frames,
            selection_hash,
            selection_categories,
            selection_details,
        ),
        status="reused",
    )


def _load_cached_metrics(
    info: FrameMetricsCacheInfo, cfg: AnalysisConfig
) -> Optional[CachedMetrics]:
    """
    Load previously computed metrics when cache metadata still matches.

    Parameters:
        info (FrameMetricsCacheInfo): Cache metadata describing the expected file and clip characteristics.
        cfg (AnalysisConfig): Current analysis configuration whose fingerprint must match the cached payload.

    Returns:
        Optional[CachedMetrics]: Cached metrics if the persisted payload is valid; otherwise ``None``.
    """

    result = probe_cached_metrics(info, cfg)
    return result.metrics if result.status == "reused" else None


def _selection_sidecar_path(info: FrameMetricsCacheInfo) -> Path:
    """Return the filesystem location for the lightweight selection sidecar."""

    return info.path.parent / "generated.selection.v1.json"


def _infer_clip_role(index: int, name: str, analyzed_file: str, total: int) -> str:
    lowered = name.lower()
    analyzed_lower = analyzed_file.lower()
    if lowered == analyzed_lower:
        return "analyze"
    if index == 0:
        return "ref"
    if index == 1:
        return "tgt"
    return f"aux{index}"


def _clip_snapshot_payloads(info: FrameMetricsCacheInfo) -> Optional[List[Dict[str, object]]]:
    if info.clips is None or len(info.clips) != len(info.files):
        return None
    return [clip.to_payload() for clip in info.clips]


def _clip_identity_from_payload(entry: object) -> Optional[ClipIdentity]:
    entry_map = _coerce_str_dict(entry)
    if entry_map is None:
        return None
    path_obj = entry_map.get("path")
    name_obj = entry_map.get("name")
    role_obj = entry_map.get("role")
    if not isinstance(path_obj, str) or not isinstance(name_obj, str):
        return None
    role = str(role_obj) if isinstance(role_obj, str) else ""
    size = _coerce_optional_int(entry_map.get("size"))
    mtime_obj = entry_map.get("mtime")
    if isinstance(mtime_obj, str):
        mtime = mtime_obj or None
    elif isinstance(mtime_obj, (int, float)) and not isinstance(mtime_obj, bool):
        mtime = str(mtime_obj)
    else:
        mtime = None
    sha1_obj = entry_map.get("sha1")
    sha1 = None
    if isinstance(sha1_obj, str):
        sha1 = sha1_obj or None
    return ClipIdentity(
        role=role or "",
        path=path_obj,
        name=name_obj,
        size=size,
        mtime=mtime,
        sha1=sha1,
    )


def _build_clip_inputs(
    info: FrameMetricsCacheInfo,
    *,
    compute_sha1: bool = False,
    env_opt_in: bool = True,
) -> List[Dict[str, object]]:
    snapshot = _clip_snapshot_payloads(info)
    if snapshot is not None:
        # Return a shallow copy so callers can mutate without affecting cache info.
        return [dict(entry) for entry in snapshot]

    root = info.path.parent
    entries: List[Dict[str, object]] = []
    total = len(info.files)
    should_hash = compute_sha1 or (env_opt_in and _cache_hash_env_requested())
    for idx, file_name in enumerate(info.files):
        candidate = Path(file_name)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        try:
            stat_result = candidate.stat()
            size = int(stat_result.st_size)
            mtime = _dt.datetime.fromtimestamp(stat_result.st_mtime, tz=_dt.timezone.utc).isoformat()
        except OSError:
            size = None
            mtime = None
        sha1 = _compute_file_sha1(candidate) if should_hash else None
        entries.append(
            {
                "role": _infer_clip_role(idx, file_name, info.analyzed_file, total),
                "path": str(candidate),
                "name": file_name,
                "size": size,
                "mtime": mtime,
                "sha1": sha1,
            }
        )
    return entries


def _clip_identities_from_info(info: FrameMetricsCacheInfo) -> List[ClipIdentity]:
    if info.clips is not None and len(info.clips) == len(info.files):
        return list(info.clips)

    computed = _build_clip_inputs(info)
    total = len(info.files)
    identities: List[ClipIdentity] = []
    for idx, entry in enumerate(computed):
        clip = _clip_identity_from_payload(entry)
        if clip is None:
            name = info.files[idx] if idx < total else str(entry.get("name", "clip"))
            role = _infer_clip_role(idx, name, info.analyzed_file, total)
            path_obj = entry.get("path")
            path = str(path_obj) if isinstance(path_obj, str) else str(info.path.parent / name)
            clip = ClipIdentity(
                role=role,
                path=path,
                name=name,
                size=None,
                mtime=None,
                sha1=None,
            )
        identities.append(clip)
    return identities


def _compare_clip_identities(
    expected: Sequence[ClipIdentity],
    observed: Sequence[ClipIdentity],
    *,
    require_sha1: bool,
) -> Optional[str]:
    if len(expected) != len(observed):
        return "inputs_count_mismatch"
    for exp, obs in zip(expected, observed, strict=False):
        if obs.path != exp.path:
            return "inputs_path_mismatch"
        if obs.name != exp.name:
            return "inputs_name_mismatch"
        if exp.role != obs.role:
            return "inputs_role_mismatch"
        if exp.size is not None and obs.size is not None and obs.size != exp.size:
            return "inputs_size_mismatch"
        if exp.mtime is not None and obs.mtime is not None and obs.mtime != exp.mtime:
            return "inputs_mtime_mismatch"
        if require_sha1:
            if exp.sha1 is None or obs.sha1 is None or obs.sha1 != exp.sha1:
                return "inputs_sha1_mismatch"
    return None


def _selection_cache_key(
    *,
    clip_inputs: Sequence[Mapping[str, object]],
    cfg: AnalysisConfig,
    selection_source: str,
) -> str:
    payload = {
        "clips": clip_inputs,
        "config_hash": _config_fingerprint(cfg),
        "selection_source": selection_source,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _selection_payload_from_inputs(
    clip_inputs: Sequence[Mapping[str, object]],
    cfg: AnalysisConfig,
    selection_hash: str,
    selection_frames: Sequence[int],
    selection_details: Mapping[int, SelectionDetail] | None,
    analyzed_file: str,
) -> Dict[str, object]:
    cache_key = _selection_cache_key(
        clip_inputs=clip_inputs,
        cfg=cfg,
        selection_source=_SELECTION_SOURCE_ID,
    )
    selection_module = _selection_module()
    detail_records = selection_module.serialize_selection_details(selection_details or {})
    return {
        "version": _SELECTION_METADATA_VERSION,
        "cache_key": cache_key,
        "selection_hash": selection_hash,
        "selection_source": _SELECTION_SOURCE_ID,
        "generated_at": _now_utc_iso(),
        "analyzed_file": analyzed_file,
        "inputs": {
            "clips": list(clip_inputs),
            "config_fingerprint": _config_fingerprint(cfg),
        },
        "selections": detail_records,
        "frames": [int(frame) for frame in selection_frames],
    }


def _build_selection_sidecar_payload(
    info: FrameMetricsCacheInfo,
    cfg: AnalysisConfig,
    selection_hash: str,
    selection_frames: Sequence[int],
    selection_details: Mapping[int, SelectionDetail] | None,
) -> Dict[str, object]:
    clip_inputs = _clip_snapshot_payloads(info) or _build_clip_inputs(info)
    return _selection_payload_from_inputs(
        clip_inputs,
        cfg,
        selection_hash,
        selection_frames,
        selection_details,
        info.analyzed_file,
    )


def _save_selection_sidecar(
    info: FrameMetricsCacheInfo,
    cfg: AnalysisConfig,
    selection_hash: Optional[str],
    selection_frames: Optional[Sequence[int]],
    selection_details: Mapping[int, SelectionDetail] | None = None,
) -> None:
    """Persist rich selection metadata for fast reloads."""

    if selection_hash is None or selection_frames is None:
        return

    payload = _build_selection_sidecar_payload(
        info,
        cfg,
        selection_hash,
        selection_frames,
        selection_details,
    )

    target = _selection_sidecar_path(info)
    try:
        _atomic_write_json(target, payload)
    except OSError:
        return


def build_clip_inputs_from_paths(
    analyzed_file: str,
    clip_paths: Sequence[Path],
    *,
    compute_sha1: bool = False,
    env_opt_in: bool = True,
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    total = len(clip_paths)
    should_hash = compute_sha1 or (env_opt_in and _cache_hash_env_requested())
    for idx, clip_path in enumerate(clip_paths):
        resolved = clip_path.resolve()
        try:
            stat_result = resolved.stat()
            size = int(stat_result.st_size)
            mtime = _dt.datetime.fromtimestamp(stat_result.st_mtime, tz=_dt.timezone.utc).isoformat()
        except OSError:
            size = None
            mtime = None
        sha1 = _compute_file_sha1(resolved) if should_hash else None
        entries.append(
            {
                "role": _infer_clip_role(idx, resolved.name, analyzed_file, total),
                "path": str(resolved),
                "name": resolved.name,
                "size": size,
                "mtime": mtime,
                "sha1": sha1,
            }
        )
    return entries


def export_selection_metadata(
    target_path: Path,
    *,
    analyzed_file: str,
    clip_paths: Sequence[Path],
    cfg: AnalysisConfig,
    selection_hash: str,
    selection_frames: Sequence[int],
    selection_details: Mapping[int, SelectionDetail],
) -> None:
    clip_inputs = build_clip_inputs_from_paths(analyzed_file, clip_paths)
    payload = _selection_payload_from_inputs(
        clip_inputs,
        cfg,
        selection_hash,
        selection_frames,
        selection_details,
        analyzed_file,
    )
    _atomic_write_json(target_path, payload)


def write_selection_cache_file(
    target_path: Path,
    *,
    analyzed_file: str,
    clip_paths: Sequence[Path],
    cfg: AnalysisConfig,
    selection_hash: str,
    selection_frames: Sequence[int],
    selection_details: Mapping[int, SelectionDetail],
    selection_categories: Mapping[int, str],
) -> None:
    """Write a generated.compframes-style JSON payload with selection annotations only."""

    selection_module = _selection_module()
    clip_inputs = build_clip_inputs_from_paths(analyzed_file, clip_paths)
    payload = _selection_payload_from_inputs(
        clip_inputs,
        cfg,
        selection_hash,
        selection_frames,
        selection_details,
        analyzed_file,
    )
    normalized_frames = [int(frame) for frame in selection_frames]

    def _detail_or_default(frame: int) -> SelectionDetail:
        existing = selection_details.get(frame)
        if existing is not None:
            return existing
        return selection_module.SelectionDetail(
            frame_index=frame,
            label="Auto",
            score=None,
            source=_SELECTION_SOURCE_ID,
            timecode=None,
            clip_role=None,
            notes=None,
        )

    categories = [
        [frame, str(selection_categories.get(frame, _detail_or_default(frame).label))]
        for frame in normalized_frames
    ]
    selection_section_obj = payload.setdefault("selection", {})
    if not isinstance(selection_section_obj, dict):
        selection_section_obj = {}
        payload["selection"] = selection_section_obj
    selection_section = cast(Dict[str, object], selection_section_obj)
    selection_section["frames"] = normalized_frames
    selection_section["categories"] = categories
    selection_section["annotations"] = {
        str(frame): selection_module.format_selection_annotation(_detail_or_default(frame))
        for frame in normalized_frames
    }
    payload.setdefault("brightness", [])
    payload.setdefault("motion", [])
    _atomic_write_json(target_path, payload)


def _load_selection_sidecar(
    info: Optional[FrameMetricsCacheInfo], cfg: AnalysisConfig, selection_hash: Optional[str]
) -> Optional[Tuple[List[int], Dict[int, SelectionDetail]]]:
    """Load previously stored selection frames if the sidecar matches current state."""

    if info is None or not selection_hash:
        return None

    selection_module = _selection_module()
    path = _selection_sidecar_path(info)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        data_raw = json.loads(raw)
    except json.JSONDecodeError:
        return None

    data = _coerce_str_dict(data_raw)
    if data is None:
        return None

    version = str(data.get("version") or "")
    if version not in {"1", _SELECTION_METADATA_VERSION}:
        return None

    inputs_section = _coerce_str_dict(data.get("inputs")) or {}
    clip_inputs = inputs_section.get("clips")
    recomputed_inputs = _clip_snapshot_payloads(info) or _build_clip_inputs(info)
    if clip_inputs is not None:
        clip_names: List[object] = []
        if isinstance(clip_inputs, list):
            for entry in cast(List[object], clip_inputs):
                entry_map = _coerce_str_dict(entry)
                if entry_map is None:
                    continue
                clip_names.append(entry_map.get("name"))
        if clip_names != list(info.files):
            return None

    cache_key = data.get("cache_key")
    expected_cache_key = _selection_cache_key(
        clip_inputs=recomputed_inputs,
        cfg=cfg,
        selection_source=str(data.get("selection_source") or _SELECTION_SOURCE_ID),
    )
    if cache_key and cache_key != expected_cache_key:
        return None

    if data.get("analyzed_file") != info.analyzed_file:
        return None

    if data.get("selection_hash") != selection_hash:
        return None

    frames_raw = data.get("frames")
    if not isinstance(frames_raw, list):
        return None

    try:
        normalized_frames = [int(value) for value in cast(List[SupportsInt | str], frames_raw)]
    except (TypeError, ValueError):
        return None

    detail_records = selection_module.deserialize_selection_details(data.get("selections"))
    return normalized_frames, detail_records


def _save_cached_metrics(
    info: FrameMetricsCacheInfo,
    cfg: AnalysisConfig,
    brightness: Sequence[tuple[int, float]],
    motion: Sequence[tuple[int, float]],
    *,
    selection_hash: Optional[str] = None,
    selection_frames: Optional[Sequence[int]] = None,
    selection_categories: Optional[Dict[int, str]] = None,
    selection_details: Optional[Mapping[int, SelectionDetail]] = None,
) -> None:
    """
    Persist metrics and optional frame selections for reuse across runs.

    Parameters:
        info (FrameMetricsCacheInfo): Cache metadata describing the target persistence location.
        cfg (AnalysisConfig): Analysis configuration whose fingerprint will be stored alongside the metrics.
        brightness (Sequence[tuple[int, float]]): Per-frame brightness measurements.
        motion (Sequence[tuple[int, float]]): Per-frame motion measurements.
        selection_hash (Optional[str]): Fingerprint describing the selection parameters that produced ``selection_frames``.
        selection_frames (Optional[Sequence[int]]): Optional frame indices chosen for screenshot generation.
        selection_categories (Optional[Dict[int, str]]): Optional per-frame category labels to persist.
    """
    selection_module = _selection_module()
    path = info.path
    clip_inputs_payload = _build_clip_inputs(info)
    payload: Dict[str, object] = {
        "version": _METRICS_PAYLOAD_VERSION,
        "config_hash": _config_fingerprint(cfg),
        "files": list(info.files),
        "analyzed_file": info.analyzed_file,
        "release_group": info.release_group,
        "trim_start": info.trim_start,
        "trim_end": info.trim_end,
        "fps": [info.fps_num, info.fps_den],
        "brightness": [(int(idx), float(val)) for idx, val in brightness],
        "motion": [(int(idx), float(val)) for idx, val in motion],
        "inputs": {
            "clips": clip_inputs_payload,
            "analyzed_file": info.analyzed_file,
        },
    }
    annotations: Dict[str, str] = {}
    if selection_hash is not None and selection_frames is not None:
        serialized_details = selection_module.serialize_selection_details(selection_details or {})
        selection_details_map = selection_details or {}

        selection_section: Dict[str, object] = {
            "hash": selection_hash,
            "frames": [int(frame) for frame in selection_frames],
        }
        payload["selection"] = selection_section

        if selection_categories is not None:
            categories_payload: List[list[object]] = []
            for frame in selection_frames:
                frame_idx = int(frame)
                category_value = selection_categories.get(frame_idx, "")
                categories_payload.append([frame_idx, str(category_value)])
            if categories_payload:
                selection_section["categories"] = categories_payload

        if serialized_details:
            selection_section["details"] = serialized_details

            for record in serialized_details:
                frame_idx = _coerce_frame_index(record.get("frame_index"))
                if frame_idx is None or frame_idx < 0:
                    continue

                detail = selection_details_map.get(frame_idx)
                if detail is None:
                    label_obj: object = record.get("type")
                    source_obj: object = record.get("source")
                    detail = selection_module.SelectionDetail(
                        frame_index=frame_idx,
                        label=str(label_obj).strip() if isinstance(label_obj, str) and label_obj.strip() else "Auto",
                        score=coerce_optional_float(record.get("score")),
                        source=str(source_obj).strip() if isinstance(source_obj, str) and source_obj.strip() else _SELECTION_SOURCE_ID,
                        timecode=coerce_optional_str(record.get("ts_tc")),
                        clip_role=coerce_optional_str(record.get("clip_role")),
                        notes=coerce_optional_str(record.get("notes")),
                    )
                annotations[str(frame_idx)] = selection_module.format_selection_annotation(detail)

        if annotations:
            payload["selection_annotations"] = annotations

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        # Failing to persist cache data should not abort the pipeline.
        return

    _save_selection_sidecar(info, cfg, selection_hash, selection_frames, selection_details or {})


coerce_str_dict = _coerce_str_dict
coerce_int_list = _coerce_int_list
coerce_selection_categories = _coerce_selection_categories
coerce_metric_series = _coerce_metric_series
coerce_frame_index = _coerce_frame_index
coerce_optional_float = _coerce_optional_float
coerce_optional_str = _coerce_optional_str
infer_clip_role = _infer_clip_role
cache_hash_env_requested = _cache_hash_env_requested
compute_file_sha1 = _compute_file_sha1
threshold_snapshot = _threshold_snapshot
config_fingerprint = _config_fingerprint
selection_sidecar_path = _selection_sidecar_path
build_clip_inputs = _build_clip_inputs
selection_cache_key = _selection_cache_key
selection_payload_from_inputs = _selection_payload_from_inputs
build_selection_sidecar_payload = _build_selection_sidecar_payload
save_selection_sidecar = _save_selection_sidecar
load_selection_sidecar = _load_selection_sidecar
save_cached_metrics = _save_cached_metrics
load_cached_metrics = _load_cached_metrics
