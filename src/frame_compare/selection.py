"""Clip initialisation and selection helpers used by the runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Final, List, Mapping, Optional, Sequence, Tuple

from rich.markup import escape

from src.datatypes import AnalysisConfig, RuntimeConfig
from src.frame_compare import vs as vs_core
from src.frame_compare.analysis import SelectionWindowSpec, compute_selection_window
from src.frame_compare.cache import (
    compute_probe_cache_key,
    load_probe_snapshot,
    persist_probe_snapshot,
)
from src.frame_compare.cli_runtime import CLIAppError, ClipProbeSnapshot

logger = logging.getLogger(__name__)

__all__: Final = [
    "extract_clip_fps",
    "init_clips",
    "probe_clip_metadata",
    "resolve_selection_windows",
    "log_selection_windows",
]

if TYPE_CHECKING:
    from src.frame_compare.cli_runtime import CliOutputManagerProtocol, ClipPlan
else:  # pragma: no cover - runtime-only fallback
    CliOutputManagerProtocol = Any  # type: ignore[assignment]
    ClipPlan = Any  # type: ignore[assignment]


def _extract_clip_fps(clip: object) -> Tuple[int, int]:
    """Return (fps_num, fps_den) from *clip*, defaulting to 24000/1001 when missing."""
    num = getattr(clip, "fps_num", None)
    den = getattr(clip, "fps_den", None)
    if isinstance(num, int) and isinstance(den, int) and den:
        return (num, den)
    return (24000, 1001)


extract_clip_fps = _extract_clip_fps


def _make_indexing_notifier(
    reporter: CliOutputManagerProtocol | None,
) -> Callable[[str], None]:
    def _indexing_note(filename: str) -> None:
        label = escape(filename)
        if reporter is not None:
            reporter.console.print(f"[dim][CACHE] Indexing {label}…[/]")
        else:
            logger.info("[CACHE] Indexing %s…", filename)

    return _indexing_note


_TONEMAP_PROP_BASE_NAMES = {"matrix", "primaries", "transfer", "colorrange"}
_TONEMAP_PROP_PREFIXES = ("masteringdisplay", "contentlightlevel")


def _merge_plan_frame_props(target_plan: ClipPlan, props: Mapping[str, Any]) -> Dict[str, Any]:
    if target_plan.source_frame_props is None:
        target_plan.source_frame_props = dict(props)
    else:
        for key, value in props.items():
            target_plan.source_frame_props.setdefault(key, value)
    return target_plan.source_frame_props or {}


def _capture_source_props_for_probe(target_plan: ClipPlan) -> Callable[[Mapping[str, Any]], None]:
    def _store(props: Mapping[str, Any]) -> None:
        _merge_plan_frame_props(target_plan, props)

    return _store


def _tonemap_prop_keys(props: Mapping[str, Any]) -> tuple[str, ...]:
    matched: list[str] = []
    for key in props:
        normalized = str(key).lstrip("_").lower()
        if normalized in _TONEMAP_PROP_BASE_NAMES:
            matched.append(str(key))
            continue
        for prefix in _TONEMAP_PROP_PREFIXES:
            if normalized.startswith(prefix):
                matched.append(str(key))
                break
    return tuple(sorted(dict.fromkeys(matched)))


def _log_cache_note(message: str, reporter: "CliOutputManagerProtocol | None") -> None:
    logger.info(message)
    if reporter is not None:
        reporter.console.print(f"[dim]{escape(message)}[/]")


def _snapshot_matches_plan(
    plan: ClipPlan,
    snapshot: ClipProbeSnapshot | None,
    fps_override: tuple[int, int] | None,
) -> bool:
    if snapshot is None:
        return False
    if plan.probe_cache_key and snapshot.cache_key and snapshot.cache_key != plan.probe_cache_key:
        return False
    if int(snapshot.trim_start) != int(plan.trim_start):
        return False
    if snapshot.trim_end != plan.trim_end:
        return False
    if snapshot.applied_fps != fps_override:
        return False
    return True


def _apply_snapshot_to_plan(
    plan: ClipPlan,
    snapshot: ClipProbeSnapshot,
    *,
    attach_clip: bool,
) -> None:
    plan.effective_fps = snapshot.effective_fps
    plan.applied_fps = snapshot.applied_fps
    plan.source_fps = snapshot.source_fps or plan.source_fps
    plan.source_num_frames = snapshot.source_num_frames
    plan.source_width = snapshot.source_width
    plan.source_height = snapshot.source_height
    _merge_plan_frame_props(plan, snapshot.source_frame_props)
    if attach_clip and snapshot.clip is not None:
        plan.clip = snapshot.clip
    plan.probe_snapshot = snapshot


def _build_snapshot_from_plan(plan: ClipPlan, applied_fps: tuple[int, int] | None) -> ClipProbeSnapshot:
    props = dict(plan.source_frame_props or {})
    snapshot = ClipProbeSnapshot(
        trim_start=int(plan.trim_start),
        trim_end=int(plan.trim_end) if plan.trim_end is not None else None,
        fps_override=plan.fps_override,
        applied_fps=applied_fps,
        effective_fps=plan.effective_fps,
        source_fps=plan.source_fps,
        source_num_frames=plan.source_num_frames,
        source_width=plan.source_width,
        source_height=plan.source_height,
        source_frame_props=props,
        tonemap_prop_keys=_tonemap_prop_keys(props),
        cache_key=plan.probe_cache_key,
        cache_path=plan.probe_cache_path,
    )
    return snapshot


def _initialise_clip_and_snapshot(
    plan: ClipPlan,
    *,
    fps_override: tuple[int, int] | None,
    cache_dir_str: str | None,
    cache_root: Path | None,
    indexing_notifier: Callable[[str], None],
    persist_snapshot: bool,
) -> tuple[ClipProbeSnapshot, bool]:
    source_props_hint = plan.source_frame_props if plan.source_frame_props else None
    frame_props_sink = _capture_source_props_for_probe(plan)
    clip = vs_core.init_clip(
        str(plan.path),
        trim_start=plan.trim_start,
        trim_end=plan.trim_end,
        fps_map=fps_override,
        cache_dir=cache_dir_str,
        indexing_notifier=indexing_notifier,
        frame_props_sink=frame_props_sink,
        source_frame_props_hint=source_props_hint,
    )
    plan.clip = clip
    plan.applied_fps = fps_override if fps_override is not None else plan.applied_fps
    plan.effective_fps = _extract_clip_fps(clip)
    if plan.source_fps is None:
        plan.source_fps = plan.effective_fps
    plan.source_num_frames = int(getattr(clip, "num_frames", 0) or 0)
    plan.source_width = int(getattr(clip, "width", 0) or 0)
    plan.source_height = int(getattr(clip, "height", 0) or 0)
    if plan.source_frame_props is None:
        plan.source_frame_props = {}
    snapshot = _build_snapshot_from_plan(plan, fps_override)
    snapshot.clip = clip
    plan.probe_snapshot = snapshot
    wrote = False
    if cache_root is not None and persist_snapshot:
        if not snapshot.cache_key:
            raise ValueError("Cannot persist probe snapshot without cache key")
        cache_path, wrote = persist_probe_snapshot(cache_root, snapshot)
        snapshot.cache_path = cache_path
    return snapshot, wrote


def _plan_needs_refresh(plan: ClipPlan, fps_override: tuple[int, int] | None) -> bool:
    snapshot = plan.probe_snapshot
    if snapshot is None:
        return True
    if int(snapshot.trim_start) != int(plan.trim_start):
        return True
    if snapshot.trim_end != plan.trim_end:
        return True
    if snapshot.applied_fps != fps_override:
        return True
    return False


def _ensure_probe_snapshot(
    plan: ClipPlan,
    *,
    fps_override: tuple[int, int] | None,
    cache_root: Path | None,
    cache_dir_str: str | None,
    indexing_notifier: Callable[[str], None],
    stats: Dict[str, int],
    force_reprobe: bool,
    reporter: "CliOutputManagerProtocol | None",
) -> ClipProbeSnapshot:
    snapshot = plan.probe_snapshot if plan.probe_snapshot is not None else None
    if not force_reprobe and snapshot is not None and _snapshot_matches_plan(plan, snapshot, fps_override):
        origin = "memory" if snapshot.clip is not None else "disk"
        stats["memory_hits" if origin == "memory" else "disk_hits"] += 1
        _apply_snapshot_to_plan(plan, snapshot, attach_clip=bool(snapshot.clip))
        tonemap_desc = ", ".join(snapshot.tonemap_prop_keys) or "none"
        _log_cache_note(
            f"[CACHE] Reused {origin} probe snapshot for {plan.path.name} (tonemap={tonemap_desc})",
            reporter,
        )
        return snapshot
    snapshot, wrote = _initialise_clip_and_snapshot(
        plan,
        fps_override=fps_override,
        cache_dir_str=cache_dir_str,
        cache_root=cache_root,
        indexing_notifier=indexing_notifier,
        persist_snapshot=True,
    )
    stats["opened"] += 1
    if wrote:
        stats["writes"] += 1
    else:
        stats["unchanged"] += 1
    tonemap_desc = ", ".join(snapshot.tonemap_prop_keys) or "none"
    _log_cache_note(
        f"[CACHE] Refreshed probe snapshot for {plan.path.name} (tonemap={tonemap_desc})",
        reporter,
    )
    return snapshot


def probe_clip_metadata(
    plans: Sequence[ClipPlan],
    runtime_cfg: RuntimeConfig,
    cache_dir: Path | None,
    *,
    reporter: CliOutputManagerProtocol | None = None,
) -> None:
    """Populate FPS, geometry, and HDR snapshot metadata for each clip plan.

    Cached probe snapshots from memory or disk are reused whenever possible so
    clip files are only opened when a snapshot is missing or stale. Any plans
    whose `plan.clip` objects remain uninitialized after this pass will be
    populated by `init_clips` before downstream processing.
    """

    if not plans:
        return

    vs_core.set_ram_limit(runtime_cfg.ram_limit_mb)
    cache_root = cache_dir
    cache_dir_str = str(cache_root) if cache_root is not None else None
    indexing_notifier = _make_indexing_notifier(reporter)
    force_reprobe = bool(getattr(runtime_cfg, "force_reprobe", False))
    if force_reprobe:
        _log_cache_note("[CACHE] force_reprobe enabled; bypassing cached probe metadata.", reporter)

    stats: Dict[str, int] = {key: 0 for key in ("opened", "memory_hits", "disk_hits", "writes", "unchanged")}

    for plan in plans:
        plan.probe_cache_key = compute_probe_cache_key(plan)
        if cache_root is not None and plan.probe_cache_key:
            plan.probe_cache_path = cache_root / "probe" / f"{plan.probe_cache_key}.json"
        else:
            plan.probe_cache_path = None
        if force_reprobe or cache_root is None or not plan.probe_cache_key:
            continue
        cached = load_probe_snapshot(cache_root, plan.probe_cache_key)
        if cached is not None:
            plan.probe_snapshot = cached

    reference_index = next((idx for idx, plan in enumerate(plans) if plan.use_as_reference), None)
    reference_fps: Optional[Tuple[int, int]] = None

    if reference_index is not None:
        reference_snapshot = _ensure_probe_snapshot(
            plans[reference_index],
            fps_override=None,
            cache_root=cache_root,
            cache_dir_str=cache_dir_str,
            indexing_notifier=indexing_notifier,
            stats=stats,
            force_reprobe=force_reprobe,
            reporter=reporter,
        )
        reference_fps = reference_snapshot.effective_fps or reference_snapshot.source_fps

    for idx, plan in enumerate(plans):
        if idx == reference_index:
            continue
        fps_override = plan.fps_override
        if fps_override is None and reference_fps is not None:
            fps_override = reference_fps

        snapshot = _ensure_probe_snapshot(
            plan,
            fps_override=fps_override,
            cache_root=cache_root,
            cache_dir_str=cache_dir_str,
            indexing_notifier=indexing_notifier,
            stats=stats,
            force_reprobe=force_reprobe,
            reporter=reporter,
        )
        plan.applied_fps = fps_override if fps_override is not None else snapshot.applied_fps

    _log_cache_note(
        "[CACHE] Probe summary: opened={opened} memory_hits={memory_hits} disk_hits={disk_hits} writes={writes} unchanged={unchanged}".format(
            **stats
        ),
        reporter,
    )


def init_clips(
    plans: Sequence[ClipPlan],
    runtime_cfg: RuntimeConfig,
    cache_dir: Path | None,
    *,
    reporter: CliOutputManagerProtocol | None = None,
) -> None:
    """Initialise VapourSynth clips and reuse previously probed metadata when possible."""

    vs_core.set_ram_limit(runtime_cfg.ram_limit_mb)
    cache_root = cache_dir
    cache_dir_str = str(cache_root) if cache_root is not None else None
    indexing_notifier = _make_indexing_notifier(reporter)
    force_reprobe = bool(getattr(runtime_cfg, "force_reprobe", False))

    reference_index = next((idx for idx, plan in enumerate(plans) if plan.use_as_reference), None)
    reference_fps: Optional[Tuple[int, int]] = None

    reuse_hits = 0
    reopened = 0

    for plan in plans:
        if not getattr(plan, "probe_cache_key", None):
            plan.probe_cache_key = compute_probe_cache_key(plan)
            if cache_root is not None and plan.probe_cache_key:
                plan.probe_cache_path = cache_root / "probe" / f"{plan.probe_cache_key}.json"

    if reference_index is not None:
        plan = plans[reference_index]
        needs_refresh = force_reprobe or plan.clip is None or _plan_needs_refresh(plan, None)
        if needs_refresh:
            snapshot, _ = _initialise_clip_and_snapshot(
                plan,
                fps_override=None,
                cache_dir_str=cache_dir_str,
                cache_root=cache_root,
                indexing_notifier=indexing_notifier,
                persist_snapshot=True,
            )
            reference_fps = snapshot.effective_fps or snapshot.source_fps
            reopened += 1
        else:
            reference_fps = plan.effective_fps or plan.source_fps
            reuse_hits += 1

    for idx, plan in enumerate(plans):
        if idx == reference_index:
            continue
        fps_override = plan.fps_override
        if fps_override is None and reference_fps is not None:
            fps_override = reference_fps

        needs_refresh = force_reprobe or plan.clip is None or _plan_needs_refresh(plan, fps_override)
        if not needs_refresh:
            if fps_override is not None:
                plan.applied_fps = fps_override
            reuse_hits += 1
            continue

        snapshot, _ = _initialise_clip_and_snapshot(
            plan,
            fps_override=fps_override,
            cache_dir_str=cache_dir_str,
            cache_root=cache_root,
            indexing_notifier=indexing_notifier,
            persist_snapshot=True,
        )
        plan.applied_fps = fps_override if fps_override is not None else snapshot.applied_fps
        reopened += 1

    _log_cache_note(
        f"[CACHE] init_clips summary: reused={reuse_hits} reopened={reopened} force_reprobe={force_reprobe}",
        reporter,
    )


def resolve_selection_windows(
    plans: Sequence[ClipPlan],
    analysis_cfg: AnalysisConfig,
) -> tuple[List[SelectionWindowSpec], tuple[int, int], bool]:
    specs: List[SelectionWindowSpec] = []
    min_total_frames: Optional[int] = None
    for plan in plans:
        clip = plan.clip
        if clip is None:
            raise CLIAppError("Clip initialisation failed")
        total_frames = int(getattr(clip, "num_frames", 0))
        if min_total_frames is None or total_frames < min_total_frames:
            min_total_frames = total_frames
        fps_num, fps_den = plan.effective_fps or _extract_clip_fps(clip)
        fps_val = fps_num / fps_den if fps_den else 0.0
        try:
            spec = compute_selection_window(
                total_frames,
                fps_val,
                analysis_cfg.ignore_lead_seconds,
                analysis_cfg.ignore_trail_seconds,
                analysis_cfg.min_window_seconds,
            )
        except TypeError as exc:
            detail = (
                f"Invalid analysis window values for {plan.path.name}: "
                f"lead={analysis_cfg.ignore_lead_seconds!r} "
                f"trail={analysis_cfg.ignore_trail_seconds!r} "
                f"min={analysis_cfg.min_window_seconds!r}"
            )
            raise CLIAppError(
                detail,
                rich_message=f"[red]{escape(detail)}[/red]",
            ) from exc
        specs.append(spec)

    if not specs:
        return [], (0, 0), False

    start = max(spec.start_frame for spec in specs)
    end = min(spec.end_frame for spec in specs)
    collapsed = False
    if end <= start:
        collapsed = True
        fallback_end = min_total_frames or 0
        start = 0
        end = fallback_end

    if end <= start:
        raise CLIAppError("No frames remain after applying ignore window")

    return specs, (start, end), collapsed


def log_selection_windows(
    plans: Sequence[ClipPlan],
    specs: Sequence[SelectionWindowSpec],
    intersection: tuple[int, int],
    *,
    collapsed: bool,
    analyze_fps: float,
    reporter: CliOutputManagerProtocol | None = None,
) -> None:
    """Log per-clip selection windows plus the common intersection summary."""

    def _emit(markup: str, plain: str, *, warning: bool = False) -> None:
        if reporter is not None:
            reporter.console.print(markup)
        else:
            log_fn = logger.warning if warning else logger.info
            log_fn(plain)

    if len(plans) != len(specs):
        warning_markup = (
            f"[yellow]Selection plans/specs length mismatch[/]: {len(plans)} plans vs {len(specs)} specs; "
            "extra entries will be skipped"
        )
        warning_plain = (
            f"Selection plans/specs length mismatch: {len(plans)} plans vs {len(specs)} specs; "
            "extra entries will be skipped"
        )
        _emit(warning_markup, warning_plain, warning=True)

    for plan, spec in zip(plans, specs, strict=False):
        raw_label = plan.metadata.get("label") or plan.path.name
        label_plain = (raw_label or plan.path.name).strip()
        label_markup = escape(label_plain)
        selection_markup = (
            f"[cyan]{label_markup}[/]: Selecting frames within [start={spec.start_seconds:.2f}s, "
            f"end={spec.end_seconds:.2f}s] (frames [{spec.start_frame}, {spec.end_frame})) — "
            f"lead={spec.applied_lead_seconds:.2f}s, trail={spec.applied_trail_seconds:.2f}s"
        )
        selection_plain = (
            f"{label_plain}: Selecting frames within start={spec.start_seconds:.2f}s, "
            f"end={spec.end_seconds:.2f}s (frames [{spec.start_frame}, {spec.end_frame})) — "
            f"lead={spec.applied_lead_seconds:.2f}s, trail={spec.applied_trail_seconds:.2f}s"
        )
        _emit(selection_markup, selection_plain)
        for warning in spec.warnings:
            warning_markup = f"[yellow]{label_markup}[/]: {warning}"
            warning_plain = f"{label_plain}: {warning}"
            _emit(warning_markup, warning_plain, warning=True)

    start_frame, end_frame = intersection
    if analyze_fps > 0 and end_frame > start_frame:
        start_seconds = start_frame / analyze_fps
        end_seconds = end_frame / analyze_fps
    else:
        start_seconds = float(start_frame)
        end_seconds = float(end_frame)

    common_markup = (
        f"[cyan]Common selection window[/]: frames [{start_frame}, {end_frame}) — "
        f"seconds [{start_seconds:.2f}s, {end_seconds:.2f}s)"
    )
    common_plain = (
        f"Common selection window: frames [{start_frame}, {end_frame}) — "
        f"seconds [{start_seconds:.2f}s, {end_seconds:.2f}s)"
    )
    _emit(common_markup, common_plain)

    if collapsed:
        collapsed_markup = (
            "[yellow]Ignore lead/trail settings did not overlap across all sources; using fallback range.[/yellow]"
        )
        collapsed_plain = (
            "Ignore lead/trail settings did not overlap across all sources; using fallback range."
        )
        _emit(collapsed_markup, collapsed_plain, warning=True)
