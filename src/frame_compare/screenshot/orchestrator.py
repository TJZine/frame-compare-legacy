"""High-level screenshot orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Sequence,
)

from src.datatypes import (
    ColorConfig,
    ExportRange,
    ScreenshotConfig,
)
from src.frame_compare import vs as vs_core
from src.frame_compare.render.errors import ScreenshotError, ScreenshotWriterError

from . import debug, helpers, naming, render

logger = logging.getLogger(__name__)


def generate_screenshots(
    clips: Sequence[Any],
    frames: Sequence[int],
    files: Sequence[str],
    metadata: Sequence[Mapping[str, str]],
    out_dir: Path,
    cfg: ScreenshotConfig,
    color_cfg: ColorConfig,
    *,
    trim_offsets: Sequence[int] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    frame_labels: Mapping[int, str] | None = None,
    selection_details: Mapping[int, Mapping[str, Any]] | None = None,
    alignment_maps: Sequence[Any] | None = None,
    warnings_sink: List[str] | None = None,
    verification_sink: List[Dict[str, Any]] | None = None,
    pivot_notifier: Callable[[str], None] | None = None,
    debug_color: bool = False,
    source_frame_props: Sequence[Mapping[str, Any] | None] | None = None,
) -> List[str]:
    """
    Render and save screenshots for the given frames from each input clip using the configured writers.

    Render each requested frame for every clip, applying geometry planning, optional overlays, alignment mapping, and the selected writer backend (fpng or ffmpeg). Created files are written into out_dir and their paths are returned in the order they were produced.

    Parameters:
        clips: Sequence of clip objects prepared for rendering.
        frames: Sequence of frame indices to render for each clip.
        files: Sequence of source file paths corresponding to clips; must match length of clips.
        metadata: Sequence of metadata mappings (one per file); must match length of files.
        out_dir: Destination directory for written screenshot files.
        cfg: ScreenshotConfig controlling writer selection, geometry and format options.
        color_cfg: ColorConfig controlling overlays, tonemapping and related color options.
        trim_offsets: Optional per-file trim start offsets; if None, treated as zeros. Must match length of files.
        source_frame_props: Optional sequence mirroring ``files`` containing cached source frame props for each clip.
        progress_callback: Optional callable invoked with 1 for each saved file to indicate progress.
        frame_labels: Optional mapping from frame index to a user-visible selection label used in overlays and filenames.
        alignment_maps: Optional sequence of alignment mappers (one per clip) used to map source frame indices.
        warnings_sink: Optional list to which non-fatal warning messages will be appended.
        verification_sink: Optional list to which per-clip verification records will be appended; each record contains keys: file, frame, average, maximum, auto_selected.
        pivot_notifier: Optional callable invoked with a short text note whenever a full-chroma pivot is applied.
        debug_color: Enable detailed colour debugging (logs, intermediate PNGs, legacy conversions) when True.

    Returns:
        List[str]: Ordered list of file paths for all created screenshot files.
    """

    if len(clips) != len(files):
        raise ScreenshotError("clips and files must have matching lengths")
    if len(metadata) != len(files):
        raise ScreenshotError("metadata and files must have matching lengths")
    if not frames:
        return []

    if trim_offsets is None:
        trim_offsets = [0] * len(files)
    if len(trim_offsets) != len(files):
        raise ScreenshotError("trim_offsets and files must have matching lengths")
    if source_frame_props is not None and len(source_frame_props) != len(files):
        raise ScreenshotError("source_frame_props and files must have matching lengths")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise ScreenshotError(
            "Unable to create screenshot directory "
            f"'{out_dir}': {exc.strerror or exc}"
        ) from exc
    except OSError as exc:
        raise ScreenshotError(
            f"Unable to prepare screenshot directory '{out_dir}': {exc}"
        ) from exc
    created: List[str] = []

    processed_results: List[vs_core.ClipProcessResult] = []
    overlay_states: List[render.OverlayState] = []
    debug_enabled = bool(debug_color or getattr(color_cfg, "debug_color", False))
    debug_root = out_dir / "debug" if debug_enabled else None
    debug_dither = helpers.normalize_rgb_dither(cfg.rgb_dither)
    use_ffmpeg_runtime = bool(cfg.use_ffmpeg and not debug_enabled)
    frame_info_allowed_default = bool(cfg.add_frame_info and not debug_enabled)
    overlays_allowed_default = not debug_enabled

    for index, (clip, file_path) in enumerate(zip(clips, files, strict=True)):
        stored_props = None
        if source_frame_props is not None and index < len(source_frame_props):
            stored_props = source_frame_props[index]
        result = vs_core.process_clip_for_screenshot(
            clip,
            file_path,
            color_cfg,
            enable_overlay=True,
            enable_verification=True,
            logger_override=logger,
            warning_sink=warnings_sink,
            debug_color=debug_enabled,
            stored_source_props=stored_props,
        )
        processed_results.append(result)
        overlay_states.append(render.new_overlay_state())
        if result.verification is not None:
            logger.info(
                "[VERIFY] %s frame=%d avg=%.4f max=%.4f",
                file_path,
                result.verification.frame,
                result.verification.average,
                result.verification.maximum,
            )
            if verification_sink is not None:
                verification_sink.append(
                    {
                        "file": str(file_path),
                        "frame": int(result.verification.frame),
                        "average": float(result.verification.average),
                        "maximum": float(result.verification.maximum),
                        "auto_selected": bool(result.verification.auto_selected),
                    }
                )

    geometry = render.plan_geometry([result.clip for result in processed_results], cfg)

    for clip_index, (result, file_path, meta, plan, trim_start, source_clip) in enumerate(
        zip(processed_results, files, metadata, geometry, trim_offsets, clips, strict=True)
    ):
        mapper = None
        if alignment_maps is not None and clip_index < len(alignment_maps):
            mapper = alignment_maps[clip_index]
        if frame_labels:
            logger.debug('frame_labels keys: %s', list(frame_labels.keys()))
        crop = plan["crop"]
        scaled = plan["scaled"]
        pad = plan["pad"]
        width = int(plan["width"])
        height = int(plan["height"])
        trim_start = int(trim_start)
        raw_label, safe_label = naming.derive_labels(file_path, meta)

        debug_state: debug.ColorDebugState | None = None
        if debug_enabled and debug_root is not None:
            artifacts = getattr(result, "debug", None)
            core = getattr(result.clip, "core", None)
            if core is None and artifacts is not None and artifacts.normalized_clip is not None:
                core = getattr(artifacts.normalized_clip, "core", None)
            if artifacts is None:
                fallback_props = {}
                if hasattr(result, "source_props"):
                    try:
                        fallback_props = dict(getattr(result, "source_props", {}) or {})
                    except (TypeError, ValueError):
                        fallback_props = {}
                fallback_clip = result.clip if result.clip is not None else None
                if fallback_clip is not None and core is None:
                    core = getattr(fallback_clip, "core", None)
                logger.debug(
                    "Colour debug artifacts unavailable for %s; falling back to processed clip",
                    file_path,
                )
                artifacts = vs_core.ColorDebugArtifacts(
                    normalized_clip=fallback_clip,
                    normalized_props=fallback_props,
                    original_props=fallback_props,
                    color_tuple=(None, None, None, None),
                )
            if core is None:
                try:
                    import vapoursynth as vs  # type: ignore

                    core = getattr(vs, "core", None)
                except ImportError:
                    core = None
            debug_state = debug.ColorDebugState(
                enabled=core is not None,
                base_dir=(debug_root / safe_label),
                label=safe_label,
                core=core,
                compression_level=int(getattr(cfg, "compression_level", 1)),
                rgb_dither=debug_dither,
                logger_obj=logger,
                artifacts=artifacts,
            )

        overlay_state = overlay_states[clip_index]
        base_overlay_text = getattr(result, "overlay_text", None)
        source_props_raw = getattr(result, "source_props", {})
        resolved_clip, resolved_source_props = render.resolve_source_props(
            result.clip,
            source_props_raw,
            color_cfg=color_cfg,
            file_name=str(file_path),
            warning_sink=warnings_sink,
        )
        source_props = resolved_source_props
        is_sdr_pipeline = render.is_sdr_pipeline(result.tonemap, resolved_source_props)
        range_full, range_limited = helpers.range_constants()
        export_range = getattr(cfg, "export_range", ExportRange.FULL)
        expand_to_full = render.should_expand_to_full(export_range)
        clip_color_range = render.resolve_output_color_range(resolved_source_props, result.tonemap)
        original_color_range = clip_color_range
        if expand_to_full:
            clip_color_range = range_full
        _, _, _, source_color_meta = vs_core.resolve_color_metadata(resolved_source_props)
        if source_color_meta not in (range_full, range_limited):
            source_color_hint = original_color_range
        else:
            source_color_hint = int(source_color_meta)

        for frame in frames:
            frame_idx = int(frame)
            mapped_idx = frame_idx
            if mapper is not None:
                try:
                    mapped_idx, _, clamped = mapper.map_frame(frame_idx)
                except (IndexError, ValueError, LookupError) as exc:
                    logger.warning(
                        "Failed to map frame %s for %s via alignment: %s",
                        frame_idx,
                        file_path,
                        exc,
                    )
                    clamped = False
                else:
                    if clamped:
                        logger.debug(
                            "Alignment clamped frame %s→%s for %s",
                            frame_idx,
                            mapped_idx,
                            file_path,
                        )
            detail_info = selection_details.get(frame_idx) if selection_details else None
            selection_label = frame_labels.get(frame_idx) if frame_labels else None
            if selection_label is None and detail_info is not None:
                derived_label = detail_info.get("label") or detail_info.get("type")
                if derived_label:
                    selection_label = str(derived_label)
            if selection_label is not None:
                logger.debug('Selection label for frame %s: %s', frame_idx, selection_label)
            actual_idx, was_clamped = render.clamp_frame_index(resolved_clip, mapped_idx)
            if was_clamped:
                logger.debug(
                    "Frame %s exceeds available frames (%s) in %s; using %s",
                    mapped_idx,
                    getattr(resolved_clip, 'num_frames', 'unknown'),
                    file_path,
                    actual_idx,
                )
            if debug_state and debug_state.normalized_clip is not None:
                debug_state.capture_stage(
                    "post_normalisation",
                    actual_idx,
                    debug_state.normalized_clip,
                    debug_state.normalized_props,
                )

            # Fetch dynamic props from the source clip for this frame
            current_props = dict(source_props)
            try:
                # Use source_clip to ensure we get original metadata (e.g. DoVi RPU stats)
                # that might be lost or static in the processed clip.
                # We use actual_idx which is clamped and mapped.
                source_frame_idx = render.resolve_source_frame_index(actual_idx, trim_start)

                if source_frame_idx is not None:
                    frame_ref = source_clip.get_frame(source_frame_idx)
                    dynamic_props = dict(frame_ref.props)
                    current_props.update(dynamic_props)
                else:
                    logger.debug("Frame %s falls within padding; skipping dynamic props", actual_idx)
            except (RuntimeError, ValueError, KeyError) as exc:
                logger.debug("Failed to fetch dynamic props for frame %s: %s", actual_idx, exc)

            if debug_enabled:
                overlay_text = None
            else:
                overlay_text = render.compose_overlay_text(
                    base_overlay_text,
                    color_cfg,
                    plan,
                    selection_label,
                    current_props,
                    tonemap_info=result.tonemap,
                    selection_detail=detail_info,
                )
            file_name = naming.prepare_filename(frame_idx, safe_label)
            target_path = out_dir / file_name

            try:
                resolved_frame = render.resolve_source_frame_index(actual_idx, trim_start)
                use_ffmpeg = use_ffmpeg_runtime and resolved_frame is not None
                if use_ffmpeg_runtime and resolved_frame is None:
                    logger.debug(
                        "Frame %s for %s falls within synthetic trim padding; "
                        "using VapourSynth writer",
                        frame_idx,
                        file_path,
                    )
                if use_ffmpeg:
                    assert resolved_frame is not None
                    if overlay_text and overlay_state.get("overlay_status") != "ok":
                        logger.info("[OVERLAY] %s applied (ffmpeg)", file_path)
                        overlay_state["overlay_status"] = "ok"
                    render.save_frame_with_ffmpeg(
                        file_path,
                        resolved_frame,
                        crop,
                        scaled,
                        pad,
                        target_path,
                        cfg,
                        width,
                        height,
                        selection_label,
                        overlay_text=overlay_text,
                        geometry_plan=plan,
                        is_sdr=is_sdr_pipeline,
                        pivot_notifier=pivot_notifier,
                        frame_info_allowed=frame_info_allowed_default,
                        overlays_allowed=overlays_allowed_default and bool(overlay_text),
                        target_range=clip_color_range,
                        expand_to_full=expand_to_full,
                        source_color_range=source_color_hint,
                    )
                else:
                    render.save_frame_with_fpng(
                        resolved_clip,
                        actual_idx,
                        crop,
                        scaled,
                        pad,
                        target_path,
                        cfg,
                        raw_label,
                        frame_idx,
                        selection_label,
                        overlay_text=overlay_text,
                        overlay_state=overlay_state,
                        strict_overlay=bool(getattr(color_cfg, "strict", False)),
                        source_props=resolved_source_props,
                        geometry_plan=plan,
                        tonemap_info=result.tonemap,
                        pivot_notifier=pivot_notifier,
                        color_cfg=color_cfg,
                        file_name=str(file_path),
                        warning_sink=warnings_sink,
                        debug_state=debug_state,
                        frame_info_allowed=frame_info_allowed_default,
                        overlays_allowed=overlays_allowed_default,
                        expand_to_full=expand_to_full,
                    )
            except ScreenshotWriterError:
                raise
            except Exception as exc:
                message = (
                    f"[RENDER] Falling back to placeholder for frame {frame_idx} of {file_path}: {exc}"
                )
                logger.exception(message)
                if warnings_sink is not None:
                    warnings_sink.append(message)
                render.save_frame_placeholder(target_path)

            created.append(str(target_path))
            if progress_callback is not None:
                progress_callback(1)

        if debug_state is not None:
            debug_state.close()

        if warnings_sink is not None:
            warnings_sink.extend(render.get_overlay_warnings(overlay_state))

    return created
