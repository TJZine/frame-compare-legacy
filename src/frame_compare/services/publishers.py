"""Publisher services wrapping slow.pics uploads and HTML report generation."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence, cast

from rich.markup import escape

from src.datatypes import ReportConfig, SlowpicsConfig
from src.frame_compare.analysis import SelectionDetail
from src.frame_compare.cli_runtime import (
    CLIAppError,
    CliOutputManagerProtocol,
    ClipPlan,
    JsonTail,
    ReportJSON,
    SlowpicsJSON,
    SlowpicsTitleBlock,
    SlowpicsTitleInputs,
)
from src.frame_compare.interfaces import (
    PublisherIO,
    ReportRendererProtocol,
    SlowpicsClientProtocol,
)
from src.frame_compare.layout_utils import color_text as _color_text
from src.frame_compare.layout_utils import format_kv as _format_kv
from src.frame_compare.layout_utils import plan_label as _plan_label
from src.frame_compare.slowpics import SlowpicsAPIError, build_shortcut_filename


@dataclass(slots=True)
class SlowpicsPublisherRequest:
    """Inputs required to publish a slow.pics collection."""

    reporter: CliOutputManagerProtocol
    json_tail: JsonTail
    layout_data: MutableMapping[str, Any]
    title_inputs: SlowpicsTitleInputs
    final_title: str
    resolved_base: str | None
    tmdb_disclosure_line: str | None
    verbose_tmdb_tag: str | None
    image_paths: Sequence[str]
    out_dir: Path
    config: SlowpicsConfig


@dataclass(slots=True)
class SlowpicsPublisherResult:
    """Outputs emitted after publishing to slow.pics."""

    url: str | None


@dataclass(slots=True)
class ReportPublisherRequest:
    """Inputs required for HTML report generation."""

    reporter: CliOutputManagerProtocol
    json_tail: JsonTail
    layout_data: MutableMapping[str, Any]
    report_enabled: bool
    root: Path
    plans: Sequence[ClipPlan]
    frames: Sequence[int]
    selection_details: Mapping[int, SelectionDetail]
    image_paths: Sequence[str]
    metadata_title: str | None
    slowpics_url: str | None
    config: ReportConfig
    collected_warnings: list[str]


@dataclass(slots=True)
class ReportPublisherResult:
    """Outputs emitted after report publishing."""

    report_path: Path | None


def _ensure_slowpics_block(json_tail: JsonTail, cfg: SlowpicsConfig) -> SlowpicsJSON:
    block = cast(SlowpicsJSON | None, json_tail.get("slowpics"))
    if block is None:
        block = cast(
            SlowpicsJSON,
            {
                "enabled": bool(cfg.auto_upload),
                "title": SlowpicsTitleBlock(
                    inputs=SlowpicsTitleInputs(
                        resolved_base=None,
                        collection_name=None,
                        collection_suffix=getattr(cfg, "collection_suffix", ""),
                    ),
                    final=None,
                ),
                "url": None,
                "shortcut_path": None,
                "shortcut_written": False,
                "shortcut_error": None,
                "deleted_screens_dir": False,
                "is_public": bool(cfg.is_public),
                "is_hentai": bool(cfg.is_hentai),
                "remove_after_days": int(cfg.remove_after_days),
            },
        )
        json_tail["slowpics"] = block
        return block

    if "url" not in block:
        block["url"] = None
    if "shortcut_path" not in block:
        block["shortcut_path"] = None
    if "shortcut_written" not in block:
        block["shortcut_written"] = False
    if "shortcut_error" not in block:
        block["shortcut_error"] = None
    if "deleted_screens_dir" not in block:
        block["deleted_screens_dir"] = False
    return block


class SlowpicsPublisher:
    """Service that encapsulates slow.pics logging and uploads."""

    def __init__(self, client: SlowpicsClientProtocol, io: PublisherIO) -> None:
        self._client = client
        self._io = io

    def publish(self, request: SlowpicsPublisherRequest) -> SlowpicsPublisherResult:
        reporter = request.reporter
        layout_data = request.layout_data
        json_tail = request.json_tail
        slowpics_cfg = request.config
        slowpics_url: str | None = None

        reporter.line(_color_text("slow.pics collection (preview):", "blue"))
        inputs_parts = [
            _format_kv(
                "collection_name",
                request.title_inputs["collection_name"],
                label_style="dim blue",
                value_style="bright_white",
            ),
            _format_kv(
                "collection_suffix",
                request.title_inputs["collection_suffix"],
                label_style="dim blue",
                value_style="bright_white",
            ),
        ]
        reporter.line("  " + "  ".join(inputs_parts))
        resolved_display = request.resolved_base or "(n/a)"
        reporter.line(
            "  "
            + _format_kv(
                "resolved_base",
                resolved_display,
                label_style="dim blue",
                value_style="bright_white",
            )
        )
        reporter.line(
            "  "
            + _format_kv(
                "final",
                f'"{request.final_title}"',
                label_style="dim blue",
                value_style="bold bright_white",
            )
        )
        if request.verbose_tmdb_tag:
            reporter.verbose_line(f"  {escape(request.verbose_tmdb_tag)}")
        if request.tmdb_disclosure_line:
            reporter.verbose_line(request.tmdb_disclosure_line)
        if slowpics_cfg.auto_upload:
            layout_data.setdefault("slowpics", {})["status"] = "preparing"
            reporter.update_values(layout_data)
            reporter.console.print("[cyan]Preparing slow.pics upload...[/cyan]")
            upload_total = len(request.image_paths)

            file_sizes = [self._io.file_size(path) for path in request.image_paths] if upload_total else []
            progress_tracker = UploadProgressTracker(file_sizes)
            total_bytes = progress_tracker.total_bytes
            console_width = getattr(reporter.console.size, "width", 80) or 80
            stats_width_limit = max(24, console_width - 32)

            def _format_duration(seconds: float | None) -> str:
                if seconds is None or not math.isfinite(seconds):
                    return "--:--"
                total = max(0, int(seconds + 0.5))
                hours, remainder = divmod(total, 3600)
                minutes, secs = divmod(remainder, 60)
                if hours:
                    return f"{hours:d}:{minutes:02d}:{secs:02d}"
                return f"{minutes:02d}:{secs:02d}"

            def _format_stats(files_done: int, bytes_done: int, elapsed: float) -> str:
                speed_bps = bytes_done / elapsed if elapsed > 0 else 0.0
                mbps = speed_bps / (1024 * 1024)
                remaining_bytes = max(total_bytes - bytes_done, 0)
                eta_seconds = (remaining_bytes / speed_bps) if speed_bps > 0 else None
                stats = f"{mbps:5.2f} MiB/s | ETA { _format_duration(eta_seconds)} | Elapsed {_format_duration(elapsed)}"
                return stats if len(stats) <= stats_width_limit else stats[: stats_width_limit - 3] + "..."

            reporter.update_progress_state(
                "upload_bar",
                current=0,
                total=upload_total,
                stats=_format_stats(0, 0, 0.0),
            )

            def _advance_upload(count: int) -> None:
                files_done, bytes_done, elapsed = progress_tracker.advance(count)
                reporter.update_progress_state(
                    "upload_bar",
                    current=min(files_done, upload_total),
                    total=upload_total,
                    stats=_format_stats(files_done, bytes_done, elapsed),
                )

            try:
                slowpics_url = self._client.upload(
                    list(request.image_paths),
                    request.out_dir,
                    slowpics_cfg,
                    progress_callback=_advance_upload,
                )
            except SlowpicsAPIError as exc:
                layout_data.setdefault("slowpics", {})["status"] = "failed"
                reporter.update_values(layout_data)
                raise CLIAppError(
                    f"slow.pics upload failed: {exc}",
                    rich_message=f"[red]slow.pics upload failed:[/red] {exc}",
                ) from exc
            else:
                layout_data.setdefault("slowpics", {})["status"] = "completed"
                reporter.update_values(layout_data)
                reporter.line(_color_text(f"[✓] slow.pics: uploading {upload_total} images", "green"))
                reporter.line(_color_text("[✓] slow.pics: assembling collection", "green"))

        slowpics_block = _ensure_slowpics_block(json_tail, slowpics_cfg)
        slowpics_block["url"] = slowpics_url
        shortcut_path_obj: Path | None = None
        shortcut_error: str | None = None
        if slowpics_cfg.create_url_shortcut and slowpics_url:
            shortcut_filename = build_shortcut_filename(slowpics_cfg.collection_name, slowpics_url)
            if shortcut_filename:
                shortcut_path_obj = request.out_dir / shortcut_filename
            else:
                shortcut_error = "invalid_shortcut_name"
        if shortcut_path_obj is not None:
            slowpics_block["shortcut_path"] = str(shortcut_path_obj)
            shortcut_written = self._io.path_exists(shortcut_path_obj)
        else:
            slowpics_block["shortcut_path"] = None
            shortcut_written = False
            if not slowpics_cfg.create_url_shortcut:
                shortcut_error = "disabled"
        if shortcut_written:
            shortcut_error = None
        elif shortcut_path_obj is not None and shortcut_error is None:
            shortcut_error = "write_failed"
        slowpics_block["shortcut_written"] = shortcut_written
        slowpics_block["shortcut_error"] = shortcut_error

        return SlowpicsPublisherResult(url=slowpics_url)


class ReportPublisher:
    """Service that encapsulates report generation and layout updates."""

    def __init__(self, renderer: ReportRendererProtocol, io: PublisherIO) -> None:
        self._renderer = renderer
        self._io = io

    def publish(self, request: ReportPublisherRequest) -> ReportPublisherResult:
        reporter = request.reporter
        json_tail = request.json_tail
        report_cfg = request.config
        report_index_path: Path | None = None
        report_block = json_tail.setdefault("report", cast(ReportJSON, {}))
        report_block["enabled"] = request.report_enabled
        report_block["path"] = None
        report_block["output_dir"] = report_cfg.output_dir
        report_block["open_after_generate"] = bool(getattr(report_cfg, "open_after_generate", True))
        request.layout_data["report"] = report_block

        if request.report_enabled:
            try:
                report_dir = self._io.resolve_report_dir(
                    request.root,
                    report_cfg.output_dir,
                    purpose="report.output_dir",
                )
                plan_payload = [
                    {
                        "label": _plan_label(plan),
                        "metadata": dict(plan.metadata),
                        "path": plan.path,
                    }
                    for plan in request.plans
                ]
                try:
                    report_index_path = self._renderer.generate(
                        report_dir=report_dir,
                        report_cfg=report_cfg,
                        frames=list(request.frames),
                        selection_details=request.selection_details,
                        image_paths=list(request.image_paths),
                        plans=plan_payload,
                        metadata_title=request.metadata_title,
                        include_metadata=str(getattr(report_cfg, "include_metadata", "minimal")),
                        slowpics_url=request.slowpics_url,
                    )
                except SlowpicsAPIError as exc:
                    request.reporter.error(f"Failed to generate report: {exc}")
                    report_index_path = None
            except CLIAppError as exc:
                message = f"HTML report generation failed: {exc}"
                reporter.warn(message)
                request.collected_warnings.append(message)
                report_block["enabled"] = False
                report_block["path"] = None
            except (OSError, ValueError, TypeError) as exc:  # pragma: no cover - defensive
                message = f"HTML report generation failed: {exc}"
                reporter.warn(message)
                request.collected_warnings.append(message)
                report_block["enabled"] = False
                report_block["path"] = None
            else:
                report_block["enabled"] = True
                report_block["path"] = str(report_index_path)
        else:
            report_block["enabled"] = False
            report_block["path"] = None

        return ReportPublisherResult(report_path=report_index_path)
class UploadProgressTracker:
    """Track uploaded file/byte counts in a thread-safe manner."""

    def __init__(self, file_sizes: Sequence[int]) -> None:
        self._file_sizes = tuple(file_sizes)
        self.total_files = len(self._file_sizes)
        self.total_bytes = sum(self._file_sizes)
        self._uploaded_files = 0
        self._uploaded_bytes = 0
        self._start_time = time.perf_counter()
        self._lock = threading.Lock()

    def advance(self, count: int) -> tuple[int, int, float]:
        increment = max(count, 0)
        with self._lock:
            target = min(self._uploaded_files + increment, self.total_files)
            while self._uploaded_files < target:
                index = self._uploaded_files
                size = self._file_sizes[index] if index < self.total_files else 0
                self._uploaded_bytes = min(self._uploaded_bytes + size, self.total_bytes)
                self._uploaded_files += 1
            elapsed = max(time.perf_counter() - self._start_time, 1e-6)
            return self._uploaded_files, self._uploaded_bytes, elapsed
