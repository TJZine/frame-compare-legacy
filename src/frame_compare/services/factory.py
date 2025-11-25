"""Factory helpers for service construction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import src.frame_compare.alignment_preview as alignment_preview
import src.frame_compare.alignment as alignment_runner
import src.frame_compare.planner as planner_utils
import src.frame_compare.preflight as preflight_utils
import src.frame_compare.report as html_report
import src.frame_compare.selection as selection_utils
import src.frame_compare.tmdb_workflow as tmdb_workflow
from src.datatypes import ReportConfig, RuntimeConfig, SlowpicsConfig, TMDBConfig
from src.frame_compare.analysis import SelectionDetail
from src.frame_compare.analyze_target import pick_analyze_file
from src.frame_compare.cli_runtime import ClipPlan
from src.frame_compare.interfaces import (
    PublisherIO,
    ReportRendererProtocol,
    SlowpicsClientProtocol,
)
from src.frame_compare.slowpics import upload_comparison
from src.frame_compare.tmdb_workflow import TMDBLookupResult

from .alignment import AlignmentWorkflow
from .metadata import (
    CliPromptProtocol,
    FilesystemProbeProtocol,
    MetadataResolver,
    PlanBuilder,
    TMDBClientProtocol,
)
from .publishers import ReportPublisher, SlowpicsPublisher

__all__ = [
    "build_alignment_workflow",
    "build_metadata_resolver",
    "build_report_publisher",
    "build_slowpics_publisher",
]


class _TMDBWorkflowClient(TMDBClientProtocol):
    """Adapter that delegates TMDB lookups to tmdb_workflow."""

    def resolve(
        self,
        *,
        files: Sequence[Path],
        metadata: Sequence[dict[str, str]],
        tmdb_cfg: TMDBConfig,
        year_hint_raw: str | None,
    ) -> TMDBLookupResult:
        return tmdb_workflow.resolve_workflow(
            files=files,
            metadata=metadata,
            tmdb_cfg=tmdb_cfg,
            year_hint_raw=year_hint_raw,
        )


class _ClipProbeAdapter(FilesystemProbeProtocol):
    """Adapter for selection_utils.probe_clip_metadata."""

    def probe(
        self,
        plans: Sequence[ClipPlan],
        runtime_cfg: RuntimeConfig,
        cache_dir: Path,
        *,
        reporter: CliPromptProtocol | None = None,
    ) -> None:
        selection_utils.probe_clip_metadata(
            plans,
            runtime_cfg,
            cache_dir,
            reporter=reporter,
        )


def build_metadata_resolver() -> MetadataResolver:
    """Construct a MetadataResolver wired to the concrete adapters."""

    plan_builder: PlanBuilder = planner_utils.build_plans
    return MetadataResolver(
        tmdb_client=_TMDBWorkflowClient(),
        plan_builder=plan_builder,
        analyze_picker=pick_analyze_file,
        clip_probe=_ClipProbeAdapter(),
    )


def build_alignment_workflow() -> AlignmentWorkflow:
    """Construct an AlignmentWorkflow with default adapters."""

    return AlignmentWorkflow(
        apply_alignment=alignment_runner.apply_audio_alignment,
        format_output=alignment_runner.format_alignment_output,
        confirm_alignment=alignment_preview.confirm_alignment_with_screenshots,
    )


class _SlowpicsClient(SlowpicsClientProtocol):
    """Adapter delegating slow.pics uploads to the shared helper."""

    def upload(
        self,
        image_paths: Sequence[str],
        out_dir: Path,
        cfg: SlowpicsConfig,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> str:
        return upload_comparison(
            list(image_paths),
            out_dir,
            cfg,
            progress_callback=progress_callback,
        )


class _ReportRenderer(ReportRendererProtocol):
    """Adapter that forwards to the HTML report implementation."""

    def generate(
        self,
        *,
        report_dir: Path,
        report_cfg: ReportConfig,
        frames: Sequence[int],
        selection_details: Mapping[int, SelectionDetail],
        image_paths: Sequence[str],
        plans: Sequence[Mapping[str, object]],
        metadata_title: str | None,
        include_metadata: str,
        slowpics_url: str | None,
    ) -> Path:
        return html_report.generate_html_report(
            report_dir=report_dir,
            report_cfg=report_cfg,
            frames=frames,
            selection_details=selection_details,
            image_paths=image_paths,
            plans=plans,
            metadata_title=metadata_title,
            include_metadata=include_metadata,
            slowpics_url=slowpics_url,
        )


class _PublisherIO(PublisherIO):
    """Filesystem helper backing publisher services."""

    def file_size(self, path: str | Path) -> int:
        try:
            return Path(path).stat().st_size
        except OSError:
            return 0

    def path_exists(self, path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    def resolve_report_dir(self, root: Path, relative: str, *, purpose: str) -> Path:
        return preflight_utils.resolve_subdir(root, relative, purpose=purpose)


def build_slowpics_publisher() -> SlowpicsPublisher:
    """Construct the slow.pics publisher wiring to concrete adapters."""

    return SlowpicsPublisher(client=_SlowpicsClient(), io=_PublisherIO())


def build_report_publisher() -> ReportPublisher:
    """Construct the report publisher with filesystem and renderer adapters."""

    return ReportPublisher(renderer=_ReportRenderer(), io=_PublisherIO())
