from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, MutableMapping, Optional, cast

from rich.console import Console

from src.datatypes import AppConfig
from src.frame_compare.analysis import FrameMetricsCacheInfo, SelectionDetail
from src.frame_compare.cli_runtime import (
    AudioAlignmentDisplayData,
    AudioAlignmentSummary,
    CliOutputManagerProtocol,
    ClipPlan,
    ClipRecord,
    JsonTail,
    SlowpicsTitleInputs,
    TrimSummary,
)
from src.frame_compare.preflight import PreflightResult
from src.frame_compare.result_snapshot import RunResultSnapshot
from src.frame_compare.services.alignment import AlignmentResult, AlignmentWorkflow
from src.frame_compare.services.metadata import MetadataResolver
from src.frame_compare.services.publishers import ReportPublisher, SlowpicsPublisher

if TYPE_CHECKING:
    from src.frame_compare.services.setup import SetupService

ReporterFactory = Callable[['RunRequest', Path, Console], CliOutputManagerProtocol]


@dataclass
class RunEnvironment:
    preflight: PreflightResult
    cfg: AppConfig
    root: Path
    out_dir: Path
    out_dir_created: bool
    out_dir_created_path: Optional[Path]
    result_snapshot_path: Path
    analysis_cache_path: Path
    offsets_path: Path
    vspreview_mode_value: str
    layout_path: Path
    reporter: CliOutputManagerProtocol
    service_mode_enabled: bool
    legacy_requested: bool
    collected_warnings: List[str]
    report_enabled: bool


@dataclass
class RunResult:
    files: List[Path]
    frames: List[int]
    out_dir: Path
    out_dir_created: bool
    out_dir_created_path: Optional[Path]
    root: Path
    config: AppConfig
    image_paths: List[str]
    slowpics_url: Optional[str] = None
    json_tail: JsonTail | None = None
    report_path: Optional[Path] = None
    snapshot: RunResultSnapshot | None = None
    snapshot_path: Optional[Path] = None


@dataclass
class RunRequest:
    config_path: str | None
    input_dir: str | None = None
    root_override: str | None = None
    audio_track_overrides: Iterable[str] | None = None
    quiet: bool = False
    verbose: bool = False
    no_color: bool = False
    report_enable_override: Optional[bool] = None
    skip_wizard: bool = False
    debug_color: bool = False
    tonemap_overrides: Optional[Dict[str, Any]] = None
    impl_module: ModuleType | None = None
    console: Console | None = None
    reporter: CliOutputManagerProtocol | None = None
    reporter_factory: ReporterFactory | None = None
    from_cache_only: bool = False
    force_cache_refresh: bool = False
    show_partial_sections: bool = False
    show_missing_sections: bool = True
    service_mode_override: bool | None = None
    diagnostic_frame_metrics: bool | None = None


@dataclass(slots=True)
class RunDependencies:
    """Container describing the service instances required by the runner."""

    metadata_resolver: MetadataResolver
    alignment_workflow: AlignmentWorkflow
    report_publisher: ReportPublisher
    slowpics_publisher: SlowpicsPublisher
    setup_service: "SetupService"


@dataclass(slots=True)
class RunContext:
    """Aggregated state shared across service boundaries."""

    plans: list[ClipPlan]
    metadata: list[dict[str, Any]]
    json_tail: JsonTail
    layout_data: MutableMapping[str, Any]
    metadata_title: str | None
    analyze_path: Path
    slowpics_title_inputs: SlowpicsTitleInputs
    slowpics_final_title: str
    slowpics_resolved_base: str | None
    slowpics_tmdb_disclosure_line: str | None
    slowpics_verbose_tmdb_tag: str | None
    tmdb_notes: list[str]
    alignment_summary: AudioAlignmentSummary | None = None
    alignment_display: AudioAlignmentDisplayData | None = None

    def update_alignment(self, result: AlignmentResult) -> None:
        """Persist alignment outputs in the shared run context."""

        self.plans = result.plans
        self.alignment_summary = result.summary
        self.alignment_display = result.display


@dataclass
class CoordinatorContext:
    """
    Comprehensive state container for the WorkflowCoordinator execution pipeline.
    Holds all state that persists between execution phases.
    """
    request: RunRequest
    dependencies: RunDependencies

    # Initialization (SetupPhase)
    env: RunEnvironment = field(init=False)

    # Discovery (DiscoveryPhase)
    # Replaces RunContext fields
    plans: List[ClipPlan] = field(default_factory=lambda: cast(List[ClipPlan], []))
    metadata: List[Dict[str, Any]] = field(default_factory=lambda: cast(List[Dict[str, Any]], []))
    json_tail: JsonTail = field(default_factory=lambda: cast(JsonTail, {}))
    layout_data: MutableMapping[str, Any] = field(default_factory=lambda: cast(MutableMapping[str, Any], {}))

    # TMDB / Metadata
    metadata_title: str | None = None
    analyze_path: Path | None = None # Initially None
    slowpics_title_inputs: SlowpicsTitleInputs | None = None
    slowpics_final_title: str | None = None
    slowpics_resolved_base: str | None = None
    slowpics_tmdb_disclosure_line: str | None = None
    slowpics_verbose_tmdb_tag: str | None = None
    tmdb_notes: List[str] = field(default_factory=lambda: cast(List[str], []))

    # Alignment (AlignmentPhase)
    alignment_summary: AudioAlignmentSummary | None = None
    alignment_display: AudioAlignmentDisplayData | None = None

    # Loader (ClipLoaderPhase)
    clip_records: List[ClipRecord] = field(default_factory=lambda: cast(List[ClipRecord], []))
    trim_details: List[TrimSummary] = field(default_factory=lambda: cast(List[TrimSummary], []))
    stored_props_seq: List[Optional[Dict[str, Any]]] = field(default_factory=lambda: cast(List[Optional[Dict[str, Any]]], []))

    # Analysis (AnalysisPhase)
    cache_info: FrameMetricsCacheInfo | None = None
    frames: List[int] = field(default_factory=lambda: cast(List[int], []))
    selection_details: Dict[int, SelectionDetail] = field(default_factory=lambda: cast(Dict[int, SelectionDetail], {}))

    # Render (RenderPhase)
    image_paths: List[str] = field(default_factory=lambda: cast(List[str], []))
    verification_records: List[Dict[str, Any]] = field(default_factory=lambda: cast(List[Dict[str, Any]], []))

    # Publish (PublishPhase)
    slowpics_url: str | None = None
    report_path: Path | None = None

    # Result (ResultPhase)
    result: RunResult | None = None

    def update_alignment(self, result: AlignmentResult) -> None:
        """Persist alignment outputs."""
        self.plans = result.plans
        self.alignment_summary = result.summary
        self.alignment_display = result.display
