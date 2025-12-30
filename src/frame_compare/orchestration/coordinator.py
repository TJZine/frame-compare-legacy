from __future__ import annotations

import logging

from src.datatypes import AppConfig
from src.frame_compare.cli_runtime import CliOutputManagerProtocol
from src.frame_compare.orchestration.phases.alignment import AlignmentPhase
from src.frame_compare.orchestration.phases.analysis import AnalysisPhase
from src.frame_compare.orchestration.phases.discovery import DiscoveryPhase
from src.frame_compare.orchestration.phases.loader import ClipLoaderPhase
from src.frame_compare.orchestration.phases.publish import PublishPhase
from src.frame_compare.orchestration.phases.render import RenderPhase
from src.frame_compare.orchestration.phases.result import ResultPhase
from src.frame_compare.orchestration.phases.setup import SetupPhase
from src.frame_compare.orchestration.state import (
    CoordinatorContext,
    RunDependencies,
    RunRequest,
    RunResult,
)
from src.frame_compare.services.alignment import AlignmentWorkflow
from src.frame_compare.services.factory import (
    build_alignment_workflow,
    build_metadata_resolver,
    build_report_publisher,
    build_slowpics_publisher,
)
from src.frame_compare.services.metadata import MetadataResolver
from src.frame_compare.services.publishers import (
    ReportPublisher,
    SlowpicsPublisher,
)
from src.frame_compare.services.setup import DefaultSetupService

logger = logging.getLogger('frame_compare')


def default_run_dependencies(
    *,
    cfg: AppConfig | None = None,
    reporter: CliOutputManagerProtocol | None = None,
    cache_dir: object | None = None, # Type hint adjusted to match usage/definition if needed, but original used Path | None
    metadata_resolver: MetadataResolver | None = None,
    alignment_workflow: AlignmentWorkflow | None = None,
    report_publisher: ReportPublisher | None = None,
    slowpics_publisher: SlowpicsPublisher | None = None,
) -> RunDependencies:
    """
    Build the default service bundle used by :func:`run`.

    The cfg/reporter/cache_dir parameters are accepted for future adapter wiring;
    callers may omit them when defaults suffice but tests can inject overrides.
    """

    del cfg, reporter, cache_dir  # Reserved for future dependency wiring.
    return RunDependencies(
        metadata_resolver=metadata_resolver or build_metadata_resolver(),
        alignment_workflow=alignment_workflow or build_alignment_workflow(),
        report_publisher=report_publisher or build_report_publisher(),
        slowpics_publisher=slowpics_publisher or build_slowpics_publisher(),
        setup_service=DefaultSetupService(),
    )


class WorkflowCoordinator:
    def __init__(self, dependencies: RunDependencies | None = None):
        self.dependencies = dependencies

    def execute(self, request: RunRequest) -> RunResult:
        """Orchestrate the CLI workflow."""
        dependencies = self.dependencies or default_run_dependencies()
        context = CoordinatorContext(request=request, dependencies=dependencies)

        pipeline = [
            SetupPhase(),
            DiscoveryPhase(),
            AlignmentPhase(),
            ClipLoaderPhase(),
            AnalysisPhase(),
            RenderPhase(),
            PublishPhase(),
            ResultPhase(),
        ]

        for phase in pipeline:
            # Check for early exit (e.g. from SetupPhase handling cache-only)
            if context.result is not None:
                break

            phase.execute(context)

        if context.result is None:
            raise RuntimeError("Workflow finished without producing a result.")

        return context.result
