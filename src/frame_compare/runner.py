from __future__ import annotations

from src.frame_compare.orchestration.coordinator import WorkflowCoordinator, default_run_dependencies
from src.frame_compare.orchestration.state import (
    ReporterFactory,
    RunContext,
    RunDependencies,
    RunRequest,
    RunResult,
)

__all__ = [
    "ReporterFactory",
    "RunContext",
    "RunDependencies",
    "RunRequest",
    "RunResult",
    "default_run_dependencies",
    "run",
]


def run(request: RunRequest, *, dependencies: RunDependencies | None = None) -> RunResult:
    """
    Orchestrate the CLI workflow.

    This function delegates the execution to the WorkflowCoordinator.
    """
    coordinator = WorkflowCoordinator(dependencies)
    return coordinator.execute(request)
