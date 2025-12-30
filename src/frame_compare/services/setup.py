from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.frame_compare.orchestration import setup as setup_impl

if TYPE_CHECKING:
    from src.frame_compare.orchestration.state import RunEnvironment, RunRequest


class SetupService(Protocol):
    """Protocol for preparing the run environment."""

    def prepare_run_environment(self, request: RunRequest) -> RunEnvironment:
        """
        Perform preflight checks, configure environment, and prepare dependencies.
        """
        ...


class DefaultSetupService:
    """Default implementation delegating to the existing setup module."""

    def prepare_run_environment(self, request: RunRequest) -> RunEnvironment:
        return setup_impl.prepare_run_environment(request)
