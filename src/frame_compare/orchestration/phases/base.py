from typing import Protocol

from src.frame_compare.orchestration.state import CoordinatorContext


class Phase(Protocol):
    def execute(self, context: CoordinatorContext) -> None:
        """Execute this phase, mutating the context."""
        ...
