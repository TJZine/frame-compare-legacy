"""Runner-specific test fixtures.

This conftest.py provides fixtures specific to CLI runner and integration tests.
General-purpose fixtures are defined in tests/conftest.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


@pytest.fixture
def integration_test(request: "FixtureRequest") -> None:
    """Apply the 'integration' marker to this test.

    Integration tests exercise the full CLI pipeline and may be slower than
    unit tests. Use `pytest -m integration` to run only integration tests,
    or `pytest -m "not integration"` to skip them.

    Usage::

        def test_full_pipeline(integration_test, cli_runner_env):
            # This test is now marked as an integration test
            ...

    Note:
        You can also use the decorator directly: @pytest.mark.integration
    """
    request.node.add_marker(pytest.mark.integration)  # type: ignore[attr-defined]
