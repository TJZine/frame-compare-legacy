from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.frame_compare.cli_runtime import JsonTail
from tests.helpers.runner_env import (
    DummyProgress,
    _CliRunnerEnv,
    _make_json_tail_stub,
    _RecordingOutputManager,
    install_dummy_progress,
    install_tty_stdin,
    install_vs_core_stub,
    install_vspreview_presence,
    install_which_map,
)


@pytest.fixture
def cli_runner_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _CliRunnerEnv:
    """Install a deterministic CLI harness for CLI-heavy tests."""

    return _CliRunnerEnv(monkeypatch, tmp_path)


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[_CliRunnerEnv]:
    """Yield a `_CliRunnerEnv` harness (Phase 10 fixture consolidation target)."""

    env = _CliRunnerEnv(monkeypatch, tmp_path)
    yield env


@pytest.fixture
def recording_output_manager() -> _RecordingOutputManager:
    """Provide a CliOutputManager test double that records emitted lines."""

    return _RecordingOutputManager()


@pytest.fixture
def json_tail_stub() -> JsonTail:
    """Expose a reusable JsonTail stub for telemetry assertions."""

    return _make_json_tail_stub()


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click runner configured for CLI smoke tests."""

    return CliRunner()


@pytest.fixture
def runner_vs_core_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatically stub VapourSynth bindings for runner-heavy suites."""

    install_vs_core_stub(monkeypatch)


@pytest.fixture
def dummy_progress(monkeypatch: pytest.MonkeyPatch) -> type[DummyProgress]:
    """Install the DummyProgress stub so runner suites share a consistent Progress helper."""

    install_dummy_progress(monkeypatch)
    return DummyProgress


@pytest.fixture
def vspreview_env(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], None]:
    """Return a toggle that marks VSPreview modules/CLI present or missing."""

    def _toggle(present: bool) -> None:
        install_vspreview_presence(monkeypatch, present=present)

    return _toggle


@pytest.fixture
def which_map(monkeypatch: pytest.MonkeyPatch) -> Callable[[set[str] | None], None]:
    """Return a helper that flags specific CLI tools as missing (others resolved under /usr/bin)."""

    def _apply(missing: set[str] | None = None) -> None:
        install_which_map(monkeypatch, missing=missing)

    return _apply


@pytest.fixture
def tty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sys.stdin appear as a TTY for interactive prompt tests.

    Use this fixture for tests that verify confirmation prompts, offset reuse
    dialogs, or any other interactive CLI behavior that requires TTY detection.
    """
    install_tty_stdin(monkeypatch)
