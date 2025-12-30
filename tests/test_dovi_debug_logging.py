from __future__ import annotations

import json

import pytest
from _pytest.capture import CaptureFixture

from src.frame_compare.orchestration import setup
from src.frame_compare.vs import tonemap as vs_tonemap


def _read_err(capsys: CaptureFixture[str]) -> str:
    captured = capsys.readouterr()
    return captured.err


def test_runner_emit_dovi_debug_skips_when_flag_false(monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    monkeypatch.setenv(setup.DOVI_DEBUG_ENV_FLAG, "0")
    setup.emit_dovi_debug({"phase": "unit"})
    assert "[DOVI_DEBUG]" not in _read_err(capsys)


def test_runner_emit_dovi_debug_emits_when_flag_true(monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    monkeypatch.setenv(setup.DOVI_DEBUG_ENV_FLAG, "1")
    setup.emit_dovi_debug({"phase": "unit", "value": 3})
    err = _read_err(capsys)
    assert "[DOVI_DEBUG]" in err
    assert json.dumps({"phase": "unit", "value": 3}) in err


def test_vs_emit_dovi_debug_skips_when_flag_false(monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    monkeypatch.setenv(vs_tonemap.DOVI_DEBUG_ENV_FLAG, "no")
    vs_tonemap._emit_vs_dovi_debug({"phase": "vs"})
    assert "[DOVI_DEBUG]" not in _read_err(capsys)


def test_vs_emit_dovi_debug_emits_when_flag_true(monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    monkeypatch.setenv(vs_tonemap.DOVI_DEBUG_ENV_FLAG, "yes")
    vs_tonemap._emit_vs_dovi_debug({"phase": "vs", "tag": "test"})
    err = _read_err(capsys)
    assert "[DOVI_DEBUG]" in err
    assert json.dumps({"phase": "vs", "tag": "test"}) in err
