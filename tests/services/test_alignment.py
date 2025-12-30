from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, cast

import pytest

from src.datatypes import AppConfig
from src.frame_compare.cli_runtime import (
    AudioAlignmentDisplayData,
    AudioAlignmentSummary,
    CliOutputManagerProtocol,
    ClipPlan,
    JsonTail,
)
from src.frame_compare.services.alignment import AlignmentRequest, AlignmentWorkflow
from src.frame_compare.services.metadata import CliPromptProtocol
from tests.services.conftest import StubReporter, build_base_json_tail, build_service_config


class _ApplyRecorder:
    def __init__(
        self,
        summary: AudioAlignmentSummary | None,
        display: AudioAlignmentDisplayData | None,
    ) -> None:
        self.summary = summary
        self.display = display
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        plans: Sequence[ClipPlan],
        cfg: AppConfig,
        analyze_path: Path,
        root: Path,
        audio_track_overrides: Mapping[str, int],
        *,
        reporter: CliPromptProtocol | None,
) -> tuple[AudioAlignmentSummary | None, AudioAlignmentDisplayData | None]:
        self.calls.append(
            {
                "plans": plans,
                "cfg": cfg,
                "analyze_path": analyze_path,
                "root": root,
                "overrides": dict(audio_track_overrides),
                "reporter": reporter,
            }
        )
        return self.summary, self.display


class _FormatRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        plans: Sequence[ClipPlan],
        summary: AudioAlignmentSummary | None,
        display: AudioAlignmentDisplayData | None,
        *,
        cfg: AppConfig,
        root: Path,
        reporter: CliPromptProtocol,
        json_tail: JsonTail,
        vspreview_mode: str,
        collected_warnings: list[str] | None,
    ) -> None:
        self.calls.append(
            {
                "plans": plans,
                "summary": summary,
                "display": display,
                "cfg": cfg,
                "root": root,
                "reporter": reporter,
                "json_tail": json_tail,
                "vspreview_mode": vspreview_mode,
                "warnings": collected_warnings,
            }
        )


class _ConfirmRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Sequence[ClipPlan], AudioAlignmentSummary, AudioAlignmentDisplayData]] = []

    def __call__(
        self,
        plans: Sequence[ClipPlan],
        summary: AudioAlignmentSummary,
        cfg: AppConfig,
        root: Path,
        reporter: CliOutputManagerProtocol,
        display: AudioAlignmentDisplayData,
    ) -> None:
        self.calls.append((plans, summary, display))


def _make_request(tmp_path: Path, cfg: AppConfig, reporter: StubReporter, json_tail: JsonTail) -> AlignmentRequest:
    plans = [ClipPlan(path=tmp_path / "a.mkv", metadata={"label": "a"})]
    audio_overrides: Mapping[str, int] = {"a.mkv": 1}
    return AlignmentRequest(
        plans=plans,
        cfg=cfg,
        root=tmp_path,
        analyze_path=plans[0].path,
        audio_track_overrides=audio_overrides,
        reporter=reporter,
        json_tail=json_tail,
        vspreview_mode="baseline",
        collected_warnings=[],
    )


def test_alignment_workflow_runs_apply_and_format(tmp_path: Path) -> None:
    cfg = build_service_config(tmp_path)
    reporter = StubReporter()
    json_tail = build_base_json_tail(cfg)
    summary = cast(AudioAlignmentSummary, SimpleNamespace(suggestion_mode=False))
    display = cast(AudioAlignmentDisplayData, SimpleNamespace())
    apply_recorder = _ApplyRecorder(summary, display)
    format_recorder = _FormatRecorder()
    confirm_recorder = _ConfirmRecorder()
    workflow = AlignmentWorkflow(
        apply_alignment=apply_recorder,
        format_output=format_recorder,
        confirm_alignment=confirm_recorder,
    )
    request = _make_request(tmp_path, cfg, reporter, json_tail)

    result = workflow.run(request)

    assert result.summary is summary
    assert result.display is display
    assert apply_recorder.calls and format_recorder.calls
    assert confirm_recorder.calls == []


def test_alignment_workflow_confirms_when_enabled(tmp_path: Path) -> None:
    cfg = build_service_config(tmp_path)
    cfg.audio_alignment.enable = True
    reporter = StubReporter()
    json_tail = build_base_json_tail(cfg)
    summary = cast(AudioAlignmentSummary, SimpleNamespace(suggestion_mode=False))
    display = cast(AudioAlignmentDisplayData, SimpleNamespace())
    apply_recorder = _ApplyRecorder(summary, display)
    confirm_recorder = _ConfirmRecorder()
    workflow = AlignmentWorkflow(
        apply_alignment=apply_recorder,
        format_output=_FormatRecorder(),
        confirm_alignment=confirm_recorder,
    )
    request = _make_request(tmp_path, cfg, reporter, json_tail)

    workflow.run(request)

    assert len(confirm_recorder.calls) == 1


def test_alignment_workflow_propagates_failures(tmp_path: Path) -> None:
    cfg = build_service_config(tmp_path)
    reporter = StubReporter()
    json_tail = build_base_json_tail(cfg)

    def _raising_apply(*_: Any, **__: Any) -> tuple[None, None]:
        raise RuntimeError("boom")

    workflow = AlignmentWorkflow(
        apply_alignment=_raising_apply,
        format_output=_FormatRecorder(),
        confirm_alignment=_ConfirmRecorder(),
    )
    request = _make_request(tmp_path, cfg, reporter, json_tail)

    with pytest.raises(RuntimeError):
        workflow.run(request)
