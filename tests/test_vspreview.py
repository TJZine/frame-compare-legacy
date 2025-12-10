"""Unit tests for the dedicated vspreview module."""
from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import frame_compare as _frame_compare  # noqa: F401  # Ensure CLI shim initialises alignment_runner.
from src.frame_compare import alignment as alignment_package
from src.frame_compare import vspreview as vspreview_module
from src.frame_compare.cli_runtime import ClipPlan
from tests.helpers.runner_env import (
    _make_config,
    _make_json_tail_stub,
    _RecordingOutputManager,
)


def _make_summary(reference: ClipPlan, target: ClipPlan, tmp_path: Path) -> alignment_package.AudioAlignmentSummary:
    """Helper constructing a minimal audio-alignment summary for VSPreview tests."""

    return alignment_package.AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.json",
        reference_name=reference.path.name,
        measurements=tuple(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference,
        final_adjustments={},
        swap_details={},
        suggested_frames={target.path.name: 3},
        suggestion_mode=True,
        manual_trim_starts={target.path.name: 0},
    )


def test_render_script_includes_manual_trims(tmp_path: Path) -> None:
    """render_script should serialise reference/target plans with trims and suggestions."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.use_vspreview = True
    cfg.audio_alignment.vspreview_mode = "seeded"

    reference = ClipPlan(path=tmp_path / "Ref.mkv", metadata={"label": "Reference"}, trim_start=2)
    target = ClipPlan(path=tmp_path / "Target.mkv", metadata={"label": "Target"}, trim_start=5)
    plans = [reference, target]
    summary = _make_summary(reference, target, tmp_path)
    summary.manual_trim_starts = {reference.path.name: 2, target.path.name: 5}

    script = vspreview_module.render_script(plans, summary, cfg, tmp_path)

    assert "REFERENCE = {" in script
    assert "'label': 'Reference'" in script
    assert "'Target': {" in script
    assert "OFFSET_MAP" in script
    assert "SUGGESTION_MAP" in script
    assert "SHOW_SUGGESTED_OVERLAY" in script
    # Seeded mode should pre-populate OFFSET_MAP entries with suggested deltas.
    assert "# Suggested delta +3f" in script


def test_render_script_marks_missing_frame_hint(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.audio_alignment.use_vspreview = True

    reference = ClipPlan(path=tmp_path / "Ref.mkv", metadata={"label": "Ref"})
    target = ClipPlan(path=tmp_path / "Target.mkv", metadata={"label": "Target"})
    plans = [reference, target]
    summary = _make_summary(reference, target, tmp_path)
    summary.suggested_frames = {}

    script = vspreview_module.render_script(plans, summary, cfg, tmp_path)

    assert "Suggested delta n/a" in script
    assert "(None, 0.0)" in script


def test_persist_vspreview_script_regenerates_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist_script should avoid overwriting existing files with deterministic naming."""

    timestamp = dt.datetime(2024, 1, 1, 12, 0, 0)

    class _FixedDatetime(dt.datetime):
        @classmethod
        def now(cls, tz: dt.tzinfo | None = None) -> dt.datetime:
            return timestamp if tz is None else timestamp.replace(tzinfo=tz)

    monkeypatch.setattr(vspreview_module._dt, "datetime", _FixedDatetime)

    uuid_values = iter(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])

    def _fake_uuid() -> SimpleNamespace:
        return SimpleNamespace(hex=next(uuid_values))

    monkeypatch.setattr(vspreview_module.uuid, "uuid4", _fake_uuid)

    script_dir = vspreview_module.resolve_subdir(tmp_path, "vspreview", purpose="vspreview workspace")
    script_dir.mkdir(parents=True, exist_ok=True)
    existing_name = f"vspreview_{timestamp.strftime('%Y%m%d-%H%M%S')}_aaaaaaaa.py"
    (script_dir / existing_name).write_text("# existing", encoding="utf-8")

    script_path = vspreview_module.persist_script("print('ok')", tmp_path)

    assert script_path.parent == script_dir
    assert script_path.name == f"vspreview_{timestamp.strftime('%Y%m%d-%H%M%S')}_bbbbbbbb.py"


def test_persist_vspreview_script_propagates_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filesystem failures should bubble up so callers can surface the error."""

    original_write = Path.write_text

    def _raise_on_vspreview(target: Path, *args: Any, **kwargs: Any) -> int:
        if "vspreview" in target.parts:
            raise PermissionError("locked")
        return original_write(target, *args, **kwargs)  # type: ignore[return-value]

    monkeypatch.setattr(Path, "write_text", _raise_on_vspreview)

    with pytest.raises(PermissionError):
        vspreview_module.persist_script("print('locked')", tmp_path)


def test_launch_vspreview_uses_injected_process_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher should call the injected process runner and record telemetry."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.use_vspreview = True
    cfg.runtime.vapoursynth_python_paths = ["~/vsp"]

    reference = ClipPlan(path=tmp_path / "Ref.mkv", metadata={"label": "Reference"})
    target = ClipPlan(path=tmp_path / "Target.mkv", metadata={"label": "Target"})
    plans = [reference, target]
    summary = _make_summary(reference, target, tmp_path)

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()
    fake_script = tmp_path / "vspreview_script.py"
    fake_script.parent.mkdir(parents=True, exist_ok=True)
    fake_script.write_text("# stub", encoding="utf-8")

    monkeypatch.setattr(vspreview_module, "write_script", lambda *args, **kwargs: fake_script)
    monkeypatch.setattr(
        vspreview_module,
        "resolve_command",
        lambda _path: (["vspreview", str(fake_script)], None),
    )
    monkeypatch.setattr(vspreview_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(vspreview_module, "prompt_offsets", lambda *args, **kwargs: None)

    recorded_calls: list[dict[str, object]] = []

    def _runner(
        command: list[str],
        *,
        env: dict[str, str],
        check: bool,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[object]:
        recorded_calls.append({"command": list(command), "env": dict(env), "kwargs": kwargs})
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    vspreview_module.launch(
        plans,
        summary,
        None,
        cfg,
        tmp_path,
        reporter,
        json_tail,
        process_runner=_runner,
    )

    assert recorded_calls, "Injected runner should be called"
    env_record = recorded_calls[0]["env"]
    assert isinstance(env_record, dict)
    assert "VAPOURSYNTH_PYTHONPATH" in env_record
    audio_block = json_tail["audio_alignment"]
    assert audio_block.get("vspreview_invoked") is True
    assert audio_block.get("vspreview_exit_code") == 0


def test_launch_vspreview_reports_backend_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing PySide/PyQt backends should propagate a specific reason into telemetry."""

    cfg = _make_config(tmp_path)
    cfg.audio_alignment.use_vspreview = True

    reference = ClipPlan(path=tmp_path / "Ref.mkv", metadata={"label": "Reference"})
    target = ClipPlan(path=tmp_path / "Target.mkv", metadata={"label": "Target"})
    plans = [reference, target]
    summary = _make_summary(reference, target, tmp_path)

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()

    fake_script = tmp_path / "vspreview_script.py"
    fake_script.parent.mkdir(parents=True, exist_ok=True)
    fake_script.write_text("# stub", encoding="utf-8")

    monkeypatch.setattr(vspreview_module, "write_script", lambda *args, **kwargs: fake_script)
    monkeypatch.setattr(
        vspreview_module,
        "resolve_command",
        lambda _path: (None, "vspreview-backend-missing"),
    )
    monkeypatch.setattr(vspreview_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(vspreview_module, "prompt_offsets", lambda *args, **kwargs: None)

    vspreview_module.launch(
        plans,
        summary,
        None,
        cfg,
        tmp_path,
        reporter,
        json_tail,
    )

    offer_entry = json_tail.get("vspreview_offer")
    assert offer_entry == {
        "vspreview_offered": False,
        "reason": "vspreview-backend-missing",
    }


def test_apply_manual_offsets_updates_json_tail(tmp_path: Path) -> None:
    """Manual offsets should update the JSON tail even when clip labels diverge from names."""

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()

    reference = ClipPlan(path=tmp_path / "Ref.mkv", metadata={"label": "Reference"}, effective_fps=(24, 1))
    target = ClipPlan(path=tmp_path / "TargetA.mkv", metadata={"label": "Target A"}, effective_fps=(24, 1))
    extra = ClipPlan(path=tmp_path / "TargetB.mkv", metadata={"label": "Target B"}, effective_fps=(24, 1))
    plans = [reference, target, extra]

    summary = alignment_package.AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.json",
        reference_name=reference.path.name,
        measurements=tuple(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference,
        final_adjustments={},
        swap_details={},
        suggested_frames={target.path.name: 0},
        suggestion_mode=True,
        manual_trim_starts={},
    )

    deltas = {
        target.path.name: 3,
        reference.path.name: -1,
        "unknown.mkv": 5,
    }

    vspreview_module.apply_manual_offsets(
        plans,
        summary,
        deltas,
        reporter,
        json_tail,
        display=None,
    )

    audio_block = json_tail["audio_alignment"]
    offsets_frames = audio_block.get("offsets_frames", {})
    detail = summary.measured_offsets[target.path.name]
    assert offsets_frames.get("Target A") == detail.frames
    warnings = reporter.get_warnings()
    assert any("not part of the current plan" in warning for warning in warnings)


def test_apply_manual_offsets_populates_seconds_without_known_fps(tmp_path: Path) -> None:
    """Manual offsets should fall back to a sane FPS when clips lack metadata."""

    reporter = _RecordingOutputManager()
    json_tail = _make_json_tail_stub()

    reference = ClipPlan(path=tmp_path / "Ref.mkv", metadata={"label": "Reference"})
    target = ClipPlan(path=tmp_path / "Target.mkv", metadata={"label": "Target"})
    summary = alignment_package.AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.json",
        reference_name=reference.path.name,
        measurements=tuple(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference,
        final_adjustments={},
        swap_details={},
        suggested_frames={},
        suggestion_mode=True,
        manual_trim_starts={},
    )

    vspreview_module.apply_manual_offsets(
        [reference, target],
        summary,
        {target.path.name: 6},
        reporter,
        json_tail,
        display=None,
    )

    detail = summary.measured_offsets[target.path.name]
    assert detail.frames == 6
    fallback_fps = 24000 / 1001
    assert detail.offset_seconds == pytest.approx(6 / fallback_fps, rel=1e-6)
    audio_block = json_tail["audio_alignment"]
    offsets_sec = audio_block.get("offsets_sec", {})
    assert offsets_sec["Target"] == pytest.approx(6 / fallback_fps, rel=1e-6)
