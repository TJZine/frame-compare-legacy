"""Unit tests for section availability heuristics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, cast

from src.frame_compare import runner as runner_module
from src.frame_compare.orchestration import reporting
from src.frame_compare.result_snapshot import SectionAvailability, SectionState
from tests.helpers.runner_env import _make_config


def _make_run_result(tmp_path: Path) -> runner_module.RunResult:
    cfg = _make_config(tmp_path)
    out_dir = tmp_path / "screens"
    out_dir.mkdir(parents=True, exist_ok=True)
    return runner_module.RunResult(
        files=[],
        frames=[],
        out_dir=out_dir,
        out_dir_created=False,
        out_dir_created_path=None,
        root=tmp_path,
        config=cfg,
        image_paths=[],
    )


def _apply(layout_data: Mapping[str, object], section_states: Dict[str, SectionState], result: runner_module.RunResult) -> None:
    def _mark(section_id: str, availability: SectionAvailability, note: str | None = None) -> None:
        if section_id not in section_states:
            return
        section_states[section_id] = SectionState(availability=availability, note=note)

    reporting.apply_section_availability_overrides(
        section_states,
        _mark,
        layout_data=cast(Mapping[str, Any], layout_data),
        result=result,
    )


def test_render_and_publish_sections_track_cache_artifacts(tmp_path: Path) -> None:
    result = _make_run_result(tmp_path)
    section_states = {
        "render": SectionState(availability=SectionAvailability.FULL, note=None),
        "publish": SectionState(availability=SectionAvailability.FULL, note=None),
    }
    layout_data = {
        "slowpics": {"status": "failed", "auto_upload": True},
    }

    _apply(layout_data, section_states, result)

    assert section_states["render"].availability is SectionAvailability.PARTIAL
    assert section_states["publish"].availability is SectionAvailability.PARTIAL


def test_audio_alignment_and_vspreview_sections_follow_configuration(tmp_path: Path) -> None:
    result = _make_run_result(tmp_path)
    section_states = {
        "audio_align": SectionState(availability=SectionAvailability.FULL, note=None),
        "vspreview_info": SectionState(availability=SectionAvailability.FULL, note=None),
        "vspreview_missing": SectionState(availability=SectionAvailability.FULL, note=None),
    }
    layout_data = {
        "audio_alignment": {"enabled": False, "use_vspreview": False},
        "vspreview": {"missing": {"active": False}},
    }

    _apply(layout_data, section_states, result)

    assert section_states["audio_align"].availability is SectionAvailability.MISSING
    assert section_states["vspreview_info"].availability is SectionAvailability.MISSING
    assert section_states["vspreview_missing"].availability is SectionAvailability.MISSING


def test_report_and_viewer_sections_require_destinations(tmp_path: Path) -> None:
    result = _make_run_result(tmp_path)
    section_states = {
        "report": SectionState(availability=SectionAvailability.FULL, note=None),
        "viewer": SectionState(availability=SectionAvailability.FULL, note=None),
    }
    layout_data = {
        "report": {"enabled": False, "path": None},
        "viewer": {"mode": "none", "destination": None},
    }

    _apply(layout_data, section_states, result)

    assert section_states["report"].availability is SectionAvailability.MISSING
    assert section_states["viewer"].availability is SectionAvailability.MISSING
