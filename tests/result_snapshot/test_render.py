"""Tests covering result snapshot rendering behaviour."""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.progress import Progress, ProgressColumn

from src.frame_compare.cli_runtime import CliOutputManagerProtocol
from src.frame_compare.result_snapshot import (
    RenderOptions,
    ResultSource,
    RunResultSnapshot,
    SectionAvailability,
    SectionSnapshot,
    render_run_result,
)


@dataclass
class RecordingReporter(CliOutputManagerProtocol):
    """Minimal reporter that captures render activity for assertions."""

    quiet: bool = False
    verbose: bool = False
    console: Console = field(default_factory=lambda: Console(file=io.StringIO(), force_terminal=False))
    values: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    rendered_sections: list[str] = field(default_factory=list)
    section_titles: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    _warnings: list[str] = field(default_factory=list)

    def set_flag(self, key: str, value: Any) -> None:  # pragma: no cover - trivial
        self.flags[key] = value

    def update_values(self, mapping: Mapping[str, Any]) -> None:  # pragma: no cover - trivial
        self.values.update(mapping)

    def render_sections(self, section_ids: Iterable[str]) -> None:
        self.rendered_sections.extend(section_ids)

    def banner(self, text: str) -> None:  # pragma: no cover - unused
        self.lines.append(text)

    def get_warnings(self) -> list[str]:  # pragma: no cover - unused
        return list(self._warnings)

    def section(self, title: str) -> None:
        self.section_titles.append(title)

    def line(self, text: str) -> None:
        self.lines.append(text)

    def warn(self, text: str) -> None:  # pragma: no cover - unused
        self._warnings.append(text)

    def error(self, text: str) -> None:  # pragma: no cover - unused
        self._warnings.append(f"ERROR: {text}")

    def iter_warnings(self) -> list[str]:  # pragma: no cover - unused
        return list(self._warnings)

    def verbose_line(self, text: str) -> None:  # pragma: no cover - unused
        self.lines.append(text)

    def create_progress(self, progress_id: str, *, transient: bool = False) -> Progress:  # pragma: no cover
        return Progress(console=self.console, transient=transient)

    def update_progress_state(self, progress_id: str, **state: Any) -> None:  # pragma: no cover
        return None

    def progress(self, *columns: ProgressColumn, transient: bool = False) -> Progress:  # pragma: no cover
        return Progress(*columns, console=self.console, transient=transient)


def _make_snapshot() -> RunResultSnapshot:
    sections = {
        "full": SectionSnapshot(id="full", label="Full", availability=SectionAvailability.FULL),
        "partial": SectionSnapshot(id="partial", label="Partial", availability=SectionAvailability.PARTIAL),
        "missing": SectionSnapshot(
            id="missing",
            label="Missing",
            availability=SectionAvailability.MISSING,
            note="rerun for data",
        ),
    }
    snapshot = RunResultSnapshot(
        source=ResultSource.CACHE,
        sections=sections,
    )
    return snapshot


def test_render_run_result_skips_partial_by_default() -> None:
    reporter = RecordingReporter()
    snapshot = _make_snapshot()
    layout_sections = [
        {"id": "full", "title": "Full"},
        {"id": "partial", "title": "Partial"},
        {"id": "missing", "title": "Missing"},
    ]
    options = RenderOptions(show_partial=False, show_missing_sections=False)

    render_run_result(snapshot=snapshot, reporter=reporter, layout_sections=layout_sections, options=options)

    assert reporter.rendered_sections == ["full"]
    assert all("incomplete" not in title.lower() for title in reporter.section_titles)


def test_render_run_result_respects_show_partial_flag() -> None:
    reporter = RecordingReporter()
    snapshot = _make_snapshot()
    layout_sections = [
        {"id": "partial", "title": "Partial"},
    ]
    options = RenderOptions(show_partial=True, show_missing_sections=False)

    render_run_result(snapshot=snapshot, reporter=reporter, layout_sections=layout_sections, options=options)

    assert reporter.rendered_sections == ["partial"]
    assert any("Partial" in title for title in reporter.section_titles)


def test_render_run_result_hides_missing_sections_when_disabled() -> None:
    reporter = RecordingReporter()
    snapshot = _make_snapshot()
    layout_sections = [
        {"id": "full", "title": "Full"},
        {"id": "missing", "title": "Missing"},
    ]
    options = RenderOptions(show_partial=False, show_missing_sections=False)

    render_run_result(snapshot=snapshot, reporter=reporter, layout_sections=layout_sections, options=options)

    assert reporter.rendered_sections == ["full"]
    assert all(title != "Missing" for title in reporter.section_titles)
    assert all("not available from cache" not in line for line in reporter.lines)


def test_missing_sections_emit_hint_when_enabled() -> None:
    reporter = RecordingReporter()
    snapshot = _make_snapshot()
    layout_sections = [
        {"id": "missing", "title": "Missing"},
    ]
    options = RenderOptions(show_partial=False, show_missing_sections=True, no_cache_hint="--rerun")

    render_run_result(snapshot=snapshot, reporter=reporter, layout_sections=layout_sections, options=options)

    assert not reporter.rendered_sections
    assert reporter.section_titles and reporter.section_titles[0] == "Missing"
    assert any(options.no_cache_hint in line for line in reporter.lines)
