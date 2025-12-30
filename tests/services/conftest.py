from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import pytest
from rich.console import Console

from src.datatypes import (
    AnalysisConfig,
    AppConfig,
    AudioAlignmentConfig,
    CLIConfig,
    ColorConfig,
    DiagnosticsConfig,
    NamingConfig,
    OverridesConfig,
    PathsConfig,
    ReportConfig,
    RunnerConfig,
    RuntimeConfig,
    ScreenshotConfig,
    SlowpicsConfig,
    SourceConfig,
    TMDBConfig,
)
from src.frame_compare.cli_runtime import JsonTail, SlowpicsTitleInputs
from src.frame_compare.services.metadata import CliPromptProtocol


class StubReporter(CliPromptProtocol):
    def __init__(self) -> None:
        self.quiet = False
        self.verbose = False
        self.console = Console(record=True, force_terminal=False)
        self.flags: dict[str, Any] = {}
        self.values: dict[str, Any] = {}
        self.lines: list[str] = []
        self.verbose_lines: list[str] = []
        self.warnings: list[str] = []

    def set_flag(self, key: str, value: Any) -> None:
        self.flags[key] = value

    def update_values(self, mapping: Mapping[str, Any]) -> None:
        self.values.update(mapping)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def error(self, text: str) -> None:
        self.warnings.append(f"ERROR: {text}")

    def get_warnings(self) -> list[str]:
        return list(self.warnings)

    def render_sections(self, section_ids: Iterable[str]) -> None:  # pragma: no cover - unused
        return None

    def create_progress(self, progress_id: str, *, transient: bool = False) -> Any:  # pragma: no cover - unused
        return None

    def update_progress_state(self, progress_id: str, **state: Any) -> None:  # pragma: no cover - unused
        return None

    def banner(self, text: str) -> None:  # pragma: no cover - unused
        self.lines.append(text)

    def section(self, title: str) -> None:  # pragma: no cover - unused
        self.lines.append(title)

    def line(self, text: str) -> None:
        self.lines.append(text)

    def verbose_line(self, text: str) -> None:
        self.verbose_lines.append(text)

    def progress(self, *columns: Any, transient: bool = False) -> Any:  # pragma: no cover - unused
        return None

    def confirm(self, text: str, *, default: bool = True) -> bool:
        return default

    def iter_warnings(self) -> list[str]:
        return list(self.warnings)


def build_service_config(root: Path) -> AppConfig:
    return AppConfig(
        analysis=AnalysisConfig(
            frame_count_dark=2,
            frame_count_bright=2,
            frame_count_motion=2,
            random_frames=0,
            user_frames=[],
        ),
        screenshots=ScreenshotConfig(directory_name="screens", add_frame_info=False),
        cli=CLIConfig(),
        color=ColorConfig(),
        slowpics=SlowpicsConfig(auto_upload=False),
        tmdb=TMDBConfig(api_key="secret"),
        naming=NamingConfig(always_full_filename=False, prefer_guessit=False),
        paths=PathsConfig(input_dir=str(root)),
        runtime=RuntimeConfig(ram_limit_mb=2048),
        overrides=OverridesConfig(),
        source=SourceConfig(preferred="lsmas"),
        audio_alignment=AudioAlignmentConfig(enable=False),
        report=ReportConfig(enable=False),
        runner=RunnerConfig(),
        diagnostics=DiagnosticsConfig(),
    )


def build_base_json_tail(cfg: AppConfig) -> JsonTail:
    slowpics_inputs: SlowpicsTitleInputs = {
        "resolved_base": None,
        "collection_name": None,
        "collection_suffix": getattr(cfg.slowpics, "collection_suffix", ""),
    }
    tail = {
        "clips": [],
        "trims": {"per_clip": {}},
        "window": {},
        "alignment": {},
        "audio_alignment": {},
        "analysis": {},
        "render": {},
        "tonemap": {},
        "overlay": {},
        "verify": {},
        "cache": {},
        "slowpics": {
            "enabled": bool(cfg.slowpics.auto_upload),
            "title": {
                "inputs": slowpics_inputs,
                "final": None,
            },
        },
        "report": {"enabled": False},
        "viewer": {},
        "warnings": [],
        "workspace": {},
        "vspreview_mode": None,
        "suggested_frames": None,
        "suggested_seconds": 0.0,
        "vspreview_offer": None,
    }
    return cast(JsonTail, tail)


@pytest.fixture
def service_reporter() -> StubReporter:
    return StubReporter()
