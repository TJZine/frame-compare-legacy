from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import pytest

from src.datatypes import AppConfig, SlowpicsConfig
from src.frame_compare.cli_runtime import CLIAppError, JsonTail
from src.frame_compare.services.publishers import (
    ReportPublisher,
    ReportPublisherRequest,
    SlowpicsPublisher,
    SlowpicsPublisherRequest,
)
from src.frame_compare.slowpics import SlowpicsAPIError
from tests.services.conftest import StubReporter, build_base_json_tail, build_service_config


class _StubRenderer:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.calls: list[Mapping[str, Any]] = []

    def generate(self, **kwargs: Any) -> Path:
        self.calls.append(dict(kwargs))
        return self.index_path


class _StubPublisherIO:
    def __init__(self) -> None:
        self.sizes: dict[str, int] = {}
        self.existing: set[Path] = set()

    def file_size(self, path: str | Path) -> int:
        return self.sizes.get(str(path), 0)

    def path_exists(self, path: Path) -> bool:
        return path in self.existing

    def resolve_report_dir(self, root: Path, relative: str, *, purpose: str) -> Path:  # noqa: ARG002
        resolved = root / relative
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


class _StubSlowpicsClient:
    def __init__(self, result_url: str | None = "https://slow.pics/c/demo") -> None:
        self.result_url = result_url
        self.calls: list[tuple[Sequence[str], Path]] = []

    def upload(
        self,
        image_paths: Sequence[str],
        out_dir: Path,
        cfg: SlowpicsConfig,
        *,
        progress_callback=None,
    ) -> str:
        self.calls.append((tuple(image_paths), out_dir))
        if self.result_url is None:
            raise SlowpicsAPIError("upload failed")
        if progress_callback is not None:
            progress_callback(len(image_paths))
        return self.result_url


@pytest.fixture
def publisher_io() -> _StubPublisherIO:
    return _StubPublisherIO()


@pytest.fixture
def service_cfg(tmp_path: Path) -> AppConfig:
    cfg = build_service_config(tmp_path)
    cfg.report.enable = True
    cfg.slowpics.auto_upload = False
    return cfg


def _make_context_payload(cfg: AppConfig) -> tuple[JsonTail, MutableMapping[str, Any]]:
    tail = build_base_json_tail(cfg)
    layout: MutableMapping[str, Any] = {
        "slowpics": tail["slowpics"],
        "report": tail["report"],
    }
    return tail, layout


def test_report_publisher_updates_report_block(tmp_path: Path, service_cfg: AppConfig, publisher_io: _StubPublisherIO) -> None:
    reporter = StubReporter()
    json_tail, layout_data = _make_context_payload(service_cfg)
    renderer = _StubRenderer(tmp_path / "report" / "index.html")
    publisher = ReportPublisher(renderer=renderer, io=publisher_io)
    request = ReportPublisherRequest(
        reporter=reporter,
        json_tail=json_tail,
        layout_data=layout_data,
        report_enabled=True,
        root=tmp_path,
        plans=[],
        frames=[1, 2, 3],
        selection_details={},
        image_paths=["img-a.png"],
        metadata_title="Demo",
        slowpics_url="https://slow.pics/c/demo",
        config=service_cfg.report,
        collected_warnings=[],
    )

    result = publisher.publish(request)

    assert result.report_path == renderer.index_path
    report_block = json_tail["report"]
    assert report_block.get("enabled") is True
    assert report_block.get("path") == str(renderer.index_path)
    assert layout_data["report"].get("path") == str(renderer.index_path)


def test_report_publisher_handles_generation_error(tmp_path: Path, service_cfg: AppConfig, publisher_io: _StubPublisherIO) -> None:
    reporter = StubReporter()
    json_tail, layout_data = _make_context_payload(service_cfg)

    class _FailingRenderer(_StubRenderer):
        def generate(self, **kwargs: Any) -> Path:  # noqa: ARG002
            raise SlowpicsAPIError("boom")

    renderer = _FailingRenderer(tmp_path / "report" / "index.html")
    publisher = ReportPublisher(renderer=renderer, io=publisher_io)
    request = ReportPublisherRequest(
        reporter=reporter,
        json_tail=json_tail,
        layout_data=layout_data,
        report_enabled=True,
        root=tmp_path,
        plans=[],
        frames=[],
        selection_details={},
        image_paths=[],
        metadata_title=None,
        slowpics_url=None,
        config=service_cfg.report,
        collected_warnings=[],
    )

    result = publisher.publish(request)

    assert result.report_path is None
    # Report generation failure should disable the feature flag and keep path unset
    assert json_tail["report"].get("enabled") is False
    assert json_tail["report"].get("path") is None
    assert "Failed to generate report: boom" in reporter.warnings[0]


def test_slowpics_publisher_updates_block(tmp_path: Path, service_cfg: AppConfig, publisher_io: _StubPublisherIO) -> None:
    reporter = StubReporter()
    json_tail, layout_data = _make_context_payload(service_cfg)
    service_cfg.slowpics.auto_upload = True
    publisher = SlowpicsPublisher(client=_StubSlowpicsClient("https://slow.pics/c/foo"), io=publisher_io)
    request = SlowpicsPublisherRequest(
        reporter=reporter,
        json_tail=json_tail,
        layout_data=layout_data,
        title_inputs=json_tail["slowpics"]["title"]["inputs"],
        final_title="Demo",
        resolved_base="Demo",
        tmdb_disclosure_line=None,
        verbose_tmdb_tag=None,
        image_paths=["img-a.png", "img-b.png"],
        out_dir=tmp_path,
        config=service_cfg.slowpics,
    )

    result = publisher.publish(request)

    assert result.url == "https://slow.pics/c/foo"
    slowpics_block = json_tail["slowpics"]
    assert slowpics_block["url"] == "https://slow.pics/c/foo"
    assert slowpics_block["shortcut_written"] is False


def test_slowpics_publisher_raises_on_failure(tmp_path: Path, service_cfg: AppConfig, publisher_io: _StubPublisherIO) -> None:
    reporter = StubReporter()
    json_tail, layout_data = _make_context_payload(service_cfg)
    service_cfg.slowpics.auto_upload = True
    client = _StubSlowpicsClient(result_url=None)
    publisher = SlowpicsPublisher(client=client, io=publisher_io)
    request = SlowpicsPublisherRequest(
        reporter=reporter,
        json_tail=json_tail,
        layout_data=layout_data,
        title_inputs=json_tail["slowpics"]["title"]["inputs"],
        final_title="Demo",
        resolved_base="Demo",
        tmdb_disclosure_line=None,
        verbose_tmdb_tag=None,
        image_paths=["img-a.png"],
        out_dir=tmp_path,
        config=service_cfg.slowpics,
    )

    with pytest.raises(CLIAppError):
        publisher.publish(request)
