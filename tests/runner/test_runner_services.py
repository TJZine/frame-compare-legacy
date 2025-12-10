"""Runner dependency wiring and service orchestration tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, cast

import pytest

from src.datatypes import AppConfig
from src.frame_compare import media as media_utils
from src.frame_compare import runner as runner_module
from src.frame_compare import selection as selection_utils
from src.frame_compare.cli_runtime import CLIAppError, CliOutputManager, ClipPlan, JsonTail
from src.frame_compare.interfaces import PublisherIO, ReportRendererProtocol, SlowpicsClientProtocol
from src.frame_compare.orchestration.coordinator import WorkflowCoordinator
from src.frame_compare.services.alignment import AlignmentRequest, AlignmentResult, AlignmentWorkflow
from src.frame_compare.services.metadata import MetadataResolver, MetadataResolveRequest, MetadataResolveResult
from src.frame_compare.services.publishers import (
    ReportPublisher,
    ReportPublisherRequest,
    SlowpicsPublisher,
    SlowpicsPublisherRequest,
)
from src.frame_compare.services.setup import DefaultSetupService
from tests.services.conftest import StubReporter, build_base_json_tail, build_service_config

pytestmark = pytest.mark.usefixtures("runner_vs_core_stub", "dummy_progress")  # type: ignore[attr-defined]


class SentinelError(RuntimeError):
    """Raised to stop the runner after service orchestration assertions."""


def _write_media(cli_runner_env: Any) -> list[Path]:
    files: list[Path] = []
    for name in ("Alpha.mkv", "Beta.mkv"):
        path = cli_runner_env.media_root / name
        path.write_bytes(b"0")
        files.append(path)
    return files


def _make_metadata_result(files: Sequence[Path]) -> MetadataResolveResult:
    plans = [
        ClipPlan(path=files[0], metadata={"label": "Alpha"}),
        ClipPlan(path=files[1], metadata={"label": "Beta"}),
    ]
    plans[0].use_as_reference = True
    return MetadataResolveResult(
        plans=plans,
        metadata=[{"label": "Alpha"}, {"label": "Beta"}],
        metadata_title="Demo Title",
        analyze_path=files[0],
        slowpics_title_inputs={
            "resolved_base": "Demo Base",
            "collection_name": "Collection",
            "collection_suffix": "Vol.1",
        },
        slowpics_final_title="Demo Title",
        slowpics_resolved_base="Demo Base",
        slowpics_tmdb_disclosure_line=None,
        slowpics_verbose_tmdb_tag=None,
        tmdb_notes=["tmdb demo note"],
    )


class _RendererStub(ReportRendererProtocol):
    def __init__(self) -> None:
        self.output = Path("report/index.html")

    def generate(  # type: ignore[override]
        self,
        *,
        report_dir: Path,
        report_cfg,  # noqa: ANN001
        frames,
        selection_details,
        image_paths,
        plans,
        metadata_title,
        include_metadata,
        slowpics_url,
    ) -> Path:
        return report_dir / self.output


class _PublisherIOStub(PublisherIO):
    def file_size(self, path: str | Path) -> int:
        return 0

    def path_exists(self, path: Path) -> bool:
        return False

    def resolve_report_dir(self, root: Path, relative: str, *, purpose: str) -> Path:  # noqa: ARG002
        resolved = root / relative
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


class _SlowpicsClientStub(SlowpicsClientProtocol):
    def upload(
        self,
        image_paths,
        out_dir: Path,
        cfg,  # noqa: ANN001
        *,
        progress_callback=None,
    ) -> str:
        if progress_callback is not None:
            progress_callback(len(list(image_paths)))
        return "https://slow.pics/c/test"


class _StubReportPublisher(ReportPublisher):
    def __init__(self) -> None:
        super().__init__(renderer=_RendererStub(), io=_PublisherIOStub())
        self.last_request: ReportPublisherRequest | None = None
        self.call_count = 0

    def publish(self, request: ReportPublisherRequest) -> object:  # type: ignore[override]
        self.last_request = request
        self.call_count += 1
        return SimpleNamespace(report_path=None)


class _StubSlowpicsPublisher(SlowpicsPublisher):
    def __init__(self) -> None:
        super().__init__(client=_SlowpicsClientStub(), io=_PublisherIOStub())
        self.last_request: SlowpicsPublisherRequest | None = None
        self.call_count = 0

    def publish(self, request: SlowpicsPublisherRequest) -> object:  # type: ignore[override]
        self.last_request = request
        self.call_count += 1
        return SimpleNamespace(url="https://slow.pics/c/test")


def _build_dependencies(
    metadata_resolver: object,
    alignment_workflow: object,
    report_publisher: object | None = None,
    slowpics_publisher: object | None = None,
) -> runner_module.RunDependencies:
    return runner_module.RunDependencies(
        metadata_resolver=cast(MetadataResolver, metadata_resolver),
        alignment_workflow=cast(AlignmentWorkflow, alignment_workflow),
        report_publisher=cast(ReportPublisher, report_publisher or _StubReportPublisher()),
        slowpics_publisher=cast(SlowpicsPublisher, slowpics_publisher or _StubSlowpicsPublisher()),
        setup_service=DefaultSetupService(),
    )


class _RecordingMetadataResolver:
    def __init__(self, result: MetadataResolveResult, log: list[str]) -> None:
        self._result = result
        self._log = log
        self.last_request: MetadataResolveRequest | None = None

    def resolve(self, request: MetadataResolveRequest) -> MetadataResolveResult:
        self._log.append("metadata")
        self.last_request = request
        return self._result


class _RecordingAlignmentWorkflow:
    def __init__(self, result: AlignmentResult, log: list[str]) -> None:
        self._result = result
        self._log = log
        self.last_request: AlignmentRequest | None = None

    def run(self, request: AlignmentRequest) -> AlignmentResult:
        self._log.append("alignment")
        self.last_request = request
        return self._result


def test_runner_calls_services_in_order(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: Any,
    recording_output_manager: CliOutputManager,
) -> None:
    """Runner should sequence MetadataResolver before AlignmentWorkflow."""

    files = _write_media(cli_runner_env)
    monkeypatch.setattr(media_utils, "discover_media", lambda _root: list(files))
    metadata_result = _make_metadata_result(files)
    alignment_result = AlignmentResult(plans=list(metadata_result.plans), summary=None, display=None)
    call_order: list[str] = []
    metadata_resolver = _RecordingMetadataResolver(metadata_result, call_order)
    alignment_workflow = _RecordingAlignmentWorkflow(alignment_result, call_order)

    def _stop(*_args: object, **_kwargs: object) -> None:
        raise SentinelError("halt after alignment")

    monkeypatch.setattr(selection_utils, "init_clips", _stop)
    dependencies = _build_dependencies(metadata_resolver, alignment_workflow)
    request = runner_module.RunRequest(
        config_path=str(cli_runner_env.config_path),
        reporter=recording_output_manager,
    )

    with pytest.raises(SentinelError):
        runner_module.run(request, dependencies=dependencies)

    assert call_order == ["metadata", "alignment"]
    assert metadata_resolver.last_request is not None
    assert list(metadata_resolver.last_request.files) == list(files)
    assert alignment_workflow.last_request is not None
    assert alignment_workflow.last_request.analyze_path == files[0]
    assert list(alignment_workflow.last_request.plans) == list(metadata_result.plans)


def test_runner_warns_when_legacy_requested(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: Any,
    recording_output_manager: CliOutputManager,
) -> None:
    """Legacy toggles should be ignored but surfaced as warnings."""

    files = _write_media(cli_runner_env)
    monkeypatch.setattr(media_utils, "discover_media", lambda _root: list(files))
    metadata_result = _make_metadata_result(files)
    alignment_result = AlignmentResult(plans=list(metadata_result.plans), summary=None, display=None)
    call_order: list[str] = []
    metadata_resolver = _RecordingMetadataResolver(metadata_result, call_order)
    alignment_workflow = _RecordingAlignmentWorkflow(alignment_result, call_order)

    def _stop(*_args: object, **_kwargs: object) -> None:
        raise SentinelError("halt after alignment")

    monkeypatch.setattr(selection_utils, "init_clips", _stop)
    dependencies = _build_dependencies(metadata_resolver, alignment_workflow)
    request = runner_module.RunRequest(
        config_path=str(cli_runner_env.config_path),
        reporter=recording_output_manager,
        service_mode_override=False,
    )

    with pytest.raises(SentinelError):
        runner_module.run(request, dependencies=dependencies)

    warnings = recording_output_manager.iter_warnings()
    assert any("Legacy runner path has been retired" in warning for warning in warnings)


def test_metadata_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: Any,
    recording_output_manager: CliOutputManager,
) -> None:
    """CLIAppError raised by the metadata service should bubble out unchanged."""

    files = _write_media(cli_runner_env)
    monkeypatch.setattr(media_utils, "discover_media", lambda _root: list(files))

    class _ExplodingMetadata:
        def resolve(self, _request: MetadataResolveRequest) -> MetadataResolveResult:
            raise CLIAppError("metadata boom")

    metadata_resolver = _ExplodingMetadata()
    alignment_result = AlignmentResult(plans=[], summary=None, display=None)
    alignment_workflow = _RecordingAlignmentWorkflow(alignment_result, [])
    dependencies = _build_dependencies(metadata_resolver, alignment_workflow)
    request = runner_module.RunRequest(
        config_path=str(cli_runner_env.config_path),
        reporter=recording_output_manager,
    )

    with pytest.raises(CLIAppError, match="metadata boom"):
        runner_module.run(request, dependencies=dependencies)


def test_alignment_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: Any,
    recording_output_manager: CliOutputManager,
) -> None:
    """Failures from AlignmentWorkflow should also surface as CLIAppError."""

    files = _write_media(cli_runner_env)
    monkeypatch.setattr(media_utils, "discover_media", lambda _root: list(files))
    metadata_result = _make_metadata_result(files)
    metadata_resolver = _RecordingMetadataResolver(metadata_result, [])

    class _ExplodingAlignment:
        def run(self, _request: AlignmentRequest) -> AlignmentResult:
            raise CLIAppError("alignment boom")

    alignment_workflow = _ExplodingAlignment()
    dependencies = _build_dependencies(metadata_resolver, alignment_workflow)
    request = runner_module.RunRequest(
        config_path=str(cli_runner_env.config_path),
        reporter=recording_output_manager,
    )

    with pytest.raises(CLIAppError, match="alignment boom"):
        runner_module.run(request, dependencies=dependencies)


def _build_context(tmp_path: Path) -> tuple[runner_module.RunContext, JsonTail, dict[str, Any], AppConfig]:
    cfg = build_service_config(tmp_path)
    cfg.report.enable = True
    cfg.slowpics.auto_upload = False
    json_tail = build_base_json_tail(cfg)
    layout_data = {
        "slowpics": json_tail["slowpics"],
        "report": json_tail["report"],
    }
    files = [tmp_path / "Alpha.mkv", tmp_path / "Beta.mkv"]
    for file in files:
        file.write_bytes(b"0")
    metadata_result = _make_metadata_result(files)
    context = runner_module.RunContext(
        plans=list(metadata_result.plans),
        metadata=list(metadata_result.metadata),
        json_tail=json_tail,
        layout_data=layout_data,
        metadata_title=metadata_result.metadata_title,
        analyze_path=metadata_result.analyze_path,
        slowpics_title_inputs=metadata_result.slowpics_title_inputs,
        slowpics_final_title=metadata_result.slowpics_final_title,
        slowpics_resolved_base=metadata_result.slowpics_resolved_base,
        slowpics_tmdb_disclosure_line=None,
        slowpics_verbose_tmdb_tag=None,
        tmdb_notes=list(metadata_result.tmdb_notes),
    )
    return context, json_tail, layout_data, cfg


def test_publish_results_uses_services(tmp_path: Path) -> None:
    context_old, json_tail, layout_data, cfg = _build_context(tmp_path)
    reporter = StubReporter()
    report_publisher = _StubReportPublisher()
    slowpics_publisher = _StubSlowpicsPublisher()

    deps = _build_dependencies(
        metadata_resolver=object(),
        alignment_workflow=object(),
        report_publisher=report_publisher,
        slowpics_publisher=slowpics_publisher,
    )

    from src.frame_compare.orchestration.state import CoordinatorContext
    from src.frame_compare.orchestration.phases.publish import PublishPhase

    request = runner_module.RunRequest(config_path=None, reporter=reporter)
    coord_context = CoordinatorContext(request=request, dependencies=deps)
    
    # Mock env
    coord_context.env = SimpleNamespace(
        cfg=cfg,
        reporter=reporter,
        out_dir=tmp_path,
        report_enabled=True,
        root=tmp_path,
        collected_warnings=[],
    )  # type: ignore

    coord_context.json_tail = json_tail
    coord_context.layout_data = layout_data
    coord_context.plans = context_old.plans
    coord_context.image_paths = ["img-a.png"]
    coord_context.frames = [1, 2]
    coord_context.selection_details = {}
    
    coord_context.slowpics_title_inputs = context_old.slowpics_title_inputs
    coord_context.slowpics_final_title = context_old.slowpics_final_title
    coord_context.slowpics_resolved_base = context_old.slowpics_resolved_base
    coord_context.slowpics_tmdb_disclosure_line = context_old.slowpics_tmdb_disclosure_line
    coord_context.slowpics_verbose_tmdb_tag = context_old.slowpics_verbose_tmdb_tag

    phase = PublishPhase()
    phase.execute(coord_context)

    assert report_publisher.call_count == 1
    assert slowpics_publisher.call_count == 1
    assert coord_context.slowpics_url == "https://slow.pics/c/test"
    assert coord_context.report_path is None


def test_reporter_flags_initialized_with_service_context(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner_env: Any,
    recording_output_manager: CliOutputManager,
) -> None:
    """Runner should set upload/tmdb flags before service sequencing."""

    cli_runner_env.cfg.slowpics.auto_upload = True
    cli_runner_env.reinstall(cli_runner_env.cfg)
    files = _write_media(cli_runner_env)
    monkeypatch.setattr(media_utils, "discover_media", lambda _root: list(files))
    metadata_result = _make_metadata_result(files)
    alignment_result = AlignmentResult(plans=list(metadata_result.plans), summary=None, display=None)
    call_order: list[str] = []
    metadata_resolver = _RecordingMetadataResolver(metadata_result, call_order)
    alignment_workflow = _RecordingAlignmentWorkflow(alignment_result, call_order)

    def _stop(*_args: object, **_kwargs: object) -> None:
        raise SentinelError("stop after metadata/alignment")

    monkeypatch.setattr(selection_utils, "init_clips", _stop)
    dependencies = _build_dependencies(metadata_resolver, alignment_workflow)
    request = runner_module.RunRequest(
        config_path=str(cli_runner_env.config_path),
        reporter=recording_output_manager,
    )

    with pytest.raises(SentinelError):
        runner_module.run(request, dependencies=dependencies)

    assert recording_output_manager.flags.get("upload_enabled") is True
    assert recording_output_manager.flags.get("tmdb_resolved") is False
