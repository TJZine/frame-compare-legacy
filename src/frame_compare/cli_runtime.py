"""Runtime data structures and CLI helpers shared between Click wiring and the runner."""

from __future__ import annotations

import io
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
    cast,
)

from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, ProgressColumn

from src.frame_compare.cli_layout import CliLayoutError, CliLayoutRenderer, load_cli_layout
from src.frame_compare.layout_utils import (
    color_text as _color_text,
)
from src.frame_compare.layout_utils import (
    format_kv as _format_kv,
)
from src.frame_compare.layout_utils import (
    normalise_vspreview_mode as _normalise_vspreview_mode,
)
from src.frame_compare.layout_utils import (
    plan_label as _plan_label,
)
from src.frame_compare.layout_utils import (
    plan_label_parts as _plan_label_parts,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.datatypes import AppConfig


def _default_props_dict() -> Dict[str, object]:
    return {}


@dataclass
class ClipProbeSnapshot:
    """
    Serialized snapshot of clip metadata captured during the probe phase.

    Attributes:
        trim_start (int): Trim start applied when the snapshot was recorded.
        trim_end (Optional[int]): Trim end applied when the snapshot was recorded.
        fps_override (Optional[Tuple[int, int]]): User-supplied FPS override at probe time.
        applied_fps (Optional[Tuple[int, int]]): FPS tuple actually supplied to VapourSynth.
        effective_fps (Optional[Tuple[int, int]]): FPS reported back by the clip.
        source_fps (Optional[Tuple[int, int]]): Native FPS reported by the source plugin.
        source_num_frames (Optional[int]): Total number of frames reported for the trimmed clip.
        source_width (Optional[int]): Clip width observed at probe time.
        source_height (Optional[int]): Clip height observed at probe time.
        source_frame_props (Dict[str, Any]): Snapshot of frame props preserved for tonemapping.
        tonemap_prop_keys (Tuple[str, ...]): Sorted list of tonemapping-critical prop keys.
        metadata_digest (str): Hash of the recorded metadata payload for quick change checks.
        cache_key (Optional[str]): Stable cache key derived from file stats + trim/FPS inputs.
        cache_path (Optional[Path]): Path to the persisted JSON payload, when available.
        cached_at (Optional[str]): ISO8601 timestamp describing when the snapshot hit disk.
        clip (Optional[object]): Live VapourSynth clip handle (never serialized) for reuse.
    """

    trim_start: int
    trim_end: Optional[int]
    fps_override: Optional[Tuple[int, int]]
    applied_fps: Optional[Tuple[int, int]]
    effective_fps: Optional[Tuple[int, int]]
    source_fps: Optional[Tuple[int, int]]
    source_num_frames: Optional[int]
    source_width: Optional[int]
    source_height: Optional[int]
    source_frame_props: Dict[str, object] = field(default_factory=_default_props_dict)
    tonemap_prop_keys: Tuple[str, ...] = field(default_factory=tuple)
    metadata_digest: str = ""
    cache_key: Optional[str] = None
    cache_path: Optional[Path] = None
    cached_at: Optional[str] = None
    clip: Optional[object] = None


@dataclass
class _ClipPlan:
    """
    Internal plan describing how a source clip should be processed.

    Attributes:
        path (Path): Path to the source media file.
        metadata (Dict[str, str]): Metadata parsed from the source file name.
        trim_start (int): Leading frames to skip before analysis.
        trim_end (Optional[int]): Final frame index (exclusive) or ``None`` to include the full clip.
        fps_override (Optional[Tuple[int, int]]): Rational frame-rate override applied during processing.
        use_as_reference (bool): Whether the clip should drive alignment decisions.
        clip (Optional[object]): Lazily populated VapourSynth clip reference.
        source_frame_props (Optional[Dict[str, Any]]): Snapshot of the source clip's frame props before any trims/padding.
        effective_fps (Optional[Tuple[int, int]]): Frame rate after alignment adjustments.
        applied_fps (Optional[Tuple[int, int]]): Frame rate enforced by user configuration.
        source_fps (Optional[Tuple[int, int]]): Native frame rate detected from the source file.
        source_num_frames (Optional[int]): Total number of frames available in the source clip.
        source_width (Optional[int]): Source clip width in pixels.
        source_height (Optional[int]): Source clip height in pixels.
        has_trim_start_override (bool): ``True`` when a manual trim start was supplied.
        has_trim_end_override (bool): ``True`` when a manual trim end was supplied.
        alignment_frames (int): Number of frames trimmed during audio alignment.
        alignment_status (str): Human-friendly status describing the alignment result.
        probe_snapshot (Optional[ClipProbeSnapshot]): Cached probe metadata for reuse.
        probe_cache_key (Optional[str]): Cache key derived from file stats + trim/FPS inputs.
        probe_cache_path (Optional[Path]): Location of the persisted probe snapshot, if any.
    """

    path: Path
    metadata: Dict[str, str]
    trim_start: int = 0
    trim_end: Optional[int] = None
    fps_override: Optional[Tuple[int, int]] = None
    use_as_reference: bool = False
    clip: Optional[object] = None
    source_frame_props: Optional[Dict[str, object]] = None
    effective_fps: Optional[Tuple[int, int]] = None
    applied_fps: Optional[Tuple[int, int]] = None
    source_fps: Optional[Tuple[int, int]] = None
    source_num_frames: Optional[int] = None
    source_width: Optional[int] = None
    source_height: Optional[int] = None
    has_trim_start_override: bool = False
    has_trim_end_override: bool = False
    alignment_frames: int = 0
    alignment_status: str = ""
    probe_snapshot: Optional[ClipProbeSnapshot] = None
    probe_cache_key: Optional[str] = None
    probe_cache_path: Optional[Path] = None


ClipPlan = _ClipPlan

_OverrideValue = TypeVar("_OverrideValue")


class SlowpicsTitleInputs(TypedDict):
    resolved_base: Optional[str]
    collection_name: Optional[str]
    collection_suffix: str


class SlowpicsTitleBlock(TypedDict):
    inputs: SlowpicsTitleInputs
    final: Optional[str]


class SlowpicsJSON(TypedDict):
    enabled: bool
    title: SlowpicsTitleBlock
    url: Optional[str]
    shortcut_path: Optional[str]
    shortcut_written: bool
    shortcut_error: Optional[str]
    deleted_screens_dir: bool
    is_public: bool
    is_hentai: bool
    remove_after_days: int


class AudioAlignmentJSON(TypedDict, total=False):
    enabled: bool
    reference_stream: Optional[str]
    target_stream: dict[str, object]
    offsets_sec: dict[str, object]
    offsets_frames: dict[str, object]
    measurements: dict[str, dict[str, object]]
    stream_lines: list[str]
    stream_lines_text: str
    offset_lines: list[str]
    offset_lines_text: str
    preview_paths: list[str]
    confirmed: bool | str | None
    offsets_filename: str
    use_vspreview: bool
    vspreview_manual_offsets: dict[str, object]
    vspreview_manual_deltas: dict[str, object]
    vspreview_reference_trim: Optional[int]
    manual_trim_summary: list[str]
    suggestion_mode: bool
    suggested_frames: dict[str, int]
    manual_trim_starts: dict[str, int]
    vspreview_script: Optional[str]
    vspreview_invoked: bool
    vspreview_exit_code: Optional[int]


class TrimClipEntry(TypedDict):
    lead_f: int
    trail_f: int
    lead_s: float
    trail_s: float


class TrimsJSON(TypedDict):
    per_clip: dict[str, TrimClipEntry]


class ReportJSON(TypedDict, total=False):
    enabled: bool
    path: Optional[str]
    output_dir: str
    open_after_generate: bool
    opened: bool
    mode: str


class ViewerJSON(TypedDict, total=False):
    mode: str
    mode_display: str
    destination: Optional[str]
    destination_label: str


class JsonTail(TypedDict):
    clips: list[dict[str, object]]
    trims: TrimsJSON
    window: dict[str, object]
    alignment: dict[str, object]
    audio_alignment: AudioAlignmentJSON
    analysis: dict[str, object]
    render: dict[str, object]
    tonemap: dict[str, object]
    overlay: dict[str, object]
    verify: dict[str, object]
    cache: dict[str, object]
    slowpics: SlowpicsJSON
    report: ReportJSON
    viewer: ViewerJSON
    warnings: list[str]
    workspace: dict[str, object]
    vspreview_mode: Optional[str]
    suggested_frames: Optional[int]
    suggested_seconds: float
    vspreview_offer: Optional[dict[str, object]]


class ClipRecord(TypedDict):
    label: str
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    duration_tc: str
    path: str


class TrimSummary(TypedDict):
    label: str
    lead_frames: int
    lead_seconds: float
    trail_frames: int
    trail_seconds: float


def _coerce_str_mapping(value: object) -> dict[str, object]:
    """Return a shallow copy of *value* if it is a mapping with string-like keys."""

    if isinstance(value, MappingABC):
        source = cast(Mapping[Any, object], value)
        result: dict[str, object] = {}
        for key, item in source.items():
            result[str(key)] = item
        return result
    return {}


coerce_str_mapping = _coerce_str_mapping


def _ensure_audio_alignment_block(json_tail: JsonTail) -> AudioAlignmentJSON:
    """Ensure the audio alignment block exists and return a mutable mapping."""

    block = cast(Optional[AudioAlignmentJSON], json_tail.get("audio_alignment"))
    if block is None:
        new_block: AudioAlignmentJSON = {}
        json_tail["audio_alignment"] = new_block
        return new_block
    return block


ensure_audio_alignment_block = _ensure_audio_alignment_block


def _ensure_slowpics_block(json_tail: JsonTail, cfg: "AppConfig") -> SlowpicsJSON:
    """Ensure that ``json_tail`` contains a slow.pics block and return it."""

    block = cast(Optional[SlowpicsJSON], json_tail.get("slowpics"))
    if block is None:
        block = SlowpicsJSON(
            enabled=bool(cfg.slowpics.auto_upload),
            title=SlowpicsTitleBlock(
                inputs=SlowpicsTitleInputs(
                    resolved_base=None,
                    collection_name=None,
                    collection_suffix=getattr(cfg.slowpics, "collection_suffix", ""),
                ),
                final=None,
            ),
            url=None,
            shortcut_path=None,
            shortcut_written=False,
            shortcut_error=None,
            deleted_screens_dir=False,
            is_public=bool(cfg.slowpics.is_public),
            is_hentai=bool(cfg.slowpics.is_hentai),
            remove_after_days=int(cfg.slowpics.remove_after_days),
        )
        json_tail["slowpics"] = block
        return block

    existing_block: SlowpicsJSON = block
    if "url" not in existing_block:
        existing_block["url"] = None
    if "shortcut_path" not in existing_block:
        existing_block["shortcut_path"] = None
    if "shortcut_written" not in existing_block:
        existing_block["shortcut_written"] = False
    if "shortcut_error" not in existing_block:
        existing_block["shortcut_error"] = None
    if "deleted_screens_dir" not in existing_block:
        existing_block["deleted_screens_dir"] = False
    return existing_block

ensure_slowpics_block = _ensure_slowpics_block


class CLIAppError(RuntimeError):
    """Raised when the CLI cannot complete its work."""

    def __init__(self, message: str, *, code: int = 1, rich_message: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.rich_message = rich_message or message





class CliOutputManagerProtocol(Protocol):
    quiet: bool
    verbose: bool
    console: Console
    flags: Dict[str, Any]
    values: Dict[str, Any]

    def set_flag(self, key: str, value: Any) -> None: ...

    def update_values(self, mapping: Mapping[str, Any]) -> None: ...

    def warn(self, text: str) -> None: ...

    def get_warnings(self) -> List[str]: ...

    def render_sections(self, section_ids: Iterable[str]) -> None: ...

    def create_progress(self, progress_id: str, *, transient: bool = False) -> Progress: ...

    def update_progress_state(self, progress_id: str, **state: Any) -> None: ...

    def banner(self, text: str) -> None: ...

    def section(self, title: str) -> None: ...

    def line(self, text: str) -> None: ...

    def verbose_line(self, text: str) -> None: ...

    def progress(self, *columns: ProgressColumn, transient: bool = False) -> Progress: ...

    def iter_warnings(self) -> List[str]: ...


class CliOutputManager:
    """Layout-driven CLI presentation controller."""

    def __init__(
        self,
        *,
        quiet: bool,
        verbose: bool,
        no_color: bool,
        layout_path: Path,
        console: Console | None = None,
    ) -> None:
        self.quiet = quiet
        self.verbose = verbose and not quiet
        self.no_color = no_color
        self.console = console or Console(no_color=no_color, highlight=False)
        try:
            self.layout = load_cli_layout(layout_path)
        except CliLayoutError as exc:
            raise CLIAppError(str(exc)) from exc
        self.renderer = CliLayoutRenderer(
            self.layout,
            self.console,
            quiet=quiet,
            verbose=self.verbose,
            no_color=no_color,
        )
        self.flags: Dict[str, Any] = {
            "quiet": quiet,
            "verbose": self.verbose,
            "no_color": no_color,
        }
        self.values: Dict[str, Any] = {
            "theme": {
                "colors": dict(self.layout.theme.colors),
                "symbols": dict(self.renderer.symbols),
            }
        }
        self._warnings: List[str] = []

    def set_flag(self, key: str, value: Any) -> None:
        self.flags[key] = value

    def update_values(self, mapping: Mapping[str, Any]) -> None:
        self.values.update(mapping)

    def warn(self, text: str) -> None:
        self._warnings.append(text)

    def get_warnings(self) -> List[str]:
        return list(self._warnings)

    def render_sections(self, section_ids: Iterable[str]) -> None:
        target_ids = set(section_ids)
        self.renderer.bind_context(self.values, self.flags)
        for section in self.layout.sections:
            section_id = section.get("id")
            if section_id in target_ids:
                self.renderer.render_section(section, self.values, self.flags)

    def create_progress(self, progress_id: str, *, transient: bool = False) -> Progress:
        self.renderer.bind_context(self.values, self.flags)
        return self.renderer.create_progress(progress_id, transient=transient)

    def update_progress_state(self, progress_id: str, **state: Any) -> None:
        self.renderer.update_progress_state(progress_id, state=state)

    def banner(self, text: str) -> None:
        if self.quiet:
            self.console.print(text)
            return
        self.console.print(f"[bold bright_cyan]{escape(text)}[/]")

    def section(self, title: str) -> None:
        if self.quiet:
            return
        self.console.print(f"[bold cyan]{title}[/]")

    def line(self, text: str) -> None:
        if self.quiet:
            return
        self.console.print(text)

    def verbose_line(self, text: str) -> None:
        if self.quiet or not self.verbose:
            return
        if not text:
            return
        self.console.print(f"[dim]{escape(text)}[/]")

    def progress(self, *columns: ProgressColumn, transient: bool = False) -> Progress:
        return Progress(*columns, console=self.console, transient=transient)

    def iter_warnings(self) -> List[str]:
        return list(self._warnings)


class NullCliOutputManager(CliOutputManagerProtocol):
    """
    Minimal CliOutputManager implementation that discards console output.

    Used by automation callers (or quiet runs) that want to suppress Rich layout
    rendering while still collecting warnings and JSON-tail metadata.
    """

    def __init__(
        self,
        *,
        quiet: bool,
        verbose: bool,
        no_color: bool,
        layout_path: Path | None = None,
        console: Console | None = None,
    ) -> None:
        self.quiet = True
        self.verbose = False
        self.no_color = no_color
        self.console = console or Console(
            file=io.StringIO(),
            no_color=True,
            highlight=False,
            force_terminal=False,
            width=80,
        )
        self.layout = None
        self.flags: Dict[str, Any] = {
            "quiet": True,
            "verbose": False,
            "no_color": no_color,
        }
        self.values: Dict[str, Any] = {}
        self._warnings: List[str] = []

    def set_flag(self, key: str, value: Any) -> None:
        self.flags[key] = value

    def update_values(self, mapping: Mapping[str, Any]) -> None:
        self.values.update(mapping)

    def warn(self, text: str) -> None:
        self._warnings.append(text)

    def get_warnings(self) -> List[str]:
        return list(self._warnings)

    def render_sections(self, section_ids: Iterable[str]) -> None:  # noqa: ARG002
        return None

    def create_progress(self, progress_id: str, *, transient: bool = False) -> Progress:  # noqa: ARG002
        return Progress(console=self.console, transient=transient)

    def update_progress_state(self, progress_id: str, **state: Any) -> None:  # noqa: ARG002
        return None

    def banner(self, text: str) -> None:  # noqa: ARG002
        return None

    def section(self, title: str) -> None:  # noqa: ARG002
        return None

    def line(self, text: str) -> None:  # noqa: ARG002
        return None

    def verbose_line(self, text: str) -> None:  # noqa: ARG002
        return None

    def progress(self, *columns: ProgressColumn, transient: bool = False) -> Progress:
        return Progress(*columns, console=self.console, transient=transient)

    def iter_warnings(self) -> List[str]:
        return list(self._warnings)



from src.frame_compare.alignment import models as _alignment_models  # noqa: E402,I001

AudioAlignmentSummary = _alignment_models.AudioAlignmentSummary
AudioMeasurementDetail = _alignment_models.AudioMeasurementDetail
AudioAlignmentDisplayData = _alignment_models.AudioAlignmentDisplayData
_AudioAlignmentSummary = AudioAlignmentSummary
_AudioMeasurementDetail = AudioMeasurementDetail
_AudioAlignmentDisplayData = AudioAlignmentDisplayData

__all__ = [
    "CLIAppError",
    "CliOutputManager",
    "CliOutputManagerProtocol",
    "NullCliOutputManager",
    "JsonTail",
    "SlowpicsJSON",
    "SlowpicsTitleBlock",
    "SlowpicsTitleInputs",
    "ViewerJSON",
    "_AudioAlignmentDisplayData",
    "_AudioAlignmentSummary",
    "_AudioMeasurementDetail",
    "_ClipPlan",
    "_coerce_str_mapping",
    "_color_text",
    "_ensure_audio_alignment_block",
    "_ensure_slowpics_block",
    "_format_kv",
    "_normalise_vspreview_mode",
    "_plan_label",
    "_plan_label_parts",
    "AudioAlignmentDisplayData",
    "AudioAlignmentSummary",
    "AudioAlignmentJSON",
    "AudioMeasurementDetail",
    "ClipPlan",
    "ClipRecord",
    "coerce_str_mapping",
    "ensure_audio_alignment_block",
    "ensure_slowpics_block",
    "ReportJSON",
    "TrimClipEntry",
    "TrimSummary",
    "TrimsJSON",
]
