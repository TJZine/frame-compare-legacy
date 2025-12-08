"""Run result snapshot types and helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeAlias, cast

from src.frame_compare import runtime_utils
from src.frame_compare.cli_runtime import CliOutputManagerProtocol
from src.frame_compare.layout_utils import color_text as _color_text

JsonPrimitive = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

RUN_RESULT_SNAPSHOT_SCHEMA = 1
RUN_RESULT_SNAPSHOT_FILENAME = ".frame_compare.run.json"


class SnapshotDecodeError(RuntimeError):
    """Raised when cached snapshot data cannot be decoded safely."""


class ResultSource(str, Enum):
    """Describe where a result snapshot originated from."""

    LIVE = "live"
    CACHE = "cache"


class SectionAvailability(str, Enum):
    """Describe how much data is available for a rendered CLI section."""

    FULL = "full"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass(frozen=True)
class SectionSnapshot:
    """Record metadata for each CLI section."""

    id: str
    label: str
    availability: SectionAvailability = SectionAvailability.FULL
    note: str | None = None


@dataclass(frozen=True)
class SectionState:
    """Mutable availability metadata recorded before snapshot serialization."""

    availability: SectionAvailability
    note: str | None = None


def _new_json_mapping() -> dict[str, JsonValue]:
    return {}


def _new_sections_mapping() -> dict[str, SectionSnapshot]:
    return {}


def _new_str_list() -> list[str]:
    return []


def _new_int_list() -> list[int]:
    return []


def _coerce_str_list(raw_items: Sequence[Any] | None) -> list[str]:
    if raw_items is None:
        return []
    coerced: list[str] = []
    for item in raw_items:
        try:
            coerced.append(str(item))
        except (ValueError, TypeError):  # pragma: no cover - extremely rare edge cases
            continue
    return coerced


def _coerce_int_list(raw_items: Sequence[Any] | None) -> list[int]:
    if raw_items is None:
        return []
    coerced: list[int] = []
    for item in raw_items:
        try:
            coerced.append(int(item))
        except (TypeError, ValueError):
            continue
    return coerced


@dataclass
class RunResultSnapshot:
    """Serializable snapshot of CLI output and metadata."""

    schema_version: int = RUN_RESULT_SNAPSHOT_SCHEMA
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: ResultSource = ResultSource.LIVE
    cli_version: str = "unknown"
    values: dict[str, JsonValue] = field(default_factory=_new_json_mapping)
    flags: dict[str, JsonValue] = field(default_factory=_new_json_mapping)
    sections: dict[str, SectionSnapshot] = field(default_factory=_new_sections_mapping)
    warnings: list[str] = field(default_factory=_new_str_list)
    files: list[str] = field(default_factory=_new_str_list)
    frames: list[int] = field(default_factory=_new_int_list)
    image_paths: list[str] = field(default_factory=_new_str_list)
    slowpics_url: str | None = None
    report_path: str | None = None
    json_tail: dict[str, JsonValue] | None = None

    def to_json_dict(self) -> dict[str, JsonValue]:
        """Convert the snapshot into a JSON-serialisable mapping."""

        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "source": self.source.value,
            "cli_version": self.cli_version,
            "values": self.values,
            "flags": self.flags,
            "sections": {
                section_id: {
                    "label": section.label,
                    "availability": section.availability.value,
                    "note": section.note,
                }
                for section_id, section in self.sections.items()
            },
            "warnings": list(self.warnings),
            "files": list(self.files),
            "frames": list(self.frames),
            "image_paths": list(self.image_paths),
            "slowpics_url": self.slowpics_url,
            "report_path": self.report_path,
            "json_tail": self.json_tail,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "RunResultSnapshot":
        """Hydrate a snapshot from the cache payload."""

        try:
            schema_version = int(payload.get("schema_version", 0))
        except (TypeError, ValueError) as exc:
            raise SnapshotDecodeError("Invalid schema_version") from exc
        created_at = datetime.now(timezone.utc)
        created_at_raw = payload.get("created_at")
        if created_at_raw is None:
            created_at = datetime.now(timezone.utc)
        elif isinstance(created_at_raw, str):
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError as exc:
                raise SnapshotDecodeError("Invalid created_at value") from exc
        else:
            raise SnapshotDecodeError("created_at must be an ISO formatted string")
        source_raw = payload.get("source")
        if source_raw is None:
            source = ResultSource.LIVE
        elif isinstance(source_raw, str):
            try:
                source = ResultSource(source_raw)
            except ValueError as exc:
                raise SnapshotDecodeError("Unrecognised snapshot source") from exc
        else:
            raise SnapshotDecodeError("Snapshot source must be a string")
        sections_payload = payload.get("sections")
        sections: dict[str, SectionSnapshot] = {}
        if isinstance(sections_payload, MappingABC):
            sections_mapping = cast(Mapping[str, Any], sections_payload)
            for section_id_raw, section_value in sections_mapping.items():
                if not isinstance(section_value, MappingABC):
                    continue
                section_map = cast(Mapping[str, Any], section_value)
                section_id = str(section_id_raw)
                label_raw = section_map.get("label")
                label = str(label_raw) if label_raw is not None else section_id
                availability_value = section_map.get("availability")
                if isinstance(availability_value, str):
                    try:
                        availability = SectionAvailability(availability_value)
                    except ValueError:
                        availability = SectionAvailability.MISSING
                else:
                    availability = SectionAvailability.MISSING
                note_value = section_map.get("note")
                note_text = str(note_value) if note_value is not None else None
                sections[section_id] = SectionSnapshot(
                    id=section_id,
                    label=label,
                    availability=availability,
                    note=note_text,
                )
        values_payload = payload.get("values")
        flags_payload = payload.get("flags")
        files_payload = payload.get("files")
        frames_payload = payload.get("frames")
        image_paths_payload = payload.get("image_paths")
        warnings_payload = payload.get("warnings")
        json_tail_sentinel = object()
        json_tail_payload = payload.get("json_tail", json_tail_sentinel)
        if json_tail_payload is json_tail_sentinel or json_tail_payload is None:
            json_tail_map: Mapping[str, Any] | None = None
        else:
            if not hasattr(json_tail_payload, "items"):
                raise SnapshotDecodeError("json_tail must be a mapping when present")
            json_tail_map = cast(Mapping[str, Any], json_tail_payload)
        values_map = (
            cast(Mapping[str, Any], values_payload) if isinstance(values_payload, MappingABC) else None
        )
        flags_map = (
            cast(Mapping[str, Any], flags_payload) if isinstance(flags_payload, MappingABC) else None
        )
        files_list = _coerce_str_list(cast(Sequence[Any], files_payload) if isinstance(files_payload, list) else None)
        frames_list = _coerce_int_list(
            cast(Sequence[Any], frames_payload) if isinstance(frames_payload, list) else None
        )
        image_paths_list = _coerce_str_list(
            cast(Sequence[Any], image_paths_payload) if isinstance(image_paths_payload, list) else None
        )
        warnings_list = _coerce_str_list(
            cast(Sequence[Any], warnings_payload) if isinstance(warnings_payload, list) else None
        )
        instance = cls(
            schema_version=schema_version,
            created_at=created_at,
            source=source,
            cli_version=str(payload.get("cli_version") or "unknown"),
            values=_coerce_json_mapping(values_map),
            flags=_coerce_json_mapping(flags_map),
            sections=sections,
            warnings=warnings_list,
            files=files_list,
            frames=frames_list,
            image_paths=image_paths_list,
            slowpics_url=str(payload.get("slowpics_url")) if payload.get("slowpics_url") is not None else None,
            report_path=str(payload.get("report_path")) if payload.get("report_path") is not None else None,
            json_tail=_coerce_json_mapping(json_tail_map) if json_tail_map is not None else None,
        )
        return instance


@dataclass(frozen=True)
class RenderOptions:
    """Toggle cached output presentation."""

    show_partial: bool = False
    show_missing_sections: bool = True
    no_cache_hint: str = "--no-cache"
    partial_label: str = "(from cache, incomplete)"
    missing_label: str = "not available from cache; rerun for details"

    @property
    def show_missing(self) -> bool:
        """Backwards-compatible alias for legacy callers."""

        return self.show_missing_sections


def snapshot_path(out_dir: Path) -> Path:
    """Return the canonical snapshot path inside an output directory."""

    return out_dir / RUN_RESULT_SNAPSHOT_FILENAME


def build_snapshot(
    *,
    values: Mapping[str, Any],
    flags: Mapping[str, Any],
    layout_sections: Iterable[Mapping[str, Any]],
    section_states: Mapping[str, SectionState] | None,
    files: Sequence[Path],
    frames: Sequence[int],
    image_paths: Sequence[str],
    slowpics_url: str | None,
    report_path: Path | None,
    warnings: Sequence[str],
    json_tail: Mapping[str, Any] | None,
    source: ResultSource,
    cli_version: str,
) -> RunResultSnapshot:
    """Build a snapshot from the runner state."""

    sections: dict[str, SectionSnapshot] = {}
    for section in layout_sections:
        section_id_raw = section.get("id")
        if not section_id_raw:
            continue
        section_id = str(section_id_raw)
        label = str(section.get("title") or section.get("title_badge") or section_id)
        state = section_states.get(section_id) if section_states else None
        sections[section_id] = SectionSnapshot(
            id=section_id,
            label=label,
            availability=state.availability if state else SectionAvailability.FULL,
            note=state.note if state else None,
        )
    sanitized_values = _coerce_json_mapping(values)
    sanitized_flags = _coerce_json_mapping(flags)
    sanitized_tail = _coerce_json_mapping(json_tail) if json_tail is not None else None
    snapshot = RunResultSnapshot(
        schema_version=RUN_RESULT_SNAPSHOT_SCHEMA,
        created_at=datetime.now(timezone.utc),
        source=source,
        cli_version=cli_version,
        values=sanitized_values,
        flags=sanitized_flags,
        sections=sections,
        warnings=list(warnings),
        files=[str(path) for path in files],
        frames=list(frames),
        image_paths=list(image_paths),
        slowpics_url=slowpics_url,
        report_path=str(report_path) if report_path is not None else None,
        json_tail=sanitized_tail,
    )
    return snapshot


def write_snapshot(path: Path, snapshot: RunResultSnapshot) -> None:
    """Persist the snapshot payload to disk atomically."""

    payload = snapshot.to_json_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_snapshot(path: Path) -> RunResultSnapshot | None:
    """Load a snapshot from the cache file."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, MappingABC):
        return None
    try:
        snapshot = RunResultSnapshot.from_json_dict(cast(Mapping[str, Any], payload))
    except SnapshotDecodeError:
        return None
    if snapshot.schema_version != RUN_RESULT_SNAPSHOT_SCHEMA:
        return None
    return snapshot


def render_run_result(
    *,
    snapshot: RunResultSnapshot,
    reporter: CliOutputManagerProtocol,
    layout_sections: Iterable[Mapping[str, Any]],
    options: RenderOptions,
) -> None:
    """Render the stored sections using the provided reporter."""

    reporter.update_values(snapshot.values)
    for key, value in snapshot.flags.items():
        reporter.set_flag(key, value)
    if snapshot.source == ResultSource.CACHE and not reporter.quiet:
        timestamp = snapshot.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        reporter.banner(
            f"Using cached analysis from {timestamp}. "
            f"Run again with {options.no_cache_hint} for a fresh capture."
        )
    rendered_summary = False
    layout_list = list(layout_sections)
    for section in layout_list:
        section_id_raw = section.get("id")
        if not section_id_raw:
            continue
        section_id = str(section_id_raw)
        section_state = snapshot.sections.get(section_id)
        if section_state is None:
            if options.show_missing_sections:
                _render_missing_section(reporter, section, options)
            continue
        if section_state.availability == SectionAvailability.MISSING:
            if options.show_missing_sections:
                _render_missing_section(reporter, section, options, section_state.note)
            continue
        if section_state.availability == SectionAvailability.PARTIAL and not options.show_partial:
            continue
        if section_state.availability == SectionAvailability.PARTIAL and options.show_partial:
            _render_partial_header(reporter, section, options)
        reporter.render_sections([section_id])
        if section_id == "summary":
            rendered_summary = True
    compatibility_required = bool(
        reporter.flags.get("compat.summary_fallback")
        or reporter.flags.get("compatibility_mode")
        or reporter.flags.get("legacy_summary_fallback")
    )
    if not rendered_summary or compatibility_required:
        fallback_lines = runtime_utils.build_legacy_summary_lines(
            snapshot.values, emit_json_tail=bool(snapshot.flags.get("emit_json_tail"))
        )
        if not reporter.quiet:
            reporter.section("Summary")
            for line in fallback_lines:
                reporter.line(_color_text(line, "green"))


def _coerce_json_mapping(raw: Mapping[str, Any] | None) -> dict[str, JsonValue]:
    if raw is None:
        return {}
    return {str(key): _coerce_json_value(value) for key, value in raw.items()}


def _coerce_json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        enum_value = value.value
        if isinstance(enum_value, (str, int, float, bool)):
            return enum_value
        return str(enum_value)
    if isinstance(value, MappingABC):
        mapping_value = cast(Mapping[Any, Any], value)
        return {str(key): _coerce_json_value(inner) for key, inner in mapping_value.items()}
    if isinstance(value, (list, tuple, set)):
        sequence_value = cast(Iterable[Any], value)
        return [_coerce_json_value(item) for item in sequence_value]
    return str(value)


def _render_missing_section(
    reporter: CliOutputManagerProtocol,
    section: Mapping[str, Any],
    options: RenderOptions,
    note: str | None = None,
) -> None:
    label = str(section.get("title") or section.get("title_badge") or section.get("id") or "Section")
    reporter.section(label)
    message = note or f"{options.missing_label}; use {options.no_cache_hint}."
    reporter.line(_color_text(message, "yellow"))


def _render_partial_header(
    reporter: CliOutputManagerProtocol,
    section: Mapping[str, Any],
    options: RenderOptions,
) -> None:
    label = str(section.get("title") or section.get("title_badge") or section.get("id") or "Section")
    reporter.section(f"{label} {options.partial_label}")


def resolve_cli_version() -> str:
    """Best-effort CLI/package version lookup for snapshot metadata."""

    try:
        return importlib_metadata.version("frame-compare")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover - fallback
        return "unknown"
