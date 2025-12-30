"""Audio-based alignment helpers for pre-analysis trim adjustments."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
import subprocess
import tempfile
import tomllib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

from src.frame_compare import subproc as _subproc

logger = logging.getLogger(__name__)
FpsHint = float | tuple[int, int]
FpsHintMap = Mapping[Path, FpsHint]


def _to_int(value: object, default: int = 0) -> int:
    """Safely convert a JSON-derived value to an integer."""

    if value is None:
        return default
    if isinstance(value, bool):  # bool is subclass of int but handle explicitly
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _as_str_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return cast(dict[str, object], value)
    return {}

_FLUSH_TO_ZERO_WARNING = "The value of the smallest subnormal"


@contextmanager
def _suppress_flush_to_zero_warning() -> Iterator[None]:
    """Temporarily silence NumPy's flush-to-zero diagnostic warning."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=f"{_FLUSH_TO_ZERO_WARNING}.*",
            category=UserWarning,
            module="numpy._core.getlimits",
        )
        yield


class AudioAlignmentError(RuntimeError):
    """Raised when audio alignment cannot be completed."""


@dataclass
class AlignmentMeasurement:
    """Measurement details for a clip relative to the chosen reference."""

    file: Path
    offset_seconds: Optional[float]
    frames: Optional[int]
    correlation: float
    reference_fps: Optional[float]
    target_fps: Optional[float]
    error: Optional[str] = None

    @property
    def key(self) -> str:
        """Return the file stem used when indexing alignment results."""
        return self.file.name


@dataclass
class AudioStreamInfo:
    """Metadata describing a single audio stream from ffprobe."""

    index: int
    language: str
    codec_name: str
    channels: int
    channel_layout: str
    sample_rate: int
    bitrate: int
    is_default: bool
    is_forced: bool


def ensure_external_tools() -> None:
    """Make sure ffmpeg/ffprobe are discoverable."""

    missing = [tool for tool in ("ffmpeg", "ffprobe") if which(tool) is None]
    if missing:
        raise AudioAlignmentError(
            f"Required tool(s) missing from PATH: {', '.join(missing)}"
        )


_OPTIONAL_MODULES: Optional[Tuple[Any, Any, Any]] = None


def _load_optional_modules() -> Tuple[Any, Any, Any]:
    global _OPTIONAL_MODULES
    if _OPTIONAL_MODULES is not None:
        return _OPTIONAL_MODULES

    try:
        with _suppress_flush_to_zero_warning():
            import librosa  # type: ignore
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:  # pragma: no cover - optional dependency
        message = (
            "Audio alignment requires optional dependencies: numpy, librosa, soundfile."
            if isinstance(exc, ModuleNotFoundError)
            else f"Audio alignment failed to load optional dependencies: {exc}"
        )
        raise AudioAlignmentError(message) from exc
    _OPTIONAL_MODULES = (np, librosa, sf)
    return _OPTIONAL_MODULES


def probe_audio_streams(path: Path) -> List[AudioStreamInfo]:
    """Return metadata for all audio streams in *path*."""

    if which("ffprobe") is None:
        raise AudioAlignmentError("ffprobe not found in PATH")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index,codec_name,channels,channel_layout,sample_rate,bit_rate,disposition:stream_tags=language",
        "-of",
        "json",
        str(path),
    ]

    try:
        result = _subproc.run_checked(
            cmd,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise AudioAlignmentError(f"ffprobe failed for {path.name}") from exc

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AudioAlignmentError(f"Unable to parse ffprobe output for {path.name}") from exc

    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        return []
    payload_dict: dict[str, object] = cast(dict[str, object], payload)
    streams_data = payload_dict.get("streams", [])
    if not isinstance(streams_data, list):
        return []
    streams: List[AudioStreamInfo] = []
    for entry in streams_data or []:
        if not isinstance(entry, dict) or not all(isinstance(key, str) for key in entry):
            continue
        entry_dict: dict[str, object] = cast(dict[str, object], entry)
        index = _to_int(entry_dict.get("index"), default=-1)
        if index < 0:
            continue
        tags = _as_str_dict(entry_dict.get("tags"))
        disposition = _as_str_dict(entry_dict.get("disposition"))
        language = str(tags.get("language") or "").strip()
        codec_name = str(entry_dict.get("codec_name") or "").strip()
        channels = _to_int(entry_dict.get("channels"))
        channel_layout = str(entry_dict.get("channel_layout") or "").strip()
        sample_rate = _to_int(entry_dict.get("sample_rate"))
        bitrate = _to_int(entry_dict.get("bit_rate"))
        is_default = bool(_to_int(disposition.get("default")))
        is_forced = bool(_to_int(disposition.get("forced")))
        streams.append(
            AudioStreamInfo(
                index=index,
                language=language.lower(),
                codec_name=codec_name.lower(),
                channels=channels,
                channel_layout=channel_layout.lower(),
                sample_rate=sample_rate,
                bitrate=bitrate,
                is_default=is_default,
                is_forced=is_forced,
            )
        )
    streams.sort(key=lambda info: info.index)
    return streams


def _extract_audio(
    infile: Path,
    *,
    sample_rate: int,
    start_seconds: Optional[float],
    duration_seconds: Optional[float],
    stream_index: int,
) -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    path = Path(handle.name)
    handle.close()
    cmd: List[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds}"]
    cmd += ["-i", str(infile)]
    if duration_seconds is not None:
        cmd += ["-t", f"{duration_seconds}"]
    cmd += [
        "-map",
        f"0:{stream_index}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        "-y",
        str(path),
    ]
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = _subproc.run_checked(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - data dependent
        try:
            os.unlink(path)
        except OSError:
            pass
        stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        detail = f": {stderr}" if stderr else ""
        raise AudioAlignmentError(
            f"ffmpeg failed to extract audio from {infile.name}{detail}"
        ) from exc
    finally:
        # Ensure ffmpeg output doesn't spam stdout when successful
        if completed is not None and completed.stdout:
            logger.debug("ffmpeg audio extract stdout: %s", completed.stdout.strip())
    return path


@contextmanager
def _temporary_audio(
    infile: Path,
    *,
    sample_rate: int,
    start_seconds: Optional[float],
    duration_seconds: Optional[float],
    stream_index: int,
) -> Iterator[Path]:
    path = _extract_audio(
        infile,
        sample_rate=sample_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        stream_index=stream_index,
    )
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _onset_envelope(
    wav_path: Path,
    *,
    sample_rate: int,
    hop_length: int,
) -> Tuple[Any, int]:
    np, librosa, sf = _load_optional_modules()

    try:
        with _suppress_flush_to_zero_warning():
            data, native_sr = sf.read(str(wav_path))
            if data.size == 0:
                raise AudioAlignmentError(f"No audio samples extracted from {wav_path}")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if native_sr != sample_rate:
                data = librosa.resample(data, orig_sr=native_sr, target_sr=sample_rate)

            peak = float(np.max(np.abs(data))) if data.size else 0.0
            if peak > 0:
                data = data / peak

            onset_env = librosa.onset.onset_strength(
                y=data,
                sr=sample_rate,
                hop_length=hop_length,
                center=True,
            )
    except RuntimeError as exc:  # pragma: no cover - optional dependency runtime
        message = (
            "Audio alignment failed during onset envelope calculation because an optional "
            f"dependency raised an error: {exc}. Install numpy, librosa, and soundfile "
            "(and their dependencies)."
        )
        raise AudioAlignmentError(message) from exc
    return onset_env.astype(np.float32), hop_length


def _cross_correlation(a, b) -> Tuple[int, float]:
    np, _, _ = _load_optional_modules()
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        raise AudioAlignmentError("Empty onset envelope encountered during correlation")
    a = (a - np.mean(a)) / (np.std(a) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)
    corr = np.correlate(b, a, mode="full")
    idx = int(np.argmax(corr)) - (len(a) - 1)
    return idx, float(np.max(corr))


def _probe_fps(infile: Path) -> Optional[float]:
    try:
        result = _subproc.run_checked(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(infile),
            ],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    text = (result.stdout or "").strip()
    if not text:
        return None
    if "/" in text:
        num_str, den_str = text.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
            if den == 0:
                return None
            return num / den
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_fps_hint(value: FpsHint | None) -> Optional[float]:
    """Convert a cached FPS hint into a usable float."""

    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 2:
            return None
        numerator, denominator = value
        if not denominator:
            return None
        return float(numerator) / float(denominator)
    try:
        fps_value = float(value)
    except (TypeError, ValueError):
        return None
    if fps_value <= 0 or math.isnan(fps_value):
        return None
    return fps_value


def measure_offsets(
    reference: Path,
    targets: Sequence[Path],
    *,
    sample_rate: int,
    hop_length: int,
    start_seconds: Optional[float],
    duration_seconds: Optional[float],
    reference_stream: int = 0,
    target_streams: Mapping[Path, int] | None = None,
    window_overrides: Mapping[Path, Tuple[Optional[float], Optional[float]]] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    fps_hints: FpsHintMap | None = None,
) -> List[AlignmentMeasurement]:
    """
    Estimate relative audio offsets for *targets* against *reference*.

    Cached FPS hints take precedence whenever provided; otherwise `_probe_fps()` is used.
    """
    ensure_external_tools()

    def _resolve_fps(path: Path) -> Optional[float]:
        hint = _normalize_fps_hint(fps_hints.get(path)) if fps_hints is not None else None
        if hint is not None:
            return hint
        return _probe_fps(path)

    ref_env: Optional[Any] = None
    hop: int
    with _temporary_audio(
        reference,
        sample_rate=sample_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        stream_index=reference_stream,
    ) as ref_audio:
        ref_env, hop = _onset_envelope(ref_audio, sample_rate=sample_rate, hop_length=hop_length)

    if ref_env is None:
        raise AudioAlignmentError(f"Failed to compute onset envelope for {reference.name}")

    results: List[AlignmentMeasurement] = []
    reference_fps = _resolve_fps(reference)

    for target in targets:
        target_fps = _resolve_fps(target)
        try:
            stream_idx = 0
            if target_streams is not None:
                stream_idx = int(target_streams.get(target, 0))
            win_start = start_seconds
            win_dur = duration_seconds
            if window_overrides is not None and target in window_overrides:
                override_start, override_dur = window_overrides[target]
                if override_start is not None:
                    win_start = override_start
                if override_dur is not None:
                    win_dur = override_dur
            with _temporary_audio(
                target,
                sample_rate=sample_rate,
                start_seconds=win_start,
                duration_seconds=win_dur,
                stream_index=stream_idx,
            ) as target_audio:
                target_env, _ = _onset_envelope(
                    target_audio,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                )
            lag_frames, strength = _cross_correlation(ref_env, target_env)
            seconds_per_onset = hop_length / float(sample_rate)
            offset_seconds = lag_frames * seconds_per_onset

            frames = None
            if target_fps and target_fps > 0:
                frames = int(round(offset_seconds * target_fps))

            results.append(
                AlignmentMeasurement(
                    file=target,
                    offset_seconds=offset_seconds,
                    frames=frames,
                    correlation=strength,
                    reference_fps=reference_fps,
                    target_fps=target_fps,
                )
            )
        except AudioAlignmentError as exc:
            logger.warning("Audio alignment failed for %s: %s", target.name, exc)
            results.append(
                AlignmentMeasurement(
                    file=target,
                    offset_seconds=0.0,
                    frames=None,
                    correlation=0.0,
                    reference_fps=reference_fps,
                    target_fps=target_fps,
                    error=str(exc),
                )
            )
        if progress_callback is not None:
            try:
                progress_callback(1)
            except Exception:  # noqa: BLE001
                pass

    return results


def load_offsets(path: Path) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
    """Load previously recorded alignment offsets from *path* if available."""
    if not path.exists():
        return None, {}
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise AudioAlignmentError(f"Failed to read {path}: {exc}") from exc
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    offsets = data.get("offsets", {}) if isinstance(data, dict) else {}
    cleaned: Dict[str, Dict[str, Any]] = {}
    if isinstance(offsets, dict):
        offsets_dict: dict[str, object] = cast(dict[str, object], offsets)
        for key, value in offsets_dict.items():
            if isinstance(value, dict):
                cleaned[key] = dict(value)
    reference_name = None
    if isinstance(meta, dict):
        meta_dict: dict[str, object] = cast(dict[str, object], meta)
        ref = meta_dict.get("reference")
        if isinstance(ref, str):
            reference_name = ref
    return reference_name, cleaned


def _toml_quote(value: str) -> str:
    """Escape TOML string characters for safe inline inclusion."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_float(value: Optional[float]) -> str:
    """Format floats with fixed precision while preserving NaN markers."""
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def update_offsets_file(
    path: Path,
    reference_name: str,
    measurements: Sequence[AlignmentMeasurement],
    existing: Mapping[str, Dict[str, Any]] | None = None,
    negative_override_notes: Mapping[str, str] | None = None,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Write updated offset measurements to *path* and return applied adjustments."""
    applied: Dict[str, int] = {}
    statuses: Dict[str, str] = {}
    existing_map: Mapping[str, Dict[str, Any]] = existing if existing is not None else {}
    notes_map: Mapping[str, str] = (
        negative_override_notes if negative_override_notes is not None else {}
    )

    lines: List[str] = []
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    lines.append("[meta]")
    lines.append(f'reference = "{_toml_quote(reference_name)}"')
    lines.append(f'generated_at = "{_toml_quote(timestamp)}"')
    lines.append("")

    lines.append("[offsets]")

    for measurement in sorted(measurements, key=lambda m: m.file.name.lower()):
        key = measurement.key
        prior = existing_map.get(key)
        prior_frames = None
        prior_status = ""
        if isinstance(prior, Mapping):
            frames_obj = prior.get("frames")
            if isinstance(frames_obj, (int, float)):
                prior_frames = frames_obj
            status_obj = prior.get("status")
            if isinstance(status_obj, str):
                prior_status = status_obj.lower()
        manual = prior_status == "manual"
        if not manual and isinstance(prior_frames, (int, float)) and isinstance(prior, Mapping):
            suggested = prior.get("suggested_frames")
            if isinstance(suggested, (int, float)) and int(prior_frames) != int(suggested):
                manual = True
        frames = prior_frames if manual else measurement.frames

        if frames is not None:
            applied[key] = int(frames)
        status = "manual" if manual else "auto"
        statuses[key] = status

        frames_line = "nan"
        if frames is not None:
            frames_line = str(int(frames))

        suggested_frames = measurement.frames
        suggested_seconds = measurement.offset_seconds
        seconds = None
        if frames is not None and measurement.target_fps and measurement.target_fps > 0:
            seconds = frames / measurement.target_fps

        block: List[str] = [f'[offsets."{_toml_quote(key)}"]']
        block.append(f"frames = {frames_line}")
        block.append(f"seconds = {_format_float(seconds)}")
        if suggested_frames is None:
            block.append("suggested_frames = nan")
        else:
            block.append(f"suggested_frames = {int(suggested_frames)}")
        block.append(f"suggested_seconds = {_format_float(suggested_seconds)}")
        block.append(f"correlation = {_format_float(measurement.correlation)}")
        block.append(
            f"target_fps = {_format_float(measurement.target_fps)}"
        )
        block.append(f"status = \"{status}\"")
        if measurement.error:
            block.append(f'error = "{_toml_quote(measurement.error)}"')
        else:
            note = notes_map.get(key)
            if isinstance(note, str):
                block.append(f'note = "{_toml_quote(note)}"')
        block.append("")
        lines.extend(block)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return applied, statuses
