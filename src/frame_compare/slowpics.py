"""Slow.pics upload orchestration."""

from __future__ import annotations

import logging
import queue
import re
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, cast
from urllib.parse import unquote, urlsplit

import requests
from requests.adapters import HTTPAdapter

from src.datatypes import SlowpicsConfig
from src.frame_compare.net import (
    ALLOWED_METHODS,
    DEFAULT_CONNECT_TIMEOUT,
    RETRY_STATUS,
    build_urllib3_retry,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from requests_toolbelt import MultipartEncoder as _MultipartEncoderType
else:  # pragma: no cover - optional dependency in tests
    _MultipartEncoderType = Any

try:
    from requests_toolbelt import MultipartEncoder as _RuntimeMultipartEncoder  # type: ignore[reportMissingImports]
except ImportError:
    _RuntimeMultipartEncoder = None

MultipartEncoder: Optional[type[_MultipartEncoderType]] = _RuntimeMultipartEncoder


class SlowpicsAPIError(RuntimeError):
    """Raised when slow.pics API interactions fail."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


logger = logging.getLogger(__name__)


_CONNECT_TIMEOUT_SECONDS = DEFAULT_CONNECT_TIMEOUT
_DEFAULT_UPLOAD_CONCURRENCY = 3
_MIN_UPLOAD_THROUGHPUT_BYTES_PER_SEC = 256 * 1024  # 256 KiB/s baseline assumption
_UPLOAD_TIMEOUT_MARGIN_SECONDS = 15.0


class _SessionPool:
    """Simple bounded pool that hands out preconfigured Requests sessions."""

    def __init__(self, session_factory: Callable[[], requests.Session], size: int) -> None:
        self._session_factory = session_factory
        self._queue: "queue.Queue[requests.Session]" = queue.Queue(maxsize=max(1, size))
        self._sessions: List[requests.Session] = []
        try:
            for _ in range(max(1, size)):
                session = session_factory()
                self._sessions.append(session)
                self._queue.put(session)
        except Exception:  # noqa: BLE001
            self.close()
            raise

    @contextmanager
    def acquire(self) -> Iterator[requests.Session]:
        """Check out a session for exclusive use within a worker thread."""

        session = self._queue.get()
        try:
            yield session
        finally:
            self._queue.put(session)

    def close(self) -> None:
        """Close every underlying session."""

        while self._sessions:
            session = self._sessions.pop()
            try:
                session.close()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to close slow.pics session cleanly", exc_info=True)


def _raise_for_status(response: requests.Response, context: str) -> None:
    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise SlowpicsAPIError(
            f"{context} failed ({response.status_code}): {detail}",
            status_code=response.status_code,
        )


def _redact_webhook(url: str) -> str:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "webhook"
    if parsed.netloc:
        return parsed.netloc
    return parsed.path or "webhook"


def _post_direct_webhook(session: requests.Session, webhook_url: str, canonical_url: str) -> None:
    redacted = _redact_webhook(webhook_url)
    payload = {"content": canonical_url}
    backoff = 1.0
    for attempt in range(1, 4):
        try:
            resp = session.post(webhook_url, json=payload, timeout=10)
            if resp.status_code < 300:
                logger.info("Posted slow.pics URL to webhook host %s", redacted)
                return
            message = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            message = exc.__class__.__name__
        logger.warning(
            "Webhook post attempt %s to %s failed: %s",
            attempt,
            redacted,
            message,
        )
        if attempt < 3:
            time.sleep(backoff)
            backoff = min(backoff * 2, 4.0)
    logger.error("Giving up on webhook delivery to %s after %s attempts", redacted, 3)


def _build_legacy_headers(session: requests.Session, encoder: Any) -> Dict[str, str]:
    xsrf = session.cookies.get_dict().get("XSRF-TOKEN")
    if not xsrf:
        raise SlowpicsAPIError("Missing XSRF token; cannot complete slow.pics upload")
    return {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
        "Access-Control-Allow-Origin": "*",
        "Content-Length": str(getattr(encoder, "len", 0)),
        "Content-Type": encoder.content_type,
        "Origin": "https://slow.pics/",
        "Referer": "https://slow.pics/comparison",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/113.0.0.0 Safari/537.36"
        ),
        "X-XSRF-TOKEN": unquote(xsrf),
    }


_TMDB_MANUAL_RE = re.compile(r"^(movie|tv)[/_:-]?(\d+)$", re.IGNORECASE)
_SHORTCUT_SANITIZE_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _format_tmdb_identifier(tmdb_id: str, category: str | None) -> str:
    """Normalize TMDB identifiers for slow.pics legacy form fields."""

    text = (tmdb_id or "").strip()
    if not text:
        return ""

    match = _TMDB_MANUAL_RE.match(text)
    if match:
        prefix, digits = match.groups()
        return f"{prefix.upper()}_{digits}"

    normalized_category = (category or "").strip().lower()
    if text.isdigit() and normalized_category in {"movie", "tv"}:
        return f"{normalized_category.upper()}_{text}"

    return text


def _sanitize_shortcut_component(value: str) -> str:
    """Return a filesystem-safe fragment for slow.pics shortcut filenames."""

    trimmed = value.strip()
    if not trimmed:
        return ""
    replaced = _SHORTCUT_SANITIZE_RE.sub("_", trimmed)
    deduped = re.sub(r"_+", "_", replaced)
    cleaned = deduped.strip("._-")
    if not cleaned:
        return ""
    # Keep filenames at a reasonable length while preserving extension space.
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rstrip("._-")
    return cleaned


def build_shortcut_filename(collection_name: str | None, canonical_url: str) -> str:
    """
    Resolve the `.url` filename used for slow.pics shortcuts.

    Preference order:
        1. Sanitised collection name.
        2. Sanitised comparison key from the canonical URL.
        3. Fallback literal ``"slowpics"``.
    """

    base = _sanitize_shortcut_component(collection_name or "")
    if not base:
        key = canonical_url.rstrip("/").rsplit("/", 1)[-1]
        base = _sanitize_shortcut_component(key)
    if not base:
        base = "slowpics"
    return f"{base}.url"


def _prepare_legacy_plan(image_files: List[str]) -> tuple[List[int], List[List[tuple[str, Path]]]]:
    groups: dict[int, List[tuple[str, Path]]] = defaultdict(list)
    for file_path in image_files:
        path = Path(file_path)
        if not path.is_file():
            raise SlowpicsAPIError(f"Image file not found: {file_path}")
        name = path.name
        if " - " not in name or not name.lower().endswith(".png"):
            raise SlowpicsAPIError(
                f"Screenshot '{name}' does not follow '<frame> - <label>.png' naming"
            )
        frame_part, label_part = name[:-4].split(" - ", 1)
        try:
            frame_idx = int(frame_part.strip())
        except ValueError as exc:
            raise SlowpicsAPIError(f"Unable to parse frame index from '{name}'") from exc
        label = label_part.strip() or "comparison"
        groups.setdefault(frame_idx, []).append((label, path))

    if not groups:
        raise SlowpicsAPIError("No screenshots available for slow.pics upload")

    frame_order = sorted(groups.keys())
    expected = len(groups[frame_order[0]])
    for frame, entries in groups.items():
        if len(entries) != expected:
            raise SlowpicsAPIError(
                "Inconsistent screenshot count for frame "
                f"{frame}; expected {expected}, found {len(entries)}"
            )
    ordered_groups = [groups[frame] for frame in frame_order]
    return frame_order, ordered_groups


def _compute_image_upload_timeout(cfg: SlowpicsConfig, size_bytes: int) -> tuple[float, float]:
    """Return (connect, read) timeouts derived from a throughput floor plus margin."""

    base = max(float(cfg.image_upload_timeout_seconds), 1.0)
    if size_bytes <= 0:
        return (_CONNECT_TIMEOUT_SECONDS, base)
    estimated = size_bytes / _MIN_UPLOAD_THROUGHPUT_BYTES_PER_SEC + _UPLOAD_TIMEOUT_MARGIN_SECONDS
    return (_CONNECT_TIMEOUT_SECONDS, max(base, estimated))


def _upload_comparison_legacy(
    session_factory: Callable[[], requests.Session],
    image_files: List[str],
    screen_dir: Path,
    cfg: SlowpicsConfig,
    *,
    progress_callback: Optional[Callable[[int], None]] = None,
    max_workers: Optional[int] = None,
) -> str:
    if MultipartEncoder is None:
        raise SlowpicsAPIError(
            "requests-toolbelt is required for slow.pics uploads. Install it to enable auto-upload."
        )
    encoder_cls = MultipartEncoder

    frame_order, grouped = _prepare_legacy_plan(image_files)
    browser_id = str(uuid.uuid4())

    fields: dict[str, str] = {
        "collectionName": cfg.collection_name or "Frame Comparison",
        "hentai": str(bool(cfg.is_hentai)).lower(),
        "optimize-images": "true",
        "browserId": browser_id,
        "public": str(bool(cfg.is_public)).lower(),
    }
    if cfg.tmdb_id:
        fields["tmdbId"] = _format_tmdb_identifier(cfg.tmdb_id, getattr(cfg, "tmdb_category", ""))
    if cfg.remove_after_days:
        fields["removeAfter"] = str(int(cfg.remove_after_days))

    upload_plan: List[List[Path]] = []
    for comp_index, frame in enumerate(frame_order):
        entries = grouped[comp_index]
        fields[f"comparisons[{comp_index}].name"] = str(frame)
        per_frame_paths: List[Path] = []
        for image_index, (label, path) in enumerate(entries):
            fields[f"comparisons[{comp_index}].imageNames[{image_index}]"] = label
            per_frame_paths.append(path)
        upload_plan.append(per_frame_paths)

    session = session_factory()
    assert encoder_cls is not None
    encoder = encoder_cls(fields, str(uuid.uuid4()))
    headers = _build_legacy_headers(session, encoder)
    response = session.post(
        "https://slow.pics/upload/comparison",
        data=encoder,
        headers=headers,
        timeout=(_CONNECT_TIMEOUT_SECONDS, 30.0),
    )
    _raise_for_status(response, "Legacy collection creation")
    try:
        comp_json = response.json()
    except ValueError as exc:
        raise SlowpicsAPIError("Invalid JSON response returned by slow.pics") from exc

    collection_uuid = comp_json.get("collectionUuid")
    key = comp_json.get("key")
    if not key:
        raise SlowpicsAPIError("Missing collection key in slow.pics response")
    canonical_url = f"https://slow.pics/c/{key}"
    images_raw = comp_json.get("images")
    if not isinstance(images_raw, list):
        raise SlowpicsAPIError("Slow.pics response missing image identifiers")
    images_raw_list = cast(List[object], images_raw)
    image_groups: List[List[str]] = []
    for group_index, image_ids_raw in enumerate(images_raw_list):
        if not isinstance(image_ids_raw, list):
            raise SlowpicsAPIError(f"Slow.pics response malformed for comparison {group_index}")
        normalized_ids: List[str] = []
        image_ids_list = cast(List[object], image_ids_raw)
        for image_id in image_ids_list:
            if not isinstance(image_id, str):
                raise SlowpicsAPIError("Slow.pics image identifiers must be strings")
            normalized_ids.append(image_id)
        image_groups.append(normalized_ids)
    if len(image_groups) != len(upload_plan):
        raise SlowpicsAPIError("Unexpected slow.pics response structure for comparisons")

    jobs: List[tuple[Path, str]] = []
    for per_frame_paths, image_ids in zip(upload_plan, image_groups, strict=False):
        if len(image_ids) != len(per_frame_paths):
            raise SlowpicsAPIError("Slow.pics returned mismatched image identifiers")
        jobs.extend(zip(per_frame_paths, image_ids, strict=False))

    worker_count = max_workers if max_workers is not None else _DEFAULT_UPLOAD_CONCURRENCY
    worker_count = max(1, min(worker_count, len(jobs) or 1))

    session_pool = _SessionPool(session_factory, worker_count)
    try:
        def _upload_single(path: Path, image_uuid: str) -> None:
            file_size = path.stat().st_size
            timeout = _compute_image_upload_timeout(cfg, file_size)
            with ExitStack() as stack:
                file_handle = stack.enter_context(path.open("rb"))
                upload_fields = {
                    "collectionUuid": collection_uuid,
                    "imageUuid": image_uuid,
                    "file": (path.name, file_handle, "image/png"),
                    "browserId": browser_id,
                }
                upload_encoder = encoder_cls(upload_fields, str(uuid.uuid4()))
                with session_pool.acquire() as local_session:
                    upload_headers = _build_legacy_headers(local_session, upload_encoder)
                    upload_resp = local_session.post(
                        "https://slow.pics/upload/image",
                        data=upload_encoder,
                        headers=upload_headers,
                        timeout=timeout,
                    )
            _raise_for_status(upload_resp, f"Upload frame {path.name}")
            if getattr(upload_resp, "content", b""):
                text = upload_resp.content.decode("utf-8", "ignore").strip()
                if text and text.upper() != "OK":
                    raise SlowpicsAPIError(f"Unexpected slow.pics response: {text}")
            if progress_callback is not None:
                progress_callback(1)

        if worker_count == 1:
            for path, image_uuid in jobs:
                _upload_single(path, image_uuid)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_upload_single, path, image_uuid) for path, image_uuid in jobs]
                try:
                    for future in as_completed(futures):
                        future.result()
                except Exception:  # noqa: BLE001
                    for future in futures:
                        future.cancel()
                    raise
    finally:
        session_pool.close()

    if cfg.webhook_url:
        _post_direct_webhook(session, cfg.webhook_url, canonical_url)
    if cfg.create_url_shortcut:
        shortcut_name = build_shortcut_filename(cfg.collection_name, canonical_url)
        shortcut_path = screen_dir / shortcut_name
        try:
            shortcut_path.write_text(f"[InternetShortcut]\nURL={canonical_url}\n", encoding="utf-8")
            logger.info("Saved slow.pics shortcut: %s", shortcut_path)
        except OSError as exc:
            logger.warning(
                "Failed to write slow.pics shortcut at %s: %s",
                shortcut_path,
                exc,
            )
    session.close()
    return canonical_url


def _configure_slowpics_session(session: requests.Session, *, workers: Optional[int] = None) -> None:
    """
    Configure the provided session with retry-capable HTTP adapters sized for concurrent uploads.

    Parameters:
        session: The Requests session that will perform slow.pics HTTP calls.
        workers: Expected parallel upload workers; defaults to `_DEFAULT_UPLOAD_CONCURRENCY`.
    """

    effective_workers = workers if workers and workers > 0 else _DEFAULT_UPLOAD_CONCURRENCY
    pool_size = max(4, effective_workers)
    retries = build_urllib3_retry(
        backoff_factor=0.1,
        status_forcelist=RETRY_STATUS,
        allowed_methods=ALLOWED_METHODS,
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    logger.info(
        "slow.pics session configured: pool=%d retries=%d backoff=%.1f",
        pool_size,
        retries.total or 0,
        retries.backoff_factor,
    )


def upload_comparison(
    image_files: List[str],
    screen_dir: Path,
    cfg: SlowpicsConfig,
    *,
    progress_callback: Optional[Callable[[int], None]] = None,
    max_workers: Optional[int] = None,
) -> str:
    """Upload screenshots to slow.pics and return the collection URL.

    Notes:
        When ``max_workers`` is greater than 1 (or left as ``None`` and defaults to
        parallel uploads), ``progress_callback`` may be invoked concurrently from
        multiple worker threads. Callers should ensure the callback is thread-safe
        (for example, by using ``threading.Lock`` or other synchronization
        primitives) if they mutate shared state or emit UI updates inside the
        callback. The CLI publishers use a dedicated lock-protected progress
        tracker so file counts/byte totals stay consistent while worker threads
        emit callbacks.
    """

    if not image_files:
        raise SlowpicsAPIError("No image files provided for upload")

    expected_workers = min(max_workers or _DEFAULT_UPLOAD_CONCURRENCY, len(image_files) or 1)
    bootstrap_session = requests.Session()
    try:
        _configure_slowpics_session(bootstrap_session, workers=expected_workers)
        try:
            bootstrap_session.get("https://slow.pics/comparison", timeout=_CONNECT_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise SlowpicsAPIError(f"Failed to establish slow.pics session: {exc}") from exc

        xsrf_token = bootstrap_session.cookies.get("XSRF-TOKEN")
        if not xsrf_token:
            raise SlowpicsAPIError("Missing XSRF token from slow.pics response")

        logger.info("Using slow.pics legacy upload endpoints")

        def _new_session() -> requests.Session:
            worker_session = requests.Session()
            worker_session.cookies.update(bootstrap_session.cookies.get_dict())
            _configure_slowpics_session(worker_session, workers=expected_workers)
            return worker_session

        url = _upload_comparison_legacy(
            _new_session,
            image_files,
            screen_dir,
            cfg,
            progress_callback=progress_callback,
            max_workers=max_workers,
        )
        logger.info("Slow.pics: %s", url)
        logger.info(
            "slow.pics upload complete: frames=%d workers=%d url=%s",
            len(image_files),
            expected_workers,
            url,
        )
        return url
    finally:
        bootstrap_session.close()
