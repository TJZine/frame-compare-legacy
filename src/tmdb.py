"""Helpers for resolving TMDB metadata from filenames or external IDs."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import httpx

from .datatypes import TMDBConfig
from .frame_compare import net

logger = logging.getLogger(__name__)

MOVIE = "MOVIE"
TV = "TV"
_BASE_URL = "https://api.themoviedb.org/3"
_SIMILARITY_THRESHOLD = 0.45
_STRONG_MATCH_THRESHOLD = 0.92
_AMBIGUITY_MARGIN = 0.08

_NON_WORD_RE = re.compile(r"[^0-9a-zA-Z\s]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_ROMAN_RE = re.compile(r"\b([IVXLCDM]+)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_IMDB_RE = re.compile(r"tt\d{7,9}", re.IGNORECASE)


class TMDBResolutionError(RuntimeError):
    """Raised when TMDB matching cannot complete."""


class TMDBAmbiguityError(TMDBResolutionError):
    """Raised when multiple TMDB results look equally plausible."""

    def __init__(self, candidates: Sequence["TMDBCandidate"]) -> None:
        self.candidates = list(candidates)
        pretty = ", ".join(
            (
                f"{cand.category.lower()}/{cand.tmdb_id} "
                f"({cand.title} – {cand.year or '????'} score={cand.score:0.3f})"
            )
            for cand in self.candidates
        )
        super().__init__(f"Ambiguous TMDB match candidates: {pretty}")


@dataclass
class TMDBCandidate:
    """Represents a TMDB search or lookup hit."""

    category: str
    tmdb_id: str
    title: str
    original_title: Optional[str]
    year: Optional[int]
    score: float
    original_language: Optional[str]
    reason: str
    used_filename_search: bool
    payload: Dict[str, Any]


@dataclass
class TMDBResolution:
    """Final TMDB resolution result."""

    candidate: TMDBCandidate
    margin: float
    source_query: str

    def __iter__(self):  # pragma: no cover - convenience for tuple unpacking
        yield self.candidate.category
        yield self.candidate.tmdb_id
        yield self.candidate.original_language
        yield self.candidate.used_filename_search

    @property
    def category(self) -> str:
        return self.candidate.category

    @property
    def tmdb_id(self) -> str:
        return self.candidate.tmdb_id

    @property
    def title(self) -> str:
        return self.candidate.title

    @property
    def original_title(self) -> Optional[str]:
        return self.candidate.original_title

    @property
    def year(self) -> Optional[int]:
        return self.candidate.year

    @property
    def original_language(self) -> Optional[str]:
        return self.candidate.original_language


class _TTLCache:
    """Bounded TTL cache shared across TMDB requests."""

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max(0, int(max_entries))
        self._data: "OrderedDict[Tuple[Any, ...], Tuple[float, int, Any]]" = OrderedDict()

    def configure(self, *, max_entries: int | None = None) -> None:
        if max_entries is not None:
            self._max_entries = max(0, int(max_entries))
            if self._max_entries == 0:
                self._data.clear()
            else:
                while len(self._data) > self._max_entries:
                    self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def get(self, key: Tuple[Any, ...], ttl_seconds: int) -> Any | None:
        if ttl_seconds <= 0:
            self._data.pop(key, None)
            return None
        entry = self._data.get(key)
        if not entry:
            return None
        timestamp, stored_ttl, value = entry
        ttl = min(stored_ttl, ttl_seconds)
        if ttl <= 0:
            self._data.pop(key, None)
            return None
        if time.monotonic() - timestamp > ttl:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: Tuple[Any, ...], value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0 or self._max_entries == 0:
            self._data.pop(key, None)
            return
        now = time.monotonic()
        self._data[key] = (now, int(ttl_seconds), value)
        self._data.move_to_end(key)
        while len(self._data) > self._max_entries:
            self._data.popitem(last=False)


_CACHE = _TTLCache()


def parse_manual_id(value: str) -> Tuple[str, str]:
    """Validate a manual TMDB identifier (movie/12345 or tv/67890)."""

    text = value.strip().lower()
    if not text:
        raise TMDBResolutionError("Manual TMDB identifier cannot be empty")
    if text.isdigit():
        raise TMDBResolutionError(
            "Manual TMDB identifier must include a category prefix (movie/ or tv/)."
        )
    match = re.match(r"^(movie|tv)[/:-]?(\d+)$", text)
    if not match:
        raise TMDBResolutionError(
            "Manual TMDB identifier must look like movie/12345 or tv/67890"
        )
    category = MOVIE if match.group(1) == "movie" else TV
    tmdb_id = match.group(2)
    return category, tmdb_id


def _normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title or "")
    text = text.replace("&", " and ")
    text = _NON_WORD_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip().lower()


def _normalized_variants(title: str) -> List[str]:
    base = _normalize_title(title)
    variants = {base}
    if base.startswith("the "):
        variants.add(base[4:])

    # Handle stylistic spelling variations where "vv" is used in place of "w" (e.g.
    # "The VVitch" -> "The Witch") and vice-versa. This helps align aliases with
    # filenames even when the main TMDB title omits the variant.
    if "vv" in base:
        variants.add(base.replace("vv", "w"))
    if "w" in base:
        variants.add(base.replace("w", "vv"))

    return [variant for variant in variants if variant]


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if not shorter:
        return 0.0
    matches = 0
    j = 0
    for ch in shorter:
        idx = longer.find(ch, j)
        if idx == -1:
            continue
        matches += 1
        j = idx + 1
    return matches / max(1, len(longer))


def _roman_to_int(text: str) -> Optional[int]:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    result = 0
    prev = 0
    for char in text.upper():
        value = values.get(char)
        if value is None:
            return None
        if value > prev:
            result += value - 2 * prev
        else:
            result += value
        prev = value
    return result if result > 0 else None


def _convert_roman_suffix(title: str) -> Optional[str]:
    match = _ROMAN_RE.search(title)
    if not match:
        return None
    numeral = match.group(1)
    converted = _roman_to_int(numeral)
    if not converted or converted <= 1:
        return None
    return f"{title[:match.start(1)]}{converted}{title[match.end(1):]}"


def _primary_title(title: str) -> Optional[str]:
    for delimiter in (":", " - ", " (", " – "):
        if delimiter in title:
            candidate = title.split(delimiter, 1)[0].strip()
            if candidate and candidate != title:
                return candidate
    return None


def _reduced_words(title: str) -> Optional[str]:
    words = re.findall(r"[A-Za-z0-9']+", title)
    filtered = [word for word in words if len(word) > 2]
    if filtered and filtered != words:
        return " ".join(filtered)
    return None


def _strip_filename_noise(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    stem = re.sub(r"^\[[^]]+\]", "", stem)
    stem = re.sub(r"[\[\]{}()]", " ", stem)
    stem = stem.replace("_", " ")
    stem = stem.replace(".", " ")
    stem = re.sub(
        r"\b(480p|576p|720p|1080p|2160p|4320p|x264|h264|hevc|x265|av1|hdr|sdr|remux|webrip|web-dl|bluray|blu-ray|dvdrip|multi|10bit|proper|repack|extended|uhd|nf|amzn)\b",
        "",
        stem,
        flags=re.IGNORECASE,
    )
    stem = _WHITESPACE_RE.sub(" ", stem)
    return stem.strip()


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_year_from_date(date_text: str) -> Optional[int]:
    if not date_text:
        return None
    match = _YEAR_RE.search(date_text)
    if match:
        return int(match.group(0))
    return None


def _extract_year_from_result(payload: Dict[str, Any], category: str) -> Optional[int]:
    if category == MOVIE:
        return _extract_year_from_date(payload.get("release_date", ""))
    return _extract_year_from_date(payload.get("first_air_date", ""))


def _call_guessit(filename: str) -> Dict[str, Any]:
    try:
        module = import_module("guessit")
    except ImportError:
        return {}
    parser = getattr(module, "guessit", None)
    if not callable(parser):
        return {}
    try:
        result = parser(filename)
    except (ValueError, TypeError, IndexError, AttributeError, LookupError) as exc:
        logger.debug("GuessIt failed for %s: %s", filename, exc)
        return {}
    if isinstance(result, dict):
        return dict(result)
    return {}


def _call_anitopy(filename: str) -> Dict[str, Any]:
    try:
        module = import_module("anitopy")
    except ImportError:
        return {}
    parser = getattr(module, "parse", None)
    if not callable(parser):
        return {}
    try:
        result = parser(filename)
    except (ValueError, TypeError, IndexError, AttributeError, LookupError) as exc:
        logger.debug("Anitopy failed for %s: %s", filename, exc)
        return {}
    if isinstance(result, dict):
        return dict(result)
    return {}


@dataclass(frozen=True)
class _QueryPlan:
    query: str
    year: Optional[int]
    category: str
    reason: str


def _expand_title_variants(title: str) -> List[str]:
    """Return additional search titles derived from *title*."""

    variants: List[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append(normalized)

    add(title)

    simplified = _WHITESPACE_RE.sub(" ", title).strip()
    if simplified and simplified != title:
        add(simplified)

    # Allow queries that drop a trailing year segment when filenames embed it.
    without_year = re.sub(r"\b(19|20)\d{2}\b", "", simplified).strip()
    if without_year and without_year != simplified:
        add(_WHITESPACE_RE.sub(" ", without_year).strip())

    no_paren = re.sub(r"\([^)]*\)", "", simplified).strip()
    if no_paren and no_paren != simplified:
        add(no_paren)

    for delimiter in (":", " - ", " – ", " — "):
        if delimiter in simplified:
            head, tail = simplified.split(delimiter, 1)
            add(head)
            add(tail)

    if re.search(r"(?i)\baka\b", simplified):
        parts = re.split(r"(?i)\baka\b", simplified)
        for part in parts:
            add(part)

    if re.search(r"(?i)vvitch", simplified):
        add(re.sub(r"(?i)vvitch", "witch", simplified))

    return variants


def _build_query_plans(
    base_title: str,
    *,
    year: Optional[int],
    category_hint: Optional[str],
    category_preference: Optional[str],
    anime_titles: Sequence[str],
    year_tolerance: int,
) -> List[_QueryPlan]:
    queries: List[_QueryPlan] = []
    seen: set[Tuple[str, Optional[int], str]] = set()

    def add(query: str, year_val: Optional[int], category: str, reason: str) -> None:
        key = (query.lower(), year_val, category)
        if not query.strip():
            return
        if key in seen:
            return
        seen.add(key)
        queries.append(
            _QueryPlan(
                query=query.strip(),
                year=year_val,
                category=category,
                reason=reason,
            )
        )

    categories: List[str] = []
    for preferred in (category_preference, category_hint, MOVIE, TV):
        if preferred and preferred not in categories:
            categories.append(preferred)

    all_titles: List[str] = []
    for raw in [base_title, *anime_titles]:
        for expanded in _expand_title_variants(raw):
            if expanded not in all_titles:
                all_titles.append(expanded)

    for category in categories:
        for title in all_titles:
            add(title, year, category, "primary-title")
            roman = _convert_roman_suffix(title)
            if roman and roman != title:
                add(roman, year, category, "roman-numeral")
            primary = _primary_title(title)
            if primary and primary != title:
                add(primary, year, category, "secondary-title")
            reduced = _reduced_words(title)
            if reduced and reduced != title:
                add(reduced, year, category, "reduced-words")

    if year is not None and year_tolerance > 0:
        deltas = {delta for delta in range(-year_tolerance, year_tolerance + 1) if delta}
        for plan in list(queries):
            for delta in sorted(deltas):
                add(plan.query, year + delta, plan.category, f"year-adjust {delta:+d}")

    return queries


async def _http_request(
    client: httpx.AsyncClient,
    *,
    cache_ttl: int,
    path: str,
    params: Dict[str, Any],
    timeout: float | httpx.Timeout | None = None,
) -> Dict[str, Any]:
    key = (path, tuple(sorted(params.items())))
    cached = _CACHE.get(key, cache_ttl)
    if cached is not None:
        return cached

    redacted_host = net.redact_url_for_logs(str(getattr(client, "base_url", "")) or path)

    async def _on_backoff(delay: float, attempt_index: int) -> None:
        net.log_backoff_attempt(redacted_host, attempt_index, delay)

    effective_timeout = timeout if timeout is not None else net.DEFAULT_HTTP_TIMEOUT

    try:
        response = await net.httpx_get_json_with_backoff(
            client,
            path,
            params,
            retries=3,
            initial_backoff=0.5,
            max_backoff=4.0,
            sleep=asyncio.sleep,
            on_backoff=_on_backoff,
            timeout=effective_timeout,
        )
    except httpx.RequestError as exc:
        raise TMDBResolutionError(f"TMDB request failed after retries: {exc}") from exc
    except net.BackoffError as exc:
        raise TMDBResolutionError("TMDB request failed after retries") from exc

    status = response.status_code
    if status >= 400:
        raise TMDBResolutionError(
            f"TMDB request failed for {path} (status={status}): {response.text[:200]}"
        )
    try:
        payload_obj = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise TMDBResolutionError("TMDB returned invalid JSON") from exc
    payload = _ensure_dict(payload_obj, context=f"{path} response")
    _CACHE.set(key, payload, ttl_seconds=cache_ttl)
    return payload


def _ensure_dict(value: object, *, context: str) -> Dict[str, Any]:
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return cast(Dict[str, Any], value)
    raise TMDBResolutionError(f"{context} was not a JSON object")


def _dict_entries(value: object) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    entries: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            entries.append(cast(Dict[str, Any], item))
    return entries


def _title_candidates(payload: Dict[str, Any]) -> List[str]:
    keys = ["title", "name", "original_title", "original_name"]
    values: List[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    seen: set[str] = set()
    unique: List[str] = []
    for title in values:
        normalized = title.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(title)
    return unique


def _score_payload(
    payload: Dict[str, Any],
    *,
    category: str,
    query_norms: Sequence[str],
    year: Optional[int],
    tolerance: int,
    index: int,
) -> float:
    titles = _title_candidates(payload)
    title_norms = [_normalize_title(title) for title in titles]
    best = 0.0
    for query_norm in query_norms:
        for title_norm in title_norms:
            sim = _similarity(query_norm, title_norm)
            if title_norm == query_norm and sim < 1.0:
                sim = max(sim, 0.99)
            best = max(best, sim)
    if not titles:
        best *= 0.8

    if category == TV and index == 0:
        best += 0.01

    candidate_year = _extract_year_from_result(payload, category)
    if year is not None:
        if candidate_year is None:
            best -= 0.05
        else:
            diff = abs(candidate_year - year)
            if diff <= tolerance:
                best += 0.05 * (tolerance - diff + 1)
            else:
                best -= min(0.4 + (0.05 * (diff - tolerance)), 0.7)
    popularity = payload.get("popularity")
    popularity_value = 0.0
    if popularity is not None:
        try:
            popularity_value = float(popularity)
        except (TypeError, ValueError):
            popularity_value = 0.0
    if popularity_value > 0:
        best += min(popularity_value / 200.0, 0.05)

    return best


async def _fetch_alias_titles(
    client: httpx.AsyncClient,
    *,
    category: str,
    tmdb_id: str,
    cache_ttl: int,
    timeout: float | httpx.Timeout | None = None,
) -> List[str]:
    """Return alternative titles for a TMDB movie or TV entry."""

    if not tmdb_id:
        return []

    if category == MOVIE:
        path = f"movie/{tmdb_id}/alternative_titles"
        key = "titles"
    else:
        path = f"tv/{tmdb_id}/alternative_titles"
        key = "results"

    payload = await _http_request(
        client,
        cache_ttl=cache_ttl,
        path=path,
        params={},
        timeout=timeout,
    )
    titles: List[str] = []
    for entry in _dict_entries(payload.get(key)):
        title = entry.get("title")
        if isinstance(title, str):
            stripped = title.strip()
            if stripped:
                titles.append(stripped)
    return titles


def _alias_similarity(
    query_norms: Sequence[str],
    aliases: Iterable[str],
    existing_norms: Iterable[str] | None = None,
) -> float:
    alias_norms: List[str] = []
    seen: set[str] = set()
    excluded = {norm for norm in (existing_norms or []) if norm}
    for alias in aliases:
        for alias_norm in _normalized_variants(alias):
            if not alias_norm or alias_norm in seen or alias_norm in excluded:
                continue
            seen.add(alias_norm)
            alias_norms.append(alias_norm)

    best = 0.0
    for query_norm in query_norms:
        if not query_norm:
            continue
        query_tokens = query_norm.split()
        for alias_norm in alias_norms:
            alias_tokens = alias_norm.split()
            if (
                len(query_tokens) >= 2
                and set(query_tokens).issubset(alias_tokens)
                and len(alias_tokens) > len(query_tokens)
            ):
                best = max(best, 0.9)
            if alias_norm.startswith(query_norm) and len(query_norm) >= 4:
                best = max(best, 0.9)
            if query_norm.startswith(alias_norm) and len(alias_norm) >= 4:
                best = max(best, 0.9)
            best = max(best, _similarity(query_norm, alias_norm))
    return best


def _best_external_candidate(
    payload: Dict[str, Any],
    *,
    category_preference: Optional[str],
    category_hint: Optional[str],
    query_norms: Sequence[str],
    year: Optional[int],
    tolerance: int,
) -> Optional[TMDBCandidate]:
    candidates: List[TMDBCandidate] = []
    for category, key in ((MOVIE, "movie_results"), (TV, "tv_results")):
        for item in _dict_entries(payload.get(key)):
            score = _score_payload(
                item,
                category=category,
                query_norms=query_norms,
                year=year,
                tolerance=tolerance,
                index=0,
            )
            candidate = TMDBCandidate(
                category=category,
                tmdb_id=str(item.get("id")),
                title=item.get("title") or item.get("name") or "",
                original_title=item.get("original_title") or item.get("original_name"),
                year=_extract_year_from_result(item, category),
                score=score,
                original_language=item.get("original_language"),
                reason="external-id",
                used_filename_search=False,
                payload=item,
            )
            candidates.append(candidate)

    if not candidates:
        return None

    def _preferred(category: str) -> int:
        if category_preference == category:
            return 0
        if category_hint == category:
            return 1
        return 2

    candidates.sort(key=lambda cand: (_preferred(cand.category), -cand.score))
    return candidates[0]


async def _perform_search(
    client: httpx.AsyncClient,
    *,
    plan: _QueryPlan,
    cache_ttl: int,
    timeout: float | httpx.Timeout | None = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "query": plan.query,
        "include_adult": "false",
    }
    if plan.year is not None:
        if plan.category == MOVIE:
            params["year"] = plan.year
        else:
            params["first_air_date_year"] = plan.year
    payload = await _http_request(
        client,
        cache_ttl=cache_ttl,
        path="search/movie" if plan.category == MOVIE else "search/tv",
        params=params,
        timeout=timeout,
    )
    return _dict_entries(payload.get("results"))


def _extract_best_candidate(
    *,
    results: List[Dict[str, Any]],
    plan: _QueryPlan,
    query_norms: Sequence[str],
    tolerance: int,
    year: Optional[int],
) -> List[TMDBCandidate]:
    candidates: List[TMDBCandidate] = []
    for index, item in enumerate(results):
        score = _score_payload(
            item,
            category=plan.category,
            query_norms=query_norms,
            year=plan.year if plan.reason.startswith("year-adjust") else year,
            tolerance=tolerance,
            index=index,
        )
        candidate = TMDBCandidate(
            category=plan.category,
            tmdb_id=str(item.get("id")),
            title=item.get("title") or item.get("name") or "",
            original_title=item.get("original_title") or item.get("original_name"),
            year=_extract_year_from_result(item, plan.category),
            score=score,
            original_language=item.get("original_language"),
            reason=plan.reason,
            used_filename_search=True,
            payload=item,
        )
        candidates.append(candidate)
    return candidates


def _extract_title_year(
    filename: str,
    *,
    enable_anime: bool,
) -> Tuple[str, Optional[int], Optional[str], List[str]]:
    guess = _call_guessit(filename)
    title = guess.get("title") if isinstance(guess, dict) else ""
    title_str = str(title or "").strip()
    year = guess.get("year") if isinstance(guess, dict) else None
    year_int = _safe_int(year)

    type_hint_raw = str(guess.get("type") or "") if isinstance(guess, dict) else ""
    category_hint: Optional[str] = None
    if type_hint_raw.lower() in {"episode", "season", "tv", "series"}:
        category_hint = TV
    elif type_hint_raw.lower() == "movie":
        category_hint = MOVIE

    anime_titles: List[str] = []
    if enable_anime:
        ani = _call_anitopy(filename)
        ani_title = ani.get("anime_title") or ani.get("title") if isinstance(ani, dict) else None
        if ani_title:
            ani_title_str = str(ani_title).strip()
            if ani_title_str and ani_title_str.lower() != title_str.lower():
                anime_titles.append(ani_title_str)
        romaji = ani.get("anime_title_romaji") if isinstance(ani, dict) else None
        if romaji:
            romaji_str = str(romaji).strip()
            if romaji_str and romaji_str.lower() not in {t.lower() for t in anime_titles}:
                anime_titles.append(romaji_str)

    cleaned = title_str or _strip_filename_noise(filename)
    if not cleaned:
        cleaned = filename
    return cleaned, year_int, category_hint, anime_titles


def _detect_external_ids(filename: str) -> Tuple[Optional[str], Optional[str]]:
    imdb = None
    match = _IMDB_RE.search(filename)
    if match:
        imdb = match.group(0)
    return imdb, None


async def resolve_tmdb(
    filename: str,
    *,
    config: TMDBConfig,
    year: Optional[int] = None,
    imdb_id: Optional[str] = None,
    tvdb_id: Optional[str] = None,
    unattended: Optional[bool] = None,
    category_preference: Optional[str] = None,
    http_transport: httpx.BaseTransport | None = None,
) -> Optional[TMDBResolution]:
    """Resolve TMDB metadata for *filename* using the provided *config*."""

    if not config.api_key:
        raise TMDBResolutionError("tmdb.api_key must be set to resolve TMDB metadata")

    _CACHE.configure(max_entries=config.cache_max_entries)

    unattended_mode = config.unattended if unattended is None else unattended
    category_pref = (category_preference or config.category_preference or "").upper() or None
    year_tolerance = max(0, int(config.year_tolerance))

    cleaned_title, parsed_year, category_hint, anime_titles = _extract_title_year(
        filename,
        enable_anime=config.enable_anime_parsing,
    )
    if year is None and parsed_year is not None:
        year = parsed_year

    imdb_auto, tvdb_auto = _detect_external_ids(filename)
    imdb_lookup = imdb_id or imdb_auto
    tvdb_lookup = tvdb_id or tvdb_auto

    query_norms = _normalized_variants(cleaned_title)

    timeout = httpx.Timeout(
        net.DEFAULT_READ_TIMEOUT,
        connect=net.DEFAULT_CONNECT_TIMEOUT,
        read=net.DEFAULT_READ_TIMEOUT,
        write=net.DEFAULT_CONNECT_TIMEOUT,
    )
    params = {"api_key": config.api_key}
    async with httpx.AsyncClient(
        base_url=_BASE_URL,
        timeout=timeout,
        params=params,
        transport=http_transport,
    ) as client:
        if imdb_lookup:
            payload = await _http_request(
                client,
                cache_ttl=config.cache_ttl_seconds,
                path=f"find/{imdb_lookup}",
                params={"external_source": "imdb_id"},
                timeout=timeout,
            )
            candidate = _best_external_candidate(
                payload,
                category_preference=category_pref,
                category_hint=category_hint,
                query_norms=query_norms,
                year=year,
                tolerance=year_tolerance,
            )
            if candidate:
                logger.info(
                    "TMDB external match via IMDb %s -> %s/%s (%s)",
                    imdb_lookup,
                    candidate.category,
                    candidate.tmdb_id,
                    candidate.title,
                )
                return TMDBResolution(candidate=candidate, margin=1.0, source_query=cleaned_title)

        if tvdb_lookup:
            payload = await _http_request(
                client,
                cache_ttl=config.cache_ttl_seconds,
                path=f"find/{tvdb_lookup}",
                params={"external_source": "tvdb_id"},
                timeout=timeout,
            )
            candidate = _best_external_candidate(
                payload,
                category_preference=category_pref,
                category_hint=category_hint,
                query_norms=query_norms,
                year=year,
                tolerance=year_tolerance,
            )
            if candidate:
                logger.info(
                    "TMDB external match via TVDB %s -> %s/%s (%s)",
                    tvdb_lookup,
                    candidate.category,
                    candidate.tmdb_id,
                    candidate.title,
                )
                return TMDBResolution(candidate=candidate, margin=1.0, source_query=cleaned_title)

        plans = _build_query_plans(
            cleaned_title,
            year=year,
            category_hint=category_hint,
            category_preference=category_pref,
            anime_titles=anime_titles,
            year_tolerance=year_tolerance,
        )

        all_candidates: List[TMDBCandidate] = []

        for plan in plans:
            results = await _perform_search(
                client,
                plan=plan,
                cache_ttl=config.cache_ttl_seconds,
                timeout=timeout,
            )
            candidates = _extract_best_candidate(
                results=results,
                plan=plan,
                query_norms=query_norms,
                tolerance=year_tolerance,
                year=year,
            )
            if not candidates:
                continue
            for candidate in candidates:
                candidate.reason = plan.reason
                all_candidates.append(candidate)
            if any(cand.score >= _STRONG_MATCH_THRESHOLD for cand in candidates):
                break

        if not all_candidates:
            logger.warning("TMDB search returned no viable candidates for %s", filename)
            return None

        all_candidates.sort(key=lambda cand: cand.score, reverse=True)

        baseline_top_score = all_candidates[0].score if all_candidates else 0.0

        viable_candidates = [cand for cand in all_candidates if cand.score >= _SIMILARITY_THRESHOLD]

        alias_scores: Dict[str, float] = {}

        # Consider up to five of the strongest unique candidates for alias lookups. If no
        # candidate cleared the similarity threshold we still want to probe the top results,
        # otherwise entries like "The VVitch" never receive their alternative-title boost.
        alias_targets: List[TMDBCandidate] = []
        seen_ids: set[str] = set()

        def _append_targets(pool: Iterable[TMDBCandidate]) -> None:
            for candidate in pool:
                if candidate.tmdb_id in seen_ids:
                    continue
                alias_targets.append(candidate)
                seen_ids.add(candidate.tmdb_id)
                if len(alias_targets) >= 5:
                    break

        _append_targets(viable_candidates)
        if len(alias_targets) < 5:
            _append_targets(all_candidates)

        needs_alias_lookup = len(alias_targets) > 1 or (
            alias_targets and alias_targets[0].score < _STRONG_MATCH_THRESHOLD
        )
        if needs_alias_lookup:
            for candidate in alias_targets:
                try:
                    aliases = await _fetch_alias_titles(
                        client,
                        category=candidate.category,
                        tmdb_id=candidate.tmdb_id,
                        cache_ttl=config.cache_ttl_seconds,
                        timeout=timeout,
                    )
                except TMDBResolutionError:
                    continue
                candidate_norms: set[str] = set()
                candidate_norms.update(_normalized_variants(candidate.title))
                if candidate.original_title:
                    candidate_norms.update(_normalized_variants(candidate.original_title))
                alias_score = _alias_similarity(
                    query_norms,
                    aliases,
                    existing_norms=candidate_norms,
                )
                if alias_score >= 0.7:
                    # Alias matches should be able to overtake otherwise stronger
                    # candidates (e.g. the ambiguous "The Witch" releases) so scale
                    # the boost with both the alias strength and the current score.
                    incremental = max(0.1, alias_score * 0.25)
                    adjusted = max(candidate.score + incremental, alias_score + 0.05)
                    adjusted = max(adjusted, baseline_top_score + 0.01)
                    alias_scores[candidate.tmdb_id] = adjusted

        if alias_scores:
            for candidate in all_candidates:
                new_score = alias_scores.get(candidate.tmdb_id)
                if new_score and new_score > candidate.score:
                    candidate.score = new_score
                    if "alias" not in candidate.reason:
                        candidate.reason = f"{candidate.reason}+alias"

            all_candidates.sort(key=lambda cand: cand.score, reverse=True)
            viable_candidates = [
                cand for cand in all_candidates if cand.score >= _SIMILARITY_THRESHOLD
            ]

        if not viable_candidates:
            top = all_candidates[:3]
            summary = ", ".join(
                (
                    f"{cand.category.lower()}/{cand.tmdb_id} {cand.title} "
                    f"({cand.year or '????'}) score={cand.score:0.3f}"
                )
                for cand in top
            )
            logger.warning("TMDB search failed for %s. Top candidates: %s", filename, summary)
            return None

        best_candidate = viable_candidates[0]
        runner_up = viable_candidates[1] if len(viable_candidates) > 1 else None

        margin = best_candidate.score - (runner_up.score if runner_up else 0.0)
        resolution = TMDBResolution(
            candidate=best_candidate,
            margin=margin,
            source_query=cleaned_title,
        )

        if not unattended_mode and runner_up and margin < _AMBIGUITY_MARGIN:
            ranked = sorted(all_candidates, key=lambda cand: cand.score, reverse=True)[:5]
            raise TMDBAmbiguityError(ranked)

        if runner_up and margin < _AMBIGUITY_MARGIN:
            logger.warning(
                "TMDB match for %s selected %s/%s (%s) with low margin %.3f",
                filename,
                best_candidate.category,
                best_candidate.tmdb_id,
                best_candidate.title,
                margin,
            )
        else:
            logger.info(
                "TMDB match via %s -> %s/%s (%s, %s) score=%.3f",
                best_candidate.reason,
                best_candidate.category,
                best_candidate.tmdb_id,
                best_candidate.title,
                best_candidate.year or "????",
                best_candidate.score,
            )

        return resolution


__all__ = [
    "MOVIE",
    "TV",
    "TMDBResolution",
    "TMDBResolutionError",
    "TMDBAmbiguityError",
    "TMDBCandidate",
    "resolve_tmdb",
    "parse_manual_id",
]
