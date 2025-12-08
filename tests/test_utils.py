import types
from typing import Any, Dict

import pytest
from pytest import MonkeyPatch

import src.utils as utils


def test_extract_release_group_brackets() -> None:
    assert utils._extract_release_group_brackets("[Team] Title.mkv") == "Team"
    assert utils._extract_release_group_brackets("Title.mkv") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (5, "5"),
        ([1, 2], "1-2"),
        ((), ""),
    ],
)
def test_normalize_episode_number(value: Any, expected: str) -> None:
    assert utils._normalize_episode_number(value) == expected


def test_bracket_only_release_group(monkeypatch: MonkeyPatch) -> None:
    original_import = utils.import_module

    def fake_import(name: str):
        if name == "anitopy":
            return types.SimpleNamespace(parse=lambda _: {})
        return original_import(name)

    monkeypatch.setattr(utils, "import_module", fake_import)

    meta = utils.parse_filename_metadata(
        "[Group] Title.S01E02.1080p.mkv",
        prefer_guessit=False,
    )

    assert meta["release_group"] == "Group"
    assert meta["label"] == "[Group] Title.S01E02.1080p.mkv"


def test_guessit_path(monkeypatch: MonkeyPatch) -> None:
    original_import = utils.import_module

    fake_guessit_result: Dict[str, Any] = {
        "title": "Show",
        "episode": 3,
        "episode_title": "Pilot",
        "release_group": "Team",
        "season": 1,
    }

    def fake_import(name: str):
        if name == "guessit":
            return types.SimpleNamespace(guessit=lambda _: fake_guessit_result)
        return original_import(name)

    monkeypatch.setattr(utils, "import_module", fake_import)

    meta = utils.parse_filename_metadata(
        "Example.mkv",
        prefer_guessit=True,
        always_full_filename=False,
    )

    assert meta["anime_title"] == "Show"
    assert meta["episode_number"] == "3"
    assert meta["episode_title"] == "Pilot"
    assert meta["release_group"] == "Team"
    assert meta["label"] == "[Team] Show S01E03 – Pilot"


def test_guessit_error_fallback(monkeypatch: MonkeyPatch) -> None:
    original_import = utils.import_module

    fake_ani_result: Dict[str, Any] = {
        "anime_title": "Fallback Show",
        "episode_number": 4,
        "episode_title": "Arc",
        "release_group": "FB",
    }

    def fake_import(name: str):
        if name == "guessit":
            def _raiser(_: str) -> Dict[str, Any]:
                raise ValueError("boom") # Raise ValueError to be caught
            return types.SimpleNamespace(guessit=_raiser)
        if name == "anitopy":
            return types.SimpleNamespace(parse=lambda _: fake_ani_result)
        return original_import(name)

        monkeypatch.setattr(utils, "import_module", fake_import)

        meta = utils.parse_filename_metadata(
            "[FB] Title - 04.mkv",
            prefer_guessit=True,
            always_full_filename=False,
        )
        assert meta["anime_title"] == "Fallback Show"
        assert meta["release_group"] == "FB"
        assert meta["label"] != "[FB] Title - 04.mkv"
        assert meta["label"].startswith("[FB] Fallback Show")

def test_metadata_includes_year_and_ids(monkeypatch: MonkeyPatch) -> None:
    original_import = utils.import_module

    def fake_import(name: str):
        if name == "anitopy":
            return types.SimpleNamespace(
                parse=lambda _: {"anime_title": "Sample", "anime_season": 2020}
            )
        return original_import(name)

    monkeypatch.setattr(utils, "import_module", fake_import)

    meta = utils.parse_filename_metadata(
        "Sample.2020.tt7654321.mkv",
        prefer_guessit=False,
        always_full_filename=True,
    )
    assert meta["year"] == "2020"
    assert meta["imdb_id"] == "tt7654321"
    assert meta["tvdb_id"] == ""
    assert meta["title"] == "Sample"
