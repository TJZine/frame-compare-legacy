"""Screenshot naming utilities."""

from __future__ import annotations

from typing import Mapping

from src.frame_compare.render import naming as _naming


def sanitise_label(label: str) -> str:
    return _naming.sanitise_label(label)


def derive_labels(source: str, metadata: Mapping[str, str]) -> tuple[str, str]:
    return _naming.derive_labels(source, metadata)


def prepare_filename(frame: int, label: str) -> str:
    return _naming.prepare_filename(frame, label)
