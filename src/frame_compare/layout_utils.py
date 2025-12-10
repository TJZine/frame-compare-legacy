"""Shared Rich/text formatting helpers for CLI output."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Mapping, Optional

from rich.markup import escape

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_SPACE_COLLAPSE_RE = re.compile(r" {2,}")
_TAB_COLLAPSE_RE = re.compile(r"\t{2,}")

if TYPE_CHECKING:  # pragma: no cover
    from src.frame_compare.cli_runtime import ClipPlan


def color_text(text: str, style: Optional[str]) -> str:
    """Wrap *text* with Rich ``style`` tags when provided."""

    if style:
        return f"[{style}]{text}[/]"
    return text


def format_kv(
    label: str,
    value: object,
    *,
    label_style: Optional[str] = "dim",
    value_style: Optional[str] = "bright_white",
    sep: str = "=",
) -> str:
    """Format a label/value pair with optional Rich styling."""

    label_text = escape(str(label))
    value_text = escape(str(value))
    return f"{color_text(label_text, label_style)}{sep}{color_text(value_text, value_style)}"


def plan_label(plan: "ClipPlan") -> str:
    """Determine a user-facing label for a clip plan using metadata fallbacks."""

    metadata = plan.metadata
    for key in ("label", "title", "anime_title", "file_name"):
        value = metadata.get(key)
        if value:
            text = str(value).strip()
            if text:
                return text
    return plan.path.name


def plan_label_parts(plan: "ClipPlan") -> tuple[str, str]:
    """Return the resolved label and canonical filename for a plan."""

    return plan_label(plan), plan.path.name


def normalise_vspreview_mode(raw: object) -> str:
    """Return a canonical VSPreview mode label (``baseline`` or ``seeded``)."""

    text = str(raw or "baseline").strip().lower()
    return "seeded" if text == "seeded" else "baseline"


def format_resolution_summary(plan: Mapping[str, Any]) -> str:
    """Human-readable summary comparing cropped and final clip dimensions."""

    def format_dimensions(width: int, height: int) -> str:
        return f"{int(width)} \u00D7 {int(height)}"

    original_w = int(plan.get("cropped_w", 0))
    original_h = int(plan.get("cropped_h", 0))
    final_w, final_h = plan.get("final", (original_w, original_h))
    original = format_dimensions(original_w, original_h)
    target = format_dimensions(int(final_w), int(final_h))
    if target == original:
        return f"{original}  (native)"
    return f"{original} \u2192 {target}  (original \u2192 target)"


def sanitize_console_text(text: str, *, max_len: int | None = 512) -> str:
    """
    Return a console-safe version of *text* that follows OWASP A07 (Cross-Site Scripting –
    Output Encoding) guidance for terminal output.

    The sanitizer is intended for console-bound data only: it strips ANSI escape
    sequences, drops non-printable control characters (preserving space/tab), collapses
    redundant spaces, and trims the result to avoid pathological payloads.
    """

    normalized = str(text or "")
    without_ansi = _ANSI_ESCAPE_RE.sub("", normalized)
    with_line_breaks = without_ansi.replace("\r", " ").replace("\n", " ")
    filtered = "".join(ch for ch in with_line_breaks if ch.isprintable() or ch in {" ", "\t"})
    collapsed_spaces = _SPACE_COLLAPSE_RE.sub(" ", filtered)
    collapsed_tabs = _TAB_COLLAPSE_RE.sub("\t", collapsed_spaces)
    clean = collapsed_tabs.strip()

    if max_len is not None and max_len >= 0 and len(clean) > max_len:
        ellipsis = "…"
        if max_len <= len(ellipsis):
            return ellipsis[:max_len]
        limit = max_len - len(ellipsis)
        truncated = clean[:limit].rstrip()
        clean = f"{truncated}{ellipsis}"

    return clean


__all__ = [
    "color_text",
    "format_kv",
    "format_resolution_summary",
    "normalise_vspreview_mode",
    "plan_label",
    "plan_label_parts",
    "sanitize_console_text",
]
