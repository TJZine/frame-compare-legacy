"""Regression coverage for VSPreview console-safe script generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Sequence, cast

import pytest

import src.frame_compare.core as core_module
from src.frame_compare.cli_runtime import ClipPlan, _AudioAlignmentSummary


@pytest.fixture
def _vspreview_script_text(tmp_path: Path) -> Sequence[str]:
    cfg = core_module._fresh_app_config()
    cfg.audio_alignment.use_vspreview = True
    cfg.audio_alignment.enable = True

    reference_path = tmp_path / "Reference.mkv"
    target_path = tmp_path / "Target.mkv"
    reference_path.write_bytes(b"ref")
    target_path.write_bytes(b"tgt")

    reference_plan = ClipPlan(
        path=reference_path,
        metadata={"label": "Reference"},
    )
    target_plan = ClipPlan(
        path=target_path,
        metadata={"label": "Target"},
    )
    plans = [reference_plan, target_plan]

    summary = _AudioAlignmentSummary(
        offsets_path=tmp_path / "offsets.toml",
        reference_name=reference_path.name,
        measurements=(),
        applied_frames={},
        baseline_shift=0,
        statuses={},
        reference_plan=reference_plan,
        final_adjustments={},
        swap_details={},
        suggested_frames={target_path.name: 2},
        suggestion_mode=True,
        manual_trim_starts={},
    )

    script_path = core_module._write_vspreview_script(plans, summary, cfg, tmp_path)
    return script_path.read_text(encoding="utf-8").splitlines()


def test_ascii_arrows_in_prints(_vspreview_script_text: Sequence[str]) -> None:
    """Ensure all printed guidance lines stick to ASCII-safe arrows."""

    for line in _vspreview_script_text:
        stripped = line.strip()
        if stripped.startswith("print(") or stripped.startswith("safe_print("):
            assert "\u2192" not in line
            assert "\u2194" not in line


def test_reconfigure_present(_vspreview_script_text: Sequence[str]) -> None:
    """The generated script should defensively prefer UTF-8 output."""

    joined = "\n".join(_vspreview_script_text)
    assert 'stdout.reconfigure(encoding="utf-8", errors="replace")' in joined
    assert 'stderr.reconfigure(encoding="utf-8", errors="replace")' in joined


def test_safe_print_fallback_handles_unicode(_vspreview_script_text: Sequence[str]) -> None:
    """`safe_print` should degrade gracefully when faced with non-ASCII glyphs."""

    pattern = re.compile(
        r"^def\s+safe_print\(msg:\s*str\)\s*->\s*None:\n(?:(?:[ \t].*\n)+)",
        re.MULTILINE,
    )
    script_text = "\n".join(_vspreview_script_text)
    match = pattern.search(script_text)
    assert match is not None, "safe_print definition should be present"

    namespace: Dict[str, object] = {}
    exec(match.group(0), namespace)  # noqa: S102 - executing generated code for test validation

    calls: list[str] = []

    def _fake_print(message: str) -> None:
        calls.append(message)
        if "\u2192" in message:
            raise UnicodeEncodeError("cp1252", message, 0, len(message), "test")

    namespace["print"] = _fake_print

    safe_print = cast(Callable[[str], None], namespace["safe_print"])

    safe_print("Unicode arrow \u2192 should be replaced")

    assert len(calls) >= 2
    assert calls[0] == "Unicode arrow \u2192 should be replaced"
    assert calls[-1] == "[log message unavailable due to encoding]"
