from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, SpinnerColumn

from src.frame_compare.layout import CliLayoutRenderer, load_cli_layout


def _build_renderer() -> CliLayoutRenderer:
    layout = load_cli_layout(Path("cli_layout.v1.json"))
    console = Console(force_terminal=False, width=120, record=True)
    return CliLayoutRenderer(layout, console, quiet=False, verbose=False, no_color=True)


def test_create_progress_uses_spinner_for_dot_style() -> None:
    renderer = _build_renderer()
    renderer.bind_context({}, {"progress_style": "dot"})
    progress = renderer.create_progress("render_bar", transient=True)
    try:
        assert any(isinstance(column, SpinnerColumn) for column in progress.columns)
    finally:
        progress.stop()


def test_create_progress_uses_bar_for_fill_style() -> None:
    renderer = _build_renderer()
    renderer.bind_context({}, {"progress_style": "fill"})
    progress = renderer.create_progress("render_bar", transient=True)
    try:
        assert any(isinstance(column, BarColumn) for column in progress.columns)
    finally:
        progress.stop()
