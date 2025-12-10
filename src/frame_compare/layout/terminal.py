"""Terminal output and color handling."""
from __future__ import annotations

import os
import re
import sys
from typing import Dict, List

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
ANSI_RESET = "\x1b[0m"

COLOR_BLIND_OVERRIDES: Dict[str, str] = {
    "bool_true": "cyan",
    "bool_false": "orange",
    "number_ok": "cyan",
    "number_warn": "orange",
    "number_bad": "purple",
    "success": "cyan",
    "warn": "orange",
    "error": "purple",
}


class AnsiColorMapper:
    """Translate theme color tokens into ANSI escape sequences."""

    _TOKEN_CODES_16 = {
        "cyan": 36,
        "blue": 34,
        "green": 32,
        "yellow": 33,
        "orange": 33,
        "red": 31,
        "grey": 37,
        "gray": 37,
        "white": 37,
        "black": 30,
        "magenta": 35,
        "purple": 35,
    }

    _TOKEN_CODES_256 = {
        "cyan": 51,
        "blue": 75,
        "green": 84,
        "yellow": 184,
        "orange": 214,
        "red": 203,
        "grey": 240,
        "gray": 240,
        "white": 15,
        "black": 0,
        "magenta": 201,
        "purple": 177,
    }

    def __init__(self, *, no_color: bool) -> None:
        """
        Initialize the color mapper and determine the terminal color capability.

        Determines whether color output is disabled by combining the explicit `no_color` flag with the `NO_COLOR` environment variable, records that state on `self.no_color`, and sets `self._capability` to `"none"` when color is disabled. When color is enabled, attempts to enable Windows VT processing and detects the terminal's color capability, storing the result on `self._capability`. This may modify console state when enabling VT mode on Windows.

        Parameters:
            no_color (bool): If True, force-disable all ANSI color output regardless of environment.
        """
        env_no_color = bool(os.environ.get("NO_COLOR"))
        self.no_color = no_color or env_no_color
        self._capability = "none"
        if not self.no_color:
            self._enable_windows_vt_mode()
            self._capability = self._detect_capability()

    @staticmethod
    def _is_truthy_flag(raw_value: str) -> bool:
        """Return True when an environment-style flag requests enabling behavior."""

        normalized = raw_value.strip().lower()
        return bool(normalized) and normalized not in {"0", "false", "no", "off"}

    @staticmethod
    def _is_modern_windows_terminal() -> bool:
        """Return True if heuristics indicate a VT-capable Windows console."""

        if os.name != "nt":
            return False
        if os.environ.get("WT_SESSION"):
            return True
        term_program = os.environ.get("TERM_PROGRAM", "").lower()
        if term_program == "windows_terminal":
            return True
        get_windows_version = getattr(sys, "getwindowsversion", None)
        if callable(get_windows_version):
            try:  # pragma: no cover - platform dependent
                version = get_windows_version()
            except OSError:
                version = None
            if version is not None:
                major = getattr(version, "major", 0)
                build = getattr(version, "build", 0)
                if major > 10 or (major == 10 and build >= 10586):
                    return True
        return False

    @staticmethod
    def _enable_windows_vt_mode() -> None:
        """
        Enable ANSI VT (virtual terminal) processing on Windows consoles so ANSI escape sequences are interpreted.

        This function is a no-op on non-Windows platforms. On Windows it first attempts to initialize color support via the optional `colorama` package; if `colorama` is unavailable it falls back to enabling VT processing through the Windows console API. Failures are handled silently — if VT mode cannot be enabled the function returns without raising.
        """
        if os.name != "nt":
            return
        try:
            import colorama

            colorama.just_fix_windows_console()
            convert = not (
                AnsiColorMapper._is_modern_windows_terminal()
                or AnsiColorMapper._is_truthy_flag(
                    os.environ.get("FRAME_COMPARE_FORCE_256_COLOR", "")
                )
            )
            colorama.init(strip=False, convert=convert)
            return
        except ImportError:
            pass
        try:  # pragma: no cover - platform dependent
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except OSError:
            # If enabling VT mode fails we silently continue; the console will
            # simply ignore the escape codes.
            pass

    @staticmethod
    def _detect_capability() -> str:
        """
        Determine the terminal color capability based on environment variables.

        Checks COLORTERM and TERM for indications of truecolor or 256-color support and returns "256" when detected, otherwise returns "16".

        Returns:
            capability (str): "256" if truecolor/256-color support is likely, "16" otherwise.
        """
        force_256 = os.environ.get("FRAME_COMPARE_FORCE_256_COLOR", "").strip().lower()
        if force_256 and force_256 not in {"0", "false", "no", "off"}:
            return "256"

        colorterm = os.environ.get("COLORTERM", "").lower()
        if any(token in colorterm for token in ("truecolor", "24bit")):
            return "256"
        term = os.environ.get("TERM", "").lower()
        if "256color" in term or "truecolor" in term:
            return "256"
        if os.name == "nt":
            if os.environ.get("WT_SESSION"):
                return "256"
            term_program = os.environ.get("TERM_PROGRAM", "").lower()
            if term_program == "windows_terminal":
                return "256"
            get_windows_version = getattr(sys, "getwindowsversion", None)
            if callable(get_windows_version):
                try:  # pragma: no cover - platform dependent
                    version = get_windows_version()
                except OSError:
                    version = None
                if version is not None:
                    major = getattr(version, "major", 0)
                    build = getattr(version, "build", 0)
                    if major > 10 or (major == 10 and build >= 10586):
                        return "256"
        return "16"

    def apply(self, token: str, text: str) -> str:
        """
        Apply the color/style represented by `token` to `text` using ANSI SGR sequences.

        Parameters:
            token (str): Color or style token name understood by the mapper.
            text (str): The text to wrap with the style; returned unchanged if empty or styling is disabled.

        Returns:
            str: `text` wrapped with the ANSI SGR sequence for `token` and an ANSI reset, or the original `text` if no style is applied.
        """
        if not text or self.no_color:
            return text
        sgr = self._lookup(token)
        if not sgr:
            return text
        return f"{sgr}{text}{ANSI_RESET}"

    def _lookup(self, token: str) -> str:
        """
        Convert a color token into the corresponding ANSI SGR escape sequence for the renderer's detected terminal capability.

        Parameters:
            token (str): Color token string (e.g. "accent.bold" or "green.bright") where the first segment names a color role and subsequent dot-separated segments are modifiers such as "bright", "bold", or "dim".

        Returns:
            str: The ANSI escape sequence that applies the token's color and modifiers, or an empty string if the token is empty or not supported for the current terminal capability.
        """
        token = (token or "").strip()
        if not token:
            return ""
        parts = token.lower().split(".")
        color = parts[0] if parts else ""
        modifiers = {part for part in parts[1:] if part}

        if self._capability == "256":
            code = self._TOKEN_CODES_256.get(color)
            if code is None:
                return ""
            attrs: List[str] = []
            if "bold" in modifiers:
                attrs.append("1")
            if "dim" in modifiers:
                attrs.append("2")
            attrs.append(f"38;5;{code}")
            return f"\x1b[{';'.join(attrs)}m"

        if self._capability == "16":
            base = self._TOKEN_CODES_16.get(color)
            if base is None:
                return ""
            if "bright" in modifiers and 30 <= base <= 37:
                base += 60
            attrs = []
            if "bold" in modifiers:
                attrs.append("1")
            if "dim" in modifiers:
                attrs.append("2")
            attrs.append(str(base))
            return f"\x1b[{';'.join(attrs)}m"

        return ""
