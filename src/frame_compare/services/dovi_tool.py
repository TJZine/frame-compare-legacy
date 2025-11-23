"""
Service wrapper for dovi_tool CLI to extract L1 metadata.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, cast

from src.frame_compare.preflight import PROJECT_ROOT

logger = logging.getLogger(__name__)

class DoviToolService:
    """Handles interaction with the dovi_tool binary."""

    def __init__(self) -> None:
        self.binary_path = self._resolve_binary()

    def _resolve_binary(self) -> Path | None:
        """Resolve the platform-specific dovi_tool binary."""
        tools_dir = PROJECT_ROOT / "tools"

        # Check for platform-specific binary names
        if sys.platform == "win32":
            binary = tools_dir / "dovi_tool.exe"
        else:
            binary = tools_dir / "dovi_tool"

        if binary.exists():
            return binary

        # Fallback: check if both exist and pick the one that matches (user might have dumped both)
        # This is redundant if the above check works, but good for safety if logic changes
        return None

    def is_available(self) -> bool:
        return self.binary_path is not None and self.binary_path.exists()

    def extract_rpu_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract RPU metadata from the given file using dovi_tool.
        Returns a list of dicts, where each dict contains L1 stats for a frame.
        """
        if not self.is_available():
            logger.warning("dovi_tool binary not found in tools/")
            return []

        # Cache file to avoid re-running expensive extraction
        cache_path = file_path.with_suffix(file_path.suffix + ".dovi_info.json")
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    logger.info("Found cached dovi info at %s with %d frames", cache_path, len(cached_data))
                    return cached_data # FORCE FRESH RUN
            except Exception as e:
                logger.warning("Failed to load cached dovi info: %s", e)

        rpu_bin = None
        json_out = None

        try:
            # Step 1: Extract RPU binary
            # dovi_tool extract-rpu -i <video> -o <temp.bin>
            fd_bin, rpu_bin = tempfile.mkstemp(suffix=".bin")
            os.close(fd_bin)

            cmd_extract = [str(self.binary_path), "extract-rpu", "-i", str(file_path), "-o", rpu_bin]
            logger.info("Running dovi_tool extract-rpu: %s", " ".join(cmd_extract))
            subprocess.run(cmd_extract, check=True, capture_output=True, text=True)

            # Step 2: Export RPU to JSON
            # dovi_tool export -i <temp.bin> -d all=<temp.json>
            fd_json, json_out = tempfile.mkstemp(suffix=".json")
            os.close(fd_json)

            cmd_export = [str(self.binary_path), "export", "-i", rpu_bin, "-d", f"all={json_out}"]
            logger.info("Running dovi_tool export: %s", " ".join(cmd_export))
            subprocess.run(cmd_export, check=True, capture_output=True, text=True)

            # Step 3: Parse JSON
            with open(json_out, "r", encoding="utf-8") as f:
                data = json.load(f)

            frames_metadata = self._parse_dovi_json(data)

            if frames_metadata:
                logger.info("Extracted metadata for %d frames.", len(frames_metadata))
            else:
                logger.warning("Extracted metadata is empty!")

            # Save to cache
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(frames_metadata, f)
            except Exception as e:
                logger.warning("Failed to write dovi info cache: %s", e)

            return frames_metadata

        except subprocess.CalledProcessError as e:
            logger.warning("dovi_tool failed: %s\nStderr: %s", e, e.stderr)
            return []
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse dovi_tool output: %s", e)
            return []
        except Exception as e:
            logger.warning("Unexpected error running dovi_tool: %s", e)
            return []
        finally:
            # Cleanup temp files
            if rpu_bin and os.path.exists(rpu_bin):
                try:
                    os.remove(rpu_bin)
                except OSError:
                    pass
            if json_out and os.path.exists(json_out):
                try:
                    os.remove(json_out)
                except OSError:
                    pass

    def _pq_to_nits(self, pq_val: int | float) -> float:
        """
        Convert PQ value (0-4095 range or 0-1 range) to nits using ST2084 EOTF.
        """
        # Constants for ST2084
        m1 = 2610.0 / 4096.0 / 4.0
        m2 = 2523.0 / 4096.0 * 128.0
        c1 = 3424.0 / 4096.0
        c2 = 2413.0 / 4096.0 * 32.0
        c3 = 2392.0 / 4096.0 * 32.0

        # Normalize to 0-1
        # If value is > 1.0, assume it's 12-bit integer
        val = float(pq_val)
        if val > 1.0:
            val = val / 4095.0

        if val <= 0.0:
            return 0.0

        # EOTF
        # L = ((max[(N^(1/m2) - c1) / (c2 - c3 * N^(1/m2)), 0])^(1/m1)) * 10000
        try:
            pow_val = val ** (1.0 / m2)
            num = max(pow_val - c1, 0.0)
            den = c2 - c3 * pow_val
            if den == 0:
                return 10000.0 # Avoid div by zero, theoretically max brightness

            linear_val = (num / den) ** (1.0 / m1)
            return linear_val * 10000.0
        except Exception:
            return 0.0

    def _parse_dovi_json(self, data: Any) -> List[Dict[str, Any]]:
        """
        Parse the raw JSON from dovi_tool into a simplified list of frame stats.
        """
        extracted: List[Dict[str, Any]] = []

        if isinstance(data, dict):
            data_dict = cast(Dict[str, Any], data)
            frames = data_dict.get("frames", [])
        else:
            frames = data

        if not isinstance(frames, list):
            logger.warning("dovi_tool output 'frames' is not a list")
            return []

        frame_list = cast(List[Any], frames)

        for frame in frame_list:
            if not isinstance(frame, dict):
                continue
            stats: Dict[str, Any] = {}

            # Check for vdr_dm_data directly (new dovi_tool structure)
            vdr_dm = cast(Dict[str, Any], frame.get("vdr_dm_data", {})) # pyright: ignore[reportUnknownMemberType]

            # Fallback: Check for rpu -> vdr_dm_data (old structure?)
            if not vdr_dm:
                rpu = cast(Dict[str, Any], frame.get("rpu", {})) # pyright: ignore[reportUnknownMemberType]
                vdr_dm = cast(Dict[str, Any], rpu.get("vdr_dm_data", {})) # pyright: ignore[reportUnknownMemberType]

            # Try to find Level 1 in various locations
            l1: Dict[str, Any] = {}

            # 1. Direct 'level1' in vdr_dm_data (some versions)
            if "level1" in vdr_dm:
                l1 = cast(Dict[str, Any], vdr_dm.get("level1", {})) # pyright: ignore[reportUnknownMemberType]

            # 2. Inside cmv29_metadata (v2.9)
            if not l1 and "cmv29_metadata" in vdr_dm:
                cmv29 = cast(Dict[str, Any], vdr_dm.get("cmv29_metadata", {})) # pyright: ignore[reportUnknownMemberType]

                # Try to find Level1 in ext_metadata_blocks
                if "ext_metadata_blocks" in cmv29:
                    blocks = cast(List[Any], cmv29["ext_metadata_blocks"])
                    for block in blocks:
                        if isinstance(block, dict) and "Level1" in block:
                            l1 = cast(Dict[str, Any], block["Level1"])
                            break

                if not l1:
                    l1 = cast(Dict[str, Any], cmv29.get("level1", {})) # pyright: ignore[reportUnknownMemberType]

            # 3. Inside cmv40_metadata (v4.0)
            if not l1 and "cmv40_metadata" in vdr_dm:
                cmv40 = cast(Dict[str, Any], vdr_dm.get("cmv40_metadata", {})) # pyright: ignore[reportUnknownMemberType]

                # Try to find Level1 in ext_metadata_blocks
                if "ext_metadata_blocks" in cmv40:
                    blocks = cast(List[Any], cmv40["ext_metadata_blocks"])
                    for block in blocks:
                        if isinstance(block, dict) and "Level1" in block:
                            l1 = cast(Dict[str, Any], block["Level1"])
                            break

                if not l1:
                    l1 = cast(Dict[str, Any], cmv40.get("level1", {})) # pyright: ignore[reportUnknownMemberType]

            if not l1:
                continue

            if "min_pq" in l1:
                val = l1["min_pq"]
                if isinstance(val, (int, float)):
                    stats["l1_min_nits"] = self._pq_to_nits(val)
            if "max_pq" in l1:
                val = l1["max_pq"]
                if isinstance(val, (int, float)):
                    stats["l1_max_nits"] = self._pq_to_nits(val)
            if "avg_pq" in l1:
                val = l1["avg_pq"]
                if isinstance(val, (int, float)):
                    stats["l1_avg_nits"] = self._pq_to_nits(val)

            extracted.append(stats)

        return extracted

# Singleton instance
dovi_tool = DoviToolService()
