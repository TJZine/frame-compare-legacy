from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Dict, List, cast

from src.frame_compare import runtime_utils
from src.frame_compare.cli_runtime import coerce_str_mapping
from src.frame_compare.orchestration import reporting
from src.frame_compare.orchestration.phases.base import Phase
from src.frame_compare.orchestration.state import CoordinatorContext, RunResult
from src.frame_compare.result_snapshot import (
    RenderOptions,
    ResultSource,
    SectionAvailability,
    SectionState,
    build_snapshot,
    render_run_result,
    resolve_cli_version,
    write_snapshot,
)

logger = logging.getLogger('frame_compare')


class ResultPhase(Phase):
    def execute(self, context: CoordinatorContext) -> None:
        env = context.env
        reporter = env.reporter
        json_tail = context.json_tail
        layout_data = context.layout_data
        request = context.request

        # 1. Initial Result Construction
        result = RunResult(
            files=[plan.path for plan in context.plans],
            frames=list(context.frames),
            out_dir=env.out_dir,
            out_dir_created=env.out_dir_created,
            out_dir_created_path=env.out_dir_created_path,
            root=env.root,
            config=env.cfg,
            image_paths=list(context.image_paths),
            slowpics_url=context.slowpics_url,
            json_tail=json_tail,
            report_path=context.report_path,
        )

        # 2. Warnings Processing
        collected_warnings = env.collected_warnings
        for warning in collected_warnings:
            reporter.warn(warning)

        warnings_list = list(dict.fromkeys(reporter.iter_warnings()))
        json_tail["warnings"] = warnings_list

        # Warnings Layout
        raw_layout_sections_obj = getattr(getattr(reporter, "layout", None), "sections", [])
        layout_sections: list[Mapping[str, Any]] = []
        if isinstance(raw_layout_sections_obj, list):
            layout_sources = cast(list[Any], raw_layout_sections_obj)
            for section_obj in layout_sources:
                if isinstance(section_obj, Mapping):
                    layout_sections.append(cast(Mapping[str, Any], section_obj))

        warnings_section: dict[str, object] | None = None
        for section_map in layout_sections:
            if section_map.get("id") == "warnings":
                warnings_section = dict(section_map)
                break
        fold_config_source = warnings_section.get("fold_labels") if warnings_section is not None else None
        if isinstance(fold_config_source, Mapping):
            fold_config = coerce_str_mapping(cast(Mapping[str, object], fold_config_source))
        else:
            fold_config = {}
        fold_head = fold_config.get("head")
        fold_tail = fold_config.get("tail")
        fold_when = fold_config.get("when")
        head = int(fold_head) if isinstance(fold_head, (int, float)) else 2
        tail = int(fold_tail) if isinstance(fold_tail, (int, float)) else 1
        joiner = str(fold_config.get("joiner", ", "))
        fold_when_text = str(fold_when) if isinstance(fold_when, str) and fold_when else None
        fold_enabled = runtime_utils.evaluate_rule_condition(fold_when_text, flags=reporter.flags)

        warnings_data: List[Dict[str, object]] = []
        if warnings_list:
            labels_text = runtime_utils.fold_sequence(warnings_list, head=head, tail=tail, joiner=joiner, enabled=fold_enabled)
            warnings_data.append(
                {
                    "warning.type": "general",
                    "warning.count": len(warnings_list),
                    "warning.labels": labels_text,
                }
            )
        else:
            warnings_data.append(
                {
                    "warning.type": "general",
                    "warning.count": 0,
                    "warning.labels": "none",
                }
            )

        layout_data["warnings"] = warnings_data
        reporter.update_values(layout_data)

        # 3. Section Availability & Snapshot
        section_states: Dict[str, SectionState] = {}
        for section in layout_sections:
            section_id_raw = section.get("id")
            if not section_id_raw:
                continue
            section_id = str(section_id_raw).strip()
            if not section_id:
                continue
            section_states[section_id] = SectionState(
                availability=SectionAvailability.MISSING,
                note=None,
            )

        def _mark_section(section_id: str, availability: SectionAvailability, note: str | None = None) -> None:
            if section_id not in section_states:
                return
            section_states[section_id] = SectionState(availability=availability, note=note)

        for section_id in list(section_states):
            _mark_section(section_id, SectionAvailability.FULL)

        reporting.apply_section_availability_overrides(
            section_states,
            _mark_section,
            layout_data=layout_data,
            result=result,
        )

        snapshot = build_snapshot(
            values=reporter.values,
            flags=reporter.flags,
            layout_sections=layout_sections,
            section_states=section_states,
            files=result.files,
            frames=result.frames,
            image_paths=result.image_paths,
            slowpics_url=result.slowpics_url,
            report_path=result.report_path,
            warnings=warnings_list,
            json_tail=result.json_tail,
            source=ResultSource.LIVE,
            cli_version=resolve_cli_version(),
        )
        result.snapshot = snapshot
        result.snapshot_path = env.result_snapshot_path

        try:
            write_snapshot(env.result_snapshot_path, snapshot)
        except OSError:
            logger.warning("Failed to persist run snapshot to %s", env.result_snapshot_path, exc_info=True)

        render_options = RenderOptions(
            show_partial=request.show_partial_sections,
            show_missing_sections=request.show_missing_sections,
        )

        render_run_result(
            snapshot=snapshot,
            reporter=reporter,
            layout_sections=layout_sections,
            options=render_options,
        )

        context.result = result
