from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.frame_compare.orchestration.phases.base import Phase
from src.frame_compare.orchestration.state import CoordinatorContext
from src.frame_compare.services.publishers import ReportPublisherRequest, SlowpicsPublisherRequest


class PublishPhase(Phase):
    def execute(self, context: CoordinatorContext) -> None:
        reporter = context.env.reporter
        json_tail = context.json_tail
        layout_data = context.layout_data
        cfg = context.env.cfg

        assert context.slowpics_title_inputs is not None, "Slowpics title inputs required for publish phase"
        assert context.slowpics_final_title is not None, "Slowpics final title required for publish phase"

        slowpics_request = SlowpicsPublisherRequest(
            reporter=reporter,
            json_tail=json_tail,
            layout_data=layout_data,
            title_inputs=context.slowpics_title_inputs,
            final_title=context.slowpics_final_title,
            resolved_base=context.slowpics_resolved_base,
            tmdb_disclosure_line=context.slowpics_tmdb_disclosure_line,
            verbose_tmdb_tag=context.slowpics_verbose_tmdb_tag,
            image_paths=list(context.image_paths),
            out_dir=context.env.out_dir,
            config=cfg.slowpics,
        )

        slowpics_publisher = context.dependencies.slowpics_publisher
        slowpics_result = slowpics_publisher.publish(slowpics_request)
        slowpics_url = slowpics_result.url

        report_request = ReportPublisherRequest(
            reporter=reporter,
            json_tail=json_tail,
            layout_data=layout_data,
            report_enabled=context.env.report_enabled,
            root=context.env.root,
            plans=context.plans,
            frames=list(context.frames),
            selection_details=context.selection_details,
            image_paths=list(context.image_paths),
            metadata_title=context.metadata_title,
            slowpics_url=slowpics_url,
            config=cfg.report,
            collected_warnings=context.env.collected_warnings,
        )

        report_publisher = context.dependencies.report_publisher
        report_result = report_publisher.publish(report_request)

        context.slowpics_url = slowpics_url
        context.report_path = report_result.report_path

        # Update Viewer Block
        report_block = json_tail["report"]
        viewer_block = json_tail.get("viewer", {})
        viewer_mode = "slow_pics" if slowpics_url else "local_report" if report_block.get("enabled") and report_block.get("path") else "none"
        viewer_destination: Optional[str]
        viewer_label: str
        if viewer_mode == "slow_pics":
            viewer_destination = slowpics_url
        elif viewer_mode == "local_report":
            raw_path = report_block.get("path")
            viewer_destination = str(raw_path) if raw_path is not None else None
        else:
            viewer_destination = None
        viewer_label = viewer_destination or ""
        if viewer_mode == "local_report" and viewer_destination:
            try:
                viewer_label = str(Path(viewer_destination).resolve().relative_to(context.env.root.resolve()))
            except ValueError:
                viewer_label = viewer_destination
        viewer_mode_display = {
            "slow_pics": "slow.pics",
            "local_report": "Local report",
            "none": "None",
        }.get(viewer_mode, viewer_mode.title())
        viewer_block.update(
            {
                "mode": viewer_mode,
                "mode_display": viewer_mode_display,
                "destination": viewer_destination,
                "destination_label": viewer_label,
            }
        )
        json_tail["viewer"] = viewer_block
        layout_data["viewer"] = viewer_block
        reporter.update_values(layout_data)
