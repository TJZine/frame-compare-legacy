from __future__ import annotations

from typing import cast

from src.frame_compare import media as media_utils
from src.frame_compare.cli_runtime import CLIAppError
from src.frame_compare.orchestration.phases.base import Phase
from src.frame_compare.orchestration.state import CoordinatorContext
from src.frame_compare.services.metadata import CliPromptProtocol, MetadataResolveRequest


class DiscoveryPhase(Phase):
    def execute(self, context: CoordinatorContext) -> None:
        root = context.env.root

        try:
            files = media_utils.discover_media(root)
        except OSError as exc:
            raise CLIAppError(
                f"Failed to list input directory: {exc}",
                rich_message=f"[red]Failed to list input directory:[/red] {exc}",
            ) from exc

        if len(files) < 2:
            raise CLIAppError(
                "Need at least two video files to compare.",
                rich_message="[red]Need at least two video files to compare.[/red]",
            )

        metadata_request = MetadataResolveRequest(
            cfg=context.env.cfg,
            root=root,
            files=files,
            reporter=cast(CliPromptProtocol, context.env.reporter),
            json_tail=context.json_tail,
            layout_data=context.layout_data,
            collected_warnings=context.env.collected_warnings,
        )

        metadata_resolver = context.dependencies.metadata_resolver
        metadata_result = metadata_resolver.resolve(metadata_request)

        # Populate context
        context.plans = list(metadata_result.plans)
        context.metadata = list(metadata_result.metadata)
        context.metadata_title = metadata_result.metadata_title
        context.analyze_path = metadata_result.analyze_path
        context.slowpics_title_inputs = metadata_result.slowpics_title_inputs
        context.slowpics_final_title = metadata_result.slowpics_final_title
        context.slowpics_resolved_base = metadata_result.slowpics_resolved_base
        context.slowpics_tmdb_disclosure_line = metadata_result.slowpics_tmdb_disclosure_line
        context.slowpics_verbose_tmdb_tag = metadata_result.slowpics_verbose_tmdb_tag
        context.tmdb_notes = list(metadata_result.tmdb_notes)
