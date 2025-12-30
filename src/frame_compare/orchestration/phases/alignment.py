from __future__ import annotations

from typing import cast

from src.frame_compare import metadata as metadata_utils
from src.frame_compare.cli_runtime import CLIAppError
from src.frame_compare.orchestration.phases.base import Phase
from src.frame_compare.orchestration.state import CoordinatorContext
from src.frame_compare.services.alignment import AlignmentRequest
from src.frame_compare.services.metadata import CliPromptProtocol


class AlignmentPhase(Phase):
    def execute(self, context: CoordinatorContext) -> None:
        if context.analyze_path is None:
            raise CLIAppError("analyze_path not set in context (DiscoveryPhase failed?)")

        audio_track_overrides = context.request.audio_track_overrides
        audio_track_override_map = metadata_utils.parse_audio_track_overrides(audio_track_overrides or [])

        alignment_request = AlignmentRequest(
            plans=context.plans,
            cfg=context.env.cfg,
            root=context.env.root,
            analyze_path=context.analyze_path,
            audio_track_overrides=audio_track_override_map,
            reporter=cast(CliPromptProtocol, context.env.reporter),
            json_tail=context.json_tail,
            vspreview_mode=context.env.vspreview_mode_value,
            collected_warnings=context.env.collected_warnings,
        )

        alignment_workflow = context.dependencies.alignment_workflow
        alignment_result = alignment_workflow.run(alignment_request)

        context.update_alignment(alignment_result)
