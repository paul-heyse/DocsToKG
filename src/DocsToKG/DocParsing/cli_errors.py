# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.cli_errors",
#   "purpose": "Exception types and formatting helpers shared by DocParsing CLIs.",
#   "sections": [
#     {
#       "id": "clivalidationerror",
#       "name": "CLIValidationError",
#       "anchor": "class-clivalidationerror",
#       "kind": "class"
#     },
#     {
#       "id": "chunkingclivalidationerror",
#       "name": "ChunkingCLIValidationError",
#       "anchor": "class-chunkingclivalidationerror",
#       "kind": "class"
#     },
#     {
#       "id": "doctagsclivalidationerror",
#       "name": "DoctagsCLIValidationError",
#       "anchor": "class-doctagsclivalidationerror",
#       "kind": "class"
#     },
#     {
#       "id": "embeddingclivalidationerror",
#       "name": "EmbeddingCLIValidationError",
#       "anchor": "class-embeddingclivalidationerror",
#       "kind": "class"
#     },
#     {
#       "id": "format-cli-error",
#       "name": "format_cli_error",
#       "anchor": "function-format-cli-error",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Exception types and formatting helpers shared by DocParsing CLIs.

The CLI layers for DocTags, chunking, and embedding all surface validation
failures through a common set of lightweight exceptions so that callers receive
consistent error messaging regardless of the stage they are invoking. This
module defines those stage-specific subclasses, embeds hints for remediation,
and provides rendering helpers that the unified CLI uses to keep terminal output
predictable for both humans and automation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "CLIValidationError",
    "ChunkingCLIValidationError",
    "DoctagsCLIValidationError",
    "EmbeddingCLIValidationError",
    "format_cli_error",
]


@dataclass(slots=True)
class CLIValidationError(ValueError):
    """Base exception capturing option names and human-friendly messages."""

    option: str
    message: str
    hint: Optional[str] = None
    stage: str = "cli"

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        """Initialise the ``ValueError`` base with the human-readable message."""

        ValueError.__init__(self, self.message)

    def __str__(self) -> str:  # pragma: no cover - formatting handled in helper
        """Return the underlying message for convenience."""

        return self.message


class ChunkingCLIValidationError(CLIValidationError):
    """Validation error raised by chunking CLI helpers."""

    stage = "chunk"

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        """Ensure the chunk stage marker is applied before chaining."""

        self.stage = "chunk"
        CLIValidationError.__post_init__(self)


class DoctagsCLIValidationError(CLIValidationError):
    """Validation error raised by DocTags CLI helpers."""

    stage = "doctags"

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        """Ensure the doctags stage marker is applied before chaining."""

        self.stage = "doctags"
        CLIValidationError.__post_init__(self)


class EmbeddingCLIValidationError(CLIValidationError):
    """Validation error raised by embedding CLI helpers."""

    stage = "embed"

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        """Ensure the embed stage marker is applied before chaining."""

        self.stage = "embed"
        CLIValidationError.__post_init__(self)


def format_cli_error(error: CLIValidationError) -> str:
    """Return a consistent error string for CLI consumption."""

    prefix = f"[{error.stage}]"
    hint = f" Hint: {error.hint}" if error.hint else ""
    return f"{prefix} {error.option}: {error.message}.{hint}".strip()
