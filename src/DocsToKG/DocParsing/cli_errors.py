"""Shared CLI validation error types for DocParsing entry points."""

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
