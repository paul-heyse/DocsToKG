"""Shared CLI validation error types for DocParsing entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "CLIValidationError",
    "ChunkingCLIValidationError",
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

    def __str__(self) -> str:  # pragma: no cover - formatting handled in helper
        return self.message


class ChunkingCLIValidationError(CLIValidationError):
    """Validation error raised by chunking CLI helpers."""

    stage = "chunk"


class EmbeddingCLIValidationError(CLIValidationError):
    """Validation error raised by embedding CLI helpers."""

    stage = "embed"


def format_cli_error(error: CLIValidationError) -> str:
    """Return a consistent error string for CLI consumption."""

    prefix = f"[{error.stage}]"
    hint = f" Hint: {error.hint}" if error.hint else ""
    return f"{prefix} {error.option}: {error.message}.{hint}".strip()
