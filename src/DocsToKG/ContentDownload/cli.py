"""Legacy CLI entry point forwarding to the modern Typer application.

This module preserves the historical ``python -m DocsToKG.ContentDownload.cli``
invocation while delegating to :mod:`DocsToKG.ContentDownload.cli_v2`. It also
registers the queue/orchestrator management commands exposed by
:mod:`DocsToKG.ContentDownload.cli_orchestrator` so existing automation can rely
on a single import path.
"""

from __future__ import annotations

from DocsToKG.ContentDownload.cli_orchestrator import (
    OrchestratorConfig,
    WorkQueue,
    app as queue_app,
)
from DocsToKG.ContentDownload.cli_v2 import app as core_app

__all__ = ["app", "main", "OrchestratorConfig", "WorkQueue"]

app = core_app

# Register queue/orchestrator subcommands exactly once to keep help text stable.
if not getattr(app, "_docs_to_kg_queue_registered", False):
    app.add_typer(
        queue_app,
        name="queue",
        help="Persistent work queue management commands.",
    )
    setattr(app, "_docs_to_kg_queue_registered", True)


def main() -> None:
    """Invoke the Typer application."""

    app()


if __name__ == "__main__":  # pragma: no cover - manual CLI invocation helper
    main()
