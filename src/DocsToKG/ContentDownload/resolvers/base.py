"""
Base Resolver Protocol and Contract

All resolvers (Unpaywall, arXiv, CrossRef, etc.) implement this protocol.
Resolvers receive work artifacts and return a ResolverResult containing
zero or more DownloadPlans to attempt.

Protocol (structural subtyping):
    class MyResolver:
        name: str
        def resolve(self, artifact, session, ctx, telemetry, run_id) -> ResolverResult:
            ...

Design:
- Pure functions (no side effects beyond HTTP requests)
- Return ResolverResult with zero or more DownloadPlans
- Empty plans() means "nothing to contribute"
- Can include optional notes for telemetry/diagnostics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from DocsToKG.ContentDownload.api import ResolverResult

    # Import telemetry protocol for type hints
    from DocsToKG.ContentDownload.telemetry import AttemptSink


class Resolver(Protocol):
    """
    Protocol for artifact resolvers.

    Every resolver must have:
    - name: str attribute (e.g., "unpaywall", "arxiv", "crossref")
    - resolve() method returning ResolverResult
    """

    name: str
    """Resolver name (e.g., 'unpaywall', 'arxiv', 'crossref', 'landing')."""

    def resolve(
        self,
        artifact: any,
        session: any,
        ctx: any,
        telemetry: Optional[AttemptSink],
        run_id: Optional[str],
    ) -> ResolverResult:
        """
        Attempt to resolve download plans for an artifact.

        Args:
            artifact: Work artifact to resolve (e.g., OpenAlex work object)
            session: HTTP session/client (httpx.Client or similar)
            ctx: Context object with config, caches, state
            telemetry: Optional AttemptSink for logging attempts
            run_id: Optional run ID for correlation

        Returns:
            ResolverResult with zero or more DownloadPlans.
            Empty plans=() means this resolver has nothing to contribute.

        Raises:
            Should not raise. Instead, return empty plans or error details in notes.
        """
        ...
