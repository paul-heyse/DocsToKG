# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.arxiv",
#   "purpose": "arXiv resolver implementation",
#   "sections": [
#     {
#       "id": "arxivresolver",
#       "name": "ArxivResolver",
#       "anchor": "class-arxivresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for arXiv preprints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from DocsToKG.ContentDownload.core import strip_prefix
from DocsToKG.ContentDownload.resolvers.base import (
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
)

from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    import httpx

    from DocsToKG.ContentDownload.core import WorkArtifact


@register_v2("arxiv")
class ArxivResolver:
    """Resolve arXiv preprints using arXiv identifier lookups."""

    name = "arxiv"

    def is_enabled(self, config: Any, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the work exposes an arXiv identifier.

        Args:
            config: Resolver configuration (unused but part of the contract).
            artifact: Work record under consideration.

        Returns:
            bool: Whether this resolver should attempt to resolve the work.
        """
        return bool(artifact.arxiv_id)

    def iter_urls(
        self,
        client: "httpx.Client",  # noqa: ARG002 - unused, kept for signature parity
        config: Any,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield the canonical arXiv PDF URL for ``artifact`` when available.

        Args:
            session: Requests session to use for follow-up HTTP calls.
            config: Resolver configuration supplied by the pipeline.
            artifact: Work record containing source identifiers.

        Yields:
            ResolverResult: Candidate PDF URL or skip event.
        """
        arxiv_id = artifact.arxiv_id
        if not arxiv_id:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_ARXIV_ID,
            )
            return
        arxiv_id = strip_prefix(arxiv_id, "arxiv:")
        yield ResolverResult(
            url=f"https://arxiv.org/pdf/{arxiv_id}.pdf", metadata={"identifier": arxiv_id}
        )
