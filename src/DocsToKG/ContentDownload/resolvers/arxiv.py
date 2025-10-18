# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.arxiv",
#   "purpose": "arXiv resolver implementation",
#   "sections": [
#     {
#       "id": "arxiv-resolver",
#       "name": "ArxivResolver",
#       "anchor": "class-arxivresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for arXiv preprints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from DocsToKG.ContentDownload.core import strip_prefix

from .base import RegisteredResolver, ResolverEvent, ResolverEventReason, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.pipeline import ResolverConfig
    from DocsToKG.ContentDownload.core import WorkArtifact
    import requests as _requests


class ArxivResolver(RegisteredResolver):
    """Resolve arXiv preprints using arXiv identifier lookups."""

    name = "arxiv"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        return bool(artifact.arxiv_id)

    def iter_urls(
        self,
        session: "_requests.Session",
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        arxiv_id = artifact.arxiv_id
        if not arxiv_id:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_ARXIV_ID,
            )
            return
        arxiv_id = strip_prefix(arxiv_id, "arxiv:")
        yield ResolverResult(url=f"https://arxiv.org/pdf/{arxiv_id}.pdf", metadata={"identifier": arxiv_id})
