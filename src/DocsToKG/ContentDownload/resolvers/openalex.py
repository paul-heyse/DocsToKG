# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.openalex",
#   "purpose": "OpenAlex resolver implementation",
#   "sections": [
#     {
#       "id": "openalex-resolver",
#       "name": "OpenAlexResolver",
#       "anchor": "class-openalexresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation that surfaces OpenAlex-provided URLs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import requests as _requests

from DocsToKG.ContentDownload.core import dedupe

from .base import RegisteredResolver, ResolverEvent, ResolverEventReason, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


class OpenAlexResolver(RegisteredResolver):
    """Resolve OpenAlex work metadata into candidate download URLs."""

    name = "openalex"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        return bool(artifact.pdf_urls or artifact.open_access_url)

    def iter_urls(
        self,
        session: _requests.Session,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        candidates = list(dedupe(artifact.pdf_urls))
        if getattr(artifact, "open_access_url", None):
            candidates.append(artifact.open_access_url)

        if not candidates:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_OPENALEX_URLS,
            )
            return

        for url in dedupe(candidates):
            if not url:
                continue
            yield ResolverResult(url=url, metadata={"source": "openalex_metadata"})
