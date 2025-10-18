# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.openalex",
#   "purpose": "OpenAlex resolver implementation",
#   "sections": [
#     {
#       "id": "openalexresolver",
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
        """Return ``True`` when OpenAlex metadata includes candidate URLs.

        Args:
            config: Resolver configuration (unused for enablement).
            artifact: Work record being evaluated.

        Returns:
            bool: Whether the resolver should attempt the work.
        """
        return bool(artifact.pdf_urls or artifact.open_access_url)

    def iter_urls(
        self,
        session: _requests.Session,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield URLs surfaced directly from OpenAlex metadata.

        Args:
            session: Requests session (unused; signature parity).
            config: Resolver configuration controlling policy headers.
            artifact: Work metadata containing PDF candidates.

        Yields:
            ResolverResult: Candidate download URLs or skip events.
        """
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
