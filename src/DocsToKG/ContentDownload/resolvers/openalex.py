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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import httpx

from DocsToKG.ContentDownload.core import dedupe
from DocsToKG.ContentDownload.resolvers.base import (
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
)
from DocsToKG.ContentDownload.urls import canonical_for_index

from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact


@register_v2("openalex")
class OpenAlexResolver:
    """Resolve OpenAlex work metadata into candidate download URLs."""

    name = "openalex"

    def is_enabled(self, config: Any, artifact: WorkArtifact) -> bool:
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
        client: httpx.Client,  # noqa: ARG002 - unused, kept for signature parity
        config: Any,
        artifact: WorkArtifact,
    ) -> Iterable[ResolverResult]:
        """Yield URLs surfaced directly from OpenAlex metadata.

        Args:
            client: HTTPX client (unused; retained for signature parity).
            config: Resolver configuration controlling policy headers.
            artifact: Work metadata containing PDF candidates.

        Yields:
            ResolverResult: Candidate download URLs or skip events.
        """
        candidates = list(dedupe(artifact.pdf_urls))
        open_access_url = getattr(artifact, "open_access_url", None)
        if open_access_url:
            candidates.append(open_access_url)

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
            # Explicitly compute canonical URL for RFC 3986 compliance and deduplication
            try:
                canonical_url = canonical_for_index(url)
            except Exception:
                canonical_url = url
            yield ResolverResult(
                url=url,
                canonical_url=canonical_url,
                metadata={"source": "openalex_metadata"},
            )
