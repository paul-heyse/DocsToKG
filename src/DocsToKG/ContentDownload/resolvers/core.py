# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.core",
#   "purpose": "CORE API resolver implementation",
#   "sections": [
#     {
#       "id": "coreresolver",
#       "name": "CoreResolver",
#       "anchor": "class-coreresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the CORE API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import httpx

from DocsToKG.ContentDownload.core import normalize_doi

from .base import ApiResolverBase, ResolverEvent, ResolverEventReason, ResolverResult
from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


@register_v2("core")
class CoreResolver(ApiResolverBase):
    """Resolve PDFs using the CORE API."""

    name = "core"
    api_display_name = "CORE"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        """Return ``True`` when a DOI is present and the CORE API is configured.

        Args:
            config: Resolver configuration containing API credentials.
            artifact: Work record under consideration.

        Returns:
            bool: ``True`` if the resolver can operate for this work.
        """

        return bool(config.core_api_key and artifact.doi)

    def iter_urls(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield download URLs discovered via the CORE search API.

        Args:
            client: HTTPX client to execute HTTP calls.
            config: Resolver configuration with credentials and limits.
            artifact: Work record providing identifiers.

        Yields:
            ResolverResult: Candidate download URLs or diagnostic events.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_DOI,
            )
            return
        headers = {"Authorization": f"Bearer {config.core_api_key}"}
        data, error = self._request_json(
            client,
            "GET",
            "https://api.core.ac.uk/v3/search/works",
            config=config,
            params={"q": f'doi:"{doi}"', "page": 1, "pageSize": 3},
            headers=headers,
        )
        if error:
            yield error
            return
        results = (data.get("results") if isinstance(data, dict) else None) or []
        for hit in results:
            if not isinstance(hit, dict):
                continue
            url = hit.get("downloadUrl") or hit.get("pdfDownloadLink")
            if url:
                yield ResolverResult(url=url, metadata={"source": "core"})
            for entry in hit.get("fullTextLinks") or []:
                if isinstance(entry, dict):
                    href = entry.get("url") or entry.get("link")
                    if href and href.lower().endswith(".pdf"):
                        yield ResolverResult(url=href, metadata={"source": "core"})
