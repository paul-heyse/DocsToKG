# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.doaj",
#   "purpose": "DOAJ resolver implementation",
#   "sections": [
#     {
#       "id": "doajresolver",
#       "name": "DoajResolver",
#       "anchor": "class-doajresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the DOAJ API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List

import httpx

from DocsToKG.ContentDownload.core import dedupe, normalize_doi

from .registry_v2 import register_v2

class ResolverResult:
    """Result from resolver attempt."""
    def __init__(self, url=None, referer=None, metadata=None, 
                 event=None, event_reason=None, **kwargs):
        self.url = url
        self.referer = referer
        self.metadata = metadata or {}
        self.event = event
        self.event_reason = event_reason
        for k, v in kwargs.items():
            setattr(self, k, v)



if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    

@register_v2("doaj")
class DoajResolver:
    """Resolve Open Access links using the DOAJ API."""

    name = "doaj"
    api_display_name = "DOAJ"

    def is_enabled(self, config: Any, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when a DOI is present for DOAJ lookups.

        Args:
            config: Resolver configuration (unused for enablement).
            artifact: Work record being processed.

        Returns:
            bool: Whether DOAJ should attempt resolution.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        client: httpx.Client,
        config: Any,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield PDF links surfaced by the DOAJ search API.

        Args:
            client: HTTPX client for outbound HTTP calls.
            config: Resolver configuration containing optional API key.
            artifact: Work record supplying DOI metadata.

        Yields:
            ResolverResult: Candidate download URLs or skip events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_DOI,
            )
            return
        extra_headers: Dict[str, str] = {}
        if config.doaj_api_key:
            extra_headers["X-API-KEY"] = config.doaj_api_key
        data, error = self._request_json(
            client,
            "GET",
            "https://doaj.org/api/v2/search/articles/",
            config=config,
            params={"pageSize": 3, "q": f'doi:"{doi}"'},
            headers=extra_headers,
        )
        if error:
            yield error
            return
        candidates: List[str] = []
        for result in data.get("results", []) or []:
            bibjson = (result or {}).get("bibjson", {})
            for link in bibjson.get("link", []) or []:
                if not isinstance(link, dict):
                    continue
                url = link.get("url")
                if url and url.lower().endswith(".pdf"):
                    candidates.append(url)
        for url in dedupe(candidates):
            yield ResolverResult(url=url, metadata={"source": "doaj"})
