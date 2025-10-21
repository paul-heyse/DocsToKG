# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.europe_pmc",
#   "purpose": "Europe PMC resolver implementation",
#   "sections": [
#     {
#       "id": "europepmcresolver",
#       "name": "EuropePmcResolver",
#       "anchor": "class-europepmcresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the Europe PMC REST API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List

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
    

@register_v2("europe_pmc")
class EuropePmcResolver:
    """Resolve Open Access links via the Europe PMC REST API."""

    name = "europe_pmc"

    def is_enabled(self, config: Any, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when a DOI is available for Europe PMC lookups.

        Args:
            config: Resolver configuration (unused but part of the interface).
            artifact: Work record we may attempt to resolve.

        Returns:
            bool: Whether this resolver should run for the work.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        client: httpx.Client,
        config: Any,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield PDF URLs announced by the Europe PMC REST API.

        Args:
            client: HTTPX client for HTTP calls.
            config: Resolver configuration controlling limits.
            artifact: Work record that triggered this resolver.

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
        data, error = self._request_json(
            client,
            "GET",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            config=config,
            params={"query": f'DOI:"{doi}"', "format": "json", "pageSize": 3},
        )
        if error:
            if error.event_reason is ResolverEventReason.HTTP_ERROR:
                return
            yield error
            return
        candidates: List[str] = []
        for result in (data.get("resultList", {}) or {}).get("result", []) or []:
            full_text = result.get("fullTextUrlList", {}) or {}
            for entry in full_text.get("fullTextUrl", []) or []:
                if not isinstance(entry, dict):
                    continue
                if (entry.get("documentStyle") or "").lower() != "pdf":
                    continue
                url = entry.get("url")
                if url:
                    candidates.append(url)
        for url in dedupe(candidates):
            yield ResolverResult(url=url, metadata={"source": "europe_pmc"})
