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

import requests as _requests

from DocsToKG.ContentDownload.core import dedupe, normalize_doi

from .base import ApiResolverBase, ResolverEvent, ResolverEventReason, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


class EuropePmcResolver(ApiResolverBase):
    """Resolve Open Access links via the Europe PMC REST API."""

    name = "europe_pmc"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
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
        session: _requests.Session,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield PDF URLs announced by the Europe PMC REST API.

        Args:
            session: Requests session for HTTP calls.
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
            session,
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
