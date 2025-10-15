"""Europe PMC resolver for European open access articles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import dedupe, normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


class EuropePmcResolver:
    """Resolve Open Access links via the Europe PMC REST API.

    Attributes:
        name: Resolver identifier exposed to the orchestration pipeline.

    Examples:
        >>> resolver = EuropePmcResolver()
        >>> resolver.name
        'europe_pmc'
    """

    name = "europe_pmc"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has a DOI suitable for lookup.

        Args:
            config: Resolver configuration providing Europe PMC preferences.
            artifact: Work artifact containing DOI metadata.

        Returns:
            Boolean indicating whether the resolver should attempt a lookup.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs from the Europe PMC API.

        Args:
            session: HTTP session used to execute Europe PMC API requests.
            config: Resolver configuration including polite headers and timeouts.
            artifact: Work artifact representing the target scholarly output.

        Returns:
            Iterable of resolver results containing discovered URLs.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            return []
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": f'DOI:"{doi}"', "format": "json", "pageSize": 3},
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
            )
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
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


__all__ = ["EuropePmcResolver"]
