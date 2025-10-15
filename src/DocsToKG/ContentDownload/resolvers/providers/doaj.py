"""DOAJ (Directory of Open Access Journals) resolver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import dedupe, normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


class DoajResolver:
    """Resolve Open Access links using the DOAJ API.

    Attributes:
        name: Resolver identifier surfaced to the orchestration pipeline.

    Examples:
        >>> resolver = DoajResolver()
        >>> resolver.name
        'doaj'
    """

    name = "doaj"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has a DOI for DOAJ lookup.

        Args:
            config: Resolver configuration containing DOAJ API credentials.
            artifact: Work artifact possibly holding a DOI identifier.

        Returns:
            Boolean indicating whether DOAJ resolution should run.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs discovered via DOAJ article metadata.

        Args:
            session: HTTP session capable of performing DOAJ API requests.
            config: Resolver configuration specifying headers and API key.
            artifact: Work artifact representing the item being resolved.

        Returns:
            Iterable of resolver results for candidate Open Access URLs.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            return []
        headers = dict(config.polite_headers)
        if config.doaj_api_key:
            headers["X-API-KEY"] = config.doaj_api_key
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://doaj.org/api/v2/search/articles/",
                params={"pageSize": 3, "q": f'doi:"{doi}"'},
                headers=headers,
                timeout=config.get_timeout(self.name),
            )
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        candidates = []
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


__all__ = ["DoajResolver"]
