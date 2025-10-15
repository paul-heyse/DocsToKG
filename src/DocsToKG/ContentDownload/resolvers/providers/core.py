"""CORE API resolver for aggregated open access content."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import requests

from ..types import Resolver, ResolverConfig, ResolverResult
from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import normalize_doi

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


class CoreResolver:
    """Resolve PDFs using the CORE API."""

    name = "core"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when a CORE API key and DOI are available."""

        return bool(config.core_api_key and artifact.doi)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs returned by the CORE API."""

        doi = normalize_doi(artifact.doi)
        if not doi:
            return []
        headers = dict(config.polite_headers)
        headers["Authorization"] = f"Bearer {config.core_api_key}"
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://api.core.ac.uk/v3/search/works",
                params={"q": f'doi:"{doi}"', "page": 1, "pageSize": 3},
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
        for hit in data.get("results", []) or []:
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


__all__ = ["CoreResolver"]
