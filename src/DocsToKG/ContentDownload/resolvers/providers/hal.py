"""HAL (Hyper Articles en Ligne) open archive resolver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import dedupe, normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


class HalResolver:
    """Resolve publications from the HAL open archive.

    Attributes:
        name: Resolver identifier communicated to the pipeline orchestration.

    Examples:
        >>> resolver = HalResolver()
        >>> resolver.name
        'hal'
    """

    name = "hal"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has a DOI for HAL lookup.

        Args:
            config: Resolver configuration providing HAL request settings.
            artifact: Work artifact potentially containing a DOI identifier.

        Returns:
            Boolean indicating whether HAL resolution is applicable.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate HAL download URLs.

        Args:
            session: HTTP session used for outbound HAL API requests.
            config: Resolver configuration specifying headers and timeouts.
            artifact: Work artifact representing the item to resolve.

        Returns:
            Iterable of resolver results describing resolved URLs.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            return []
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://api.archives-ouvertes.fr/search/",
                params={"q": f"doiId_s:{doi}", "fl": "fileMain_s,file_s"},
                headers=config.polite_headers,
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
        docs = (data.get("response") or {}).get("docs") or []
        urls: List[str] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            main = doc.get("fileMain_s")
            if isinstance(main, str):
                urls.append(main)
            files = doc.get("file_s")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, str):
                        urls.append(item)
        for url in dedupe(urls):
            if url.lower().endswith(".pdf"):
                yield ResolverResult(url=url, metadata={"source": "hal"})


__all__ = ["HalResolver"]
