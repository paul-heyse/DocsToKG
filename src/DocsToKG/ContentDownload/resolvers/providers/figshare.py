"""Figshare repository resolver for DOI-indexed research outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


class FigshareResolver:
    """Resolve Figshare repository metadata into download URLs.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> resolver = FigshareResolver()
        >>> resolver.name
        'figshare'
    """

    name = "figshare"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return True when the artifact exposes a DOI for Figshare lookup.

        Args:
            config: Resolver configuration (unused but part of the protocol signature).
            artifact: Work metadata that may reference a Figshare DOI.

        Returns:
            bool: ``True`` when a DOI is present, otherwise ``False``.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Search the Figshare API by DOI and yield PDF file URLs.

        Args:
            session: Requests session used for API calls (supports retry injection).
            config: Resolver configuration providing polite headers and timeouts.
            artifact: Work metadata containing the DOI search key.

        Returns:
            Iterable[ResolverResult]: Iterator yielding resolver results for each candidate URL.

        Raises:
            None
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return

        headers = dict(config.polite_headers)
        headers.setdefault("Content-Type", "application/json")

        try:
            response = request_with_retries(
                session,
                "post",
                "https://api.figshare.com/v2/articles/search",
                json={
                    "search_for": f':doi: "{doi}"',
                    "page": 1,
                    "page_size": 3,
                },
                timeout=config.get_timeout(self.name),
                headers=headers,
            )
        except requests.RequestException as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"message": str(exc)},
            )
            return

        if response.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=response.status_code,
            )
            return

        try:
            articles = response.json()
        except ValueError:
            yield ResolverResult(url=None, event="error", event_reason="json-error")
            return

        if not isinstance(articles, list):
            return

        for article in articles:
            if not isinstance(article, dict):
                continue
            files = article.get("files", []) or []
            if not isinstance(files, list):
                continue
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    continue
                filename = (file_entry.get("name") or "").lower()
                download_url = file_entry.get("download_url")

                if filename.endswith(".pdf") and download_url:
                    yield ResolverResult(
                        url=download_url,
                        metadata={
                            "source": "figshare",
                            "article_id": article.get("id"),
                            "filename": file_entry.get("name"),
                        },
                    )


__all__ = ["FigshareResolver"]
