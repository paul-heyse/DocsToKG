"""Zenodo repository resolver for DOI-indexed research outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


class ZenodoResolver:
    """Resolve Zenodo records into downloadable open-access PDF URLs.

    Attributes:
        name: Resolver identifier registered with the content download pipeline.

    Examples:
        >>> resolver = ZenodoResolver()
        >>> resolver.name
        'zenodo'
    """

    name = "zenodo"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return True when the artifact publishes a DOI for Zenodo lookup.

        Args:
            config: Resolver configuration (unused but part of the protocol signature).
            artifact: Work metadata potentially referencing Zenodo.

        Returns:
            bool: ``True`` when a DOI is available, otherwise ``False``.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Query the Zenodo API by DOI and yield PDF file URLs.

        Args:
            session: Requests session used for making Zenodo API calls.
            config: Resolver configuration providing polite headers and timeouts.
            artifact: Work metadata containing the DOI search key.

        Returns:
            Iterable[ResolverResult]: Iterator yielding resolver results for accessible PDFs.

        Raises:
            None
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return

        try:
            response = request_with_retries(
                session,
                "get",
                "https://zenodo.org/api/records/",
                params={"q": f'doi:"{doi}"', "size": 3, "sort": "mostrecent"},
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
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
            data = response.json()
        except ValueError:
            yield ResolverResult(url=None, event="error", event_reason="json-error")
            return

        hits = data.get("hits", {})
        hits_list = hits.get("hits", []) if isinstance(hits, dict) else []
        for record in hits_list or []:
            if not isinstance(record, dict):
                continue
            files = record.get("files", []) or []
            if not isinstance(files, list):
                continue
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    continue
                file_type = (file_entry.get("type") or "").lower()
                file_key = (file_entry.get("key") or "").lower()
                if file_type == "pdf" or file_key.endswith(".pdf"):
                    links = file_entry.get("links")
                    url = links.get("self") if isinstance(links, dict) else None
                    if url:
                        yield ResolverResult(
                            url=url,
                            metadata={
                                "source": "zenodo",
                                "record_id": record.get("id"),
                                "filename": file_entry.get("key"),
                            },
                        )


__all__ = ["ZenodoResolver"]
