"""Zenodo repository resolver for DOI-indexed research outputs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


LOGGER = logging.getLogger(__name__)


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

        Notes:
            All HTTP calls honour per-resolver timeouts by delegating to
            :meth:`ResolverConfig.get_timeout`.
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
        except requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={
                    "timeout": config.get_timeout(self.name),
                    "error": str(exc),
                },
            )
            return
        except requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except requests.RequestException as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"error": str(exc)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error contacting Zenodo API")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return

        if response.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=response.status_code,
                metadata={
                    "error_detail": f"Zenodo API returned {response.status_code}",
                },
            )
            return

        try:
            data = response.json()
        except ValueError as json_err:
            preview = response.text[:200] if hasattr(response, "text") else ""
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(json_err), "content_preview": preview},
            )
            return

        hits = data.get("hits", {})
        if not isinstance(hits, dict):
            LOGGER.warning("Zenodo API returned malformed hits payload: %s", type(hits).__name__)
            return
        hits_list = hits.get("hits", [])
        if not isinstance(hits_list, list):
            LOGGER.warning("Zenodo API returned malformed hits list: %s", type(hits_list).__name__)
            return
        for record in hits_list or []:
            if not isinstance(record, dict):
                LOGGER.warning("Skipping malformed Zenodo record: %r", record)
                continue
            files = record.get("files", []) or []
            if not isinstance(files, list):
                LOGGER.warning("Skipping Zenodo record with invalid files payload: %r", files)
                continue
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    LOGGER.warning(
                        "Skipping non-dict Zenodo file entry in record %s", record.get("id")
                    )
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
