"""Figshare repository resolver for DOI-indexed research outputs."""

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

        Notes:
            Requests honour resolver-specific timeouts using
            :meth:`ResolverConfig.get_timeout` and reuse
            :func:`request_with_retries` for resilient execution.
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
            LOGGER.exception("Unexpected error in Figshare resolver")
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
                    "error_detail": f"Figshare API returned {response.status_code}",
                },
            )
            return

        try:
            articles = response.json()
        except ValueError as json_err:
            preview = response.text[:200] if hasattr(response, "text") else ""
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(json_err), "content_preview": preview},
            )
            return

        if not isinstance(articles, list):
            LOGGER.warning(
                "Figshare API returned non-list articles payload: %s", type(articles).__name__
            )
            return

        for article in articles:
            if not isinstance(article, dict):
                LOGGER.warning("Skipping malformed Figshare article: %r", article)
                continue
            files = article.get("files", []) or []
            if not isinstance(files, list):
                LOGGER.warning("Skipping Figshare article with invalid files payload: %r", files)
                continue
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    LOGGER.warning("Skipping non-dict Figshare file entry: %r", file_entry)
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
