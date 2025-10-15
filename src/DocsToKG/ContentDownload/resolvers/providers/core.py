"""CORE API resolver for aggregated open access content."""

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


class CoreResolver:
    """Resolve PDFs using the CORE API.

    Attributes:
        name: Resolver identifier exposed to the orchestration pipeline.

    Examples:
        >>> resolver = CoreResolver()
        >>> resolver.name
        'core'
    """

    name = "core"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when a CORE API key and DOI are available.

        Args:
            config: Resolver configuration containing CORE credentials.
            artifact: Work artifact that may include a DOI identifier.

        Returns:
            Boolean indicating whether CORE resolution should proceed.
        """

        return bool(config.core_api_key and artifact.doi)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs returned by the CORE API.

        Args:
            session: HTTP session for issuing API requests to CORE.
            config: Resolver configuration offering headers and timeouts.
            artifact: Work artifact representing the scholarly work under consideration.

        Returns:
            Iterable of resolver results containing download URLs.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
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
        except requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
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
            LOGGER.exception("Unexpected error in CORE resolver")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        if resp.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=resp.status_code,
                metadata={"error_detail": f"CORE API returned {resp.status_code}"},
            )
            return
        try:
            data = resp.json()
        except ValueError as json_err:
            preview = resp.text[:200] if hasattr(resp, "text") else ""
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(json_err), "content_preview": preview},
            )
            return
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
