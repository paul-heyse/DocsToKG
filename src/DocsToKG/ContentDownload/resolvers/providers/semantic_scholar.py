"""
Semantic Scholar Resolver Provider

This module integrates with the Semantic Scholar Graph API to locate open
access PDFs associated with DOI-indexed papers.

Key Features:
- Memoised API lookups to respect rate limits and improve performance.
- Optional API key support via standard ``x-api-key`` headers.
- Structured error emission covering HTTP and JSON decoding failures.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers.semantic_scholar import SemanticScholarResolver

    resolver = SemanticScholarResolver()
    results = list(resolver.iter_urls(session, config, artifact))
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple
from urllib.parse import quote

import requests

from DocsToKG.ContentDownload.utils import normalize_doi

from ..headers import headers_cache_key
from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1000)
def _fetch_semantic_scholar_data(
    doi: str,
    api_key: Optional[str],
    timeout: float,
    headers_key: Tuple[Tuple[str, str], ...],
) -> Dict[str, Any]:
    """Fetch Semantic Scholar Graph API metadata for ``doi`` with caching.

    Args:
        doi: Normalised DOI string to query.
        api_key: Optional Semantic Scholar API key to include in the request.
        timeout: Request timeout in seconds.
        headers_key: Hashable representation of polite headers for cache lookups.

    Returns:
        Decoded JSON payload returned by the Semantic Scholar Graph API.

    Raises:
        requests.HTTPError: If the API responds with a non-success status code.
    """

    headers = dict(headers_key)
    if api_key:
        headers["x-api-key"] = api_key
    response = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}",
        params={"fields": "title,openAccessPdf"},
        timeout=timeout,
        headers=headers,
    )
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()


class SemanticScholarResolver:
    """Resolve PDFs via the Semantic Scholar Graph API.

    Attributes:
        name: Resolver identifier exposed to the orchestration layer.

    Examples:
        >>> resolver = SemanticScholarResolver()
        >>> resolver.name
        'semantic_scholar'
    """

    name = "semantic_scholar"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has a DOI suitable for lookup.

        Args:
            config: Resolver configuration containing Semantic Scholar settings.
            artifact: Work artifact potentially carrying a DOI.

        Returns:
            Boolean indicating whether the resolver can operate on the artifact.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs discovered via Semantic Scholar.

        Args:
            session: HTTP session for outbound API communication.
            config: Resolver configuration providing headers and timeouts.
            artifact: Work artifact describing the scholarly work to resolve.

        Returns:
            Iterable of resolver results containing download URLs or metadata events.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        try:
            data = _fetch_semantic_scholar_data(
                doi,
                config.semantic_scholar_api_key,
                config.get_timeout(self.name),
                headers_cache_key(config.polite_headers),
            )
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            detail = status if status is not None else "unknown"
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=status,
                metadata={"error_detail": f"Semantic Scholar HTTPError: {detail}"},
            )
            return
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
        except ValueError as json_err:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(json_err)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected Semantic Scholar resolver error")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return

        open_access = (data.get("openAccessPdf") or {}) if isinstance(data, dict) else {}
        url = open_access.get("url") if isinstance(open_access, dict) else None
        if url:
            yield ResolverResult(url=url, metadata={"source": "semantic-scholar"})
        else:
            yield ResolverResult(
                url=None,
                event="skipped",
                event_reason="no-openaccess-pdf",
                metadata={"doi": doi},
            )


__all__ = [
    "SemanticScholarResolver",
    "_fetch_semantic_scholar_data",
]
