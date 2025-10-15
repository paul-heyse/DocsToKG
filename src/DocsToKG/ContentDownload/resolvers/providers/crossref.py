"""
Crossref Resolver Provider

This module integrates with the Crossref metadata API to surface direct and
publisher-hosted PDF links for scholarly works. It uses polite rate limiting
and caching strategies to comply with Crossref service guidelines.

Key Features:
- Cached metadata retrieval with header normalisation for polite access.
- Robust error handling for HTTP, timeout, and JSON parsing failures.
- Deduplication of returned URLs to minimise redundant download attempts.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers.crossref import CrossrefResolver

    resolver = CrossrefResolver()
    results = list(resolver.iter_urls(session, config, artifact))
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests

from DocsToKG.ContentDownload.utils import dedupe, normalize_doi

from ..types import ResolverConfig, ResolverResult
from .unpaywall import _headers_cache_key

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1000)
def _fetch_crossref_data(
    doi: str,
   mailto: Optional[str],
    timeout: float,
    headers_key: Tuple[Tuple[str, str], ...],
) -> Dict[str, Any]:
    """Retrieve Crossref metadata for ``doi`` with polite header caching.

    Args:
        doi: Normalised DOI string to request metadata for.
        mailto: Contact email used for Crossref's polite rate limiting headers.
        timeout: Request timeout in seconds.
        headers_key: Hashable representation of polite headers for cache lookups.

    Returns:
        Decoded JSON payload returned by the Crossref API.

    Raises:
        requests.HTTPError: If the Crossref API responds with a non-success status.
    """

    headers = dict(headers_key)
    params = {"mailto": mailto} if mailto else None
    response = requests.get(
        f"https://api.crossref.org/works/{quote(doi)}",
        params=params,
        timeout=timeout,
        headers=headers,
    )
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()


class CrossrefResolver:
    """Resolve candidate URLs from the Crossref metadata API.

    Attributes:
        name: Resolver identifier advertised to the orchestrating pipeline.

    Examples:
        >>> resolver = CrossrefResolver()
        >>> resolver.name
        'crossref'
    """

    name = "crossref"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has a DOI available for lookup.

        Args:
            config: Resolver configuration providing Crossref connectivity details.
            artifact: Work artifact containing bibliographic metadata.

        Returns:
            Boolean indicating whether a Crossref query should be attempted.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield URLs discovered via the Crossref API for a given artifact.

        Args:
            session: HTTP session capable of issuing outbound requests.
            config: Resolver configuration including timeouts and polite headers.
            artifact: Work artifact carrying DOI metadata.

        Returns:
            Iterable of resolver results describing discovered URLs or errors.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(
                url=None,
                event="skipped",
                event_reason="no-doi",
            )
            return
        email = config.mailto or config.unpaywall_email
        endpoint = f"https://api.crossref.org/works/{quote(doi)}"
        params = {"mailto": email} if email else None
        headers = dict(config.polite_headers)
        if hasattr(session, "get"):
            try:
                response = session.get(
                    endpoint,
                    params=params,
                    timeout=config.get_timeout(self.name),
                    headers=headers,
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
                LOGGER.exception("Unexpected error in Crossref resolver")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                )
                return

            status = getattr(response, "status_code", 200)
            if status != 200:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=status,
                    metadata={"error_detail": f"Crossref API returned {status}"},
                )
                return

            try:
                data = response.json()
            except ValueError as json_err:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
                    metadata={
                        "error_detail": str(json_err),
                        "content_preview": response.text[:200] if hasattr(response, "text") else "",
                    },
                )
                return
        else:
            try:
                data = _fetch_crossref_data(
                    doi,
                    email,
                    config.get_timeout(self.name),
                    _headers_cache_key(config.polite_headers),
                )
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=status,
                    metadata={"error_detail": f"Crossref HTTPError: {status}"},
                )
                return
            except requests.RequestException as exc:  # pragma: no cover - network errors
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
                LOGGER.exception("Unexpected cached request error in Crossref resolver")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                )
                return

        message = ((data or {}).get("message") or {}) if isinstance(data, dict) else {}
        link_section = message.get("link") or []
        if not isinstance(link_section, list):
            link_section = []

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for entry in link_section:
            if not isinstance(entry, dict):
                continue
            url = entry.get("URL")
            content_type = entry.get("content-type")
            if url and (content_type or "").lower() in {"application/pdf", "text/html"}:
                candidates.append((url, {"content_type": content_type}))

        for url in dedupe([candidate_url for candidate_url, _ in candidates]):
            for candidate_url, metadata in candidates:
                if candidate_url == url:
                    yield ResolverResult(url=url, metadata=metadata)
                    break


__all__ = [
    "CrossrefResolver",
    "_fetch_crossref_data",
]
