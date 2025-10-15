"""
Unpaywall Resolver Provider

This module interfaces with the Unpaywall API to discover open-access PDFs for
scholarly works that expose DOI metadata.

Key Features:
- Configurable polite headers and email registration for API compliance.
- Fallback path that leverages memoised requests when no session is supplied.
- Deduplication of candidate URLs across best and alternate OA locations.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers.unpaywall import UnpaywallResolver

    resolver = UnpaywallResolver()
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

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


LOGGER = logging.getLogger(__name__)


def _headers_cache_key(headers: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """Create a hashable cache key for polite header dictionaries.

    Args:
        headers: Mapping of header names to values used in Unpaywall requests.

    Returns:
        Tuple of key/value pairs sorted to ensure deterministic hashing.
    """

    return tuple(sorted((headers or {}).items()))


@lru_cache(maxsize=1000)
def _fetch_unpaywall_data(
    doi: str,
    email: Optional[str],
    timeout: float,
    headers_key: Tuple[Tuple[str, str], ...],
) -> Dict[str, Any]:
    """Fetch Unpaywall metadata for ``doi`` using polite caching.

    Args:
        doi: DOI identifier to query against the Unpaywall API.
        email: Registered contact email required by the Unpaywall terms.
        timeout: Request timeout in seconds.
        headers_key: Hashable representation of polite headers for cache lookups.

    Returns:
        Parsed JSON payload describing open-access locations for the DOI.

    Raises:
        requests.HTTPError: If the Unpaywall API returns a non-success status code.
    """

    headers = dict(headers_key)
    response = requests.get(
        f"https://api.unpaywall.org/v2/{quote(doi)}",
        params={"email": email} if email else None,
        timeout=timeout,
        headers=headers,
    )
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()


class UnpaywallResolver:
    """Resolve PDFs via the Unpaywall API.

    Attributes:
        name: Resolver identifier announced to the pipeline.

    Examples:
        >>> resolver = UnpaywallResolver()
        >>> resolver.name
        'unpaywall'
    """

    name = "unpaywall"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when Unpaywall is configured and the work has a DOI.

        Args:
            config: Resolver configuration containing Unpaywall credentials.
            artifact: Work artifact potentially containing a DOI identifier.

        Returns:
            Boolean indicating whether the Unpaywall resolver should run.
        """

        return bool(config.unpaywall_email and artifact.doi)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate PDF URLs discovered via the Unpaywall API.

        Args:
            session: HTTP session used to issue API requests.
            config: Resolver configuration defining headers, timeouts, and email.
            artifact: Work artifact describing the scholarly record to resolve.

        Returns:
            Iterable of resolver results with download URLs or status events.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(
                url=None,
                event="skipped",
                event_reason="no-doi",
            )
            return
        endpoint = f"https://api.unpaywall.org/v2/{quote(doi)}"
        headers = dict(config.polite_headers)
        params = {"email": config.unpaywall_email} if config.unpaywall_email else None
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
            except Exception as exc:  # pragma: no cover - safety
                LOGGER.exception("Unexpected error in Unpaywall resolver session path")
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
                    metadata={"error_detail": f"Unpaywall returned {status}"},
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
                data = _fetch_unpaywall_data(
                    doi,
                    config.unpaywall_email,
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
                    metadata={"error_detail": f"Unpaywall HTTPError: {status}"},
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
                LOGGER.exception("Unexpected cached request error in Unpaywall resolver")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                )
                return

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        best = (data or {}).get("best_oa_location") or {}
        url = best.get("url_for_pdf")
        if url:
            candidates.append((url, {"source": "best_oa_location"}))

        for loc in (data or {}).get("oa_locations", []) or []:
            if not isinstance(loc, dict):
                continue
            url = loc.get("url_for_pdf")
            if url:
                candidates.append((url, {"source": "oa_location"}))

        unique_urls = dedupe([candidate_url for candidate_url, _ in candidates])
        for unique_url in unique_urls:
            for candidate_url, metadata in candidates:
                if candidate_url == unique_url:
                    yield ResolverResult(url=unique_url, metadata=metadata)
                    break


__all__ = [
    "UnpaywallResolver",
    "_fetch_unpaywall_data",
    "_headers_cache_key",
]
