"""Resolver that integrates with the Unpaywall API to locate open access PDFs."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests

from DocsToKG.ContentDownload.utils import dedupe, normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


def _headers_cache_key(headers: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((headers or {}).items()))


@lru_cache(maxsize=1000)
def _fetch_unpaywall_data(
    doi: str,
    email: Optional[str],
    timeout: float,
    headers_key: Tuple[Tuple[str, str], ...],
) -> Dict[str, Any]:
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
            except Exception as exc:  # pragma: no cover - safety
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"message": str(exc)},
                )
                return

            status = getattr(response, "status_code", 200)
            if status != 200:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=status,
                )
                return

            try:
                data = response.json()
            except Exception:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
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
                )
                return
            except requests.RequestException as exc:  # pragma: no cover - network errors
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"message": str(exc)},
                )
                return
            except ValueError:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
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
