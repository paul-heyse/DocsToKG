"""Resolver that queries the Crossref metadata API to surface publisher-hosted PDFs."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests

from DocsToKG.ContentDownload.utils import dedupe, normalize_doi

from ..types import ResolverConfig, ResolverResult
from .unpaywall import _headers_cache_key

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


@lru_cache(maxsize=1000)
def _fetch_crossref_data(
    doi: str,
    mailto: Optional[str],
    timeout: float,
    headers_key: Tuple[Tuple[str, str], ...],
) -> Dict[str, Any]:
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
            except Exception as exc:  # pragma: no cover - unexpected session errors
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
