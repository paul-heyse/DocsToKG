"""Semantic Scholar Graph API resolver."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple
from urllib.parse import quote

import requests

from ..types import Resolver, ResolverConfig, ResolverResult
from .unpaywall import _headers_cache_key
from DocsToKG.ContentDownload.utils import normalize_doi

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


@lru_cache(maxsize=1000)
def _fetch_semantic_scholar_data(
    doi: str,
    api_key: Optional[str],
    timeout: float,
    headers_key: Tuple[Tuple[str, str], ...],
) -> Dict[str, Any]:
        headers = dict(headers_key)
        if api_key:
            headers = dict(headers)
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
    """Resolve PDFs via the Semantic Scholar Graph API."""

    name = "semantic_scholar"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has a DOI suitable for lookup."""

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs discovered via Semantic Scholar."""

        doi = normalize_doi(artifact.doi)
        if not doi:
            return []
        try:
            data = _fetch_semantic_scholar_data(
                doi,
                config.semantic_scholar_api_key,
                config.get_timeout(self.name),
                _headers_cache_key(config.polite_headers),
            )
        except requests.HTTPError:
            return []
        except requests.RequestException:
            return []
        except ValueError:
            return []

        open_access = (data.get("openAccessPdf") or {}) if isinstance(data, dict) else {}
        url = open_access.get("url") if isinstance(open_access, dict) else None
        if url:
            return [ResolverResult(url=url, metadata={"source": "semantic-scholar"})]
        return []


__all__ = [
    "SemanticScholarResolver",
    "_fetch_semantic_scholar_data",
]
