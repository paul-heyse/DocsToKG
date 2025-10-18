# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.doaj",
#   "purpose": "DOAJ resolver implementation",
#   "sections": [
#     {
#       "id": "doaj-resolver",
#       "name": "DoajResolver",
#       "anchor": "class-doajresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the DOAJ API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List

import requests as _requests

from DocsToKG.ContentDownload.core import dedupe, normalize_doi

from .base import ApiResolverBase, ResolverEvent, ResolverEventReason, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


class DoajResolver(ApiResolverBase):
    """Resolve Open Access links using the DOAJ API."""

    name = "doaj"
    api_display_name = "DOAJ"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_DOI,
            )
            return
        extra_headers: Dict[str, str] = {}
        if config.doaj_api_key:
            extra_headers["X-API-KEY"] = config.doaj_api_key
        data, error = self._request_json(
            session,
            "GET",
            "https://doaj.org/api/v2/search/articles/",
            config=config,
            params={"pageSize": 3, "q": f'doi:"{doi}"'},
            headers=extra_headers,
        )
        if error:
            yield error
            return
        candidates: List[str] = []
        for result in data.get("results", []) or []:
            bibjson = (result or {}).get("bibjson", {})
            for link in bibjson.get("link", []) or []:
                if not isinstance(link, dict):
                    continue
                url = link.get("url")
                if url and url.lower().endswith(".pdf"):
                    candidates.append(url)
        for url in dedupe(candidates):
            yield ResolverResult(url=url, metadata={"source": "doaj"})
