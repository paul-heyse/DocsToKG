# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.core",
#   "purpose": "CORE API resolver implementation",
#   "sections": [
#     {
#       "id": "core-resolver",
#       "name": "CoreResolver",
#       "anchor": "class-coreresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the CORE API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable

import requests as _requests

from DocsToKG.ContentDownload.core import normalize_doi

from .base import ApiResolverBase, ResolverEvent, ResolverEventReason, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


class CoreResolver(ApiResolverBase):
    """Resolve PDFs using the CORE API."""

    name = "core"
    api_display_name = "CORE"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        return bool(config.core_api_key and artifact.doi)

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
        headers = {"Authorization": f"Bearer {config.core_api_key}"}
        data, error = self._request_json(
            session,
            "GET",
            "https://api.core.ac.uk/v3/search/works",
            config=config,
            params={"q": f'doi:"{doi}"', "page": 1, "pageSize": 3},
            headers=headers,
        )
        if error:
            yield error
            return
        results = (data.get("results") if isinstance(data, dict) else None) or []
        for hit in results:
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
