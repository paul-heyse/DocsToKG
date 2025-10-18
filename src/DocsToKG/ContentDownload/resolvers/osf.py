# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.osf",
#   "purpose": "OSF resolver implementation",
#   "sections": [
#     {
#       "id": "osf-resolver",
#       "name": "OsfResolver",
#       "anchor": "class-osfresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the Open Science Framework API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List

import requests as _requests

from DocsToKG.ContentDownload.core import dedupe, normalize_doi

from .base import ApiResolverBase, ResolverEvent, ResolverEventReason, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


class OsfResolver(ApiResolverBase):
    """Resolve artefacts hosted on the Open Science Framework."""

    name = "osf"

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
        data, error = self._request_json(
            session,
            "GET",
            "https://api.osf.io/v2/preprints/",
            config=config,
            params={"filter[doi]": doi},
        )
        if error:
            yield error
            return
        urls: List[str] = []
        for item in data.get("data", []) or []:
            if not isinstance(item, dict):
                continue
            links = item.get("links") or {}
            download = links.get("download")
            if isinstance(download, str):
                urls.append(download)
            attributes = item.get("attributes") or {}
            primary = attributes.get("primary_file") or {}
            if isinstance(primary, dict):
                file_links = primary.get("links") or {}
                href = file_links.get("download")
                if isinstance(href, str):
                    urls.append(href)
        for url in dedupe(urls):
            yield ResolverResult(url=url, metadata={"source": "osf"})
