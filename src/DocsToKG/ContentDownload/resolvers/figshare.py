# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.figshare",
#   "purpose": "Figshare resolver implementation",
#   "sections": [
#     {
#       "id": "figshare-resolver",
#       "name": "FigshareResolver",
#       "anchor": "class-figshareresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the Figshare repository API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, List, Optional

import requests as _requests

from DocsToKG.ContentDownload.core import normalize_doi

from .base import ApiResolverBase, ResolverEvent, ResolverEventReason, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


class FigshareResolver(ApiResolverBase):
    """Resolve Figshare repository metadata into download URLs."""

    name = "figshare"
    api_display_name = "Figshare"

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

        extra_headers = {"Content-Type": "application/json"}
        data, error = self._request_json(
            session,
            "POST",
            "https://api.figshare.com/v2/articles/search",
            config=config,
            json={"search_for": f':doi: "{doi}"', "page": 1, "page_size": 3},
            headers=extra_headers,
        )
        if error:
            yield error
            return

        if isinstance(data, list):
            articles: List[dict] = data
        else:
            LOGGER.warning(
                "Figshare API returned non-list articles payload: %s",
                type(data).__name__ if data is not None else "None",
            )
            articles = []

        for article in articles:
            if not isinstance(article, dict):
                LOGGER.warning("Skipping malformed Figshare article: %r", article)
                continue
            files = article.get("files", []) or []
            if not isinstance(files, list):
                LOGGER.warning("Skipping Figshare article with invalid files payload: %r", files)
                continue
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    LOGGER.warning("Skipping non-dict Figshare file entry: %r", file_entry)
                    continue
                filename = (file_entry.get("name") or "").lower()
                download_url: Optional[str] = file_entry.get("download_url")

                if filename.endswith(".pdf") and download_url:
                    yield ResolverResult(
                        url=download_url,
                        metadata={
                            "source": "figshare",
                            "article_id": article.get("id"),
                            "filename": file_entry.get("name"),
                        },
                    )
