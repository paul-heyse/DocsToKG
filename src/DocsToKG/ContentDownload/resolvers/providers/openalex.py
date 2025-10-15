"""OpenAlex direct URL resolver (position 0 in pipeline)."""

from __future__ import annotations

from typing import Iterable

import requests

from ..types import Resolver, ResolverConfig, ResolverResult
from DocsToKG.ContentDownload.utils import dedupe


class OpenAlexResolver:
    """Resolver for PDF URLs directly provided by OpenAlex metadata."""

    name = "openalex"

    def is_enabled(self, config: ResolverConfig, artifact) -> bool:  # noqa: D401
        """Enable when artifact has pdf_urls or open_access_url."""

        return bool(artifact.pdf_urls or artifact.open_access_url)

    def iter_urls(
        self,
        session: requests.Session,  # noqa: D401 - interface defined by protocol
        config: ResolverConfig,
        artifact,
    ) -> Iterable[ResolverResult]:
        """Yield all PDF URLs from OpenAlex work metadata."""

        candidates = list(artifact.pdf_urls)
        if artifact.open_access_url:
            candidates.append(artifact.open_access_url)

        for url in dedupe(candidates):
            if not url:
                continue
            yield ResolverResult(
                url=url,
                metadata={"source": "openalex_metadata"},
            )


__all__ = ["OpenAlexResolver"]

