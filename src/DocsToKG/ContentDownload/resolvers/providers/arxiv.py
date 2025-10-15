"""arXiv preprint resolver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import requests

from ..types import Resolver, ResolverConfig, ResolverResult
from DocsToKG.ContentDownload.utils import strip_prefix

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


class ArxivResolver:
    """Resolve arXiv preprints using arXiv identifier lookups."""

    name = "arxiv"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has an arXiv identifier."""

        return bool(artifact.arxiv_id)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate arXiv download URLs."""

        arxiv_id = artifact.arxiv_id
        if not arxiv_id:
            return []
        arxiv_id = strip_prefix(arxiv_id, "arxiv:")
        return [
            ResolverResult(
                url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                metadata={"identifier": arxiv_id},
            )
        ]


__all__ = ["ArxivResolver"]
