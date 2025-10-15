"""
arXiv Resolver Provider

This module implements a lightweight resolver that converts arXiv identifiers
into direct PDF download URLs. It participates in the modular content download
pipeline when arXiv metadata is available for a work.

Key Features:
- Normalises arXiv identifiers sourced from OpenAlex metadata.
- Emits deterministic PDF URLs suitable for direct download.
- Records skip events when metadata is incomplete.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers.arxiv import ArxivResolver

    resolver = ArxivResolver()
    list(resolver.iter_urls(session, config, artifact))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import requests

from DocsToKG.ContentDownload.utils import strip_prefix

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


class ArxivResolver:
    """Resolve arXiv preprints using arXiv identifier lookups.

    Attributes:
        name: Resolver identifier announced to the pipeline.

    Examples:
        >>> resolver = ArxivResolver()
        >>> resolver.name
        'arxiv'
    """

    name = "arxiv"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has an arXiv identifier.

        Args:
            config: Resolver configuration providing arXiv availability toggles.
            artifact: Work artifact potentially containing an arXiv identifier.

        Returns:
            Boolean indicating whether arXiv resolution should be attempted.
        """

        return bool(artifact.arxiv_id)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate arXiv download URLs.

        Args:
            session: HTTP session available to perform network requests.
            config: Resolver configuration describing timeouts and headers.
            artifact: Work artifact with arXiv metadata for resolution.

        Returns:
            Iterable of resolver results containing download URLs or metadata events.
        """

        arxiv_id = artifact.arxiv_id
        if not arxiv_id:
            yield ResolverResult(url=None, event="skipped", event_reason="no-arxiv-id")
            return
        arxiv_id = strip_prefix(arxiv_id, "arxiv:")
        yield ResolverResult(
            url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            metadata={"identifier": arxiv_id},
        )


__all__ = ["ArxivResolver"]
