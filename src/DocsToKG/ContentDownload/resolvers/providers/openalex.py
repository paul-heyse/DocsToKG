"""OpenAlex direct URL resolver (position 0 in pipeline)."""

from __future__ import annotations

from typing import Iterable

from DocsToKG.ContentDownload.utils import dedupe

from ..types import ResolverConfig, ResolverResult


class OpenAlexResolver:
    """Resolve OpenAlex work metadata into candidate download URLs.

    Attributes:
        name: Resolver identifier advertised to the pipeline.

    Examples:
        >>> resolver = OpenAlexResolver()
        >>> resolver.name
        'openalex'
    """

    name = "openalex"

    def is_enabled(self, config: ResolverConfig, artifact) -> bool:
        """Return True when the OpenAlex artifact exposes accessible PDF URLs.

        Args:
            config: Resolver configuration (unused but required for interface parity).
            artifact: OpenAlex metadata object containing URL fields.

        Returns:
            bool: ``True`` when at least one candidate PDF URL is available.
        """

        return bool(artifact.pdf_urls or artifact.open_access_url)

    def iter_urls(
        self,
        session: requests.Session,  # noqa: D401 - interface defined by protocol
        config: ResolverConfig,
        artifact,
    ) -> Iterable[ResolverResult]:
        """Yield resolver results for each accessible PDF URL in the artifact.

        Args:
            session: Requests session forwarded for interface compatibility.
            config: Resolver configuration (unused but accepted for uniform signature).
            artifact: OpenAlex metadata object providing URL candidates.

        Returns:
            Iterable[ResolverResult]: Iterator producing resolver results for each unique URL.

        Raises:
            None
        """

        candidates = list(dedupe(artifact.pdf_urls))
        if artifact.open_access_url:
            candidates.append(artifact.open_access_url)

        if not candidates:
            yield ResolverResult(
                url=None,
                event="skipped",
                event_reason="no-openalex-urls",
            )
            return

        for url in dedupe(candidates):
            if not url:
                continue
            yield ResolverResult(
                url=url,
                metadata={"source": "openalex_metadata"},
            )


__all__ = ["OpenAlexResolver"]
