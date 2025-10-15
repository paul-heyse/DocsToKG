"""
OpenAIRE Resolver Provider

This module integrates with the OpenAIRE research infrastructure to discover
open-access artefacts linked to DOI-indexed works.

Key Features:
- Recursive traversal of OpenAIRE JSON payloads to locate URL candidates.
- Resilient handling of cases where responses arrive as either JSON or text.
- Deduplication of candidate URLs before yielding resolver results.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers.openaire import OpenAireResolver

    resolver = OpenAireResolver()
    results = list(resolver.iter_urls(session, config, artifact))
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Iterable, List

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import dedupe, normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


def _collect_candidate_urls(node: object, results: List[str]) -> None:
    """Recursively collect HTTP(S) URLs from nested OpenAIRE response payloads.

    Args:
        node: Arbitrary node from the OpenAIRE response payload.
        results: Mutable list to append discovered URL strings into.
    """

    if isinstance(node, dict):
        for value in node.values():
            _collect_candidate_urls(value, results)
    elif isinstance(node, list):
        for item in node:
            _collect_candidate_urls(item, results)
    elif isinstance(node, str):
        if node.lower().startswith("http"):
            results.append(node)


LOGGER = logging.getLogger(__name__)


class OpenAireResolver:
    """Resolve URLs using the OpenAIRE API.

    Attributes:
        name: Resolver identifier used when interacting with the pipeline.

    Examples:
        >>> resolver = OpenAireResolver()
        >>> resolver.name
        'openaire'
    """

    name = "openaire"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has a DOI.

        Args:
            config: Resolver configuration controlling OpenAIRE behaviour.
            artifact: Work artifact containing metadata such as DOI.

        Returns:
            Boolean indicating whether an OpenAIRE lookup should be attempted.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs discovered via OpenAIRE search.

        Args:
            session: HTTP session available for outbound requests.
            config: Resolver configuration with polite headers and timeouts.
            artifact: Work artifact describing the item being resolved.

        Returns:
            Iterable of resolver results representing discovered URLs.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://api.openaire.eu/search/publications",
                params={"doi": doi},
                headers=config.polite_headers,
                timeout=config.get_timeout(self.name),
            )
        except requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except requests.RequestException as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"error": str(exc)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error in OpenAIRE resolver")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        if resp.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=resp.status_code,
                metadata={"error_detail": f"OpenAIRE API returned {resp.status_code}"},
            )
            return
        try:
            data = resp.json()
        except ValueError:
            try:
                data = json.loads(resp.text)
            except ValueError as json_err:
                preview = resp.text[:200] if hasattr(resp, "text") else ""
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
                    metadata={"error_detail": str(json_err), "content_preview": preview},
                )
                return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected JSON error in OpenAIRE resolver")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        results: List[str] = []
        _collect_candidate_urls(data, results)
        for url in dedupe(results):
            if url.lower().endswith(".pdf"):
                yield ResolverResult(url=url, metadata={"source": "openaire"})


__all__ = ["OpenAireResolver"]
