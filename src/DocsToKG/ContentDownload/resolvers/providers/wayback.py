"""Internet Archive Wayback Machine fallback resolver."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

import requests

from DocsToKG.ContentDownload.http import request_with_retries

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


LOGGER = logging.getLogger(__name__)


class WaybackResolver:
    """Fallback resolver that queries the Internet Archive Wayback Machine.

    Attributes:
        name: Resolver identifier surfaced to the pipeline.

    Examples:
        >>> resolver = WaybackResolver()
        >>> resolver.name
        'wayback'
    """

    name = "wayback"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when previous resolvers have recorded failed URLs.

        Args:
            config: Resolver configuration governing Wayback usage.
            artifact: Work artifact containing details of failed downloads.

        Returns:
            Boolean indicating whether the Wayback resolver should run.
        """

        return bool(artifact.failed_pdf_urls)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield archived URLs from the Internet Archive when available.

        Args:
            session: HTTP session used to contact the Wayback API.
            config: Resolver configuration exposing headers and timeouts.
            artifact: Work artifact describing failed PDF URLs to recover.

        Returns:
            Iterable of resolver results referencing archived snapshots.
        """

        if not artifact.failed_pdf_urls:
            yield ResolverResult(url=None, event="skipped", event_reason="no-failed-urls")
            return

        for original in artifact.failed_pdf_urls:
            try:
                resp = request_with_retries(
                    session,
                    "get",
                    "https://archive.org/wayback/available",
                    params={"url": original},
                    timeout=config.get_timeout(self.name),
                    headers=config.polite_headers,
                )
            except requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={
                        "original": original,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                continue
            except requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except requests.RequestException as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected Wayback resolver error")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={
                        "original": original,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            if resp.status_code != 200:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=resp.status_code,
                    metadata={
                        "original": original,
                        "error_detail": f"Wayback returned {resp.status_code}",
                    },
                )
                continue
            try:
                data = resp.json()
            except ValueError as json_err:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
                    metadata={"original": original, "error_detail": str(json_err)},
                )
                continue
            closest = (data.get("archived_snapshots") or {}).get("closest") or {}
            if closest.get("available") and closest.get("url"):
                metadata = {"original": original}
                if closest.get("timestamp"):
                    metadata["timestamp"] = closest["timestamp"]
                yield ResolverResult(url=closest["url"], metadata=metadata)


__all__ = ["WaybackResolver"]
