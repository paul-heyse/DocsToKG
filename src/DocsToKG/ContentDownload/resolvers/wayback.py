# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.wayback",
#   "purpose": "Wayback Machine resolver",
#   "sections": [
#     {
#       "id": "waybackresolver",
#       "name": "WaybackResolver",
#       "anchor": "class-waybackresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver that queries the Internet Archive Wayback Machine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Optional

import httpx

from DocsToKG.ContentDownload.networking import request_with_retries

from .base import (
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
)

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


class WaybackResolver(RegisteredResolver):
    """Fallback resolver that queries the Internet Archive Wayback Machine."""

    name = "wayback"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        """Return ``True`` when prior resolver attempts have failed.

        Args:
            config: Resolver configuration (unused for enablement).
            artifact: Work record containing failed PDF URLs.

        Returns:
            bool: Whether the Wayback resolver should run.
        """
        return bool(artifact.failed_pdf_urls)

    def iter_urls(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Query the Wayback Machine for archived versions of failed URLs.

        Args:
            client: HTTPX client for HTTP calls.
            config: Resolver configuration providing timeouts and headers.
            artifact: Work metadata listing failed PDF URLs to retry.

        Yields:
            ResolverResult: Archived download URLs or diagnostic events.
        """
        if not artifact.failed_pdf_urls:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_FAILED_URLS,
            )
            return

        for original in artifact.failed_pdf_urls:
            resp: Optional[httpx.Response] = None
            try:
                resp = request_with_retries(
                    client,
                    "get",
                    "https://archive.org/wayback/available",
                    params={"url": original},
                    timeout=config.get_timeout(self.name),
                    headers=config.polite_headers,
                    retry_after_cap=config.retry_after_cap,
                )
                resp.raise_for_status()
            except httpx.TimeoutException as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.TIMEOUT,
                    metadata={
                        "original": original,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                continue
            except httpx.TransportError as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.CONNECTION_ERROR,
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.HTTP_ERROR,
                    http_status=status,
                    metadata={
                        "original": original,
                        "error_detail": str(exc),
                    },
                )
                continue
            except httpx.RequestError as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.REQUEST_ERROR,
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Unexpected Wayback resolver error")
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                    metadata={
                        "original": original,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            try:
                try:
                    data = resp.json()
                except ValueError as json_err:
                    yield ResolverResult(
                        url=None,
                        event=ResolverEvent.ERROR,
                        event_reason=ResolverEventReason.JSON_ERROR,
                        metadata={"original": original, "error_detail": str(json_err)},
                    )
                    continue
                closest = (data.get("archived_snapshots") or {}).get("closest") or {}
                if closest.get("available") and closest.get("url"):
                    metadata = {"original": original}
                    if closest.get("timestamp"):
                        metadata["timestamp"] = closest["timestamp"]
                    yield ResolverResult(url=closest["url"], metadata=metadata)
            finally:
                if resp is not None:
                    resp.close()
