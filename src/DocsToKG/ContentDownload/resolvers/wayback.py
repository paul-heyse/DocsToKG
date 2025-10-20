# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.wayback",
#   "purpose": "Wayback Machine resolver with CDX-first discovery algorithm",
#   "sections": [
#     {
#       "id": "waybackresolver",
#       "name": "WaybackResolver",
#       "anchor": "class-waybackresolver",
#       "kind": "class"
#     },
#     {
#       "id": "wayback-discovery",
#       "name": "_discover_snapshots",
#       "anchor": "function-wayback-discovery",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-availability",
#       "name": "_check_availability",
#       "anchor": "function-wayback-availability",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-cdx",
#       "name": "_query_cdx",
#       "anchor": "function-wayback-cdx",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-html-parse",
#       "name": "_parse_html_for_pdf",
#       "anchor": "function-wayback-html-parse",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-verify-pdf",
#       "name": "_verify_pdf_snapshot",
#       "anchor": "function-wayback-verify-pdf",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver that queries the Internet Archive Wayback Machine with CDX-first discovery."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx

from DocsToKG.ContentDownload.networking import request_with_retries
from DocsToKG.ContentDownload.urls import canonical_for_index

from .base import (
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    find_pdf_via_anchor,
    find_pdf_via_link,
    find_pdf_via_meta,
)

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


class WaybackResolver(RegisteredResolver):
    """Fallback resolver that queries the Internet Archive Wayback Machine with CDX-first discovery."""

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
        """Query the Wayback Machine for archived versions of failed URLs using CDX-first discovery.

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

        # Get Wayback-specific configuration
        wayback_config = getattr(config, "wayback_config", {})
        year_window = wayback_config.get("year_window", 2)
        max_snapshots = wayback_config.get("max_snapshots", 8)
        min_pdf_bytes = wayback_config.get("min_pdf_bytes", 4096)
        html_parse_enabled = wayback_config.get("html_parse", True)

        for original_url in artifact.failed_pdf_urls:
            try:
                canonical_url = canonical_for_index(original_url)
                publication_year = getattr(artifact, "publication_year", None)

                # Discover snapshots using CDX-first approach
                snapshot_url, metadata = self._discover_snapshots(
                    client,
                    config,
                    original_url,
                    canonical_url,
                    publication_year,
                    year_window,
                    max_snapshots,
                    min_pdf_bytes,
                    html_parse_enabled,
                )

                if snapshot_url:
                    yield ResolverResult(
                        url=snapshot_url,
                        metadata={
                            "original": original_url,
                            "canonical": canonical_url,
                            "source": "wayback",
                            **metadata,
                        },
                    )
                else:
                    yield ResolverResult(
                        url=None,
                        event=ResolverEvent.SKIPPED,
                        event_reason=ResolverEventReason.NO_FAILED_URLS,
                        metadata={
                            "original": original_url,
                            "canonical": canonical_url,
                            "reason": "no_snapshot",
                        },
                    )

            except Exception as exc:
                LOGGER.exception("Unexpected Wayback resolver error for %s", original_url)
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                    metadata={
                        "original": original_url,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

    def _discover_snapshots(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        original_url: str,
        canonical_url: str,
        publication_year: Optional[int],
        year_window: int,
        max_snapshots: int,
        min_pdf_bytes: int,
        html_parse_enabled: bool,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Discover the best PDF snapshot using CDX-first approach.

        Returns:
            Tuple of (snapshot_url, metadata) or (None, metadata) if no suitable snapshot found.
        """
        metadata = {"discovery_method": "none"}

        # Step 1: Try Availability API as fast-path
        availability_url, availability_metadata = self._check_availability(
            client, config, original_url
        )
        metadata.update(availability_metadata)

        if availability_url:
            # Verify the availability result is a PDF
            if self._verify_pdf_snapshot(client, config, availability_url, min_pdf_bytes):
                metadata["discovery_method"] = "availability"
                return availability_url, metadata

        # Step 2: Use CDX API for comprehensive search
        cdx_snapshots = self._query_cdx(
            client, config, original_url, publication_year, year_window, max_snapshots
        )

        # Step 3: Evaluate CDX snapshots
        for snapshot in cdx_snapshots:
            snapshot_url = snapshot.get("archive_url")
            if not snapshot_url:
                continue

            # Check if it's a direct PDF snapshot
            if snapshot.get("mimetype") == "application/pdf" and snapshot.get("statuscode") == 200:
                if self._verify_pdf_snapshot(client, config, snapshot_url, min_pdf_bytes):
                    metadata.update(
                        {
                            "discovery_method": "cdx_pdf_direct",
                            "memento_ts": str(snapshot.get("timestamp", "")),
                            "statuscode": str(snapshot.get("statuscode", "")),
                            "mimetype": str(snapshot.get("mimetype", "")),
                        }
                    )
                    return snapshot_url, metadata

            # Step 4: If HTML parse is enabled, try parsing archived HTML for PDF links
            if html_parse_enabled and snapshot.get("mimetype") in [
                "text/html",
                "application/xhtml+xml",
            ]:
                pdf_url = self._parse_html_for_pdf(client, config, snapshot_url)
                if pdf_url and self._verify_pdf_snapshot(client, config, pdf_url, min_pdf_bytes):
                    metadata.update(
                        {
                            "discovery_method": "cdx_html_parse",
                            "html_snapshot_ts": str(snapshot.get("timestamp", "")),
                            "discovered_pdf_url": pdf_url,
                        }
                    )
                    return pdf_url, metadata

        metadata["discovery_method"] = "none"
        return None, metadata

    def _check_availability(
        self, client: httpx.Client, config: "ResolverConfig", url: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Check Wayback Availability API for a quick snapshot.

        Returns:
            Tuple of (snapshot_url, metadata) or (None, metadata) if not available.
        """
        metadata: Dict[str, Any] = {"availability_checked": True, "availability_available": False}

        try:
            resp = request_with_retries(
                client,
                "get",
                "https://archive.org/wayback/available",
                role="metadata",
                params={"url": url},
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            resp.raise_for_status()

            data = resp.json()
            closest = (data.get("archived_snapshots") or {}).get("closest") or {}

            if closest.get("available") and closest.get("url"):
                metadata.update(
                    {
                        "availability_available": True,
                        "availability_timestamp": closest.get("timestamp"),
                        "availability_status": closest.get("status"),
                    }
                )
                return closest["url"], metadata

        except Exception as exc:
            LOGGER.debug("Availability check failed for %s: %s", url, exc)
            metadata["availability_error"] = str(exc)

        finally:
            if "resp" in locals():
                resp.close()

        return None, metadata

    def _query_cdx(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        url: str,
        publication_year: Optional[int],
        year_window: int,
        max_snapshots: int,
    ) -> List[Dict[str, Any]]:
        """Query CDX API for snapshots.

        Returns:
            List of snapshot dictionaries sorted by relevance.
        """
        snapshots: List[Dict[str, Any]] = []

        try:
            # Build CDX query parameters
            params = {
                "url": url,
                "output": "json",
                "limit": max_snapshots,
                "filter": "statuscode:200",
            }

            # Add year window if publication year is available
            if publication_year:
                start_year = publication_year - year_window
                end_year = publication_year + year_window
                params["from"] = f"{start_year}0101000000"
                params["to"] = f"{end_year}1231235959"

            resp = request_with_retries(
                client,
                "get",
                "https://web.archive.org/cdx/search/cdx",
                role="metadata",
                params=params,
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            resp.raise_for_status()

            # Parse CDX response (JSON format)
            data = resp.json()
            if not data or len(data) < 2:  # Empty or header-only response
                return snapshots

            # CDX JSON format: first row is headers, rest are data
            headers = data[0]
            for row in data[1:]:
                snapshot = dict(zip(headers, row))
                snapshots.append(snapshot)

            # Sort by timestamp (newest first) and filter for relevant content types
            snapshots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        except Exception as exc:
            LOGGER.debug("CDX query failed for %s: %s", url, exc)

        finally:
            if "resp" in locals():
                resp.close()

        return snapshots

    def _parse_html_for_pdf(
        self, client: httpx.Client, config: "ResolverConfig", html_url: str
    ) -> Optional[str]:
        """Parse archived HTML page to find PDF links.

        Returns:
            PDF URL if found, None otherwise.
        """
        try:
            resp = request_with_retries(
                client,
                "get",
                html_url,
                role="landing",
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            resp.raise_for_status()

            # Check if response is HTML
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None

            html_content = resp.text

            # Try to import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                LOGGER.debug("BeautifulSoup not available for HTML parsing")
                return None

            soup = BeautifulSoup(html_content, "html.parser")

            # Try different PDF discovery methods
            pdf_url = find_pdf_via_meta(soup, html_url)
            if pdf_url:
                return pdf_url

            pdf_url = find_pdf_via_link(soup, html_url)
            if pdf_url:
                return pdf_url

            pdf_url = find_pdf_via_anchor(soup, html_url)
            if pdf_url:
                return pdf_url

        except Exception as exc:
            LOGGER.debug("HTML parsing failed for %s: %s", html_url, exc)

        finally:
            if "resp" in locals():
                resp.close()

        return None

    def _verify_pdf_snapshot(
        self, client: httpx.Client, config: "ResolverConfig", url: str, min_bytes: int
    ) -> bool:
        """Verify that a snapshot URL points to a valid PDF.

        Returns:
            True if the snapshot is a valid PDF, False otherwise.
        """
        try:
            # Use HEAD request to check content type and size
            resp = request_with_retries(
                client,
                "head",
                url,
                role="artifact",
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            resp.raise_for_status()

            # Check content type
            content_type = resp.headers.get("content-type", "").lower()
            if "application/pdf" not in content_type:
                return False

            # Check content length
            content_length = resp.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    if size < min_bytes:
                        return False
                except ValueError:
                    pass

            # For small files, do a GET to check PDF signature
            if content_length and int(content_length) < min_bytes * 2:
                try:
                    get_resp = request_with_retries(
                        client,
                        "get",
                        url,
                        role="artifact",
                        timeout=config.get_timeout(self.name),
                        headers=config.polite_headers,
                        retry_after_cap=config.retry_after_cap,
                    )
                    get_resp.raise_for_status()

                    # Check first few bytes for PDF signature
                    chunk = get_resp.content[:8]
                    if not chunk.startswith(b"%PDF-"):
                        return False

                finally:
                    if "get_resp" in locals():
                        get_resp.close()

            return True

        except Exception as exc:
            LOGGER.debug("PDF verification failed for %s: %s", url, exc)
            return False

        finally:
            if "resp" in locals():
                resp.close()
