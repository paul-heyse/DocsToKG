# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.probe",
#   "purpose": "GET-first probing with Range headers; audited redirects via URL gate",
#   "sections": [
#     {
#       "id": "proberesult",
#       "name": "ProbeResult",
#       "anchor": "class-proberesult",
#       "kind": "class"
#     },
#     {
#       "id": "probe-url",
#       "name": "probe_url",
#       "anchor": "function-probe-url",
#       "kind": "function"
#     },
#     {
#       "id": "extract-probe-result",
#       "name": "_extract_probe_result",
#       "anchor": "function-extract-probe-result",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""GET-first HTTP probing with Range headers and redirect auditing.

Implements smart probing strategy:
- GET with Range: bytes=0-0 for untrusted hosts (only 1 byte + headers)
- HEAD only for allow-listed trusted hosts
- All redirects validated through URL gate
- Comprehensive header extraction without full body download
"""

from __future__ import annotations

import logging
from typing import Callable, NamedTuple, Optional

import httpx

logger = logging.getLogger(__name__)

# Hosts known to support HEAD reliably
TRUSTS_HEAD = {"ebi.ac.uk", "data.bioontology.org", "www.w3.org"}


class ProbeResult(NamedTuple):
    """Result of probing a URL for metadata."""

    status: int
    """HTTP status code"""

    content_type: Optional[str]
    """Content-Type header value"""

    content_length: Optional[int]
    """Content length (from Content-Length or Content-Range)"""

    etag: Optional[str]
    """ETag header value"""

    last_modified: Optional[str]
    """Last-Modified header value"""


def probe_url(
    client: httpx.Client,
    url: str,
    *,
    validate_redirect: Optional[Callable] = None,
) -> ProbeResult:
    """Probe URL for metadata using GET-first strategy.

    Strategy:
    - Trusted hosts: use HEAD (single round-trip)
    - Untrusted hosts: use GET with Range: bytes=0-0 (1 byte + headers)
    - All redirects validated via URL gate if provided

    Args:
        client: HTTP client (should have redirect auditing hooks)
        url: URL to probe
        validate_redirect: Optional function to validate redirects (called by client hooks)

    Returns:
        ProbeResult with status, headers, and size metadata

    Raises:
        httpx.HTTPError: If probe fails or redirect validation fails
    """
    u = httpx.URL(url)
    host = u.host or ""

    # Use HEAD for trusted hosts only
    if any(host.endswith(trusted) for trusted in TRUSTS_HEAD):
        logger.debug(f"Probing {host} via HEAD (trusted)")
        resp = client.head(url, follow_redirects=False)
        return _extract_probe_result(resp, url)

    # Use GET with Range: bytes=0-0 for untrusted hosts
    logger.debug(f"Probing {host} via GET with Range: bytes=0-0 (untrusted)")
    resp = client.get(url, headers={"Range": "bytes=0-0"}, follow_redirects=False)
    try:
        return _extract_probe_result(resp, url)
    finally:
        resp.close()


def _extract_probe_result(response: httpx.Response, original_url: str) -> ProbeResult:
    """Extract metadata from probe response.

    Handles:
    - 206 Partial Content with Content-Range header
    - 200 OK with Content-Length
    - Standard headers (ETag, Last-Modified, Content-Type)

    Args:
        response: HTTP response from probe
        original_url: Original URL (for logging)

    Returns:
        ProbeResult with extracted metadata
    """
    status = response.status_code

    # Extract content length
    content_length: Optional[int] = None

    # 206 Partial Content: extract total size from Content-Range
    if status == 206:
        content_range = response.headers.get("Content-Range")
        if content_range and "/" in content_range:
            try:
                total_str = content_range.rsplit("/", 1)[-1]
                if total_str.isdigit():
                    content_length = int(total_str)
                    logger.debug(
                        f"Probe {original_url}: extracted size {content_length} from Content-Range"
                    )
            except (ValueError, IndexError):
                pass

    # Fallback to Content-Length for 200 or if Content-Range didn't work
    if not content_length and status == 200:
        clen = response.headers.get("Content-Length")
        if clen and str(clen).isdigit():
            try:
                content_length = int(clen)
                logger.debug(
                    f"Probe {original_url}: extracted size {content_length} from Content-Length"
                )
            except ValueError:
                pass

    return ProbeResult(
        status=status,
        content_type=response.headers.get("Content-Type"),
        content_length=content_length,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
    )
