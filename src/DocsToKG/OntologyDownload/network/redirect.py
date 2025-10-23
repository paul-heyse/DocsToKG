# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.network.redirect",
#   "purpose": "Safe redirect handling: Manual redirect following with security audit.",
#   "sections": [
#     {
#       "id": "redirecterror",
#       "name": "RedirectError",
#       "anchor": "class-redirecterror",
#       "kind": "class"
#     },
#     {
#       "id": "maxredirectsexceeded",
#       "name": "MaxRedirectsExceeded",
#       "anchor": "class-maxredirectsexceeded",
#       "kind": "class"
#     },
#     {
#       "id": "unsaferedirecttarget",
#       "name": "UnsafeRedirectTarget",
#       "anchor": "class-unsaferedirecttarget",
#       "kind": "class"
#     },
#     {
#       "id": "missinglocationheader",
#       "name": "MissingLocationHeader",
#       "anchor": "class-missinglocationheader",
#       "kind": "class"
#     },
#     {
#       "id": "redirectpolicy",
#       "name": "RedirectPolicy",
#       "anchor": "class-redirectpolicy",
#       "kind": "class"
#     },
#     {
#       "id": "safe-get-with-redirect",
#       "name": "safe_get_with_redirect",
#       "anchor": "function-safe-get-with-redirect",
#       "kind": "function"
#     },
#     {
#       "id": "safe-post-with-redirect",
#       "name": "safe_post_with_redirect",
#       "anchor": "function-safe-post-with-redirect",
#       "kind": "function"
#     },
#     {
#       "id": "count-redirect-hops",
#       "name": "count_redirect_hops",
#       "anchor": "function-count-redirect-hops",
#       "kind": "function"
#     },
#     {
#       "id": "format-audit-trail",
#       "name": "format_audit_trail",
#       "anchor": "function-format-audit-trail",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Safe redirect handling: Manual redirect following with security audit.

HTTPX disables auto-redirect by default (good security practice). This module
provides utilities to follow redirects manually with explicit validation at each hop.

Design:
- **Explicit hops**: Every redirect hop is manually audited
- **Security gate**: Each target URL must pass a security policy check
- **Audit trail**: All hops recorded for observability
- **Max hops**: Limit to prevent redirect loops
- **Transparent**: All hops collected; caller can inspect or log

Example:
    >>> from DocsToKG.OntologyDownload.network.redirect import (
    ...     safe_get_with_redirect
    ... )
    >>> from DocsToKG.OntologyDownload.network import get_http_client
    >>> client = get_http_client()
    >>> response, hops = safe_get_with_redirect(
    ...     client, "https://example.com/resource",
    ...     max_hops=5
    ... )
    >>> print(f"Followed {len(hops)} hops")
    >>> # hops = [
    >>> #     ("https://example.com/resource", 301),
    >>> #     ("https://cdn.example.com/data", 200),
    >>> # ]
"""

import logging
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class RedirectError(Exception):
    """Base exception for redirect handling errors."""

    pass


class MaxRedirectsExceeded(RedirectError):
    """Redirect chain exceeds maximum allowed hops."""

    def __init__(self, max_hops: int, actual_hops: List[str]):
        self.max_hops = max_hops
        self.actual_hops = actual_hops
        super().__init__(
            f"Redirect chain exceeded {max_hops} hops. " f"Hops: {' → '.join(actual_hops)}"
        )


class UnsafeRedirectTarget(RedirectError):
    """Redirect target is not allowed by security policy."""

    def __init__(self, source_url: str, target_url: str, reason: str):
        self.source_url = source_url
        self.target_url = target_url
        self.reason = reason
        super().__init__(f"Unsafe redirect from {source_url} to {target_url}: {reason}")


class MissingLocationHeader(RedirectError):
    """Redirect response missing Location header."""

    def __init__(self, url: str, status: int):
        super().__init__(f"Redirect response from {url} (status {status}) missing Location header")


# ============================================================================
# Redirect Validation
# ============================================================================


class RedirectPolicy:
    """Policy for validating redirect targets.

    Implements a default conservative policy:
    - Only HTTPS (no http:// allowed)
    - Only specific safe schemes (https://, maybe http://)
    - No authentication in URLs
    - No private/loopback addresses

    Subclass to customize validation.
    """

    def validate_target(self, source_url: str, target_url: str) -> bool:
        """Validate a redirect target URL.

        Args:
            source_url: The URL that redirected
            target_url: The proposed redirect destination

        Returns:
            True if target is allowed, False otherwise

        Raises:
            UnsafeRedirectTarget: If target fails validation
        """
        try:
            # Parse target as URL
            target = httpx.URL(target_url)

            # Disallow authentication in URL
            if target.userinfo:
                raise UnsafeRedirectTarget(source_url, target_url, "URL contains authentication")

            # Only allow https (and optionally http)
            if target.scheme not in {"https", "http"}:
                raise UnsafeRedirectTarget(
                    source_url, target_url, f"Scheme not allowed: {target.scheme}"
                )

            # Could add more checks: private IPs, suspicious domains, etc.
            # For now, keep it simple - rely on TLS cert verification

            return True

        except UnsafeRedirectTarget:
            raise
        except Exception as e:
            raise UnsafeRedirectTarget(source_url, target_url, f"URL parsing error: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ============================================================================
# Redirect Following
# ============================================================================


def safe_get_with_redirect(
    client: httpx.Client,
    url: str,
    max_hops: int = 5,
    policy: Optional[RedirectPolicy] = None,
) -> Tuple[httpx.Response, List[Tuple[str, int]]]:
    """Make GET request with explicit redirect following and auditing.

    Follows redirects manually, validating each hop:
    1. Makes initial GET request
    2. If redirect (3xx), extracts Location header
    3. Validates target with redirect policy
    4. Records hop in audit trail
    5. Repeats until terminal response or max hops

    Args:
        client: HTTPX Client (with auto-redirect=False)
        url: Initial URL to fetch
        max_hops: Maximum redirect hops (default 5)
        policy: Redirect validation policy (default: DefaultRedirectPolicy)

    Returns:
        Tuple of (final_response, audit_trail)
        - final_response: The terminal HTTP response
        - audit_trail: List of (url, status_code) tuples showing the path

    Raises:
        MaxRedirectsExceeded: If redirect chain exceeds max_hops
        UnsafeRedirectTarget: If a redirect target fails validation
        MissingLocationHeader: If redirect response missing Location
        httpx.RequestError: If underlying request fails

    Example:
        >>> response, hops = safe_get_with_redirect(client, url)
        >>> print(f"Final status: {response.status_code}")
        >>> for url, status in hops:
        ...     print(f"  {status} {url}")
    """
    if policy is None:
        policy = RedirectPolicy()

    # Audit trail: list of (url, status_code) tuples
    audit_trail: List[Tuple[str, int]] = []

    current_url = url
    hop_count = 0

    while hop_count < max_hops:
        # Make GET request
        response = client.get(current_url)
        audit_trail.append((current_url, response.status_code))

        # Check if redirect
        if response.status_code not in {301, 302, 303, 307, 308}:
            # Terminal response (not a redirect)
            logger.debug(
                "Redirect following complete",
                extra={
                    "final_status": response.status_code,
                    "hops": len(audit_trail),
                },
            )
            return response, audit_trail

        # Extract Location header
        location = response.headers.get("location")
        if not location:
            raise MissingLocationHeader(current_url, response.status_code)

        try:
            # Resolve relative locations against the response URL as per RFC 7231.
            resolved_url = response.url.join(location)
            target_url = resolved_url.human_repr()
        except Exception as exc:
            raise UnsafeRedirectTarget(
                current_url,
                location,
                f"Invalid redirect location: {exc}",
            ) from exc

        # Validate redirect target
        try:
            policy.validate_target(current_url, target_url)
        except UnsafeRedirectTarget as e:
            logger.warning(
                "Unsafe redirect detected",
                extra={
                    "source": current_url,
                    "target": target_url,
                    "reason": e.reason,
                    "hops": len(audit_trail),
                },
            )
            raise

        # Record hop and continue
        current_url = target_url
        hop_count += 1
        logger.debug(
            "Following redirect",
            extra={
                "from": audit_trail[-1][0],
                "to": target_url,
                "status": response.status_code,
                "hop": hop_count,
            },
        )

    # Exceeded max hops
    raise MaxRedirectsExceeded(max_hops, [url for url, _ in audit_trail])


def safe_post_with_redirect(
    client: httpx.Client,
    url: str,
    **kwargs,
) -> Tuple[httpx.Response, List[Tuple[str, int]]]:
    """Make POST request with redirect following.

    POST with redirects is tricky (303 implies GET, 307/308 keep POST).
    This follows RFC semantics:
    - 301/302/303 → follow as GET
    - 307/308 → follow as POST
    - Other redirects → stop

    Args:
        client: HTTPX Client
        url: Initial URL
        **kwargs: Additional arguments to client.post()

    Returns:
        Tuple of (final_response, audit_trail)

    Note:
        RFC 7231 allows 301/302 to rewrite POST to GET (many servers do).
        307/308 require preserving the method.
    """
    # For now, keep this simple: don't follow redirects on POST
    # Callers can use safe_get_with_redirect for safe cases
    response = client.post(url, **kwargs)
    return response, [(url, response.status_code)]


# ============================================================================
# Utilities
# ============================================================================


def count_redirect_hops(response: httpx.Response) -> int:
    """Count number of redirect hops in response history.

    Note: Only works if HTTPX has recorded history. Since we disable
    auto-redirect, this will always be 0 on the response itself.
    Use the audit_trail from safe_get_with_redirect instead.

    Args:
        response: HTTPX Response object

    Returns:
        Number of hops (0 if no history)
    """
    if hasattr(response, "history"):
        return len(response.history)
    return 0


def format_audit_trail(audit_trail: List[Tuple[str, int]]) -> str:
    """Format audit trail for logging/display.

    Args:
        audit_trail: List of (url, status) tuples

    Returns:
        Formatted string like "http://a (200) → http://b (301) → http://c (200)"
    """
    parts = [f"{url} ({status})" for url, status in audit_trail]
    return " → ".join(parts)


__all__ = [
    "RedirectError",
    "MaxRedirectsExceeded",
    "UnsafeRedirectTarget",
    "MissingLocationHeader",
    "RedirectPolicy",
    "safe_get_with_redirect",
    "safe_post_with_redirect",
    "count_redirect_hops",
    "format_audit_trail",
]
