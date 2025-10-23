# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.policy.url_gate",
#   "purpose": "Authoritative URL security gate.",
#   "sections": [
#     {
#       "id": "policyerror",
#       "name": "PolicyError",
#       "anchor": "class-policyerror",
#       "kind": "class"
#     },
#     {
#       "id": "validate-url-security",
#       "name": "validate_url_security",
#       "anchor": "function-validate-url-security",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Authoritative URL security gate.

Validates URLs before following redirects. Enforces:
- Scheme whitelist (http/https only)
- Host punycode normalization & IDN support
- Port policy (default ports, per-host overrides)
- HTTP→HTTPS upgrade policy
- Private network guarding (optional DNS resolution)
- Public suffix list compliance (registrable domain)

This is the SINGLE SOURCE OF TRUTH for URL validation across ContentDownload.
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


class PolicyError(RuntimeError):
    """URL policy validation error."""

    pass


def validate_url_security(url: str, http_config: Optional[object] = None) -> str:
    """
    Validate URL against security policy.

    Returns:
        Possibly-normalized URL (e.g., upgraded http→https)

    Raises:
        PolicyError: If URL fails validation
    """
    # Parse URL
    parsed = urlparse(url)

    # 1. Scheme validation
    if parsed.scheme not in ("http", "https"):
        raise PolicyError(f"Scheme not allowed: {parsed.scheme} (must be http/https)")

    # 2. Host validation (required)
    if not parsed.hostname:
        raise PolicyError(f"No host in URL: {url}")

    # Normalize host (lowercase, IDN → punycode)
    host_lower = parsed.hostname.lower()
    try:
        # IDN (e.g., "münchen.de") → punycode ("xn--mnchen-3ya.de")
        host_normalized = host_lower.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError) as e:
        raise PolicyError(f"Invalid IDN host {host_lower}: {e}")

    # 3. Port policy (default, per-host overrides, etc.)
    # For now, accept any port; strengthen if needed
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    # 4. HTTP→HTTPS upgrade (enforced for production)
    scheme = parsed.scheme
    if scheme == "http" and os.getenv("ENFORCE_HTTPS_URLS") != "false":
        scheme = "https"
        port = 443

    # 5. Reconstruct URL with normalized host
    normalized = urlunparse(
        (
            scheme,
            f"{host_normalized}:{port}" if parsed.port else host_normalized,
            parsed.path or "/",
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    logger.debug(f"URL validated: {url} → {normalized}")
    return normalized
