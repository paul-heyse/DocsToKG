# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.urls_networking",
#   "purpose": "URL normalization instrumentation for the networking hub.",
#   "sections": [
#     {
#       "id": "get-strict-mode",
#       "name": "get_strict_mode",
#       "anchor": "function-get-strict-mode",
#       "kind": "function"
#     },
#     {
#       "id": "set-strict-mode",
#       "name": "set_strict_mode",
#       "anchor": "function-set-strict-mode",
#       "kind": "function"
#     },
#     {
#       "id": "record-url-normalization",
#       "name": "record_url_normalization",
#       "anchor": "function-record-url-normalization",
#       "kind": "function"
#     },
#     {
#       "id": "log-url-change-once",
#       "name": "log_url_change_once",
#       "anchor": "function-log-url-change-once",
#       "kind": "function"
#     },
#     {
#       "id": "apply-role-headers",
#       "name": "apply_role_headers",
#       "anchor": "function-apply-role-headers",
#       "kind": "function"
#     },
#     {
#       "id": "get-url-normalization-stats",
#       "name": "get_url_normalization_stats",
#       "anchor": "function-get-url-normalization-stats",
#       "kind": "function"
#     },
#     {
#       "id": "reset-url-normalization-stats-for-tests",
#       "name": "reset_url_normalization_stats_for_tests",
#       "anchor": "function-reset-url-normalization-stats-for-tests",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""URL normalization instrumentation for the networking hub.

Responsibilities
-----------------
- Track metrics for URL normalization effectiveness (counters for total normalized,
  number changed, unique hosts, roles used).
- Enforce strict mode validation during development/canary (reject non-canonical URLs).
- Apply role-based request headers (Accept, Cache-Control by role).
- Log parameter changes once per host to surface unexpected filtering.

This module bridges urls.py (policy/canonicalization) and networking.py (HTTP layer).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Thread-safe metrics storage
_stats_lock = threading.Lock()
_url_normalization_stats: dict[str, Any] = {
    "normalized_total": 0,
    "changed_total": 0,
    "hosts_seen": set(),
    "roles_used": {},
    "logged_url_changes": set(),  # (host, pattern) tuples to avoid spam
}

# Strict mode flag
_strict_mode = os.getenv("DOCSTOKG_URL_STRICT", "0").lower() in {"1", "true", "yes", "on"}

# Role-based Accept header defaults
ROLE_HEADERS: dict[str, dict[str, str]] = {
    "metadata": {
        "Accept": "application/json, text/javascript;q=0.9, */*;q=0.1",
    },
    "landing": {
        "Accept": "text/html, application/xhtml+xml;q=0.9, */*;q=0.8",
    },
    "artifact": {
        "Accept": "application/pdf, */*;q=0.1",
    },
}


def get_strict_mode() -> bool:
    """Return whether strict URL validation is enabled."""
    return _strict_mode


def set_strict_mode(enabled: bool) -> None:
    """Enable or disable strict URL validation (for tests)."""
    global _strict_mode
    _strict_mode = enabled


def record_url_normalization(
    original_url: str,
    canonical_url: str,
    role: str,
) -> None:
    """Record metrics for a canonicalized URL.

    Args:
        original_url: The input URL before canonicalization
        canonical_url: The URL after normalization
        role: The request role (metadata/landing/artifact)

    Raises:
        ValueError: If strict mode is enabled and URL was modified
    """
    from DocsToKG.ContentDownload.urls import canonical_host

    with _stats_lock:
        # Track total normalized
        _url_normalization_stats["normalized_total"] += 1

        # Track if URL changed
        if original_url != canonical_url:
            _url_normalization_stats["changed_total"] += 1

        # Track unique hosts
        try:
            host = canonical_host(canonical_url)
            _url_normalization_stats["hosts_seen"].add(host)
        except Exception:
            pass  # Silently ignore host extraction failures

        # Track role usage
        roles = _url_normalization_stats["roles_used"]
        roles[role] = roles.get(role, 0) + 1

    # Strict mode validation
    if _strict_mode and original_url != canonical_url:
        raise ValueError(
            f"Non-canonical URL in DOCSTOKG_URL_STRICT=1 mode: '{original_url}' → '{canonical_url}'"
        )


def log_url_change_once(
    original_url: str,
    canonical_url: str,
    host: str | None = None,
) -> None:
    """Log URL changes once per host to avoid spam.

    Args:
        original_url: The input URL
        canonical_url: The normalized URL
        host: Optional canonical host (for grouping)
    """
    if original_url == canonical_url:
        return  # No change, nothing to log

    from DocsToKG.ContentDownload.urls import canonical_host

    try:
        if host is None:
            host = canonical_host(canonical_url)
    except Exception:
        host = "unknown"

    # Extract pattern (scheme + host) to avoid logging every URL for same host
    pattern = f"{host}"

    with _stats_lock:
        logged = _url_normalization_stats["logged_url_changes"]
        if pattern not in logged:
            logged.add(pattern)
            logger.warning(
                f"URL normalized for host '{host}': params filtered or URL shape changed. "
                f"Example: {original_url} → {canonical_url}. This is expected for landing pages."
            )


def apply_role_headers(
    headers: dict[str, str] | None,
    role: str,
) -> dict[str, str]:
    """Apply role-based headers to request.

    Adds default Accept header per role unless already specified by caller.

    Args:
        headers: Existing headers dict (or None)
        role: The request role (metadata/landing/artifact)

    Returns:
        Headers dict with role defaults applied
    """
    if headers is None:
        headers = {}
    else:
        headers = dict(headers)  # Don't mutate caller's dict

    # Only set if not already present (caller can override)
    if "Accept" not in headers and "accept" not in headers:
        role_defaults = ROLE_HEADERS.get(role, {})
        headers.update(role_defaults)

    return headers


def get_url_normalization_stats() -> dict[str, Any]:
    """Return current normalization metrics.

    Returns a snapshot suitable for metrics/logging.
    """
    with _stats_lock:
        return {
            "normalized_total": _url_normalization_stats["normalized_total"],
            "changed_total": _url_normalization_stats["changed_total"],
            "unique_hosts": len(_url_normalization_stats["hosts_seen"]),
            "hosts_seen": sorted(_url_normalization_stats["hosts_seen"]),
            "roles_used": dict(_url_normalization_stats["roles_used"]),
            "strict_mode": _strict_mode,
        }


def reset_url_normalization_stats_for_tests() -> None:
    """Reset metrics to defaults (for tests only)."""
    with _stats_lock:
        _url_normalization_stats["normalized_total"] = 0
        _url_normalization_stats["changed_total"] = 0
        _url_normalization_stats["hosts_seen"] = set()
        _url_normalization_stats["roles_used"] = {}
        _url_normalization_stats["logged_url_changes"] = set()


__all__ = [
    "get_strict_mode",
    "set_strict_mode",
    "record_url_normalization",
    "log_url_change_once",
    "apply_role_headers",
    "get_url_normalization_stats",
    "reset_url_normalization_stats_for_tests",
    "ROLE_HEADERS",
]
