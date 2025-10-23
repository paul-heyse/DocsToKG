# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.cache_control",
#   "purpose": "RFC 9111 cache-control directive parsing and interpretation.",
#   "sections": [
#     {
#       "id": "cachecontroldirective",
#       "name": "CacheControlDirective",
#       "anchor": "class-cachecontroldirective",
#       "kind": "class"
#     },
#     {
#       "id": "parse-cache-control",
#       "name": "parse_cache_control",
#       "anchor": "function-parse-cache-control",
#       "kind": "function"
#     },
#     {
#       "id": "is-fresh",
#       "name": "is_fresh",
#       "anchor": "function-is-fresh",
#       "kind": "function"
#     },
#     {
#       "id": "can-serve-stale",
#       "name": "can_serve_stale",
#       "anchor": "function-can-serve-stale",
#       "kind": "function"
#     },
#     {
#       "id": "should-cache",
#       "name": "should_cache",
#       "anchor": "function-should-cache",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""RFC 9111 cache-control directive parsing and interpretation.

Responsibilities
----------------
- Parse HTTP Cache-Control header into structured directives
- Interpret directives per RFC 9111 specification
- Compute cache freshness (fresh vs stale)
- Support both response and request directives
- Handle malformed headers gracefully with logging

Design Notes
------------
- Frozen dataclasses ensure immutability of parsed directives
- Conservative defaults (assume must-revalidate if unclear)
- Per-RFC: no-cache requires revalidation, no-store forbids caching
- max-age in seconds; s-maxage for shared caches only
- stale-while-revalidate allows serving stale during revalidation
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

# RFC 9111 cache-control directive regex
# Matches: directive-name=value or directive-name (with optional spaces)
_DIRECTIVE_PATTERN = re.compile(r'([a-zA-Z\-]+)(?:=(["\']?)(\d+|[^,\s"\']+)\2)?')


@dataclass(frozen=True)
class CacheControlDirective:
    """Immutable representation of parsed cache-control directives.

    Per RFC 9111, the Cache-Control header field contains directives that
    specify caching policies for both requests and responses.

    Attributes:
        no_cache: Must revalidate with origin before using cached response
        no_store: Must not store response (most restrictive)
        public: Response may be cached by any cache (public/private distinction)
        private: Response may only be cached by private cache (not shared)
        max_age: Response freshness lifetime in seconds (0 = already stale)
        s_maxage: Shared cache freshness lifetime (overrides max_age for shared caches)
        must_revalidate: Must not serve stale response when origin unreachable
        proxy_revalidate: Like must_revalidate but only for shared caches
        stale_while_revalidate: Serve stale for N seconds while revalidating in background
        stale_if_error: Serve stale for N seconds even if revalidation fails
    """

    no_cache: bool = False
    no_store: bool = False
    public: bool = False
    private: bool = False
    max_age: int | None = None
    s_maxage: int | None = None
    must_revalidate: bool = False
    proxy_revalidate: bool = False
    stale_while_revalidate: int = 0
    stale_if_error: int = 0
    # Additional directives
    immutable: bool = False


def parse_cache_control(headers: Mapping[str, str]) -> CacheControlDirective:
    """Parse Cache-Control header into structured directives.

    Args:
        headers: HTTP headers (case-insensitive key access)

    Returns:
        CacheControlDirective with parsed values

    Examples:
        >>> headers = {"cache-control": "max-age=3600, public"}
        >>> directive = parse_cache_control(headers)
        >>> directive.max_age
        3600
        >>> directive.public
        True

    Notes:
        - Performs case-insensitive header lookup
        - Handles quoted values (e.g., `private="true"`)
        - Gracefully ignores unknown directives with logging
        - Conservative defaults: missing directives default to False/None
    """
    # Find Cache-Control header (case-insensitive)
    cc_header = None
    for key, value in headers.items():
        if key.lower() == "cache-control":
            cc_header = value
            break

    if not cc_header:
        return CacheControlDirective()

    # Parse directives
    kwargs: dict = {}
    for match in _DIRECTIVE_PATTERN.finditer(cc_header):
        directive_name = match.group(1).lower()
        value_str = match.group(3) if match.group(3) else None

        # Boolean directives (value is optional or always true)
        if directive_name in ("no-cache", "no_cache"):
            kwargs["no_cache"] = True
        elif directive_name in ("no-store", "no_store"):
            kwargs["no_store"] = True
        elif directive_name == "public":
            kwargs["public"] = True
        elif directive_name == "private":
            kwargs["private"] = True
        elif directive_name in ("must-revalidate", "must_revalidate"):
            kwargs["must_revalidate"] = True
        elif directive_name in ("proxy-revalidate", "proxy_revalidate"):
            kwargs["proxy_revalidate"] = True
        elif directive_name == "immutable":
            kwargs["immutable"] = True
        # Integer directives
        elif directive_name in ("max-age", "max_age"):
            try:
                kwargs["max_age"] = int(value_str) if value_str else 0
            except (ValueError, TypeError):
                LOGGER.debug(f"Invalid max-age value: {value_str}")
        elif directive_name in ("s-maxage", "s_maxage"):
            try:
                kwargs["s_maxage"] = int(value_str) if value_str else 0
            except (ValueError, TypeError):
                LOGGER.debug(f"Invalid s-maxage value: {value_str}")
        elif directive_name in ("stale-while-revalidate", "stale_while_revalidate"):
            try:
                kwargs["stale_while_revalidate"] = int(value_str) if value_str else 0
            except (ValueError, TypeError):
                LOGGER.debug(f"Invalid stale-while-revalidate value: {value_str}")
        elif directive_name in ("stale-if-error", "stale_if_error"):
            try:
                kwargs["stale_if_error"] = int(value_str) if value_str else 0
            except (ValueError, TypeError):
                LOGGER.debug(f"Invalid stale-if-error value: {value_str}")

    return CacheControlDirective(**kwargs)


def is_fresh(directive: CacheControlDirective, age_seconds: float) -> bool:
    """Determine if cached response is fresh per cache-control directives.

    A response is fresh if its age is less than its freshness lifetime.

    Args:
        directive: Parsed cache-control directives
        age_seconds: Response age in seconds (time since generation)

    Returns:
        True if response should be considered fresh, False if stale

    Notes:
        - Treats max-age=0 as immediately stale
        - For shared caches, s-maxage (if set) takes precedence over max-age
        - Does not consider Cache-Control: public/private for freshness
        - Conservative: if both max-age and s-maxage missing, returns False (stale)

    Examples:
        >>> directive = CacheControlDirective(max_age=3600)
        >>> is_fresh(directive, 1800.0)  # 30 min age vs 1 hour TTL
        True
        >>> is_fresh(directive, 3600.1)  # Just past expiry
        False
    """
    # no-store means never cache, so never fresh
    if directive.no_store:
        return False

    # no-cache means must revalidate, so never fresh
    if directive.no_cache:
        return False

    # Determine applicable freshness lifetime
    freshness_lifetime = directive.max_age
    if directive.s_maxage is not None:
        freshness_lifetime = directive.s_maxage

    # If no TTL specified, consider stale (conservative)
    if freshness_lifetime is None:
        return False

    # Response is fresh if age <= freshness_lifetime per RFC 7234 §4.2.4
    return age_seconds <= freshness_lifetime


def can_serve_stale(
    directive: CacheControlDirective,
    age_seconds: float,
    is_revalidation_error: bool = False,
) -> bool:
    """Determine if stale response can be served (stale-while-revalidate/stale-if-error).

    Args:
        directive: Parsed cache-control directives
        age_seconds: Response age in seconds
        is_revalidation_error: True if revalidation failed (for stale-if-error)

    Returns:
        True if stale response can be served, False if must be discarded

    Notes:
        - stale-while-revalidate: serve stale N seconds while revalidating in background
        - stale-if-error: serve stale N seconds even if revalidation fails
        - must-revalidate: forbids serving stale even during errors
        - Conservative: if both extensions missing, returns False

    Examples:
        >>> directive = CacheControlDirective(max_age=3600, stale_while_revalidate=60)
        >>> can_serve_stale(directive, 3605.0)  # 5 sec past stale-while-revalidate
        True
        >>> can_serve_stale(directive, 3700.0)  # Past grace period
        False
    """
    # must-revalidate forbids serving stale
    if directive.must_revalidate:
        return False

    # If revalidation error, check stale-if-error
    if is_revalidation_error:
        freshness_lifetime = directive.max_age
        if freshness_lifetime is None:
            return False
        grace_period = directive.stale_if_error
        return age_seconds <= (freshness_lifetime + grace_period)

    # Normal stale-while-revalidate
    freshness_lifetime = directive.max_age
    if freshness_lifetime is None:
        return False

    grace_period = directive.stale_while_revalidate
    return age_seconds <= (freshness_lifetime + grace_period)


def should_cache(directive: CacheControlDirective) -> bool:
    """Determine if response should be cached at all.

    Args:
        directive: Parsed cache-control directives

    Returns:
        False if response must not be cached, True otherwise

    Notes:
        - no-store: must not cache
        - no-cache: may cache but must revalidate
        - Assumes private caches unless explicitly marked as shared-only
        - Conservative: if unclear, allows caching (caller applies policy)

    Examples:
        >>> CacheControlDirective(no_store=True)
        False
        >>> CacheControlDirective()
        True
    """
    # no-store explicitly forbids caching
    return not directive.no_store
