# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.urls",
#   "purpose": "URL canonicalization policies for content downloads",
#   "sections": [
#     {
#       "id": "urlpolicy",
#       "name": "UrlPolicy",
#       "anchor": "class-urlpolicy",
#       "kind": "class"
#     },
#     {
#       "id": "strip-fragment",
#       "name": "_strip_fragment",
#       "anchor": "function-strip-fragment",
#       "kind": "function"
#     },
#     {
#       "id": "parse-bool",
#       "name": "_parse_bool",
#       "anchor": "function-parse-bool",
#       "kind": "function"
#     },
#     {
#       "id": "parse-param-allowlist-spec",
#       "name": "parse_param_allowlist_spec",
#       "anchor": "function-parse-param-allowlist-spec",
#       "kind": "function"
#     },
#     {
#       "id": "configure-url-policy",
#       "name": "configure_url_policy",
#       "anchor": "function-configure-url-policy",
#       "kind": "function"
#     },
#     {
#       "id": "apply-environment-overrides",
#       "name": "_apply_environment_overrides",
#       "anchor": "function-apply-environment-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "get-url-policy",
#       "name": "get_url_policy",
#       "anchor": "function-get-url-policy",
#       "kind": "function"
#     },
#     {
#       "id": "reset-url-policy-for-tests",
#       "name": "reset_url_policy_for_tests",
#       "anchor": "function-reset-url-policy-for-tests",
#       "kind": "function"
#     },
#     {
#       "id": "select-param-allowlist",
#       "name": "_select_param_allowlist",
#       "anchor": "function-select-param-allowlist",
#       "kind": "function"
#     },
#     {
#       "id": "canonical-for-index",
#       "name": "canonical_for_index",
#       "anchor": "function-canonical-for-index",
#       "kind": "function"
#     },
#     {
#       "id": "canonical-for-request",
#       "name": "canonical_for_request",
#       "anchor": "function-canonical-for-request",
#       "kind": "function"
#     },
#     {
#       "id": "canonical-host",
#       "name": "canonical_host",
#       "anchor": "function-canonical-host",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Centralized URL canonicalization helpers for ContentDownload.

Responsibilities
----------------
- Normalise URLs before they enter dedupe indices, cache keys, or rate limiter
  lookups via :func:`canonical_for_index` and :func:`canonical_for_request`.
- Parse and apply allowlist / default-domain policy through
  :func:`configure_url_policy`, honouring CLI and environment overrides.
- Provide test hooks (:func:`reset_url_policy_for_tests`) so fixtures can tweak
  canonicalisation in isolation.
- Derive standardized host keys for limiter, breaker, and HTTPX transport routing
  via :func:`canonical_host`.

Policy & Canonicalization Rules
--------------------------------
URL normalization follows RFC 3986/3987 best practices:

1. **Scheme and host casing**: lowercase scheme and host (e.g., `HTTP://Example.COM` → `http://example.com`).
2. **Percent-encoding**: uppercase hex escapes and decode unreserved characters (e.g., `%2a` → `%2A`).
3. **Dot-segment removal**: normalize paths with `.` and `..` (e.g., `/a/./b/../c` → `/a/c`).
4. **Path defaults**: add `/` for empty paths on http/https (e.g., `http://example.com` → `http://example.com/`).
5. **Default port dropping**: remove port **only if it matches the scheme default** (80 for http, 443 for https).
   **Gotcha**: with `DEFAULT_SCHEME="https"`, a URL like `www.example.com:80/foo` will keep `:80` because
   80 is not the default for https; only drop when input scheme matches or defaults to http.
6. **IDN normalization**: convert international domain names to ASCII punycode (e.g., `münchen.example` → `xn--mnich-kva.example`).
7. **Fragment removal**: strip URL fragments (client-side only; not sent to origin).
8. **Query parameter handling**:
   - **No reordering**: parameter order may be semantically meaningful; never sort or shuffle.
   - **Filtering** (landing role only): optionally drop known trackers (`utm_*`, `gclid`, etc.) if enabled.
   - **Allowlists** (per-domain): preserve only specified params when filtering is active.

Role-Based Behavior
-------------------
Three roles control request shaping and filtering:

- `metadata`: REST/JSON API calls. No param filtering; preserve all query strings.
  Suitable for signed/authenticated requests; CDN signatures must remain untouched.
- `landing`: HTML discovery from landing pages. Param filtering enabled; drop trackers.
  Remove noise (`utm_*`, `fbclid`, `gclid`, etc.) but preserve application params (e.g., `page`, `id`).
- `artifact`: Direct downloads (PDF, XML). No param filtering; preserve all semantics.
  Treat as immutable; never remove parts of authenticated/CDN URLs.

Gotchas & Constraints
---------------------
- **Signature/Auth preservation**: If a URL includes signed parameters (`X-Amz-Signature`, tokens, expires),
  never use `landing` role; use `artifact` or create a custom role. The default drop list omits known
  sig param names intentionally.
- **Scheme inference**: default scheme is `https` (v2.0+ behavior). Relative URLs and scheme-less inputs
  will be upgraded to https unless explicitly overridden.
- **Port gotcha**: only the scheme-default port (80→http, 443→https) gets dropped. Explicit non-default
  ports (e.g., 8080, 9000) are preserved.
- **CDN & shortlinks**: shortlink params like `?download=1`, `?pdf=1` are **preserved** for `artifact` role,
  but dropped for `landing` if not in the allowlist. Always verify before deploying if a provider uses
  shortlink parameters.

Environment Overrides
---------------------
The following environment variables are honoured during module import:

``DOCSTOKG_URL_DEFAULT_SCHEME`` – override the default scheme (default: https).
``DOCSTOKG_URL_FILTER_LANDING`` – toggle landing-page query filtering (default: True).
``DOCSTOKG_URL_PARAM_ALLOWLIST`` – allowlist specification
    (e.g., ``site.com:page,id;example.org:id`` or ``page,id`` for a global list).

Call :func:`configure_url_policy` at runtime (CLI bootstrap) to apply parsed
configuration. Tests may call :func:`reset_url_policy_for_tests` to revert to
defaults.

Typical Integration
-------------------
1. **Resolvers** compute `canonical_url = canonical_for_index(original_url)` and forward it to the pipeline.
2. **Networking hub** calls `canonical_for_request(url, role=<role>)` just before sending to HTTPX.
3. **Limiter/Breaker** use `canonical_host(url)` to derive consistent keys.
4. **Hishel cache** receives the canonical URL from HTTPX, ensuring cache hits are stable.
5. **Manifest** persists both `original_url` and `canonical_url` for audit/debug.

Example Usage
-------------
    from DocsToKG.ContentDownload.urls import (
        canonical_for_index,
        canonical_for_request,
        canonical_host,
    )

    # For deduplication
    orig_url = "HTTP://EXAMPLE.COM:443/path?utm_source=x&id=1"
    index_url = canonical_for_index(orig_url)  # "https://example.com/path?utm_source=x&id=1"

    # For sending (landing role strips trackers)
    request_url = canonical_for_request(orig_url, role="landing")
    # "https://example.com/path?id=1"

    # For limiter key
    host_key = canonical_host(orig_url)  # "example.com"
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from url_normalize import url_normalize

Role = Literal["metadata", "landing", "artifact"]

DEFAULT_SCHEME = "https"
FILTER_FOR: Mapping[Role, bool] = {"landing": True, "metadata": False, "artifact": False}
PARAM_ALLOWLIST: Mapping[str, Sequence[str]] = {}
DEFAULT_DOMAIN_PER_HOST: Mapping[str, str] = {}

# Conservative set of tracker/marketing parameters to drop when filtering landing pages.
# Excludes known authentication, CDN signature, and semantic shortlink parameters.
DROP_PARAMS_DEFAULT = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",  # Google Ads
        "fbclid",  # Facebook Ads
        "yclid",  # Yandex Ads
        "mc_eid",  # Mailchimp Email ID
        "mc_cid",  # Mailchimp Campaign ID
        "ref",
        "ref_",
        "referrer",  # Referrer tracking
        "spm",  # Alibaba tracking
        "igshid",  # Instagram share ID
        "mkt_tok",  # Marketo token
        "msclkid",  # Microsoft Click ID
        "_hsenc",
        "_hsmi",  # HubSpot
    }
)


@dataclass
class UrlPolicy:
    """Policy container for canonicalization behaviour."""

    default_scheme: str = DEFAULT_SCHEME
    filter_for: dict[Role, bool] = field(
        default_factory=lambda: dict(FILTER_FOR)  # copy to avoid mutations on the module constant
    )
    param_allowlist_global: tuple[str, ...] = ()
    param_allowlist_per_domain: dict[str, tuple[str, ...]] = field(default_factory=dict)
    default_domain_per_host: dict[str, str] = field(default_factory=dict)

    def copy(self) -> UrlPolicy:
        return UrlPolicy(
            default_scheme=self.default_scheme,
            filter_for=dict(self.filter_for),
            param_allowlist_global=tuple(self.param_allowlist_global),
            param_allowlist_per_domain={
                domain: tuple(values) for domain, values in self.param_allowlist_per_domain.items()
            },
            default_domain_per_host=dict(self.default_domain_per_host),
        )


_POLICY = UrlPolicy()


def _strip_fragment(value: str) -> str:
    """Return ``value`` without a URL fragment."""

    parts = urlsplit(value)
    if not parts.fragment:
        return value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def parse_param_allowlist_spec(
    spec: str,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Parse allowlist specification strings into global and per-domain mappings."""

    if not spec:
        return (), {}

    global_params: list[str] = []
    per_domain: dict[str, list[str]] = {}

    for chunk in spec.split(";"):
        text = chunk.strip()
        if not text:
            continue
        if ":" in text:
            domain, _, params_text = text.partition(":")
            domain_key = domain.strip().lower()
            if not domain_key:
                continue
            params = [param.strip() for param in params_text.split(",") if param.strip()]
            if not params:
                continue
            unique_params = []
            seen = set()
            for param in params:
                if param not in seen:
                    seen.add(param)
                    unique_params.append(param)
            per_domain[domain_key] = unique_params
        else:
            for param in text.split(","):
                value = param.strip()
                if value and value not in global_params:
                    global_params.append(value)

    return tuple(global_params), {host: tuple(values) for host, values in per_domain.items()}


def configure_url_policy(
    *,
    default_scheme: str | None = None,
    filter_landing: bool | None = None,
    param_allowlist_global: Iterable[str] | None = None,
    param_allowlist_per_domain: Mapping[str, Iterable[str]] | None = None,
    default_domain_per_host: Mapping[str, str] | None = None,
) -> None:
    """Override the global URL policy."""

    global _POLICY
    policy = _POLICY.copy()

    if default_scheme:
        policy.default_scheme = default_scheme.strip().lower() or DEFAULT_SCHEME

    if filter_landing is not None:
        policy.filter_for["landing"] = bool(filter_landing)

    if param_allowlist_global is not None:
        unique = []
        seen = set()
        for value in param_allowlist_global:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                unique.append(text)
        policy.param_allowlist_global = tuple(unique)

    if param_allowlist_per_domain is not None:
        policy.param_allowlist_per_domain = {
            domain.strip().lower(): tuple(
                dict.fromkeys(str(param).strip() for param in params if str(param).strip())
            )
            for domain, params in param_allowlist_per_domain.items()
            if domain and params
        }

    if default_domain_per_host is not None:
        policy.default_domain_per_host = {
            host.strip().lower(): domain.strip().lower()
            for host, domain in default_domain_per_host.items()
            if host and domain
        }

    _POLICY = policy


def _apply_environment_overrides() -> None:
    scheme = os.getenv("DOCSTOKG_URL_DEFAULT_SCHEME")
    if scheme:
        configure_url_policy(default_scheme=scheme)

    filter_value = _parse_bool(os.getenv("DOCSTOKG_URL_FILTER_LANDING"))
    if filter_value is not None:
        configure_url_policy(filter_landing=filter_value)

    allowlist_spec = os.getenv("DOCSTOKG_URL_PARAM_ALLOWLIST")
    if allowlist_spec:
        global_params, per_domain = parse_param_allowlist_spec(allowlist_spec)
        configure_url_policy(
            param_allowlist_global=global_params,
            param_allowlist_per_domain=per_domain,
        )


def get_url_policy() -> UrlPolicy:
    """Return a shallow copy of the active URL policy."""

    return _POLICY.copy()


def reset_url_policy_for_tests() -> None:
    """Reset the URL policy to defaults (used in unit tests)."""

    global _POLICY
    _POLICY = UrlPolicy()
    _apply_environment_overrides()


def _select_param_allowlist() -> list[str] | dict[str, list[str]] | None:
    global_allowlist = _POLICY.param_allowlist_global
    domain_map = _POLICY.param_allowlist_per_domain

    if domain_map:
        return {domain: list(values) for domain, values in domain_map.items() if values}
    if global_allowlist:
        return list(global_allowlist)
    return None


def canonical_for_index(url: str) -> str:
    """Return the canonical representation used for manifests and dedupe keys."""

    if url is None:
        raise TypeError("canonical_for_index expected a URL string, received None.")
    canonical = url_normalize(url, default_scheme=_POLICY.default_scheme)
    return _strip_fragment(canonical or "")


def canonical_for_request(
    url: str,
    *,
    role: Role,
    origin_host: str | None = None,
) -> str:
    """Normalize ``url`` before issuing an HTTP request."""

    if url is None:
        raise TypeError("canonical_for_request expected a URL string, received None.")

    # Build kwargs dict with proper typing for url_normalize
    kwargs_dict: dict[str, Any] = {"default_scheme": _POLICY.default_scheme}
    default_domain: str | None = None
    if origin_host:
        host_key = origin_host.strip().lower()
        default_domain = _POLICY.default_domain_per_host.get(host_key, host_key)
    if default_domain:
        kwargs_dict["default_domain"] = default_domain

    if _POLICY.filter_for.get(role, False):
        kwargs_dict["filter_params"] = True
        allowlist = _select_param_allowlist()
        if allowlist:
            kwargs_dict["param_allowlist"] = allowlist

    canonical = url_normalize(url, **kwargs_dict)  # type: ignore[arg-type]
    return _strip_fragment(canonical or "")


def canonical_host(url: str) -> str:
    """Extract the canonical hostname from a URL for limiter/breaker/mount keys.

    Returns the lowercase, IDN-normalized (punycode) hostname without port,
    suitable for use as a key in rate limiters, circuit breakers, and HTTPX
    transport mounts.

    Raises
    ------
    TypeError
        If ``url`` is None or not a string.
    ValueError
        If the URL has no extractable hostname.

    Examples
    --------
    >>> canonical_host("HTTP://EXAMPLE.COM:443/path")
    'example.com'
    >>> canonical_host("https://münchen.example/test")
    'xn--mnich-kva.example'
    """

    if url is None:
        raise TypeError("canonical_host expected a URL string, received None.")
    if not isinstance(url, str):
        raise TypeError(f"canonical_host expected a string, got {type(url).__name__}")

    # Canonicalize to ensure IDN normalization, then extract host
    canonical = canonical_for_index(url)
    if not canonical:
        raise ValueError(f"Could not extract hostname from URL: {url}")

    try:
        parts = urlsplit(canonical)
        hostname = parts.hostname or parts.netloc.split(":")[0]
        if not hostname:
            raise ValueError(f"URL has no hostname: {canonical}")
        return hostname.lower()
    except Exception as e:
        raise ValueError(f"Could not extract hostname from canonical URL {canonical}: {e}") from e


_apply_environment_overrides()

__all__ = [
    "DEFAULT_SCHEME",
    "DEFAULT_DOMAIN_PER_HOST",
    "DROP_PARAMS_DEFAULT",
    "FILTER_FOR",
    "PARAM_ALLOWLIST",
    "Role",
    "UrlPolicy",
    "canonical_for_index",
    "canonical_for_request",
    "canonical_host",
    "configure_url_policy",
    "get_url_policy",
    "parse_param_allowlist_spec",
    "reset_url_policy_for_tests",
]
