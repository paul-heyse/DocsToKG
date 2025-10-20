"""Centralized URL canonicalization helpers for ContentDownload.

This module owns the policy surface for how URLs are normalized before they
enter dedupe indices, Hishel cache keys, rate limiters, and download workers.
It wraps :func:`url_normalize.url_normalize` with project defaults and exposes
two narrow entry points that other modules should consume:

``canonical_for_index`` – produce deterministic keys for manifests and global
dedupe state without mutating query parameters.
``canonical_for_request`` – normalize right before an HTTP request, optionally
filtering tracking parameters for landing pages while preserving semantic ones.

Environment overrides
---------------------
The following environment variables are honoured during module import:

``DOCSTOKG_URL_DEFAULT_SCHEME`` – override the default scheme (default: https).
``DOCSTOKG_URL_FILTER_LANDING`` – toggle landing-page query filtering.
``DOCSTOKG_URL_PARAM_ALLOWLIST`` – allowlist specification
    (e.g., ``site.com:page,id;example.org:id`` or ``page,id`` for a global list).

Call :func:`configure_url_policy` at runtime (CLI bootstrap) to apply parsed
configuration. Tests may call :func:`reset_url_policy_for_tests` to revert to
defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from typing_extensions import Literal
from url_normalize import url_normalize

Role = Literal["metadata", "landing", "artifact"]

DEFAULT_SCHEME = "https"
FILTER_FOR: Mapping[Role, bool] = {"landing": True, "metadata": False, "artifact": False}
PARAM_ALLOWLIST: Mapping[str, Sequence[str]] = {}
DEFAULT_DOMAIN_PER_HOST: Mapping[str, str] = {}


@dataclass
class UrlPolicy:
    """Policy container for canonicalization behaviour."""

    default_scheme: str = DEFAULT_SCHEME
    filter_for: Dict[Role, bool] = field(
        default_factory=lambda: dict(FILTER_FOR)  # copy to avoid mutations on the module constant
    )
    param_allowlist_global: Tuple[str, ...] = ()
    param_allowlist_per_domain: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    default_domain_per_host: Dict[str, str] = field(default_factory=dict)

    def copy(self) -> "UrlPolicy":
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


def _parse_bool(value: Optional[str]) -> Optional[bool]:
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
) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]:
    """Parse allowlist specification strings into global and per-domain mappings."""

    if not spec:
        return (), {}

    global_params: List[str] = []
    per_domain: Dict[str, List[str]] = {}

    for chunk in spec.split(";"):
        text = chunk.strip()
        if not text:
            continue
        if ":" in text:
            domain, _, params_text = text.partition(":")
            domain_key = domain.strip().lower()
            if not domain_key:
                continue
            params = [
                param.strip()
                for param in params_text.split(",")
                if param.strip()
            ]
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
    default_scheme: Optional[str] = None,
    filter_landing: Optional[bool] = None,
    param_allowlist_global: Optional[Iterable[str]] = None,
    param_allowlist_per_domain: Optional[Mapping[str, Iterable[str]]] = None,
    default_domain_per_host: Optional[Mapping[str, str]] = None,
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


def _select_param_allowlist() -> Optional[Union[List[str], Dict[str, List[str]]]]:
    global_allowlist = _POLICY.param_allowlist_global
    domain_map = _POLICY.param_allowlist_per_domain

    if domain_map:
        return {
            domain: list(values)
            for domain, values in domain_map.items()
            if values
        }
    if global_allowlist:
        return list(global_allowlist)
    return None


def canonical_for_index(url: str) -> str:
    """Return the canonical representation used for manifests and dedupe keys."""

    if url is None:
        raise TypeError("canonical_for_index expected a URL string, received None.")
    return url_normalize(url, default_scheme=_POLICY.default_scheme)


def canonical_for_request(
    url: str,
    *,
    role: Role,
    origin_host: Optional[str] = None,
) -> str:
    """Normalize ``url`` before issuing an HTTP request."""

    if url is None:
        raise TypeError("canonical_for_request expected a URL string, received None.")

    kwargs: MutableMapping[str, object] = {"default_scheme": _POLICY.default_scheme}
    default_domain: Optional[str] = None
    if origin_host:
        host_key = origin_host.strip().lower()
        default_domain = _POLICY.default_domain_per_host.get(host_key, host_key)
    if default_domain:
        kwargs["default_domain"] = default_domain

    if _POLICY.filter_for.get(role, False):
        kwargs["filter_params"] = True
        allowlist = _select_param_allowlist()
        if allowlist:
            kwargs["param_allowlist"] = allowlist

    return url_normalize(url, **kwargs)


_apply_environment_overrides()

__all__ = [
    "DEFAULT_SCHEME",
    "DEFAULT_DOMAIN_PER_HOST",
    "FILTER_FOR",
    "PARAM_ALLOWLIST",
    "UrlPolicy",
    "Role",
    "canonical_for_index",
    "canonical_for_request",
    "configure_url_policy",
    "get_url_policy",
    "parse_param_allowlist_spec",
    "reset_url_policy_for_tests",
]
