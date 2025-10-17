# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.pipeline",
#   "purpose": "Resolver definitions and pipeline orchestration",
#   "sections": [
#     {
#       "id": "resolverresult",
#       "name": "ResolverResult",
#       "anchor": "class-resolverresult",
#       "kind": "class"
#     },
#     {
#       "id": "normalise-domain-bytes-budget",
#       "name": "_normalise_domain_bytes_budget",
#       "anchor": "function-normalise-domain-bytes-budget",
#       "kind": "function"
#     },
#     {
#       "id": "normalise-domain-content-rules",
#       "name": "_normalise_domain_content_rules",
#       "anchor": "function-normalise-domain-content-rules",
#       "kind": "function"
#     },
#     {
#       "id": "resolverconfig",
#       "name": "ResolverConfig",
#       "anchor": "class-resolverconfig",
#       "kind": "class"
#     },
#     {
#       "id": "read-resolver-config",
#       "name": "read_resolver_config",
#       "anchor": "function-read-resolver-config",
#       "kind": "function"
#     },
#     {
#       "id": "seed-resolver-toggle-defaults",
#       "name": "_seed_resolver_toggle_defaults",
#       "anchor": "function-seed-resolver-toggle-defaults",
#       "kind": "function"
#     },
#     {
#       "id": "apply-config-overrides",
#       "name": "apply_config_overrides",
#       "anchor": "function-apply-config-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "load-resolver-config",
#       "name": "load_resolver_config",
#       "anchor": "function-load-resolver-config",
#       "kind": "function"
#     },
#     {
#       "id": "attemptrecord",
#       "name": "AttemptRecord",
#       "anchor": "class-attemptrecord",
#       "kind": "class"
#     },
#     {
#       "id": "downloadoutcome",
#       "name": "DownloadOutcome",
#       "anchor": "class-downloadoutcome",
#       "kind": "class"
#     },
#     {
#       "id": "pipelineresult",
#       "name": "PipelineResult",
#       "anchor": "class-pipelineresult",
#       "kind": "class"
#     },
#     {
#       "id": "resolver",
#       "name": "Resolver",
#       "anchor": "class-resolver",
#       "kind": "class"
#     },
#     {
#       "id": "resolvermetrics",
#       "name": "ResolverMetrics",
#       "anchor": "class-resolvermetrics",
#       "kind": "class"
#     },
#     {
#       "id": "fetch-semantic-scholar-data",
#       "name": "_fetch_semantic_scholar_data",
#       "anchor": "function-fetch-semantic-scholar-data",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-unpaywall-data",
#       "name": "_fetch_unpaywall_data",
#       "anchor": "function-fetch-unpaywall-data",
#       "kind": "function"
#     },
#     {
#       "id": "resolverregistry",
#       "name": "ResolverRegistry",
#       "anchor": "class-resolverregistry",
#       "kind": "class"
#     },
#     {
#       "id": "registeredresolver",
#       "name": "RegisteredResolver",
#       "anchor": "class-registeredresolver",
#       "kind": "class"
#     },
#     {
#       "id": "apiresolverbase",
#       "name": "ApiResolverBase",
#       "anchor": "class-apiresolverbase",
#       "kind": "class"
#     },
#     {
#       "id": "absolute-url",
#       "name": "_absolute_url",
#       "anchor": "function-absolute-url",
#       "kind": "function"
#     },
#     {
#       "id": "collect-candidate-urls",
#       "name": "_collect_candidate_urls",
#       "anchor": "function-collect-candidate-urls",
#       "kind": "function"
#     },
#     {
#       "id": "find-pdf-via-meta",
#       "name": "find_pdf_via_meta",
#       "anchor": "function-find-pdf-via-meta",
#       "kind": "function"
#     },
#     {
#       "id": "find-pdf-via-link",
#       "name": "find_pdf_via_link",
#       "anchor": "function-find-pdf-via-link",
#       "kind": "function"
#     },
#     {
#       "id": "find-pdf-via-anchor",
#       "name": "find_pdf_via_anchor",
#       "anchor": "function-find-pdf-via-anchor",
#       "kind": "function"
#     },
#     {
#       "id": "default-resolvers",
#       "name": "default_resolvers",
#       "anchor": "function-default-resolvers",
#       "kind": "function"
#     },
#     {
#       "id": "arxivresolver",
#       "name": "ArxivResolver",
#       "anchor": "class-arxivresolver",
#       "kind": "class"
#     },
#     {
#       "id": "coreresolver",
#       "name": "CoreResolver",
#       "anchor": "class-coreresolver",
#       "kind": "class"
#     },
#     {
#       "id": "crossrefresolver",
#       "name": "CrossrefResolver",
#       "anchor": "class-crossrefresolver",
#       "kind": "class"
#     },
#     {
#       "id": "doajresolver",
#       "name": "DoajResolver",
#       "anchor": "class-doajresolver",
#       "kind": "class"
#     },
#     {
#       "id": "europepmcresolver",
#       "name": "EuropePmcResolver",
#       "anchor": "class-europepmcresolver",
#       "kind": "class"
#     },
#     {
#       "id": "figshareresolver",
#       "name": "FigshareResolver",
#       "anchor": "class-figshareresolver",
#       "kind": "class"
#     },
#     {
#       "id": "halresolver",
#       "name": "HalResolver",
#       "anchor": "class-halresolver",
#       "kind": "class"
#     },
#     {
#       "id": "landingpageresolver",
#       "name": "LandingPageResolver",
#       "anchor": "class-landingpageresolver",
#       "kind": "class"
#     },
#     {
#       "id": "openaireresolver",
#       "name": "OpenAireResolver",
#       "anchor": "class-openaireresolver",
#       "kind": "class"
#     },
#     {
#       "id": "openalexresolver",
#       "name": "OpenAlexResolver",
#       "anchor": "class-openalexresolver",
#       "kind": "class"
#     },
#     {
#       "id": "osfresolver",
#       "name": "OsfResolver",
#       "anchor": "class-osfresolver",
#       "kind": "class"
#     },
#     {
#       "id": "pmcresolver",
#       "name": "PmcResolver",
#       "anchor": "class-pmcresolver",
#       "kind": "class"
#     },
#     {
#       "id": "semanticscholarresolver",
#       "name": "SemanticScholarResolver",
#       "anchor": "class-semanticscholarresolver",
#       "kind": "class"
#     },
#     {
#       "id": "unpaywallresolver",
#       "name": "UnpaywallResolver",
#       "anchor": "class-unpaywallresolver",
#       "kind": "class"
#     },
#     {
#       "id": "waybackresolver",
#       "name": "WaybackResolver",
#       "anchor": "class-waybackresolver",
#       "kind": "class"
#     },
#     {
#       "id": "zenodoresolver",
#       "name": "ZenodoResolver",
#       "anchor": "class-zenodoresolver",
#       "kind": "class"
#     },
#     {
#       "id": "callable-accepts-argument",
#       "name": "_callable_accepts_argument",
#       "anchor": "function-callable-accepts-argument",
#       "kind": "function"
#     },
#     {
#       "id": "runstate",
#       "name": "_RunState",
#       "anchor": "class-runstate",
#       "kind": "class"
#     },
#     {
#       "id": "resolverpipeline",
#       "name": "ResolverPipeline",
#       "anchor": "class-resolverpipeline",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Content Download Resolver Orchestration

This module centralises resolver configuration, provider registration,
pipeline orchestration, and cache helpers for the DocsToKG content download
stack. Resolver classes encapsulate provider-specific discovery logic while
the shared pipeline coordinates rate limiting, concurrency, and polite HTTP
behaviour.

Key Features:
- Resolver registry supplying default provider ordering and toggles.
- Shared retry helper integration to ensure consistent network backoff.
- Manifest and attempt bookkeeping for detailed diagnostics.
- Utility functions for cache invalidation and signature normalisation.

Dependencies:
- requests: Outbound HTTP traffic and session management.
- BeautifulSoup: Optional HTML parsing for resolver implementations.
- DocsToKG.ContentDownload.network: Shared retry and session helpers.

Usage:
    from DocsToKG.ContentDownload import pipeline

    config = pipeline.ResolverConfig()
    active_resolvers = pipeline.default_resolvers()
    runner = pipeline.ResolverPipeline(
        resolvers=active_resolvers,
        config=config,
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import threading
import time as _time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import BoundedSemaphore, Lock
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
)
from urllib.parse import quote, urljoin, urlparse, urlsplit

import requests as _requests

from DocsToKG.ContentDownload.core import (
    DEFAULT_MIN_PDF_BYTES,
    DEFAULT_SNIFF_BYTES,
    DEFAULT_TAIL_CHECK_BYTES,
    PDF_LIKE,
    Classification,
    ReasonCode,
    dedupe,
    normalize_doi,
    normalize_pmcid,
    normalize_url,
    parse_size,
    strip_prefix,
)
from DocsToKG.ContentDownload.network import (
    CircuitBreaker,
    TokenBucket,
    head_precheck,
    request_with_retries,
)
from DocsToKG.ContentDownload.telemetry import AttemptSink

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact

try:  # Optional dependency guarded at runtime
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    BeautifulSoup = None

# --- Globals ---

LOGGER = logging.getLogger(__name__)

DEFAULT_RESOLVER_ORDER: List[str] = [
    "openalex",
    "unpaywall",
    "crossref",
    "landing_page",
    "arxiv",
    "pmc",
    "europe_pmc",
    "core",
    "zenodo",
    "figshare",
    "doaj",
    "semantic_scholar",
    "openaire",
    "hal",
    "osf",
    "wayback",
]

_DEFAULT_RESOLVER_TOGGLES: Dict[str, bool] = {
    name: name not in {"openaire", "hal", "osf"} for name in DEFAULT_RESOLVER_ORDER
}
DEFAULT_RESOLVER_TOGGLES = MappingProxyType(_DEFAULT_RESOLVER_TOGGLES)

__all__ = [
    "AttemptRecord",
    "AttemptSink",
    "DownloadFunc",
    "DownloadOutcome",
    "PipelineResult",
    "Resolver",
    "ResolverConfig",
    "ResolverMetrics",
    "ResolverPipeline",
    "ResolverRegistry",
    "ResolverResult",
    "RegisteredResolver",
    "apply_config_overrides",
    "default_resolvers",
    "load_resolver_config",
    "read_resolver_config",
]

# --- Resolver Configuration ---


@dataclass
class ResolverResult:
    """Either a candidate download URL or an informational resolver event.

    Attributes:
        url: Candidate download URL emitted by the resolver (``None`` for events).
        referer: Optional referer header to accompany the download request.
        metadata: Arbitrary metadata recorded alongside the result.
        event: Optional event label (e.g., ``"error"`` or ``"skipped"``).
        event_reason: Human-readable reason describing the event.
        http_status: HTTP status associated with the event, when available.

    Examples:
        >>> ResolverResult(url="https://example.org/file.pdf", metadata={"resolver": "core"})
    """

    url: Optional[str]
    referer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event: Optional[str] = None
    event_reason: Optional[str] = None
    http_status: Optional[int] = None

    @property
    def is_event(self) -> bool:
        """Return ``True`` when this result represents an informational event.

        Args:
            self: Resolver result instance under inspection.

        Returns:
            bool: ``True`` if the resolver emitted an event instead of a URL.
        """

        return self.url is None


def _normalise_domain_bytes_budget(data: Mapping[str, Any]) -> Dict[str, int]:
    """Normalize domain byte budgets to lower-case host keys and integer byte limits."""

    if data is None:
        return {}
    if not isinstance(data, Mapping):  # pragma: no cover - defensive guard
        raise ValueError("domain_bytes_budget must be a mapping of host -> bytes")

    normalized: Dict[str, int] = {}
    for host, value in data.items():
        host_key = str(host).strip().lower()
        if not host_key:
            continue
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(
                ("domain_bytes_budget['{name}'] must be an integer, got {value!r}").format(
                    name=host, value=value
                )
            ) from exc
        if limit <= 0:
            raise ValueError(
                ("domain_bytes_budget['{name}'] must be positive, got {value}").format(
                    name=host, value=value
                )
            )
        normalized[host_key] = limit
    return normalized


def _normalise_domain_content_rules(data: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize domain content policies to lower-case host keys and canonical values."""

    if data is None:
        return {}
    if not isinstance(data, Mapping):  # pragma: no cover - defensive guard
        raise ValueError("domain_content_rules must be a mapping of host -> policy")

    normalized: Dict[str, Dict[str, Any]] = {}
    for host, raw_spec in data.items():
        if not host:
            continue
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                ("domain_content_rules['{name}'] must be a mapping, got {value!r}").format(
                    name=host, value=raw_spec
                )
            )
        host_key = str(host).strip().lower()
        if not host_key:
            continue

        policy: Dict[str, Any] = {}
        if "max_bytes" in raw_spec and raw_spec["max_bytes"] is not None:
            raw_limit = raw_spec["max_bytes"]
            try:
                limit_value = (
                    parse_size(raw_limit) if isinstance(raw_limit, str) else int(raw_limit)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    (
                        "domain_content_rules['{name}'].max_bytes must be an integer or size string,"
                        " got {value!r}"
                    ).format(name=host, value=raw_limit)
                ) from exc
            if limit_value <= 0:
                raise ValueError(
                    (
                        "domain_content_rules['{name}'].max_bytes must be positive, got {value}"
                    ).format(name=host, value=limit_value)
                )
            policy["max_bytes"] = int(limit_value)

        allowed_raw = raw_spec.get("allowed_types")
        if allowed_raw:
            candidates: List[str] = []
            if isinstance(allowed_raw, str):
                candidates.extend(allowed_raw.split(","))
            elif isinstance(allowed_raw, (list, tuple, set)):
                for entry in allowed_raw:
                    if entry is None:
                        continue
                    candidates.extend(str(entry).split(","))
            else:
                raise ValueError(
                    (
                        "domain_content_rules['{name}'].allowed_types must be a string or sequence,"
                        " got {value!r}"
                    ).format(name=host, value=allowed_raw)
                )
            allowed: List[str] = []
            for candidate in candidates:
                token = candidate.strip()
                if not token:
                    continue
                token_lower = token.lower()
                if token_lower not in allowed:
                    allowed.append(token_lower)
            if allowed:
                policy["allowed_types"] = tuple(allowed)

        if policy:
            normalized[host_key] = policy
            if host_key.startswith("www."):
                bare = host_key[4:]
                if bare and bare not in normalized:
                    normalized[bare] = policy
            else:
                prefixed = f"www.{host_key}"
                normalized.setdefault(prefixed, policy)

    return normalized


@dataclass
class ResolverConfig:
    """Runtime configuration options applied across resolvers.

    Attributes:
        resolver_order: Ordered list of resolver names to execute.
        resolver_toggles: Mapping toggling individual resolvers on/off.
        max_attempts_per_work: Maximum number of resolver attempts per work item.
        timeout: Default HTTP timeout applied to resolvers.
        sleep_jitter: Random jitter added between retries.
        polite_headers: HTTP headers to apply for polite crawling.
        unpaywall_email: Contact email registered with Unpaywall.
        core_api_key: API key used for the CORE resolver.
        semantic_scholar_api_key: API key for Semantic Scholar resolver.
        doaj_api_key: API key for DOAJ resolver.
        resolver_timeouts: Resolver-specific timeout overrides.
        resolver_min_interval_s: Minimum interval between resolver _requests.
        domain_min_interval_s: Optional per-domain rate limits overriding resolver settings.
        enable_head_precheck: Toggle applying HEAD filtering before downloads.
        resolver_head_precheck: Per-resolver overrides for HEAD filtering behaviour.
        host_accept_overrides: Mapping of hostname to Accept header override.
        mailto: Contact email appended to polite headers and user agent string.
        max_concurrent_resolvers: Upper bound on concurrent resolver threads per work.
        max_concurrent_per_host: Upper bound on simultaneous downloads per hostname.
        enable_global_url_dedup: Enable global URL deduplication across works when True.
        domain_token_buckets: Mapping of hostname to token bucket parameters.
        domain_bytes_budget: Mapping of hostname to maximum bytes permitted this run.
        domain_content_rules: Mapping of hostname to MIME allow-lists and max-bytes caps.
        resolver_circuit_breakers: Mapping of resolver name to breaker thresholds/cooldowns.

    Notes:
        ``enable_head_precheck`` toggles inexpensive HEAD lookups before downloads
        to filter obvious HTML responses. ``resolver_head_precheck`` allows
        per-resolver overrides when specific providers reject HEAD _requests.
        ``max_concurrent_resolvers`` bounds the number of resolver threads used
        per work while still respecting configured rate limits. ``max_concurrent_per_host``
        limits simultaneous downloads hitting the same hostname across workers.

    Examples:
        >>> config = ResolverConfig()
        >>> config.max_attempts_per_work
        25
    """

    resolver_order: List[str] = field(default_factory=lambda: list(DEFAULT_RESOLVER_ORDER))
    resolver_toggles: Dict[str, bool] = field(
        default_factory=lambda: dict(_DEFAULT_RESOLVER_TOGGLES)
    )
    max_attempts_per_work: int = 25
    timeout: float = 30.0
    sleep_jitter: float = 0.35
    polite_headers: Dict[str, str] = field(default_factory=dict)
    unpaywall_email: Optional[str] = None
    core_api_key: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    doaj_api_key: Optional[str] = None
    resolver_timeouts: Dict[str, float] = field(default_factory=dict)
    resolver_min_interval_s: Dict[str, float] = field(default_factory=dict)
    domain_min_interval_s: Dict[str, float] = field(default_factory=dict)
    enable_head_precheck: bool = True
    resolver_head_precheck: Dict[str, bool] = field(default_factory=dict)
    head_precheck_host_overrides: Dict[str, bool] = field(default_factory=dict)
    host_accept_overrides: Dict[str, str] = field(default_factory=dict)
    mailto: Optional[str] = None
    max_concurrent_resolvers: int = 1
    max_concurrent_per_host: int = 3
    enable_global_url_dedup: bool = True
    domain_token_buckets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    domain_bytes_budget: Dict[str, int] = field(default_factory=dict)
    domain_content_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resolver_circuit_breakers: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Heuristic knobs (defaults preserve current CLI behaviour)
    sniff_bytes: int = DEFAULT_SNIFF_BYTES
    min_pdf_bytes: int = DEFAULT_MIN_PDF_BYTES
    tail_check_bytes: int = DEFAULT_TAIL_CHECK_BYTES

    def get_timeout(self, resolver_name: str) -> float:
        """Return the timeout to use for a resolver, falling back to defaults.

        Args:
            resolver_name: Name of the resolver requesting a timeout.

        Returns:
            float: Timeout value in seconds.
        """

        return self.resolver_timeouts.get(resolver_name, self.timeout)

    def get_content_policy(self, host: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the normalized content policy for ``host`` when configured."""

        if not host:
            return None
        host_key = host.strip().lower()
        if not host_key:
            return None
        return self.domain_content_rules.get(host_key)

    def is_enabled(self, resolver_name: str) -> bool:
        """Return ``True`` when the resolver is enabled for the current run.

        Args:
            resolver_name: Name of the resolver.

        Returns:
            bool: ``True`` if the resolver is enabled.
        """

        return self.resolver_toggles.get(resolver_name, True)

    def __post_init__(self) -> None:
        """Validate configuration fields and apply defaults for missing values.

        Args:
            self: Configuration instance requiring validation.

        Returns:
            None
        """

        if self.max_concurrent_resolvers < 1:
            raise ValueError(
                f"max_concurrent_resolvers must be >= 1, got {self.max_concurrent_resolvers}"
            )
        if self.max_concurrent_resolvers > 10:
            warnings.warn(
                (
                    "max_concurrent_resolvers="
                    f"{self.max_concurrent_resolvers} > 10 may violate rate limits. "
                    "Ensure resolver_min_interval_s is configured appropriately for all resolvers."
                ),
                UserWarning,
                stacklevel=2,
            )

        if self.max_concurrent_per_host < 0:
            raise ValueError(
                f"max_concurrent_per_host must be >= 0, got {self.max_concurrent_per_host}"
            )

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        for resolver_name, timeout_val in self.resolver_timeouts.items():
            if timeout_val <= 0:
                raise ValueError(
                    ("resolver_timeouts['{name}'] must be positive, got {value}").format(
                        name=resolver_name, value=timeout_val
                    )
                )

        for resolver_name, interval in self.resolver_min_interval_s.items():
            if interval < 0:
                raise ValueError(
                    ("resolver_min_interval_s['{name}'] must be non-negative, got {value}").format(
                        name=resolver_name, value=interval
                    )
                )

        normalized_domain_limits: Dict[str, float] = {}
        for host, interval in self.domain_min_interval_s.items():
            if interval < 0:
                raise ValueError(
                    ("domain_min_interval_s['{name}'] must be non-negative, got {value}").format(
                        name=host, value=interval
                    )
                )
            normalized_domain_limits[host.lower()] = interval
        if normalized_domain_limits:
            self.domain_min_interval_s = normalized_domain_limits

        if self.domain_bytes_budget:
            self.domain_bytes_budget = _normalise_domain_bytes_budget(self.domain_bytes_budget)
        if self.domain_content_rules:
            self.domain_content_rules = _normalise_domain_content_rules(self.domain_content_rules)

        if self.max_attempts_per_work < 1:
            raise ValueError(
                f"max_attempts_per_work must be >= 1, got {self.max_attempts_per_work}"
            )

        if self.host_accept_overrides:
            overrides: Dict[str, str] = {}
            for host, header in self.host_accept_overrides.items():
                if not host:
                    continue
                overrides[host.lower()] = str(header)
            self.host_accept_overrides = overrides

        if self.domain_token_buckets:
            buckets: Dict[str, Dict[str, float]] = {}
            for host, spec in self.domain_token_buckets.items():
                if not host or not isinstance(spec, dict):
                    continue
                rate = float(spec.get("rate_per_second", spec.get("rate", 1.0)))
                capacity = float(spec.get("capacity", spec.get("burst", 1.0)))
                if rate <= 0 or capacity <= 0:
                    raise ValueError(
                        (
                            "domain_token_buckets['{host}'] requires positive rate and capacity,"
                            " got rate={rate} capacity={capacity}"
                        ).format(host=host, rate=rate, capacity=capacity)
                    )
                payload: Dict[str, float] = {
                    "rate_per_second": rate,
                    "capacity": capacity,
                }
                if "breaker_threshold" in spec or "failure_threshold" in spec:
                    threshold = int(spec.get("breaker_threshold", spec.get("failure_threshold", 5)))
                    if threshold < 1:
                        raise ValueError(
                            (
                                "domain_token_buckets['{host}'] breaker_threshold must be >= 1, got {value}"
                            ).format(host=host, value=threshold)
                        )
                    payload["breaker_threshold"] = threshold
                if "breaker_cooldown" in spec or "cooldown_seconds" in spec:
                    cooldown = float(
                        spec.get("breaker_cooldown", spec.get("cooldown_seconds", 60.0))
                    )
                    if cooldown < 0:
                        raise ValueError(
                            (
                                "domain_token_buckets['{host}'] breaker_cooldown must be >= 0, got {value}"
                            ).format(host=host, value=cooldown)
                        )
                    payload["breaker_cooldown"] = cooldown
                buckets[host.lower()] = payload
            self.domain_token_buckets = buckets

        if self.resolver_circuit_breakers:
            breakers: Dict[str, Dict[str, float]] = {}
            for resolver_name, spec in self.resolver_circuit_breakers.items():
                if not resolver_name or not isinstance(spec, dict):
                    continue
                threshold = int(spec.get("failure_threshold", spec.get("threshold", 5)))
                cooldown = float(spec.get("cooldown_seconds", spec.get("cooldown", 60.0)))
                if threshold < 1:
                    raise ValueError(
                        (
                            "resolver_circuit_breakers['{name}'] failure_threshold must be >= 1, got {value}"
                        ).format(name=resolver_name, value=threshold)
                    )
                if cooldown < 0:
                    raise ValueError(
                        (
                            "resolver_circuit_breakers['{name}'] cooldown_seconds must be >= 0, got {value}"
                        ).format(name=resolver_name, value=cooldown)
                    )
                breakers[resolver_name] = {
                    "failure_threshold": threshold,
                    "cooldown_seconds": cooldown,
                }
            self.resolver_circuit_breakers = breakers


def read_resolver_config(path: Path) -> Dict[str, Any]:
    """Read resolver configuration from JSON or YAML files."""

    text = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install PyYAML to load YAML resolver configs, or provide JSON."
            ) from exc
        return yaml.safe_load(text) or {}

    if ext in {".json", ".jsn"}:
        return json.loads(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Unable to parse resolver config. Install PyYAML or provide JSON."
            ) from exc
        return yaml.safe_load(text) or {}


def _seed_resolver_toggle_defaults(config: "ResolverConfig", resolver_names: Sequence[str]) -> None:
    """Ensure resolver toggles include defaults for every known resolver."""

    for name in resolver_names:
        default_enabled = DEFAULT_RESOLVER_TOGGLES.get(name, True)
        config.resolver_toggles.setdefault(name, default_enabled)


def apply_config_overrides(
    config: "ResolverConfig",
    data: Dict[str, Any],
    resolver_names: Sequence[str],
) -> None:
    """Apply overrides from configuration data onto a ResolverConfig."""

    for field_name in (
        "resolver_order",
        "resolver_toggles",
        "max_attempts_per_work",
        "timeout",
        "sleep_jitter",
        "polite_headers",
        "unpaywall_email",
        "core_api_key",
        "semantic_scholar_api_key",
        "doaj_api_key",
        "resolver_timeouts",
        "resolver_min_interval_s",
        "mailto",
        "resolver_head_precheck",
        "head_precheck_host_overrides",
        "host_accept_overrides",
        "domain_token_buckets",
        "resolver_circuit_breakers",
        "max_concurrent_per_host",
        "domain_bytes_budget",
        "domain_content_rules",
    ):
        if field_name in data and data[field_name] is not None:
            if field_name == "domain_bytes_budget":
                setattr(config, field_name, _normalise_domain_bytes_budget(data[field_name]))
            elif field_name == "domain_content_rules":
                setattr(config, field_name, _normalise_domain_content_rules(data[field_name]))
            else:
                setattr(config, field_name, data[field_name])

    if "resolver_rate_limits" in data:
        raise ValueError(
            "resolver_rate_limits is no longer supported. Rename entries to resolver_min_interval_s."
        )

    _seed_resolver_toggle_defaults(config, resolver_names)


def load_resolver_config(
    args: argparse.Namespace,
    resolver_names: Sequence[str],
    resolver_order_override: Optional[List[str]] = None,
) -> "ResolverConfig":
    """Construct resolver configuration combining CLI, config files, and env vars."""

    config = ResolverConfig()
    if args.resolver_config:
        config_data = read_resolver_config(Path(args.resolver_config))
        apply_config_overrides(config, config_data, resolver_names)

    config.unpaywall_email = (
        getattr(args, "unpaywall_email", None)
        or config.unpaywall_email
        or os.getenv("UNPAYWALL_EMAIL")
        or getattr(args, "mailto", None)
    )
    config.core_api_key = (
        getattr(args, "core_api_key", None) or config.core_api_key or os.getenv("CORE_API_KEY")
    )
    config.semantic_scholar_api_key = (
        getattr(args, "semantic_scholar_api_key", None)
        or config.semantic_scholar_api_key
        or os.getenv("S2_API_KEY")
    )
    config.doaj_api_key = (
        getattr(args, "doaj_api_key", None) or config.doaj_api_key or os.getenv("DOAJ_API_KEY")
    )
    config.mailto = getattr(args, "mailto", None) or config.mailto

    if getattr(args, "max_resolver_attempts", None):
        config.max_attempts_per_work = args.max_resolver_attempts
    if getattr(args, "resolver_timeout", None):
        config.timeout = args.resolver_timeout
    if hasattr(args, "concurrent_resolvers") and args.concurrent_resolvers is not None:
        config.max_concurrent_resolvers = args.concurrent_resolvers
    if hasattr(args, "max_concurrent_per_host") and args.max_concurrent_per_host is not None:
        config.max_concurrent_per_host = args.max_concurrent_per_host

    if resolver_order_override:
        ordered: List[str] = []
        for name in resolver_order_override:
            if name not in ordered:
                ordered.append(name)
        for name in resolver_names:
            if name not in ordered:
                ordered.append(name)
        config.resolver_order = ordered

    for disabled in getattr(args, "disable_resolver", []) or []:
        config.resolver_toggles[disabled] = False

    for enabled in getattr(args, "enable_resolver", []) or []:
        config.resolver_toggles[enabled] = True

    _seed_resolver_toggle_defaults(config, resolver_names)

    if hasattr(args, "global_url_dedup") and args.global_url_dedup is not None:
        config.enable_global_url_dedup = args.global_url_dedup

    if getattr(args, "domain_min_interval", None):
        domain_limits = dict(config.domain_min_interval_s)
        for domain, interval in args.domain_min_interval:
            domain_limits[domain] = interval
        config.domain_min_interval_s = domain_limits
    if getattr(args, "domain_bytes_budget", None):
        budget_map = dict(config.domain_bytes_budget)
        for domain, limit in args.domain_bytes_budget:
            budget_map[domain] = limit
        config.domain_bytes_budget = _normalise_domain_bytes_budget(budget_map)
    if getattr(args, "domain_token_bucket", None):
        bucket_map: Dict[str, Dict[str, float]] = dict(config.domain_token_buckets)
        for domain, spec in args.domain_token_bucket:
            bucket_map[domain] = dict(spec)
        config.domain_token_buckets = bucket_map

    headers = dict(config.polite_headers)
    existing_mailto = headers.get("mailto")
    mailto_value = config.mailto or existing_mailto
    base_agent = headers.get("User-Agent") or "DocsToKGDownloader/1.0"
    if mailto_value:
        config.mailto = config.mailto or mailto_value
        headers["mailto"] = mailto_value
        user_agent = f"DocsToKGDownloader/1.0 (+{mailto_value}; mailto:{mailto_value})"
    else:
        headers.pop("mailto", None)
        user_agent = base_agent
    headers["User-Agent"] = user_agent
    accept_override = getattr(args, "accept", None)
    if accept_override:
        headers["Accept"] = accept_override
    elif not headers.get("Accept"):
        headers["Accept"] = "application/pdf, text/html;q=0.9, */*;q=0.8"
    config.polite_headers = headers

    if hasattr(args, "head_precheck") and args.head_precheck is not None:
        config.enable_head_precheck = args.head_precheck

    config.resolver_min_interval_s.setdefault("unpaywall", 1.0)

    return config


@dataclass(frozen=True)
class AttemptRecord:
    """Structured log record describing a resolver attempt.

    Attributes:
        run_id: Identifier shared across a downloader run.
        work_id: Identifier of the work being processed.
        resolver_name: Name of the resolver that produced the record.
        resolver_order: Ordinal position of the resolver in the pipeline.
        url: Candidate URL that was attempted.
        status: :class:`Classification` describing the attempt result.
        http_status: HTTP status code (when available).
        content_type: Response content type.
        elapsed_ms: Approximate elapsed time for the attempt in milliseconds.
        resolver_wall_time_ms: Wall-clock time spent inside the resolver including
            rate limiting, measured in milliseconds.
        reason: Optional :class:`ReasonCode` describing failure or skip reason.
        metadata: Arbitrary metadata supplied by the resolver.
        sha256: SHA-256 digest of downloaded content, when available.
        content_length: Size of the downloaded content in bytes.
        dry_run: Flag indicating whether the attempt occurred in dry-run mode.
        retry_after: Optional cooldown seconds suggested by the upstream service.

    Examples:
        >>> AttemptRecord(
        ...     work_id="W1",
        ...     resolver_name="unpaywall",
        ...     resolver_order=1,
        ...     url="https://example.org/pdf",
        ...     status="pdf",
        ...     http_status=200,
        ...     content_type="application/pdf",
        ...     elapsed_ms=120.5,
        ... )
    """

    work_id: str
    resolver_name: str
    resolver_order: Optional[int]
    url: Optional[str]
    status: Classification
    http_status: Optional[int]
    content_type: Optional[str]
    elapsed_ms: Optional[float]
    reason: Optional[ReasonCode | str] = None
    reason_detail: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sha256: Optional[str] = None
    content_length: Optional[int] = None
    dry_run: bool = False
    resolver_wall_time_ms: Optional[float] = None
    retry_after: Optional[float] = None
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        status_value: Classification | str
        if not isinstance(self.status, Classification):
            converted = Classification.from_wire(self.status)
            if (
                isinstance(self.status, str)
                and converted is Classification.UNKNOWN
                and self.status.lower() not in {member.value for member in Classification}
            ):
                status_value = self.status
            else:
                status_value = converted
        else:
            status_value = self.status
        object.__setattr__(self, "status", status_value)

        reason_value = self.reason
        if isinstance(reason_value, ReasonCode):
            normalized_reason: ReasonCode | str | None = reason_value
        elif reason_value is not None:
            normalized = str(reason_value).replace("-", "_")
            candidate = ReasonCode.from_wire(normalized)
            if candidate is not ReasonCode.UNKNOWN or normalized == ReasonCode.UNKNOWN.value:
                normalized_reason = candidate
            else:
                normalized_reason = str(reason_value)
        else:
            normalized_reason = None
        object.__setattr__(self, "reason", normalized_reason)


@dataclass
class DownloadOutcome:
    """Outcome of a resolver download attempt.

    Attributes:
        classification: Classification label describing the outcome (e.g., 'pdf').
        path: Local filesystem path to the stored artifact.
        http_status: HTTP status code when available.
        content_type: Content type reported by the server.
        elapsed_ms: Time spent downloading in milliseconds.
        reason: Optional :class:`ReasonCode` describing failures.
        reason_detail: Optional human-readable diagnostic detail.
        sha256: SHA-256 digest of the downloaded content.
        content_length: Size of the downloaded content in bytes.
        etag: HTTP ETag header value when provided.
        last_modified: HTTP Last-Modified timestamp.
        extracted_text_path: Optional path to extracted text artefacts.
        retry_after: Optional cooldown value derived from HTTP ``Retry-After`` headers.

    Examples:
        >>> DownloadOutcome(classification="pdf", path="pdfs/sample.pdf", http_status=200,
        ...                 content_type="application/pdf", elapsed_ms=150.0)
    """

    classification: Classification
    path: Optional[str] = None
    http_status: Optional[int] = None
    content_type: Optional[str] = None
    elapsed_ms: Optional[float] = None
    reason: Optional[ReasonCode | str] = None
    reason_detail: Optional[str] = None
    sha256: Optional[str] = None
    content_length: Optional[int] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    extracted_text_path: Optional[str] = None
    retry_after: Optional[float] = None
    error: Optional[str] = None

    @property
    def is_pdf(self) -> bool:
        """Return ``True`` when the classification represents a PDF.

        Args:
            self: Download outcome to evaluate.

        Returns:
            bool: ``True`` if the outcome corresponds to a PDF download.
        """

        return self.classification in PDF_LIKE

    def __post_init__(self) -> None:
        if not isinstance(self.classification, Classification):
            self.classification = Classification.from_wire(self.classification)
        reason_value = self.reason
        if isinstance(reason_value, ReasonCode):
            normalized_reason: ReasonCode | str = reason_value
        elif reason_value is not None:
            normalized = str(reason_value).replace("-", "_")
            candidate = ReasonCode.from_wire(normalized)
            if candidate is not ReasonCode.UNKNOWN or normalized == ReasonCode.UNKNOWN.value:
                normalized_reason = candidate
            else:
                normalized_reason = str(reason_value)
        else:
            normalized_reason = None
        self.reason = normalized_reason


@dataclass
class PipelineResult:
    """Aggregate result returned by the resolver pipeline.

    Attributes:
        success: Indicates whether the pipeline found a suitable asset.
        resolver_name: Resolver that produced the successful result.
        url: URL that was ultimately fetched.
        outcome: Download outcome associated with the result.
        html_paths: Collected HTML artefacts from the pipeline.
        failed_urls: Candidate URLs that failed during this run.
        reason: Optional reason string explaining failures.

    Examples:
        >>> PipelineResult(success=True, resolver_name="unpaywall", url="https://example")
    """

    success: bool
    resolver_name: Optional[str] = None
    url: Optional[str] = None
    outcome: Optional[DownloadOutcome] = None
    html_paths: List[str] = field(default_factory=list)
    failed_urls: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    reason_detail: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.reason, ReasonCode):
            self.reason = self.reason.value.replace("_", "-")


class Resolver(Protocol):
    """Protocol that resolver implementations must follow.

    Attributes:
        name: Resolver identifier used within configuration.

    Examples:
        Concrete implementations are provided in classes such as
        :class:`UnpaywallResolver` and :class:`CrossrefResolver` below.
    """

    name: str

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` if this resolver should run for the given artifact.

        Args:
            config: Resolver configuration options.
            artifact: Work artifact under consideration.

        Returns:
            bool: ``True`` when the resolver should run for the artifact.
        """

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs or events for the given artifact.

        Args:
            session: HTTP session used for outbound _requests.
            config: Resolver configuration.
            artifact: Work artifact describing the current item.

        Returns:
            Iterable[ResolverResult]: Stream of download candidates or events.
        """


@dataclass
class ResolverMetrics:
    """Lightweight metrics collector for resolver execution.

    Attributes:
        attempts: Counter of attempts per resolver.
        successes: Counter of successful PDF downloads per resolver.
        html: Counter of HTML-only results per resolver.
        skips: Counter of skip events keyed by resolver and reason.

    Examples:
        >>> metrics = ResolverMetrics()
        >>> metrics.record_attempt("unpaywall", DownloadOutcome("pdf", None, 200, None, 10.0))
        >>> metrics.summary()["attempts"]["unpaywall"]
        1
    """

    attempts: Counter = field(default_factory=Counter)
    successes: Counter = field(default_factory=Counter)
    html: Counter = field(default_factory=Counter)
    skips: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)
    latency_ms: defaultdict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    status_counts: defaultdict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    error_reasons: defaultdict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record_attempt(self, resolver_name: str, outcome: DownloadOutcome) -> None:
        """Record a resolver attempt and update success/html counters.

        Args:
            resolver_name: Name of the resolver that executed.
            outcome: Download outcome produced by the resolver.

        Returns:
            None
        """

        with self._lock:
            self.attempts[resolver_name] += 1
            if outcome.classification is Classification.HTML:
                self.html[resolver_name] += 1
            if outcome.is_pdf:
                self.successes[resolver_name] += 1
            classification_key = outcome.classification.value
            self.status_counts[resolver_name][classification_key] += 1
            if outcome.elapsed_ms is not None:
                self.latency_ms[resolver_name].append(float(outcome.elapsed_ms))
            reason_code = outcome.reason
            if reason_code is None and outcome.classification not in PDF_LIKE:
                reason_code = ReasonCode.from_wire(classification_key)
            if reason_code:
                key = reason_code.value if isinstance(reason_code, ReasonCode) else str(reason_code)
                self.error_reasons[resolver_name][key] += 1

    def record_skip(self, resolver_name: str, reason: str) -> None:
        """Record a skip event for a resolver with a reason tag.

        Args:
            resolver_name: Resolver that was skipped.
            reason: Short description explaining the skip.

        Returns:
            None
        """

        with self._lock:
            key = f"{resolver_name}:{reason}"
            self.skips[key] += 1

    def record_failure(self, resolver_name: str) -> None:
        """Record a resolver failure occurrence.

        Args:
            resolver_name: Resolver that raised an exception during execution.

        Returns:
            None
        """

        with self._lock:
            self.failures[resolver_name] += 1

    def summary(self) -> Dict[str, Any]:
        """Return aggregated metrics summarizing resolver behaviour.

        Args:
            self: Metrics collector instance aggregating resolver statistics.

        Returns:
            Dict[str, Any]: Snapshot of attempts, successes, HTML hits, and skips.

        Examples:
            >>> metrics = ResolverMetrics()
            >>> metrics.record_attempt("unpaywall", DownloadOutcome("pdf", None, 200, None, 10.0))
            >>> metrics.summary()["attempts"]["unpaywall"]
            1
        """

        def _percentile(sorted_values: List[float], percentile: float) -> float:
            if not sorted_values:
                return 0.0
            if len(sorted_values) == 1:
                return sorted_values[0]
            k = (len(sorted_values) - 1) * (percentile / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_values[int(k)]
            d0 = sorted_values[f] * (c - k)
            d1 = sorted_values[c] * (k - f)
            return d0 + d1

        with self._lock:
            latency_summary: Dict[str, Dict[str, float]] = {}
            for resolver_name, samples in self.latency_ms.items():
                if not samples:
                    continue
                ordered = sorted(samples)
                count = len(samples)
                total = sum(samples)
                latency_summary[resolver_name] = {
                    "count": count,
                    "mean_ms": total / count,
                    "min_ms": ordered[0],
                    "p50_ms": _percentile(ordered, 50.0),
                    "p95_ms": _percentile(ordered, 95.0),
                    "p99_ms": _percentile(ordered, 99.0),
                    "max_ms": ordered[-1],
                }

            status_summary = {
                resolver: dict(counter)
                for resolver, counter in self.status_counts.items()
                if counter
            }
            classification_totals: Dict[str, int] = defaultdict(int)
            for counter in self.status_counts.values():
                for status, count in counter.items():
                    classification_totals[status] += count
            reason_summary = {
                resolver: [
                    {"reason": reason, "count": count} for reason, count in counter.most_common(5)
                ]
                for resolver, counter in self.error_reasons.items()
                if counter
            }
            reason_totals: Dict[str, int] = defaultdict(int)
            for counter in self.error_reasons.values():
                for reason, count in counter.items():
                    reason_totals[reason] += count

            return {
                "attempts": dict(self.attempts),
                "successes": dict(self.successes),
                "html": dict(self.html),
                "skips": dict(self.skips),
                "failures": dict(self.failures),
                "latency_ms": latency_summary,
                "status_counts": status_summary,
                "error_reasons": reason_summary,
                "classification_totals": dict(classification_totals),
                "reason_totals": dict(reason_totals),
            }


# --- Shared Helpers ---


def _fetch_semantic_scholar_data(
    session: _requests.Session,
    config: ResolverConfig,
    doi: str,
) -> Dict[str, Any]:
    """Return Semantic Scholar metadata for ``doi`` using configured headers."""

    headers = dict(config.polite_headers)
    if config.semantic_scholar_api_key:
        headers["x-api-key"] = config.semantic_scholar_api_key
    response = request_with_retries(
        session,
        "GET",
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}",
        params={"fields": "title,openAccessPdf"},
        timeout=config.get_timeout("semantic_scholar"),
        headers=headers,
    )
    try:
        if response.status_code != 200:
            error = _requests.HTTPError(f"Semantic Scholar HTTPError: {response.status_code}")
            error.response = response
            raise error
        return response.json()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _fetch_unpaywall_data(
    session: _requests.Session,
    config: ResolverConfig,
    doi: str,
) -> Dict[str, Any]:
    """Return Unpaywall metadata for ``doi`` using configured headers."""

    endpoint = f"https://api.unpaywall.org/v2/{quote(doi)}"
    headers = dict(config.polite_headers)
    params = {"email": config.unpaywall_email} if config.unpaywall_email else None
    response = request_with_retries(
        session,
        "GET",
        endpoint,
        params=params,
        timeout=config.get_timeout("unpaywall"),
        headers=headers,
    )
    try:
        if response.status_code != 200:
            error = _requests.HTTPError(f"Unpaywall HTTPError: {response.status_code}")
            error.response = response
            raise error
        return response.json()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


DownloadFunc = Callable[..., DownloadOutcome]


# --- Resolver Registry ---


class ResolverRegistry:
    """Registry tracking resolver classes by their ``name`` attribute.

    Attributes:
        _providers: Mapping of resolver names to resolver classes.

    Examples:
        >>> ResolverRegistry.register(type("Tmp", (RegisteredResolver,), {"name": "tmp"}))  # doctest: +SKIP
        <class 'Tmp'>
    """

    _providers: Dict[str, Type[Resolver]] = {}

    @classmethod
    def register(cls, resolver_cls: Type[Resolver]) -> Type[Resolver]:
        """Register a resolver class under its declared ``name`` attribute.

        Args:
            resolver_cls: Resolver implementation to register.

        Returns:
            Type[Resolver]: The registered resolver class for chaining.
        """
        name = getattr(resolver_cls, "name", None)
        if not name:
            raise ValueError(f"Resolver class {resolver_cls.__name__} missing 'name' attribute")
        cls._providers[name] = resolver_cls
        return resolver_cls

    @classmethod
    def create_default(cls) -> List[Resolver]:
        """Instantiate resolver instances in priority order.

        Args:
            None

        Returns:
            List[Resolver]: Resolver instances ordered by default priority.

        Raises:
            None.
        """
        instances: List[Resolver] = []
        seen: set[str] = set()
        for name in DEFAULT_RESOLVER_ORDER:
            resolver_cls = cls._providers.get(name)
            if resolver_cls is not None:
                instances.append(resolver_cls())
                seen.add(name)
        for name in sorted(cls._providers):
            if name not in seen:
                instances.append(cls._providers[name]())
        return instances


class RegisteredResolver:
    """Mixin ensuring subclasses register with :class:`ResolverRegistry`.

    Attributes:
        None: Subclasses inherit registration behaviour automatically.

    Examples:
        >>> class ExampleResolver(RegisteredResolver):
        ...     name = "example"
        ...     def is_enabled(self, config, artifact):
        ...         return True
        ...     def iter_urls(self, session, config, artifact):
        ...         yield ResolverResult(url="https://example.org")
    """

    def __init_subclass__(cls, register: bool = True, **kwargs: Any) -> None:  # type: ignore[override]
        super().__init_subclass__(**kwargs)
        if register:
            ResolverRegistry.register(cls)  # type: ignore[arg-type]


class ApiResolverBase(RegisteredResolver, register=False):
    """Shared helper for resolvers interacting with JSON-based HTTP APIs.

    Attributes:
        name: Resolver identifier used for logging and registration.

    Examples:
        >>> class ExampleApiResolver(ApiResolverBase):
        ...     name = "example-api"
        ...     def iter_urls(self, session, config, artifact):
        ...         payload, result = self._request_json(
        ...             session,
        ...             \"GET\",
        ...             \"https://example.org/api\",
        ...             config=config,
        ...         )
        ...         if payload:
        ...             yield result  # doctest: +SKIP
    """

    def _request_json(
        self,
        session: _requests.Session,
        method: str,
        url: str,
        *,
        config: ResolverConfig,
        timeout: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[Any], Optional[ResolverResult]]:
        timeout_value = timeout if timeout is not None else config.get_timeout(self.name)
        request_headers: Dict[str, str] = dict(config.polite_headers)
        if headers:
            request_headers.update(headers)

        kwargs.setdefault("allow_redirects", True)

        try:
            response = request_with_retries(
                session,
                method,
                url,
                params=params,
                json=json,
                headers=request_headers,
                timeout=timeout_value,
                **kwargs,
            )
        except _requests.Timeout as exc:
            return (
                None,
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={
                        "url": url,
                        "timeout": timeout_value,
                        "error": str(exc),
                    },
                ),
            )
        except _requests.ConnectionError as exc:
            return (
                None,
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"url": url, "error": str(exc)},
                ),
            )
        except _requests.RequestException as exc:
            return (
                None,
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"url": url, "error": str(exc)},
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Unexpected error issuing %s request to %s", method, url)
            return (
                None,
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={
                        "url": url,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                ),
            )

        try:
            if response.status_code != 200:
                return (
                    None,
                    ResolverResult(
                        url=None,
                        event="error",
                        event_reason="http-error",
                        http_status=response.status_code,
                        metadata={
                            "url": url,
                            "error_detail": f"{getattr(self, 'api_display_name', self.name)} API returned {response.status_code}",
                        },
                    ),
                )
            data = response.json()
        except ValueError as json_err:
            preview = response.text[:200] if hasattr(response, "text") else ""
            return (
                None,
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
                    metadata={
                        "url": url,
                        "error_detail": str(json_err),
                        "content_preview": preview,
                    },
                ),
            )
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        return data, None


def _absolute_url(base: str, href: str) -> str:
    """Resolve relative ``href`` values against ``base`` to obtain absolute URLs."""

    parsed = urlparse(href)
    if parsed.scheme and parsed.netloc:
        return href
    base_parts = urlparse(base)
    if base_parts.scheme and base_parts.netloc:
        origin = f"{base_parts.scheme}://{base_parts.netloc}/"
        return urljoin(origin, href)
    return urljoin(base, href)


def _collect_candidate_urls(node: object, results: List[str]) -> None:
    """Recursively collect HTTP(S) URLs from nested response payloads."""

    if isinstance(node, dict):
        for value in node.values():
            _collect_candidate_urls(value, results)
    elif isinstance(node, list):
        for item in node:
            _collect_candidate_urls(item, results)
    elif isinstance(node, str) and node.lower().startswith("http"):
        results.append(node)


def find_pdf_via_meta(soup: "BeautifulSoup", base_url: str) -> Optional[str]:
    """Return PDF URL declared via ``citation_pdf_url`` meta tags.

    Args:
        soup: Parsed HTML document to inspect for metadata tags.
        base_url: URL used to resolve relative PDF links.

    Returns:
        Optional[str]: Absolute PDF URL when one is advertised; otherwise ``None``.
    """

    tag = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if tag is None:
        return None
    href = (tag.get("content") or "").strip()
    if not href:
        return None
    return _absolute_url(base_url, href)


def find_pdf_via_link(soup: "BeautifulSoup", base_url: str) -> Optional[str]:
    """Return PDF URL referenced via ``<link rel=\"alternate\" type=\"application/pdf\">``.

    Args:
        soup: Parsed HTML document to inspect for ``<link>`` elements.
        base_url: URL used to resolve relative ``href`` attributes.

    Returns:
        Optional[str]: Absolute PDF URL if link metadata is present; otherwise ``None``.
    """

    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel") or []).lower()
        typ = (link.get("type") or "").lower()
        href = (link.get("href") or "").strip()
        if "alternate" in rel and "application/pdf" in typ and href:
            return _absolute_url(base_url, href)
    return None


def find_pdf_via_anchor(soup: "BeautifulSoup", base_url: str) -> Optional[str]:
    """Return PDF URL inferred from anchor elements mentioning PDFs.

    Args:
        soup: Parsed HTML document to search for anchor tags referencing PDFs.
        base_url: URL used to resolve relative anchor targets.

    Returns:
        Optional[str]: Absolute PDF URL when an anchor appears to reference one.
    """

    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        if not href:
            continue
        text = (anchor.get_text() or "").strip().lower()
        href_lower = href.lower()
        if href_lower.endswith(".pdf") or "pdf" in text:
            candidate = _absolute_url(base_url, href)
            if candidate.lower().endswith(".pdf"):
                return candidate
    return None


def default_resolvers() -> List[Resolver]:
    """Instantiate the default resolver stack in priority order.

    Args:
        None

    Returns:
        List[Resolver]: Resolver instances ordered according to ``DEFAULT_RESOLVER_ORDER``.
    """

    return ResolverRegistry.create_default()


# --- Resolver implementations ---


class ArxivResolver(RegisteredResolver):
    """Resolve arXiv preprints using arXiv identifier lookups.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> ArxivResolver().name
        'arxiv'
    """

    name = "arxiv"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.arxiv_id)

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        arxiv_id = artifact.arxiv_id
        if not arxiv_id:
            yield ResolverResult(url=None, event="skipped", event_reason="no-arxiv-id")
            return
        arxiv_id = strip_prefix(arxiv_id, "arxiv:")
        yield ResolverResult(
            url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            metadata={"identifier": arxiv_id},
        )


class CoreResolver(ApiResolverBase):
    """Resolve PDFs using the CORE API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> CoreResolver().name
        'core'
    """

    name = "core"
    api_display_name = "CORE"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(config.core_api_key and artifact.doi)

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        headers = {"Authorization": f"Bearer {config.core_api_key}"}
        data, error = self._request_json(
            session,
            "GET",
            "https://api.core.ac.uk/v3/search/works",
            config=config,
            params={"q": f'doi:"{doi}"', "page": 1, "pageSize": 3},
            headers=headers,
        )
        if error:
            yield error
            return
        results = (data.get("results") if isinstance(data, dict) else None) or []
        for hit in results:
            if not isinstance(hit, dict):
                continue
            url = hit.get("downloadUrl") or hit.get("pdfDownloadLink")
            if url:
                yield ResolverResult(url=url, metadata={"source": "core"})
            for entry in hit.get("fullTextLinks") or []:
                if isinstance(entry, dict):
                    href = entry.get("url") or entry.get("link")
                    if href and href.lower().endswith(".pdf"):
                        yield ResolverResult(url=href, metadata={"source": "core"})


class CrossrefResolver(ApiResolverBase):
    """Resolve candidate URLs from the Crossref metadata API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> CrossrefResolver().name
        'crossref'
    """

    name = "crossref"
    api_display_name = "Crossref"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        email = config.mailto or config.unpaywall_email
        params = {"mailto": email} if email else None
        data, error = self._request_json(
            session,
            "GET",
            f"https://api.crossref.org/works/{quote(doi)}",
            config=config,
            params=params,
        )
        if error:
            yield error
            return

        message = (data.get("message") if isinstance(data, dict) else None) or {}
        link_section = message.get("link") or []
        if not isinstance(link_section, list):
            link_section = []

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for entry in link_section:
            if not isinstance(entry, dict):
                continue
            url = entry.get("URL")
            content_type = entry.get("content-type")
            ctype = (content_type or "").lower()
            if url and ctype in {"application/pdf", "application/x-pdf", "text/html"}:
                candidates.append((url, {"content_type": content_type}))

        for url in dedupe([candidate_url for candidate_url, _ in candidates]):
            for candidate_url, metadata in candidates:
                if candidate_url == url:
                    yield ResolverResult(url=url, metadata=metadata)
                    break


class DoajResolver(ApiResolverBase):
    """Resolve Open Access links using the DOAJ API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> DoajResolver().name
        'doaj'
    """

    name = "doaj"
    api_display_name = "DOAJ"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        extra_headers: Dict[str, str] = {}
        if config.doaj_api_key:
            extra_headers["X-API-KEY"] = config.doaj_api_key
        data, error = self._request_json(
            session,
            "GET",
            "https://doaj.org/api/v2/search/articles/",
            config=config,
            params={"pageSize": 3, "q": f'doi:"{doi}"'},
            headers=extra_headers,
        )
        if error:
            yield error
            return
        candidates = []
        for result in data.get("results", []) or []:
            bibjson = (result or {}).get("bibjson", {})
            for link in bibjson.get("link", []) or []:
                if not isinstance(link, dict):
                    continue
                url = link.get("url")
                if url and url.lower().endswith(".pdf"):
                    candidates.append(url)
        for url in dedupe(candidates):
            yield ResolverResult(url=url, metadata={"source": "doaj"})


class EuropePmcResolver(ApiResolverBase):
    """Resolve Open Access links via the Europe PMC REST API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> EuropePmcResolver().name
        'europe_pmc'
    """

    name = "europe_pmc"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        data, error = self._request_json(
            session,
            "GET",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            config=config,
            params={"query": f'DOI:"{doi}"', "format": "json", "pageSize": 3},
        )
        if error:
            if error.event_reason == "http-error":
                return
            yield error
            return
        candidates: List[str] = []
        for result in (data.get("resultList", {}) or {}).get("result", []) or []:
            full_text = result.get("fullTextUrlList", {}) or {}
            for entry in full_text.get("fullTextUrl", []) or []:
                if not isinstance(entry, dict):
                    continue
                if (entry.get("documentStyle") or "").lower() != "pdf":
                    continue
                url = entry.get("url")
                if url:
                    candidates.append(url)
        for url in dedupe(candidates):
            yield ResolverResult(url=url, metadata={"source": "europe_pmc"})


class FigshareResolver(ApiResolverBase):
    """Resolve Figshare repository metadata into download URLs.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> FigshareResolver().name
        'figshare'
    """

    name = "figshare"
    api_display_name = "Figshare"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return

        extra_headers = {"Content-Type": "application/json"}
        data, error = self._request_json(
            session,
            "POST",
            "https://api.figshare.com/v2/articles/search",
            config=config,
            json={"search_for": f':doi: "{doi}"', "page": 1, "page_size": 3},
            headers=extra_headers,
        )
        if error:
            yield error
            return

        if isinstance(data, list):
            articles = data
        else:
            LOGGER.warning(
                "Figshare API returned non-list articles payload: %s",
                type(data).__name__ if data is not None else "None",
            )
            articles = []

        for article in articles:
            if not isinstance(article, dict):
                LOGGER.warning("Skipping malformed Figshare article: %r", article)
                continue
            files = article.get("files", []) or []
            if not isinstance(files, list):
                LOGGER.warning("Skipping Figshare article with invalid files payload: %r", files)
                continue
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    LOGGER.warning("Skipping non-dict Figshare file entry: %r", file_entry)
                    continue
                filename = (file_entry.get("name") or "").lower()
                download_url = file_entry.get("download_url")

                if filename.endswith(".pdf") and download_url:
                    yield ResolverResult(
                        url=download_url,
                        metadata={
                            "source": "figshare",
                            "article_id": article.get("id"),
                            "filename": file_entry.get("name"),
                        },
                    )


class HalResolver(ApiResolverBase):
    """Resolve publications from the HAL open archive.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> HalResolver().name
        'hal'
    """

    name = "hal"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        data, error = self._request_json(
            session,
            "GET",
            "https://api.archives-ouvertes.fr/search/",
            config=config,
            params={"q": f"doiId_s:{doi}", "fl": "fileMain_s,file_s"},
        )
        if error:
            yield error
            return
        docs = (data.get("response") or {}).get("docs") or []
        urls: List[str] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            main = doc.get("fileMain_s")
            if isinstance(main, str):
                urls.append(main)
            files = doc.get("file_s")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, str):
                        urls.append(item)
        for url in dedupe(urls):
            if url.lower().endswith(".pdf"):
                yield ResolverResult(url=url, metadata={"source": "hal"})


class LandingPageResolver(RegisteredResolver):
    """Attempt to scrape landing pages when explicit PDFs are unavailable.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> LandingPageResolver().name
        'landing_page'
    """

    name = "landing_page"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.landing_urls)

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        if BeautifulSoup is None:
            yield ResolverResult(url=None, event="skipped", event_reason="no-beautifulsoup")
            return
        for landing in artifact.landing_urls:
            try:
                resp = request_with_retries(
                    session,
                    "get",
                    landing,
                    headers=config.polite_headers,
                    timeout=config.get_timeout(self.name),
                )
            except _requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={
                        "landing": landing,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                continue
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"landing": landing, "error": str(exc)},
                )
                continue
            except _requests.RequestException as exc:  # pragma: no cover - network errors
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"landing": landing, "error": str(exc)},
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected error scraping landing page")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={
                        "landing": landing,
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
                        "landing": landing,
                        "error_detail": f"Landing page returned {resp.status_code}",
                    },
                )
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            for pattern, finder in (
                ("meta", find_pdf_via_meta),
                ("link", find_pdf_via_link),
                ("anchor", find_pdf_via_anchor),
            ):
                candidate = finder(soup, landing)
                if candidate:
                    yield ResolverResult(
                        url=candidate,
                        referer=landing,
                        metadata={"pattern": pattern},
                    )
                    break


class OpenAireResolver(RegisteredResolver):
    """Resolve URLs using the OpenAIRE API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> OpenAireResolver().name
        'openaire'
    """

    name = "openaire"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://api.openaire.eu/search/publications",
                params={"doi": doi},
                headers=config.polite_headers,
                timeout=config.get_timeout(self.name),
            )
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except _requests.RequestException as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"error": str(exc)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error in OpenAIRE resolver")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        if resp.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=resp.status_code,
                metadata={"error_detail": f"OpenAIRE API returned {resp.status_code}"},
            )
            return
        try:
            data = resp.json()
        except ValueError:
            try:
                data = json.loads(resp.text)
            except ValueError as json_err:
                preview = resp.text[:200] if hasattr(resp, "text") else ""
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
                    metadata={"error_detail": str(json_err), "content_preview": preview},
                )
                return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected JSON error in OpenAIRE resolver")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        results: List[str] = []
        _collect_candidate_urls(data, results)
        for url in dedupe(results):
            if url.lower().endswith(".pdf"):
                yield ResolverResult(url=url, metadata={"source": "openaire"})


class OpenAlexResolver(RegisteredResolver):
    """Resolve OpenAlex work metadata into candidate download URLs.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> OpenAlexResolver().name
        'openalex'
    """

    name = "openalex"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.pdf_urls or artifact.open_access_url)

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        candidates = list(dedupe(artifact.pdf_urls))
        if getattr(artifact, "open_access_url", None):
            candidates.append(artifact.open_access_url)

        if not candidates:
            yield ResolverResult(url=None, event="skipped", event_reason="no-openalex-urls")
            return

        for url in dedupe(candidates):
            if not url:
                continue
            yield ResolverResult(url=url, metadata={"source": "openalex_metadata"})


class OsfResolver(ApiResolverBase):
    """Resolve artefacts hosted on the Open Science Framework.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> OsfResolver().name
        'osf'
    """

    name = "osf"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        data, error = self._request_json(
            session,
            "GET",
            "https://api.osf.io/v2/preprints/",
            config=config,
            params={"filter[doi]": doi},
        )
        if error:
            yield error
            return
        urls: List[str] = []
        for item in data.get("data", []) or []:
            if not isinstance(item, dict):
                continue
            links = item.get("links") or {}
            download = links.get("download")
            if isinstance(download, str):
                urls.append(download)
            attributes = item.get("attributes") or {}
            primary = attributes.get("primary_file") or {}
            if isinstance(primary, dict):
                file_links = primary.get("links") or {}
                href = file_links.get("download")
                if isinstance(href, str):
                    urls.append(href)
        for url in dedupe(urls):
            yield ResolverResult(url=url, metadata={"source": "osf"})


class PmcResolver(RegisteredResolver):
    """Resolve PubMed Central articles via identifiers and lookups.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> PmcResolver().name
        'pmc'
    """

    name = "pmc"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.pmcid or artifact.pmid or artifact.doi)

    def _lookup_pmcids(
        self, session: _requests.Session, identifiers: List[str], config: ResolverConfig
    ) -> List[str]:
        if not identifiers:
            return []
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={
                    "ids": ",".join(identifiers),
                    "format": "json",
                    "tool": "docs-to-kg",
                    "email": config.unpaywall_email or "",
                },
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
            )
        except _requests.Timeout as exc:
            LOGGER.debug("PMC ID lookup timed out: %s", exc)
            return []
        except _requests.ConnectionError as exc:
            LOGGER.debug("PMC ID lookup connection error: %s", exc)
            return []
        except _requests.RequestException as exc:
            LOGGER.debug("PMC ID lookup request error: %s", exc)
            return []
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error looking up PMC IDs")
            return []
        if resp.status_code != 200:
            LOGGER.debug("PMC ID lookup HTTP error status: %s", resp.status_code)
            return []
        try:
            data = resp.json()
        except ValueError as json_err:
            LOGGER.debug("PMC ID lookup JSON error: %s", json_err)
            return []
        results: List[str] = []
        for record in data.get("records", []) or []:
            pmcid = record.get("pmcid")
            if pmcid:
                results.append(normalize_pmcid(pmcid))
        return [pmc for pmc in results if pmc]

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        pmcids: List[str] = []
        if artifact.pmcid:
            pmcids.append(normalize_pmcid(artifact.pmcid))
        identifiers: List[str] = []
        doi = normalize_doi(artifact.doi)
        if doi:
            identifiers.append(doi)
        elif artifact.pmid:
            identifiers.append(artifact.pmid)
        if not pmcids:
            pmcids.extend(self._lookup_pmcids(session, identifiers, config))

        if not pmcids:
            yield ResolverResult(url=None, event="skipped", event_reason="no-pmcid")
            return

        for pmcid in dedupe(pmcids):
            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
            fallback_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
            try:
                resp = request_with_retries(
                    session,
                    "get",
                    oa_url,
                    timeout=config.get_timeout(self.name),
                    headers=config.polite_headers,
                )
            except _requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={
                        "pmcid": pmcid,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"pmcid": pmcid, "error": str(exc)},
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except _requests.RequestException as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"pmcid": pmcid, "error": str(exc)},
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected PMC OA lookup error")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={"pmcid": pmcid, "error": str(exc), "error_type": type(exc).__name__},
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            if resp.status_code != 200:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=resp.status_code,
                    metadata={
                        "pmcid": pmcid,
                        "error_detail": f"OA endpoint returned {resp.status_code}",
                    },
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            pdf_links_emitted = False
            try:
                root = ET.fromstring(resp.text)
            except ET.ParseError as exc:
                LOGGER.debug("PMC OA XML parse error for %s: %s", pmcid, exc)
                for href in re.findall(r'href=["\']([^"\']+)["\']', resp.text or ""):
                    candidate = href.strip()
                    if candidate.lower().endswith(".pdf"):
                        pdf_links_emitted = True
                        yield ResolverResult(
                            url=_absolute_url(oa_url, candidate),
                            metadata={"pmcid": pmcid, "source": "oa"},
                        )
            else:
                for link in root.iter():
                    tag = link.tag.rsplit("}", 1)[-1].lower()
                    if tag not in {"link", "a"}:
                        continue
                    href = (
                        link.attrib.get("href")
                        or link.attrib.get("{http://www.w3.org/1999/xlink}href")
                        or ""
                    ).strip()
                    fmt = (link.attrib.get("format") or "").lower()
                    mime = (link.attrib.get("type") or "").lower()
                    if not href:
                        continue
                    if fmt == "pdf" or mime == "application/pdf" or href.lower().endswith(".pdf"):
                        pdf_links_emitted = True
                        yield ResolverResult(
                            url=_absolute_url(oa_url, href),
                            metadata={"pmcid": pmcid, "source": "oa"},
                        )
            if pdf_links_emitted:
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
            else:
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )


class SemanticScholarResolver(RegisteredResolver):
    """Resolve PDFs via the Semantic Scholar Graph API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> SemanticScholarResolver().name
        'semantic_scholar'
    """

    name = "semantic_scholar"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        try:
            data = _fetch_semantic_scholar_data(session, config, doi)
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except _requests.HTTPError as exc:
            status = None
            response_obj = getattr(exc, "response", None)
            if response_obj is not None and hasattr(response_obj, "status_code"):
                status = response_obj.status_code
            if status is None:
                match = re.search(r"(\\d{3})", str(exc))
                if match:
                    status = int(match.group(1))
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=status,
                metadata={"error_detail": str(exc)},
            )
            return
        except _requests.RequestException as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"error": str(exc)},
            )
            return
        except ValueError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(exc)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected Semantic Scholar resolver error")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return

        open_access = (data.get("openAccessPdf") or {}) if isinstance(data, dict) else {}
        url = open_access.get("url") if isinstance(open_access, dict) else None
        if url:
            yield ResolverResult(url=url, metadata={"source": "semantic-scholar"})
        else:
            yield ResolverResult(
                url=None,
                event="skipped",
                event_reason="no-openaccess-pdf",
                metadata={"doi": doi},
            )


class UnpaywallResolver(RegisteredResolver):
    """Resolve PDFs via the Unpaywall API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> UnpaywallResolver().name
        'unpaywall'
    """

    name = "unpaywall"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(config.unpaywall_email and artifact.doi)

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        try:
            data = _fetch_unpaywall_data(session, config, doi)
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except _requests.HTTPError as exc:
            status = None
            response_obj = getattr(exc, "response", None)
            if response_obj is not None and hasattr(response_obj, "status_code"):
                status = response_obj.status_code
            if status is None:
                match = re.search(r"(\\d{3})", str(exc))
                if match:
                    status = int(match.group(1))
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=status,
                metadata={"error_detail": str(exc)},
            )
            return
        except _requests.RequestException as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"error": str(exc)},
            )
            return
        except ValueError as json_err:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(json_err)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error in Unpaywall resolver session path")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        best = (data or {}).get("best_oa_location") or {}
        url = best.get("url_for_pdf")
        if url:
            candidates.append((url, {"source": "best_oa_location"}))

        for loc in (data or {}).get("oa_locations", []) or []:
            if not isinstance(loc, dict):
                continue
            url = loc.get("url_for_pdf")
            if url:
                candidates.append((url, {"source": "oa_location"}))

        unique_urls = dedupe([candidate_url for candidate_url, _ in candidates])
        for unique_url in unique_urls:
            for candidate_url, metadata in candidates:
                if candidate_url == unique_url:
                    yield ResolverResult(url=unique_url, metadata=metadata)
                    break


class WaybackResolver(RegisteredResolver):
    """Fallback resolver that queries the Internet Archive Wayback Machine.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> WaybackResolver().name
        'wayback'
    """

    name = "wayback"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.failed_pdf_urls)

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
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
            except _requests.Timeout as exc:
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
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except _requests.RequestException as exc:
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


class ZenodoResolver(ApiResolverBase):
    """Resolve Zenodo records into downloadable open-access PDF URLs.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> ZenodoResolver().name
        'zenodo'
    """

    name = "zenodo"
    api_display_name = "Zenodo"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.

        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.

        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.

        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.

        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return

        data, error = self._request_json(
            session,
            "GET",
            "https://zenodo.org/api/records/",
            config=config,
            params={"q": f'doi:"{doi}"', "size": 3, "sort": "mostrecent"},
        )
        if error:
            if error.event_reason == "connection-error":
                error.event_reason = "request-error"
            yield error
            return

        hits = data.get("hits", {})
        if not isinstance(hits, dict):
            LOGGER.warning("Zenodo API returned malformed hits payload: %s", type(hits).__name__)
            return
        hits_list = hits.get("hits", [])
        if not isinstance(hits_list, list):
            LOGGER.warning("Zenodo API returned malformed hits list: %s", type(hits_list).__name__)
            return
        for record in hits_list or []:
            if not isinstance(record, dict):
                LOGGER.warning("Skipping malformed Zenodo record: %r", record)
                continue
            files = record.get("files", []) or []
            if not isinstance(files, list):
                LOGGER.warning("Skipping Zenodo record with invalid files payload: %r", files)
                continue
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    LOGGER.warning(
                        "Skipping non-dict Zenodo file entry in record %s", record.get("id")
                    )
                    continue
                file_type = (file_entry.get("type") or "").lower()
                file_key = (file_entry.get("key") or "").lower()
                if file_type == "pdf" or file_key.endswith(".pdf"):
                    links = file_entry.get("links")
                    url = links.get("self") if isinstance(links, dict) else None
                    if url:
                        yield ResolverResult(
                            url=url,
                            metadata={
                                "source": "zenodo",
                                "record_id": record.get("id"),
                                "filename": file_entry.get("key"),
                            },
                        )


# --- Resolver Pipeline ---

# --- Resolver Pipeline ---


def _callable_accepts_argument(func: DownloadFunc, name: str) -> bool:
    """Return ``True`` when ``func`` accepts an argument named ``name``.

    Args:
        func: Download function whose call signature should be inspected.
        name: Argument name whose presence should be detected.

    Returns:
        bool: ``True`` when ``func`` accepts the argument or variable parameters.
    """

    try:
        from inspect import Parameter, signature
    except ImportError:  # pragma: no cover - inspect always available
        return True

    try:
        func_signature = signature(func)
    except (TypeError, ValueError):
        return True

    for parameter in func_signature.parameters.values():
        if parameter.kind in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            return True
        if parameter.name == name:
            return True
    return False


class _RunState:
    """Mutable pipeline execution state shared across resolvers.

    Args:
        dry_run: Indicates whether downloads should be skipped.

    Attributes:
        dry_run: Indicates whether downloads should be skipped.
        seen_urls: Set of URLs already attempted.
        html_paths: Collected HTML fallback paths.
        failed_urls: URLs that failed during resolution.
        attempt_counter: Total number of resolver attempts performed.

    Examples:
        >>> state = _RunState(dry_run=True)
        >>> state.dry_run
        True
    """

    __slots__ = (
        "dry_run",
        "seen_urls",
        "html_paths",
        "failed_urls",
        "attempt_counter",
        "last_reason",
        "last_reason_detail",
    )

    def __init__(self, dry_run: bool) -> None:
        """Initialise run-state bookkeeping for a pipeline execution.

        Args:
            dry_run: Flag indicating whether downloads should be skipped.

        Returns:
            None
        """

        self.dry_run = dry_run
        self.seen_urls: set[str] = set()
        self.html_paths: List[str] = []
        self.failed_urls: List[str] = []
        self.attempt_counter = 0
        self.last_reason: Optional[ReasonCode] = None
        self.last_reason_detail: Optional[str] = None


class ResolverPipeline:
    """Executes resolvers in priority order until a PDF download succeeds.

    The pipeline is safe to reuse across worker threads when
    :attr:`ResolverConfig.max_concurrent_resolvers` is greater than one. All
    mutable shared state is protected by :class:`threading.Lock` instances and
    only read concurrently without mutation. HTTP ``_requests.Session`` objects
    are treated as read-only; callers must avoid mutating shared sessions after
    handing them to the pipeline.

    Attributes:
        config: Resolver configuration containing ordering and rate limits.
        download_func: Callable responsible for downloading resolved URLs.
        logger: Structured attempt logger capturing resolver telemetry.
        metrics: Metrics collector tracking resolver performance.

    Examples:
        >>> pipeline = ResolverPipeline([], ResolverConfig(), lambda *args, **kwargs: None, None)  # doctest: +SKIP
        >>> isinstance(pipeline.metrics, ResolverMetrics)  # doctest: +SKIP
        True
    """

    def __init__(
        self,
        resolvers: Sequence[Resolver],
        config: ResolverConfig,
        download_func: DownloadFunc,
        logger: AttemptSink,
        metrics: Optional[ResolverMetrics] = None,
        initial_seen_urls: Optional[Set[str]] = None,
        global_manifest_index: Optional[Dict[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Create a resolver pipeline with ordering, download, and metric hooks.

        Args:
            resolvers: Resolver instances available for execution.
            config: Pipeline configuration controlling ordering and limits.
            download_func: Callable responsible for downloading resolved URLs.
            logger: Logger that records resolver attempt metadata.
            metrics: Optional metrics collector used for resolver telemetry.

        Returns:
            None
        """

        self._resolver_map = {resolver.name: resolver for resolver in resolvers}
        self.config = config
        self.download_func = download_func
        self.logger = logger
        self.metrics = metrics or ResolverMetrics()
        self._run_id = run_id
        self._last_invocation: Dict[str, float] = defaultdict(lambda: 0.0)
        self._lock = threading.Lock()
        self._global_seen_urls: set[str] = set(initial_seen_urls or ())
        self._global_manifest_index: Dict[str, Dict[str, Any]] = dict(global_manifest_index or {})
        self._global_lock = threading.Lock()
        self._last_host_hit: Dict[str, float] = {}
        self._host_lock = threading.Lock()
        self._host_buckets: Dict[str, TokenBucket] = {}
        self._host_bucket_lock = threading.Lock()
        self._host_breakers: Dict[str, CircuitBreaker] = {}
        self._host_breaker_lock = threading.Lock()
        self._host_semaphores: Dict[str, BoundedSemaphore] = {}
        self._host_semaphore_lock = threading.Lock()
        self._domain_bytes_consumed: Dict[str, int] = defaultdict(int)
        self._domain_bytes_lock = threading.Lock()
        self._resolver_breakers: Dict[str, CircuitBreaker] = {}
        for name, spec in self.config.resolver_circuit_breakers.items():
            threshold = int(spec.get("failure_threshold", 5))
            cooldown = float(spec.get("cooldown_seconds", 60.0))
            self._resolver_breakers[name] = CircuitBreaker(
                failure_threshold=max(threshold, 1),
                cooldown_seconds=max(cooldown, 0.0),
                name=f"resolver:{name}",
            )
        self._download_accepts_context = _callable_accepts_argument(download_func, "context")
        self._download_accepts_head_flag = _callable_accepts_argument(
            download_func, "head_precheck_passed"
        )

    def _emit_attempt(
        self,
        record: AttemptRecord,
        *,
        timestamp: Optional[str] = None,
    ) -> None:
        """Invoke the configured logger using the structured attempt contract.

        Args:
            record: Attempt payload emitted by the pipeline.
            timestamp: Optional ISO timestamp forwarded to sinks that accept it.

        Returns:
            None
        """

        if record.run_id is None and self._run_id is not None:
            record = replace(record, run_id=self._run_id)

        if not hasattr(self.logger, "log_attempt") or not callable(
            getattr(self.logger, "log_attempt")
        ):
            raise AttributeError("ResolverPipeline logger must provide log_attempt().")
        self.logger.log_attempt(record, timestamp=timestamp)

    def _respect_rate_limit(self, resolver_name: str) -> None:
        """Sleep as required to respect per-resolver rate limiting policies.

        The method performs an atomic read-modify-write on
        :attr:`_last_invocation` guarded by :attr:`_lock` to ensure that
        concurrent threads honour resolver spacing requirements.

        Args:
            resolver_name: Name of the resolver to rate limit.

        Returns:
            None
        """

        limit = self.config.resolver_min_interval_s.get(resolver_name)
        if not limit:
            return
        wait = 0.0
        with self._lock:
            last = self._last_invocation[resolver_name]
            now = _time.monotonic()
            delta = now - last
            if delta < limit:
                wait = limit - delta
            self._last_invocation[resolver_name] = now + wait
        if wait > 0:
            _time.sleep(wait)

    def _respect_domain_limit(self, url: str) -> float:
        """Enforce per-domain throttling when configured.

        Args:
            url: Candidate URL whose host may be throttled.

        Returns:
            Total seconds slept enforcing the domain policies.
        """

        if not url:
            return 0.0
        host = urlsplit(url).netloc.lower()
        if not host:
            return 0.0

        waited = 0.0
        bucket = self._ensure_host_bucket(host)
        if bucket is not None:
            wait_seconds = bucket.acquire()
            if wait_seconds > 0:
                _time.sleep(wait_seconds)
                waited += wait_seconds

        interval_map = self.config.domain_min_interval_s
        if not interval_map:
            return waited
        interval = interval_map.get(host)
        if not interval:
            return waited

        now = _time.monotonic()
        wait = 0.0
        with self._host_lock:
            last = self._last_host_hit.get(host)
            if last is None:
                self._last_host_hit[host] = now if now > 0.0 else 1e-9
                return waited
            elapsed = now - last
            if elapsed >= interval:
                self._last_host_hit[host] = now
                return waited
            wait = interval - elapsed

        if wait > 0:
            jitter = random.random() * 0.05
            sleep_for = wait + jitter
            _time.sleep(sleep_for)
            waited += sleep_for
            with self._host_lock:
                self._last_host_hit[host] = _time.monotonic()

        return waited

    def _ensure_host_bucket(self, host: str) -> Optional[TokenBucket]:
        spec = self.config.domain_token_buckets.get(host)
        if not spec:
            return None
        with self._host_bucket_lock:
            bucket = self._host_buckets.get(host)
            if bucket is None:
                bucket = TokenBucket(
                    rate_per_second=float(spec["rate_per_second"]),
                    capacity=float(spec["capacity"]),
                )
                self._host_buckets[host] = bucket
            return bucket

    def _resolve_budget_host_key(self, host: str) -> Optional[str]:
        if not host:
            return None
        host_key = host.lower()
        if host_key in self.config.domain_bytes_budget:
            return host_key
        if host_key.startswith("www."):
            bare = host_key[4:]
            if bare in self.config.domain_bytes_budget:
                return bare
        return None

    def _domain_budget_remaining(self, host_key: str) -> Optional[int]:
        budget = self.config.domain_bytes_budget.get(host_key)
        if budget is None:
            return None
        with self._domain_bytes_lock:
            consumed = self._domain_bytes_consumed.get(host_key, 0)
        return budget - consumed

    def _estimate_outcome_bytes(self, outcome: DownloadOutcome) -> Optional[int]:
        length = outcome.content_length
        bytes_value: Optional[int]
        if isinstance(length, str):
            try:
                bytes_value = int(length)
            except ValueError:
                bytes_value = None
        elif isinstance(length, int):
            bytes_value = length
        elif length is not None:
            try:
                bytes_value = int(length)
            except (TypeError, ValueError):
                bytes_value = None
        else:
            bytes_value = None
        if bytes_value is not None and bytes_value > 0:
            return bytes_value
        path = outcome.path
        if not path:
            return None
        try:
            return max(int(Path(path).stat().st_size), 0)
        except (OSError, ValueError):
            return None

    def _record_domain_bytes(self, host_key: str, outcome: DownloadOutcome) -> None:
        if host_key not in self.config.domain_bytes_budget:
            return
        bytes_used = self._estimate_outcome_bytes(outcome)
        if bytes_used is None or bytes_used <= 0:
            return
        with self._domain_bytes_lock:
            consumed = self._domain_bytes_consumed.get(host_key, 0)
            self._domain_bytes_consumed[host_key] = consumed + bytes_used

    def _acquire_host_slot(self, host: str) -> Optional[Callable[[], None]]:
        limit = self.config.max_concurrent_per_host
        if limit <= 0:
            return None
        host_key = host.lower()
        with self._host_semaphore_lock:
            semaphore = self._host_semaphores.get(host_key)
            if semaphore is None:
                semaphore = BoundedSemaphore(limit)
                self._host_semaphores[host_key] = semaphore
        semaphore.acquire()

        def _release() -> None:
            try:
                semaphore.release()
            except ValueError:  # pragma: no cover - defensive
                pass

        return _release

    def _get_existing_host_breaker(self, host: str) -> Optional[CircuitBreaker]:
        with self._host_breaker_lock:
            return self._host_breakers.get(host)

    def _ensure_host_breaker(self, host: str) -> CircuitBreaker:
        with self._host_breaker_lock:
            breaker = self._host_breakers.get(host)
            if breaker is None:
                spec = self.config.domain_token_buckets.get(host) or {}
                threshold = int(spec.get("breaker_threshold", 5))
                cooldown = float(spec.get("breaker_cooldown", 120.0))
                breaker = CircuitBreaker(
                    failure_threshold=max(threshold, 1),
                    cooldown_seconds=max(cooldown, 0.0),
                    name=f"host:{host}",
                )
                self._host_breakers[host] = breaker
            return breaker

    def _host_breaker_allows(self, host: str) -> Tuple[bool, float]:
        breaker = self._get_existing_host_breaker(host)
        if breaker is None:
            return True, 0.0
        allowed = breaker.allow()
        remaining = breaker.cooldown_remaining() if not allowed else 0.0
        return allowed, remaining

    def _update_breakers(
        self,
        resolver_name: str,
        host: Optional[str],
        outcome: DownloadOutcome,
    ) -> None:
        breaker = self._resolver_breakers.get(resolver_name)
        if breaker:
            success_classes = {
                Classification.PDF,
                Classification.HTML,
                Classification.CACHED,
                Classification.SKIPPED,
            }
            if outcome.classification in success_classes:
                breaker.record_success()
            else:
                breaker.record_failure(retry_after=outcome.retry_after)

        if not host:
            return
        host_key = host.lower()
        status = outcome.http_status or 0
        should_record_failure = status in {429, 500, 502, 503, 504}
        if not should_record_failure and outcome.reason is ReasonCode.REQUEST_EXCEPTION:
            should_record_failure = True

        if should_record_failure:
            breaker = self._ensure_host_breaker(host_key)
            breaker.record_failure(retry_after=outcome.retry_after)
            return

        if outcome.classification in PDF_LIKE or outcome.classification is Classification.HTML:
            breaker = self._get_existing_host_breaker(host_key)
            if breaker:
                breaker.record_success()

    def _jitter_sleep(self) -> None:
        """Introduce a small delay to avoid stampeding downstream services.

        Args:
            self: Pipeline instance executing resolver scheduling logic.

        Returns:
            None
        """

        if self.config.sleep_jitter <= 0:
            return
        _time.sleep(self.config.sleep_jitter + random.random() * 0.1)

    def _should_attempt_head_check(self, resolver_name: str, url: Optional[str]) -> bool:
        """Return ``True`` when a resolver should perform a HEAD preflight request.

        Args:
            resolver_name: Name of the resolver under consideration.

        Returns:
            Boolean indicating whether the resolver should issue a HEAD request.
        """

        if resolver_name in self.config.resolver_head_precheck:
            return self.config.resolver_head_precheck[resolver_name]
        if url:
            parsed = urlparse(url)
            host = (parsed.hostname or parsed.netloc or "").lower()
            if host:
                override = self.config.head_precheck_host_overrides.get(host)
                if override is None and host.startswith("www."):
                    override = self.config.head_precheck_host_overrides.get(host[4:])
                if override is not None:
                    return override
        return self.config.enable_head_precheck

    def _head_precheck_url(
        self,
        session: _requests.Session,
        url: str,
        timeout: float,
        content_policy: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Delegate to the shared network-layer preflight helper.

        Args:
            session: Requests session used to issue the HEAD preflight.
            url: Candidate URL that may host a downloadable document.
            timeout: Maximum duration in seconds to wait for the HEAD response.

        Returns:
            bool: ``True`` when the URL passes preflight checks, ``False`` otherwise.
        """

        return head_precheck(session, url, timeout, content_policy=content_policy)

    def run(
        self,
        session: _requests.Session,
        artifact: "WorkArtifact",
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute resolvers until a PDF is obtained or resolvers are exhausted.

        Args:
            session: Requests session used for resolver HTTP calls.
            artifact: Work artifact describing the document to resolve.
            context: Optional execution context containing flags such as ``dry_run``.

        Returns:
            PipelineResult capturing resolver attempts and successful downloads.
        """

        context_data: Dict[str, Any] = context or {}
        state = _RunState(dry_run=bool(context_data.get("dry_run", False)))

        if self.config.max_concurrent_resolvers == 1:
            return self._run_sequential(session, artifact, context_data, state)

        return self._run_concurrent(session, artifact, context_data, state)

    def _run_sequential(
        self,
        session: _requests.Session,
        artifact: "WorkArtifact",
        context_data: Dict[str, Any],
        state: _RunState,
    ) -> PipelineResult:
        """Execute resolvers in order using the current thread.

        Args:
            session: Shared requests session for resolver HTTP calls.
            artifact: Work artifact describing the document being processed.
            context_data: Execution context dictionary.
            state: Mutable run state tracking attempts and duplicates.

        Returns:
            PipelineResult summarising the sequential run outcome.
        """

        for order_index, resolver_name in enumerate(self.config.resolver_order, start=1):
            resolver = self._prepare_resolver(resolver_name, order_index, artifact, state)
            if resolver is None:
                continue

            results, wall_ms = self._collect_resolver_results(
                resolver_name,
                resolver,
                session,
                artifact,
            )

            for result in results:
                pipeline_result = self._process_result(
                    session,
                    artifact,
                    resolver_name,
                    order_index,
                    result,
                    context_data,
                    state,
                    resolver_wall_time_ms=wall_ms,
                )
                if pipeline_result is not None:
                    return pipeline_result

        return PipelineResult(
            success=False,
            html_paths=list(state.html_paths),
            failed_urls=list(state.failed_urls),
            reason=state.last_reason,
            reason_detail=state.last_reason_detail,
        )

    def _run_concurrent(
        self,
        session: _requests.Session,
        artifact: "WorkArtifact",
        context_data: Dict[str, Any],
        state: _RunState,
    ) -> PipelineResult:
        """Execute resolvers concurrently using a thread pool.

        Args:
            session: Shared requests session for resolver HTTP calls.
            artifact: Work artifact describing the document being processed.
            context_data: Execution context dictionary.
            state: Mutable run state tracking attempts and duplicates.

        Returns:
            PipelineResult summarising the concurrent run outcome.
        """

        max_workers = self.config.max_concurrent_resolvers
        active_futures: Dict[Future[Tuple[List[ResolverResult], float]], Tuple[str, int]] = {}

        def submit_next(
            executor: ThreadPoolExecutor,
            start_index: int,
        ) -> int:
            """Queue additional resolvers until reaching concurrency limits.

            Args:
                executor: Thread pool responsible for executing resolver calls.
                start_index: Index in ``resolver_order`` where submission should resume.

            Returns:
                Updated index pointing to the next resolver candidate that has not been submitted.
            """
            index = start_index
            while len(active_futures) < max_workers and index < len(self.config.resolver_order):
                resolver_name = self.config.resolver_order[index]
                order_index = index + 1
                index += 1
                resolver = self._prepare_resolver(resolver_name, order_index, artifact, state)
                if resolver is None:
                    continue
                future = executor.submit(
                    self._collect_resolver_results,
                    resolver_name,
                    resolver,
                    session,
                    artifact,
                )
                active_futures[future] = (resolver_name, order_index)
            return index

        next_index = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            next_index = submit_next(executor, next_index)

            while active_futures:
                done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    resolver_name, order_index = active_futures.pop(future)
                    try:
                        results, wall_ms = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        results = [
                            ResolverResult(
                                url=None,
                                event="error",
                                event_reason="resolver-exception",
                                metadata={"message": str(exc)},
                            )
                        ]
                        wall_ms = 0.0

                    for result in results:
                        pipeline_result = self._process_result(
                            session,
                            artifact,
                            resolver_name,
                            order_index,
                            result,
                            context_data,
                            state,
                            resolver_wall_time_ms=wall_ms,
                        )
                        if pipeline_result is not None:
                            executor.shutdown(wait=False, cancel_futures=True)
                            return pipeline_result

                next_index = submit_next(executor, next_index)

        return PipelineResult(
            success=False,
            html_paths=list(state.html_paths),
            failed_urls=list(state.failed_urls),
        )

    def _prepare_resolver(
        self,
        resolver_name: str,
        order_index: int,
        artifact: "WorkArtifact",
        state: _RunState,
    ) -> Optional[Resolver]:
        """Return a prepared resolver or log skip events when unavailable.

        Args:
            resolver_name: Name of the resolver to prepare.
            order_index: Execution order index for the resolver.
            artifact: Work artifact being processed.
            state: Mutable run state tracking skips and duplicates.

        Returns:
            Resolver instance when available and enabled, otherwise ``None``.
        """

        resolver = self._resolver_map.get(resolver_name)
        if resolver is None:
            self._emit_attempt(
                AttemptRecord(
                    run_id=self._run_id,
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=None,
                    status=Classification.SKIPPED,
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason=ReasonCode.RESOLVER_MISSING,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=0.0,
                )
            )
            self.metrics.record_skip(resolver_name, "missing")
            return None

        if not self.config.is_enabled(resolver_name):
            self._emit_attempt(
                AttemptRecord(
                    run_id=self._run_id,
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=None,
                    status=Classification.SKIPPED,
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason=ReasonCode.RESOLVER_DISABLED,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=0.0,
                )
            )
            self.metrics.record_skip(resolver_name, "disabled")
            return None

        if not resolver.is_enabled(self.config, artifact):
            self._emit_attempt(
                AttemptRecord(
                    run_id=self._run_id,
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=None,
                    status=Classification.SKIPPED,
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason=ReasonCode.RESOLVER_NOT_APPLICABLE,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=0.0,
                )
            )
            self.metrics.record_skip(resolver_name, "not-applicable")
            return None

        breaker = self._resolver_breakers.get(resolver_name)
        if breaker and not breaker.allow():
            remaining = breaker.cooldown_remaining()
            self._emit_attempt(
                AttemptRecord(
                    run_id=self._run_id,
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=None,
                    status=Classification.SKIPPED,
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason=ReasonCode.RESOLVER_BREAKER_OPEN,
                    reason_detail=f"cooldown-{remaining:.1f}s",
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=0.0,
                    retry_after=remaining if remaining > 0 else None,
                )
            )
            self.metrics.record_skip(resolver_name, "breaker-open")
            return None

        return resolver

    def _collect_resolver_results(
        self,
        resolver_name: str,
        resolver: Resolver,
        session: _requests.Session,
        artifact: "WorkArtifact",
    ) -> Tuple[List[ResolverResult], float]:
        """Collect resolver results while applying rate limits and error handling.

        Args:
            resolver_name: Name of the resolver being executed (for logging and limits).
            resolver: Resolver instance that will generate candidate URLs.
            session: Requests session forwarded to the resolver.
            artifact: Work artifact describing the current document.

        Returns:
            Tuple of resolver results and the resolver wall time (ms).
        """

        results: List[ResolverResult] = []
        self._respect_rate_limit(resolver_name)
        start = _time.monotonic()
        try:
            for result in resolver.iter_urls(session, self.config, artifact):
                results.append(result)
        except Exception as exc:
            self.metrics.record_failure(resolver_name)
            results.append(
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="resolver-exception",
                    metadata={"message": str(exc)},
                )
            )
        wall_ms = (_time.monotonic() - start) * 1000.0
        return results, wall_ms

    def _process_result(
        self,
        session: _requests.Session,
        artifact: "WorkArtifact",
        resolver_name: str,
        order_index: int,
        result: ResolverResult,
        context_data: Dict[str, Any],
        state: _RunState,
        *,
        resolver_wall_time_ms: Optional[float] = None,
    ) -> Optional[PipelineResult]:
        """Process a single resolver result and return a terminal pipeline outcome.

        Args:
            session: Requests session used for follow-up download calls.
            artifact: Work artifact describing the document being processed.
            resolver_name: Name of the resolver that produced the result.
            order_index: 1-based index of the resolver in the execution order.
            result: Resolver result containing either a URL or event metadata.
            context_data: Execution context dictionary.
            state: Mutable run state tracking attempts and duplicates.
            resolver_wall_time_ms: Wall-clock time spent in the resolver.

        Returns:
            PipelineResult when resolution succeeds, otherwise ``None``.
        """

        if result.is_event:
            reason_text = result.event_reason or result.event
            status_value: Any = result.event or Classification.SKIPPED
            self._emit_attempt(
                AttemptRecord(
                    run_id=self._run_id,
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=None,
                    status=status_value,
                    http_status=result.http_status,
                    content_type=None,
                    elapsed_ms=None,
                    reason=reason_text,
                    reason_detail=reason_text,
                    metadata=result.metadata,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=resolver_wall_time_ms,
                )
            )
            if result.event_reason:
                self.metrics.record_skip(resolver_name, result.event_reason)
            return None

        url = result.url
        if not url:
            return None
        url = normalize_url(url)
        if self.config.enable_global_url_dedup:
            with self._global_lock:
                duplicate = url in self._global_seen_urls
                if not duplicate:
                    self._global_seen_urls.add(url)
            if duplicate:
                self._emit_attempt(
                    AttemptRecord(
                        run_id=self._run_id,
                        work_id=artifact.work_id,
                        resolver_name=resolver_name,
                        resolver_order=order_index,
                        url=url,
                        status=Classification.SKIPPED,
                        http_status=None,
                        content_type=None,
                        elapsed_ms=None,
                        reason=ReasonCode.DUPLICATE_URL_GLOBAL,
                        reason_detail="duplicate-url-global",
                        metadata=result.metadata,
                        dry_run=state.dry_run,
                        resolver_wall_time_ms=resolver_wall_time_ms,
                    )
                )
                self.metrics.record_skip(resolver_name, "duplicate-url-global")
                return None
        if url in state.seen_urls:
            self._emit_attempt(
                AttemptRecord(
                    run_id=self._run_id,
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=url,
                    status=Classification.SKIPPED,
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason=ReasonCode.DUPLICATE_URL,
                    reason_detail="duplicate-url",
                    metadata=result.metadata,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=resolver_wall_time_ms,
                )
            )
            self.metrics.record_skip(resolver_name, "duplicate-url")
            return None

        state.seen_urls.add(url)
        parsed_url = urlsplit(url)
        host_for_policy = parsed_url.hostname or parsed_url.netloc
        if context_data.get("list_only"):
            self._emit_attempt(
                AttemptRecord(
                    run_id=self._run_id,
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=url,
                    status=Classification.SKIPPED,
                    http_status=None,
                    content_type=None,
                    elapsed_ms=0.0,
                    reason=ReasonCode.LIST_ONLY,
                    reason_detail="list-only",
                    metadata=result.metadata,
                    dry_run=True,
                    resolver_wall_time_ms=resolver_wall_time_ms,
                )
            )
            self.metrics.record_skip(resolver_name, "list-only")
            return None

        download_context = dict(context_data)
        # Ensure downstream download heuristics are present with config defaults.
        download_context.setdefault("sniff_bytes", self.config.sniff_bytes)
        download_context.setdefault("min_pdf_bytes", self.config.min_pdf_bytes)
        download_context.setdefault("tail_check_bytes", self.config.tail_check_bytes)
        download_context.setdefault("host_accept_overrides", self.config.host_accept_overrides)
        download_context.setdefault("global_manifest_index", self._global_manifest_index)
        download_context.setdefault("domain_content_rules", self.config.domain_content_rules)
        head_precheck_passed = False
        if self._should_attempt_head_check(resolver_name, url):
            head_precheck_passed = self._head_precheck_url(
                session,
                url,
                self.config.get_timeout(resolver_name),
                self.config.get_content_policy(host_for_policy),
            )
            if not head_precheck_passed:
                self._emit_attempt(
                    AttemptRecord(
                        run_id=self._run_id,
                        work_id=artifact.work_id,
                        resolver_name=resolver_name,
                        resolver_order=order_index,
                        url=url,
                        status=Classification.SKIPPED,
                        http_status=None,
                        content_type=None,
                        elapsed_ms=None,
                        reason=ReasonCode.HEAD_PRECHECK_FAILED,
                        metadata=result.metadata,
                        dry_run=state.dry_run,
                        resolver_wall_time_ms=resolver_wall_time_ms,
                    )
                )
                self.metrics.record_skip(resolver_name, "head-precheck-failed")
                return None
            head_precheck_passed = True

        release_host_slot: Optional[Callable[[], None]] = None
        budget_key: Optional[str] = None
        try:
            state.attempt_counter += 1
            host_value = (parsed_url.netloc or "").lower()
            if host_value:
                budget_key = self._resolve_budget_host_key(host_value)
                if budget_key is not None:
                    remaining_budget = self._domain_budget_remaining(budget_key)
                    if remaining_budget is not None and remaining_budget <= 0:
                        self._emit_attempt(
                            AttemptRecord(
                                run_id=self._run_id,
                                work_id=artifact.work_id,
                                resolver_name=resolver_name,
                                resolver_order=order_index,
                                url=url,
                                status=Classification.SKIPPED,
                                http_status=None,
                                content_type=None,
                                elapsed_ms=None,
                                reason=ReasonCode.BUDGET_EXHAUSTED,
                                reason_detail="domain-bytes-budget",
                                metadata=result.metadata,
                                dry_run=state.dry_run,
                                resolver_wall_time_ms=resolver_wall_time_ms,
                            )
                        )
                        self.metrics.record_skip(resolver_name, "domain-bytes-budget")
                        state.last_reason = ReasonCode.BUDGET_EXHAUSTED
                        state.last_reason_detail = "domain-bytes-budget"
                        if url not in state.failed_urls:
                            state.failed_urls.append(url)
                        return None
                allowed, remaining = self._host_breaker_allows(host_value)
                if not allowed:
                    detail = f"cooldown-{remaining:.1f}s"
                    self._emit_attempt(
                        AttemptRecord(
                            run_id=self._run_id,
                            work_id=artifact.work_id,
                            resolver_name=resolver_name,
                            resolver_order=order_index,
                            url=url,
                            status=Classification.SKIPPED,
                            http_status=None,
                            content_type=None,
                            elapsed_ms=None,
                            reason=ReasonCode.DOMAIN_BREAKER_OPEN,
                            reason_detail=detail,
                            metadata=result.metadata,
                            dry_run=state.dry_run,
                            resolver_wall_time_ms=resolver_wall_time_ms,
                            retry_after=remaining if remaining > 0 else None,
                        )
                    )
                    self.metrics.record_skip(resolver_name, "domain-breaker-open")
                    state.last_reason = ReasonCode.DOMAIN_BREAKER_OPEN
                    state.last_reason_detail = detail
                    return None
                release_host_slot = self._acquire_host_slot(host_value)
            domain_wait = self._respect_domain_limit(url)
            kwargs: Dict[str, Any] = {}
            if self._download_accepts_head_flag:
                kwargs["head_precheck_passed"] = head_precheck_passed

            if self._download_accepts_context:
                outcome = self.download_func(
                    session,
                    artifact,
                    url,
                    result.referer,
                    self.config.get_timeout(resolver_name),
                    download_context,
                    **kwargs,
                )
            else:
                outcome = self.download_func(
                    session,
                    artifact,
                    url,
                    result.referer,
                    self.config.get_timeout(resolver_name),
                    **kwargs,
                )

            retry_after_hint = outcome.retry_after
            if retry_after_hint is None and domain_wait > 0:
                retry_after_hint = domain_wait

            self._emit_attempt(
                AttemptRecord(
                    run_id=self._run_id,
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=url,
                    status=outcome.classification,
                    http_status=outcome.http_status,
                    content_type=outcome.content_type,
                    elapsed_ms=outcome.elapsed_ms,
                    reason=outcome.reason,
                    reason_detail=outcome.reason_detail,
                    metadata=result.metadata,
                    sha256=outcome.sha256,
                    content_length=outcome.content_length,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=resolver_wall_time_ms,
                    retry_after=retry_after_hint,
                )
            )
            self.metrics.record_attempt(resolver_name, outcome)
            self._update_breakers(resolver_name, host_value, outcome)
            if budget_key is not None:
                self._record_domain_bytes(budget_key, outcome)

            classification = outcome.classification
            if classification is Classification.HTML and outcome.path:
                state.html_paths.append(outcome.path)

            if classification not in PDF_LIKE and url:
                if url not in state.failed_urls:
                    state.failed_urls.append(url)
                if url not in artifact.failed_pdf_urls:
                    artifact.failed_pdf_urls.append(url)

            if classification in PDF_LIKE:
                return PipelineResult(
                    success=True,
                    resolver_name=resolver_name,
                    url=url,
                    outcome=outcome,
                    html_paths=list(state.html_paths),
                    failed_urls=list(state.failed_urls),
                )

            state.last_reason = outcome.reason
            state.last_reason_detail = outcome.reason_detail
            if state.attempt_counter >= self.config.max_attempts_per_work:
                return PipelineResult(
                    success=False,
                    resolver_name=resolver_name,
                    url=url,
                    outcome=outcome,
                    html_paths=list(state.html_paths),
                    failed_urls=list(state.failed_urls),
                    reason=ReasonCode.MAX_ATTEMPTS_REACHED,
                    reason_detail="max-attempts-reached",
                )

            self._jitter_sleep()
            return None
        finally:
            if release_host_slot:
                release_host_slot()
