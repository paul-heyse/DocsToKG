# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.pipeline",
#   "purpose": "Resolver definitions and pipeline orchestration",
#   "sections": [
#     {
#       "id": "coerce-download-context",
#       "name": "_coerce_download_context",
#       "anchor": "function-coerce-download-context",
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
#       "id": "resolvermetrics",
#       "name": "ResolverMetrics",
#       "anchor": "class-resolvermetrics",
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

"""Resolver orchestration, configuration, and metrics for content downloads.

Responsibilities
----------------
- Own the resolver registry (built-ins plus plugin hooks) and expose helpers
  such as :func:`default_resolvers`, :func:`load_resolver_config`, and
  :func:`apply_config_overrides` that normalise command-line overrides.
- Define the data structures that describe pipeline activity:
  :class:`ResolverConfig`, :class:`AttemptRecord`, :class:`DownloadOutcome`,
  :class:`PipelineResult`, and :class:`ResolverMetrics`.
- Coordinate threaded resolver execution via :class:`ResolverPipeline`,
  including concurrency controls, per-domain token buckets, circuit breakers,
  and retry logic wired into :mod:`DocsToKG.ContentDownload.networking`.
- Track manifest/bookkeeping state (URL dedupe, cache verification, latency
  counters) so downstream modules can persist consistent telemetry snapshots.

Key Components
--------------
- ``ResolverConfig`` – typed view of pipeline and HTTP toggle settings.
- ``ResolverPipeline`` – orchestrates resolver attempts, concurrency, and
  result aggregation.
- ``AttemptRecord`` / ``DownloadOutcome`` / ``PipelineResult`` – structured
  payloads consumed by telemetry sinks.
- ``load_resolver_config`` / ``read_resolver_config`` – parse YAML/JSON config
  files and merge them with defaults and CLI toggles.

Design Notes
------------
- Resolver implementations live in :mod:`DocsToKG.ContentDownload.resolvers`;
  this module keeps orchestration logic separate from individual provider code.
- Concurrency primitives favour cooperative cancellation and reuse fixtures
  that the unit tests in ``tests/content_download`` exercise directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import threading
import time as _time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
from urllib.parse import urlparse, urlsplit

import requests as _requests

from DocsToKG.ContentDownload.core import (
    DEFAULT_MIN_PDF_BYTES,
    DEFAULT_SNIFF_BYTES,
    DEFAULT_TAIL_CHECK_BYTES,
    PDF_LIKE,
    Classification,
    DownloadContext,
    ReasonCode,
    normalize_classification,
    normalize_reason,
    normalize_url,
)
from DocsToKG.ContentDownload.networking import (
    CircuitBreaker,
    TokenBucket,
    head_precheck,
)
from DocsToKG.ContentDownload.resolvers import (
    DEFAULT_RESOLVER_ORDER,
    DEFAULT_RESOLVER_TOGGLES,
    ApiResolverBase,
    ArxivResolver,
    CoreResolver,
    CrossrefResolver,
    DoajResolver,
    EuropePmcResolver,
    FigshareResolver,
    HalResolver,
    LandingPageResolver,
    OpenAireResolver,
    OpenAlexResolver,
    OsfResolver,
    PmcResolver,
    RegisteredResolver,
    Resolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverRegistry,
    ResolverResult,
    SemanticScholarResolver,
    UnpaywallResolver,
    WaybackResolver,
    ZenodoResolver,
    default_resolvers,
)
from DocsToKG.ContentDownload.telemetry import AttemptSink

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.download import DownloadConfig

# --- Globals ---

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESOLVER_CREDENTIALS_PATH = PROJECT_ROOT / "config" / "resolver_credentials.yaml"


def _coerce_download_context(
    context: Optional[Union["DownloadConfig", DownloadContext, Mapping[str, Any]]],
) -> DownloadContext:
    """Normalise context-like inputs into a :class:`DownloadContext` instance."""

    if isinstance(context, DownloadContext):
        return context

    if context is None:
        return DownloadContext.from_mapping({})

    if hasattr(context, "to_context"):
        try:
            return context.to_context({})  # type: ignore[call-arg]
        except TypeError:
            return context.to_context()  # type: ignore[call-arg]

    from DocsToKG.ContentDownload.download import DownloadConfig

    return DownloadConfig.from_options(context).to_context({})


__all__ = [
    "ApiResolverBase",
    "ArxivResolver",
    "AttemptRecord",
    "AttemptSink",
    "CoreResolver",
    "CrossrefResolver",
    "DownloadFunc",
    "DownloadOutcome",
    "DoajResolver",
    "EuropePmcResolver",
    "FigshareResolver",
    "HalResolver",
    "PipelineResult",
    "LandingPageResolver",
    "Resolver",
    "ResolverConfig",
    "ResolverMetrics",
    "ResolverPipeline",
    "ResolverRegistry",
    "ResolverResult",
    "ResolverEvent",
    "ResolverEventReason",
    "RegisteredResolver",
    "OpenAireResolver",
    "OpenAlexResolver",
    "OsfResolver",
    "PmcResolver",
    "SemanticScholarResolver",
    "UnpaywallResolver",
    "WaybackResolver",
    "ZenodoResolver",
    "apply_config_overrides",
    "default_resolvers",
    "load_resolver_config",
    "read_resolver_config",
]

# --- Resolver Configuration ---


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
        retry_after_cap: Ceiling (seconds) applied to ``Retry-After`` hints when retrying HTTP calls.
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
        domain_content_rules: Mapping of hostname to MIME allow-lists.
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
        default_factory=lambda: dict(DEFAULT_RESOLVER_TOGGLES)
    )
    max_attempts_per_work: int = 25
    timeout: float = 30.0
    retry_after_cap: float = 120.0
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
        "retry_after_cap",
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
        "domain_content_rules",
    ):
        if field_name in data and data[field_name] is not None:
            if field_name == "domain_content_rules":
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
    config_paths: List[Path] = []

    if DEFAULT_RESOLVER_CREDENTIALS_PATH.is_file():
        config_paths.append(DEFAULT_RESOLVER_CREDENTIALS_PATH)

    if args.resolver_config:
        cli_config_path = Path(args.resolver_config).expanduser().resolve(strict=False)
        config_paths.append(cli_config_path)

    for config_path in config_paths:
        config_data = read_resolver_config(config_path)
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
    if getattr(args, "retry_after_cap", None):
        config.retry_after_cap = float(args.retry_after_cap)
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
    if getattr(args, "domain_token_bucket", None):
        bucket_map: Dict[str, Dict[str, float]] = dict(config.domain_token_buckets)
        for domain, spec in args.domain_token_bucket:
            bucket_map[domain] = dict(spec)
        config.domain_token_buckets = bucket_map

    if config.retry_after_cap <= 0:
        raise ValueError("retry_after_cap must be positive")

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
        normalized_status = normalize_classification(self.status)
        object.__setattr__(self, "status", normalized_status)

        normalized_reason = normalize_reason(self.reason)
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
    metadata: Dict[str, Any] = field(default_factory=dict)
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
        normalized_status = normalize_classification(self.classification)
        if isinstance(normalized_status, Classification):
            self.classification = normalized_status
        else:
            self.classification = Classification.from_wire(normalized_status)

        self.reason = normalize_reason(self.reason)
        metadata_value = self.metadata
        if metadata_value is None:
            self.metadata = {}
        elif not isinstance(metadata_value, dict):
            self.metadata = dict(metadata_value)


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
    xml: Counter = field(default_factory=Counter)
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
            if outcome.classification is Classification.XML:
                self.xml[resolver_name] += 1
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
                "xml": dict(self.xml),
                "skips": dict(self.skips),
                "failures": dict(self.failures),
                "latency_ms": latency_summary,
                "status_counts": status_summary,
                "error_reasons": reason_summary,
                "classification_totals": dict(classification_totals),
                "reason_totals": dict(reason_totals),
            }


# --- Shared Helpers ---


DownloadFunc = Callable[..., DownloadOutcome]


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
    only read concurrently without mutation. Resolver execution retrieves
    thread-local HTTP sessions via the supplied factory so each worker maintains
    its own connection pool without cross-thread sharing.

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
        self._global_manifest_index = global_manifest_index or {}
        self._global_lock = threading.Lock()
        self._last_host_hit: Dict[str, float] = {}
        self._host_lock = threading.Lock()
        self._host_buckets: Dict[str, TokenBucket] = {}
        self._host_bucket_lock = threading.Lock()
        self._host_breakers: Dict[str, CircuitBreaker] = {}
        self._host_breaker_lock = threading.Lock()
        self._host_semaphores: Dict[str, BoundedSemaphore] = {}
        self._host_semaphore_lock = threading.Lock()
        self._resolver_breakers: Dict[str, CircuitBreaker] = {}
        circuit_breakers = getattr(self.config, "resolver_circuit_breakers", {}) or {}
        for name, spec in circuit_breakers.items():
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

        if isinstance(record.reason, ReasonCode):
            reason_text = record.reason.value.replace("_", "-")
            record = replace(record, reason=reason_text)
        if isinstance(record.reason_detail, ReasonCode):
            detail_text = record.reason_detail.value.replace("_", "-")
            record = replace(record, reason_detail=detail_text)

        if not hasattr(self.logger, "log_attempt") or not callable(
            getattr(self.logger, "log_attempt")
        ):
            raise AttributeError("ResolverPipeline logger must provide log_attempt().")
        self.logger.log_attempt(record, timestamp=timestamp)

    def _record_skip(
        self,
        resolver_name: str,
        reason: Union[str, ReasonCode],
        detail: Optional[Union[str, ReasonCode]] = None,
    ) -> None:
        """Normalise and record skip telemetry for resolver events."""

        reason_token = normalize_reason(reason)
        if isinstance(reason_token, ReasonCode):
            reason_text = reason_token.value.replace("_", "-")
        else:
            reason_text = str(reason_token if reason_token is not None else reason)
        self.metrics.record_skip(resolver_name, reason_text)
        if detail is None:
            return
        detail_token = normalize_reason(detail)
        if isinstance(detail_token, ReasonCode):
            detail_text = detail_token.value.replace("_", "-")
        elif detail_token is None:
            return
        else:
            detail_text = str(detail_token)
        if detail_text and detail_text != reason_text:
            self.metrics.record_skip(resolver_name, detail_text)

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

        base_jitter = self.config.sleep_jitter
        if base_jitter <= 0:
            return
        concurrency = max(self.config.max_concurrent_resolvers, 1)
        effective = base_jitter if concurrency == 1 else base_jitter / concurrency
        _time.sleep(effective + random.random() * 0.1)

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
        context: Optional[Union["DownloadConfig", DownloadContext, Mapping[str, Any]]] = None,
        *,
        session_factory: Optional[Callable[[], _requests.Session]] = None,
    ) -> PipelineResult:
        """Execute resolvers until a PDF is obtained or resolvers are exhausted.

        Args:
            session: Requests session used for resolver HTTP calls.
            artifact: Work artifact describing the document to resolve.
            context: Optional :class:`DownloadContext` (or mapping) containing execution flags.
            session_factory: Optional callable returning a thread-local session for
                resolver execution. When omitted, the provided ``session`` is reused.

        Returns:
            PipelineResult capturing resolver attempts and successful downloads.
        """

        context_obj = _coerce_download_context(context)
        state = _RunState(dry_run=context_obj.dry_run)
        if session_factory is None:

            def session_provider() -> _requests.Session:
                """Return the shared HTTP session when no factory override is provided."""
                return session

        else:
            session_provider = session_factory

        if self.config.max_concurrent_resolvers == 1:
            return self._run_sequential(session, session_provider, artifact, context_obj, state)

        return self._run_concurrent(session, session_provider, artifact, context_obj, state)

    def _run_sequential(
        self,
        session: _requests.Session,
        session_provider: Callable[[], _requests.Session],
        artifact: "WorkArtifact",
        context: DownloadContext,
        state: _RunState,
    ) -> PipelineResult:
        """Execute resolvers in order using the current thread.

        Args:
            session: Shared requests session for resolver HTTP calls.
            session_provider: Callable returning the session for the active thread.
            artifact: Work artifact describing the document being processed.
            context: Execution context object.
            state: Mutable run state tracking attempts and duplicates.

        Returns:
            PipelineResult summarising the sequential run outcome.
        """

        override = list(context.resolver_order)
        if override:
            ordered_names = [name for name in override if name in self._resolver_map]
            ordered_names.extend(
                name for name in self.config.resolver_order if name not in ordered_names
            )
        else:
            ordered_names = list(self.config.resolver_order)

        for order_index, resolver_name in enumerate(ordered_names, start=1):
            resolver = self._prepare_resolver(resolver_name, order_index, artifact, state)
            if resolver is None:
                continue

            results, wall_ms = self._collect_resolver_results(
                resolver_name,
                resolver,
                session_provider,
                artifact,
            )

            for result in results:
                pipeline_result = self._process_result(
                    session,
                    artifact,
                    resolver_name,
                    order_index,
                    result,
                    context,
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
        session_provider: Callable[[], _requests.Session],
        artifact: "WorkArtifact",
        context: DownloadContext,
        state: _RunState,
    ) -> PipelineResult:
        """Execute resolvers concurrently using a thread pool.

        Args:
            session: Shared requests session for resolver HTTP calls.
            session_provider: Callable returning the session for the active thread.
            artifact: Work artifact describing the document being processed.
            context: Execution context object.
            state: Mutable run state tracking attempts and duplicates.

        Returns:
            PipelineResult summarising the concurrent run outcome.
        """

        override = list(context.resolver_order)
        if override:
            resolver_order = [name for name in override if name in self._resolver_map]
            resolver_order.extend(
                name for name in self.config.resolver_order if name not in resolver_order
            )
        else:
            resolver_order = list(self.config.resolver_order)

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
            while len(active_futures) < max_workers and index < len(resolver_order):
                resolver_name = resolver_order[index]
                order_index = index + 1
                index += 1
                resolver = self._prepare_resolver(resolver_name, order_index, artifact, state)
                if resolver is None:
                    continue
                future = executor.submit(
                    self._collect_resolver_results,
                    resolver_name,
                    resolver,
                    session_provider,
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
                                event=ResolverEvent.ERROR,
                                event_reason=ResolverEventReason.RESOLVER_EXCEPTION,
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
                            context,
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
            self._record_skip(resolver_name, "missing")
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
            self._record_skip(resolver_name, "disabled")
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
            self._record_skip(resolver_name, "not-applicable")
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
            self._record_skip(resolver_name, "breaker-open")
            return None

        return resolver

    def _collect_resolver_results(
        self,
        resolver_name: str,
        resolver: Resolver,
        session_provider: Callable[[], _requests.Session],
        artifact: "WorkArtifact",
    ) -> Tuple[List[ResolverResult], float]:
        """Collect resolver results while applying rate limits and error handling.

        Args:
            resolver_name: Name of the resolver being executed (for logging and limits).
            resolver: Resolver instance that will generate candidate URLs.
            session_provider: Callable returning the requests session for the current thread.
            artifact: Work artifact describing the current document.

        Returns:
            Tuple of resolver results and the resolver wall time (ms).
        """

        session = session_provider()
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
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.RESOLVER_EXCEPTION,
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
        context: DownloadContext,
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
            context: Execution context object.
            state: Mutable run state tracking attempts and duplicates.
            resolver_wall_time_ms: Wall-clock time spent in the resolver.

        Returns:
            PipelineResult when resolution succeeds, otherwise ``None``.
        """

        if result.is_event:
            if result.event_reason is not None:
                reason_text: Optional[str] = result.event_reason.value
            elif result.event is not None:
                reason_text = result.event.value
            else:
                reason_text = None
            status_value: Any = (
                result.event.value if result.event is not None else Classification.SKIPPED
            )
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
                self._record_skip(resolver_name, result.event_reason.value)
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
                self._record_skip(resolver_name, "duplicate-url-global")
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
            self._record_skip(resolver_name, "duplicate-url")
            return None

        state.seen_urls.add(url)
        parsed_url = urlsplit(url)
        host_for_policy = parsed_url.hostname or parsed_url.netloc
        if context.list_only:
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
            self._record_skip(resolver_name, "list-only")
            return None

        download_context = context.clone_for_download()
        # Ensure downstream download heuristics are present with config defaults.
        if not context.is_explicit("sniff_bytes"):
            download_context.sniff_bytes = self.config.sniff_bytes
        if not context.is_explicit("min_pdf_bytes"):
            download_context.min_pdf_bytes = self.config.min_pdf_bytes
        if not context.is_explicit("tail_check_bytes"):
            download_context.tail_check_bytes = self.config.tail_check_bytes
        if not context.is_explicit("host_accept_overrides"):
            download_context.host_accept_overrides = self.config.host_accept_overrides
        if not context.is_explicit("global_manifest_index"):
            download_context.global_manifest_index = self._global_manifest_index
        if not context.is_explicit("domain_content_rules"):
            download_context.domain_content_rules = self.config.domain_content_rules
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
                self._record_skip(resolver_name, "head-precheck-failed")
                return None
            head_precheck_passed = True

        release_host_slot: Optional[Callable[[], None]] = None
        try:
            state.attempt_counter += 1
            host_value = (parsed_url.netloc or "").lower()
            if host_value:
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
                    self._record_skip(resolver_name, "domain-breaker-open", detail)
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

            metadata_payload: Dict[str, Any] = {}
            resolver_metadata = getattr(result, "metadata", None)
            if isinstance(resolver_metadata, dict):
                metadata_payload.update(resolver_metadata)
            outcome_metadata = getattr(outcome, "metadata", None)
            if isinstance(outcome_metadata, dict):
                metadata_payload.update(outcome_metadata)
            extra_context = getattr(download_context, "extra", None)
            if isinstance(extra_context, dict) and extra_context.get("resume_disabled"):
                metadata_payload["resume_disabled"] = True
            # Strip any legacy resume hints to avoid re-enabling deprecated range flows.
            for key in list(metadata_payload):
                if key.startswith("resume_") and key != "resume_disabled":
                    metadata_payload.pop(key, None)

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
                    metadata=metadata_payload,
                    sha256=outcome.sha256,
                    content_length=outcome.content_length,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=resolver_wall_time_ms,
                    retry_after=retry_after_hint,
                )
            )
            self.metrics.record_attempt(resolver_name, outcome)
            self._update_breakers(resolver_name, host_value, outcome)

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
