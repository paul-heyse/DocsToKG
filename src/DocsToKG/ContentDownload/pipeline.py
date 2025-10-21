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
  including centralized rate limiting, circuit breakers,
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
import re
import threading
import time as _time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
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

import httpx

from DocsToKG.ContentDownload.breakers import BreakerConfig, BreakerOpenError, RequestRole
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
)
from DocsToKG.ContentDownload.networking import head_precheck
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
from DocsToKG.ContentDownload.breakers_loader import load_breaker_config

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


def _validate_max_concurrent_resolvers(value: Any) -> int:
    """Validate ``max_concurrent_resolvers`` sourced from configuration files."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_concurrent_resolvers must be an integer >= 1")
    if value < 1:
        raise ValueError("max_concurrent_resolvers must be >= 1")
    return value


def _validate_enable_global_url_dedup(value: Any) -> bool:
    """Validate ``enable_global_url_dedup`` sourced from configuration files."""

    if not isinstance(value, bool):
        raise ValueError("enable_global_url_dedup must be a boolean value")
    return value


@dataclass
class ResolverConfig:
    """Runtime configuration options applied across resolvers.

    Attributes:
        resolver_order: Ordered list of resolver names to execute.
        resolver_toggles: Mapping toggling individual resolvers on/off.
        max_attempts_per_work: Maximum number of resolver attempts per work item.
        timeout: Default HTTP timeout applied to resolvers.
        retry_after_cap: Ceiling (seconds) applied to ``Retry-After`` hints when retrying HTTP calls.
        polite_headers: HTTP headers to apply for polite crawling.
        unpaywall_email: Contact email registered with Unpaywall.
        core_api_key: API key used for the CORE resolver.
        semantic_scholar_api_key: API key for Semantic Scholar resolver.
        doaj_api_key: API key for DOAJ resolver.
        resolver_timeouts: Resolver-specific timeout overrides.
        enable_head_precheck: Toggle applying HEAD filtering before downloads.
        resolver_head_precheck: Per-resolver overrides for HEAD filtering behaviour.
        host_accept_overrides: Mapping of hostname to Accept header override.
        mailto: Contact email appended to polite headers and user agent string.
        max_concurrent_resolvers: Upper bound on concurrent resolver threads per work.
        enable_global_url_dedup: Enable global URL deduplication across works when True.
        global_url_dedup_cap: Maximum URLs hydrated into the global dedupe cache.
        domain_content_rules: Mapping of hostname to MIME allow-lists.
        breaker_config: Fully resolved circuit breaker configuration shared with networking.
        wayback_config: Wayback-specific configuration options (year_window, max_snapshots, etc.).

    Notes:
        ``enable_head_precheck`` toggles inexpensive HEAD lookups before downloads
        to filter obvious HTML responses. ``resolver_head_precheck`` allows
        per-resolver overrides when specific providers reject HEAD requests.
        ``max_concurrent_resolvers`` bounds the number of resolver threads used
        per work while still respecting configured rate limits.

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
    polite_headers: Dict[str, str] = field(default_factory=dict)
    unpaywall_email: Optional[str] = None
    core_api_key: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    doaj_api_key: Optional[str] = None
    resolver_timeouts: Dict[str, float] = field(default_factory=dict)
    enable_head_precheck: bool = True
    resolver_head_precheck: Dict[str, bool] = field(default_factory=dict)
    head_precheck_host_overrides: Dict[str, bool] = field(default_factory=dict)
    host_accept_overrides: Dict[str, str] = field(default_factory=dict)
    mailto: Optional[str] = None
    max_concurrent_resolvers: int = 1
    enable_global_url_dedup: bool = True
    global_url_dedup_cap: Optional[int] = 100_000
    domain_content_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    breaker_config: BreakerConfig = field(default_factory=BreakerConfig)
    # Wayback-specific configuration
    wayback_config: Dict[str, Any] = field(default_factory=dict)
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
                    f"{self.max_concurrent_resolvers} > 10 may violate provider rate limits. "
                    "Review centralized limiter policies before increasing concurrency."
                ),
                UserWarning,
                stacklevel=2,
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

        if self.domain_content_rules:
            self.domain_content_rules = _normalise_domain_content_rules(self.domain_content_rules)

        if self.max_attempts_per_work < 1:
            raise ValueError(
                f"max_attempts_per_work must be >= 1, got {self.max_attempts_per_work}"
            )

        # Accept overrides (host -> custom Accept header value)
        overrides: Dict[str, str] = {}
        if self.host_accept_overrides:
            # Use deferred import to avoid circular dependency at module load time
            from DocsToKG.ContentDownload.breakers_loader import _normalize_host_key

            for host, header in self.host_accept_overrides.items():
                if not host:
                    continue
                overrides[_normalize_host_key(host)] = str(header)
        self.host_accept_overrides = overrides

        # Legacy resolver_circuit_breakers validation removed - now handled by pybreaker-based BreakerRegistry


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

    validators: Dict[str, Callable[[Any], Any]] = {
        "max_concurrent_resolvers": _validate_max_concurrent_resolvers,
        "enable_global_url_dedup": _validate_enable_global_url_dedup,
    }

    if "domain_min_interval_s" in data and data["domain_min_interval_s"] not in (None, {}):
        raise ValueError(
            "domain_min_interval_s is no longer supported. Configure host policies via centralized rate limiter overrides."
        )
    if "domain_token_buckets" in data and data["domain_token_buckets"] not in (None, {}):
        raise ValueError(
            "domain_token_buckets is no longer supported. Configure host policies via centralized rate limiter overrides."
        )
    if "resolver_min_interval_s" in data and data["resolver_min_interval_s"] not in (None, {}):
        raise ValueError(
            "resolver_min_interval_s is no longer supported. Configure resolver-friendly rate policies via centralized limiter overrides."
        )

    for field_name in (
        "resolver_order",
        "resolver_toggles",
        "max_attempts_per_work",
        "timeout",
        "retry_after_cap",
        "polite_headers",
        "unpaywall_email",
        "core_api_key",
        "semantic_scholar_api_key",
        "doaj_api_key",
        "resolver_timeouts",
        "mailto",
        "resolver_head_precheck",
        "head_precheck_host_overrides",
        "host_accept_overrides",
        # "resolver_circuit_breakers",  # Legacy - now handled by pybreaker-based BreakerRegistry
        "max_concurrent_resolvers",
        "enable_global_url_dedup",
        "domain_content_rules",
        "global_url_dedup_cap",
        "resolver_min_interval_s",
    ):
        if field_name in data and data[field_name] is not None:
            value = data[field_name]
            if field_name == "domain_content_rules":
                value = _normalise_domain_content_rules(value)
            elif field_name in validators:
                value = validators[field_name](value)
            setattr(config, field_name, value)

    if "resolver_rate_limits" in data:
        raise ValueError(
            "resolver_rate_limits is no longer supported. Configure resolver-specific rate policies via centralized limiter overrides."
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
    breaker_fragments: List[Mapping[str, Any]] = []
    breaker_yaml_paths: List[Path] = []

    def _collect_breaker_section(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            breaker_fragments.append(dict(value))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _collect_breaker_section(item)
            return
        if isinstance(value, (str, os.PathLike)):
            breaker_yaml_paths.append(Path(value).expanduser().resolve(strict=False))
            return
        raise ValueError("breaker_config entries must be mappings, sequences, or filesystem paths")

    if DEFAULT_RESOLVER_CREDENTIALS_PATH.is_file():
        config_paths.append(DEFAULT_RESOLVER_CREDENTIALS_PATH)

    if args.resolver_config:
        cli_config_path = Path(args.resolver_config).expanduser().resolve(strict=False)
        config_paths.append(cli_config_path)

    for config_path in config_paths:
        config_data = read_resolver_config(config_path)
        for breaker_key in ("breaker_config", "breaker_policies"):
            if breaker_key in config_data and config_data[breaker_key] not in (None, {}):
                _collect_breaker_section(config_data[breaker_key])
        apply_config_overrides(config, config_data, resolver_names)

    env_breaker_yaml = os.getenv("DOCSTOKG_BREAKERS_YAML")
    if env_breaker_yaml:
        _collect_breaker_section(env_breaker_yaml)

    cli_breaker_path = getattr(args, "breaker_config_path", None)
    if cli_breaker_path:
        _collect_breaker_section(cli_breaker_path)

    cli_host_overrides = list(getattr(args, "breaker_host_overrides", []) or [])
    cli_role_overrides = list(getattr(args, "breaker_role_overrides", []) or [])
    cli_resolver_overrides = list(getattr(args, "breaker_resolver_overrides", []) or [])
    cli_defaults_override = getattr(args, "breaker_defaults_override", None)
    cli_classify_override = getattr(args, "breaker_classify_override", None)
    cli_rolling_override = getattr(args, "breaker_rolling_override", None)

    env_has_breaker_overrides = any(key.startswith("DOCSTOKG_BREAKER") for key in os.environ)

    breaker_overrides_requested = bool(
        breaker_fragments
        or breaker_yaml_paths
        or cli_host_overrides
        or cli_role_overrides
        or cli_resolver_overrides
        or cli_defaults_override
        or cli_classify_override
        or cli_rolling_override
        or env_has_breaker_overrides
    )

    if breaker_overrides_requested:
        try:
            from DocsToKG.ContentDownload.breakers_loader import (
                load_breaker_config,
                merge_breaker_docs,
            )
        except (ImportError, RuntimeError) as exc:
            raise ValueError(
                "Breaker configuration requested but dependencies are unavailable. Install PyYAML or remove breaker overrides."
            ) from exc

        base_doc: Dict[str, Any] = {}
        for fragment in breaker_fragments:
            base_doc = merge_breaker_docs(base_doc, fragment)

        config.breaker_config = load_breaker_config(
            None,
            env=os.environ,
            cli_host_overrides=cli_host_overrides or None,
            cli_role_overrides=cli_role_overrides or None,
            cli_resolver_overrides=cli_resolver_overrides or None,
            cli_defaults_override=cli_defaults_override,
            cli_classify_override=cli_classify_override,
            cli_rolling_override=cli_rolling_override,
            base_doc=base_doc or None,
            extra_yaml_paths=breaker_yaml_paths,
        )
    else:
        config.breaker_config = BreakerConfig()

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

    # Wayback-specific configuration
    wayback_config = {}
    if hasattr(args, "wayback_year_window"):
        wayback_config["year_window"] = args.wayback_year_window
    if hasattr(args, "wayback_max_snapshots"):
        wayback_config["max_snapshots"] = args.wayback_max_snapshots
    if hasattr(args, "wayback_min_pdf_bytes"):
        wayback_config["min_pdf_bytes"] = args.wayback_min_pdf_bytes
    if hasattr(args, "wayback_html_parse") and args.wayback_html_parse is not None:
        wayback_config["html_parse"] = args.wayback_html_parse
    if hasattr(args, "wayback_availability_first") and args.wayback_availability_first is not None:
        wayback_config["availability_first"] = args.wayback_availability_first
    config.wayback_config = wayback_config

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

    if hasattr(args, "global_url_dedup_cap") and args.global_url_dedup_cap is not None:
        config.global_url_dedup_cap = args.global_url_dedup_cap

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
    canonical_url: Optional[str] = None
    original_url: Optional[str] = None
    reason: Optional[ReasonCode | str] = None
    reason_detail: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sha256: Optional[str] = None
    content_length: Optional[int] = None
    dry_run: bool = False
    resolver_wall_time_ms: Optional[float] = None
    retry_after: Optional[float] = None
    rate_limiter_wait_ms: Optional[float] = None
    rate_limiter_backend: Optional[str] = None
    rate_limiter_mode: Optional[str] = None
    rate_limiter_role: Optional[str] = None
    from_cache: Optional[bool] = None
    run_id: Optional[str] = None
    # Breaker state fields
    breaker_host_state: Optional[str] = None
    breaker_resolver_state: Optional[str] = None
    breaker_open_remaining_ms: Optional[int] = None
    breaker_recorded: Optional[str] = None

    def __post_init__(self) -> None:
        normalized_status = normalize_classification(self.status)
        object.__setattr__(self, "status", normalized_status)

        normalized_reason = normalize_reason(self.reason)
        object.__setattr__(self, "reason", normalized_reason)

        canonical = self.canonical_url or self.url
        object.__setattr__(self, "canonical_url", canonical)
        original = self.original_url
        if original is None:
            original = self.url
        object.__setattr__(self, "original_url", original)


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
    canonical_url: Optional[str] = None
    canonical_index: Optional[str] = None
    original_url: Optional[str] = None
    # Circuit breaker state information
    breaker_host_state: Optional[str] = None
    breaker_resolver_state: Optional[str] = None
    breaker_open_remaining_ms: Optional[int] = None
    breaker_recorded: Optional[str] = None

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

        if self.original_url is None:
            self.original_url = self.canonical_url
        if self.canonical_url is None and self.original_url is not None:
            self.canonical_url = self.original_url
        if self.canonical_index is None:
            self.canonical_index = self.canonical_url or self.original_url


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
    canonical_url: Optional[str] = None
    canonical_index: Optional[str] = None
    original_url: Optional[str] = None
    outcome: Optional[DownloadOutcome] = None
    html_paths: List[str] = field(default_factory=list)
    failed_urls: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    reason_detail: Optional[str] = None

    def __post_init__(self) -> None:
        if self.canonical_url is None:
            self.canonical_url = self.url
        if self.original_url is None:
            self.original_url = self.canonical_url
        if self.canonical_index is None:
            self.canonical_index = self.canonical_url
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
    thread-local HTTPX clients via the supplied factory so each worker maintains
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
        self._lock = threading.Lock()
        self._global_seen_urls: set[str] = set(initial_seen_urls or ())
        self._global_manifest_index = global_manifest_index or {}
        self._global_lock = threading.Lock()
        # Legacy breaker initialization removed - now handled by pybreaker-based BreakerRegistry
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
        client: httpx.Client,
        url: str,
        timeout: float,
        content_policy: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Delegate to the shared network-layer preflight helper.

        Args:
            client: HTTPX client used to issue the HEAD preflight.
            url: Candidate URL that may host a downloadable document.
            timeout: Maximum duration in seconds to wait for the HEAD response.

        Returns:
            bool: ``True`` when the URL passes preflight checks, ``False`` otherwise.
        """

        return head_precheck(client, url, timeout, content_policy=content_policy)

    def run(
        self,
        client: httpx.Client,
        artifact: "WorkArtifact",
        context: Optional[Union["DownloadConfig", DownloadContext, Mapping[str, Any]]] = None,
        *,
        client_provider: Optional[Callable[[], httpx.Client]] = None,
    ) -> PipelineResult:
        """Execute resolvers until a PDF is obtained or resolvers are exhausted.

        Args:
            client: HTTPX client used for resolver HTTP calls.
            artifact: Work artifact describing the document to resolve.
            context: Optional :class:`DownloadContext` (or mapping) containing execution flags.
            client_provider: Optional callable returning a client for resolver execution.
                When omitted, the provided ``client`` is reused.

        Returns:
            PipelineResult capturing resolver attempts and successful downloads.
        """

        context_obj = _coerce_download_context(context)
        state = _RunState(dry_run=context_obj.dry_run)
        if client_provider is None:

            def client_provider_fn() -> httpx.Client:
                return client

        else:
            client_provider_fn = client_provider

        if self.config.max_concurrent_resolvers == 1:
            return self._run_sequential(client, client_provider_fn, artifact, context_obj, state)

        return self._run_concurrent(client, client_provider_fn, artifact, context_obj, state)

    def _run_sequential(
        self,
        client: httpx.Client,
        client_provider: Callable[[], httpx.Client],
        artifact: "WorkArtifact",
        context: DownloadContext,
        state: _RunState,
    ) -> PipelineResult:
        """Execute resolvers in order using the current thread.

        Args:
            client: Shared HTTPX client for resolver HTTP calls.
            client_provider: Callable returning the client for the active thread.
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
                client_provider,
                artifact,
            )

            for result in results:
                pipeline_result = self._process_result(
                    client,
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
        client: httpx.Client,
        client_provider: Callable[[], httpx.Client],
        artifact: "WorkArtifact",
        context: DownloadContext,
        state: _RunState,
    ) -> PipelineResult:
        """Execute resolvers concurrently using a thread pool.

        Args:
            client: Shared HTTPX client for resolver HTTP calls.
            client_provider: Callable returning the client for the active thread.
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
                    client_provider,
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
                            client,
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

        # Legacy resolver breaker check removed - now handled by pybreaker-based BreakerRegistry

        return resolver

    def _collect_resolver_results(
        self,
        resolver_name: str,
        resolver: Resolver,
        client_provider: Callable[[], httpx.Client],
        artifact: "WorkArtifact",
    ) -> Tuple[List[ResolverResult], float]:
        """Collect resolver results while applying rate limits and error handling.

        Args:
            resolver_name: Name of the resolver being executed (for logging and limits).
            resolver: Resolver instance that will generate candidate URLs.
            client_provider: Callable returning the HTTPX client for the current thread.
            artifact: Work artifact describing the current document.

        Returns:
            Tuple of resolver results and the resolver wall time (ms).
        """

        client = client_provider()
        results: List[ResolverResult] = []
        start = _time.monotonic()
        try:
            for result in resolver.iter_urls(client, self.config, artifact):
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
        client: httpx.Client,
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
            client: HTTPX client used for follow-up download calls.
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

        url = result.canonical_url or result.url
        if not url:
            return None
        original_url = result.original_url or url
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
                        canonical_url=url,
                        original_url=original_url,
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
                    canonical_url=url,
                    original_url=original_url,
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
                    canonical_url=url,
                    original_url=original_url,
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
                client,
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
                        canonical_url=url,
                        original_url=original_url,
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

        host_value = (parsed_url.netloc or "").lower()
        # Legacy breaker preflight removed - now handled by pybreaker-based BreakerRegistry in networking layer
        breaker_state = None
        if breaker_state is not None:
            reason_token, detail_token, retry_after_hint = breaker_state
            self._emit_attempt(
                AttemptRecord(
                    run_id=self._run_id,
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=url,
                    canonical_url=url,
                    original_url=original_url,
                    status=Classification.SKIPPED,
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason=reason_token,
                    reason_detail=detail_token,
                    metadata=result.metadata,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=resolver_wall_time_ms,
                    retry_after=retry_after_hint,
                )
            )
            skip_reason = reason_token.replace("_", "-")
            self._record_skip(resolver_name, skip_reason, detail_token)
            return None

        state.attempt_counter += 1
        kwargs: Dict[str, Any] = {}
        if self._download_accepts_head_flag:
            kwargs["head_precheck_passed"] = head_precheck_passed
        if result.origin_host:
            kwargs.setdefault("origin_host", result.origin_host)
        if original_url is not None:
            kwargs.setdefault("original_url", original_url)

        if self._download_accepts_context:
            outcome = self.download_func(
                client,
                artifact,
                url,
                result.referer,
                self.config.get_timeout(resolver_name),
                download_context,
                telemetry=self.logger,
                run_id=self._run_id,
                **kwargs,
            )
        else:
            outcome = self.download_func(
                client,
                artifact,
                url,
                result.referer,
                self.config.get_timeout(resolver_name),
                telemetry=self.logger,
                run_id=self._run_id,
                **kwargs,
            )

        retry_after_hint = outcome.retry_after

        metadata_payload: Dict[str, Any] = {}
        resolver_metadata = getattr(result, "metadata", None)
        if isinstance(resolver_metadata, dict):
            metadata_payload.update(resolver_metadata)
        outcome_metadata = getattr(outcome, "metadata", None)
        if isinstance(outcome_metadata, dict):
            metadata_payload.update(outcome_metadata)

        rate_limiter_info = metadata_payload.get("rate_limiter")
        if not isinstance(rate_limiter_info, Mapping):
            rate_limiter_info = {}
        network_info = metadata_payload.get("network")
        if not isinstance(network_info, Mapping):
            network_info = {}

        wait_ms_value: Optional[float]
        try:
            wait_raw = rate_limiter_info.get("wait_ms")
            wait_ms_value = float(wait_raw) if wait_raw is not None else None
        except (TypeError, ValueError):
            wait_ms_value = None

        rate_limiter_role = rate_limiter_info.get("role")
        if isinstance(rate_limiter_role, str):
            rate_limiter_role = rate_limiter_role.lower()
        else:
            rate_limiter_role = None

        attempt_record = AttemptRecord(
            run_id=self._run_id,
            work_id=artifact.work_id,
            resolver_name=resolver_name,
            resolver_order=order_index,
            url=url,
            canonical_url=outcome.canonical_url or url,
            original_url=original_url,
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
            rate_limiter_wait_ms=wait_ms_value,
            rate_limiter_backend=rate_limiter_info.get("backend"),
            rate_limiter_mode=rate_limiter_info.get("mode"),
            rate_limiter_role=rate_limiter_role,
            from_cache=network_info.get("from_cache"),
            # Circuit breaker state from DownloadOutcome
            breaker_host_state=outcome.breaker_host_state,
            breaker_resolver_state=outcome.breaker_resolver_state,
            breaker_open_remaining_ms=outcome.breaker_open_remaining_ms,
            breaker_recorded=outcome.breaker_recorded,
        )

        self._emit_attempt(attempt_record)
        self.metrics.record_attempt(resolver_name, outcome)
        # Legacy breaker update removed - now handled by pybreaker-based BreakerRegistry

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
                    canonical_url=outcome.canonical_url or url,
                    canonical_index=outcome.canonical_index or url,
                    original_url=outcome.original_url or original_url,
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
                    canonical_url=outcome.canonical_url or url,
                    canonical_index=outcome.canonical_index or url,
                    original_url=outcome.original_url or original_url,
                    outcome=outcome,
                    html_paths=list(state.html_paths),
                    failed_urls=list(state.failed_urls),
                    reason=ReasonCode.MAX_ATTEMPTS_REACHED,
                    reason_detail="max-attempts-reached",
                )

            return None
