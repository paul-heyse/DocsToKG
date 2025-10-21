# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.args",
#   "purpose": "CLI option parsing and configuration bootstrap for content downloads",
#   "sections": [
#     {"id": "resolved-config", "name": "ResolvedConfig", "anchor": "class-resolved-config", "kind": "class"},
#     {"id": "bootstrap-run-environment", "name": "bootstrap_run_environment", "anchor": "function-bootstrap-run-environment", "kind": "function"},
#     {"id": "resolve-config", "name": "resolve_config", "anchor": "function-resolve-config", "kind": "function"},
#     {"id": "build-query", "name": "build_query", "anchor": "function-build-query", "kind": "function"},
#     {"id": "resolve-topic-id-if-needed", "name": "resolve_topic_id_if_needed", "anchor": "function-resolve-topic-id-if-needed", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""CLI argument resolution and run bootstrap for DocsToKG content downloads.

Responsibilities
----------------
- Define the public parser surface that maps CLI flags onto a typed
  :class:`ResolvedConfig` snapshot consumed by the runner.
- Assemble OpenAlex ``Works`` queries (or topic lookups) from user input while
  deferring all network traffic until execution time.
- Provision output directories, manifest destinations, and resolver instances
  via helpers such as :func:`bootstrap_run_environment` and
  :func:`resolve_config`.
- Hydrate manifest indexes and global URL dedupe sets (via
  :class:`ManifestUrlIndex`) so runs can reuse cached artifacts before
  contacting resolvers.
- Detect when ``--sleep`` is supplied explicitly so concurrent runs avoid
  inheriting sequential throttle defaults unless requested.
- Expose compatibility shims (``build_query()``, ``resolve_topic_id_if_needed``)
  that tests and automation can exercise without pulling in the full CLI stack.

Design Notes
------------
- Import-time side effects are intentionally avoided so the module can be
  reused in unit tests and other tooling without hitting the network.
- The dataclasses defined here are immutable to keep configuration hand-offs
  explicit; stateful operations live in ``DocsToKG.ContentDownload.runner`` and
  downstream modules.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from pyalex import Topics, Works
from pyrate_limiter import Duration, Rate

from DocsToKG.ContentDownload.cache_loader import load_cache_config
from DocsToKG.ContentDownload.cache_policy import CacheRouter
from DocsToKG.ContentDownload.core import (
    DEFAULT_MIN_PDF_BYTES,
    DEFAULT_SNIFF_BYTES,
    DEFAULT_TAIL_CHECK_BYTES,
    Classification,
)
from DocsToKG.ContentDownload.download import RobotsCache, ensure_dir
from DocsToKG.ContentDownload.pipeline import load_resolver_config
from DocsToKG.ContentDownload.pyalex_shim import apply_mailto
from DocsToKG.ContentDownload.ratelimit import (
    DEFAULT_ROLE,
    ROLE_ORDER,
    BackendConfig,
    RolePolicy,
    clone_policies,
    clone_role_policy,
    configure_rate_limits,
    get_rate_limiter_manager,
    validate_policies,
)
from DocsToKG.ContentDownload.ratelimits_loader import load_rate_config
from DocsToKG.ContentDownload.resolvers import DEFAULT_RESOLVER_ORDER, default_resolvers
from DocsToKG.ContentDownload.telemetry import ManifestUrlIndex
from DocsToKG.ContentDownload.urls import configure_url_policy, parse_param_allowlist_spec

__all__ = [
    "ResolvedConfig",
    "build_parser",
    "parse_args",
    "resolve_config",
    "bootstrap_run_environment",
    "build_query",
    "resolve_topic_id_if_needed",
]

LOGGER = logging.getLogger("DocsToKG.ContentDownload")

DEFAULT_SLEEP_SECONDS = 0.05


@dataclass(frozen=True)
class ResolvedConfig:
    """Immutable configuration derived from CLI arguments.

    The dataclass is frozen to prevent callers from mutating configuration at
    runtime. Any operational side effects (filesystem initialisation, telemetry
    bootstrapping, etc.) must be performed explicitly via helper functions such
    as :func:`bootstrap_run_environment` rather than during configuration
    resolution.
    """

    args: argparse.Namespace
    run_id: str
    query: Works
    pdf_dir: Path
    html_dir: Path
    xml_dir: Path
    manifest_path: Path
    csv_path: Optional[Path]
    sqlite_path: Path
    resolver_instances: List[Any]
    resolver_config: Any
    previous_url_index: ManifestUrlIndex
    persistent_seen_urls: Set[str]
    robots_checker: Optional[RobotsCache]
    concurrency_product: int
    extract_html_text: bool
    verify_cache_digest: bool
    openalex_retry_attempts: int
    openalex_retry_backoff: float
    openalex_retry_max_delay: float
    retry_after_cap: float
    rate_policies: Mapping[str, RolePolicy]
    rate_backend: BackendConfig
    rate_config: Optional[Any] = None  # RateConfig from ratelimits_loader
    cache_config: Optional[Any] = None  # CacheConfig from cache_loader
    cache_disabled: bool = False  # If True, bypass all caching


def bootstrap_run_environment(resolved: ResolvedConfig) -> None:
    """Initialise directories required for a resolved download run."""

    ensure_dir(resolved.pdf_dir)
    ensure_dir(resolved.html_dir)
    ensure_dir(resolved.xml_dir)


def build_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""

    class _RecordSleepAction(argparse.Action):
        """Track explicit ``--sleep`` usage to disambiguate from the default."""

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: Any,
            option_string: Optional[str] = None,
        ) -> None:
            setattr(namespace, self.dest, values)
            setattr(namespace, "_sleep_explicit", True)

    parser = argparse.ArgumentParser(
        description="Download OpenAlex PDFs for a topic and year range with resolvers.",
    )
    parser.add_argument("--topic", type=str, help="Free-text topic search.")
    parser.add_argument(
        "--topic-id",
        type=str,
        help="OpenAlex Topic ID (e.g., https://openalex.org/T12345). Overrides --topic.",
    )
    parser.add_argument("--year-start", type=int, required=True, help="Start year (inclusive).")
    parser.add_argument("--year-end", type=int, required=True, help="End year (inclusive).")
    parser.add_argument("--out", type=Path, default=Path("./pdfs"), help="Output folder for PDFs.")
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Folder for HTML responses (default: sibling 'HTML').",
    )
    parser.add_argument(
        "--xml-out",
        type=Path,
        default=None,
        help="Folder for XML responses (default: sibling 'XML').",
    )
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Create timestamped run directories under --out with separate PDF, HTML, and XML folders.",
    )
    parser.add_argument(
        "--content-addressed",
        action="store_true",
        help="Store PDFs using content-addressed paths with friendly symlinks.",
    )
    parser.add_argument(
        "--warm-manifest-cache",
        action="store_true",
        help="Eagerly load the manifest URL index into memory (legacy behaviour).",
    )
    parser.add_argument(
        "--verify-cache-digest",
        action="store_true",
        help="Force cached artifact validation to recompute SHA-256 even when size/mtime match.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSONL log.",
    )
    parser.add_argument(
        "--mailto", type=str, default=None, help="Email for the OpenAlex polite pool."
    )
    parser.add_argument("--per-page", type=int, default=200, help="Results per page (1-200).")
    parser.add_argument("--max", type=int, default=None, help="Maximum works to process.")
    parser.add_argument("--oa-only", action="store_true", help="Only consider open-access works.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        action=_RecordSleepAction,
        help="Sleep seconds between works (sequential mode).",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Bypass robots.txt checks (defaults to respecting policies).",
    )
    rate_group = parser.add_argument_group("Rate limiting")
    rate_group.add_argument(
        "--rate",
        dest="rate_override",
        action="append",
        default=[],
        metavar="HOST[.ROLE]=LIMIT/INTERVAL[,…]",
        help=(
            "Override limiter windows (e.g., --rate api.openalex.org=10/s,5000/h "
            "or --rate export.arxiv.org.artifact=1/min)."
        ),
    )
    rate_group.add_argument(
        "--rate-mode",
        dest="rate_mode_override",
        action="append",
        default=[],
        metavar="HOST[.ROLE]=MODE",
        help="Set limiter mode per host (raise or wait[:ms], e.g., api.crossref.org=wait:250).",
    )
    rate_group.add_argument(
        "--rate-max-delay",
        dest="rate_max_delay_override",
        action="append",
        default=[],
        metavar="HOST.ROLE=MS",
        help="Override maximum wait milliseconds for a specific host role (e.g., host.artifact=5000).",
    )
    rate_group.add_argument(
        "--rate-backend",
        dest="rate_backend",
        default=None,
        choices=["memory", "sqlite", "redis", "postgres"],
        help="Pyrate-limiter backend for rate limiting storage (default: memory).",
    )
    rate_group.add_argument(
        "--rate-max-inflight",
        dest="rate_max_inflight",
        type=int,
        default=None,
        metavar="N",
        help="Global in-flight request ceiling (default: 500).",
    )
    rate_group.add_argument(
        "--rate-aimd-enabled",
        dest="rate_aimd_enabled",
        action="store_true",
        help="Enable AIMD dynamic rate tuning based on 429 responses (default: disabled).",
    )
    rate_group.add_argument(
        "--rate-config",
        dest="rate_config_path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to ratelimits.yaml configuration file.",
    )

    url_group = parser.add_argument_group("URL normalization")
    url_group.add_argument(
        "--url-default-scheme",
        type=str,
        default=None,
        metavar="SCHEME",
        help="Override the default scheme applied to URLs without an explicit scheme (default: https).",
    )
    url_group.add_argument(
        "--url-param-allowlist",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "Query parameter allowlist specification (e.g., 'page,id' or 'example.com:id,token;site.org:page')."
        ),
    )
    url_group.add_argument(
        "--url-filter-landing",
        dest="url_filter_landing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable (or disable with --no-url-filter-landing) landing-page parameter filtering.",
    )
    rate_group.add_argument(
        "--rate-backend",
        dest="rate_backend_spec",
        type=str,
        default=None,
        metavar="BACKEND[:key=value,…]",
        help="Select limiter backend (memory, multiprocess, sqlite:path=/tmp/rl.db, redis:url=...).",
    )
    rate_group.add_argument(
        "--rate-disable",
        dest="rate_disable",
        action="store_true",
        help="Bypass the centralized rate limiter (pilot fallback; disables quota enforcement).",
    )

    # Circuit breaker arguments
    breaker_group = parser.add_argument_group("Circuit breakers")
    breaker_group.add_argument(
        "--breaker",
        dest="breaker_host_overrides",
        action="append",
        default=[],
        help="Host-specific breaker settings: HOST=fail_max:5,reset:60,retry_after_cap:900",
    )
    breaker_group.add_argument(
        "--breaker-role",
        dest="breaker_role_overrides",
        action="append",
        default=[],
        help="Role-specific breaker settings: HOST:ROLE=fail_max:4,reset:45,trial_calls:2",
    )
    breaker_group.add_argument(
        "--breaker-resolver",
        dest="breaker_resolver_overrides",
        action="append",
        default=[],
        help="Resolver-specific breaker settings: NAME=fail_max:4,reset:45",
    )
    breaker_group.add_argument(
        "--breaker-defaults",
        dest="breaker_defaults_override",
        type=str,
        help="Default breaker settings: fail_max:5,reset:60,retry_after_cap:900",
    )
    breaker_group.add_argument(
        "--breaker-classify",
        dest="breaker_classify_override",
        type=str,
        help="Failure classification: failure=429,500,... neutral=401,403,...",
    )
    breaker_group.add_argument(
        "--breaker-rolling",
        dest="breaker_rolling_override",
        type=str,
        help="Rolling window settings: enabled:true,window:30,thresh:6,cooldown:60",
    )
    breaker_group.add_argument(
        "--breaker-config",
        dest="breaker_config_path",
        type=Path,
        help="Path to breaker configuration YAML file",
    )

    parser.add_argument(
        "--log-rotate",
        type=_parse_size,
        default=None,
        metavar="SIZE",
        help="Rotate JSONL logs after SIZE bytes (e.g., 250MB).",
    )

    openalex_group = parser.add_argument_group("OpenAlex pagination")
    openalex_group.add_argument(
        "--openalex-retry-attempts",
        type=int,
        default=3,
        help="Number of times to retry a failed OpenAlex page request before giving up.",
    )
    openalex_group.add_argument(
        "--openalex-retry-backoff",
        type=float,
        default=1.0,
        help="Base backoff (seconds) used between OpenAlex pagination retries.",
    )
    openalex_group.add_argument(
        "--openalex-retry-max-delay",
        type=float,
        default=60.0,
        help=(
            "Ceiling applied to OpenAlex pagination retry sleeps (seconds)."
            " Retry-After headers and exponential backoff are honoured up to this limit."
        ),
    )

    classifier_group = parser.add_argument_group("Classifier settings")
    classifier_group.add_argument(
        "--sniff-bytes",
        type=int,
        default=DEFAULT_SNIFF_BYTES,
        help=f"Bytes to buffer before inferring payload type (default: {DEFAULT_SNIFF_BYTES}).",
    )
    classifier_group.add_argument(
        "--min-pdf-bytes",
        type=int,
        default=DEFAULT_MIN_PDF_BYTES,
        help=f"Minimum PDF size required when HEAD precheck fails (default: {DEFAULT_MIN_PDF_BYTES}).",
    )
    classifier_group.add_argument(
        "--tail-check-bytes",
        type=int,
        default=DEFAULT_TAIL_CHECK_BYTES,
        help=f"Tail window size used to detect embedded HTML (default: {DEFAULT_TAIL_CHECK_BYTES}).",
    )

    resolver_group = parser.add_argument_group("Resolver settings")
    resolver_group.add_argument(
        "--resolver-config", type=str, default=None, help="Path to resolver config (YAML/JSON)."
    )
    resolver_group.add_argument(
        "--resolver-order",
        type=str,
        default=None,
        help="Comma-separated resolver order override (e.g., 'unpaywall,crossref').",
    )
    resolver_group.add_argument(
        "--resolver-preset",
        choices=["fast", "broad"],
        default=None,
        help="Shortcut resolver ordering preset ('fast' prioritises OA, 'broad' keeps defaults).",
    )
    resolver_group.add_argument(
        "--unpaywall-email", type=str, default=None, help="Override Unpaywall email credential."
    )
    resolver_group.add_argument(
        "--core-api-key", type=str, default=None, help="CORE API key override."
    )
    resolver_group.add_argument(
        "--semantic-scholar-api-key",
        type=str,
        default=None,
        help="Semantic Scholar Graph API key override.",
    )
    resolver_group.add_argument(
        "--doaj-api-key", type=str, default=None, help="DOAJ API key override."
    )
    resolver_group.add_argument(
        "--disable-resolver",
        action="append",
        default=[],
        help="Disable a resolver by name (can be repeated).",
    )
    resolver_group.add_argument(
        "--enable-resolver",
        action="append",
        default=[],
        help="Enable a resolver by name (can be repeated).",
    )
    resolver_group.add_argument(
        "--max-resolver-attempts",
        type=int,
        default=None,
        help="Maximum resolver attempts per work.",
    )
    resolver_group.add_argument(
        "--resolver-timeout",
        type=float,
        default=None,
        help="Default timeout (seconds) for resolver HTTP requests.",
    )

    # Wayback-specific options
    wayback_group = parser.add_argument_group("Wayback Machine settings")
    wayback_group.add_argument(
        "--wayback-year-window",
        type=int,
        default=2,
        help="Search window around publication year for Wayback snapshots (default: 2).",
    )
    wayback_group.add_argument(
        "--wayback-max-snapshots",
        type=int,
        default=8,
        help="Maximum CDX snapshots to evaluate per URL (default: 8).",
    )
    wayback_group.add_argument(
        "--wayback-min-pdf-bytes",
        type=int,
        default=4096,
        help="Minimum PDF size in bytes to accept (default: 4096).",
    )
    wayback_group.add_argument(
        "--wayback-html-parse",
        dest="wayback_html_parse",
        action="store_true",
        help="Enable HTML parsing to find PDF links in archived pages.",
    )
    wayback_group.add_argument(
        "--no-wayback-html-parse",
        action="store_false",
        dest="wayback_html_parse",
        help="Disable HTML parsing for Wayback resolver.",
    )
    wayback_group.add_argument(
        "--wayback-availability",
        dest="wayback_availability_first",
        action="store_true",
        help="Attempt the availability API before querying CDX (default behaviour).",
    )
    wayback_group.add_argument(
        "--no-wayback-availability",
        dest="wayback_availability_first",
        action="store_false",
        help="Skip availability API lookups and rely on CDX snapshots only.",
    )
    wayback_group.set_defaults(wayback_html_parse=True, wayback_availability_first=True)
    resolver_group.add_argument(
        "--retry-after-cap",
        type=float,
        default=120.0,
        help=(
            "Maximum seconds honoured from Retry-After headers during resolver HTTP retries. "
            "Retries use Tenacity-managed exponential jitter; this cap prevents multi-minute sleeps."
        ),
    )
    resolver_group.add_argument(
        "--concurrent-resolvers",
        type=int,
        default=None,
        help="Maximum resolver threads per work item (default: 1).",
    )
    resolver_group.add_argument(
        "--global-url-dedup",
        dest="global_url_dedup",
        action="store_true",
        help="Skip downloads when a URL was already fetched in this run (default).",
    )
    resolver_group.add_argument(
        "--no-global-url-dedup",
        dest="global_url_dedup",
        action="store_false",
        help="Disable global URL deduplication.",
    )
    resolver_group.add_argument(
        "--global-url-dedup-cap",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of URLs hydrated from prior manifests into the persistent dedupe "
            "set (default: 100000). Set to 0 to disable the cap."
        ),
    )
    resolver_group.add_argument(
        "--head-precheck",
        dest="head_precheck",
        action="store_true",
        help="Enable resolver HEAD preflight filtering (default).",
    )
    resolver_group.add_argument(
        "--no-head-precheck",
        dest="head_precheck",
        action="store_false",
        help="Disable resolver HEAD preflight filtering.",
    )
    resolver_group.add_argument(
        "--accept",
        type=str,
        default=None,
        help="Override the Accept header sent with resolver HTTP requests.",
    )

    # HTTP caching arguments
    cache_group = parser.add_argument_group("HTTP caching (RFC 9111)")
    cache_group.add_argument(
        "--cache-config",
        dest="cache_config_path",
        type=Path,
        default=None,
        help="Path to cache configuration YAML file (see cache.yaml for examples).",
    )
    cache_group.add_argument(
        "--cache-host",
        dest="cache_host_overrides",
        action="append",
        default=[],
        metavar="HOST=TTL_S",
        help=(
            "Override cache policy for a host (e.g., --cache-host api.crossref.org=259200 "
            "or --cache-host example.com=0 to disable)."
        ),
    )
    cache_group.add_argument(
        "--cache-role",
        dest="cache_role_overrides",
        action="append",
        default=[],
        metavar="HOST:ROLE=TTL_S",
        help=(
            "Override cache policy for a host:role pair (e.g., "
            "--cache-role api.openalex.org:metadata=259200,swrv_s:180)."
        ),
    )
    cache_group.add_argument(
        "--cache-defaults",
        dest="cache_defaults_override",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "Override cache controller defaults (e.g., "
            "'cacheable_methods:GET,cacheable_statuses:200,301,allow_heuristics:false')."
        ),
    )
    cache_group.add_argument(
        "--cache-storage",
        dest="cache_storage_kind",
        choices=["file", "memory", "redis", "sqlite", "s3"],
        default=None,
        help="Cache storage backend (default: file). Choose 'memory' for ephemeral, or persistent options.",
    )
    cache_group.add_argument(
        "--cache-disable",
        dest="cache_disabled",
        action="store_true",
        help="Disable HTTP caching (bypass all cache checks, use raw HTTP client).",
    )

    parser.set_defaults(head_precheck=True, global_url_dedup=None, _sleep_explicit=False)

    parser.add_argument(
        "--log-format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help=(
            "Log format for attempts (default: jsonl). Use 'csv' to emit only CSV logs "
            "alongside SQLite/summary outputs."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str.lower,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging verbosity for CLI output (default: INFO).",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Optional CSV attempts log output path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 for sequential).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Measure resolver coverage without writing files."
    )
    parser.add_argument(
        "--list-only", action="store_true", help="Discover candidate URLs but do not fetch content."
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Resume from a manifest log, skipping completed works. When a matching "
            "manifest.sqlite3 cache is available the CLI streams resume data directly "
            "from SQLite to minimise memory usage, falling back to JSONL parsing only "
            "when necessary. Provide JSONL files for direct parsing; CSV attempt logs "
            "rely on a SQLite cache stored alongside the CSV file (e.g., resume.csv with "
            "resume.sqlite3)."
        ),
    )
    parser.add_argument(
        "--extract-text",
        dest="extract_text",
        choices=["html", "never"],
        default="never",
        help="Extract plaintext sidecars for HTML downloads ('html') or disable ('never', default).",
    )
    parser.add_argument(
        "--extract-html-text",
        dest="extract_text",
        action="store_const",
        const="html",
        help=argparse.SUPPRESS,
    )

    return parser


def parse_args(
    parser: argparse.ArgumentParser, argv: Optional[List[str]] = None
) -> argparse.Namespace:
    """Parse CLI arguments using ``parser`` and optional argv override."""
    args = parser.parse_args(argv)
    _apply_rate_env_overrides(args)
    _apply_url_env_overrides(args)
    return args


def _apply_rate_env_overrides(args: argparse.Namespace) -> None:
    """Augment CLI-specified rate overrides with environment variables."""

    def _extend_list(target: List[str], env_var: str) -> None:
        raw = os.environ.get(env_var)
        if not raw:
            return
        for segment in raw.split(";"):
            token = segment.strip()
            if token:
                target.append(token)

    if not hasattr(args, "rate_override"):
        args.rate_override = []
    if not hasattr(args, "rate_mode_override"):
        args.rate_mode_override = []
    if not hasattr(args, "rate_max_delay_override"):
        args.rate_max_delay_override = []
    if not hasattr(args, "rate_disable"):
        args.rate_disable = False

    _extend_list(args.rate_override, "DOCSTOKG_RATE")
    _extend_list(args.rate_mode_override, "DOCSTOKG_RATE_MODE")
    _extend_list(args.rate_max_delay_override, "DOCSTOKG_RATE_MAX_DELAY")

    if getattr(args, "rate_backend_spec", None) is None:
        env_backend = os.environ.get("DOCSTOKG_RATE_BACKEND")
        if env_backend:
            args.rate_backend_spec = env_backend.strip()

    env_disable = os.environ.get("DOCSTOKG_RATE_DISABLED")
    if env_disable:
        token = env_disable.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            args.rate_disable = True
        elif token in {"0", "false", "no", "off"}:
            if not getattr(args, "rate_disable", False):
                args.rate_disable = False
        else:
            LOGGER.warning(
                "Ignoring unrecognised DOCSTOKG_RATE_DISABLED value %r (expected true/false).",
                env_disable,
            )


def _apply_url_env_overrides(args: argparse.Namespace) -> None:
    """Overlay URL policy overrides sourced from environment variables."""

    def _coerce_bool(env_name: str, raw: str) -> Optional[bool]:
        token = raw.strip().lower()
        if not token:
            return None
        if token in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "f", "no", "n", "off"}:
            return False
        LOGGER.warning(
            "Ignoring unrecognised %s value %r (expected true/false).",
            env_name,
            raw,
        )
        return None

    if getattr(args, "url_default_scheme", None) in (None, ""):
        scheme = os.environ.get("DOCSTOKG_URL_DEFAULT_SCHEME")
        if scheme:
            args.url_default_scheme = scheme.strip()

    if getattr(args, "url_filter_landing", None) is None:
        raw_filter = os.environ.get("DOCSTOKG_URL_FILTER_LANDING")
        if raw_filter is not None:
            parsed = _coerce_bool("DOCSTOKG_URL_FILTER_LANDING", raw_filter)
            if parsed is not None:
                args.url_filter_landing = parsed

    if not getattr(args, "url_param_allowlist", None):
        allowlist = os.environ.get("DOCSTOKG_URL_PARAM_ALLOWLIST")
        if allowlist:
            args.url_param_allowlist = allowlist.strip()


def _parse_rate_override_spec(value: str) -> Tuple[str, Optional[str], List[Rate]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("rate override must use HOST[.role]=rate format.")
    target, spec = value.split("=", 1)
    host, role = _split_host_role(target)
    tokens = [segment.strip() for segment in spec.split(",") if segment.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("rate override requires at least one limit/interval pair.")
    rates = [_parse_rate_window_token(token) for token in tokens]
    return host, role, rates


def _parse_rate_mode_spec(value: str) -> Tuple[str, Optional[str], str, Optional[int]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("rate mode override must use HOST[.role]=MODE syntax.")
    target, spec = value.split("=", 1)
    host, role = _split_host_role(target)
    spec = spec.strip().lower()
    if not spec:
        raise argparse.ArgumentTypeError("rate mode specification cannot be empty.")

    if spec.startswith("wait"):
        delay_ms: Optional[int] = None
        if ":" in spec:
            _, delay_text = spec.split(":", 1)
            delay_ms = _parse_delay_ms(delay_text.strip())
        return host, role, "wait", delay_ms

    if spec in {"raise", "block"}:
        return host, role, "raise", 0

    raise argparse.ArgumentTypeError(
        "rate mode must be 'raise' or 'wait[:milliseconds]' (e.g., wait:250)."
    )


def _parse_rate_max_delay_spec(value: str) -> Tuple[str, str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("rate max delay must use HOST.role=milliseconds syntax.")
    target, delay_text = value.split("=", 1)
    host, role = _split_host_role(target)
    if role is None:
        raise argparse.ArgumentTypeError(
            "rate max delay requires explicit role (e.g., host.metadata=250)."
        )
    return host, role, _parse_delay_ms(delay_text.strip())


def _parse_backend_spec(value: Optional[str]) -> Optional[BackendConfig]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if ":" not in text:
        return BackendConfig(backend=text.lower(), options={})
    backend_name, raw_options = text.split(":", 1)
    backend_name = backend_name.strip().lower()
    options: Dict[str, Any] = {}
    for part in raw_options.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"backend option '{part}' must use key=value syntax (e.g., path=/tmp/rl.db)."
            )
        key, raw_value = part.split("=", 1)
        options[key.strip()] = _coerce_backend_option(raw_value.strip())
    return BackendConfig(backend=backend_name, options=options)


def _coerce_backend_option(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_host_role(target: str) -> Tuple[str, Optional[str]]:
    token = target.strip().lower()
    if not token:
        raise argparse.ArgumentTypeError("host identifier cannot be empty.")
    if "." in token:
        host_part, candidate_role = token.rsplit(".", 1)
        host_part = host_part.strip()
        candidate_role = candidate_role.strip()
        if candidate_role in ROLE_ORDER or candidate_role == DEFAULT_ROLE:
            if not host_part:
                raise argparse.ArgumentTypeError("host segment cannot be empty.")
            return host_part, candidate_role
    return token, None


def _parse_rate_window_token(token: str) -> Rate:
    if "/" not in token:
        raise argparse.ArgumentTypeError(
            f"invalid rate token '{token}'. Expected limit/interval (e.g., 10/s or 300/m)."
        )
    limit_text, interval_text = token.split("/", 1)
    try:
        limit_value = float(limit_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid rate limit '{limit_text}'") from exc
    limit = int(math.ceil(limit_value))
    if limit <= 0:
        raise argparse.ArgumentTypeError("rate limits must be positive integers.")
    interval_ms = _parse_interval_ms(interval_text.strip())
    return Rate(limit, interval_ms)


_INTERVAL_KEYWORDS: Dict[str, int] = {
    "ms": 1,
    "millisecond": 1,
    "milliseconds": 1,
    "s": Duration.SECOND,
    "sec": Duration.SECOND,
    "second": Duration.SECOND,
    "seconds": Duration.SECOND,
    "m": Duration.MINUTE,
    "min": Duration.MINUTE,
    "minute": Duration.MINUTE,
    "minutes": Duration.MINUTE,
    "h": Duration.HOUR,
    "hr": Duration.HOUR,
    "hour": Duration.HOUR,
    "hours": Duration.HOUR,
    "d": Duration.DAY,
    "day": Duration.DAY,
    "days": Duration.DAY,
    "w": Duration.WEEK,
    "week": Duration.WEEK,
    "weeks": Duration.WEEK,
}


def _parse_interval_ms(value: str) -> int:
    token = value.strip().lower()
    if not token:
        raise argparse.ArgumentTypeError("interval component cannot be empty.")
    if token in _INTERVAL_KEYWORDS:
        return int(_INTERVAL_KEYWORDS[token])

    match = re.fullmatch(r"(?P<amount>[\d.]+)(?P<unit>[a-z]+)", token)
    if match:
        amount = float(match.group("amount"))
        unit = match.group("unit")
        if unit not in _INTERVAL_KEYWORDS:
            raise argparse.ArgumentTypeError(f"unsupported interval unit '{unit}'")
        base = _INTERVAL_KEYWORDS[unit]
        return int(math.ceil(amount * base))

    raise argparse.ArgumentTypeError(
        f"unable to parse interval '{value}'. Use formats like 3s, 1m, or 500ms."
    )


def _parse_delay_ms(value: str) -> int:
    value = value.strip().lower()
    if not value:
        raise argparse.ArgumentTypeError("delay must be a positive duration (e.g., 250 or 0.5s).")
    if value in {"inf", "infinite"}:
        raise argparse.ArgumentTypeError("infinite waits are not supported.")
    try:
        numeric = float(value)
    except ValueError:
        return _parse_interval_ms(value)
    if numeric <= 0:
        raise argparse.ArgumentTypeError("delay must be greater than 0 milliseconds.")
    return int(math.ceil(numeric))


def merge_rate_overrides(
    base_policies: Mapping[str, RolePolicy],
    *,
    rate_overrides: Iterable[Tuple[str, Optional[str], List[Rate]]],
    mode_overrides: Iterable[Tuple[str, Optional[str], str, Optional[int]]],
    delay_overrides: Iterable[Tuple[str, str, int]],
) -> Dict[str, RolePolicy]:
    """Merge CLI/env overrides into cloned host policies."""

    policies = clone_policies(base_policies)
    default_template = policies.get("default")

    def _ensure_policy(hostname: str) -> RolePolicy:
        key = hostname.lower()
        if key in policies:
            return policies[key]
        if default_template is not None:
            policies[key] = clone_role_policy(default_template)
        else:
            policies[key] = RolePolicy()
        return policies[key]

    for host, role, rates in rate_overrides:
        policy = _ensure_policy(host)
        targets = ROLE_ORDER if role is None else (role,)
        for target_role in targets:
            policy.rates[target_role] = list(rates)

    for host, role, mode, delay in mode_overrides:
        policy = _ensure_policy(host)
        targets = ROLE_ORDER if role is None else (role,)
        for target_role in targets:
            policy.mode[target_role] = mode
            if mode == "raise":
                policy.max_delay_ms[target_role] = 0
            elif delay is not None:
                policy.max_delay_ms[target_role] = delay
            elif target_role not in policy.max_delay_ms:
                policy.max_delay_ms[target_role] = 250

    for host, role, delay_ms in delay_overrides:
        policy = _ensure_policy(host)
        policy.max_delay_ms[role] = delay_ms

    return policies


def resolve_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    resolver_factory: Optional[Callable[[], List[Any]]] = None,
) -> ResolvedConfig:
    """Validate arguments, resolve configuration, and prepare run state."""
    extract_html_text = args.extract_text == "html"

    topic = args.topic.strip() if isinstance(args.topic, str) else args.topic
    topic_id_input = args.topic_id.strip() if isinstance(args.topic_id, str) else args.topic_id

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.concurrent_resolvers is not None and args.concurrent_resolvers < 1:
        parser.error("--concurrent-resolvers must be >= 1")
    if not 1 <= args.per_page <= 200:
        parser.error("--per-page must be between 1 and 200")
    if args.sleep < 0:
        parser.error("--sleep must be greater than or equal to 0")
    if args.max is not None and args.max < 0:
        parser.error("--max must be greater than or equal to 0")
    if args.year_start > args.year_end:
        parser.error("--year-start must be less than or equal to --year-end")
    if not topic and not topic_id_input:
        parser.error("Provide --topic or --topic-id.")
    if args.openalex_retry_attempts < 0:
        parser.error("--openalex-retry-attempts must be >= 0")
    if args.openalex_retry_backoff < 0:
        parser.error("--openalex-retry-backoff must be >= 0")
    if args.openalex_retry_max_delay <= 0:
        parser.error("--openalex-retry-max-delay must be > 0")
    if args.retry_after_cap <= 0:
        parser.error("--retry-after-cap must be > 0")
    if args.global_url_dedup_cap is not None and args.global_url_dedup_cap < 0:
        parser.error("--global-url-dedup-cap must be >= 0")
    for field_name in ("sniff_bytes", "min_pdf_bytes", "tail_check_bytes"):
        value = getattr(args, field_name, None)
        if value is not None and value < 0:
            parser.error(f"--{field_name.replace('_', '-')} must be non-negative")

    if args.mailto:
        apply_mailto(args.mailto)

    allowlist_global: Optional[Tuple[str, ...]] = None
    allowlist_per_domain: Optional[Dict[str, Tuple[str, ...]]] = None
    if args.url_param_allowlist:
        allowlist_global, allowlist_per_domain = parse_param_allowlist_spec(
            args.url_param_allowlist
        )

    configure_url_policy(
        default_scheme=args.url_default_scheme,
        filter_landing=args.url_filter_landing,
        param_allowlist_global=allowlist_global,
        param_allowlist_per_domain=allowlist_per_domain,
    )

    topic_id = topic_id_input
    if not topic_id and topic:
        try:
            topic_id = resolve_topic_id_if_needed(topic)
        except Exception as exc:
            LOGGER.warning("Failed to resolve topic ID for '%s': %s", topic, exc)
            topic_id = None
    query_kwargs = vars(args).copy()
    if topic is not None:
        query_kwargs["topic"] = topic
    if topic_id:
        query_kwargs["topic_id"] = topic_id
    query = build_query(argparse.Namespace(**query_kwargs))
    run_id = uuid.uuid4().hex

    def _expand_path(value: Optional[Path | str]) -> Optional[Path]:
        if value is None:
            return None
        return Path(value).expanduser().resolve(strict=False)

    base_pdf_dir = _expand_path(args.out)
    if base_pdf_dir is None:
        parser.error("--out is required")

    if not getattr(args, "_sleep_explicit", False) and getattr(args, "workers", 1) > 1:
        args.sleep = 0.0

    html_override = _expand_path(args.html_out)
    xml_override = _expand_path(args.xml_out)
    manifest_override = _expand_path(args.manifest)
    csv_override = _expand_path(args.log_csv)
    if args.staging:
        run_dir = (base_pdf_dir / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")).resolve(strict=False)
        pdf_dir = (run_dir / "PDF").resolve(strict=False)
        html_dir = (run_dir / "HTML").resolve(strict=False)
        xml_dir = (run_dir / "XML").resolve(strict=False)
        manifest_path = (run_dir / "manifest.jsonl").resolve(strict=False)
        if args.html_out:
            LOGGER.info("Staging mode overrides --html-out; using %s", html_dir)
        if args.xml_out:
            LOGGER.info("Staging mode overrides --xml-out; using %s", xml_dir)
        if manifest_override:
            LOGGER.info("Staging mode overrides --manifest; writing to %s", manifest_path)
    else:
        pdf_dir = base_pdf_dir
        html_dir = html_override or (pdf_dir.parent / "HTML")
        xml_dir = xml_override or (pdf_dir.parent / "XML")
        manifest_path = manifest_override or (pdf_dir / "manifest.jsonl")

        html_dir = Path(html_dir).expanduser().resolve(strict=False)
        xml_dir = Path(xml_dir).expanduser().resolve(strict=False)
        manifest_path = Path(manifest_path).expanduser().resolve(strict=False)

    resolver_factory = resolver_factory or default_resolvers
    resolver_instances = resolver_factory()
    resolver_names = [resolver.name for resolver in resolver_instances]
    resolver_order_override: Optional[List[str]] = None

    def _normalise_order(order: List[str]) -> List[str]:
        cleaned: List[str] = []
        unknown: List[str] = []
        for name in order:
            if name not in resolver_names:
                unknown.append(name)
            elif name not in cleaned:
                cleaned.append(name)
        if unknown:
            parser.error(
                f"Unknown resolver(s) in order override: {', '.join(sorted(set(unknown)))}"
            )
        cleaned.extend(name for name in resolver_names if name not in cleaned)
        return cleaned

    if args.resolver_order:
        raw_order = [name.strip() for name in args.resolver_order.split(",") if name.strip()]
        if not raw_order:
            parser.error("--resolver-order requires at least one resolver name.")
        resolver_order_override = _normalise_order(raw_order)
    elif getattr(args, "resolver_preset", None):
        if args.resolver_preset == "fast":
            preset = [
                "openalex",
                "unpaywall",
                "crossref",
                "landing_page",
                "arxiv",
                "pmc",
                "europe_pmc",
                "core",
                "wayback",
            ]
        else:
            preset = list(DEFAULT_RESOLVER_ORDER)
        resolver_order_override = _normalise_order(preset)

    try:
        config = load_resolver_config(args, resolver_names, resolver_order_override)
    except ValueError as exc:
        parser.error(str(exc))

    mailto_value = getattr(config, "mailto", None)
    if mailto_value:
        apply_mailto(mailto_value)

    concurrency_product = max(args.workers, 1) * max(config.max_concurrent_resolvers, 1)
    if concurrency_product > 32:
        LOGGER.warning(
            "High parallelism detected (workers x concurrent_resolvers = %s). Ensure resolver and domain rate limits are configured appropriately.",
            concurrency_product,
        )

    if manifest_path.suffix != ".jsonl":
        manifest_path = manifest_path.with_suffix(".jsonl")
    csv_path = csv_override
    if args.log_format == "csv":
        csv_path = csv_path or manifest_path.with_suffix(".csv")
    sqlite_path = manifest_path.with_suffix(".sqlite3")

    previous_url_index = ManifestUrlIndex(sqlite_path, eager=args.warm_manifest_cache)
    persistent_seen_urls: Set[str]
    if config.enable_global_url_dedup:
        allowed_classifications = {
            Classification.PDF.value.lower(),
            Classification.CACHED.value.lower(),
            Classification.XML.value.lower(),
        }
        existing_iterator = previous_url_index.iter_existing_paths()
        cap_value = config.global_url_dedup_cap
        limited_iterator = (
            islice(existing_iterator, cap_value)
            if cap_value is not None and cap_value > 0
            else existing_iterator
        )
        persistent_seen_urls = {
            url
            for url, meta in limited_iterator
            if str(meta.get("classification", "")).lower() in allowed_classifications
        }
        if cap_value is not None and cap_value > 0 and next(existing_iterator, None) is not None:
            LOGGER.info(
                "Hydrated %s persistent resolver URLs (truncated to configured cap of %s).",
                len(persistent_seen_urls),
                cap_value,
            )
    else:
        persistent_seen_urls = set()

    robots_checker: Optional[RobotsCache] = None
    if not args.ignore_robots:
        user_agent = config.polite_headers.get("User-Agent", "DocsToKGDownloader/1.0")
        robots_checker = RobotsCache(user_agent)

    manager = get_rate_limiter_manager()

    try:
        rate_override_specs = [
            _parse_rate_override_spec(value) for value in getattr(args, "rate_override", [])
        ]
        rate_mode_specs = [
            _parse_rate_mode_spec(value) for value in getattr(args, "rate_mode_override", [])
        ]
        rate_delay_specs = [
            _parse_rate_max_delay_spec(value)
            for value in getattr(args, "rate_max_delay_override", [])
        ]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    rate_disabled = bool(getattr(args, "rate_disable", False))

    if rate_disabled:
        conflicting_flags: List[str] = []
        if rate_override_specs:
            conflicting_flags.append("--rate")
        if rate_mode_specs:
            conflicting_flags.append("--rate-mode")
        if rate_delay_specs:
            conflicting_flags.append("--rate-max-delay")
        if getattr(args, "rate_backend_spec", None):
            conflicting_flags.append("--rate-backend")
        if conflicting_flags:
            parser.error(
                "--rate-disable cannot be combined with rate override options (%s)."
                % ", ".join(conflicting_flags)
            )

        configure_rate_limits(
            policies={},
            backend=BackendConfig(backend="disabled", options={}),
        )
        LOGGER.info(
            "Centralized rate limiter disabled; HTTP requests will bypass quota enforcement."
        )
        configured_policies: Dict[str, RolePolicy] = {}
        backend_config = BackendConfig(backend="disabled", options={})
    else:
        policies = merge_rate_overrides(
            manager.policies(),
            rate_overrides=rate_override_specs,
            mode_overrides=rate_mode_specs,
            delay_overrides=rate_delay_specs,
        )

        backend_config = None
        try:
            backend_config = _parse_backend_spec(getattr(args, "rate_backend_spec", None))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

        if backend_config is None:
            base_backend = manager.backend
            backend_config = BackendConfig(
                backend=base_backend.backend,
                options=(
                    dict(base_backend.options) if isinstance(base_backend.options, Mapping) else {}
                ),
            )

        try:
            validate_policies(policies)
        except ValueError as exc:
            parser.error(str(exc))

        configured_policies = clone_policies(policies)
        configure_rate_limits(policies=policies, backend=backend_config)

    rate_policies = configured_policies

    cache_config_path = _expand_path(args.cache_config_path)
    cache_disabled = bool(getattr(args, "cache_disabled", False))

    if cache_disabled:
        cache_config = None
        LOGGER.info("HTTP caching disabled via --cache-disable")
    elif cache_config_path:
        try:
            cache_config = load_cache_config(
                cache_config_path,
                env=os.environ,
                cli_host_overrides=getattr(args, "cache_host_overrides", []),
                cli_role_overrides=getattr(args, "cache_role_overrides", []),
                cli_defaults_override=getattr(args, "cache_defaults_override", None),
            )
            cache_router = CacheRouter(cache_config)
            LOGGER.info(
                "Loaded cache configuration from %s with %d hosts",
                cache_config_path,
                len(cache_config.hosts),
            )
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("Cache policy table:\n%s", cache_router.print_effective_policy())
        except Exception as exc:
            LOGGER.warning(
                "Failed to load cache configuration from %s: %s; continuing without caching",
                cache_config_path,
                exc,
            )
            cache_config = None
    else:
        cache_config = None

    # Phase 7: Load Pyrate-Limiter rate limiting configuration
    rate_config_path = _expand_path(getattr(args, "rate_config_path", None))
    rate_config = None

    if rate_config_path:
        try:
            rate_config = load_rate_config(
                rate_config_path,
                env=os.environ,
                cli_backend=getattr(args, "rate_backend", None),
                cli_global_max_inflight=getattr(args, "rate_max_inflight", None),
                cli_aimd_enabled=getattr(args, "rate_aimd_enabled", False),
            )
            LOGGER.info(
                "Loaded Pyrate-Limiter rate config from %s with %d hosts",
                rate_config_path,
                len(rate_config.hosts),
            )
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "Rate config backend: %s, global_max_inflight: %s, AIMD: %s",
                    rate_config.backend.kind,
                    rate_config.global_max_inflight,
                    rate_config.aimd.enabled,
                )
        except Exception as exc:
            LOGGER.warning(
                "Failed to load Pyrate-Limiter rate config from %s: %s; continuing with defaults",
                rate_config_path,
                exc,
            )
            rate_config = None
    else:
        # Try loading with defaults if CLI overrides are provided
        if (
            getattr(args, "rate_backend", None)
            or getattr(args, "rate_max_inflight", None) is not None
            or getattr(args, "rate_aimd_enabled", False)
        ):
            try:
                rate_config = load_rate_config(
                    None,
                    env=os.environ,
                    cli_backend=getattr(args, "rate_backend", None),
                    cli_global_max_inflight=getattr(args, "rate_max_inflight", None),
                    cli_aimd_enabled=getattr(args, "rate_aimd_enabled", False),
                )
                LOGGER.info(
                    "Loaded Pyrate-Limiter defaults with CLI overrides (backend: %s)",
                    rate_config.backend.kind,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load Pyrate-Limiter config with CLI overrides: %s; continuing with defaults",
                    exc,
                )
                rate_config = None

    return ResolvedConfig(
        args=args,
        run_id=run_id,
        query=query,
        pdf_dir=pdf_dir,
        html_dir=html_dir,
        xml_dir=xml_dir,
        manifest_path=manifest_path,
        csv_path=csv_path,
        sqlite_path=sqlite_path,
        resolver_instances=resolver_instances,
        resolver_config=config,
        previous_url_index=previous_url_index,
        persistent_seen_urls=persistent_seen_urls,
        robots_checker=robots_checker,
        concurrency_product=concurrency_product,
        extract_html_text=extract_html_text,
        verify_cache_digest=args.verify_cache_digest,
        openalex_retry_attempts=args.openalex_retry_attempts,
        openalex_retry_backoff=args.openalex_retry_backoff,
        openalex_retry_max_delay=args.openalex_retry_max_delay,
        retry_after_cap=args.retry_after_cap,
        rate_policies=rate_policies,
        rate_backend=backend_config,
        rate_config=rate_config,  # This field is now populated.
        cache_config=cache_config,
        cache_disabled=cache_disabled,
    )


def _parse_size(value: str) -> int:
    """Parse human-friendly size strings (e.g., ``10MB``) into bytes."""

    text = (value or "").strip().lower().replace(",", "").replace("_", "")
    if not text:
        raise argparse.ArgumentTypeError("size value cannot be empty")
    from DocsToKG.ContentDownload.core import parse_size

    try:
        return parse_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_query(args: argparse.Namespace) -> Works:
    """Build a pyalex Works query based on CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured Works query object ready for iteration.
    """
    query = Works()
    if args.topic_id:
        query = query.filter(topics={"id": args.topic_id})
    else:
        query = query.search(args.topic)

    query = query.filter(publication_year=f"{args.year_start}-{args.year_end}")
    if args.oa_only:
        query = query.filter(open_access={"is_oa": True})

    query = query.select(
        [
            "id",
            "title",
            "publication_year",
            "ids",
            "open_access",
            "best_oa_location",
            "primary_location",
            "locations",
        ]
    )

    query = query.sort(publication_date="desc")
    return query


@lru_cache(maxsize=128)
def _lookup_topic_id(topic_text: str) -> Optional[str]:
    """Cached helper to resolve an OpenAlex topic identifier."""
    try:
        query = Topics().search(topic_text).select(["id"]).per_page(1)
        hits = query.get()
    except Exception as exc:  # pragma: no cover - network guard
        LOGGER.warning("Topic lookup failed for %s: %s", topic_text, exc)
        return None
    if not hits:
        return None
    candidate: Optional[Dict[str, Any]]
    if isinstance(hits, list):
        if not hits:
            return None
        candidate = hits[0] if isinstance(hits[0], dict) else None
    elif isinstance(hits, dict):
        candidate = hits
    else:
        candidate = None

    if not candidate:
        return None

    resolved = candidate.get("id")
    if resolved:
        LOGGER.info("Resolved topic '%s' -> %s", topic_text, resolved)
    return resolved


def resolve_topic_id_if_needed(topic_text: Optional[str]) -> Optional[str]:
    """Resolve a textual topic label into an OpenAlex topic identifier.

    Args:
        topic_text: Free-form topic text supplied via CLI.

    Returns:
        OpenAlex topic identifier string if resolved, else None.
    """
    if not topic_text:
        return None
    normalized = topic_text.strip()
    if not normalized:
        return None
    return _lookup_topic_id(normalized)
