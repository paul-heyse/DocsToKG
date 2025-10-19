"""Command-line argument parsing for DocsToKG content downloads.

Provides helpers that transform CLI inputs into structured configuration
objects used by the downloader. The utilities here manage resolver bootstrap,
run directory preparation, and OpenAlex query assembly without triggering any
network activity at import time.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests
from pyalex import Topics, Works

from DocsToKG.ContentDownload.core import (
    DEFAULT_MIN_PDF_BYTES,
    DEFAULT_SNIFF_BYTES,
    DEFAULT_TAIL_CHECK_BYTES,
    Classification,
)
from DocsToKG.ContentDownload.download import RobotsCache, ensure_dir
from DocsToKG.ContentDownload.pipeline import load_resolver_config
from DocsToKG.ContentDownload.resolvers import DEFAULT_RESOLVER_ORDER, default_resolvers
from DocsToKG.ContentDownload.telemetry import ManifestUrlIndex
from DocsToKG.ContentDownload.pyalex_shim import apply_mailto

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


def bootstrap_run_environment(resolved: ResolvedConfig) -> None:
    """Initialise directories required for a resolved download run."""

    ensure_dir(resolved.pdf_dir)
    ensure_dir(resolved.html_dir)
    ensure_dir(resolved.xml_dir)


def build_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
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
        default=0.05,
        help="Sleep seconds between works (sequential mode).",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Bypass robots.txt checks (defaults to respecting policies).",
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
    resolver_group.add_argument(
        "--concurrent-resolvers",
        type=int,
        default=None,
        help="Maximum resolver threads per work item (default: 1).",
    )
    resolver_group.add_argument(
        "--max-concurrent-per-host",
        type=int,
        default=None,
        help="Maximum concurrent downloads per host (default: 3). Set to 0 to disable.",
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
        "--domain-min-interval",
        dest="domain_min_interval",
        type=_parse_domain_interval,
        action="append",
        default=[],
        metavar="DOMAIN=SECONDS",
        help="Enforce a minimum interval between requests to a domain. Repeat to configure multiple domains.",
    )
    resolver_group.add_argument(
        "--domain-token-bucket",
        dest="domain_token_bucket",
        type=_parse_domain_token_bucket,
        action="append",
        default=[],
        metavar="DOMAIN=RPS[:capacity=N]",
        help="Configure per-domain token buckets (e.g., example.org=0.5:capacity=2).",
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

    parser.set_defaults(head_precheck=True, global_url_dedup=None)

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
    return parser.parse_args(argv)


def resolve_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    resolver_factory: Optional[Callable[[], List[Any]]] = None,
) -> ResolvedConfig:
    """Validate arguments, resolve configuration, and prepare run state."""
    extract_html_text = args.extract_text == "html"

    topic = args.topic.strip() if isinstance(args.topic, str) else args.topic
    topic_id_input = (
        args.topic_id.strip() if isinstance(args.topic_id, str) else args.topic_id
    )

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.concurrent_resolvers is not None and args.concurrent_resolvers < 1:
        parser.error("--concurrent-resolvers must be >= 1")
    if args.max_concurrent_per_host is not None and args.max_concurrent_per_host < 0:
        parser.error("--max-concurrent-per-host must be >= 0")
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
    for field_name in ("sniff_bytes", "min_pdf_bytes", "tail_check_bytes"):
        value = getattr(args, field_name, None)
        if value is not None and value < 0:
            parser.error(f"--{field_name.replace('_', '-')} must be non-negative")

    if args.mailto:
        apply_mailto(args.mailto)

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

    html_override = _expand_path(args.html_out)
    xml_override = _expand_path(args.xml_out)
    manifest_override = _expand_path(args.manifest)
    if args.staging:
        run_dir = (base_pdf_dir / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")).resolve(
            strict=False
        )
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
    csv_path = args.log_csv
    if args.log_format == "csv":
        csv_path = csv_path or manifest_path.with_suffix(".csv")
    sqlite_path = manifest_path.with_suffix(".sqlite3")

    previous_url_index = ManifestUrlIndex(sqlite_path, eager=args.warm_manifest_cache)
    persistent_seen_urls: Set[str] = {
        url
        for url, meta in previous_url_index.iter_existing_paths()
        if str(meta.get("classification", "")).lower()
        in {Classification.PDF.value, Classification.CACHED.value, Classification.XML.value}
    }

    robots_checker: Optional[RobotsCache] = None
    if not args.ignore_robots:
        user_agent = config.polite_headers.get("User-Agent", "DocsToKGDownloader/1.0")
        robots_checker = RobotsCache(user_agent)
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


def _parse_domain_interval(value: str) -> Tuple[str, float]:
    """Parse ``DOMAIN=SECONDS`` CLI arguments for domain throttling.

    Args:
        value: Argument provided via ``--domain-min-interval``.

    Returns:
        Tuple containing the normalized domain name and interval seconds.

    Raises:
        argparse.ArgumentTypeError: If the argument is malformed or negative.
    """

    if "=" not in value:
        raise argparse.ArgumentTypeError("domain interval must use the format domain=seconds")
    domain, interval = value.split("=", 1)
    domain = domain.strip().lower()
    if not domain:
        raise argparse.ArgumentTypeError("domain component cannot be empty")
    try:
        seconds = float(interval)
    except ValueError as exc:  # pragma: no cover - defensive parsing guard
        raise argparse.ArgumentTypeError(
            f"invalid interval for domain '{domain}': {interval}"
        ) from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"interval for domain '{domain}' must be non-negative")
    return domain, seconds


def _parse_domain_token_bucket(value: str) -> Tuple[str, Dict[str, float]]:
    """Parse ``DOMAIN=RPS[:capacity=X]`` specifications into bucket configs."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "domain token bucket must use the format domain=rate[:capacity=N]"
        )
    domain, spec = value.split("=", 1)
    domain = domain.strip().lower()
    if not domain:
        raise argparse.ArgumentTypeError("domain component cannot be empty")
    rate: Optional[float] = None
    capacity: Optional[float] = None
    parts = [segment.strip() for segment in spec.split(":") if segment.strip()]
    for index, part in enumerate(parts):
        if "=" in part:
            key, raw = part.split("=", 1)
            key = key.strip().lower()
            raw = raw.strip()
            try:
                value_float = float(raw)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid numeric value '{raw}' in token bucket spec"
                ) from exc
            if key in {"rate", "rps", "rate_per_second"}:
                rate = value_float
            elif key in {"capacity", "burst"}:
                capacity = value_float
            else:
                raise argparse.ArgumentTypeError(f"unknown token bucket key '{key}'")
        else:
            try:
                value_float = float(part)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid token bucket value '{part}'") from exc
            if rate is None:
                rate = value_float
            elif capacity is None:
                capacity = value_float
            else:
                raise argparse.ArgumentTypeError("unexpected extra token bucket value")

    if rate is None or rate <= 0:
        raise argparse.ArgumentTypeError("token bucket rate must be a positive number")
    if capacity is None or capacity <= 0:
        capacity = 1.0

    return domain, {"rate_per_second": float(rate), "capacity": float(capacity)}


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
    except requests.RequestException as exc:  # pragma: no cover - network guard
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
