# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.network",
#   "purpose": "Provide secure streaming downloads, DNS validation, and retry-aware HTTP helpers",
#   "sections": [
#     {"id": "infrastructure", "name": "Networking Infrastructure & Constants", "anchor": "INF", "kind": "infra"},
#     {"id": "dns", "name": "DNS & Host Validation", "anchor": "DNS", "kind": "helpers"},
#     {"id": "session", "name": "Session Pools & Rate Limiting", "anchor": "SES", "kind": "api"},
#     {"id": "streaming", "name": "Streaming Downloader", "anchor": "STR", "kind": "api"},
#     {"id": "helpers", "name": "Download Helpers & Security Checks", "anchor": "HLP", "kind": "helpers"}
#   ]
# }
# === /NAVMAP ===

"""Networking utilities for ontology downloads.

This module manages resilient HTTP downloads: DNS caching, session pooling,
range resume, provenance logging, retry-after aware throttling, and security
guards around redirects, content types, and host allowlists.  It provides the
streaming helpers consumed by resolvers and the planner when fetching ontology
artefacts.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import random
import shutil
import socket
import threading
import time
import unicodedata
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
)
from urllib.parse import ParseResult, urljoin, urlparse, urlunparse

import pooch
import requests

from ..cancellation import CancellationToken
from ..errors import ConfigError, DownloadFailure, OntologyDownloadError, PolicyError
from ..settings import DownloadConfiguration
from .filesystem import _compute_file_hash, _materialize_cached_file, sanitize_filename
from .rate_limit import TokenBucket, apply_retry_after, get_bucket

try:  # pragma: no cover - psutil may be unavailable in minimal environments
    import psutil  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when psutil cannot be imported
    psutil = None  # type: ignore[assignment]
    _PROCESS = None
else:
    try:
        _PROCESS = psutil.Process()
    except Exception:  # pragma: no cover - defensive against exotic psutil failures
        _PROCESS = None

_DNS_CACHE_TTL = 120.0
_DNS_CACHE_MAX_ENTRIES = 4096
_DNS_CACHE_LOCK = threading.Lock()
_DNS_CACHE: "OrderedDict[str, Tuple[float, List[Tuple]]]" = OrderedDict()
_DNS_STUB_LOCK = threading.Lock()
_DNS_STUBS: Dict[str, Callable[[str], List[Tuple]]] = {}

_RDF_FORMAT_LABELS = {
    "application/rdf+xml": "RDF/XML",
    "text/turtle": "Turtle",
    "application/n-triples": "N-Triples",
    "application/trig": "TriG",
    "application/ld+json": "JSON-LD",
}
_RDF_ALIAS_GROUPS = {
    "application/rdf+xml": {"application/rdf+xml", "application/xml", "text/xml"},
    "text/turtle": {"text/turtle", "application/x-turtle"},
    "application/n-triples": {"application/n-triples", "text/plain"},
    "application/trig": {"application/trig"},
    "application/ld+json": {"application/ld+json"},
}
RDF_MIME_ALIASES: set[str] = set()
RDF_MIME_FORMAT_LABELS: Dict[str, str] = {}
for canonical, aliases in _RDF_ALIAS_GROUPS.items():
    label = _RDF_FORMAT_LABELS[canonical]
    for alias in aliases:
        RDF_MIME_ALIASES.add(alias)
        RDF_MIME_FORMAT_LABELS[alias] = label

SESSION_POOL_CACHE_DEFAULT = 2
_RETRYABLE_HTTP_STATUSES = {408, 409, 416, 425, 429, 500, 502, 503, 504}

T = TypeVar("T")


def _enforce_idn_safety(host: str) -> None:
    """Validate internationalized hostnames and reject suspicious patterns."""

    if all(ord(char) < 128 for char in host):
        return

    scripts = set()
    for char in host:
        if ord(char) < 128:
            if char.isalpha():
                scripts.add("LATIN")
            continue

        category = unicodedata.category(char)
        if category in {"Mn", "Me", "Cf"}:
            raise ConfigError("Internationalized host contains invisible characters")

        try:
            name = unicodedata.name(char)
        except ValueError as exc:
            raise ConfigError("Internationalized host contains unknown characters") from exc

        for script in ("LATIN", "CYRILLIC", "GREEK"):
            if script in name:
                scripts.add(script)
                break

    if len(scripts) > 1:
        raise ConfigError("Internationalized host mixes multiple scripts")


def _rebuild_netloc(parsed: ParseResult, ascii_host: str) -> str:
    """Reconstruct URL netloc with a normalized hostname."""

    host_component = ascii_host
    if ":" in host_component and not host_component.startswith("["):
        host_component = f"[{host_component}]"

    port = f":{parsed.port}" if parsed.port else ""
    return f"{host_component}{port}"


def _cached_getaddrinfo(host: str) -> List[Tuple]:
    """Resolve *host* using an expiring LRU cache to avoid repeated DNS lookups."""

    ascii_host = host.lower()
    with _DNS_STUB_LOCK:
        stub = _DNS_STUBS.get(ascii_host)
    if stub is not None:
        results = stub(host)
        now = time.monotonic()
        expires_at = now + _DNS_CACHE_TTL
        with _DNS_CACHE_LOCK:
            _DNS_CACHE[ascii_host] = (expires_at, results)
            _DNS_CACHE.move_to_end(ascii_host)
            _prune_dns_cache(now)
        return results

    now = time.monotonic()
    with _DNS_CACHE_LOCK:
        cached = _DNS_CACHE.get(ascii_host)
        if cached is not None:
            expires_at, results = cached
            if expires_at > now:
                _DNS_CACHE.move_to_end(ascii_host)
                _prune_dns_cache(now)
                return results
            _DNS_CACHE.pop(ascii_host, None)
        _prune_dns_cache(now)

    results = socket.getaddrinfo(host, None)
    expires_at = now + _DNS_CACHE_TTL
    with _DNS_CACHE_LOCK:
        _DNS_CACHE[ascii_host] = (expires_at, results)
        _DNS_CACHE.move_to_end(ascii_host)
        _prune_dns_cache(now)
    return results


def _prune_dns_cache(current_time: float) -> None:
    """Expire stale DNS entries and enforce the cache size bound."""

    while _DNS_CACHE:
        first_key, (expires_at, _) = next(iter(_DNS_CACHE.items()))
        if expires_at > current_time:
            break
        _DNS_CACHE.pop(first_key, None)

    while len(_DNS_CACHE) > _DNS_CACHE_MAX_ENTRIES:
        _DNS_CACHE.popitem(last=False)


def register_dns_stub(host: str, handler: Callable[[str], List[Tuple]]) -> None:
    """Register a DNS stub callable for ``host`` used during testing."""

    ascii_host = host.lower()
    with _DNS_STUB_LOCK:
        _DNS_STUBS[ascii_host] = handler
    with _DNS_CACHE_LOCK:
        _DNS_CACHE.pop(ascii_host, None)


def clear_dns_stubs() -> None:
    """Remove all registered DNS stubs and purge cached stub lookups."""

    with _DNS_STUB_LOCK:
        stubbed_hosts = list(_DNS_STUBS.keys())
        _DNS_STUBS.clear()

    if not stubbed_hosts:
        return

    with _DNS_CACHE_LOCK:
        for ascii_host in stubbed_hosts:
            _DNS_CACHE.pop(ascii_host, None)


def validate_url_security(url: str, http_config: Optional[DownloadConfiguration] = None) -> str:
    """Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists."""

    parsed = urlparse(url)
    logger = logging.getLogger("DocsToKG.OntologyDownload")
    if parsed.username or parsed.password:
        raise PolicyError("Credentials in URLs are not allowed")

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise PolicyError("Only HTTP(S) URLs are allowed for ontology downloads")

    host = parsed.hostname
    if not host:
        raise PolicyError("URL must include hostname")

    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False

    ascii_host = host.lower()
    if not is_ip:
        _enforce_idn_safety(host)
        try:
            ascii_host = host.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise PolicyError(f"Invalid internationalized hostname: {host}") from exc

    parsed = parsed._replace(netloc=_rebuild_netloc(parsed, ascii_host))

    allowed_exact: Set[str] = set()
    allowed_suffixes: Set[str] = set()
    allowed_host_ports: Dict[str, Set[int]] = {}
    allowed_ip_literals: Set[str] = set()
    allowed_port_set = http_config.allowed_port_set() if http_config else {80, 443}
    if http_config:
        normalized = http_config.normalized_allowed_hosts()
        if normalized:
            allowed_exact, allowed_suffixes, allowed_host_ports, allowed_ip_literals = normalized

    allow_private = False
    if allowed_exact or allowed_suffixes:
        matched_exact = ascii_host in allowed_exact
        matched_suffix = any(
            ascii_host == suffix or ascii_host.endswith(f".{suffix}") for suffix in allowed_suffixes
        )
        if not (matched_exact or matched_suffix):
            raise PolicyError(f"Host {host} not in allowlist")

        if matched_exact and ascii_host in allowed_ip_literals:
            allow_private = True
        elif http_config and http_config.allow_private_networks_for_host_allowlist:
            allow_private = True

    if scheme == "http":
        if allow_private:
            logger.warning(
                "allowing http url for explicit allowlist host",
                extra={"stage": "download", "original_url": url},
            )
        else:
            logger.warning(
                "upgrading http url to https",
                extra={"stage": "download", "original_url": url},
            )
            parsed = parsed._replace(scheme="https")
            scheme = "https"

    if scheme != "https" and not allow_private:
        raise PolicyError("Only HTTPS URLs are allowed for ontology downloads")

    port = parsed.port
    if port is None:
        port = 80 if scheme == "http" else 443

    host_port_allowances = allowed_host_ports.get(ascii_host, set())
    if port not in allowed_port_set and port not in host_port_allowances:
        raise PolicyError(f"Port {port} is not permitted for ontology downloads")

    if is_ip:
        address = ipaddress.ip_address(ascii_host)
        if not allow_private and (
            address.is_private or address.is_loopback or address.is_reserved or address.is_multicast
        ):
            raise ConfigError(f"Refusing to download from private address {host}")
        return urlunparse(parsed)

    try:
        infos = _cached_getaddrinfo(ascii_host)
    except socket.gaierror as exc:
        logger.warning(
            "dns resolution failed",
            extra={"stage": "download", "hostname": host, "error": str(exc)},
        )
        if http_config and getattr(http_config, "strict_dns", False):
            raise ConfigError(f"DNS resolution failed for {host}: {exc}") from exc
        return urlunparse(parsed)

    for info in infos:
        candidate_ip = ipaddress.ip_address(info[4][0])
        if not allow_private and (
            candidate_ip.is_private
            or candidate_ip.is_loopback
            or candidate_ip.is_reserved
            or candidate_ip.is_multicast
        ):
            raise ConfigError(f"Refusing to download from private address resolved for {host}")

    return urlunparse(parsed)


class SessionPool:
    """Lightweight pool that reuses requests sessions per (service, host)."""

    def __init__(self, max_per_key: int = 2) -> None:
        self._lock = threading.Lock()
        self._pool: Dict[Tuple[str, str], List[requests.Session]] = {}
        self._max_per_key = max_per_key

    def _normalize(self, service: Optional[str], host: Optional[str]) -> Tuple[str, str]:
        service_key = (service or "_").lower()
        host_key = (host or "default").lower()
        return service_key, host_key

    @contextmanager
    def lease(
        self,
        *,
        service: Optional[str],
        host: Optional[str],
        http_config: Optional["DownloadConfiguration"] = None,
    ) -> Iterator[requests.Session]:
        """Yield a session associated with ``service``/``host`` and return it to the pool."""

        key = self._normalize(service, host)
        factory: Optional[Callable[[], requests.Session]] = None
        if http_config is not None:
            getter = getattr(http_config, "get_session_factory", None)
            if callable(getter):
                candidate = getter()
                if candidate is not None and not callable(candidate):
                    raise TypeError("session_factory getter must return a callable or None")
                factory = candidate

        with self._lock:
            stack = self._pool.get(key)
            if stack:
                session = stack.pop()
                if not stack:
                    self._pool.pop(key, None)
            else:
                if factory is not None:
                    session = factory()
                    if session is None:
                        session = requests.Session()
                    elif not isinstance(session, requests.Session):
                        raise TypeError(
                            "session_factory must return a requests.Session instance or None"
                        )
                else:
                    session = requests.Session()
        try:
            yield session
        finally:
            with self._lock:
                stack = self._pool.setdefault(key, [])
                if len(stack) < self._max_per_key:
                    stack.append(session)
                else:
                    session.close()
                    if not stack:
                        self._pool.pop(key, None)

    def clear(self) -> None:
        """Close and forget all pooled sessions (testing helper)."""

        with self._lock:
            for stack in self._pool.values():
                for session in stack:
                    session.close()
            self._pool.clear()


def retry_with_backoff(
    func: Callable[[], T],
    *,
    retryable: Callable[[Exception], bool],
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    jitter: float = 0.5,
    callback: Optional[Callable[[int, Exception, float], None]] = None,
    retry_after: Optional[Callable[[Exception], Optional[float]]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Execute ``func`` with exponential backoff until it succeeds."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as exc:  # pragma: no cover - behaviour verified via callers
            if attempt >= max_attempts or not retryable(exc):
                raise
            delay = backoff_base * (2 ** (attempt - 1))
            if retry_after is not None:
                try:
                    hint = retry_after(exc)
                except Exception:  # pragma: no cover - defensive against callbacks
                    hint = None
                else:
                    if hint is not None:
                        delay = max(hint, 0.0)
            if jitter > 0:
                delay += random.uniform(0.0, jitter)
            if callback is not None:
                try:
                    callback(attempt, exc, delay)
                except Exception:  # pragma: no cover - defensive against callbacks
                    pass
            sleep(max(delay, 0.0))


def log_memory_usage(
    logger: logging.Logger,
    *,
    stage: str,
    event: str,
    validator: Optional[str] = None,
) -> None:
    """Emit debug-level memory usage snapshots when enabled."""
    is_enabled = getattr(logger, "isEnabledFor", None)
    if callable(is_enabled):
        enabled = is_enabled(logging.DEBUG)
    else:  # pragma: no cover - fallback for stub loggers
        enabled = False
    if not enabled:
        return
    if not psutil or _PROCESS is None:
        return
    try:
        memory_mb = _PROCESS.memory_info().rss / (1024**2)
    except (psutil.Error, OSError):  # pragma: no cover - defensive guard
        return
    extra: Dict[str, object] = {"stage": stage, "event": event, "memory_mb": round(memory_mb, 2)}
    if validator:
        extra["validator"] = validator
    logger.debug("memory usage", extra=extra)


@dataclass(slots=True)
class DownloadResult:
    """Result metadata for a completed download operation.

    Attributes:
        path: Final file path where the ontology document was stored.
        status: Download status (`fresh`, `updated`, or `cached`).
        sha256: SHA-256 checksum of the downloaded artifact.
        etag: HTTP ETag returned by the upstream server, when available.
        last_modified: Upstream last-modified header value if provided.
        content_type: Reported MIME type when available (HEAD or GET).
        content_length: Reported content length when available.

    Examples:
        >>> result = DownloadResult(Path("ontology.owl"), "fresh", "deadbeef", None, None, None, None)
        >>> result.status
        'fresh'
    """

    path: Path
    status: str
    sha256: str
    etag: Optional[str]
    last_modified: Optional[str]
    content_type: Optional[str]
    content_length: Optional[int]


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        seconds = float(candidate)
    except ValueError:
        try:
            retry_time = parsedate_to_datetime(candidate)
        except (TypeError, ValueError, IndexError):
            return None
        if retry_time is None:
            return None
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        seconds = (retry_time - now).total_seconds()
    return max(seconds, 0.0)


def _is_retryable_status(status_code: Optional[int]) -> bool:
    if status_code is None:
        return True
    if status_code >= 500:
        return True
    return status_code in _RETRYABLE_HTTP_STATUSES


def is_retryable_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` represents a retryable network failure."""

    if isinstance(exc, DownloadFailure):
        return exc.retryable
    if isinstance(
        exc,
        (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.SSLError,
        ),
    ):
        return True
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        return _is_retryable_status(status)
    if isinstance(exc, requests.RequestException):
        return True
    return False


SESSION_POOL = SessionPool()


@contextmanager
def request_with_redirect_audit(
    *,
    session: requests.Session,
    method: str,
    url: str,
    headers: Dict[str, str],
    timeout: float,
    stream: bool,
    http_config: DownloadConfiguration,
    assume_url_validated: bool = False,
) -> Iterator[requests.Response]:
    """Issue an HTTP request while validating every redirect target."""

    redirects = 0
    response: Optional[requests.Response] = None
    current_url = url
    last_validated_url: Optional[str] = None
    assume_validated_flag = assume_url_validated
    raw_limit = getattr(http_config, "max_redirects", 5)
    try:
        max_redirects = int(raw_limit)
    except (TypeError, ValueError):
        max_redirects = 5
    if max_redirects < 0:
        max_redirects = 0

    try:
        while True:
            if assume_validated_flag:
                secure_url = current_url
            else:
                secure_url = validate_url_security(current_url, http_config)
            assume_validated_flag = False
            last_validated_url = secure_url
            try:
                response = session.request(
                    method,
                    secure_url,
                    headers=headers,
                    timeout=timeout,
                    stream=stream,
                    allow_redirects=False,
                )
            except requests.RequestException:
                raise

            if response.is_redirect:
                redirects += 1
                if redirects > max_redirects:
                    response.close()
                    raise PolicyError("Too many redirects during download")
                location = response.headers.get("Location")
                if not location:
                    response.close()
                    raise PolicyError("Redirect response missing Location header")
                next_url = urljoin(secure_url, location)
                try:
                    current_url = validate_url_security(next_url, http_config)
                    last_validated_url = current_url
                    assume_validated_flag = True
                finally:
                    response.close()
                    response = None
                continue

            try:
                if response.url != last_validated_url:
                    last_validated_url = validate_url_security(response.url, http_config)
            except Exception:
                response.close()
                raise

            setattr(response, "validated_url", last_validated_url)
            yield response
            return
    finally:
        if response is not None:
            response.close()


class StreamingDownloader(pooch.HTTPDownloader):
    """Custom downloader supporting HEAD validation, conditional requests, resume, and caching.

    The downloader shares a :mod:`requests` session so it can issue a HEAD probe
    prior to streaming content, verifies Content-Type and Content-Length against
    expectations, and persists ETag/Last-Modified headers for cache-friendly
    revalidation.

    Attributes:
        destination: Final location where the ontology will be stored.
        custom_headers: HTTP headers supplied by the resolver.
        http_config: Download configuration governing retries and limits.
        previous_manifest: Manifest from prior runs used for caching.
        logger: Logger used for structured telemetry.
        status: Final download status (`fresh`, `updated`, or `cached`).
        response_etag: ETag returned by the upstream server, if present.
        response_last_modified: Last-modified timestamp provided by the server.
        expected_media_type: MIME type provided by the resolver for validation.
        streamed_digests: Mapping of hash algorithm names to hex digests computed during streaming.

    Examples:
        >>> from pathlib import Path
        >>> from DocsToKG.OntologyDownload.settings import DownloadConfiguration
        >>> downloader = StreamingDownloader(
        ...     destination=Path("/tmp/ontology.owl"),
        ...     headers={},
        ...     http_config=DownloadConfiguration(),
        ...     previous_manifest={},
        ...     logger=logging.getLogger("test"),
        ... )
        >>> downloader.status
        'fresh'
    """

    def __init__(
        self,
        *,
        destination: Path,
        headers: Dict[str, str],
        http_config: DownloadConfiguration,
        previous_manifest: Optional[Dict[str, object]],
        logger: logging.Logger,
        expected_media_type: Optional[str] = None,
        service: Optional[str] = None,
        origin_host: Optional[str] = None,
        bucket: Optional[TokenBucket] = None,
        hash_algorithms: Optional[Iterable[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        url_already_validated: bool = False,
    ) -> None:
        super().__init__(headers={}, progressbar=False, timeout=http_config.timeout_sec)
        self.destination = destination
        self.custom_headers = dict(headers)
        self.http_config = http_config
        self.previous_manifest = previous_manifest or {}
        self.logger = logger
        self.status = "fresh"
        self.response_etag: Optional[str] = None
        self.response_last_modified: Optional[str] = None
        self.expected_media_type = expected_media_type
        self.head_content_type: Optional[str] = None
        self.head_content_length: Optional[int] = None
        self.response_content_type: Optional[str] = None
        self.response_content_length: Optional[int] = None
        self.service = service
        self.origin_host = origin_host
        self.invoked = False
        self.bucket = bucket
        requested_algorithms: List[str] = ["sha256"]
        if hash_algorithms:
            for algorithm in hash_algorithms:
                normalized = algorithm.strip().lower()
                if normalized and normalized not in requested_algorithms:
                    requested_algorithms.append(normalized)
        self._requested_hash_algorithms: Tuple[str, ...] = tuple(requested_algorithms)
        self._unsupported_hash_algorithms: Set[str] = set()
        self._hashers: Dict[str, object] = {}
        self.streamed_digests: Dict[str, str] = {}
        self.cancellation_token = cancellation_token
        self._reset_hashers()
        self._reuse_head_token = False
        self._assume_url_validated = url_already_validated

    def _reset_hashers(self) -> None:
        """Initialise hashlib objects for all supported algorithms."""

        self.streamed_digests = {}
        self._hashers = {}
        for algorithm in self._requested_hash_algorithms:
            try:
                self._hashers[algorithm] = hashlib.new(algorithm)
            except ValueError:
                if algorithm not in self._unsupported_hash_algorithms:
                    self.logger.warning(
                        "unsupported checksum algorithm for streaming",
                        extra={
                            "stage": "download",
                            "algorithm": algorithm,
                        },
                    )
                    self._unsupported_hash_algorithms.add(algorithm)

    def _seed_hashers_from_file(self, path: Path) -> None:
        """Update hashers with the bytes already present on disk."""

        if not self._hashers:
            return
        if not path.exists():
            return
        try:
            with path.open("rb") as existing:
                for chunk in iter(lambda: existing.read(1 << 20), b""):
                    if not chunk:
                        continue
                    for hasher in self._hashers.values():
                        hasher.update(chunk)
        except OSError as exc:
            self.logger.debug(
                "failed to seed hashers from partial file",
                extra={"stage": "download", "error": str(exc)},
            )

    @contextmanager
    def _request_with_redirect_audit(
        self,
        *,
        session: requests.Session,
        method: str,
        url: str,
        headers: Dict[str, str],
        timeout: float,
        stream: bool,
    ) -> Iterator[requests.Response]:
        """Issue an HTTP request while validating every redirect target."""

        with request_with_redirect_audit(
            session=session,
            method=method,
            url=url,
            headers=headers,
            timeout=timeout,
            stream=stream,
            http_config=self.http_config,
            assume_url_validated=self._assume_url_validated,
        ) as response:
            yield response

    def _sleep_with_cancellation(
        self,
        delay: float,
        *,
        remaining_budget: Optional[Callable[[], float]] = None,
        timeout_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Sleep in short increments while monitoring cancellation state."""

        if delay <= 0:
            return

        callback_invoked = False
        deadline = time.monotonic() + delay
        poll_interval = 0.5

        while True:
            if self.cancellation_token and self.cancellation_token.is_cancelled():
                self.logger.info(
                    "download cancelled during HEAD retry backoff",
                    extra={
                        "stage": "download",
                        "status": "cancelled",
                    },
                )
                raise DownloadFailure(
                    "Download was cancelled during HEAD retry backoff",
                    retryable=False,
                )

            if remaining_budget is not None:
                budget_remaining = remaining_budget()
                if (
                    timeout_callback is not None
                    and budget_remaining <= 0
                    and not callback_invoked
                ):
                    timeout_callback()
                    callback_invoked = True

            remaining_sleep = deadline - time.monotonic()
            if remaining_sleep <= 0:
                break

            time.sleep(min(poll_interval, remaining_sleep))

    def _preliminary_head_check(
        self,
        url: str,
        session: requests.Session,
        *,
        headers: Optional[Mapping[str, str]] = None,
        token_consumed: bool = False,
        remaining_budget: Optional[Callable[[], float]] = None,
        timeout_callback: Optional[Callable[[], None]] = None,
    ) -> tuple[Optional[str], Optional[int]]:
        """Probe the origin with HEAD to audit media type and size before downloading.

        The HEAD probe allows the pipeline to abort before streaming large
        payloads that exceed configured limits and to log early warnings for
        mismatched Content-Type headers reported by the origin.

        Args:
            url: Fully qualified download URL resolved by the planner.
            session: Prepared requests session used for outbound calls.
            headers: Headers to include with the HEAD probe. When omitted the
                downloader will send the polite header set merged with any
                resolver-supplied headers.
            token_consumed: Indicates whether the caller already consumed a
                rate-limit token prior to invoking the HEAD request.
            remaining_budget: Optional callable returning the remaining time
                budget (in seconds) before the download timeout is reached.
            timeout_callback: Optional callable invoked when the requested
                backoff would exhaust the remaining timeout budget.

        Returns:
            Tuple ``(content_type, content_length)`` extracted from response
            headers. Each element is ``None`` when the origin omits it.

        Raises:
            PolicyError: Propagates download policy errors encountered during the HEAD request.
            DownloadFailure: Raised when the timeout budget is exhausted prior to completing the HEAD request.
        """

        # Check for cancellation before HEAD request
        if self.cancellation_token and self.cancellation_token.is_cancelled():
            self.logger.info(
                "download cancelled before HEAD request",
                extra={
                    "stage": "download",
                    "status": "cancelled",
                },
            )
            raise DownloadFailure(
                "Download was cancelled before HEAD request",
                retryable=False,
            )

        self._reuse_head_token = False
        consumed_here = False
        if self.bucket is not None and not token_consumed:
            self.bucket.consume()
            token_consumed = True
            consumed_here = True

        if headers is None:
            request_headers = self.http_config.polite_http_headers(
                correlation_id=_extract_correlation_id(self.logger)
            )
            request_headers.update(self.custom_headers)
        else:
            request_headers = dict(headers)

        if remaining_budget is not None and timeout_callback is not None:
            remaining = remaining_budget()
            if remaining <= 0:
                timeout_callback()

        try:
            with self._request_with_redirect_audit(
                session=session,
                method="HEAD",
                url=url,
                headers=request_headers,
                timeout=self.http_config.timeout_sec,
                stream=False,
            ) as response:
                if response.status_code >= 400:
                    retry_after_header = response.headers.get("Retry-After")
                    retry_after_delay = _parse_retry_after(retry_after_header)
                    if retry_after_delay is not None:
                        self.logger.warning(
                            "head retry-after",
                            extra={
                                "stage": "download",
                                "method": "HEAD",
                                "status_code": response.status_code,
                                "retry_after_sec": round(retry_after_delay, 2),
                                "service": self.service,
                                "host": self.origin_host,
                            },
                        )
                        apply_retry_after(
                            http_config=self.http_config,
                            service=self.service,
                            host=self.origin_host,
                            delay=retry_after_delay,
                        )
                        if retry_after_delay > 0:
                            if remaining_budget is not None and timeout_callback is not None:
                                remaining = remaining_budget()
                                if retry_after_delay >= max(remaining, 0.0):
                                    timeout_callback()
                            if self.bucket is not None and token_consumed:
                                self._reuse_head_token = True
                            self._sleep_with_cancellation(
                                retry_after_delay,
                                remaining_budget=remaining_budget,
                                timeout_callback=timeout_callback,
                            )
                    self.logger.debug(
                        "HEAD request failed, proceeding with GET",
                        extra={
                            "stage": "download",
                            "method": "HEAD",
                            "status_code": response.status_code,
                            "url": url,
                            "headers": self.custom_headers,
                            "token_consumed": token_consumed,
                            "consumed_here": consumed_here,
                        },
                    )
                    return None, None

                content_type = response.headers.get("Content-Type")
                content_length_header = response.headers.get("Content-Length")
                content_length = None
                if content_length_header:
                    try:
                        content_length = int(content_length_header)
                    except (TypeError, ValueError):
                        # Invalid Content-Length header, ignore it
                        content_length = None

                return content_type, content_length
        except requests.RequestException as exc:
            self.logger.debug(
                "HEAD request exception, proceeding with GET",
                extra={
                    "stage": "download",
                    "error": str(exc),
                    "url": url,
                    "headers": request_headers,
                },
            )
            return None, None

    def _validate_media_type(
        self,
        actual_content_type: Optional[str],
        expected_media_type: Optional[str],
        url: str,
    ) -> None:
        """Validate that the received ``Content-Type`` header is acceptable, tolerating aliases.

        RDF endpoints often return generic XML or Turtle aliases, so the
        validator accepts a small set of known MIME variants while still
        surfacing actionable warnings for unexpected types.

        Args:
            actual_content_type: Raw header value reported by the origin server.
            expected_media_type: MIME type declared by resolver metadata.
            url: Download URL logged when mismatches occur.

        Returns:
            None
        """

        if not self.http_config.validate_media_type:
            return
        if not expected_media_type:
            return
        if not actual_content_type:
            self.logger.warning(
                "server did not provide Content-Type header",
                extra={
                    "stage": "download",
                    "expected_media_type": expected_media_type,
                    "url": url,
                },
            )
            return

        actual_mime = actual_content_type.split(";")[0].strip().lower()
        expected_mime = expected_media_type.strip().lower()
        if actual_mime == expected_mime:
            return

        expected_label = RDF_MIME_FORMAT_LABELS.get(expected_mime)
        actual_label = RDF_MIME_FORMAT_LABELS.get(actual_mime)
        if expected_label and actual_label:
            if expected_label == actual_label:
                if actual_mime != expected_mime:
                    self.logger.info(
                        "acceptable media type variation",
                        extra={
                            "stage": "download",
                            "expected": expected_mime,
                            "actual": actual_mime,
                            "label": expected_label,
                            "url": url,
                        },
                    )
                return
            variation_hint = {
                "stage": "download",
                "expected_media_type": expected_mime,
                "expected_label": expected_label,
                "actual_media_type": actual_mime,
                "actual_label": actual_label,
                "url": url,
            }
            self.logger.warning(
                "media type mismatch detected",
                extra={
                    **variation_hint,
                    "action": "proceeding with download",
                    "override_hint": "Set defaults.http.validate_media_type: false to disable validation",
                },
            )
            return

        self.logger.warning(
            "media type mismatch detected",
            extra={
                "stage": "download",
                "expected_media_type": expected_mime,
                "actual_media_type": actual_mime,
                "url": url,
                "action": "proceeding with download",
                "override_hint": "Set defaults.http.validate_media_type: false to disable validation",
            },
        )

    def __call__(self, url: str, output_file: str, pooch_logger: logging.Logger) -> None:  # type: ignore[override]
        """Stream ontology content to disk while enforcing download policies.

        Args:
            url: Secure download URL resolved by the planner.
            output_file: Temporary filename managed by pooch during download.
            pooch_logger: Logger instance supplied by pooch (unused).

        Raises:
            PolicyError: If download policies are violated (e.g., invalid URLs or disallowed MIME types).
            OntologyDownloadError: If filesystem errors occur.
            requests.HTTPError: Propagated when HTTP status codes indicate failure.

        Returns:
            None
        """
        self.invoked = True

        manifest_headers: Dict[str, str] = {}
        if self.previous_manifest:
            etag_value = self.previous_manifest.get("etag")
            if isinstance(etag_value, str) and etag_value.strip():
                manifest_headers["If-None-Match"] = etag_value
            last_modified_value = self.previous_manifest.get("last_modified")
            if isinstance(last_modified_value, str) and last_modified_value.strip():
                manifest_headers["If-Modified-Since"] = last_modified_value
        base_headers = {**self.custom_headers, **manifest_headers}
        part_path = Path(output_file + ".part")
        destination_part_path = Path(str(self.destination) + ".part")
        if not part_path.exists() and destination_part_path.exists():
            part_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination_part_path, part_path)

        self.head_content_type = None
        self.head_content_length = None
        self.response_content_type = None
        self.response_content_length = None

        overall_start = time.monotonic()
        timeout_limit = float(self.http_config.download_timeout_sec)

        with SESSION_POOL.lease(
            service=self.service,
            host=self.origin_host,
            http_config=self.http_config,
        ) as session:
            head_token_consumed = False
            if self.bucket is not None:
                self.bucket.consume()
                head_token_consumed = True

            def _clear_partial_files() -> None:
                for candidate in (part_path, destination_part_path):
                    try:
                        candidate.unlink(missing_ok=True)
                    except OSError:
                        pass

            def _raise_timeout(elapsed: float) -> None:
                _clear_partial_files()
                timeout_sec = timeout_limit
                self.logger.error(
                    "download timeout",
                    extra={
                        "stage": "download",
                        "error": "timeout",
                        "elapsed_sec": round(elapsed, 2),
                        "timeout_sec": timeout_sec,
                        "service": self.service,
                        "host": self.origin_host,
                    },
                )
                raise DownloadFailure(
                    (
                        f"Download exceeded timeout of {timeout_sec:.2f} seconds "
                        f"(elapsed {elapsed:.2f} seconds)"
                    ),
                    retryable=False,
                )

            def _remaining_budget() -> float:
                return timeout_limit - (time.monotonic() - overall_start)

            def _fail_for_timeout() -> None:
                _raise_timeout(time.monotonic() - overall_start)

            head_content_type, head_content_length = self._preliminary_head_check(
                url,
                session,
                headers=base_headers,
                token_consumed=head_token_consumed,
                remaining_budget=_remaining_budget,
                timeout_callback=_fail_for_timeout,
            )
            self.head_content_type = head_content_type
            self.head_content_length = head_content_length
            if head_content_type:
                self._validate_media_type(head_content_type, self.expected_media_type, url)

            resume_position = part_path.stat().st_size if part_path.exists() else 0

            def _enforce_timeout() -> None:
                elapsed = time.monotonic() - overall_start
                if elapsed > timeout_limit:
                    _raise_timeout(elapsed)

            def _stream_once() -> str:
                nonlocal resume_position

                # Check for cancellation before starting the download
                if self.cancellation_token and self.cancellation_token.is_cancelled():
                    self.logger.info(
                        "download cancelled before request",
                        extra={
                            "stage": "download",
                            "status": "cancelled",
                        },
                    )
                    raise DownloadFailure(
                        "Download was cancelled before request",
                        retryable=False,
                    )

                self._reset_hashers()
                resume_position = part_path.stat().st_size if part_path.exists() else 0
                original_resume_position = resume_position
                want_range = original_resume_position > 0
                request_headers = dict(base_headers)
                if want_range:
                    request_headers["Range"] = f"bytes={original_resume_position}-"

                if self.bucket is not None:
                    if self._reuse_head_token:
                        self.logger.debug(
                            "reusing head token after retry-after",
                            extra={
                                "stage": "download",
                                "service": self.service,
                                "host": self.origin_host,
                            },
                        )
                        self._reuse_head_token = False
                    else:
                        self.bucket.consume()

                request_timeout = self.http_config.timeout_sec

                # Check for cancellation immediately before HTTP request
                if self.cancellation_token and self.cancellation_token.is_cancelled():
                    self.logger.info(
                        "download cancelled before GET request",
                        extra={
                            "stage": "download",
                            "status": "cancelled",
                        },
                    )
                    raise DownloadFailure(
                        "Download was cancelled before GET request",
                        retryable=False,
                    )

                with self._request_with_redirect_audit(
                    session=session,
                    method="GET",
                    url=url,
                    headers=request_headers,
                    timeout=request_timeout,
                    stream=True,
                ) as response:
                    if response.status_code == 304 and Path(self.destination).exists():
                        self.status = "cached"
                        self.response_etag = response.headers.get(
                            "ETag"
                        ) or self.previous_manifest.get("etag")
                        self.response_last_modified = response.headers.get(
                            "Last-Modified"
                        ) or self.previous_manifest.get("last_modified")
                        manifest_type = self.previous_manifest.get("content_type")
                        self.response_content_type = (
                            manifest_type if isinstance(manifest_type, str) else None
                        )
                        manifest_length = self.previous_manifest.get("content_length")
                        try:
                            self.response_content_length = (
                                int(manifest_length) if manifest_length is not None else None
                            )
                        except (TypeError, ValueError):
                            self.response_content_length = None
                        part_path.unlink(missing_ok=True)
                        return "cached"

                    if response.status_code in {429, 503}:
                        retry_after_header = response.headers.get("Retry-After")
                        retry_after_delay = _parse_retry_after(retry_after_header)
                        if retry_after_delay is not None:
                            self.logger.warning(
                                "download retry-after",
                                extra={
                                    "stage": "download",
                                    "status_code": response.status_code,
                                    "retry_after_sec": round(retry_after_delay, 2),
                                    "service": self.service,
                                    "host": self.origin_host,
                                },
                            )
                            apply_retry_after(
                                http_config=self.http_config,
                                service=self.service,
                                host=self.origin_host,
                                delay=retry_after_delay,
                            )
                        http_error = requests.HTTPError(
                            f"HTTP error {response.status_code}", response=response
                        )
                        setattr(http_error, "_retry_after_delay", retry_after_delay)
                        response.close()
                        raise http_error

                    if response.status_code == 416:
                        self.logger.warning(
                            "range request rejected; retrying without resume",
                            extra={
                                "stage": "download",
                                "status_code": response.status_code,
                                "resume_position": original_resume_position,
                                "service": self.service,
                                "host": self.origin_host,
                            },
                        )
                        _clear_partial_files()
                        resume_position = 0
                        want_range = False
                        raise requests.HTTPError(
                            f"HTTP error {response.status_code}", response=response
                        )

                    range_honored = response.status_code == 206
                    if want_range and range_honored:
                        content_range = response.headers.get("Content-Range")
                        reported_offset: Optional[int] = None
                        if content_range and content_range.startswith("bytes "):
                            try:
                                reported_offset = int(content_range.split()[1].split("-")[0])
                            except (IndexError, ValueError):
                                reported_offset = None
                        if (
                            reported_offset is not None
                            and reported_offset != original_resume_position
                        ):
                            self.logger.warning(
                                "range resume misaligned; restarting from beginning",
                                extra={
                                    "stage": "download",
                                    "expected_offset": original_resume_position,
                                    "reported_offset": reported_offset,
                                    "status_code": response.status_code,
                                },
                            )
                            _clear_partial_files()
                            range_honored = False
                            resume_position = 0
                            want_range = False
                    if want_range and not range_honored:
                        self.logger.warning(
                            "range resume not honored; restarting from beginning",
                            extra={
                                "stage": "download",
                                "status_code": response.status_code,
                                "resume_position": original_resume_position,
                            },
                        )
                        _clear_partial_files()
                        range_honored = False
                        resume_position = 0
                        want_range = False
                    elif range_honored:
                        resume_position = original_resume_position
                    else:
                        resume_position = 0

                    if range_honored:
                        self.status = "updated"
                        if resume_position > 0:
                            self._seed_hashers_from_file(part_path)
                    response.raise_for_status()

                    self._validate_media_type(
                        response.headers.get("Content-Type"),
                        self.expected_media_type,
                        url,
                    )
                    self.response_content_type = response.headers.get("Content-Type")
                    length_header = response.headers.get("Content-Length")
                    total_bytes: Optional[int] = None
                    next_progress: Optional[float] = 0.1
                    parsed_length: Optional[int] = None
                    if length_header:
                        try:
                            parsed_length = int(length_header)
                        except ValueError:
                            parsed_length = None
                    self.response_content_length = parsed_length
                    if parsed_length is not None:
                        total_bytes = parsed_length
                    if total_bytes:
                        completed_fraction = resume_position / total_bytes
                        if completed_fraction >= 1:
                            next_progress = None
                        else:
                            next_progress = ((int(completed_fraction * 10)) + 1) / 10
                    self.response_etag = response.headers.get("ETag")
                    self.response_last_modified = response.headers.get("Last-Modified")
                    mode = "ab" if range_honored else "wb"
                    bytes_downloaded = resume_position
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with part_path.open(mode) as fh:
                            for chunk in response.iter_content(chunk_size=1 << 20):
                                _enforce_timeout()
                                if not chunk:
                                    continue

                                # Check for cancellation before processing chunk
                                if (
                                    self.cancellation_token
                                    and self.cancellation_token.is_cancelled()
                                ):
                                    part_path.unlink(missing_ok=True)
                                    self.logger.info(
                                        "download cancelled",
                                        extra={
                                            "stage": "download",
                                            "status": "cancelled",
                                            "bytes_downloaded": bytes_downloaded,
                                        },
                                    )
                                    raise DownloadFailure(
                                        "Download was cancelled",
                                        retryable=False,
                                    )

                                fh.write(chunk)
                                for hasher in self._hashers.values():
                                    hasher.update(chunk)
                                bytes_downloaded += len(chunk)
                                if total_bytes and next_progress:
                                    progress = bytes_downloaded / total_bytes
                                    while next_progress and progress >= next_progress:
                                        self.logger.info(
                                            "download progress",
                                            extra={
                                                "stage": "download",
                                                "status": "in-progress",
                                                "progress": {
                                                    "percent": round(min(progress, 1.0) * 100, 1)
                                                },
                                            },
                                        )
                                        next_progress += 0.1
                                        if next_progress > 1:
                                            next_progress = None
                                            break
                    except OSError as exc:
                        part_path.unlink(missing_ok=True)
                        self.logger.error(
                            "filesystem error during download",
                            extra={"stage": "download", "error": str(exc)},
                        )
                        if "No space left" in str(exc):
                            raise OntologyDownloadError(
                                "No space left on device while writing download"
                            ) from exc
                        raise OntologyDownloadError(f"Failed to write download: {exc}") from exc

                    if self._hashers:
                        self.streamed_digests = {
                            algorithm: hasher.hexdigest()
                            for algorithm, hasher in self._hashers.items()
                        }
                    return "success"

            def _stream_once_with_timeout() -> str:
                _enforce_timeout()
                result = _stream_once()
                _enforce_timeout()
                return result

            def _retry_after_hint(exc: BaseException) -> Optional[float]:
                delay = getattr(exc, "_retry_after_delay", None)
                if delay is not None:
                    return delay
                response = getattr(exc, "response", None)
                if response is not None:
                    return _parse_retry_after(response.headers.get("Retry-After"))
                return None

            def _on_retry(attempt_number: int, exc: BaseException, delay: float) -> None:
                self.logger.warning(
                    "download retry",
                    extra={
                        "stage": "download",
                        "attempt": attempt_number,
                        "sleep_sec": round(delay, 2),
                        "error": str(exc),
                    },
                )

            result_state = retry_with_backoff(
                _stream_once_with_timeout,
                retryable=is_retryable_error,
                max_attempts=max(1, self.http_config.max_retries),
                backoff_base=self.http_config.backoff_factor,
                jitter=self.http_config.backoff_factor,
                callback=_on_retry,
                retry_after=_retry_after_hint,
            )

        if result_state == "cached":
            destination_part_path.unlink(missing_ok=True)
            return

        part_path.replace(Path(output_file))
        destination_part_path.unlink(missing_ok=True)


def _extract_correlation_id(logger: logging.Logger) -> Optional[str]:
    extra = getattr(logger, "extra", None)
    if isinstance(extra, dict):
        value = extra.get("correlation_id")
        if isinstance(value, str):
            return value
    return None


def download_stream(
    *,
    url: str,
    destination: Path,
    headers: Dict[str, str],
    previous_manifest: Optional[Dict[str, object]],
    http_config: DownloadConfiguration,
    cache_dir: Path,
    logger: logging.Logger,
    expected_media_type: Optional[str] = None,
    service: Optional[str] = None,
    expected_hash: Optional[str] = None,
    cancellation_token: Optional[CancellationToken] = None,
    url_already_validated: bool = False,
) -> DownloadResult:
    """Download ontology content with HEAD validation, rate limiting, caching, retries, and hash checks.

    Args:
        url: URL of the ontology document to download.
        destination: Target file path for the downloaded content.
        headers: HTTP headers forwarded to the download request.
        previous_manifest: Manifest metadata from a prior run, used for caching.
        http_config: Download configuration containing timeouts, limits, and rate controls.
        cache_dir: Directory where intermediary cached files are stored.
        logger: Logger adapter for structured download telemetry.
        expected_media_type: Expected Content-Type for validation, if known.
        service: Logical service identifier for per-service rate limiting.
        expected_hash: Optional ``<algorithm>:<hex>`` string enforcing a known hash.
        cancellation_token: Optional token for cooperative cancellation.
        url_already_validated: When ``True``, assumes *url* has already passed
            :func:`validate_url_security` checks and skips redundant
            validations.

    Returns:
        DownloadResult describing the final artifact and metadata.

    Raises:
        PolicyError: If policy validation fails or limits are exceeded.
        OntologyDownloadError: If retryable download mechanisms exhaust or IO fails.
    """
    secure_url = url if url_already_validated else validate_url_security(url, http_config)
    parsed = urlparse(secure_url)
    host = parsed.hostname
    bucket = get_bucket(http_config=http_config, host=host, service=service)

    log_memory_usage(logger, stage="download", event="before")
    polite_headers = http_config.polite_http_headers(correlation_id=_extract_correlation_id(logger))
    merged_headers: Dict[str, str] = dict(polite_headers)
    merged_headers.update({str(k): str(v) for k, v in headers.items()})

    def _resolved_content_metadata(
        current_downloader: StreamingDownloader,
        manifest: Optional[Dict[str, object]],
    ) -> tuple[Optional[str], Optional[int]]:
        content_type = (
            current_downloader.response_content_type or current_downloader.head_content_type
        )
        if content_type is None and manifest:
            manifest_type = manifest.get("content_type")
            if isinstance(manifest_type, str):
                content_type = manifest_type

        content_length = current_downloader.response_content_length
        if content_length is None and current_downloader.head_content_length is not None:
            content_length = current_downloader.head_content_length
        if content_length is None and manifest:
            manifest_length = manifest.get("content_length")
            try:
                content_length = int(manifest_length) if manifest_length is not None else None
            except (TypeError, ValueError):
                content_length = None
        return content_type, content_length

    expected_algorithm: Optional[str] = None
    expected_digest: Optional[str] = None
    pooch_known_hash: Optional[str] = None
    if expected_hash:
        parts = expected_hash.split(":", 1)
        if len(parts) == 2:
            candidate_algorithm = parts[0].strip().lower()
            candidate_digest = parts[1].strip().lower()
            if candidate_algorithm and candidate_digest:
                expected_algorithm = candidate_algorithm
                expected_digest = candidate_digest
                if candidate_algorithm in {"md5", "sha256", "sha512"}:
                    pooch_known_hash = f"{candidate_algorithm}:{candidate_digest}"
        else:
            logger.warning(
                "expected checksum malformed",
                extra={"stage": "download", "checksum": expected_hash, "url": secure_url},
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(destination.name)
    url_hash = hashlib.sha256(secure_url.encode("utf-8")).hexdigest()[:12]
    cache_key = f"{url_hash}_{safe_name}"

    def _verify_expected_checksum(
        digests: Dict[str, str],
        *,
        artifact_path: Path,
        cache_path: Optional[Path],
    ) -> None:
        if not expected_algorithm or not expected_digest:
            return
        try:
            actual: Optional[str]
            digest_value = digests.get(expected_algorithm)
            if digest_value is not None:
                actual = digest_value.lower()
            else:
                actual = _compute_file_hash(artifact_path, expected_algorithm).lower()
        except ValueError:
            logger.warning(
                "unsupported checksum algorithm",
                extra={
                    "stage": "download",
                    "algorithm": expected_algorithm,
                    "url": secure_url,
                },
            )
            return
        if actual != expected_digest:
            logger.error(
                "checksum mismatch detected",
                extra={
                    "stage": "download",
                    "expected": f"{expected_algorithm}:{expected_digest}",
                    "actual": actual,
                    "url": secure_url,
                },
            )
            artifact_path.unlink(missing_ok=True)
            if cache_path is not None and cache_path != artifact_path:
                cache_path.unlink(missing_ok=True)
            raise DownloadFailure(
                f"Checksum mismatch for {secure_url}",
                retryable=False,
            )

    def _resolve_digests(
        *,
        current_downloader: StreamingDownloader,
        manifest: Optional[Dict[str, object]],
        artifact_path: Path,
    ) -> Dict[str, str]:
        digests = {
            algorithm: value.lower()
            for algorithm, value in current_downloader.streamed_digests.items()
        }
        if not digests and manifest:
            manifest_sha = manifest.get("sha256") if manifest else None
            if isinstance(manifest_sha, str) and manifest_sha:
                digests.setdefault("sha256", manifest_sha.lower())
        if "sha256" not in digests:
            try:
                digests["sha256"] = _compute_file_hash(artifact_path, "sha256").lower()
            except ValueError:
                pass
        return digests

    raw_attempts = getattr(http_config, "checksum_mismatch_retries", 3)
    try:
        max_checksum_attempts = int(raw_attempts)
    except (TypeError, ValueError):
        max_checksum_attempts = 3
    if max_checksum_attempts < 1:
        max_checksum_attempts = 1

    manifest_for_attempt = previous_manifest

    for attempt in range(1, max_checksum_attempts + 1):
        downloader = StreamingDownloader(
            destination=destination,
            headers=merged_headers,
            http_config=http_config,
            previous_manifest=manifest_for_attempt,
            logger=logger,
            expected_media_type=expected_media_type,
            service=service,
            origin_host=host,
            bucket=bucket,
            hash_algorithms=[expected_algorithm] if expected_algorithm else None,
            cancellation_token=cancellation_token,
            url_already_validated=True,
        )
        attempt_start = time.monotonic()
        try:
            cached_path = Path(
                pooch.retrieve(
                    secure_url,
                    path=cache_dir,
                    fname=cache_key,
                    known_hash=pooch_known_hash,
                    downloader=downloader,
                    progressbar=False,
                )
            )
        except requests.HTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            message = f"HTTP error while downloading {secure_url}: {exc}"
            retryable = _is_retryable_status(status_code)
            logger.error(
                "download request failed",
                extra={
                    "stage": "download",
                    "url": secure_url,
                    "error": str(exc),
                    "status_code": status_code,
                },
            )
            raise DownloadFailure(message, status_code=status_code, retryable=retryable) from exc
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.SSLError,
        ) as exc:
            logger.error(
                "download request failed",
                extra={"stage": "download", "url": secure_url, "error": str(exc)},
            )
            raise DownloadFailure(
                f"HTTP error while downloading {secure_url}: {exc}", retryable=True
            ) from exc
        except PolicyError:
            raise
        except DownloadFailure:
            raise
        except Exception as exc:  # pragma: no cover - defensive catch for pooch errors
            logger.error(
                "pooch download error",
                extra={"stage": "download", "url": secure_url, "error": str(exc)},
            )
            raise OntologyDownloadError(f"Download failed for {secure_url}: {exc}") from exc

        if not downloader.invoked and previous_manifest:
            downloader.status = "cached"
            etag_value = previous_manifest.get("etag")
            downloader.response_etag = etag_value if isinstance(etag_value, str) else None
            last_modified_value = previous_manifest.get("last_modified")
            downloader.response_last_modified = (
                last_modified_value if isinstance(last_modified_value, str) else None
            )

        if downloader.status == "cached":
            elapsed_cached = (time.monotonic() - attempt_start) * 1000
            logger.info(
                "cache hit",
                extra={
                    "stage": "download",
                    "status": "cached",
                    "elapsed_ms": round(elapsed_cached, 2),
                },
            )
            if destination.exists():
                artifact_path = destination
                cache_reference: Optional[Path] = cached_path if cached_path.exists() else None
            else:
                artifact_path, cache_reference = _materialize_cached_file(cached_path, destination)
            digest_map = _resolve_digests(
                current_downloader=downloader,
                manifest=manifest_for_attempt,
                artifact_path=artifact_path,
            )
            sha256 = digest_map.get("sha256")
            if sha256 is None:
                raise OntologyDownloadError(
                    f"failed to compute sha256 for cached artifact: {secure_url}"
                )
            _verify_expected_checksum(
                digest_map,
                artifact_path=artifact_path,
                cache_path=cache_reference,
            )
            log_memory_usage(logger, stage="download", event="after")
            content_type, content_length = _resolved_content_metadata(
                downloader, manifest_for_attempt
            )
            return DownloadResult(
                path=artifact_path,
                status="cached",
                sha256=sha256,
                etag=downloader.response_etag,
                last_modified=downloader.response_last_modified,
                content_type=content_type,
                content_length=content_length,
            )

        artifact_path, cache_reference = _materialize_cached_file(cached_path, destination)

        digest_map = _resolve_digests(
            current_downloader=downloader,
            manifest=manifest_for_attempt,
            artifact_path=artifact_path,
        )
        sha256 = digest_map.get("sha256")
        if sha256 is None:
            raise OntologyDownloadError(
                f"failed to compute sha256 for downloaded artifact: {secure_url}"
            )
        _verify_expected_checksum(
            digest_map,
            artifact_path=artifact_path,
            cache_path=cache_reference,
        )
        previous_sha256 = manifest_for_attempt.get("sha256") if manifest_for_attempt else None
        if previous_sha256 and previous_sha256 != sha256:
            logger.error(
                "sha256 mismatch detected",
                extra={
                    "stage": "download",
                    "expected": expected_hash,
                    "actual": sha256,
                    "url": secure_url,
                },
            )
            artifact_path.unlink(missing_ok=True)
            if cache_reference != artifact_path:
                cache_reference.unlink(missing_ok=True)
            if attempt >= max_checksum_attempts:
                raise OntologyDownloadError(
                    f"checksum mismatch after {max_checksum_attempts} attempts: {secure_url}"
                )
            manifest_for_attempt = None
            continue

        elapsed = (time.monotonic() - attempt_start) * 1000
        logger.info(
            "download complete",
            extra={
                "stage": "download",
                "status": downloader.status,
                "elapsed_ms": round(elapsed, 2),
                "sha256": sha256,
            },
        )
        log_memory_usage(logger, stage="download", event="after")
        content_type, content_length = _resolved_content_metadata(downloader, manifest_for_attempt)
        return DownloadResult(
            path=artifact_path,
            status=downloader.status,
            sha256=sha256,
            etag=downloader.response_etag,
            last_modified=downloader.response_last_modified,
            content_type=content_type,
            content_length=content_length,
        )

    raise OntologyDownloadError(
        f"checksum mismatch after {max_checksum_attempts} attempts: {secure_url}"
    )
