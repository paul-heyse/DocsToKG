# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.network",
#   "purpose": "Provide secure streaming downloads, DNS validation, and retry-aware HTTP helpers",
#   "sections": [
#     {"id": "infrastructure", "name": "Networking Infrastructure & Constants", "anchor": "INF", "kind": "infra"},
#     {"id": "dns", "name": "DNS & Host Validation", "anchor": "DNS", "kind": "helpers"},
#     {"id": "httpx", "name": "HTTPX Client & Rate Limiting", "anchor": "HTX", "kind": "api"},
#     {"id": "streaming", "name": "Streaming Downloader", "anchor": "STR", "kind": "api"},
#     {"id": "helpers", "name": "Download Helpers & Security Checks", "anchor": "HLP", "kind": "helpers"}
#   ]
# }
# === /NAVMAP ===

"""Networking utilities for ontology downloads.

This module manages resilient HTTP downloads: DNS caching, a shared HTTPX
client with RFC-9111 caching, range resume, provenance logging, retry-after
aware throttling, and security guards around redirects, content types, and
host allowlists. It provides the streaming helpers consumed by resolvers and
the planner when fetching ontology artefacts. The legacy `requests` session
pool has been retired—every code path now goes through the shared HTTPX +
Hishel transport defined in `DocsToKG.OntologyDownload.net`.
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import logging
import os
import random
import ssl
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

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_base

from ..cancellation import CancellationToken
from ..errors import ConfigError, DownloadFailure, OntologyDownloadError, PolicyError
from ..net import get_http_client
from ..settings import DownloadConfiguration
from .filesystem import _compute_file_hash, _materialize_cached_file, sanitize_filename
from .rate_limit import TokenBucket, apply_retry_after, get_bucket

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

_DOCUMENTATION_NETWORKS: Tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
)


def _is_documentation_address(address: IPAddress) -> bool:
    """Return ``True`` when *address* belongs to an IANA documentation prefix."""

    return any(address in network for network in _DOCUMENTATION_NETWORKS)


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

    allow_private_networks = False
    allow_plain_http = False
    if allowed_exact or allowed_suffixes:
        matched_exact = ascii_host in allowed_exact
        matched_suffix = any(
            ascii_host == suffix or ascii_host.endswith(f".{suffix}") for suffix in allowed_suffixes
        )
        if not (matched_exact or matched_suffix):
            raise PolicyError(f"Host {host} not in allowlist")

        if matched_exact and ascii_host in allowed_ip_literals:
            allow_private_networks = True
            allow_plain_http = True
        elif http_config and http_config.allow_private_networks_for_host_allowlist:
            allow_private_networks = True

        if http_config and http_config.allow_plain_http_for_host_allowlist:
            allow_plain_http = True

    if scheme == "http":
        if allow_plain_http:
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

    if scheme != "https" and not allow_plain_http:
        raise PolicyError("Only HTTPS URLs are allowed for ontology downloads")

    port = parsed.port
    if port is None:
        port = 80 if scheme == "http" else 443

    host_port_allowances = allowed_host_ports.get(ascii_host, set())
    if port not in allowed_port_set and port not in host_port_allowances:
        raise PolicyError(f"Port {port} is not permitted for ontology downloads")

    if is_ip:
        address = ipaddress.ip_address(ascii_host)
        is_doc_address = _is_documentation_address(address)
        if (
            not allow_private_networks
            and not is_doc_address
            and (address.is_private or address.is_loopback or address.is_multicast)
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
        is_doc_address = _is_documentation_address(candidate_ip)
        if (
            not allow_private_networks
            and not is_doc_address
            and (candidate_ip.is_private or candidate_ip.is_loopback or candidate_ip.is_multicast)
        ):
            raise ConfigError(f"Refusing to download from private address resolved for {host}")

    return urlunparse(parsed)


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

    class _BackoffWait(wait_base):
        def __call__(self, retry_state) -> float:  # type: ignore[override]
            outcome = retry_state.outcome
            exc: Optional[Exception]
            if outcome is None or not outcome.failed:
                exc = None
            else:
                exc = outcome.exception()

            attempt_number = max(retry_state.attempt_number, 1)
            delay = backoff_base * (2 ** (attempt_number - 1))

            if retry_after is not None and exc is not None:
                try:
                    hint = retry_after(exc)
                except Exception:  # pragma: no cover - defensive against callbacks
                    hint = None
                else:
                    if hint is not None:
                        delay = max(hint, 0.0)

            if jitter > 0:
                delay += random.uniform(0.0, jitter)

            delay = max(delay, 0.0)
            setattr(retry_state.retry_object, "_ontology_retry_delay", delay)
            setattr(retry_state.retry_object, "_ontology_retry_exception", exc)
            return delay

    def _before_sleep(retry_state) -> None:
        if callback is None:
            return

        delay = getattr(retry_state.retry_object, "_ontology_retry_delay", 0.0)
        exc = getattr(retry_state.retry_object, "_ontology_retry_exception", None)
        if exc is None and retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()

        if exc is None:
            return

        try:
            callback(retry_state.attempt_number, exc, delay)
        except Exception:  # pragma: no cover - defensive against callbacks
            pass

    retry_controller = Retrying(
        retry=retry_if_exception(lambda exc: retryable(exc)),
        wait=_BackoffWait(),
        stop=stop_after_attempt(max_attempts),
        sleep=sleep,
        reraise=True,
        before_sleep=_before_sleep,
    )

    return retry_controller(func)


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


def _extract_correlation_id(logger: logging.Logger) -> Optional[str]:
    """Return the correlation identifier stored on a logger adapter, when present."""

    extra = getattr(logger, "extra", None)
    if isinstance(extra, dict):
        value = extra.get("correlation_id")
        if isinstance(value, str) and value:
            return value
    return None


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
    if isinstance(exc, ssl.SSLError):
        return True
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.TransportError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", None)
        return _is_retryable_status(status)
    if isinstance(exc, httpx.RequestError):
        return True
    return False


@contextmanager
def request_with_redirect_audit(
    *,
    client: httpx.Client,
    method: str,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    stream: bool,
    http_config: DownloadConfiguration,
    assume_url_validated: bool = False,
    extensions: Optional[Mapping[str, object]] = None,
) -> Iterator[httpx.Response]:
    """Issue an HTTP request while validating every redirect target."""

    redirects = 0
    response: Optional[httpx.Response] = None
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
            secure_url = (
                current_url
                if assume_validated_flag
                else validate_url_security(current_url, http_config)
            )
            assume_validated_flag = False
            last_validated_url = secure_url

            request = client.build_request(
                method,
                secure_url,
                headers=headers,
                timeout=timeout,
            )
            if method.upper() == "HEAD":
                request.extensions["ontology_skip_status_check"] = True
            if extensions:
                request.extensions["ontology_headers"] = dict(extensions)

            try:
                response = client.send(request, stream=stream)
            except httpx.RequestError:
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
                response_url = str(response.url)
                if response_url != last_validated_url:
                    last_validated_url = validate_url_security(response_url, http_config)
            except Exception:
                response.close()
                raise

            setattr(response, "validated_url", last_validated_url)
            yield response
            return
    finally:
        if response is not None:
            response.close()


@dataclass(slots=True)
class _StreamOutcome:
    """Captured outcome metadata for a download attempt."""

    status: str
    cache_path: Path
    etag: Optional[str]
    last_modified: Optional[str]
    content_type: Optional[str]
    content_length: Optional[int]
    from_cache: bool


def _conditional_headers_from_manifest(
    manifest: Optional[Mapping[str, object]]
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if not manifest:
        return headers
    etag = manifest.get("etag") if isinstance(manifest, Mapping) else None
    if isinstance(etag, str) and etag.strip():
        headers["If-None-Match"] = etag.strip()
    last_modified = manifest.get("last_modified") if isinstance(manifest, Mapping) else None
    if isinstance(last_modified, str) and last_modified.strip():
        headers["If-Modified-Since"] = last_modified.strip()
    return headers


def _parse_expected_hash(expected_hash: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not expected_hash:
        return None, None
    parts = expected_hash.split(':', 1)
    if len(parts) != 2:
        return None, None
    algorithm = parts[0].strip().lower()
    digest = parts[1].strip().lower()
    if not algorithm or not digest:
        return None, None
    return algorithm, digest


def _initial_progress_state(
    *,
    resume_position: int,
    total_bytes: Optional[int],
    percent_step: float,
) -> Dict[str, object]:
    state: Dict[str, object] = {"last_bytes": resume_position, "next_percent": None}
    if total_bytes and total_bytes > 0 and percent_step > 0:
        progress = resume_position / total_bytes
        if progress < 1:
            next_step = ((int(progress / percent_step)) + 1) * percent_step
            state["next_percent"] = None if next_step >= 1 else next_step
    return state


def _log_stream_progress(
    *,
    logger: logging.Logger,
    bytes_downloaded: int,
    total_bytes: Optional[int],
    state: Dict[str, object],
    percent_step: float,
    bytes_threshold: int,
) -> None:
    if total_bytes and total_bytes > 0 and percent_step > 0:
        next_percent = state.get("next_percent")
        while isinstance(next_percent, (int, float)) and total_bytes:
            progress = bytes_downloaded / total_bytes
            if progress + 1e-9 < next_percent:
                break
            logger.info(
                "download progress",
                extra={
                    "stage": "download",
                    "status": "in-progress",
                    "event": "download_progress",
                    "progress": {
                        "percent": round(min(progress, 1.0) * 100, 1),
                        "bytes_downloaded": bytes_downloaded,
                        "total_bytes": total_bytes,
                    },
                },
            )
            next_percent += percent_step
            if next_percent >= 1:
                next_percent = None
        state["next_percent"] = next_percent
        return

    if bytes_threshold <= 0:
        return
    last_bytes = state.get("last_bytes", 0)
    if bytes_downloaded - last_bytes >= bytes_threshold:
        logger.info(
            "download progress",
            extra={
                "stage": "download",
                "status": "in-progress",
                "event": "download_progress",
                "progress": {
                    "bytes_downloaded": bytes_downloaded,
                },
            },
        )
        state["last_bytes"] = bytes_downloaded


def _apply_retry_after_from_response(
    *,
    response: httpx.Response,
    http_config: DownloadConfiguration,
    service: Optional[str],
    host: Optional[str],
) -> Optional[float]:
    retry_after_header = response.headers.get("Retry-After")
    retry_delay = _parse_retry_after(retry_after_header)
    if retry_delay is not None:
        apply_retry_after(
            http_config=http_config,
            service=service,
            host=host,
            delay=retry_delay,
        )
    return retry_delay


def _validate_media_type(
    *,
    actual_content_type: Optional[str],
    expected_media_type: Optional[str],
    http_config: DownloadConfiguration,
    logger: logging.Logger,
    url: str,
) -> None:
    if not http_config.validate_media_type:
        return
    if not expected_media_type:
        return
    if not actual_content_type:
        logger.warning(
            "server did not provide Content-Type header",
            extra={
                "stage": "download",
                "expected_media_type": expected_media_type,
                "url": url,
            },
        )
        return

    actual_mime = actual_content_type.split(';')[0].strip().lower()
    expected_mime = expected_media_type.strip().lower()
    if actual_mime == expected_mime:
        return

    expected_label = RDF_MIME_FORMAT_LABELS.get(expected_mime)
    actual_label = RDF_MIME_FORMAT_LABELS.get(actual_mime)
    if expected_label and actual_label and expected_label == actual_label:
        if actual_mime != expected_mime:
            logger.info(
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

    logger.warning(
        "media type mismatch detected",
        extra={
            "stage": "download",
            "expected_media_type": expected_mime,
            "actual_media_type": actual_mime,
            "url": url,
            "override_hint": "Set defaults.http.validate_media_type: false to disable validation",
        },
    )


def _total_bytes_from_response(response: httpx.Response, resume_position: int) -> Optional[int]:
    if response.status_code == 206:
        content_range = response.headers.get("Content-Range")
        if content_range and '/' in content_range:
            try:
                total = int(content_range.split('/')[-1])
                return total
            except (ValueError, IndexError):
                return None
        return None
    length_header = response.headers.get("Content-Length")
    if not length_header:
        return None
    try:
        return int(length_header)
    except (TypeError, ValueError):
        return None


def _stream_body_to_cache(
    *,
    response: httpx.Response,
    cache_path: Path,
    resume_position: int,
    http_config: DownloadConfiguration,
    cancellation_token: Optional[CancellationToken],
    logger: logging.Logger,
    progress_percent_step: float,
    progress_bytes_threshold: int,
) -> int:
    part_path = cache_path.with_suffix(cache_path.suffix + '.part')
    part_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'ab' if response.status_code == 206 and resume_position > 0 else 'wb'
    if mode == 'wb':
        part_path.unlink(missing_ok=True)
    total_bytes = _total_bytes_from_response(response, resume_position)
    state = _initial_progress_state(
        resume_position=resume_position,
        total_bytes=total_bytes,
        percent_step=progress_percent_step,
    )
    max_bytes = http_config.max_uncompressed_bytes()
    bytes_downloaded = resume_position
    try:
        with part_path.open(mode) as stream:
            for chunk in response.iter_bytes(1 << 20):
                if not chunk:
                    continue
                if cancellation_token and cancellation_token.is_cancelled():
                    raise DownloadFailure("Download was cancelled", retryable=False)
                stream.write(chunk)
                bytes_downloaded += len(chunk)
                if max_bytes and bytes_downloaded > max_bytes:
                    raise PolicyError(
                        f"Download exceeded size limit of {http_config.max_uncompressed_size_gb} GB"
                    )
                _log_stream_progress(
                    logger=logger,
                    bytes_downloaded=bytes_downloaded,
                    total_bytes=total_bytes,
                    state=state,
                    percent_step=progress_percent_step,
                    bytes_threshold=progress_bytes_threshold,
                )
    except DownloadFailure:
        part_path.unlink(missing_ok=True)
        raise
    except PolicyError:
        part_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        part_path.unlink(missing_ok=True)
        logger.error(
            "filesystem error during download",
            extra={"stage": "download", "error": str(exc)},
        )
        raise OntologyDownloadError(f"Failed to write download: {exc}") from exc

    try:
        os.replace(part_path, cache_path)
    except OSError as exc:
        part_path.unlink(missing_ok=True)
        logger.error(
            "filesystem error finalising download",
            extra={"stage": "download", "error": str(exc)},
        )
        raise OntologyDownloadError(f"Failed to finalise download: {exc}") from exc
    return bytes_downloaded


def _download_once(
    *,
    client: httpx.Client,
    url: str,
    cache_path: Path,
    headers: Mapping[str, str],
    http_config: DownloadConfiguration,
    bucket: Optional[TokenBucket],
    logger: logging.Logger,
    correlation_id: str,
    cancellation_token: Optional[CancellationToken],
    expected_media_type: Optional[str],
    progress_percent_step: float,
    progress_bytes_threshold: int,
    perform_head: bool,
    service: Optional[str],
    host: Optional[str],
) -> _StreamOutcome:
    if cancellation_token and cancellation_token.is_cancelled():
        raise DownloadFailure("Download was cancelled", retryable=False)

    resume_position = 0
    part_path = cache_path.with_suffix(cache_path.suffix + '.part')
    if part_path.exists():
        try:
            resume_position = part_path.stat().st_size
        except OSError:
            resume_position = 0
    elif cache_path.exists():
        resume_position = 0

    content_type_hint: Optional[str] = None
    content_length_hint: Optional[int] = None
    etag_hint: Optional[str] = None
    last_modified_hint: Optional[str] = None

    if perform_head:
        if bucket is not None:
            bucket.consume()
        extensions = {
            "config": http_config,
            "headers": headers,
            "correlation_id": correlation_id,
        }
        try:
            with request_with_redirect_audit(
                client=client,
                method="HEAD",
                url=url,
                headers=headers,
                timeout=http_config.timeout_sec,
                stream=False,
                http_config=http_config,
                assume_url_validated=True,
                extensions=extensions,
            ) as response:
                status_code = response.status_code
                if status_code in {429, 503}:
                    retry_delay = _apply_retry_after_from_response(
                        response=response,
                        http_config=http_config,
                        service=service,
                        host=host,
                    )
                    http_error = httpx.HTTPStatusError(
                        f"HTTP error {status_code}", request=response.request, response=response
                    )
                    if retry_delay is not None:
                        setattr(http_error, "_retry_after_delay", retry_delay)
                    raise http_error
                if status_code == 304:
                    return _StreamOutcome(
                        status="cached",
                        cache_path=cache_path,
                        etag=response.headers.get("ETag") or etag_hint,
                        last_modified=response.headers.get("Last-Modified") or last_modified_hint,
                        content_type=content_type_hint,
                        content_length=content_length_hint,
                        from_cache=True,
                    )
                response.raise_for_status()
                content_type_header = response.headers.get("Content-Type")
                content_length_header = response.headers.get("Content-Length")
                if content_type_header:
                    content_type_hint = content_type_header
                if content_length_header:
                    try:
                        content_length_hint = int(content_length_header)
                    except (TypeError, ValueError):
                        content_length_hint = None
                etag_hint = response.headers.get("ETag")
                last_modified_hint = response.headers.get("Last-Modified")
        except httpx.HTTPStatusError:
            raise
        except httpx.RequestError as exc:
            logger.debug(
                "HEAD request exception, proceeding with GET",
                extra={"stage": "download", "error": str(exc), "url": url},
            )

    request_headers = dict(headers)
    if resume_position > 0:
        request_headers["Range"] = f"bytes={resume_position}-"

    if bucket is not None:
        bucket.consume()
    extensions = {
        "config": http_config,
        "headers": request_headers,
        "correlation_id": correlation_id,
    }

    with request_with_redirect_audit(
        client=client,
        method="GET",
        url=url,
        headers=request_headers,
        timeout=http_config.download_timeout_sec,
        stream=True,
        http_config=http_config,
        assume_url_validated=True,
        extensions=extensions,
    ) as response:
        status_code = response.status_code
        if status_code in {429, 503}:
            retry_delay = _apply_retry_after_from_response(
                response=response,
                http_config=http_config,
                service=service,
                host=host,
            )
            http_error = httpx.HTTPStatusError(
                f"HTTP error {status_code}", request=response.request, response=response
            )
            if retry_delay is not None:
                setattr(http_error, "_retry_after_delay", retry_delay)
            raise http_error
        if status_code == 416 and resume_position > 0:
            part_path.unlink(missing_ok=True)
            raise DownloadFailure("Range request rejected by origin", retryable=True)
        if status_code == 304:
            return _StreamOutcome(
                status="cached",
                cache_path=cache_path,
                etag=response.headers.get("ETag") or etag_hint,
                last_modified=response.headers.get("Last-Modified") or last_modified_hint,
                content_type=content_type_hint,
                content_length=content_length_hint,
                from_cache=True,
            )

        response.raise_for_status()

        _validate_media_type(
            actual_content_type=response.headers.get("Content-Type"),
            expected_media_type=expected_media_type,
            http_config=http_config,
            logger=logger,
            url=url,
        )

        bytes_downloaded = _stream_body_to_cache(
            response=response,
            cache_path=cache_path,
            resume_position=resume_position,
            http_config=http_config,
            cancellation_token=cancellation_token,
            logger=logger,
            progress_percent_step=progress_percent_step,
            progress_bytes_threshold=progress_bytes_threshold,
        )

        status = "updated" if resume_position > 0 else "fresh"
        raw_length = _safe_int(response.headers.get("Content-Length"))
        if raw_length is None:
            raw_length = content_length_hint
        if raw_length is None:
            raw_length = bytes_downloaded if bytes_downloaded >= 0 else None
        return _StreamOutcome(
            status=status,
            cache_path=cache_path,
            etag=response.headers.get("ETag") or etag_hint,
            last_modified=response.headers.get("Last-Modified") or last_modified_hint,
            content_type=response.headers.get("Content-Type") or content_type_hint,
            content_length=raw_length,
            from_cache=False,
        )


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
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
    """Download ontology content via HTTPX streaming."""

    secure_url = url if url_already_validated else validate_url_security(url, http_config)
    parsed = urlparse(secure_url)
    host = parsed.hostname
    bucket = get_bucket(http_config=http_config, host=host, service=service)
    http_client = get_http_client(http_config)

    log_memory_usage(logger, stage="download", event="before")
    correlation_id = _extract_correlation_id(logger)

    polite_headers = http_config.polite_http_headers(correlation_id=correlation_id)

    def _build_headers(manifest: Optional[Dict[str, object]]) -> Dict[str, str]:
        merged: Dict[str, str] = {str(k): str(v) for k, v in polite_headers.items()}
        for key, value in headers.items():
            merged[str(key)] = str(value)
        merged.update(_conditional_headers_from_manifest(manifest))
        return merged

    progress_percent_step = float(getattr(http_config, "progress_log_percent_step", 0.1) or 0.0)
    raw_bytes_threshold = getattr(http_config, "progress_log_bytes_threshold", 5 * (1 << 20))
    progress_bytes_threshold = int(raw_bytes_threshold or 0)
    perform_head = bool(getattr(http_config, "perform_head_precheck", True))

    expected_algorithm, expected_digest = _parse_expected_hash(expected_hash)
    if expected_hash and (not expected_algorithm or not expected_digest):
        logger.warning(
            "expected checksum malformed",
            extra={"stage": "download", "checksum": expected_hash, "url": secure_url},
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(destination.name)
    url_hash = hashlib.sha256(secure_url.encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / f"{url_hash}_{safe_name}"

    raw_checksum_attempts = getattr(http_config, "checksum_mismatch_retries", 3)
    try:
        checksum_attempts = max(1, int(raw_checksum_attempts))
    except (TypeError, ValueError):
        checksum_attempts = 3

    manifest_for_attempt: Optional[Dict[str, object]] = previous_manifest

    def _compute_sha256(path: Path) -> str:
        return sha256_file(path)

    for attempt in range(1, checksum_attempts + 1):
        attempt_start = time.monotonic()
        request_headers = _build_headers(manifest_for_attempt)

        def _perform_download() -> _StreamOutcome:
            try:
                return _download_once(
                    client=http_client,
                    url=secure_url,
                    cache_path=cache_path,
                    headers=request_headers,
                    http_config=http_config,
                    bucket=bucket,
                    logger=logger,
                    correlation_id=correlation_id,
                    cancellation_token=cancellation_token,
                    expected_media_type=expected_media_type,
                    progress_percent_step=progress_percent_step,
                    progress_bytes_threshold=progress_bytes_threshold,
                    perform_head=perform_head,
                    service=service,
                    host=host,
                )
            except httpx.HTTPStatusError as exc:
                status_code = getattr(exc.response, "status_code", None)
                retryable = _is_retryable_status(status_code)
                message = f"HTTP error {status_code} while downloading {secure_url}" if status_code else str(exc)
                failure = DownloadFailure(message, status_code=status_code, retryable=retryable)
                retry_after_delay = getattr(exc, "_retry_after_delay", None)
                if retry_after_delay is not None:
                    setattr(failure, "_retry_after_delay", retry_after_delay)
                raise failure from exc
            except httpx.TransportError as exc:
                raise DownloadFailure(
                    f"Network error while downloading {secure_url}: {exc}",
                    retryable=True,
                ) from exc

        def _on_retry(attempt_idx: int, exc: Exception, delay: float) -> None:
            logger.warning(
                "download retrying",
                extra={
                    "stage": "download",
                    "attempt": attempt_idx,
                    "retry_delay_sec": round(delay, 2),
                    "error": str(exc),
                    "url": secure_url,
                },
            )

        try:
            outcome = retry_with_backoff(
                _perform_download,
                retryable=is_retryable_error,
                max_attempts=max(1, http_config.max_retries),
                backoff_base=http_config.backoff_factor,
                jitter=http_config.backoff_factor,
                callback=_on_retry,
                retry_after=lambda exc: getattr(exc, "_retry_after_delay", None),
            )
        except DownloadFailure:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "unexpected download error",
                extra={"stage": "download", "url": secure_url, "error": str(exc)},
            )
            raise OntologyDownloadError(f"Download failed for {secure_url}: {exc}") from exc

        if outcome.from_cache:
            if not cache_path.exists():
                raise OntologyDownloadError(
                    f"Cached file for {secure_url} is missing at {cache_path}"
                )
            elapsed_cached = (time.monotonic() - attempt_start) * 1000
            logger.info(
                "cache hit",
                extra={
                    "stage": "download",
                    "status": "cached",
                    "elapsed_ms": round(elapsed_cached, 2),
                },
            )
            artifact_path, cache_reference = _materialize_cached_file(cache_path, destination)
        else:
            artifact_path, cache_reference = _materialize_cached_file(cache_path, destination)
            elapsed = (time.monotonic() - attempt_start) * 1000
            logger.info(
                "download complete",
                extra={
                    "stage": "download",
                    "status": outcome.status,
                    "elapsed_ms": round(elapsed, 2),
                },
            )

        sha256 = _compute_sha256(artifact_path)
        expected_value: Optional[str] = None
        if expected_algorithm:
            if expected_algorithm == "sha256":
                expected_value = sha256
            else:
                expected_value = _compute_file_hash(artifact_path, expected_algorithm).lower()
        if expected_algorithm and expected_digest:
            if expected_value != expected_digest:
                logger.error(
                    "checksum mismatch detected",
                    extra={
                        "stage": "download",
                        "expected": expected_hash,
                        "actual": expected_value,
                        "url": secure_url,
                    },
                )
                artifact_path.unlink(missing_ok=True)
                if cache_reference != artifact_path:
                    cache_reference.unlink(missing_ok=True)
                cache_path.unlink(missing_ok=True)
                if attempt < checksum_attempts:
                    manifest_for_attempt = None
                    continue
                raise OntologyDownloadError(
                    f"checksum mismatch after {checksum_attempts} attempts: {secure_url}"
                )

        content_type = outcome.content_type
        if not content_type and manifest_for_attempt:
            manifest_type = manifest_for_attempt.get("content_type")
            if isinstance(manifest_type, str):
                content_type = manifest_type

        content_length = outcome.content_length
        if content_length is None and manifest_for_attempt:
            manifest_length = manifest_for_attempt.get("content_length")
            try:
                content_length = int(manifest_length) if manifest_length is not None else None
            except (TypeError, ValueError):
                content_length = None

        etag = outcome.etag
        if not etag and manifest_for_attempt:
            manifest_etag = manifest_for_attempt.get("etag")
            if isinstance(manifest_etag, str):
                etag = manifest_etag

        last_modified = outcome.last_modified
        if not last_modified and manifest_for_attempt:
            manifest_last_modified = manifest_for_attempt.get("last_modified")
            if isinstance(manifest_last_modified, str):
                last_modified = manifest_last_modified

        log_memory_usage(logger, stage="download", event="after")

        return DownloadResult(
            path=artifact_path,
            status=outcome.status,
            sha256=sha256,
            etag=etag,
            last_modified=last_modified,
            content_type=content_type,
            content_length=content_length,
        )

    raise OntologyDownloadError(
        f"checksum mismatch after {checksum_attempts} attempts: {secure_url}"
    )

    raise OntologyDownloadError(
        f"checksum mismatch after {max_checksum_attempts} attempts: {secure_url}"
    )
