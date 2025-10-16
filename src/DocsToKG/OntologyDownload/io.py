"""Filesystem safety, rate limiting, and networking helpers for ontology downloads."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import random
import re
import shutil
import socket
import stat
import tarfile
import threading
import time
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
from urllib.parse import ParseResult, urlparse, urlunparse

import pooch
import psutil
import requests

from .errors import ConfigError, DownloadFailure, OntologyDownloadError, PolicyError
from .settings import DownloadConfiguration

_DNS_CACHE: Dict[str, Tuple[float, List[Tuple]]] = {}
_DNS_CACHE_TTL = 120.0


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename derived from ``filename``."""

    original = filename
    safe = filename.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", safe)
    safe = safe.strip("._") or "ontology"
    if len(safe) > 255:
        safe = safe[:255]
    if safe != original:
        logging.getLogger("DocsToKG.OntologyDownload").warning(
            "sanitized unsafe filename",
            extra={"stage": "sanitize", "original": original, "sanitized": safe},
        )
    return safe


def generate_correlation_id() -> str:
    """Return a short-lived identifier that links related log entries."""

    return uuid.uuid4().hex[:12]


def mask_sensitive_data(payload: Dict[str, object]) -> Dict[str, object]:
    """Return a copy of ``payload`` with common secret fields masked."""

    sensitive_keys = {"authorization", "api_key", "apikey", "token", "secret", "password"}
    token_pattern = re.compile(r"^[A-Za-z0-9+/=_-]{32,}$")

    def _mask_value(value: object, key_hint: Optional[str] = None) -> object:
        if isinstance(value, dict):
            return {
                sub_key: _mask_value(sub_value, sub_key.lower())
                for sub_key, sub_value in value.items()
            }
        if isinstance(value, list):
            return [_mask_value(item, key_hint) for item in value]
        if isinstance(value, tuple):
            return tuple(_mask_value(item, key_hint) for item in value)
        if isinstance(value, set):
            return {_mask_value(item, key_hint) for item in value}
        if isinstance(value, str):
            lowered = value.lower()
            if key_hint in sensitive_keys:
                return "***masked***"
            if "apikey" in lowered:
                return "***masked***"
            if key_hint == "authorization":
                token = value.strip()
                if "bearer " in lowered or token_pattern.match(token):
                    return "***masked***"
            if "bearer " in lowered:
                return "***masked***"
        return value

    masked: Dict[str, object] = {}
    for key, value in payload.items():
        lower = key.lower()
        if lower in sensitive_keys:
            masked[key] = "***masked***"
        else:
            masked[key] = _mask_value(value, lower)
    return masked


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
    """Resolve *host* using a short-lived cache to avoid repeated DNS lookups."""

    now = time.monotonic()
    cached = _DNS_CACHE.get(host)
    if cached is not None:
        timestamp, results = cached
        if now - timestamp <= _DNS_CACHE_TTL:
            return results
    results = socket.getaddrinfo(host, None)
    _DNS_CACHE[host] = (now, results)
    return results


def validate_url_security(url: str, http_config: Optional[DownloadConfiguration] = None) -> str:
    """Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists."""

    parsed = urlparse(url)
    logger = logging.getLogger("DocsToKG.OntologyDownload")
    if parsed.username or parsed.password:
        raise ConfigError("Credentials in URLs are not allowed")

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ConfigError("Only HTTP(S) URLs are allowed for ontology downloads")

    host = parsed.hostname
    if not host:
        raise ConfigError("URL must include hostname")

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
            raise ConfigError(f"Invalid internationalized hostname: {host}") from exc

    parsed = parsed._replace(netloc=_rebuild_netloc(parsed, ascii_host))

    allowed_exact: Set[str] = set()
    allowed_suffixes: Set[str] = set()
    allowed_host_ports: Dict[str, Set[int]] = {}
    allowed_port_set = http_config.allowed_port_set() if http_config else {80, 443}
    if http_config:
        normalized = http_config.normalized_allowed_hosts()
        if normalized:
            allowed_exact, allowed_suffixes, allowed_host_ports = normalized

    allow_private = False
    if allowed_exact or allowed_suffixes:
        if ascii_host in allowed_exact or any(
            ascii_host == suffix or ascii_host.endswith(f".{suffix}") for suffix in allowed_suffixes
        ):
            allow_private = True
        else:
            raise ConfigError(f"Host {host} not in allowlist")

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
        raise ConfigError("Only HTTPS URLs are allowed for ontology downloads")

    port = parsed.port
    if port is None:
        port = 80 if scheme == "http" else 443

    host_port_allowances = allowed_host_ports.get(ascii_host, set())
    if port not in allowed_port_set and port not in host_port_allowances:
        raise ConfigError(f"Port {port} is not permitted for ontology downloads")

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


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for the provided file."""

    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_MAX_COMPRESSION_RATIO = 10.0


def _validate_member_path(member_name: str) -> Path:
    """Validate archive member paths to prevent traversal attacks."""

    normalized = member_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute():
        raise ConfigError(f"Unsafe absolute path detected in archive: {member_name}")
    if not relative.parts:
        raise ConfigError(f"Empty path detected in archive: {member_name}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ConfigError(f"Unsafe path detected in archive: {member_name}")
    return Path(*relative.parts)


def _check_compression_ratio(
    *,
    total_uncompressed: int,
    compressed_size: int,
    archive: Path,
    logger: Optional[logging.Logger],
    archive_type: str,
) -> None:
    """Ensure compressed archives do not expand beyond the permitted ratio."""

    if compressed_size <= 0:
        return
    ratio = total_uncompressed / float(compressed_size)
    if ratio > _MAX_COMPRESSION_RATIO:
        if logger:
            logger.error(
                "archive compression ratio too high",
                extra={
                    "stage": "extract",
                    "archive": str(archive),
                    "ratio": round(ratio, 2),
                    "compressed_bytes": compressed_size,
                    "uncompressed_bytes": total_uncompressed,
                    "limit": _MAX_COMPRESSION_RATIO,
                },
            )
        raise ConfigError(
            f"{archive_type} archive {archive} expands to {total_uncompressed} bytes, "
            f"exceeding {_MAX_COMPRESSION_RATIO}:1 compression ratio"
        )


def extract_zip_safe(
    zip_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Extract a ZIP archive while preventing traversal and compression bombs."""

    if not zip_path.exists():
        raise ConfigError(f"ZIP archive not found: {zip_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        safe_members: List[tuple[zipfile.ZipInfo, Path]] = []
        total_uncompressed = 0
        for member in members:
            member_path = _validate_member_path(member.filename)
            mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise ConfigError(f"Unsafe link detected in archive: {member.filename}")
            if member.is_dir():
                safe_members.append((member, member_path))
                continue
            total_uncompressed += int(member.file_size)
            safe_members.append((member, member_path))
        compressed_size = max(
            zip_path.stat().st_size,
            sum(int(member.compress_size) for member in members) or 0,
        )
        _check_compression_ratio(
            total_uncompressed=total_uncompressed,
            compressed_size=compressed_size,
            archive=zip_path,
            logger=logger,
            archive_type="ZIP",
        )
        for member, member_path in safe_members:
            target_path = destination / member_path
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(target_path)
    if logger:
        logger.info(
            "extracted zip archive",
            extra={"stage": "extract", "archive": str(zip_path), "files": len(extracted)},
        )
    return extracted


def extract_tar_safe(
    tar_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks."""

    if not tar_path.exists():
        raise ConfigError(f"TAR archive not found: {tar_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    try:
        with tarfile.open(tar_path, mode="r:*") as archive:
            members = archive.getmembers()
            safe_members: List[tuple[tarfile.TarInfo, Path]] = []
            total_uncompressed = 0
            for member in members:
                member_path = _validate_member_path(member.name)
                if member.isdir():
                    safe_members.append((member, member_path))
                    continue
                if member.islnk() or member.issym():
                    raise ConfigError(f"Unsafe link detected in archive: {member.name}")
                if member.isdev():
                    raise ConfigError(
                        f"Unsupported special file detected in archive: {member.name}"
                    )
                if not member.isfile():
                    raise ConfigError(f"Unsupported tar member type encountered: {member.name}")
                total_uncompressed += int(member.size)
                safe_members.append((member, member_path))
            compressed_size = tar_path.stat().st_size
            _check_compression_ratio(
                total_uncompressed=total_uncompressed,
                compressed_size=compressed_size,
                archive=tar_path,
                logger=logger,
                archive_type="TAR",
            )
            for member, member_path in safe_members:
                if member.isdir():
                    (destination / member_path).mkdir(parents=True, exist_ok=True)
                    continue
                target_path = destination / member_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_file = archive.extractfile(member)
                if extracted_file is None:
                    raise ConfigError(f"Failed to extract member: {member.name}")
                with extracted_file as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted.append(target_path)
    except tarfile.TarError as exc:
        raise ConfigError(f"Failed to extract tar archive {tar_path}: {exc}") from exc
    if logger:
        logger.info(
            "extracted tar archive",
            extra={"stage": "extract", "archive": str(tar_path), "files": len(extracted)},
        )
    return extracted


_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2")


def extract_archive_safe(
    archive_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Extract archives by dispatching to the appropriate safe handler."""

    lower_name = archive_path.name.lower()
    if lower_name.endswith(".zip"):
        return extract_zip_safe(archive_path, destination, logger=logger)
    if any(lower_name.endswith(suffix) for suffix in _TAR_SUFFIXES):
        return extract_tar_safe(archive_path, destination, logger=logger)
    raise ConfigError(f"Unsupported archive format: {archive_path}")


# --- Rate limiter utilities merged from ratelimit.py ---

try:  # pragma: no cover - POSIX only
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no cover - Windows only
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - POSIX fallback
    msvcrt = None  # type: ignore[assignment]


class TokenBucket:
    """Token bucket used to enforce per-host and per-service rate limits."""

    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity or rate_per_sec
        self.tokens = self.capacity
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> None:
        """Consume tokens from the bucket, sleeping until capacity is available."""

        while True:
            with self.lock:
                now = time.monotonic()
                delta = now - self.timestamp
                self.timestamp = now
                self.tokens = min(self.capacity, self.tokens + delta * self.rate)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                needed = tokens - self.tokens
            time.sleep(max(needed / self.rate, 0.0))


class SharedTokenBucket(TokenBucket):
    """Token bucket backed by a filesystem state file for multi-process usage."""

    def __init__(
        self,
        *,
        rate_per_sec: float,
        capacity: float,
        state_path: Path,
    ) -> None:
        super().__init__(rate_per_sec=rate_per_sec, capacity=capacity)
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _acquire_file_lock(self, handle) -> None:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        elif msvcrt is not None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]

    def _release_file_lock(self, handle) -> None:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        elif msvcrt is not None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]

    def _read_state(self, handle) -> Tuple[float, float]:
        try:
            handle.seek(0)
            raw = handle.read()
            if not raw:
                return self.capacity, time.monotonic()
            data = json.loads(raw)
            return data.get("tokens", self.capacity), data.get("timestamp", time.monotonic())
        except (json.JSONDecodeError, OSError):
            return self.capacity, time.monotonic()

    def _write_state(self, handle, state) -> None:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(state))
        handle.flush()

    def _try_consume(self, tokens: float) -> Optional[float]:
        locked = False
        handle = None
        try:
            handle = self.state_path.open("a+")
            self._acquire_file_lock(handle)
            locked = True
            available, timestamp = self._read_state(handle)
            now = time.monotonic()
            delta = now - timestamp
            available = min(self.capacity, available + delta * self.rate)
            if available >= tokens:
                available -= tokens
                state = {"tokens": available, "timestamp": now}
                self._write_state(handle, state)
                return None
            state = {"tokens": available, "timestamp": now}
            self._write_state(handle, state)
            return tokens - available
        finally:
            if handle is not None:
                if locked:
                    self._release_file_lock(handle)
                handle.close()

    def consume(self, tokens: float = 1.0) -> None:  # type: ignore[override]
        """Consume tokens from the shared bucket, waiting when insufficient."""

        while True:
            with self.lock:
                needed = self._try_consume(tokens)
            if needed is None:
                return
            time.sleep(max(needed / self.rate, 0.0))


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
    ) -> Iterator[requests.Session]:
        """Yield a session associated with ``service``/``host`` and return it to the pool."""

        key = self._normalize(service, host)
        with self._lock:
            stack = self._pool.get(key)
            if stack:
                session = stack.pop()
                if not stack:
                    self._pool.pop(key, None)
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


def _shared_bucket_path(http_config: DownloadConfiguration, key: str) -> Optional[Path]:
    """Return the filesystem path for the shared token bucket state."""

    root = getattr(http_config, "shared_rate_limit_dir", None)
    if not root:
        return None
    base = Path(root).expanduser()
    token = re.sub(r"[^A-Za-z0-9._-]", "_", key).strip("._")
    if not token:
        token = "bucket"
    return base / f"{token}.json"


@dataclass(slots=True)
class _BucketEntry:
    bucket: TokenBucket
    rate: float
    capacity: float
    shared_path: Optional[Path]


class RateLimiterRegistry:
    """Manage shared token buckets keyed by (service, host)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[Tuple[str, str], _BucketEntry] = {}

    def _qualify(self, service: Optional[str], host: Optional[str]) -> Tuple[str, str]:
        service_key = (service or "_").lower()
        host_key = (host or "default").lower()
        return service_key, host_key

    def _normalize_rate(
        self, *, http_config: DownloadConfiguration, service: Optional[str]
    ) -> Tuple[float, float]:
        rate = http_config.rate_limit_per_second()
        if service:
            service_rate = http_config.parse_service_rate_limit(service)
            if service_rate:
                rate = service_rate
        rate = max(rate, 0.1)
        capacity = max(rate, 1.0)
        return rate, capacity

    def get_bucket(
        self,
        *,
        http_config: DownloadConfiguration,
        service: Optional[str],
        host: Optional[str],
    ) -> TokenBucket:
        """Return a token bucket for ``service``/``host`` using shared registry."""

        key = self._qualify(service, host)
        rate, capacity = self._normalize_rate(http_config=http_config, service=service)
        path_key = f"{key[0]}:{key[1]}"
        shared_path = _shared_bucket_path(http_config, path_key)
        with self._lock:
            entry = self._buckets.get(key)
            if (
                entry is None
                or entry.rate != rate
                or entry.capacity != capacity
                or entry.shared_path != shared_path
            ):
                if shared_path is not None:
                    bucket = SharedTokenBucket(
                        rate_per_sec=rate,
                        capacity=capacity,
                        state_path=shared_path,
                    )
                else:
                    bucket = TokenBucket(rate_per_sec=rate, capacity=capacity)
                entry = _BucketEntry(
                    bucket=bucket, rate=rate, capacity=capacity, shared_path=shared_path
                )
                self._buckets[key] = entry
            return entry.bucket

    def apply_retry_after(
        self,
        *,
        http_config: DownloadConfiguration,
        service: Optional[str],
        host: Optional[str],
        delay: float,
    ) -> None:
        """Reduce available tokens to honor server-provided retry-after hints."""

        if delay <= 0:
            return
        bucket = self.get_bucket(http_config=http_config, service=service, host=host)
        with bucket.lock:
            bucket.tokens = min(bucket.tokens, max(bucket.tokens - delay * bucket.rate, 0.0))
            bucket.timestamp = time.monotonic()

    def reset(self) -> None:
        """Clear all registered buckets (used in tests)."""

        with self._lock:
            self._buckets.clear()


REGISTRY = RateLimiterRegistry()
SESSION_POOL = SessionPool()


def get_bucket(
    *,
    http_config: DownloadConfiguration,
    service: Optional[str],
    host: Optional[str],
) -> TokenBucket:
    """Return a registry-managed bucket."""

    return REGISTRY.get_bucket(http_config=http_config, service=service, host=host)


def apply_retry_after(
    *,
    http_config: DownloadConfiguration,
    service: Optional[str],
    host: Optional[str],
    delay: float,
) -> None:
    """Adjust bucket capacity after receiving a Retry-After hint."""

    REGISTRY.apply_retry_after(
        http_config=http_config,
        service=service,
        host=host,
        delay=delay,
    )


def reset() -> None:
    """Clear all buckets (testing hook)."""

    REGISTRY.reset()
    SESSION_POOL.clear()



T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
    retryable: Callable[[BaseException], bool],
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    jitter: float = 0.5,
    callback: Optional[Callable[[int, BaseException, float], None]] = None,
    retry_after: Optional[Callable[[BaseException], Optional[float]]] = None,
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
        except BaseException as exc:  # pragma: no cover - behaviour verified via callers
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
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024**2)
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


_RETRYABLE_HTTP_STATUSES = {403, 408, 425, 429, 500, 502, 503, 504}


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
RDF_MIME_ALIASES: Set[str] = set()
RDF_MIME_FORMAT_LABELS: Dict[str, str] = {}
for canonical, aliases in _RDF_ALIAS_GROUPS.items():
    label = _RDF_FORMAT_LABELS[canonical]
    for alias in aliases:
        RDF_MIME_ALIASES.add(alias)
        RDF_MIME_FORMAT_LABELS[alias] = label


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

    Examples:
        >>> from pathlib import Path
        >>> from DocsToKG.OntologyDownload import DownloadConfiguration
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
    ) -> None:
        super().__init__(headers={}, progressbar=False, timeout=http_config.timeout_sec)
        self.destination = destination
        self.custom_headers = headers
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

    def _preliminary_head_check(
        self, url: str, session: requests.Session
    ) -> tuple[Optional[str], Optional[int]]:
        """Probe the origin with HEAD to audit media type and size before downloading.

        The HEAD probe allows the pipeline to abort before streaming large
        payloads that exceed configured limits and to log early warnings for
        mismatched Content-Type headers reported by the origin.

        Args:
            url: Fully qualified download URL resolved by the planner.
            session: Prepared requests session used for outbound calls.

        Returns:
            Tuple ``(content_type, content_length)`` extracted from response
            headers. Each element is ``None`` when the origin omits it.

        Raises:
            PolicyError: If the origin reports a payload larger than the
                configured ``max_download_size_gb`` limit.
        """

        try:
            response = session.head(
                url,
                headers=self.custom_headers,
                timeout=self.http_config.timeout_sec,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            self.logger.debug(
                "HEAD request exception, proceeding with GET",
                extra={"stage": "download", "error": str(exc), "url": url},
            )
            return None, None

        if response.status_code >= 400:
            self.logger.debug(
                "HEAD request failed, proceeding with GET",
                extra={
                    "stage": "download",
                    "method": "HEAD",
                    "status_code": response.status_code,
                    "url": url,
                },
            )
            return None, None

        content_type = response.headers.get("Content-Type")
        content_length_header = response.headers.get("Content-Length")
        content_length = int(content_length_header) if content_length_header else None

        if content_length:
            max_bytes = self.http_config.max_download_size_gb * (1024**3)
            if content_length > max_bytes:
                self.logger.error(
                    "file exceeds size limit (HEAD check)",
                    extra={
                        "stage": "download",
                        "content_length": content_length,
                        "limit_bytes": max_bytes,
                        "url": url,
                    },
                )
                raise PolicyError(
                    "File size {size} bytes exceeds limit of {limit} GB (detected via HEAD)".format(
                        size=content_length,
                        limit=self.http_config.max_download_size_gb,
                    )
                )

        return content_type, content_length

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
            PolicyError: If download limits are exceeded.
            OntologyDownloadError: If filesystem errors occur.
            requests.HTTPError: Propagated when HTTP status codes indicate failure.

        Returns:
            None
        """
        manifest_headers: Dict[str, str] = {}
        if "etag" in self.previous_manifest:
            manifest_headers["If-None-Match"] = self.previous_manifest["etag"]
        if "last_modified" in self.previous_manifest:
            manifest_headers["If-Modified-Since"] = self.previous_manifest["last_modified"]
        request_headers = {**self.custom_headers, **manifest_headers}
        part_path = Path(output_file + ".part")
        destination_part_path = Path(str(self.destination) + ".part")
        if not part_path.exists() and destination_part_path.exists():
            part_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination_part_path, part_path)

        self.head_content_type = None
        self.head_content_length = None
        self.response_content_type = None
        self.response_content_length = None

        with SESSION_POOL.lease(service=self.service, host=self.origin_host) as session:
            head_content_type, head_content_length = self._preliminary_head_check(url, session)
            self.head_content_type = head_content_type
            self.head_content_length = head_content_length
            if head_content_type:
                self._validate_media_type(head_content_type, self.expected_media_type, url)

            resume_position = part_path.stat().st_size if part_path.exists() else 0

            def _stream_once() -> str:
                nonlocal resume_position
                resume_position = part_path.stat().st_size if part_path.exists() else 0
                if resume_position:
                    request_headers["Range"] = f"bytes={resume_position}-"
                else:
                    request_headers.pop("Range", None)

                with session.get(
                    url,
                    headers=request_headers,
                    stream=True,
                    timeout=self.http_config.download_timeout_sec,
                    allow_redirects=True,
                ) as response:
                    if response.status_code == 304 and Path(self.destination).exists():
                        self.status = "cached"
                        self.response_etag = response.headers.get("ETag") or self.previous_manifest.get(
                            "etag"
                        )
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

                    if response.status_code == 206:
                        self.status = "updated"
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
                    max_bytes = self.http_config.max_download_size_gb * (1024**3)
                    if total_bytes is not None and total_bytes > max_bytes:
                        self.logger.error(
                            "file exceeds size limit",
                            extra={
                                "stage": "download",
                                "size": total_bytes,
                                "limit": max_bytes,
                            },
                        )
                        raise PolicyError(
                            f"File size {total_bytes} exceeds configured limit of "
                            f"{self.http_config.max_download_size_gb} GB"
                        )
                    if total_bytes:
                        completed_fraction = resume_position / total_bytes
                        if completed_fraction >= 1:
                            next_progress = None
                        else:
                            next_progress = ((int(completed_fraction * 10)) + 1) / 10
                    self.response_etag = response.headers.get("ETag")
                    self.response_last_modified = response.headers.get("Last-Modified")
                    mode = "ab" if resume_position else "wb"
                    bytes_downloaded = resume_position
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with part_path.open(mode) as fh:
                            for chunk in response.iter_content(chunk_size=1 << 20):
                                if not chunk:
                                    continue
                                fh.write(chunk)
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
                                if bytes_downloaded > self.http_config.max_download_size_gb * (
                                    1024**3
                                ):
                                    self.logger.error(
                                        "download exceeded size limit",
                                        extra={
                                            "stage": "download",
                                            "size": bytes_downloaded,
                                            "limit": self.http_config.max_download_size_gb
                                            * (1024**3),
                                        },
                                    )
                                    raise PolicyError(
                                        "Download exceeded maximum configured size while streaming"
                                    )
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

                    return "success"

            def _should_retry(exc: BaseException) -> bool:
                if isinstance(
                    exc,
                    (requests.ConnectionError, requests.Timeout, requests.exceptions.SSLError),
                ):
                    return True
                if isinstance(exc, requests.HTTPError):
                    response = getattr(exc, "response", None)
                    status = getattr(response, "status_code", None)
                    return _is_retryable_status(status)
                return False

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
                _stream_once,
                retryable=_should_retry,
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

    Returns:
        DownloadResult describing the final artifact and metadata.

    Raises:
        PolicyError: If policy validation fails or limits are exceeded.
        OntologyDownloadError: If retryable download mechanisms exhaust or IO fails.
    """
    secure_url = validate_url_security(url, http_config)
    parsed = urlparse(secure_url)
    host = parsed.hostname
    bucket = get_bucket(http_config=http_config, host=host, service=service)
    bucket.consume()

    start_time = time.monotonic()
    log_memory_usage(logger, stage="download", event="before")
    downloader = StreamingDownloader(
        destination=destination,
        headers=headers,
        http_config=http_config,
        previous_manifest=previous_manifest,
        logger=logger,
        expected_media_type=expected_media_type,
        service=service,
        origin_host=host,
    )

    def _resolved_content_metadata() -> tuple[Optional[str], Optional[int]]:
        content_type = downloader.response_content_type or downloader.head_content_type
        if content_type is None and previous_manifest:
            manifest_type = previous_manifest.get("content_type")
            if isinstance(manifest_type, str):
                content_type = manifest_type

        content_length = downloader.response_content_length
        if content_length is None and downloader.head_content_length is not None:
            content_length = downloader.head_content_length
        if content_length is None and previous_manifest:
            manifest_length = previous_manifest.get("content_length")
            try:
                content_length = int(manifest_length) if manifest_length is not None else None
            except (TypeError, ValueError):
                content_length = None
        return content_type, content_length

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(destination.name)
    url_hash = hashlib.sha256(secure_url.encode("utf-8")).hexdigest()[:12]
    cache_key = f"{url_hash}_{safe_name}"
    try:
        cached_path = Path(
            pooch.retrieve(
                secure_url,
                path=cache_dir,
                fname=cache_key,
                known_hash=expected_hash,
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
    except Exception as exc:  # pragma: no cover - defensive catch for pooch errors
        logger.error(
            "pooch download error",
            extra={"stage": "download", "url": secure_url, "error": str(exc)},
        )
        raise OntologyDownloadError(f"Download failed for {secure_url}: {exc}") from exc
    if downloader.status == "cached":
        elapsed = (time.monotonic() - start_time) * 1000
        logger.info(
            "cache hit",
            extra={"stage": "download", "status": "cached", "elapsed_ms": round(elapsed, 2)},
        )
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, destination)
        sha256 = sha256_file(destination)
        log_memory_usage(logger, stage="download", event="after")
        content_type, content_length = _resolved_content_metadata()
        return DownloadResult(
            path=destination,
            status="cached",
            sha256=sha256,
            etag=downloader.response_etag,
            last_modified=downloader.response_last_modified,
            content_type=content_type,
            content_length=content_length,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, destination)
    sha256 = sha256_file(destination)
    expected_hash = previous_manifest.get("sha256") if previous_manifest else None
    if expected_hash and expected_hash != sha256:
        logger.error(
            "sha256 mismatch detected",
            extra={
                "stage": "download",
                "expected": expected_hash,
                "actual": sha256,
                "url": secure_url,
            },
        )
        destination.unlink(missing_ok=True)
        cached_path.unlink(missing_ok=True)
        return download_stream(
            url=url,
            destination=destination,
            headers=headers,
            previous_manifest=None,
            http_config=http_config,
            cache_dir=cache_dir,
            logger=logger,
            expected_media_type=expected_media_type,
            service=service,
        )
    elapsed = (time.monotonic() - start_time) * 1000
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
    content_type, content_length = _resolved_content_metadata()
    return DownloadResult(
        path=destination,
        status=downloader.status,
        sha256=sha256,
        etag=downloader.response_etag,
        last_modified=downloader.response_last_modified,
        content_type=content_type,
        content_length=content_length,
    )


__all__ = [
    "sanitize_filename",
    "generate_correlation_id",
    "mask_sensitive_data",
    "validate_url_security",
    "sha256_file",
    "extract_zip_safe",
    "extract_tar_safe",
    "extract_archive_safe",
    "TokenBucket",
    "SharedTokenBucket",
    "RateLimiterRegistry",
    "REGISTRY",
    "SessionPool",
    "SESSION_POOL",
    "get_bucket",
    "apply_retry_after",
    "reset",
    "retry_with_backoff",
    "DownloadResult",
    "DownloadFailure",
    "StreamingDownloader",
    "download_stream",
    "log_memory_usage",
    "RDF_MIME_ALIASES",
    "RDF_MIME_FORMAT_LABELS",
    "format_bytes",
]


def __getattr__(name: str):
    """Lazily proxy pipeline helpers without incurring import cycles."""

    if name == "validate_manifest_dict":
        from .planning import validate_manifest_dict as _validate_manifest_dict

        return _validate_manifest_dict
    raise AttributeError(name)


# --- Lightweight helpers merged from utils.py ---


def format_bytes(num: int) -> str:
    """Return a human-readable representation for ``num`` bytes."""

    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"
