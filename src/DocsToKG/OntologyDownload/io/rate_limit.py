"""Rate limiting primitives for ontology downloads."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..settings import DownloadConfiguration

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


def get_bucket(
    *,
    http_config: DownloadConfiguration,
    service: Optional[str],
    host: Optional[str],
) -> TokenBucket:
    """Return a registry-managed bucket."""

    provider_getter = getattr(http_config, "get_bucket_provider", None)
    if callable(provider_getter):
        candidate = provider_getter()
        if candidate is not None:
            return candidate(service, http_config, host)
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


REGISTRY = RateLimiterRegistry()
