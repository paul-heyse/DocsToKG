# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.rate_limit",
#   "purpose": "Expose pyrate-limiter backed throttling for ontology downloads",
#   "sections": [
#     {
#       "id": "ratelimiterhandle",
#       "name": "RateLimiterHandle",
#       "anchor": "class-ratelimiterhandle",
#       "kind": "class"
#     },
#     {
#       "id": "table-name-for-key",
#       "name": "_table_name_for_key",
#       "anchor": "function-table-name-for-key",
#       "kind": "function"
#     },
#     {
#       "id": "parse-rate-string",
#       "name": "_parse_rate_string",
#       "anchor": "function-parse-rate-string",
#       "kind": "function"
#     },
#     {
#       "id": "normalise-shared-dir",
#       "name": "_normalise_shared_dir",
#       "anchor": "function-normalise-shared-dir",
#       "kind": "function"
#     },
#     {
#       "id": "limiteradapter",
#       "name": "_LimiterAdapter",
#       "anchor": "class-limiteradapter",
#       "kind": "class"
#     },
#     {
#       "id": "limiterentry",
#       "name": "_LimiterEntry",
#       "anchor": "class-limiterentry",
#       "kind": "class"
#     },
#     {
#       "id": "pyratelimitermanager",
#       "name": "_PyrateLimiterManager",
#       "anchor": "class-pyratelimitermanager",
#       "kind": "class"
#     },
#     {
#       "id": "get-bucket",
#       "name": "get_bucket",
#       "anchor": "function-get-bucket",
#       "kind": "function"
#     },
#     {
#       "id": "apply-retry-after",
#       "name": "apply_retry_after",
#       "anchor": "function-apply-retry-after",
#       "kind": "function"
#     },
#     {
#       "id": "reset",
#       "name": "reset",
#       "anchor": "function-reset",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Rate limiting façade for ontology downloads.

This module front-loads a pyrate-limiter manager that hands back blocking
handles keyed by ``(service, host)``.  The limits are derived from
:class:`DownloadConfiguration`, optionally persisted in SQLite so multiple
processes respect shared quotas.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set, Tuple

from pyrate_limiter import Duration, Limiter, Rate
from pyrate_limiter.buckets import InMemoryBucket
from pyrate_limiter.buckets.sqlite_bucket import SQLiteBucket
from pyrate_limiter.utils import validate_rate_list

from ..settings import DownloadConfiguration

logger = logging.getLogger("DocsToKG.OntologyDownload.rate_limit")

_RATE_LIMIT_PATTERN = re.compile(r"^([\d.]+)/(second|sec|s|minute|min|m|hour|h)$", re.IGNORECASE)
_UNIT_TO_MILLISECONDS = {
    "second": int(Duration.SECOND),
    "sec": int(Duration.SECOND),
    "s": int(Duration.SECOND),
    "minute": int(Duration.MINUTE),
    "min": int(Duration.MINUTE),
    "m": int(Duration.MINUTE),
    "hour": int(Duration.HOUR),
    "h": int(Duration.HOUR),
}
_BLOCKING_MAX_DELAY_MS = int(Duration.DAY) * 365  # wait up to ~one year before giving up
_BUFFER_MS = 0
_RETRY_SLEEP_SECONDS = 0.1


class RateLimiterHandle(Protocol):
    """Minimal interface exposed by pyrate limiter adapters."""

    def consume(self, tokens: float = 1.0) -> None:
        """Acquire tokens, blocking until the limiter admits the request."""


def _table_name_for_key(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", name)
    token = token.strip("_") or "bucket"
    return f"rl_{token}"


def _parse_rate_string(limit_text: str) -> List[Rate]:
    match = _RATE_LIMIT_PATTERN.match(limit_text.strip())
    if not match:
        raise ValueError(
            f"Invalid rate limit '{limit_text}'. Expected format: <number>/<unit> "
            "(e.g., '5/second', '60/minute', '1/hour')"
        )
    raw_value, unit_token = match.groups()
    value = float(raw_value)
    if value <= 0:
        raise ValueError("Rate limit values must be positive")

    base_ms = _UNIT_TO_MILLISECONDS[unit_token.lower()]
    fraction = Fraction(value).limit_denominator(1000)
    limit = max(1, fraction.numerator)
    interval_ms = int(base_ms * fraction.denominator)

    rates = [Rate(limit, interval_ms)]
    validate_rate_list(rates)
    return rates


def _normalise_shared_dir(shared_dir: Optional[Path]) -> Optional[Path]:
    if shared_dir is None:
        return None
    base = Path(shared_dir).expanduser()
    try:
        return base.resolve(strict=False)
    except TypeError:  # pragma: no cover - for Python versions without strict kwarg
        return base.resolve()


class _LimiterAdapter(RateLimiterHandle):
    """Thin wrapper providing the ``consume`` interface and cleanup hooks."""

    __slots__ = ("_limiter", "_name")

    def __init__(self, limiter: Limiter, name: str) -> None:
        self._limiter = limiter
        self._name = name

    def consume(self, tokens: float = 1.0) -> None:
        weight = max(1, math.ceil(tokens))
        wait_started_at: float | None = None

        while True:
            result = self._limiter.try_acquire(self._name, weight=weight)

            if not isinstance(result, bool):
                raise RuntimeError(
                    "Limiter produced an asynchronous result in a synchronous context"
                )

            if result:
                if wait_started_at is not None:
                    elapsed = time.monotonic() - wait_started_at
                    logger.debug(
                        "rate limiter admitted request after waiting",
                        extra={
                            "stage": "rate-limit",
                            "key": self._name,
                            "weight": weight,
                            "wait_s": round(elapsed, 6),
                        },
                    )
                return

            if wait_started_at is None:
                wait_started_at = time.monotonic()
                logger.debug(
                    "rate limiter throttled request; waiting for available tokens",
                    extra={
                        "stage": "rate-limit",
                        "key": self._name,
                        "weight": weight,
                    },
                )

            time.sleep(_RETRY_SLEEP_SECONDS)

    def dispose(self, bucket: InMemoryBucket | SQLiteBucket) -> None:
        """Release any background leakers and close bucket resources."""

        factory = getattr(self._limiter, "bucket_factory", None)
        leaker = getattr(factory, "_leaker", None)
        if leaker is not None:
            if hasattr(leaker, "deregister"):
                try:
                    leaker.deregister(bucket)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
            if hasattr(leaker, "join"):
                try:
                    leaker.join(timeout=1)  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
        dispose = getattr(self._limiter, "dispose", None)
        if callable(dispose):
            try:
                dispose(bucket)
            except Exception:  # pragma: no cover - defensive cleanup
                pass


@dataclass(slots=True)
class _LimiterEntry:
    adapter: _LimiterAdapter
    bucket: InMemoryBucket | SQLiteBucket
    rate_signature: str
    backend_signature: str


class _PyrateLimiterManager:
    """Construct and cache pyrate-limiter handles keyed by (service, host)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._limiters: Dict[Tuple[str, str], _LimiterEntry] = {}
        self._logged: Set[str] = set()

    def get_bucket(
        self,
        *,
        http_config: DownloadConfiguration,
        service: Optional[str],
        host: Optional[str],
    ) -> RateLimiterHandle:
        """Return a cached limiter handle for ``(service, host)``."""
        service_key = (service or "_").lower()
        host_key = (host or "default").lower()
        limiter_name = f"{service_key}:{host_key}"
        key = (service_key, host_key)
        limit_text = self._resolve_limit_text(http_config, service)
        backend_signature = self._expected_backend_signature(
            getattr(http_config, "shared_rate_limit_dir", None), limiter_name
        )

        with self._lock:
            entry = self._limiters.get(key)
            if (
                entry is not None
                and entry.rate_signature == limit_text
                and entry.backend_signature == backend_signature
            ):
                return entry.adapter

            adapter, bucket, actual_backend_signature = self._build_adapter(
                limit_text=limit_text,
                http_config=http_config,
                service_key=service_key,
                host_key=host_key,
                limiter_name=limiter_name,
            )
            if entry is not None:
                entry.adapter.dispose(entry.bucket)
            self._limiters[key] = _LimiterEntry(
                adapter=adapter,
                bucket=bucket,
                rate_signature=limit_text,
                backend_signature=actual_backend_signature,
            )
            return adapter

    def reset(self) -> None:
        with self._lock:
            for entry in self._limiters.values():
                entry.adapter.dispose(entry.bucket)
            self._limiters.clear()
            self._logged.clear()

    def _resolve_limit_text(
        self, http_config: DownloadConfiguration, service: Optional[str]
    ) -> str:
        if service:
            override = http_config.rate_limits.get(service)
            if override:
                return override
        return http_config.per_host_rate_limit

    def _build_adapter(
        self,
        *,
        limit_text: str,
        http_config: DownloadConfiguration,
        service_key: str,
        host_key: str,
        limiter_name: str,
    ) -> Tuple[_LimiterAdapter, str]:
        rates = _parse_rate_string(limit_text)
        bucket, backend_signature = self._create_bucket(
            rates=rates,
            shared_dir=getattr(http_config, "shared_rate_limit_dir", None),
            limiter_name=limiter_name,
        )
        limiter = Limiter(
            bucket,
            raise_when_fail=False,
            max_delay=_BLOCKING_MAX_DELAY_MS,
            retry_until_max_delay=True,
            buffer_ms=_BUFFER_MS,
        )
        adapter = _LimiterAdapter(limiter, limiter_name)
        self._log_backend_once(limiter_name, limit_text, backend_signature)
        return adapter, bucket, backend_signature

    def _create_bucket(
        self,
        *,
        rates: List[Rate],
        shared_dir: Optional[Path],
        limiter_name: str,
    ) -> Tuple[InMemoryBucket | SQLiteBucket, str]:
        normalised = _normalise_shared_dir(shared_dir)
        if normalised is None:
            return InMemoryBucket(rates), "memory"

        normalised.mkdir(parents=True, exist_ok=True)
        db_path = normalised / "ratelimit.sqlite"
        try:
            resolved = db_path.resolve(strict=False)
        except TypeError:  # pragma: no cover - for Python versions without strict kwarg
            resolved = db_path.resolve()
        table_name = _table_name_for_key(limiter_name)
        bucket = SQLiteBucket.init_from_file(
            rates,
            table=table_name,
            db_path=str(resolved),
            create_new_table=True,
            use_file_lock=True,
        )
        backend_signature = f"sqlite:{resolved}:{table_name}"
        return bucket, backend_signature

    def _expected_backend_signature(self, shared_dir: Optional[Path], limiter_name: str) -> str:
        normalised = _normalise_shared_dir(shared_dir)
        if normalised is None:
            return "memory"
        db_path = normalised / "ratelimit.sqlite"
        try:
            resolved = db_path.resolve(strict=False)
        except TypeError:  # pragma: no cover
            resolved = db_path.resolve()
        table_name = _table_name_for_key(limiter_name)
        return f"sqlite:{resolved}:{table_name}"

    def _log_backend_once(self, name: str, limit_text: str, backend_signature: str) -> None:
        if name in self._logged:
            return
        backend = "sqlite" if backend_signature.startswith("sqlite:") else "in-memory"
        logger.debug(
            "initialised rate limiter",
            extra={
                "stage": "rate-limit",
                "key": name,
                "backend": backend,
                "limit": limit_text,
            },
        )
        self._logged.add(name)


_PYRATE_MANAGER = _PyrateLimiterManager()


def get_bucket(
    *,
    http_config: DownloadConfiguration,
    service: Optional[str],
    host: Optional[str],
) -> RateLimiterHandle:
    """Return the configured rate limiter handle for ``service``/``host``."""

    provider_getter = getattr(http_config, "get_bucket_provider", None)
    if callable(provider_getter):
        candidate = provider_getter()
        if candidate is not None:
            return candidate(service, http_config, host)

    return _PYRATE_MANAGER.get_bucket(
        http_config=http_config,
        service=service,
        host=host,
    )


def apply_retry_after(
    *,
    http_config: DownloadConfiguration,
    service: Optional[str],
    host: Optional[str],
    delay: float,
) -> Optional[float]:
    """Return the parsed delay."""

    if delay <= 0:
        return None
    return delay


def reset() -> None:
    """Clear cached limiter state (used in tests)."""

    _PYRATE_MANAGER.reset()
