"""Centralised rate-limiting utilities built on pyrate-limiter.

This module owns the lifecycle of Limiter instances, policy validation,
and the HTTP transport shim that enforces host/role quotas beneath the
Hishel cache layer. Callers configure policies via
``configure_rate_limits`` and retrieve the singleton manager through
``get_rate_limiter_manager``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import httpx
from pyrate_limiter import (
    Duration,
    Limiter,
    Rate,
    BucketFullException,
    LimiterDelayException,
)
from pyrate_limiter.buckets import (
    InMemoryBucket,
    MultiprocessBucket,
    RedisBucket,
    SQLiteBucket,
    PostgresBucket,
)
from pyrate_limiter.utils import validate_rate_list

from DocsToKG.ContentDownload.errors import RateLimitError

LOGGER = logging.getLogger("DocsToKG.ContentDownload.ratelimit")

RoleName = str

ROLE_METADATA = "metadata"
ROLE_LANDING = "landing"
ROLE_ARTIFACT = "artifact"

DEFAULT_ROLE = ROLE_METADATA
ROLE_ORDER: Tuple[RoleName, ...] = (ROLE_METADATA, ROLE_LANDING, ROLE_ARTIFACT)


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(frozen=True)
class RoleConfig:
    """Resolved limiter configuration for a single role."""

    role: RoleName
    rates: Tuple[Rate, ...]
    mode: str
    max_delay_ms: int
    count_head: bool
    weight: int

    def fingerprint(self) -> Tuple:
        """Return a stable fingerprint used to detect policy changes."""

        return (
            self.role,
            tuple((rate.limit, rate.interval) for rate in self.rates),
            self.mode,
            self.max_delay_ms,
            self.count_head,
            self.weight,
        )


@dataclass
class RolePolicy:
    """Host-level policy keyed by role."""

    rates: Dict[RoleName, List[Rate]] = field(default_factory=dict)
    max_delay_ms: Dict[RoleName, int] = field(default_factory=dict)
    mode: Dict[RoleName, str] = field(default_factory=dict)
    count_head: Dict[RoleName, bool] = field(default_factory=dict)
    weight: Dict[RoleName, int] = field(default_factory=dict)

    def for_role(self, role: RoleName) -> Optional[RoleConfig]:
        role_name = role or DEFAULT_ROLE
        rate_list = self.rates.get(role_name)
        if not rate_list:
            return None

        mode_value = self.mode.get(role_name, "wait").lower()
        if mode_value not in {"wait", "raise"}:
            LOGGER.warning(
                "Unknown limiter mode %s for role=%s; defaulting to 'raise'",
                mode_value,
                role_name,
            )
            mode_value = "raise"

        max_delay = self.max_delay_ms.get(role_name, 0 if mode_value == "raise" else 250)
        if mode_value == "raise":
            max_delay = 0
        count_head = self.count_head.get(role_name, False)
        weight = self.weight.get(role_name, 1)
        weight = max(1, weight)

        return RoleConfig(
            role=role_name,
            rates=tuple(rate_list),
            mode=mode_value,
            max_delay_ms=int(max_delay),
            count_head=bool(count_head),
            weight=weight,
        )


@dataclass(frozen=True)
class BackendConfig:
    """Runtime limiter backend configuration."""

    backend: str = "memory"
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LimiterEntry:
    """Cache entry storing the constructed limiter and metadata."""

    limiter: Limiter
    config: RoleConfig
    backend: str
    created_at: datetime = field(default_factory=_now_utc)


@dataclass(frozen=True)
class AcquisitionResult:
    """Telemetry bundle returned after successful limiter acquisition."""

    host: str
    role: RoleName
    wait_ms: int
    backend: str
    mode: str


class LimiterCache:
    """Thread-safe limiter cache keyed by (host, role)."""

    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, RoleName], LimiterEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: Tuple[str, RoleName]) -> Optional[LimiterEntry]:
        with self._lock:
            return self._entries.get(key)

    def set(self, key: Tuple[str, RoleName], entry: LimiterEntry) -> LimiterEntry:
        with self._lock:
            self._entries[key] = entry
            return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _build_inmemory_limiter(config: RoleConfig) -> Limiter:
    bucket = InMemoryBucket(list(config.rates))
    max_delay = None if config.mode == "raise" else config.max_delay_ms
    limiter = Limiter(bucket, max_delay=max_delay)
    return limiter


def _build_multiprocess_limiter(config: RoleConfig) -> Limiter:
    bucket = MultiprocessBucket.init(list(config.rates))
    max_delay = None if config.mode == "raise" else config.max_delay_ms
    limiter = Limiter(bucket, max_delay=max_delay)
    return limiter


def _build_sqlite_limiter(config: RoleConfig, options: Mapping[str, Any]) -> Limiter:
    table = options.get("table", "docstokg_rate_bucket")
    db_path = options.get("path")
    use_file_lock = bool(options.get("use_file_lock", False))
    create_new = bool(options.get("create_table", True))
    bucket = SQLiteBucket.init_from_file(
        list(config.rates),
        table=table,
        db_path=db_path,
        create_new_table=create_new,
        use_file_lock=use_file_lock,
    )
    max_delay = None if config.mode == "raise" else config.max_delay_ms
    limiter = Limiter(bucket, max_delay=max_delay)
    return limiter


def _build_redis_limiter(config: RoleConfig, options: Mapping[str, Any]) -> Limiter:
    try:
        from redis import Redis
        from redis.asyncio import Redis as AsyncRedis  # noqa: F401 - imported for typing
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Redis backend selected but 'redis' package is not installed. "
            "Install redis-py or choose a different backend."
        ) from exc

    url = options.get("url")
    if not url:
        raise ValueError("Redis backend requires 'url' option (e.g. redis://localhost:6379/0)")

    namespace = options.get("namespace", "docstokg:ratelimit")
    redis_client = Redis.from_url(url)
    bucket_key = f"{namespace}:{config.role}"
    bucket = RedisBucket.init(list(config.rates), redis_client, bucket_key)
    if asyncio.iscoroutine(bucket):
        bucket = asyncio.get_event_loop().run_until_complete(bucket)  # pragma: no cover
    max_delay = None if config.mode == "raise" else config.max_delay_ms
    limiter = Limiter(bucket, max_delay=max_delay)
    return limiter


def _build_postgres_limiter(config: RoleConfig, options: Mapping[str, Any]) -> Limiter:
    dsn = options.get("dsn") or options.get("url")
    if not dsn:
        raise ValueError("Postgres backend requires 'dsn' (connection string) option.")
    table = options.get("table", "docstokg_rate_bucket")
    from psycopg import connect  # type: ignore  # pragma: no cover - optional dependency

    conn = connect(dsn)
    bucket = PostgresBucket.init(conn, list(config.rates), table=table)
    max_delay = None if config.mode == "raise" else config.max_delay_ms
    limiter = Limiter(bucket, max_delay=max_delay)
    return limiter


BACKEND_BUILDERS = MappingProxyType(
    {
        "memory": _build_inmemory_limiter,
        "multiprocess": _build_multiprocess_limiter,
        "sqlite": _build_sqlite_limiter,
        "redis": _build_redis_limiter,
        "postgres": _build_postgres_limiter,
    }
)


class LimiterManager:
    """Central registry managing rate limiter creation and acquisition."""

    def __init__(
        self,
        *,
        policies: Optional[Mapping[str, RolePolicy]] = None,
        backend_config: Optional[BackendConfig] = None,
    ) -> None:
        self._policies: Dict[str, RolePolicy] = {
            (host or "").lower(): policy for host, policy in (policies or {}).items()
        }
        self._backend_config = backend_config or BackendConfig()
        self._cache = LimiterCache()

    @property
    def backend(self) -> BackendConfig:
        return self._backend_config

    def configure_backend(self, backend_config: BackendConfig) -> None:
        if backend_config.backend != self._backend_config.backend:
            LOGGER.info(
                "Switching limiter backend from %s to %s",
                self._backend_config.backend,
                backend_config.backend,
            )
        self._backend_config = backend_config
        self._cache.clear()

    def configure_policies(self, policies: Mapping[str, RolePolicy]) -> None:
        self._policies = {host.lower(): policy for host, policy in policies.items()}
        self._cache.clear()

    def policies(self) -> Mapping[str, RolePolicy]:
        return MappingProxyType(self._policies)

    def get_policy(self, host: str) -> Optional[RolePolicy]:
        host_key = (host or "").lower()
        if host_key in self._policies:
            return self._policies[host_key]
        return self._policies.get("default")

    def _build_limiter(self, host: str, role_config: RoleConfig) -> LimiterEntry:
        backend_name = self._backend_config.backend.lower()
        builder = BACKEND_BUILDERS.get(backend_name)
        if builder is None:
            raise ValueError(f"Unsupported rate limiter backend '{backend_name}'")

        options = dict(self._backend_config.options)
        limiter = builder(role_config, options)
        LOGGER.debug(
            "Initialised limiter backend=%s host=%s role=%s rates=%s",
            backend_name,
            host,
            role_config.role,
            [(rate.limit, rate.interval) for rate in role_config.rates],
        )
        return LimiterEntry(limiter=limiter, config=role_config, backend=backend_name)

    def _get_limiter_entry(self, host: str, role_config: RoleConfig) -> LimiterEntry:
        key = (host, role_config.role)
        cached = self._cache.get(key)
        if cached and cached.config.fingerprint() == role_config.fingerprint():
            return cached

        entry = self._build_limiter(host, role_config)
        return self._cache.set(key, entry)

    def acquire(self, *, host: str, role: RoleName, method: str) -> AcquisitionResult:
        policy = self.get_policy(host)
        if not policy:
            return AcquisitionResult(host=host, role=role, wait_ms=0, backend="disabled", mode="disabled")

        role_config = policy.for_role(role)
        if not role_config or not role_config.rates:
            return AcquisitionResult(host=host, role=role, wait_ms=0, backend="disabled", mode="disabled")

        if method.upper() == "HEAD" and not role_config.count_head:
            return AcquisitionResult(
                host=host,
                role=role_config.role,
                wait_ms=0,
                backend=self._backend_config.backend,
                mode=role_config.mode,
            )

        entry = self._get_limiter_entry(host, role_config)
        limiter = entry.limiter
        key = f"{host}:{role_config.role}"

        start = time.perf_counter()
        try:
            limiter.try_acquire(key, weight=role_config.weight)
        except BucketFullException as exc:
            waited = int((time.perf_counter() - start) * 1000)
            wait_ms = self._estimate_wait_ms(limiter, exc.item)
            raise self._build_error(
                host=host,
                role=role_config.role,
                waited_ms=waited,
                required_wait_ms=wait_ms,
                backend=entry.backend,
                mode=role_config.mode,
                reason="bucket-full",
            ) from exc
        except LimiterDelayException as exc:
            waited = int((time.perf_counter() - start) * 1000)
            raise self._build_error(
                host=host,
                role=role_config.role,
                waited_ms=waited,
                required_wait_ms=exc.actual_delay,
                backend=entry.backend,
                mode=role_config.mode,
                reason="delay-exceeded",
                max_delay_ms=exc.max_delay,
            ) from exc

        waited_ms = int((time.perf_counter() - start) * 1000)
        return AcquisitionResult(
            host=host,
            role=role_config.role,
            wait_ms=waited_ms,
            backend=entry.backend,
            mode=role_config.mode,
        )

    def _estimate_wait_ms(self, limiter: Limiter, item) -> Optional[int]:
        try:
            bucket = limiter.bucket_factory.get(item)  # type: ignore[attr-defined]
            if asyncio.iscoroutine(bucket):  # pragma: no cover - async bucket rarely used
                loop = asyncio.get_event_loop()
                bucket = loop.run_until_complete(bucket)
            wait = bucket.waiting(item)  # type: ignore[union-attr]
            if asyncio.iscoroutine(wait):  # pragma: no cover
                wait = asyncio.get_event_loop().run_until_complete(wait)
            if isinstance(wait, int):
                return max(0, wait)
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("Failed to compute wait time for limiter item=%s", item, exc_info=True)
        return None

    def _build_error(
        self,
        *,
        host: str,
        role: RoleName,
        waited_ms: int,
        required_wait_ms: Optional[int],
        backend: str,
        mode: str,
        reason: str,
        max_delay_ms: Optional[int] = None,
    ) -> RateLimitError:
        next_allowed: Optional[datetime] = None
        retry_after: Optional[float] = None
        if required_wait_ms is not None:
            required_wait_ms = max(required_wait_ms, 0)
            next_allowed = _now_utc() + timedelta(milliseconds=required_wait_ms)
            retry_after = required_wait_ms / 1000.0

        message = (
            f"Rate limit exceeded for host={host} role={role} "
            f"(backend={backend}, mode={mode}, reason={reason})"
        )
        return RateLimitError(
            message,
            host=host,
            role=role,
            waited_ms=waited_ms,
            next_allowed_at=next_allowed,
            backend=backend,
            mode=mode,
            retry_after=retry_after,
            domain=host,
            details={"reason": reason, "max_delay_ms": max_delay_ms},
        )


def validate_policies(policies: Mapping[str, RolePolicy]) -> None:
    """Validate rate ordering for every host/role pair."""

    for host, policy in policies.items():
        for role in ROLE_ORDER:
            rates = policy.rates.get(role)
            if not rates:
                continue
            if not validate_rate_list(rates):
                raise ValueError(
                    f"Invalid rate ordering for host='{host}' role='{role}'. "
                    "Ensure rates are sorted by ascending interval/limit."
                )


def _build_default_policy() -> Dict[str, RolePolicy]:
    """Return conservative built-in defaults for known hosts."""

    def _bp(
        metadata_rates: Iterable[Rate],
        landing_rates: Iterable[Rate],
        artifact_rates: Iterable[Rate],
        *,
        metadata_delay: int = 200,
        landing_delay: int = 250,
        artifact_delay: int = 3000,
        count_head: bool = False,
    ) -> RolePolicy:
        return RolePolicy(
            rates={
                ROLE_METADATA: list(metadata_rates),
                ROLE_LANDING: list(landing_rates),
                ROLE_ARTIFACT: list(artifact_rates),
            },
            max_delay_ms={
                ROLE_METADATA: metadata_delay,
                ROLE_LANDING: landing_delay,
                ROLE_ARTIFACT: artifact_delay,
            },
            mode={
                ROLE_METADATA: "wait",
                ROLE_LANDING: "wait",
                ROLE_ARTIFACT: "wait",
            },
            count_head={
                ROLE_METADATA: count_head,
                ROLE_LANDING: count_head,
                ROLE_ARTIFACT: False,
            },
            weight={
                ROLE_METADATA: 1,
                ROLE_LANDING: 1,
                ROLE_ARTIFACT: 1,
            },
        )

    default_policy = RolePolicy(
        rates={
            ROLE_METADATA: [Rate(10, Duration.SECOND), Rate(5000, Duration.HOUR)],
            ROLE_LANDING: [Rate(6, Duration.SECOND), Rate(2000, Duration.HOUR)],
            ROLE_ARTIFACT: [Rate(2, Duration.SECOND), Rate(180, Duration.MINUTE)],
        },
        max_delay_ms={
            ROLE_METADATA: 150,
            ROLE_LANDING: 250,
            ROLE_ARTIFACT: 3000,
        },
        mode={
            ROLE_METADATA: "wait",
            ROLE_LANDING: "wait",
            ROLE_ARTIFACT: "wait",
        },
        count_head={
            ROLE_METADATA: False,
            ROLE_LANDING: False,
            ROLE_ARTIFACT: False,
        },
        weight={
            ROLE_METADATA: 1,
            ROLE_LANDING: 1,
            ROLE_ARTIFACT: 1,
        },
    )

    policies: Dict[str, RolePolicy] = {
        "default": default_policy,
        "api.openalex.org": _bp(
            [Rate(10, Duration.SECOND), Rate(4000, Duration.HOUR)],
            [Rate(5, Duration.SECOND), Rate(1500, Duration.HOUR)],
            [Rate(2, Duration.SECOND), Rate(120, Duration.MINUTE)],
            metadata_delay=150,
            landing_delay=250,
            artifact_delay=4000,
        ),
        "api.crossref.org": _bp(
            [Rate(50, Duration.SECOND), Rate(7500, Duration.HOUR)],
            [Rate(20, Duration.SECOND), Rate(5000, Duration.HOUR)],
            [Rate(5, Duration.SECOND), Rate(500, Duration.MINUTE)],
            metadata_delay=250,
            landing_delay=350,
            artifact_delay=5000,
            count_head=False,
        ),
        "export.arxiv.org": _bp(
            [Rate(1, Duration.SECOND), Rate(60, Duration.MINUTE)],
            [Rate(1, Duration.SECOND), Rate(30, Duration.MINUTE)],
            [Rate(1, Duration.SECOND), Rate(20, Duration.MINUTE)],
            metadata_delay=0,
            landing_delay=0,
            artifact_delay=6000,
        ),
        "api.unpaywall.org": _bp(
            [Rate(10, Duration.SECOND), Rate(1000, Duration.HOUR)],
            [Rate(5, Duration.SECOND), Rate(500, Duration.HOUR)],
            [Rate(2, Duration.SECOND), Rate(150, Duration.MINUTE)],
            metadata_delay=200,
            landing_delay=200,
            artifact_delay=4000,
        ),
    }
    return policies


def clone_role_policy(policy: RolePolicy) -> RolePolicy:
    """Return a deep copy of ``policy`` so callers can mutate safely."""

    return RolePolicy(
        rates={role: list(rates) for role, rates in policy.rates.items()},
        max_delay_ms=dict(policy.max_delay_ms),
        mode=dict(policy.mode),
        count_head=dict(policy.count_head),
        weight=dict(policy.weight),
    )


def clone_policies(policies: Mapping[str, RolePolicy]) -> Dict[str, RolePolicy]:
    """Deep copy host policies for mutation."""

    return {host: clone_role_policy(policy) for host, policy in policies.items()}


_GLOBAL_MANAGER = LimiterManager(
    policies=_build_default_policy(),
    backend_config=BackendConfig(backend="memory", options={}),
)


def get_rate_limiter_manager() -> LimiterManager:
    """Return the process-wide limiter manager."""

    return _GLOBAL_MANAGER


def configure_rate_limits(
    *,
    policies: Optional[Mapping[str, RolePolicy]] = None,
    backend: Optional[BackendConfig] = None,
) -> LimiterManager:
    """Configure the singleton limiter manager and return it."""

    manager = get_rate_limiter_manager()
    if policies is not None:
        validate_policies(policies)
        manager.configure_policies(policies)
    if backend is not None:
        manager.configure_backend(backend)
    return manager


class RateLimitedTransport(httpx.BaseTransport):
    """HTTPX transport wrapper that enforces rate limiter policies."""

    def __init__(
        self,
        inner: httpx.BaseTransport,
        manager: Optional[LimiterManager] = None,
    ) -> None:
        self._inner = inner
        self._manager = manager or get_rate_limiter_manager()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host or ""
        role = str(request.extensions.get("role") or DEFAULT_ROLE).lower()
        method = request.method.upper()

        try:
            outcome = self._manager.acquire(host=host, role=role, method=method)
        except RateLimitError as exc:
            raise
        else:
            meta: MutableMapping[str, Any] = request.extensions.setdefault(
                "docs_network_meta", {}
            )  # type: ignore[assignment]
            meta["rate_limiter_wait_ms"] = outcome.wait_ms
            meta["rate_limiter_backend"] = outcome.backend
            meta["rate_limiter_mode"] = outcome.mode
            meta["rate_limiter_role"] = outcome.role
            return self._inner.handle_request(request)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
