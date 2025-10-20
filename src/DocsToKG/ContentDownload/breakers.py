# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.breakers",
#   "purpose": "Per-host and per-resolver circuit breakers using pybreaker",
#   "sections": [
#     {
#       "id": "requestrole",
#       "name": "RequestRole",
#       "anchor": "class-requestrole",
#       "kind": "class"
#     },
#     {
#       "id": "breakeropenerror",
#       "name": "BreakerOpenError",
#       "anchor": "class-breakeropenerror",
#       "kind": "class"
#     },
#     {
#       "id": "breakerrolepolicy",
#       "name": "BreakerRolePolicy",
#       "anchor": "class-breakerrolepolicy",
#       "kind": "class"
#     },
#     {
#       "id": "breakerpolicy",
#       "name": "BreakerPolicy",
#       "anchor": "class-breakerpolicy",
#       "kind": "class"
#     },
#     {
#       "id": "breakerclassification",
#       "name": "BreakerClassification",
#       "anchor": "class-breakerclassification",
#       "kind": "class"
#     },
#     {
#       "id": "rollingwindowpolicy",
#       "name": "RollingWindowPolicy",
#       "anchor": "class-rollingwindowpolicy",
#       "kind": "class"
#     },
#     {
#       "id": "halfopenpolicy",
#       "name": "HalfOpenPolicy",
#       "anchor": "class-halfopenpolicy",
#       "kind": "class"
#     },
#     {
#       "id": "breakerconfig",
#       "name": "BreakerConfig",
#       "anchor": "class-breakerconfig",
#       "kind": "class"
#     },
#     {
#       "id": "cooldownstore",
#       "name": "CooldownStore",
#       "anchor": "class-cooldownstore",
#       "kind": "class"
#     },
#     {
#       "id": "inmemorycooldownstore",
#       "name": "InMemoryCooldownStore",
#       "anchor": "class-inmemorycooldownstore",
#       "kind": "class"
#     },
#     {
#       "id": "breakerlistenerfactory",
#       "name": "BreakerListenerFactory",
#       "anchor": "class-breakerlistenerfactory",
#       "kind": "class"
#     },
#     {
#       "id": "breakerregistry",
#       "name": "BreakerRegistry",
#       "anchor": "class-breakerregistry",
#       "kind": "class"
#     },
#     {
#       "id": "is-failure-for-breaker",
#       "name": "is_failure_for_breaker",
#       "anchor": "function-is-failure-for-breaker",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Per-host (and optional per-resolver) circuit breakers using pybreaker.

This module provides a centralized circuit breaker implementation that replaces
the legacy CircuitBreaker in networking.py with a pybreaker-based solution.

Key Features:
- Per-host breakers with configurable failure thresholds and reset timeouts
- Optional per-resolver breakers for resolver-specific failure patterns
- Retry-After aware cooldowns that honor server directives
- Rolling window manual-open for intermittent failure patterns
- Half-open trial calls with configurable limits per role
- Pluggable cooldown storage (in-memory, SQLite, Redis)
- Comprehensive telemetry integration

Typical Usage:
    from DocsToKG.ContentDownload.breakers import BreakerRegistry, BreakerConfig

    config = BreakerConfig()  # Load from YAML/env/CLI
    registry = BreakerRegistry(config, listener_factory=my_listener_factory)
    
    # Pre-flight check
    registry.allow(host="api.crossref.org", role=RequestRole.METADATA)
    
    # After request
    registry.on_success(host="api.crossref.org", role=RequestRole.METADATA)
    # or
    registry.on_failure(host="api.crossref.org", role=RequestRole.METADATA, 
                        status=503, retry_after_s=60.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple
import time
import threading
from collections import deque

# Optional import: keep module import-safe if pybreaker is absent in some test envs.
try:
    import pybreaker  # type: ignore
except Exception:  # pragma: no cover
    pybreaker = None  # type: ignore


# ────────────────────────────────────────────────────────────────────────────────
# Roles & Exceptions
# ────────────────────────────────────────────────────────────────────────────────

class RequestRole(str, Enum):
    """Request roles for role-specific breaker policies."""
    METADATA = "metadata"
    LANDING  = "landing"
    ARTIFACT = "artifact"


class BreakerOpenError(RuntimeError):
    """Raised by BreakerRegistry.allow() when a breaker is open (cooldown in effect)."""
    pass


# ────────────────────────────────────────────────────────────────────────────────
# Policy dataclasses (config → runtime)
# ────────────────────────────────────────────────────────────────────────────────

DEFAULT_FAILURE_STATUSES = frozenset({429, 500, 502, 503, 504, 408})
DEFAULT_NEUTRAL_STATUSES  = frozenset({401, 403, 404, 410, 451})


@dataclass(frozen=True)
class BreakerRolePolicy:
    """Overrides for a specific role (metadata|landing|artifact)."""
    fail_max: Optional[int] = None
    reset_timeout_s: Optional[int] = None
    trial_calls: int = 1  # half-open: number of probe calls allowed


@dataclass(frozen=True)
class BreakerPolicy:
    """Base per-host policy, with optional per-role overrides and Retry-After cap."""
    fail_max: int = 5
    reset_timeout_s: int = 60
    retry_after_cap_s: int = 900
    roles: Mapping[RequestRole, BreakerRolePolicy] = field(default_factory=dict)


@dataclass(frozen=True)
class BreakerClassification:
    """What counts as failure vs neutral (non-failure) for breaker accounting."""
    failure_statuses: frozenset[int] = DEFAULT_FAILURE_STATUSES
    neutral_statuses: frozenset[int] = DEFAULT_NEUTRAL_STATUSES
    # Exceptions are provided by networking when constructing the registry (httpx exceptions, etc.)
    failure_exceptions: Tuple[type, ...] = tuple()


@dataclass(frozen=True)
class RollingWindowPolicy:
    """Best-effort manual-open if too many failures land in a short window."""
    enabled: bool = False
    window_s: int = 30
    threshold_failures: int = 6
    cooldown_s: int = 60


@dataclass(frozen=True)
class HalfOpenPolicy:
    """Global knobs for half-open behavior."""
    jitter_ms: int = 150  # stagger probe calls a little


@dataclass(frozen=True)
class BreakerConfig:
    """
    Fully-resolved config loaded at startup (from YAML/env/CLI):
    - defaults: applied to unknown hosts
    - hosts: host-specific policies
    - resolvers: optional per-resolver policies (applied in addition to host)
    """
    defaults: BreakerPolicy = field(default_factory=BreakerPolicy)
    classify: BreakerClassification = field(default_factory=BreakerClassification)
    half_open: HalfOpenPolicy = field(default_factory=HalfOpenPolicy)
    rolling: RollingWindowPolicy = field(default_factory=RollingWindowPolicy)
    hosts: Mapping[str, BreakerPolicy] = field(default_factory=dict)          # host -> policy
    resolvers: Mapping[str, BreakerPolicy] = field(default_factory=dict)      # resolver-name -> policy


# ────────────────────────────────────────────────────────────────────────────────
# Cooldown store abstraction (for cross-process sharing)
# ────────────────────────────────────────────────────────────────────────────────

class CooldownStore(Protocol):
    """
    External store for host cooldown overrides (Retry-After or rolling-window).
    Times are monotonic deadlines (time.monotonic()) to avoid wall clock drift.
    """
    def get_until(self, host: str) -> Optional[float]: ...
    def set_until(self, host: str, until_monotonic: float, reason: str) -> None: ...
    def clear(self, host: str) -> None: ...


@dataclass
class InMemoryCooldownStore:
    """Process-local cooldown store; safe default."""
    _until: Dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_until(self, host: str) -> Optional[float]:
        with self._lock:
            return self._until.get(host)

    def set_until(self, host: str, until_monotonic: float, reason: str) -> None:
        with self._lock:
            self._until[host] = until_monotonic

    def clear(self, host: str) -> None:
        with self._lock:
            self._until.pop(host, None)


# TODO: Add SQLiteCooldownStore / RedisCooldownStore skeletons if/when needed.


# ────────────────────────────────────────────────────────────────────────────────
# Listener factory protocol (to emit transitions)
# ────────────────────────────────────────────────────────────────────────────────

class BreakerListenerFactory(Protocol):
    """
    Given (host, scope, resolver) returns a pybreaker listener (or None) to attach.
    Scope is 'host' or 'resolver'. Use your NetworkBreakerListener from
    networking_breaker_listener.py to build these.
    """
    def __call__(self, host: str, scope: str, resolver: Optional[str]) -> Optional[object]: ...


# ────────────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class _HalfOpenCounter:
    """Track how many probes are allowed in half-open per (host, role)."""
    remaining: int
    last_open_at: float


class BreakerRegistry:
    """
    Central registry for per-host and per-resolver breakers.

    Typical usage (networking layer):
        reg = BreakerRegistry(config, cooldown_store=InMemoryCooldownStore(), ...)
        reg.allow(host, role=RequestRole.METADATA, resolver="crossref")
        try:
            # send http request here
            ...
            reg.on_success(host, role=RequestRole.METADATA, resolver="crossref")
        except Exception as e:
            reg.on_failure(host, role=RequestRole.METADATA, resolver="crossref",
                          status=get_status_if_any(e), exception=e, retry_after_s=retry_after)
            raise
    """

    def __init__(
        self,
        config: BreakerConfig,
        *,
        cooldown_store: Optional[CooldownStore] = None,
        listener_factory: Optional[BreakerListenerFactory] = None,
        now_monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if pybreaker is None:  # pragma: no cover
            raise RuntimeError("pybreaker is required for BreakerRegistry")

        self.config = config
        self.cooldowns = cooldown_store or InMemoryCooldownStore()
        self.listener_factory = listener_factory
        self._now = now_monotonic

        # Internal storages
        self._host_breakers: Dict[str, pybreaker.CircuitBreaker] = {}
        self._resolver_breakers: Dict[str, pybreaker.CircuitBreaker] = {}
        self._rolling_fails: Dict[str, deque[float]] = {}  # host -> failure timestamps (monotonic)
        self._half_open: Dict[Tuple[str, RequestRole], _HalfOpenCounter] = {}
        self._lock = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────────────

    def allow(self, host: str, *, role: RequestRole, resolver: Optional[str] = None) -> None:
        """
        Pre-flight check before sending a network request.

        Raises:
            BreakerOpenError if the breaker is open OR a manual cooldown override is active.
        """
        now = self._now()
        host_key = host.lower()  # host is expected to be canonicalized upstream

        with self._lock:
            # 1) Cooldown override (Retry-After or rolling window)
            until = self.cooldowns.get_until(host_key)
            if until and until > now:
                raise BreakerOpenError(f"host={host_key} cooldown_remaining_ms={int((until-now)*1000)}")

            # 2) Manual-open from rolling-window policy (check window and maybe set cooldown)
            if self._should_manual_open(host_key, now):
                # If opened now, the cooldown store holds the deadline; deny this call.
                until2 = self.cooldowns.get_until(host_key)
                if until2 and until2 > now:
                    raise BreakerOpenError(f"host={host_key} cooldown_remaining_ms={int((until2-now)*1000)}")

            # 3) pybreaker state (host)
            h_cb = self._get_or_create_host_breaker(host_key)
            if h_cb.current_state == pybreaker.STATE_OPEN:
                remaining = self._remaining_cooldown_ms(h_cb, now)
                raise BreakerOpenError(f"host={host_key} breaker=open remaining_ms={remaining}")

            # 4) pybreaker state (resolver) — optional
            if resolver:
                r_cb = self._get_or_create_resolver_breaker(resolver)
                if r_cb.current_state == pybreaker.STATE_OPEN:
                    remaining = self._remaining_cooldown_ms(r_cb, now)
                    raise BreakerOpenError(f"resolver={resolver} breaker=open remaining_ms={remaining}")

            # 5) Half-open trial calls per role
            self._enforce_half_open_probe_limit(host_key, role, h_cb, now)

    def on_success(self, host: str, *, role: RequestRole, resolver: Optional[str] = None) -> None:
        """
        Call when a request succeeded (2xx/3xx or healthy terminal path).
        Resets counters and clears cooldown overrides.
        """
        host_key = host.lower()
        with self._lock:
            self._get_or_create_host_breaker(host_key).call_success()
            if resolver:
                self._get_or_create_resolver_breaker(resolver).call_success()
            # Clear any manual override cooldown
            self.cooldowns.clear(host_key)
            # Reset half-open probe counter for this (host, role)
            self._half_open.pop((host_key, role), None)

    def on_failure(
        self,
        host: str,
        *,
        role: RequestRole,
        resolver: Optional[str] = None,
        status: Optional[int] = None,
        exception: Optional[BaseException] = None,
        retry_after_s: Optional[float] = None,
    ) -> None:
        """
        Call when a request failed due to retryable server status or network exception.
        May open pybreaker and/or set a cooldown override (Retry-After or rolling-window).
        """
        host_key = host.lower()
        now = self._now()
        with self._lock:
            # Record pybreaker failures (host + optional resolver)
            self._get_or_create_host_breaker(host_key).call_failed()
            if resolver:
                self._get_or_create_resolver_breaker(resolver).call_failed()

            # Rolling window accounting
            self._record_failure_for_rolling(host_key, now)

            # Retry-After aware cooldown (host-level)
            if status in (429, 503) and retry_after_s:
                cap = self._policy_for_host(host_key).retry_after_cap_s
                duration = min(float(retry_after_s), float(cap))
                self.cooldowns.set_until(host_key, now + duration, reason="retry-after")

    # ── Query helpers (optional, useful for telemetry) ────────────────────────

    def current_state(self, host: str, *, resolver: Optional[str] = None) -> str:
        """Return the pybreaker state for ``host`` or an optional ``resolver``."""

        host_key = host.lower()
        with self._lock:
            if resolver:
                return self._state_name(self._get_or_create_resolver_breaker(resolver))
            return self._state_name(self._get_or_create_host_breaker(host_key))

    def cooldown_remaining_ms(self, host: str, *, resolver: Optional[str] = None) -> Optional[int]:
        """Return remaining cooldown in milliseconds for host/resolver if open."""

        host_key = host.lower()
        now = self._now()
        remaining: List[int] = []

        with self._lock:
            until = self.cooldowns.get_until(host_key)
            if until and until > now:
                remaining.append(int(max(0.0, (until - now)) * 1000))

            h_cb = self._host_breakers.get(host_key)
            if h_cb and h_cb.current_state == pybreaker.STATE_OPEN:
                remaining.append(self._remaining_cooldown_ms(h_cb, now))

            if resolver:
                r_cb = self._resolver_breakers.get(resolver)
                if r_cb and r_cb.current_state == pybreaker.STATE_OPEN:
                    remaining.append(self._remaining_cooldown_ms(r_cb, now))

        if remaining:
            return max(remaining)
        return None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _policy_for_host(self, host: str) -> BreakerPolicy:
        # Host-specific overrides, else defaults
        pol = self.config.hosts.get(host)
        return pol or self.config.defaults

    def _role_policy(self, base: BreakerPolicy, role: RequestRole) -> BreakerRolePolicy:
        return base.roles.get(role, BreakerRolePolicy())

    def _get_or_create_host_breaker(self, host: str) -> pybreaker.CircuitBreaker:
        cb = self._host_breakers.get(host)
        if cb is not None:
            return cb
        base = self._policy_for_host(host)
        # Apply role overrides in allow() / half-open enforcement (breaker stays host-wide)
        listeners = []
        if self.listener_factory:
            l = self.listener_factory(host, "host", None)
            if l is not None:
                listeners.append(l)
        cb = pybreaker.CircuitBreaker(
            fail_max=base.fail_max,
            reset_timeout=base.reset_timeout_s,
            state_storage=None,                # in-proc; cooldown overrides handled externally
            listeners=listeners,
        )
        self._host_breakers[host] = cb
        return cb

    def _get_or_create_resolver_breaker(self, resolver: str) -> pybreaker.CircuitBreaker:
        cb = self._resolver_breakers.get(resolver)
        if cb is not None:
            return cb
        base = self.config.resolvers.get(resolver, self.config.defaults)
        listeners = []
        if self.listener_factory:
            l = self.listener_factory(resolver, "resolver", resolver)
            if l is not None:
                listeners.append(l)
        cb = pybreaker.CircuitBreaker(
            fail_max=base.fail_max,
            reset_timeout=base.reset_timeout_s,
            state_storage=None,
            listeners=listeners,
        )
        self._resolver_breakers[resolver] = cb
        return cb

    def _remaining_cooldown_ms(self, cb: pybreaker.CircuitBreaker, now: float) -> int:
        """
        Estimate remaining cooldown for pybreaker-open. pybreaker doesn't expose
        a deadline, so we approximate using cb._state_storage.opened_at (impl detail).
        """
        try:
            opened_at = getattr(cb._state_storage, "opened_at", None)  # type: ignore[attr-defined]
            if opened_at is None:
                return 0
            elapsed = now - opened_at
            remain = max(0.0, float(cb.reset_timeout) - elapsed)
            return int(remain * 1000)
        except Exception:
            return 0

    def _enforce_half_open_probe_limit(self, host: str, role: RequestRole, cb: pybreaker.CircuitBreaker, now: float) -> None:
        """Allow only N trial calls in half-open per (host, role)."""
        if cb.current_state != pybreaker.STATE_HALF_OPEN:
            return
        base = self._policy_for_host(host)
        rp = self._role_policy(base, role)
        allowed = max(1, rp.trial_calls or 1)
        key = (host, role)
        counter = self._half_open.get(key)
        if counter is None or (now - counter.last_open_at) > float(base.reset_timeout_s):
            counter = _HalfOpenCounter(remaining=allowed, last_open_at=now)
            self._half_open[key] = counter
        if counter.remaining <= 0:
            raise BreakerOpenError(f"host={host} half-open probes exhausted for role={role.value}")
        # Jitter: space out probes a bit
        if self.config.half_open.jitter_ms > 0:
            # Callers may add a tiny sleep here if desired; registry only enforces budget.
            pass
        counter.remaining -= 1

    def _record_failure_for_rolling(self, host: str, now: float) -> None:
        rp = self.config.rolling
        if not rp.enabled:
            return
        dq = self._rolling_fails.get(host)
        if dq is None:
            dq = deque(maxlen=rp.threshold_failures)
            self._rolling_fails[host] = dq
        dq.append(now)
        # Drop old entries outside window
        cutoff = now - rp.window_s
        while dq and dq[0] < cutoff:
            dq.popleft()
        # If threshold reached, set a manual cooldown
        if len(dq) >= rp.threshold_failures:
            self.cooldowns.set_until(host, now + rp.cooldown_s, reason="rolling-window")

    def _should_manual_open(self, host: str, now: float) -> bool:
        rp = self.config.rolling
        if not rp.enabled:
            return False
        # If a cooldown is set, it's already open. If not, ensure window accounting is current.
        dq = self._rolling_fails.get(host)
        if not dq:
            return False
        cutoff = now - rp.window_s
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= rp.threshold_failures:
            # Set if not already set by _record_failure_for_rolling (race-safe)
            until = self.cooldowns.get_until(host)
            if not until or until <= now:
                self.cooldowns.set_until(host, now + rp.cooldown_s, reason="rolling-window")
            return True
        return False

    @staticmethod
    def _state_name(cb: pybreaker.CircuitBreaker) -> str:
        if cb.current_state == pybreaker.STATE_CLOSED:
            return "closed"
        if cb.current_state == pybreaker.STATE_OPEN:
            return "open"
        return "half_open"


# ────────────────────────────────────────────────────────────────────────────────
# Helpers for networking (classification)
# ────────────────────────────────────────────────────────────────────────────────

def is_failure_for_breaker(
    classify: BreakerClassification,
    *,
    status: Optional[int],
    exception: Optional[BaseException],
) -> bool:
    """
    Return True if this outcome should count as a breaker failure.
    - Retryable server statuses (429/5xx/408) count.
    - Exceptions matching failure_exceptions count.
    - Neutral statuses (401/403/404/410/451) do not count.
    - Anything else is treated as success for breaker accounting.
    """
    if exception is not None and classify.failure_exceptions:
        if isinstance(exception, classify.failure_exceptions):
            return True
    if status is None:
        return False
    if status in classify.neutral_statuses:
        return False
    return status in classify.failure_statuses
