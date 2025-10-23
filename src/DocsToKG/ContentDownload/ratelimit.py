# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.ratelimit",
#   "purpose": "Per-host and per-role rate limiting with pyrate-limiter.",
#   "sections": [
#     {
#       "id": "ratelimitexceeded",
#       "name": "RateLimitExceeded",
#       "anchor": "class-ratelimitexceeded",
#       "kind": "class"
#     },
#     {
#       "id": "ratetelemetrysink",
#       "name": "RateTelemetrySink",
#       "anchor": "class-ratetelemetrysink",
#       "kind": "class"
#     },
#     {
#       "id": "rolerates",
#       "name": "RoleRates",
#       "anchor": "class-rolerates",
#       "kind": "class"
#     },
#     {
#       "id": "hostpolicy",
#       "name": "HostPolicy",
#       "anchor": "class-hostpolicy",
#       "kind": "class"
#     },
#     {
#       "id": "backendconfig",
#       "name": "BackendConfig",
#       "anchor": "class-backendconfig",
#       "kind": "class"
#     },
#     {
#       "id": "aimdconfig",
#       "name": "AIMDConfig",
#       "anchor": "class-aimdconfig",
#       "kind": "class"
#     },
#     {
#       "id": "rateconfig",
#       "name": "RateConfig",
#       "anchor": "class-rateconfig",
#       "kind": "class"
#     },
#     {
#       "id": "rateacquisition",
#       "name": "RateAcquisition",
#       "anchor": "class-rateacquisition",
#       "kind": "class"
#     },
#     {
#       "id": "ratelimitregistry",
#       "name": "RateLimitRegistry",
#       "anchor": "class-ratelimitregistry",
#       "kind": "class"
#     },
#     {
#       "id": "ratelimitedtransport",
#       "name": "RateLimitedTransport",
#       "anchor": "class-ratelimitedtransport",
#       "kind": "class"
#     },
#     {
#       "id": "get-rate-limiter-manager",
#       "name": "get_rate_limiter_manager",
#       "anchor": "function-get-rate-limiter-manager",
#       "kind": "function"
#     },
#     {
#       "id": "set-rate-limiter-manager",
#       "name": "set_rate_limiter_manager",
#       "anchor": "function-set-rate-limiter-manager",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Per-host and per-role rate limiting with pyrate-limiter.

Provides:
- Multi-window rate limits (e.g., 10/SECOND + 5000/HOUR)
- Per-host and per-role separation
- HEAD discount support
- Bounded wait with configurable maximums
- Optional AIMD dynamic tuning
- Global in-flight ceiling
- HTTPX transport integration
- Comprehensive telemetry
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx

try:
    from pyrate_limiter import Limiter, Rate
except ImportError as e:
    raise RuntimeError("pyrate-limiter is required for rate limiting") from e

LOGGER = logging.getLogger(__name__)

# Type aliases
Role = Literal["metadata", "landing", "artifact"]


class RateLimitExceeded(RuntimeError):
    """Raised when a bounded wait elapses without acquiring capacity."""


class RateTelemetrySink(Protocol):
    """Protocol for rate limiting telemetry."""

    def emit_acquire(self, *, host: str, role: Role, delay_ms: int) -> None:
        """Emit when a token was successfully acquired."""
        ...

    def emit_block(self, *, host: str, role: Role, waited_ms: int, max_delay_ms: int) -> None:
        """Emit when acquisition was blocked due to max_delay_ms."""
        ...

    def emit_head_skipped(self, *, host: str) -> None:
        """Emit when HEAD request skipped rate limiting."""
        ...

    def emit_429(self, *, host: str, role: Role) -> None:
        """Emit when server returns 429 response."""
        ...

    def emit_success(self, *, host: str, role: Role) -> None:
        """Emit when request succeeds (non-429/503)."""
        ...

    def emit_aimd_adjust(
        self, *, host: str, role: Role, old: float, new: float, reason: str
    ) -> None:
        """Emit when AIMD adjusts rate multiplier."""
        ...


@dataclass(frozen=True)
class RoleRates:
    """Role-specific rate policy."""

    rates: list[str] = field(default_factory=list)
    max_delay_ms: int = 200
    count_head: bool = False
    max_concurrent: int | None = None


@dataclass(frozen=True)
class HostPolicy:
    """Per-host policy grouping role policies."""

    metadata: RoleRates | None = None
    landing: RoleRates | None = None
    artifact: RoleRates | None = None


@dataclass(frozen=True)
class BackendConfig:
    """Pyrate-limiter bucket backend configuration."""

    kind: str = "memory"
    dsn: str = ""


@dataclass(frozen=True)
class AIMDConfig:
    """Optional AIMD dynamic tuning configuration."""

    enabled: bool = False
    window_s: int = 60
    high_429_ratio: float = 0.05
    increase_step_pct: int = 5
    decrease_step_pct: int = 20
    min_multiplier: float = 0.3
    max_multiplier: float = 1.0


@dataclass(frozen=True)
class RateConfig:
    """Complete rate limiting configuration."""

    defaults: Mapping[Role, RoleRates]
    hosts: Mapping[str, HostPolicy]
    backend: BackendConfig = BackendConfig()
    aimd: AIMDConfig = AIMDConfig()
    global_max_inflight: int | None = 500


@dataclass
class RateAcquisition:
    """Result of an acquisition attempt."""

    acquired: bool
    delay_ms: int
    policy_max_delay_ms: int
    host: str = ""
    role: Role = "metadata"
    concurrency_held: bool = False


class RateLimitRegistry:
    """Process-global rate limiting registry."""

    def __init__(
        self,
        cfg: RateConfig,
        *,
        telemetry: RateTelemetrySink | None = None,
        run_id: str | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._tele = telemetry
        self._run_id = run_id
        self._now = now

        self._lock = threading.RLock()
        self._limiters: dict[tuple[str, Role], Limiter] = {}
        self._sem: dict[tuple[str, Role], threading.BoundedSemaphore] = {}
        self._aimd_mult: dict[tuple[str, Role], float] = {}
        self._counters: dict[tuple[str, Role], dict[str, int]] = {}

        # Global in-flight ceiling
        self._global_sem = (
            threading.BoundedSemaphore(cfg.global_max_inflight)
            if cfg.global_max_inflight and cfg.global_max_inflight > 0
            else None
        )

        # Initialize limiters and semaphores
        self._initialize()

    def _initialize(self) -> None:
        """Initialize limiters and semaphores for all configured hosts/roles."""
        for host, host_policy in self._cfg.hosts.items():
            for role in ["metadata", "landing", "artifact"]:
                policy = self._effective_policy(host, role)
                if policy.rates:
                    key = (host, role)
                    self._aimd_mult[key] = 1.0
                    self._counters[key] = {"total": 0, "status_429": 0}
                    if policy.max_concurrent:
                        self._sem[key] = threading.BoundedSemaphore(policy.max_concurrent)

    def _effective_policy(self, host: str, role: Role) -> RoleRates:
        """Resolve RoleRates for (host, role): host override -> defaults."""
        if host in self._cfg.hosts:
            host_policy = self._cfg.hosts[host]
            if role == "metadata" and host_policy.metadata:
                return host_policy.metadata
            if role == "landing" and host_policy.landing:
                return host_policy.landing
            if role == "artifact" and host_policy.artifact:
                return host_policy.artifact

        return self._cfg.defaults.get(role, RoleRates())

    def _parse_rates(self, rates: list[str]) -> list[Rate]:
        """Parse rate strings like '10/SECOND', '5000/HOUR', '1/3SECOND'."""
        parsed = []
        for rate_str in rates:
            if "/" not in rate_str:
                LOGGER.warning(f"Invalid rate format: {rate_str}")
                continue

            try:
                limit_part, duration_part = rate_str.split("/", 1)
                limit = int(limit_part.strip())

                # Handle fractions like "1/3SECOND"
                if "/" in duration_part:
                    denom, unit = duration_part.split("/", 1)
                    denom_int = int(denom.strip())
                    duration_ms = self._parse_duration_unit(unit.strip()) // denom_int
                else:
                    duration_ms = self._parse_duration_unit(duration_part.strip())

                parsed.append(Rate(limit, duration_ms))
            except Exception as e:
                LOGGER.warning(f"Failed to parse rate '{rate_str}': {e}")

        return parsed

    @staticmethod
    def _parse_duration_unit(unit: str) -> int:
        """Convert duration unit to milliseconds."""
        unit_upper = unit.upper()
        if unit_upper == "SECOND":
            return 1000
        elif unit_upper == "MINUTE":
            return 60000
        elif unit_upper == "HOUR":
            return 3600000
        elif unit_upper == "DAY":
            return 86400000
        else:
            # Default to seconds
            return 1000

    def _get_or_create_limiter(self, host: str, role: Role) -> Limiter | None:
        """Get or create a Limiter for (host, role)."""
        key = (host, role)
        if key in self._limiters:
            return self._limiters[key]

        policy = self._effective_policy(host, role)
        if not policy.rates:
            return None

        # Parse rates and apply AIMD multiplier
        rates = self._parse_rates(policy.rates)
        multiplier = self._aimd_mult.get(key, 1.0)

        # Scale rates by AIMD multiplier
        scaled_rates = [Rate(int(r.limit * multiplier), r.interval) for r in rates]

        try:
            limiter = Limiter(scaled_rates, raise_when_fail=False, max_delay=None)
            self._limiters[key] = limiter
            return limiter
        except Exception as e:
            LOGGER.error(f"Failed to create limiter for {host}/{role}: {e}")
            return None

    def acquire(self, *, host: str, role: Role, method: str) -> RateAcquisition:
        """Acquire capacity for (host, role) with bounded wait."""
        # Handle global in-flight ceiling
        if self._global_sem:
            if not self._global_sem.acquire(blocking=False):
                raise RateLimitExceeded("Global in-flight ceiling exceeded")

        concurrency_held = False
        concurrency_key: tuple[str, Role] | None = None

        try:
            policy = self._effective_policy(host, role)

            # HEAD discount
            if method == "HEAD" and not policy.count_head:
                if self._tele:
                    self._tele.emit_head_skipped(host=host)
                return RateAcquisition(
                    acquired=True,
                    delay_ms=0,
                    policy_max_delay_ms=0,
                    host=host,
                    role=role,
                    concurrency_held=False,
                )

            # Per-role concurrency cap
            key = (host, role)
            if key in self._sem:
                if not self._sem[key].acquire(blocking=False):
                    raise RateLimitExceeded(
                        f"Per-role concurrency ceiling exceeded for {host}/{role}"
                    )
                concurrency_held = True
                concurrency_key = key

            # Get or create limiter
            limiter = self._get_or_create_limiter(host, role)
            if not limiter:
                return RateAcquisition(
                    acquired=True,
                    delay_ms=0,
                    policy_max_delay_ms=policy.max_delay_ms,
                    host=host,
                    role=role,
                    concurrency_held=concurrency_held,
                )

            # Bounded wait acquisition
            start_time = self._now()
            acquired = limiter.try_acquire(f"{host}:{role}", weight=1)

            if not acquired:
                # Try with waits up to max_delay_ms
                elapsed = 0
                while elapsed < policy.max_delay_ms:
                    time.sleep(min(0.025, (policy.max_delay_ms - elapsed) / 1000))
                    acquired = limiter.try_acquire(f"{host}:{role}", weight=1)
                    if acquired:
                        break
                    elapsed = int((self._now() - start_time) * 1000)

            if not acquired:
                elapsed_ms = int((self._now() - start_time) * 1000)
                if self._tele:
                    self._tele.emit_block(
                        host=host, role=role, waited_ms=elapsed_ms, max_delay_ms=policy.max_delay_ms
                    )
                # Emit structured rate event
                from DocsToKG.ContentDownload.telemetry_helpers import emit_rate_event

                emit_rate_event(
                    telemetry=self._tele,
                    run_id=self._run_id or "unknown",
                    host=host,
                    role=role,
                    action="block",
                    delay_ms=elapsed_ms,
                    max_delay_ms=policy.max_delay_ms,
                )
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {host}/{role} after {elapsed_ms}ms"
                )

            elapsed_ms = int((self._now() - start_time) * 1000)
            if self._tele:
                self._tele.emit_acquire(host=host, role=role, delay_ms=elapsed_ms)
            # Emit structured rate event
            from DocsToKG.ContentDownload.telemetry_helpers import emit_rate_event

            emit_rate_event(
                telemetry=self._tele,
                run_id=self._run_id or "unknown",
                host=host,
                role=role,
                action="acquire",
                delay_ms=elapsed_ms,
                max_delay_ms=policy.max_delay_ms,
            )

            return RateAcquisition(
                acquired=True,
                delay_ms=elapsed_ms,
                policy_max_delay_ms=policy.max_delay_ms,
                host=host,
                role=role,
                concurrency_held=concurrency_held,
            )

        except Exception:
            if concurrency_held and concurrency_key:
                try:
                    self._sem[concurrency_key].release()
                except ValueError:
                    LOGGER.warning(
                        "Semaphore imbalance detected while rolling back %s/%s", host, role
                    )
            # Release global semaphore on error
            if self._global_sem:
                self._global_sem.release()
            raise

    def record_429(self, *, host: str, role: Role) -> None:
        """Record a server 429 for AIMD and diagnostics."""
        key = (host, role)
        with self._lock:
            if key in self._counters:
                self._counters[key]["status_429"] += 1
                self._counters[key]["total"] += 1
            if self._tele:
                self._tele.emit_429(host=host, role=role)

    def record_success(self, *, host: str, role: Role) -> None:
        """Record a success for AIMD recovery and diagnostics."""
        key = (host, role)
        with self._lock:
            if key in self._counters:
                self._counters[key]["total"] += 1
            if self._tele:
                self._tele.emit_success(host=host, role=role)

    def release_inflight(self) -> None:
        """Release the global in-flight semaphore (called after request completes)."""
        if self._global_sem:
            self._global_sem.release()

    def release_concurrency(self, *, host: str, role: Role) -> None:
        """Release the per-role concurrency semaphore, if held."""
        key = (host, role)
        sem = self._sem.get(key)
        if sem:
            try:
                sem.release()
            except ValueError:
                LOGGER.warning("Per-role semaphore imbalance detected for %s/%s", host, role)

    def tick_aimd(self) -> None:
        """Periodic AIMD adjustment (call every aimd.window_s seconds)."""
        if not self._cfg.aimd.enabled:
            return

        with self._lock:
            for key, counters in self._counters.items():
                host, role = key
                total = counters["total"]
                status_429 = counters["status_429"]

                if total == 0:
                    continue

                ratio = status_429 / total
                old_mult = self._aimd_mult.get(key, 1.0)

                if ratio >= self._cfg.aimd.high_429_ratio:
                    # Decrease multiplier
                    new_mult = max(
                        self._cfg.aimd.min_multiplier,
                        old_mult * (1 - self._cfg.aimd.decrease_step_pct / 100.0),
                    )
                else:
                    # Increase multiplier
                    new_mult = min(
                        self._cfg.aimd.max_multiplier,
                        old_mult * (1 + self._cfg.aimd.increase_step_pct / 100.0),
                    )

                if abs(new_mult - old_mult) > 0.001:
                    self._aimd_mult[key] = new_mult
                    self._limiters.pop(key, None)  # Force rebuild on next acquire
                    if self._tele:
                        reason = (
                            "high_429_ratio" if ratio >= self._cfg.aimd.high_429_ratio else "stable"
                        )
                        self._tele.emit_aimd_adjust(
                            host=host, role=role, old=old_mult, new=new_mult, reason=reason
                        )

                # Reset counters
                self._counters[key] = {"total": 0, "status_429": 0}


class RateLimitedTransport(httpx.BaseTransport):
    """HTTPX transport with per-(host, role) rate limiting."""

    def __init__(
        self,
        inner: httpx.BaseTransport,
        *,
        registry: RateLimitRegistry,
        telemetry: RateTelemetrySink | None = None,
    ) -> None:
        self._inner = inner
        self._reg = registry
        self._tele = telemetry

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Apply rate limiting and delegate to inner transport."""
        # Extract role and host
        role: Role = request.extensions.get("role", "metadata")  # type: ignore[assignment]
        host: str = str(request.url.host or "unknown").lower()

        # Acquire rate limit tokens
        acquisition: RateAcquisition | None = None
        try:
            acquisition = self._reg.acquire(host=host, role=role, method=request.method)
        except RateLimitExceeded as e:
            LOGGER.warning(f"Rate limit exceeded: {e}")
            raise

        try:
            # Send request
            response = self._inner.handle_request(request)

            # Record 429/503 or success
            if response.status_code in (429, 503):
                self._reg.record_429(host=host, role=role)
            else:
                self._reg.record_success(host=host, role=role)

            return response
        finally:
            if acquisition and acquisition.concurrency_held:
                self._reg.release_concurrency(host=host, role=role)
            # Release global in-flight ceiling
            self._reg.release_inflight()

    def close(self) -> None:
        """Close the inner transport."""
        try:
            self._inner.close()
        except Exception:
            pass


# Global rate limiter manager singleton
_GLOBAL_RATE_LIMITER_MANAGER: RateLimitRegistry | None = None
_RATE_LIMITER_LOCK = threading.Lock()


def get_rate_limiter_manager(
    cfg: RateConfig | None = None,
    *,
    telemetry: RateTelemetrySink | None = None,
    run_id: str | None = None,
) -> RateLimitRegistry:
    """Get or create the global rate limiter manager.

    Args:
        cfg: Configuration (used only on first call)
        telemetry: Optional telemetry sink
        run_id: Optional run identifier for telemetry

    Returns:
        Global RateLimitRegistry instance
    """
    global _GLOBAL_RATE_LIMITER_MANAGER

    if _GLOBAL_RATE_LIMITER_MANAGER is not None:
        return _GLOBAL_RATE_LIMITER_MANAGER

    with _RATE_LIMITER_LOCK:
        if _GLOBAL_RATE_LIMITER_MANAGER is not None:
            return _GLOBAL_RATE_LIMITER_MANAGER

        # Use provided config or create default
        if cfg is None:
            cfg = RateConfig(
                defaults={
                    "metadata": RoleRates(rates=["10/SECOND", "5000/HOUR"], max_delay_ms=200),
                    "landing": RoleRates(rates=["5/SECOND", "2000/HOUR"], max_delay_ms=250),
                    "artifact": RoleRates(rates=["2/SECOND", "500/HOUR"], max_delay_ms=2000),
                },
                hosts={},
            )

        manager = RateLimitRegistry(cfg, telemetry=telemetry, run_id=run_id)
        _GLOBAL_RATE_LIMITER_MANAGER = manager
        return manager


def set_rate_limiter_manager(manager: RateLimitRegistry | None) -> None:
    """Set the global rate limiter manager (for testing).

    Args:
        manager: Registry instance to use globally, or None to reset
    """
    global _GLOBAL_RATE_LIMITER_MANAGER
    with _RATE_LIMITER_LOCK:
        _GLOBAL_RATE_LIMITER_MANAGER = manager
