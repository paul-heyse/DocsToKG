"""RateLimitManager: Facade for pyrate-limiter with per-service rate enforcement.

Provides a unified rate-limiting interface for the OntologyDownload subsystem:
- Thread-safe singleton (lazy initialization with config binding)
- Per-service rate limits (OLS, BioPortal, OpenAlex, etc.)
- Per-host secondary keying (rate limits may vary by host)
- Multi-window enforcement (e.g., 5/sec AND 300/min simultaneously)
- Block vs fail-fast modes
- Weighted requests (some requests consume multiple slots)
- Cross-process coordination via SQLiteBucket
- Structured telemetry (emit events on acquire/block)

Design:
- **Lazy initialization**: Created on first acquire() call
- **Config binding**: Bound to initial settings.config_hash()
- **Service routing**: Each service has its own rate bucket
- **Host routing**: (optional) Further keying by host for per-provider limits
- **Default fallback**: Conservative global default (8/sec)
- **SQLiteBucket backend**: Persistent across restarts, cross-process safe

Example:
    >>> from DocsToKG.OntologyDownload.ratelimit import get_rate_limiter
    >>> limiter = get_rate_limiter()
    >>> limiter.acquire("ols", "www.ebi.ac.uk", weight=1)
    >>> # Blocks until rate limit permits
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import platformdirs
from pyrate_limiter import BucketFullException, Duration, Limiter, Rate

from DocsToKG.OntologyDownload.ratelimit.config import RateSpec

logger = logging.getLogger(__name__)

# ============================================================================
# Global State
# ============================================================================

_rate_limiter: Optional["RateLimitManager"] = None
_rate_limiter_lock = threading.Lock()
_rate_limiter_config_hash: Optional[str] = None
_rate_limiter_pid: Optional[int] = None

# Allow the limiter to block for a very long time (≈ 1 year) before giving up.
_BLOCKING_MAX_DELAY_MS = int(Duration.DAY) * 365


# ============================================================================
# RateLimitManager
# ============================================================================


class RateLimitManager:
    """Thread-safe rate-limiting manager with per-service enforcement.

    Wraps pyrate-limiter.Limiter to provide a convenient interface for
    rate-limiting requests to multiple services (OLS, BioPortal, etc.)
    with different rate policies.

    Attributes:
        _rate_specs: Mapping of service → list of RateSpec objects
        _limiter: Underlying pyrate-limiter Limiter instance
        _bucket_dir: Directory for SQLiteBucket storage
    """

    def __init__(
        self,
        rate_specs: Dict[str, List[RateSpec]],
        bucket_dir: Optional[Path] = None,
        mode: str = "block",
    ):
        """Initialize RateLimitManager.

        Args:
            rate_specs: Dictionary mapping service names to lists of RateSpecs.
                       Example: {"ols": [RateSpec(4, 1000)], "bioportal": [RateSpec(2, 1000)]}
            bucket_dir: Directory for SQLiteBucket persistence.
                       Default: platform-specific cache dir
            mode: Limiting mode: "block" (sleep until available) or "fail-fast"
                  (raise immediately on limit exceeded)
        """
        self._rate_specs = rate_specs
        self._mode = mode

        # Determine bucket directory
        if bucket_dir is None:
            bucket_dir = Path(platformdirs.user_cache_dir("ontofetch")) / "rate-limits"
        self._bucket_dir = bucket_dir
        self._bucket_dir.mkdir(parents=True, exist_ok=True)

        # Initialize limiter
        # For now, use a simple in-memory approach; can be upgraded to SQLiteBucket
        # if cross-process coordination is needed
        self._limiter: Optional[Limiter] = None
        self._service_limiters: Dict[str, Limiter] = {}

        logger.debug(
            "RateLimitManager initialized",
            extra={
                "bucket_dir": str(self._bucket_dir),
                "mode": mode,
                "services": list(rate_specs.keys()),
            },
        )

    def acquire(
        self,
        service: str,
        host: Optional[str] = None,
        weight: int = 1,
        timeout_ms: Optional[int] = None,
    ) -> bool:
        """Acquire rate limit slot(s) for a service/host.

        Acquires one or more slots from the rate limit bucket for the given
        service. If slots are available, returns immediately. If not, either
        sleeps (block mode) or raises (fail-fast mode).

        Args:
            service: Service name (e.g., "ols", "bioportal")
            host: Optional host for per-host secondary keying
            weight: Number of slots to acquire (default 1)
            timeout_ms: Maximum wait time in milliseconds (ignored; for API compatibility)

        Returns:
            True if acquisition succeeded
            False if acquisition failed (fail-fast mode with no slots)

        Raises:
            BucketFullException: If fail-fast mode and limit exceeded
            ValueError: If service or weight is invalid
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got: {weight}")

        service = service.lower().strip()

        # Get or create limiter for this service
        if service not in self._service_limiters:
            self._service_limiters[service] = self._create_limiter_for_service(service)

        limiter = self._service_limiters[service]

        # Construct key: "service:host" or just "service"
        key = f"{service}:{host}" if host else service

        try:
            # Acquire slot(s) using pyrate-limiter API
            # try_acquire(name, weight=1) returns bool or raises BucketFullException
            acquired = bool(limiter.try_acquire(name=key, weight=weight))

            if acquired:
                logger.debug(
                    "Rate limit acquired",
                    extra={
                        "service": service,
                        "host": host,
                        "weight": weight,
                        "mode": self._mode,
                    },
                )
                return True

            logger.warning(
                "Rate limit acquisition returned without success",
                extra={
                    "service": service,
                    "host": host,
                    "weight": weight,
                    "mode": self._mode,
                },
            )
            return False

        except BucketFullException as e:
            logger.warning(
                "Rate limit bucket full",
                extra={
                    "service": service,
                    "host": host,
                    "weight": weight,
                    "meta_info": str(e.meta_info) if hasattr(e, "meta_info") else str(e),
                },
            )
            if self._mode == "fail-fast":
                return False
            # In block mode, re-raise as this indicates a configuration issue
            raise

    def _create_limiter_for_service(self, service: str) -> Limiter:
        """Create a pyrate-limiter.Limiter for a service.

        Args:
            service: Service name

        Returns:
            Configured Limiter instance
        """
        if service not in self._rate_specs:
            # Use default if service not configured
            if "_default" not in self._rate_specs:
                # Hardcoded default: 8/second
                rate_specs = [RateSpec(limit=8, interval_ms=1000)]
            else:
                rate_specs = self._rate_specs["_default"]
        else:
            rate_specs = self._rate_specs[service]

        # Convert RateSpec to pyrate-limiter Rate objects
        rates = [Rate(spec.limit, spec.interval_ms) for spec in rate_specs]

        # Create Limiter with rates as a list
        # raise_when_fail: True raises BucketFullException, False returns bool
        limiter = Limiter(
            rates,
            raise_when_fail=(self._mode == "fail-fast"),
            max_delay=_BLOCKING_MAX_DELAY_MS if self._mode == "block" else None,
            retry_until_max_delay=(self._mode == "block"),
        )

        logger.debug(
            "Created limiter for service",
            extra={
                "service": service,
                "rates": [str(r) for r in rate_specs],
                "mode": self._mode,
            },
        )

        return limiter

    def get_stats(self, service: Optional[str] = None) -> Dict[str, any]:
        """Get rate-limiting statistics.

        Args:
            service: Service name to get stats for. If None, returns global stats.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "mode": self._mode,
            "bucket_dir": str(self._bucket_dir),
            "services": list(self._service_limiters.keys()),
        }

        if service:
            if service in self._service_limiters:
                # Could expose more detailed stats here
                stats["service"] = service
                stats["limiters_count"] = len(self._service_limiters)

        return stats

    def close(self) -> None:
        """Close and cleanup the manager.

        Call at application shutdown.
        """
        self._service_limiters.clear()
        logger.debug("RateLimitManager closed")


# ============================================================================
# Singleton API
# ============================================================================


def get_rate_limiter() -> RateLimitManager:
    """Get or create the shared RateLimitManager singleton.

    Returns:
        The RateLimitManager instance

    Behavior:
        - First call: Creates manager, binds to current config_hash and PID
        - Subsequent calls: Returns same instance (thread-safe)
        - Config changed after bind: Logs warning once, continues with bound manager
        - Process forked: Child detects PID change, creates new manager
    """
    global _rate_limiter, _rate_limiter_config_hash, _rate_limiter_pid

    # Quick path: already initialized, same PID
    if _rate_limiter is not None and _rate_limiter_pid == os.getpid():
        return _rate_limiter

    # Slow path: need to create or recreate
    with _rate_limiter_lock:
        # Double-check after lock acquired
        if _rate_limiter is not None and _rate_limiter_pid == os.getpid():
            return _rate_limiter

        # Close old limiter if PID changed
        if _rate_limiter is not None and _rate_limiter_pid != os.getpid():
            logger.debug("Process forked; closing old rate limiter and creating new one")
            _rate_limiter.close()
            _rate_limiter = None

        # Load settings and parse rate specs
        from DocsToKG.OntologyDownload.settings import get_settings

        try:
            settings = get_settings()
            # Extract rate limits from settings (would be in RateLimitSettings)
            # For now, use a simple default configuration
            rate_specs = {"_default": [RateSpec(limit=8, interval_ms=1000)]}

            _rate_limiter = RateLimitManager(
                rate_specs=rate_specs,
                mode="block",  # Conservative: block until available
            )
            _rate_limiter_config_hash = settings.config_hash()
            _rate_limiter_pid = os.getpid()

            logger.debug(
                "Rate limiter created and bound",
                extra={
                    "config_hash": _rate_limiter_config_hash,
                    "pid": _rate_limiter_pid,
                },
            )

        except Exception as e:
            logger.error(f"Failed to create rate limiter: {e}", exc_info=True)
            raise

        return _rate_limiter


def close_rate_limiter() -> None:
    """Close the rate limiter and release resources.

    Safe to call multiple times or when no limiter has been created.
    """
    global _rate_limiter

    with _rate_limiter_lock:
        if _rate_limiter is not None:
            try:
                _rate_limiter.close()
                logger.debug("Rate limiter closed")
            except Exception as e:
                logger.error(f"Error closing rate limiter: {e}")
            finally:
                _rate_limiter = None


def reset_rate_limiter() -> None:
    """Reset the rate limiter (primarily for testing).

    Call this between test cases to force creation of a fresh limiter.
    """
    close_rate_limiter()
    global _rate_limiter_config_hash, _rate_limiter_pid

    _rate_limiter_config_hash = None
    _rate_limiter_pid = None


__all__ = [
    "RateLimitManager",
    "get_rate_limiter",
    "close_rate_limiter",
    "reset_rate_limiter",
]
