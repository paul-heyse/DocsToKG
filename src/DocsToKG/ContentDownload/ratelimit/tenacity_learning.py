"""Per-provider rate limit learning using Tenacity callbacks.

Tenacity's `before_sleep` callbacks let us track provider behavior without
a separate manager. State lives in process memory + optional JSON persistence.

This module implements progressive rate limit reduction based on consecutive
429 responses (rate limit errors). When a provider consistently returns 429s,
we progressively reduce the configured rate limit:

- 0-2 consecutive 429s: No change (transient)
- 3-4 consecutive 429s: -10% reduction (early signal)
- 5-9 consecutive 429s: -20% additional (provider congested)
- 10+ consecutive 429s: -30% additional (severe congestion)

When the provider succeeds, we reset the consecutive counter.

Usage:
    from DocsToKG.ContentDownload.ratelimit.tenacity_learning import (
        ProviderBehaviorTracker,
        create_learning_retry_policy,
    )

    tracker = ProviderBehaviorTracker(
        persistence_path=Path.home() / ".cache" / "docstokg" / "provider_learns.json"
    )

    policy = create_learning_retry_policy(
        provider="crossref",
        host="api.crossref.org",
        tracker=tracker,
        max_delay_seconds=60,
    )

    for attempt in policy:
        with attempt:
            response = client.get(url)

    # After success
    tracker.on_success("crossref", "api.crossref.org")

    # Get effective limit
    effective_limit = tracker.get_effective_limit("crossref", "api.crossref.org", 10)
    print(effective_limit)  # May be 7 (30% reduction) if provider congested
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tenacity import RetryCallState, Retrying, stop_after_delay, wait_random_exponential

logger = logging.getLogger(__name__)


@dataclass
class ProviderBehavior:
    """Learned behavior for a provider:host pair.

    Tracks consecutive 429 responses, recovery times (from Retry-After headers),
    and applied rate limit reductions.
    """

    provider: str
    host: str
    consecutive_429s: int = 0
    recovery_times: list = field(default_factory=list)
    applied_reduction_pct: float = 0.0

    def record_429(self, retry_after: Optional[int] = None) -> None:
        """Record a 429 response.

        Args:
            retry_after: Seconds to wait (from Retry-After header, if available)
        """
        self.consecutive_429s += 1
        if retry_after:
            self.recovery_times.append(retry_after)
            # Keep only last 50 recovery times (bounded memory)
            if len(self.recovery_times) > 50:
                self.recovery_times = self.recovery_times[-50:]

    def record_success(self) -> None:
        """Record successful request - reset consecutive 429 counter."""
        self.consecutive_429s = 0

    def estimate_recovery_time(self) -> float:
        """Estimate provider's recovery time from Retry-After samples."""
        if not self.recovery_times:
            return 2.0  # Default: 2 seconds
        sorted_times = sorted(self.recovery_times)
        # Return median recovery time
        return float(sorted_times[len(sorted_times) // 2])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON persistence."""
        return asdict(self)


class ProviderBehaviorTracker:
    """Track learned rate limit behavior per provider:host pair.

    Maintains in-memory state of provider behavior, with optional JSON
    persistence across process restarts.
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        """Initialize tracker.

        Args:
            persistence_path: Optional path to JSON file for persistence
        """
        self.persistence_path = persistence_path
        self.behaviors: Dict[Tuple[str, str], ProviderBehavior] = {}

        if persistence_path and persistence_path.exists():
            self._load()

    def on_retry(
        self,
        retry_state: RetryCallState,
        provider: str,
        host: str,
    ) -> None:
        """Called by Tenacity before sleep - track the failure.

        Args:
            retry_state: Tenacity retry state
            provider: Provider name (e.g., "crossref")
            host: Hostname (e.g., "api.crossref.org")
        """
        exc = retry_state.outcome.exception() or retry_state.outcome.result()

        key = (provider, host)
        if key not in self.behaviors:
            self.behaviors[key] = ProviderBehavior(provider, host)

        behavior = self.behaviors[key]

        # Check for 429
        if hasattr(exc, "response") and exc.response:
            if exc.response.status_code == 429:
                retry_after = exc.response.headers.get("Retry-After")
                retry_after_int: Optional[int] = None
                if retry_after:
                    try:
                        retry_after_int = int(retry_after)
                    except ValueError:
                        pass

                behavior.record_429(retry_after_int)

                # Apply progressive reduction
                if behavior.consecutive_429s >= 3:
                    self._apply_reduction(behavior)

                logger.info(
                    f"429 from {provider}@{host}: "
                    f"consecutive={behavior.consecutive_429s}, "
                    f"reduction={behavior.applied_reduction_pct}%"
                )

    def on_success(self, provider: str, host: str) -> None:
        """Called by caller when request succeeds.

        Args:
            provider: Provider name
            host: Hostname
        """
        key = (provider, host)
        if key in self.behaviors:
            self.behaviors[key].record_success()

    def _apply_reduction(self, behavior: ProviderBehavior) -> None:
        """Apply progressive rate limit reduction.

        Args:
            behavior: ProviderBehavior instance
        """
        if behavior.applied_reduction_pct >= 80.0:
            return  # Already reduced significantly

        # Progressive: 10% → 20% → 30% based on consecutive 429s
        if behavior.consecutive_429s < 5:
            reduction = 10.0
        elif behavior.consecutive_429s < 10:
            reduction = 20.0
        else:
            reduction = 30.0

        old_reduction = behavior.applied_reduction_pct
        behavior.applied_reduction_pct = min(behavior.applied_reduction_pct + reduction, 80.0)

        if behavior.applied_reduction_pct > old_reduction:
            logger.warning(
                f"Reduced rate limit for {behavior.provider}@{behavior.host} "
                f"by {reduction}% (total: {behavior.applied_reduction_pct}%, "
                f"consecutive_429s: {behavior.consecutive_429s})"
            )

    def get_effective_limit(self, provider: str, host: str, config_limit: int) -> int:
        """Get effective rate limit with learned reductions.

        Args:
            provider: Provider name
            host: Hostname
            config_limit: Configured limit (no reduction)

        Returns:
            Effective limit with reductions applied
        """
        key = (provider, host)
        if key not in self.behaviors:
            return config_limit

        behavior = self.behaviors[key]
        reduction_factor = 1.0 - (behavior.applied_reduction_pct / 100.0)
        effective = max(1, int(config_limit * reduction_factor))
        return effective

    def get_provider_status(self, provider: str, host: str) -> Dict[str, Any]:
        """Get current learning state.

        Args:
            provider: Provider name
            host: Hostname

        Returns:
            Dictionary with status info
        """
        key = (provider, host)
        if key not in self.behaviors:
            return {
                "status": "unknown",
                "consecutive_429s": 0,
                "reduction_pct": 0.0,
                "recovery_time_estimate": 2.0,
            }

        behavior = self.behaviors[key]
        return {
            "status": "reducing" if behavior.applied_reduction_pct > 0 else "normal",
            "consecutive_429s": behavior.consecutive_429s,
            "reduction_pct": behavior.applied_reduction_pct,
            "recovery_time_estimate": behavior.estimate_recovery_time(),
        }

    def _save(self) -> None:
        """Persist learned config to JSON."""
        if not self.persistence_path:
            return

        data = {f"{k[0]}@{k[1]}": v.to_dict() for k, v in self.behaviors.items()}
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self.persistence_path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved learned config to {self.persistence_path}")

    def _load(self) -> None:
        """Load persisted learned config."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            data = json.loads(self.persistence_path.read_text())
            for key_str, behavior_dict in data.items():
                provider, host = key_str.split("@")
                b = ProviderBehavior(provider, host)
                b.consecutive_429s = behavior_dict.get("consecutive_429s", 0)
                b.applied_reduction_pct = behavior_dict.get("applied_reduction_pct", 0.0)
                b.recovery_times = behavior_dict.get("recovery_times", [])
                self.behaviors[(provider, host)] = b

            logger.info(f"Loaded learned config for {len(self.behaviors)} providers")
        except Exception as e:
            logger.error(f"Failed to load learned config: {e}")


def create_learning_retry_policy(
    provider: str,
    host: str,
    tracker: ProviderBehaviorTracker,
    max_delay_seconds: int = 60,
) -> Retrying:
    """Create retry policy with integrated provider learning.

    Every retry attempt updates the tracker. Caller is responsible for
    calling `tracker.on_success()` after successful request.

    Args:
        provider: Provider name (e.g., "crossref")
        host: Hostname (e.g., "api.crossref.org")
        tracker: ProviderBehaviorTracker instance
        max_delay_seconds: Max retry duration

    Returns:
        Configured Tenacity Retrying object

    Example:
        >>> from DocsToKG.ContentDownload.ratelimit.tenacity_learning import (
        ...     ProviderBehaviorTracker,
        ...     create_learning_retry_policy,
        ... )
        >>> tracker = ProviderBehaviorTracker()
        >>> policy = create_learning_retry_policy(
        ...     provider="crossref",
        ...     host="api.crossref.org",
        ...     tracker=tracker,
        ... )
        >>> for attempt in policy:
        ...     with attempt:
        ...         response = client.get(url)
        ...     tracker.on_success("crossref", "api.crossref.org")
    """

    def before_sleep_learning(retry_state: RetryCallState) -> None:
        """Track before sleeping."""
        tracker.on_retry(retry_state, provider, host)

    def retry_on_429_or_5xx(response: Any) -> bool:
        """Retry on 429 or 5xx."""
        if hasattr(response, "status_code"):
            return response.status_code in {429, 500, 502, 503, 504}
        return False

    import httpx
    from tenacity import retry_if_exception_type, retry_if_result

    return Retrying(
        stop=stop_after_delay(max_delay_seconds),
        wait=wait_random_exponential(
            multiplier=0.5,
            max=min(60, max_delay_seconds),
        ),
        retry=(
            retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout))
            | retry_if_result(retry_on_429_or_5xx)
        ),
        before_sleep=before_sleep_learning,
        reraise=True,
    )


__all__ = [
    "ProviderBehavior",
    "ProviderBehaviorTracker",
    "create_learning_retry_policy",
]
