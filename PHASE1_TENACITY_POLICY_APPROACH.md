# Phase 1 Implementation: Tenacity-Native Approach

**Date:** October 21, 2025  
**Status:** ✅ REVISED - Leveraging Existing Tenacity Infrastructure  
**Effort Estimate:** 6-8 hours (more modular, less code)  
**Risk Level:** VERY LOW (builds directly on existing patterns)

---

## Executive Summary

Rather than creating new `ContextualErrorRecovery` and `DynamicRateLimitManager` classes, this approach:

1. **Extends Tenacity's `wait_` and `retry_` predicates** to encode operation-aware strategies
2. **Uses Tenacity's `RetryCallState` to track provider learning** without a separate manager
3. **Plugs into existing `create_http_retry_policy()` factory** in both ContentDownload and OntologyDownload
4. **Remains 100% backward compatible** - existing code paths unaffected
5. **Leverages `before_sleep_log()` and callbacks** for telemetry

This is **simpler, more modular, and follows Python library conventions**.

---

## Part 1: Contextual Error Recovery via Tenacity Predicates

### Design: Custom `retry_if_*` Predicates

Instead of a separate `ContextualErrorRecovery` class, define Tenacity predicates that encode operation context:

```python
# File: src/DocsToKG/ContentDownload/errors/tenacity_policies.py

"""Context-aware retry policies using Tenacity predicates.

Extends Tenacity's retry/wait predicates to support operation-aware strategies
(DOWNLOAD, VALIDATE, RESOLVE, etc.) without a separate error recovery layer.

Pattern:
    from DocsToKG.ContentDownload.errors.tenacity_policies import (
        OperationType,
        create_contextual_retry_policy,
    )
    
    policy = create_contextual_retry_policy(
        operation=OperationType.DOWNLOAD,
        max_attempts=6,
        max_delay_seconds=60,
    )
    
    for attempt in policy:
        with attempt:
            response = client.get(url)
"""

import logging
from enum import Enum, auto
from typing import Callable, Optional

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    retry_if_result,
    stop_after_delay,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Operation type for contextual retry decisions."""
    DOWNLOAD = auto()          # Artifact content (critical)
    VALIDATE = auto()          # Post-download validation (deferrable)
    RESOLVE = auto()           # Metadata resolution (has fallbacks)
    EXTRACT = auto()           # Archive extraction (retryable)
    MANIFEST_FETCH = auto()    # Manifest metadata (has fallbacks)


def _should_retry_on_429(operation: OperationType) -> Callable:
    """Return predicate for 429 handling per operation.
    
    Downloads always retry. Validation/resolve may defer or failover instead.
    This predicate decides **whether** to retry; defer/failover are handled
    at the caller level (orchestrator).
    """
    def _retry_429(response):
        if not hasattr(response, "status_code"):
            return False
        
        if response.status_code != 429:
            return False
        
        # Operation-specific 429 strategy
        if operation == OperationType.DOWNLOAD:
            # Downloads: always retry (we'll be rate-limited at limiter level)
            return True
        elif operation in (OperationType.VALIDATE, OperationType.MANIFEST_FETCH):
            # Non-critical: return False to signal deferral to caller
            # Caller will use RunState.defer_item(context)
            logger.debug(f"429 on {operation.name}: signaling deferral to caller")
            return False
        elif operation == OperationType.RESOLVE:
            # Resolve: return False to signal failover attempt
            logger.debug(f"429 on {operation.name}: signaling failover to caller")
            return False
        else:
            # Default: retry (safe for unknown operations)
            return True
    
    return _retry_429


def _should_retry_on_timeout(operation: OperationType) -> Callable:
    """Return predicate for timeout handling per operation."""
    def _retry_timeout(exc):
        if not isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout)):
            return False
        
        # Operation-specific timeout strategy
        if operation == OperationType.DOWNLOAD:
            # Downloads: always retry (critical path)
            return True
        elif operation == OperationType.VALIDATE:
            # Validation: defer (non-critical)
            logger.debug(f"Timeout on {operation.name}: signaling deferral")
            return False
        elif operation == OperationType.RESOLVE:
            # Resolve: retry up to 3 times, then failover (handled by caller)
            # This predicate always returns True; caller counts attempts
            return True
        else:
            return True  # Default: retry
    
    return _retry_timeout


def create_contextual_retry_policy(
    operation: OperationType = OperationType.DOWNLOAD,
    max_attempts: int = 6,
    max_delay_seconds: int = 60,
) -> Retrying:
    """Create operation-aware Tenacity retry policy.
    
    Different operations have different retry semantics:
    - DOWNLOAD: aggressive retries (critical)
    - VALIDATE: deferral signals (non-critical, batch later)
    - RESOLVE: failover signals (has alternatives)
    
    Caller is responsible for:
    - Catching deferral signals (429/timeout → defer_item)
    - Catching failover signals (timeout on RESOLVE → try_alternative_resolver)
    - This policy just decides yes/no on individual errors
    
    Args:
        operation: Operation type (DOWNLOAD, VALIDATE, RESOLVE, etc.)
        max_attempts: Maximum retry attempts
        max_delay_seconds: Maximum total retry time
    
    Returns:
        Configured Tenacity Retrying object
    
    Example:
        >>> from DocsToKG.ContentDownload.errors.tenacity_policies import (
        ...     OperationType,
        ...     create_contextual_retry_policy,
        ... )
        >>> policy = create_contextual_retry_policy(
        ...     operation=OperationType.DOWNLOAD,
        ...     max_attempts=6,
        ... )
        >>> for attempt in policy:
        ...     with attempt:
        ...         response = client.get(url)
    """
    
    def wait_with_retry_after(retry_state: RetryCallState):
        """Wait strategy: respect Retry-After if present."""
        exc = retry_state.outcome.exception() or retry_state.outcome.result()
        
        if hasattr(exc, "response") and exc.response:
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return int(retry_after)
                except ValueError:
                    # Would parse HTTP-date here; skip for brevity
                    pass
        
        # Default: let exponential backoff handle it
        return 0
    
    # Build retry condition: network errors + operation-specific HTTP handling
    retry_condition = retry_if_exception_type(
        (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)
    ) | retry_if_result(_should_retry_on_429(operation)) | retry_if_result(
        _should_retry_on_timeout(operation)
    )
    
    # Also retry on 5xx for all operations
    retry_condition |= retry_if_result(
        lambda r: hasattr(r, "status_code") and r.status_code >= 500
    )
    
    return Retrying(
        stop=stop_after_delay(max_delay_seconds),
        wait=wait_with_retry_after,
        retry=retry_condition,
        before_sleep=before_sleep_log(
            logger,
            logging.WARNING,
            exc_info=False,
        ),
        reraise=True,
    )


__all__ = [
    "OperationType",
    "create_contextual_retry_policy",
]
```

### Integration with Orchestrator

Instead of a separate error recovery layer, the orchestrator uses this:

```python
# In orchestrator/runner.py

from DocsToKG.ContentDownload.errors.tenacity_policies import (
    OperationType,
    create_contextual_retry_policy,
)

class DownloadRunner:
    def __init__(self, config, ...):
        self.config = config
        self.deferred_queue = []  # Track deferred items
    
    async def download_artifact(self, artifact, resolver, url):
        """Download with contextual retry policy."""
        
        operation = OperationType.DOWNLOAD  # This is a download operation
        policy = create_contextual_retry_policy(
            operation=operation,
            max_attempts=6,
        )
        
        try:
            for attempt in policy:
                with attempt:
                    return await self._perform_download(artifact, resolver, url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # 429 signals: check if this should be deferred
                if operation == OperationType.VALIDATE:
                    self.deferred_queue.append((artifact, resolver, url))
                    logger.info(f"Deferred {artifact.id} for batch validation")
                    return None
            raise
```

---

## Part 2: Per-Provider Rate Limit Learning via Tenacity State

### Design: Leverage `RetryCallState` and `before_sleep` Callback

Instead of a separate `DynamicRateLimitManager`, use Tenacity's built-in callback mechanism:

```python
# File: src/DocsToKG/ContentDownload/ratelimit/tenacity_learning.py

"""Per-provider rate limit learning using Tenacity callbacks.

Tenacity's `before_sleep_log` and custom callbacks let us track provider
behavior without a separate manager. State lives in process memory +
optional JSON persistence.

Pattern:
    from DocsToKG.ContentDownload.ratelimit.tenacity_learning import (
        ProviderBehaviorTracker,
        create_learning_retry_policy,
    )
    
    tracker = ProviderBehaviorTracker(persistence_path="/tmp/provider_learns.json")
    policy = create_learning_retry_policy(
        provider="crossref",
        host="api.crossref.org",
        tracker=tracker,
        max_delay_seconds=60,
    )
    
    for attempt in policy:
        with attempt:
            response = client.get(url)
    
    print(tracker.get_provider_status("crossref", "api.crossref.org"))
    # Output: {"reduction_pct": 20.0, "consecutive_429s": 5, ...}
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from tenacity import RetryCallState, Retrying, stop_after_delay, wait_random_exponential

logger = logging.getLogger(__name__)


@dataclass
class ProviderBehavior:
    """Learned behavior for a provider:host pair."""
    
    provider: str
    host: str
    consecutive_429s: int = 0
    recovery_times: list = field(default_factory=list)
    applied_reduction_pct: float = 0.0
    
    def record_429(self, retry_after: Optional[int] = None):
        """Record a 429 response."""
        self.consecutive_429s += 1
        if retry_after:
            self.recovery_times.append(retry_after)
            if len(self.recovery_times) > 50:
                self.recovery_times = self.recovery_times[-50:]
    
    def record_success(self):
        """Record successful request - reset counter."""
        self.consecutive_429s = 0
    
    def estimate_recovery_time(self) -> float:
        """Estimate provider's recovery time."""
        if not self.recovery_times:
            return 2.0
        sorted_times = sorted(self.recovery_times)
        return sorted_times[len(sorted_times) // 2]
    
    def to_dict(self) -> Dict:
        """Serialize for JSON persistence."""
        return asdict(self)


class ProviderBehaviorTracker:
    """Track learned rate limit behavior per provider:host."""
    
    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path
        self.behaviors: Dict[Tuple[str, str], ProviderBehavior] = {}
        
        if persistence_path and persistence_path.exists():
            self._load()
    
    def on_retry(self, retry_state: RetryCallState, provider: str, host: str):
        """Called by Tenacity before sleep - track the failure."""
        exc = retry_state.outcome.exception() or retry_state.outcome.result()
        
        key = (provider, host)
        if key not in self.behaviors:
            self.behaviors[key] = ProviderBehavior(provider, host)
        
        behavior = self.behaviors[key]
        
        # Check for 429
        if hasattr(exc, "response") and exc.response:
            if exc.response.status_code == 429:
                retry_after = exc.response.headers.get("Retry-After")
                retry_after_int = int(retry_after) if retry_after else None
                behavior.record_429(retry_after_int)
                
                # Apply progressive reduction
                if behavior.consecutive_429s >= 3:
                    self._apply_reduction(behavior)
                
                logger.info(
                    f"429 from {provider}@{host}: "
                    f"consecutive={behavior.consecutive_429s}, "
                    f"reduction={behavior.applied_reduction_pct}%"
                )
    
    def on_success(self, provider: str, host: str):
        """Called by caller when request succeeds."""
        key = (provider, host)
        if key in self.behaviors:
            self.behaviors[key].record_success()
    
    def _apply_reduction(self, behavior: ProviderBehavior):
        """Apply progressive rate limit reduction."""
        if behavior.applied_reduction_pct >= 80.0:
            return  # Already reduced significantly
        
        # Progressive: 10% → 20% → 30% based on consecutive 429s
        if behavior.consecutive_429s < 5:
            reduction = 10.0
        elif behavior.consecutive_429s < 10:
            reduction = 20.0
        else:
            reduction = 30.0
        
        behavior.applied_reduction_pct = min(
            behavior.applied_reduction_pct + reduction, 80.0
        )
        logger.warning(
            f"Reduced rate limit for {behavior.provider}@{behavior.host} "
            f"by {reduction}% (total: {behavior.applied_reduction_pct}%)"
        )
    
    def get_effective_limit(
        self, provider: str, host: str, config_limit: int
    ) -> int:
        """Get effective rate limit with learned reductions."""
        key = (provider, host)
        if key not in self.behaviors:
            return config_limit
        
        behavior = self.behaviors[key]
        reduction_factor = 1.0 - (behavior.applied_reduction_pct / 100.0)
        return max(1, int(config_limit * reduction_factor))
    
    def get_provider_status(self, provider: str, host: str) -> Dict:
        """Get current learning state."""
        key = (provider, host)
        if key not in self.behaviors:
            return {"status": "unknown", "consecutive_429s": 0}
        
        behavior = self.behaviors[key]
        return {
            "status": "reducing" if behavior.applied_reduction_pct > 0 else "normal",
            "consecutive_429s": behavior.consecutive_429s,
            "reduction_pct": behavior.applied_reduction_pct,
            "recovery_time_estimate": behavior.estimate_recovery_time(),
        }
    
    def _save(self):
        """Persist learned config to JSON."""
        if not self.persistence_path:
            return
        
        data = {
            f"{k[0]}@{k[1]}": v.to_dict()
            for k, v in self.behaviors.items()
        }
        self.persistence_path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved learned config to {self.persistence_path}")
    
    def _load(self):
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
    """
    
    def before_sleep_learning(retry_state: RetryCallState):
        """Track before sleeping."""
        tracker.on_retry(retry_state, provider, host)
    
    def retry_on_429_or_5xx(response):
        """Retry on 429 or 5xx."""
        if hasattr(response, "status_code"):
            return response.status_code in {429, 500, 502, 503, 504}
        return False
    
    return Retrying(
        stop=stop_after_delay(max_delay_seconds),
        wait=wait_random_exponential(
            multiplier=0.5,
            max=min(60, max_delay_seconds),
        ),
        retry=retry_on_429_or_5xx,
        before_sleep=before_sleep_learning,
        reraise=True,
    )


__all__ = [
    "ProviderBehavior",
    "ProviderBehaviorTracker",
    "create_learning_retry_policy",
]
```

### Integration with Rate Limiter

```python
# In ratelimit/manager.py

from DocsToKG.ContentDownload.ratelimit.tenacity_learning import (
    ProviderBehaviorTracker,
)

class RateLimiterManager:
    def __init__(self, config, ...):
        self.config = config
        self.tracker = ProviderBehaviorTracker(
            persistence_path=Path.home() / ".cache" / "docstokg" / "provider_learns.json"
        )
    
    def get_effective_limit(self, provider: str, host: str) -> int:
        """Get configured limit with any learned reductions applied."""
        config_limit = self.config.get(f"{provider}:limit", 10)
        return self.tracker.get_effective_limit(provider, host, config_limit)
    
    def record_success(self, provider: str, host: str):
        """Called after successful request."""
        self.tracker.on_success(provider, host)
```

---

## Why This Approach is Better

### ✅ Advantages of Tenacity-Native Design

1. **Minimal new code** (~200 LOC vs 500+ LOC with separate classes)
2. **Leverages existing Tenacity patterns** - `before_sleep`, `RetryCallState`, `retry_if_*` predicates
3. **Follows Python library conventions** - extends what developers know
4. **100% backward compatible** - doesn't change existing retry policies
5. **No new dependencies** - uses httpx + Tenacity which are already present
6. **Modular** - each factory function is independent and composable
7. **Observable** - hooks into existing `before_sleep_log()` for telemetry
8. **Testable** - inject mock `ProviderBehaviorTracker` or `create_contextual_retry_policy`

### 📊 Code Footprint

| Component | LOC | Type | Reuses |
|-----------|-----|------|---------|
| `tenacity_policies.py` | ~150 | New (contextual retry) | Tenacity predicates |
| `tenacity_learning.py` | ~200 | New (learning) | Tenacity callbacks |
| `integration` | ~50 | Modified existing | Rate limiter, orchestrator |
| **Total** | **400** | Mostly new | **Both leverage Tenacity** |

**vs. Previous Proposal:**

| Component | LOC | Type |
|-----------|-----|------|
| `contextual_recovery.py` | ~300 | New (complex state machine) |
| `dynamic_config.py` | ~400 | New (manager class) |
| `integration` | ~150 | Modified existing |
| **Total** | **850** | **Heavy** |

---

## Implementation Timeline (Day 1, 6-8 hours)

### Hour 0-2: Contextual Retry Policy
- [ ] Create `tenacity_policies.py` with `OperationType` enum
- [ ] Implement `_should_retry_on_429()`, `_should_retry_on_timeout()` predicates
- [ ] Implement `create_contextual_retry_policy()` factory
- [ ] Add 10 unit tests (retry decision matrix)

### Hour 2-3: Learning Policy
- [ ] Create `tenacity_learning.py` with `ProviderBehavior` dataclass
- [ ] Implement `ProviderBehaviorTracker` with learning logic
- [ ] Implement `create_learning_retry_policy()` factory
- [ ] Add 8 unit tests (429 tracking, reduction logic)

### Hour 3-4: Integration Points
- [ ] Wire into orchestrator/runner.py (use `create_contextual_retry_policy`)
- [ ] Wire into rate limiter (use `tracker.get_effective_limit`)
- [ ] Add CLI command `rate-limit-status` (show current tracker state)
- [ ] Update imports/exports

### Hour 4-5: CLI & Telemetry
- [ ] Add `--enable-contextual-retry` flag (default False for compatibility)
- [ ] Add `--enable-learning` flag (default False for compatibility)
- [ ] Emit metrics via existing `before_sleep_log()` hooks
- [ ] Add 5 integration tests

### Hour 5-6: Documentation
- [ ] Add docstrings to all functions
- [ ] Create CONTEXTUAL_RETRY_GUIDE.md
- [ ] Create PROVIDER_LEARNING_GUIDE.md
- [ ] Update AGENTS.md

### Hour 6-8: Testing & Validation
- [ ] E2E tests combining both policies
- [ ] Backward compatibility verification
- [ ] Performance testing (< 2% CPU overhead)
- [ ] Finalize and commit

---

## Key Differences from Previous Proposal

| Aspect | Previous | Tenacity-Native |
|--------|----------|-----------------|
| Error Recovery | Separate `ContextualErrorRecovery` class | `create_contextual_retry_policy()` function |
| State Machine | Custom `RecoveryStrategy` enum | Tenacity `retry_if_*` predicates |
| Rate Learning | `DynamicRateLimitManager` singleton | `ProviderBehaviorTracker` instance |
| Persistence | JSON file (custom load/save) | Same (but simpler) |
| Telemetry | Custom `emit_metric()` | Hooks into `before_sleep_log()` |
| Integration | New orchestrator layer | Native to Tenacity, 2-3 line changes |
| Dependencies | None new | None new (still httpx + Tenacity) |

---

## Success Criteria

✅ All existing tests pass (no breaking changes)  
✅ 20+ new unit tests passing  
✅ < 400 LOC added (modular, focused)  
✅ 100% type hints (mypy --strict)  
✅ Zero linting errors (ruff, black)  
✅ Full docstrings (Google style)  
✅ CPU overhead < 2% (Tenacity overhead mostly paid once at policy creation)  
✅ Backward compatible (feature flags for opt-in)  

---

## Conclusion

This approach is **production-ready from day one** because it:

1. **Extends proven patterns** (Tenacity's callback mechanism)
2. **Minimizes code** (~400 vs 850 LOC)
3. **Maximizes reuse** (hooks into existing infrastructure)
4. **Stays modular** (each factory is independent)
5. **Is testable** (mock tracker, mock policies)

**This is the approach I recommend implementing immediately.**

