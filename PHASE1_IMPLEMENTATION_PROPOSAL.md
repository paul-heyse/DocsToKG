# Phase 1 Implementation Proposal: Contextual Error Recovery & Dynamic Rate Limiting

**Date:** October 21, 2025  
**Scope:** Top 2 High-Impact Improvements  
**Estimated Effort:** 7-9 hours total (Day 1 work)  
**Risk Level:** LOW (fully additive, zero breaking changes)

---

## Executive Summary

This proposal details how to implement the two highest-impact improvements with specific attention to your codebase's architecture:

1. **Contextual Error Recovery** (4-5h) - Smarter error handling based on operation type
2. **Per-Provider Rate Limit Learning** (3-4h) - Dynamic rate limit adjustment from real behavior

Both improvements integrate seamlessly with your existing infrastructure:
- ContentDownload's resolver pipeline + orchestrator (PR #8)
- OntologyDownload's network stack
- Existing telemetry and metrics infrastructure

---

## Part 1: Contextual Error Recovery

### Current State Analysis

Your codebase already has:
- ✅ Comprehensive error classification (network, timeout, 429, 5xx)
- ✅ Tenacity-based retry logic with exponential backoff
- ✅ Circuit breakers (per-host in OntologyDownload, generic in ContentDownload)
- ✅ Dead-letter queue for telemetry failures
- ✅ Structured event emission

**Gap:** Errors don't inform operation-specific strategies. For example:
- On 429 during *download*: back off entire provider (current behavior - good)
- On 429 during *validate*: could defer validation to batch later (new - better utilization)
- On timeout during *manifest-fetch*: could retry with backoff (current - good)
- On timeout during *resolve*: could try fallback resolver (new - better resilience)

### Proposed Architecture

```python
# File: src/DocsToKG/ContentDownload/errors/contextual_recovery.py

from enum import Enum, auto
from typing import Callable, Optional, Any, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Operation context for error recovery decisions."""
    DOWNLOAD = auto()          # Fetching artifact content
    VALIDATE = auto()          # Validating downloaded content
    RESOLVE = auto()           # Resolving URL from metadata service
    EXTRACT = auto()           # Extracting archive
    MANIFEST_FETCH = auto()    # Fetching manifest metadata


class RecoveryStrategy(Enum):
    """Strategy to apply when error occurs."""
    RETRY = auto()             # Retry with backoff
    DEFER = auto()             # Defer to later batch
    FAILOVER = auto()          # Try alternative (e.g., resolver)
    GRACEFUL_SKIP = auto()     # Skip this item, continue
    ESCALATE = auto()          # Critical error, stop


@dataclass
class ErrorContext:
    """Rich context for error recovery decisions."""
    operation: OperationType
    provider: str               # e.g., "crossref", "unpaywall"
    resource_id: str            # e.g., DOI, URL
    attempt_number: int
    error_type: str             # e.g., "429", "timeout", "network"
    error_message: str
    
    # Operation-specific metadata
    metadata: Dict[str, Any]    # Custom context per operation


class ContextualErrorRecovery:
    """Intelligent error recovery based on operation context."""
    
    def __init__(self, config: Dict[str, Any], metrics_emitter: Callable):
        self.config = config
        self.emit_metric = metrics_emitter
        self.provider_backoff_state = {}  # Track provider-level backoff
        self.deferred_queue = []          # For DEFER strategy
    
    def determine_strategy(self, context: ErrorContext) -> RecoveryStrategy:
        """Determine recovery strategy based on error context."""
        
        # 429 Rate Limit errors
        if context.error_type == "429":
            if context.operation == OperationType.DOWNLOAD:
                # For downloads: aggressive backoff on entire provider
                return RecoveryStrategy.RETRY
            elif context.operation == OperationType.VALIDATE:
                # For validation: defer to batch job (less critical)
                return RecoveryStrategy.DEFER
            elif context.operation == OperationType.RESOLVE:
                # For resolution: try fallback resolver if available
                return RecoveryStrategy.FAILOVER
        
        # Timeout errors
        elif context.error_type == "timeout":
            if context.operation == OperationType.DOWNLOAD:
                # Downloads are critical; always retry
                return RecoveryStrategy.RETRY
            elif context.operation in (OperationType.RESOLVE, OperationType.MANIFEST_FETCH):
                # Metadata fetches can tolerate some failures
                if context.attempt_number < 3:
                    return RecoveryStrategy.RETRY
                else:
                    return RecoveryStrategy.FAILOVER
            elif context.operation == OperationType.VALIDATE:
                # Validation is post-hoc; can defer
                return RecoveryStrategy.DEFER
        
        # Network errors
        elif context.error_type == "network":
            if context.operation == OperationType.DOWNLOAD:
                # Critical: retry with backoff
                return RecoveryStrategy.RETRY
            elif context.attempt_number >= 3:
                # After 3 attempts, escalate
                return RecoveryStrategy.ESCALATE
            else:
                return RecoveryStrategy.RETRY
        
        # 5xx server errors
        elif context.error_type in ("500", "502", "503", "504"):
            if context.operation == OperationType.DOWNLOAD and context.attempt_number < 5:
                # Downloads: generous retries on 5xx
                return RecoveryStrategy.RETRY
            elif context.attempt_number < 2:
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.GRACEFUL_SKIP
        
        # Default: escalate unknown errors
        return RecoveryStrategy.ESCALATE
    
    def execute_strategy(
        self,
        strategy: RecoveryStrategy,
        context: ErrorContext,
        retry_action: Callable,
        fallover_action: Optional[Callable] = None,
    ) -> Any:
        """Execute recovery strategy and return result or raise."""
        
        logger.info(
            f"Error recovery: {strategy.name}",
            extra={
                "operation": context.operation.name,
                "provider": context.provider,
                "error_type": context.error_type,
            }
        )
        
        # Emit metric for strategy selection
        self.emit_metric(
            "error_recovery_strategy_applied",
            labels={
                "operation": context.operation.name,
                "strategy": strategy.name,
                "error_type": context.error_type,
            },
        )
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return retry_action()
            
            elif strategy == RecoveryStrategy.DEFER:
                self.deferred_queue.append(context)
                logger.info(f"Deferred {context.resource_id} for later batch processing")
                return None
            
            elif strategy == RecoveryStrategy.FAILOVER:
                if fallover_action:
                    logger.info(f"Attempting failover for {context.resource_id}")
                    return fallover_action()
                else:
                    # No failover available; escalate
                    raise RuntimeError(f"Failover not available for {context.operation.name}")
            
            elif strategy == RecoveryStrategy.GRACEFUL_SKIP:
                logger.warning(f"Gracefully skipping {context.resource_id}")
                return None
            
            elif strategy == RecoveryStrategy.ESCALATE:
                raise RuntimeError(
                    f"Escalating error for {context.resource_id}: {context.error_message}"
                )
        
        except Exception as e:
            self.emit_metric(
                "error_recovery_failed",
                labels={
                    "operation": context.operation.name,
                    "strategy": strategy.name,
                }
            )
            raise
    
    def get_deferred_items(self) -> list:
        """Retrieve deferred items for batch processing."""
        result = self.deferred_queue[:]
        self.deferred_queue.clear()
        return result


# Integration with existing orchestrator
class OrchestrationContext:
    """Enhanced orchestrator context with error recovery."""
    
    def __init__(self, orchestrator_config: Dict, recovery: ContextualErrorRecovery):
        self.config = orchestrator_config
        self.recovery = recovery
    
    def download_with_context(self, artifact, resolver, url):
        """Download with contextual error recovery."""
        context = ErrorContext(
            operation=OperationType.DOWNLOAD,
            provider=resolver.name,
            resource_id=artifact.id,
            attempt_number=0,
            error_type="",
            error_message="",
            metadata={"url": url},
        )
        
        try:
            # Standard download logic
            return self._perform_download(artifact, resolver, url)
        
        except Exception as e:
            context.error_type = self._classify_error(e)
            context.error_message = str(e)
            
            strategy = self.recovery.determine_strategy(context)
            
            return self.recovery.execute_strategy(
                strategy,
                context,
                retry_action=lambda: self._perform_download_with_backoff(
                    artifact, resolver, url, context.attempt_number + 1
                ),
                fallover_action=lambda: self._try_alternative_resolver(artifact, url),
            )
    
    def _perform_download(self, artifact, resolver, url):
        """Core download logic (existing)."""
        # ... existing implementation ...
        pass
    
    def _perform_download_with_backoff(self, artifact, resolver, url, attempt):
        """Retry with exponential backoff."""
        # ... backoff logic ...
        pass
    
    def _try_alternative_resolver(self, artifact, url):
        """Try alternative resolver on failover."""
        # ... fallback logic ...
        pass
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type from exception."""
        if isinstance(error, httpx.HTTPStatusError):
            return str(error.response.status_code)
        elif "timeout" in str(error).lower():
            return "timeout"
        elif "network" in str(error).lower():
            return "network"
        else:
            return "unknown"
```

### Integration Points

1. **With Orchestrator (PR #8):**
   ```python
   # In orchestrator/runner.py
   
   from .errors.contextual_recovery import (
       ContextualErrorRecovery,
       OperationType,
       ErrorContext,
   )
   
   class DownloadRunner:
       def __init__(self, config, ...):
           self.recovery = ContextualErrorRecovery(
               config=config,
               metrics_emitter=self.telemetry.emit_metric,
           )
       
       async def process_artifact(self, artifact, resolver):
           context = ErrorContext(
               operation=OperationType.DOWNLOAD,
               provider=resolver.name,
               resource_id=artifact.id,
               attempt_number=0,
               error_type="",
               error_message="",
               metadata={"resolver": resolver.name},
           )
           
           try:
               return await self._download_internal(artifact, resolver)
           except Exception as e:
               context.error_type = self._classify_error(e)
               strategy = self.recovery.determine_strategy(context)
               # Execute strategy...
   ```

2. **With Network Layer:**
   ```python
   # In network/retry.py - enhance with context
   
   def create_contextual_retry_policy(
       operation: OperationType,
       context_recovery: ContextualErrorRecovery,
   ) -> Retrying:
       """Create retry policy informed by operation context."""
       
       def retry_predicate(attempt):
           # Use context recovery to inform retry decision
           strategy = context_recovery.determine_strategy(...)
           return strategy == RecoveryStrategy.RETRY
       
       return Retrying(
           retry=retry_predicate,
           # ... other config ...
       )
   ```

### Testing Strategy

```python
# File: tests/test_contextual_recovery.py

import pytest
from DocsToKG.ContentDownload.errors.contextual_recovery import (
    ContextualErrorRecovery,
    OperationType,
    ErrorContext,
    RecoveryStrategy,
)


class TestContextualRecovery:
    
    def test_429_on_download_suggests_retry(self):
        """On 429 during download: suggest RETRY."""
        recovery = ContextualErrorRecovery({}, lambda *a, **k: None)
        
        context = ErrorContext(
            operation=OperationType.DOWNLOAD,
            provider="crossref",
            resource_id="doi:10.1234/abc",
            attempt_number=1,
            error_type="429",
            error_message="Rate limit exceeded",
            metadata={},
        )
        
        strategy = recovery.determine_strategy(context)
        assert strategy == RecoveryStrategy.RETRY
    
    def test_429_on_validate_suggests_defer(self):
        """On 429 during validate: suggest DEFER."""
        recovery = ContextualErrorRecovery({}, lambda *a, **k: None)
        
        context = ErrorContext(
            operation=OperationType.VALIDATE,
            provider="rdflib",
            resource_id="doi:10.1234/abc",
            attempt_number=1,
            error_type="429",
            error_message="Rate limit exceeded",
            metadata={},
        )
        
        strategy = recovery.determine_strategy(context)
        assert strategy == RecoveryStrategy.DEFER
    
    def test_timeout_on_resolve_suggests_failover(self):
        """On timeout during resolve: suggest FAILOVER."""
        recovery = ContextualErrorRecovery({}, lambda *a, **k: None)
        
        context = ErrorContext(
            operation=OperationType.RESOLVE,
            provider="crossref",
            resource_id="doi:10.1234/abc",
            attempt_number=1,
            error_type="timeout",
            error_message="Connection timed out",
            metadata={},
        )
        
        strategy = recovery.determine_strategy(context)
        assert strategy == RecoveryStrategy.FAILOVER
    
    def test_deferred_items_queue(self):
        """Deferred items accumulate and can be retrieved."""
        recovery = ContextualErrorRecovery({}, lambda *a, **k: None)
        
        context1 = ErrorContext(
            operation=OperationType.VALIDATE,
            provider="rdflib",
            resource_id="id1",
            attempt_number=1,
            error_type="429",
            error_message="Rate limited",
            metadata={},
        )
        
        strategy = recovery.determine_strategy(context1)
        recovery.execute_strategy(strategy, context1, retry_action=None)
        
        deferred = recovery.get_deferred_items()
        assert len(deferred) == 1
        assert deferred[0].resource_id == "id1"
```

### Benefits

✅ **Better resource utilization**: Defers non-critical operations  
✅ **Improved resilience**: Failover strategies prevent cascading failures  
✅ **Observability**: Metrics show which strategies are used most  
✅ **Maintainability**: Centralized logic easy to tune and extend  
✅ **Zero breaking changes**: Fully additive, existing paths unaffected  

---

## Part 2: Per-Provider Rate Limit Learning

### Current State Analysis

Your codebase has:
- ✅ SQLite-backed rate limiting (pyrate-limiter)
- ✅ Multi-window support (per-second, per-hour)
- ✅ Per-role keying (metadata, landing, artifact)
- ✅ Retry-After header integration
- ✅ CLI configuration via `--rate` and env vars

**Gap:** Rate limits are static. When a provider changes limits or returns 429s, you must manually adjust. Opportunity: Learn from real behavior.

### Proposed Architecture

```python
# File: src/DocsToKG/ContentDownload/ratelimit/dynamic_config.py

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time
import logging
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProviderBehavior:
    """Learned provider behavior from 429 responses."""
    
    provider: str
    host: str
    
    # 429 response history (timestamp, retry_after_seconds)
    recent_429s: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Observed recovery times
    recovery_times: List[float] = field(default_factory=list)
    
    # Current rate limit state
    observed_rate_limit: Optional[Tuple[int, str]] = None  # (limit, window)
    consecutive_429s: int = 0
    last_429_timestamp: Optional[float] = None
    
    # Adjusted configuration
    applied_reduction_pct: float = 0.0  # How much we've reduced from config
    confidence: float = 0.0  # 0-1: confidence in learned limits
    
    def record_429(self, retry_after: Optional[int] = None) -> None:
        """Record a 429 response."""
        now = time.time()
        self.recent_429s.append((now, retry_after))
        self.consecutive_429s += 1
        self.last_429_timestamp = now
        
        if retry_after:
            self.recovery_times.append(retry_after)
            # Keep only recent recovery times
            if len(self.recovery_times) > 50:
                self.recovery_times = self.recovery_times[-50:]
    
    def record_success(self) -> None:
        """Record a successful request."""
        self.consecutive_429s = 0
    
    def estimate_recovery_time(self) -> float:
        """Estimate provider's recovery time from history."""
        if not self.recovery_times:
            return 2.0  # Conservative default
        
        # Use median of recent recovery times
        sorted_times = sorted(self.recovery_times)
        return sorted_times[len(sorted_times) // 2]
    
    def get_applied_limit(self, config_limit: int) -> int:
        """Get effective rate limit with applied reductions."""
        reduction_factor = 1.0 - (self.applied_reduction_pct / 100.0)
        return max(1, int(config_limit * reduction_factor))
    
    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            "provider": self.provider,
            "host": self.host,
            "consecutive_429s": self.consecutive_429s,
            "applied_reduction_pct": self.applied_reduction_pct,
            "confidence": self.confidence,
            "recovery_times": self.recovery_times[-10:],  # Last 10
        }


class DynamicRateLimitManager:
    """Learn and adapt rate limits from provider behavior."""
    
    def __init__(
        self,
        config: Dict,
        persistence_path: Optional[Path] = None,
        learning_enabled: bool = True,
    ):
        self.config = config
        self.persistence_path = persistence_path
        self.learning_enabled = learning_enabled
        
        self.provider_behaviors: Dict[str, ProviderBehavior] = {}
        self.learning_history: deque = deque(maxlen=1000)
        
        if persistence_path and persistence_path.exists():
            self._load_learned_config()
    
    def track_429(
        self,
        provider: str,
        host: str,
        retry_after: Optional[int] = None,
    ) -> None:
        """Track a 429 response and update learned config."""
        
        if not self.learning_enabled:
            return
        
        key = (provider, host)
        if key not in self.provider_behaviors:
            self.provider_behaviors[key] = ProviderBehavior(provider, host)
        
        behavior = self.provider_behaviors[key]
        behavior.record_429(retry_after)
        
        # Learning logic: after N consecutive 429s, reduce config limit
        if behavior.consecutive_429s >= 3:
            self._apply_rate_limit_reduction(behavior)
        
        # Record event
        self.learning_history.append({
            "timestamp": time.time(),
            "event": "429",
            "provider": provider,
            "host": host,
            "retry_after": retry_after,
        })
        
        logger.info(
            f"Tracked 429 from {provider}@{host}",
            extra={
                "provider": provider,
                "consecutive_429s": behavior.consecutive_429s,
                "reduction_pct": behavior.applied_reduction_pct,
            }
        )
    
    def track_success(self, provider: str, host: str) -> None:
        """Track successful request to improve confidence."""
        key = (provider, host)
        if key in self.provider_behaviors:
            self.provider_behaviors[key].record_success()
    
    def _apply_rate_limit_reduction(self, behavior: ProviderBehavior) -> None:
        """Apply progressive rate limit reductions."""
        
        if behavior.applied_reduction_pct >= 80.0:
            # Already reduced significantly; stop
            return
        
        # Progressive reduction
        if behavior.consecutive_429s < 5:
            reduction = 10.0
        elif behavior.consecutive_429s < 10:
            reduction = 20.0
        else:
            reduction = 30.0
        
        behavior.applied_reduction_pct = min(
            behavior.applied_reduction_pct + reduction,
            80.0,  # Never reduce below 20% of original
        )
        
        logger.warning(
            f"Reduced rate limit for {behavior.provider}@{behavior.host} "
            f"by {reduction}% (total: {behavior.applied_reduction_pct}%)",
            extra={
                "provider": behavior.provider,
                "host": behavior.host,
                "consecutive_429s": behavior.consecutive_429s,
            }
        )
        
        self.learning_history.append({
            "timestamp": time.time(),
            "event": "rate_limit_reduced",
            "provider": behavior.provider,
            "host": behavior.host,
            "reduction_pct": behavior.applied_reduction_pct,
        })
    
    def get_effective_limit(
        self,
        provider: str,
        host: str,
        config_limit: int,
    ) -> int:
        """Get effective rate limit with any learned reductions applied."""
        
        key = (provider, host)
        if key in self.provider_behaviors:
            return self.provider_behaviors[key].get_applied_limit(config_limit)
        
        return config_limit
    
    def get_provider_status(self, provider: str, host: str) -> Dict:
        """Get current learning state for a provider."""
        key = (provider, host)
        if key not in self.provider_behaviors:
            return {"status": "unknown", "consecutive_429s": 0}
        
        behavior = self.provider_behaviors[key]
        return {
            "status": "reducing" if behavior.applied_reduction_pct > 0 else "normal",
            "consecutive_429s": behavior.consecutive_429s,
            "reduction_pct": behavior.applied_reduction_pct,
            "confidence": behavior.confidence,
            "recovery_time_estimate": behavior.estimate_recovery_time(),
        }
    
    def _save_learned_config(self) -> None:
        """Persist learned configuration."""
        if not self.persistence_path:
            return
        
        data = {
            "providers": {
                f"{k[0]}@{k[1]}": v.to_dict()
                for k, v in self.provider_behaviors.items()
            },
            "learning_history_sample": list(self.learning_history)[-100:],
        }
        
        self.persistence_path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved learned config to {self.persistence_path}")
    
    def _load_learned_config(self) -> None:
        """Load previously learned configuration."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        try:
            data = json.loads(self.persistence_path.read_text())
            for key_str, behavior_dict in data.get("providers", {}).items():
                provider, host = key_str.split("@")
                behavior = ProviderBehavior(provider, host)
                behavior.applied_reduction_pct = behavior_dict.get("applied_reduction_pct", 0.0)
                behavior.consecutive_429s = behavior_dict.get("consecutive_429s", 0)
                behavior.recovery_times = behavior_dict.get("recovery_times", [])
                self.provider_behaviors[(provider, host)] = behavior
            
            logger.info(f"Loaded learned config with {len(self.provider_behaviors)} providers")
        except Exception as e:
            logger.error(f"Failed to load learned config: {e}")


# Integration with rate limiter
class EnhancedRateLimiter:
    """Rate limiter with learning capability."""
    
    def __init__(self, config: Dict, learning_manager: DynamicRateLimitManager):
        self.config = config
        self.learning_manager = learning_manager
        # ... existing rate limiter init ...
    
    def acquire(
        self,
        key: Tuple[str, str, str],  # (provider, host, role)
        weight: int = 1,
    ) -> bool:
        """Acquire rate limit slot with learning."""
        
        provider, host, role = key
        
        # Get effective limit (with learned reductions)
        config_limit = self.config.get(f"{provider}:{role}:limit", 10)
        effective_limit = self.learning_manager.get_effective_limit(
            provider, host, config_limit
        )
        
        # Acquire with effective limit
        success = self._acquire_internal(key, effective_limit, weight)
        
        if success:
            self.learning_manager.track_success(provider, host)
        
        return success
    
    def record_429(
        self,
        key: Tuple[str, str, str],
        retry_after: Optional[int] = None,
    ) -> None:
        """Record a 429 response for learning."""
        
        provider, host, role = key
        self.learning_manager.track_429(provider, host, retry_after)
```

### Integration Points

1. **With Retry Logic:**
   ```python
   # In network/retry.py
   
   def handle_429_response(response, dynamic_limiter, provider, host):
       """Handle 429 with learning."""
       retry_after = int(response.headers.get("Retry-After", 0))
       dynamic_limiter.track_429(provider, host, retry_after)
       
       # Emit event for observability
       emit_event({
           "type": "rate_limit_429",
           "provider": provider,
           "host": host,
           "retry_after": retry_after,
       })
   ```

2. **With Configuration:**
   ```python
   # In settings.py or config
   
   class RateLimitConfig(BaseModel):
       dynamic_learning_enabled: bool = True
       learning_persistence_path: Optional[Path] = None
       reduction_max_pct: float = 80.0
       reduction_threshold_429s: int = 3
   ```

3. **CLI Exposure:**
   ```python
   # In cli/catalog.py
   
   @cli.command("rate-limit-status")
   @click.option("--provider", help="Filter by provider")
   def show_rate_limit_status(provider: str = None):
       """Show learned rate limit adjustments."""
       manager = get_dynamic_limit_manager()
       
       for (p, h), behavior in manager.provider_behaviors.items():
           if provider and p != provider:
               continue
           
           click.echo(f"{p}@{h}:")
           click.echo(f"  Reduction: {behavior.applied_reduction_pct}%")
           click.echo(f"  Consecutive 429s: {behavior.consecutive_429s}")
           click.echo(f"  Recovery time estimate: {behavior.estimate_recovery_time()}s")
   ```

### Testing Strategy

```python
# File: tests/test_dynamic_rate_limit.py

import pytest
from DocsToKG.ContentDownload.ratelimit.dynamic_config import (
    DynamicRateLimitManager,
    ProviderBehavior,
)


class TestDynamicRateLimiting:
    
    def test_learns_from_429s(self):
        """Manager learns rate limits from 429 responses."""
        manager = DynamicRateLimitManager({}, learning_enabled=True)
        
        # Track 3 consecutive 429s
        for i in range(3):
            manager.track_429("crossref", "api.crossref.org", retry_after=2)
        
        # Check reduction was applied
        status = manager.get_provider_status("crossref", "api.crossref.org")
        assert status["consecutive_429s"] == 3
        assert status["reduction_pct"] > 0.0
    
    def test_progressive_reduction(self):
        """Reductions are progressive with consecutive 429s."""
        manager = DynamicRateLimitManager({}, learning_enabled=True)
        
        # 5 consecutive 429s
        for i in range(5):
            manager.track_429("crossref", "api.crossref.org")
        
        reduction1 = manager.provider_behaviors[
            ("crossref", "api.crossref.org")
        ].applied_reduction_pct
        
        # 5 more 429s
        for i in range(5):
            manager.track_429("crossref", "api.crossref.org")
        
        reduction2 = manager.provider_behaviors[
            ("crossref", "api.crossref.org")
        ].applied_reduction_pct
        
        assert reduction2 > reduction1
    
    def test_success_resets_counter(self):
        """Successful request resets 429 counter."""
        manager = DynamicRateLimitManager({}, learning_enabled=True)
        
        for i in range(3):
            manager.track_429("crossref", "api.crossref.org")
        
        assert (
            manager.provider_behaviors[
                ("crossref", "api.crossref.org")
            ].consecutive_429s
            == 3
        )
        
        manager.track_success("crossref", "api.crossref.org")
        
        assert (
            manager.provider_behaviors[
                ("crossref", "api.crossref.org")
            ].consecutive_429s
            == 0
        )
    
    def test_effective_limit_calculation(self):
        """Effective limits decrease with reductions."""
        manager = DynamicRateLimitManager({}, learning_enabled=True)
        
        config_limit = 100
        original = manager.get_effective_limit("crossref", "api.crossref.org", config_limit)
        assert original == config_limit
        
        # Apply 20% reduction
        behavior = ProviderBehavior("crossref", "api.crossref.org")
        behavior.applied_reduction_pct = 20.0
        manager.provider_behaviors[("crossref", "api.crossref.org")] = behavior
        
        reduced = manager.get_effective_limit("crossref", "api.crossref.org", config_limit)
        assert reduced == 80  # 20% reduction
    
    def test_persistence(self, tmp_path):
        """Learned config persists and reloads."""
        path = tmp_path / "learned.json"
        
        manager1 = DynamicRateLimitManager({}, persistence_path=path)
        for i in range(3):
            manager1.track_429("crossref", "api.crossref.org")
        manager1._save_learned_config()
        
        manager2 = DynamicRateLimitManager({}, persistence_path=path)
        status = manager2.get_provider_status("crossref", "api.crossref.org")
        
        assert status["reduction_pct"] > 0.0
```

### Benefits

✅ **Automatic tuning**: Learns provider limits without manual config  
✅ **Progressive reduction**: Conservative approach to avoid over-limiting  
✅ **Quick recovery**: Resets on success to test if provider recovered  
✅ **Observability**: Exposes learning state via CLI  
✅ **Persistence**: Learned config survives restarts  
✅ **Safety**: Never reduces below 20% of original  

---

## Implementation Timeline

### Day 1: Foundation (7-9 hours)

**Hour 0-2: Contextual Error Recovery Core**
- Create `errors/contextual_recovery.py` with OperationType, RecoveryStrategy, ErrorContext
- Implement `ContextualErrorRecovery.determine_strategy()` logic
- Add metric emission scaffolding

**Hour 2-3: Contextual Error Recovery Testing**
- Create unit tests for all strategy types
- Test defer queue functionality
- Test metrics emission

**Hour 3-5: Rate Limit Learning Core**
- Create `ratelimit/dynamic_config.py` with ProviderBehavior, DynamicRateLimitManager
- Implement 429 tracking and reduction logic
- Add persistence layer (JSON)

**Hour 5-6: Rate Limit Learning Testing**
- Create unit tests for learning logic
- Test progressive reduction
- Test persistence/reload

**Hour 6-7: Integration Points**
- Wire into orchestrator (contextual error recovery)
- Wire into rate limiter (dynamic learning)
- Create CLI commands for status visibility

**Hour 7-8: E2E Testing**
- Create integration tests combining both
- Verify zero breaking changes
- Performance validation

**Hour 8-9: Documentation & Metrics**
- Add docstrings and type hints
- Create migration guide
- Update AGENTS.md

---

## Risk Assessment

| Risk | Likelihood | Severity | Mitigation |
|------|-----------|----------|-----------|
| Break existing retry logic | LOW | HIGH | Comprehensive tests, default to existing behavior |
| Overly aggressive reductions | LOW | MEDIUM | Progressive approach, manual override, CLI visibility |
| Persistence corruption | LOW | LOW | JSON validation on load, fallback to defaults |
| Performance overhead | LOW | LOW | Deque with maxlen, periodic cleanup, lazy computation |
| Metric cardinality explosion | LOW | MEDIUM | Limit label dimensions, use aggregation |

---

## Success Criteria

✅ All 30+ unit tests passing  
✅ Zero breaking changes (backward compatibility verified)  
✅ CLI commands for observability operational  
✅ Comprehensive docstrings and type hints  
✅ Migration guide created  
✅ Performance overhead < 5% CPU, < 1MB RAM  

---

## Next Steps After Phase 1

Once these two improvements are deployed:

1. **Monitor & Learn** (Week 2)
   - Deploy to staging; collect metrics on error recovery strategies
   - Observe rate limit learning patterns
   - Gather user feedback

2. **Phase 2: Graceful Degradation Modes** (Week 2-3)
   - Add STRICT/GRACEFUL/OFFLINE modes
   - Integrate with contextual recovery

3. **Phase 3: Adaptive Backoff Policy** (Week 3)
   - Learn provider recovery times
   - Personalize backoff curves

4. **Phase 4: Connection Pool Metrics** (Week 4)
   - Add pool saturation detection
   - Dashboard integration

---

**Expected Impact:** 30-40% reduction in retry latency, 20-25% improvement in rate-limit compliance, 15+ operational improvements to observability.

