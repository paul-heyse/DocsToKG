# Phase 1 Integration Guide - CLI & Orchestrator Wiring

**Date:** October 21, 2025  
**Status:** Implementation Guide (Foundation Complete)

---

## 🎯 Overview

Phase 1 core implementation (contextual retry, provider learning, feature gates) is **production-ready**. This guide outlines CLI/orchestrator integration points for Hour 4-5.

---

## ✅ What's Already Done

✅ Contextual Retry Policies (`errors/tenacity_policies.py`)
✅ Provider Learning (`ratelimit/tenacity_learning.py`)
✅ Feature Gates Config (`config/models.py` - `FeatureGatesConfig`)
✅ Package Infrastructure (`errors/__init__.py`)

---

## 🔄 Integration Points (Hour 4-5)

### 1. CLI Argument Registration

**Location:** Main CLI entry point (find where global args are registered)

**Add these arguments:**

```python
@app.command()
def download(
    # ... existing arguments ...
    
    # === Phase 1 Feature Gates ===
    enable_contextual_retry: bool = typer.Option(
        False,
        "--enable-contextual-retry",
        help="Enable context-aware retry policies (DOWNLOAD/VALIDATE/RESOLVE)",
    ),
    enable_provider_learning: bool = typer.Option(
        False,
        "--enable-provider-learning",
        help="Enable per-provider rate limit learning with progressive reduction",
    ),
    provider_learning_path: Optional[str] = typer.Option(
        None,
        "--provider-learning-path",
        help="Path to persist provider learning state (JSON file, optional)",
    ),
):
    """Download artifacts with optional Phase 1 optimizations."""
    
    # ... rest of function ...
    config.feature_gates.enable_contextual_retry = enable_contextual_retry
    config.feature_gates.enable_provider_learning = enable_provider_learning
    if provider_learning_path:
        config.feature_gates.provider_learning_path = provider_learning_path
```

### 2. Orchestrator Policy Selection

**Location:** `src/DocsToKG/ContentDownload/orchestrator/runner.py` (or equivalent)

**Add conditional policy selection:**

```python
from DocsToKG.ContentDownload.errors import (
    OperationType,
    create_contextual_retry_policy,
)
from DocsToKG.ContentDownload.ratelimit.tenacity_learning import (
    ProviderBehaviorTracker,
    create_learning_retry_policy,
)

def setup_retry_policy(config: ContentDownloadConfig) -> Retrying:
    """Create retry policy based on feature gates."""
    
    if config.feature_gates.enable_contextual_retry:
        # Use contextual policy (operation = DOWNLOAD for this layer)
        return create_contextual_retry_policy(
            operation=OperationType.DOWNLOAD,
            max_attempts=config.http.retry_policy.max_attempts,
            max_delay_seconds=config.http.retry_policy.max_delay_ms / 1000,
        )
    else:
        # Fall back to existing policy
        return create_http_retry_policy(
            max_attempts=config.http.retry_policy.max_attempts,
            max_delay_ms=config.http.retry_policy.max_delay_ms,
        )

def setup_rate_limiter(config: ContentDownloadConfig) -> RateLimiterManager:
    """Create rate limiter with optional provider learning."""
    
    limiter = RateLimiterManager(config)
    
    if config.feature_gates.enable_provider_learning:
        tracker = ProviderBehaviorTracker(
            persistence_path=(
                Path(config.feature_gates.provider_learning_path)
                if config.feature_gates.provider_learning_path
                else None
            )
        )
        limiter.tracker = tracker
        logger.info("Provider learning enabled")
    
    return limiter
```

### 3. Rate Limiter Integration

**Location:** Rate limiter implementation (attach tracker)

**Add tracker field and hook:**

```python
class RateLimiterManager:
    """Rate limiter with optional provider learning."""
    
    def __init__(self, config: ContentDownloadConfig):
        # ... existing init ...
        self.tracker: Optional[ProviderBehaviorTracker] = None
    
    def get_effective_limit(
        self, 
        provider: str, 
        host: str, 
        config_limit: int
    ) -> int:
        """Get effective limit with learned reductions applied."""
        
        if self.tracker:
            return self.tracker.get_effective_limit(provider, host, config_limit)
        return config_limit
    
    def on_request_success(self, provider: str, host: str) -> None:
        """Called after successful request (for learning)."""
        
        if self.tracker:
            self.tracker.on_success(provider, host)
```

### 4. HTTP Client Retry Hook

**Location:** `networking.request_with_retries()` or retry policy application

**Wire tracker callbacks:**

```python
def request_with_retries(
    url: str,
    method: str = "GET",
    provider: str = "unknown",
    host: str = "unknown",
    **kwargs
) -> httpx.Response:
    """Request with retries and optional provider learning."""
    
    # Determine which policy to use
    policy = setup_retry_policy(config)
    
    # Attach learning callback if enabled
    if config.feature_gates.enable_provider_learning:
        tracker = ProviderBehaviorTracker(...)
        
        def before_sleep_callback(retry_state):
            tracker.on_retry(retry_state, provider, host)
        
        # Attach to policy (depends on implementation)
        policy.before_sleep = before_sleep_callback
    
    try:
        for attempt in policy:
            with attempt:
                response = client.request(method, url, **kwargs)
        
        # Mark success
        if config.feature_gates.enable_provider_learning:
            tracker.on_success(provider, host)
        
        return response
    except Exception as exc:
        # Learning tracker has already recorded failures via before_sleep
        raise
```

---

## 🧪 Integration Tests

### Test 1: Backward Compatibility (Flags OFF)

```python
def test_download_with_flags_off():
    """Verify existing behavior when flags are OFF."""
    config = ContentDownloadConfig()
    assert config.feature_gates.enable_contextual_retry is False
    assert config.feature_gates.enable_provider_learning is False
    
    # Run download - should use existing policy
    # Verify behavior identical to pre-feature-gate version
```

### Test 2: Contextual Retry Enabled

```python
def test_download_with_contextual_retry():
    """Verify contextual retry policy is used when enabled."""
    config = ContentDownloadConfig()
    config.feature_gates.enable_contextual_retry = True
    
    # Mock policy factory to verify it's called
    with patch('orchestrator.setup_retry_policy') as mock_setup:
        # Run download
        # Verify contextual policy was created
        mock_setup.assert_called_once()
        created_policy = mock_setup.return_value
        assert created_policy.operation == OperationType.DOWNLOAD
```

### Test 3: Provider Learning Enabled

```python
def test_download_with_provider_learning():
    """Verify provider learning tracker is attached."""
    config = ContentDownloadConfig()
    config.feature_gates.enable_provider_learning = True
    
    # Run download with multiple 429s
    # Verify tracker records consecutive 429s
    # Verify reduction % increases: 0% → 10% → 20% → 30%
    # Verify limiter.get_effective_limit() returns reduced value
```

### Test 4: Persistence

```python
def test_provider_learning_persistence():
    """Verify learned state survives restart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "provider_learns.json"
        
        # First run: learn behavior
        tracker1 = ProviderBehaviorTracker(persistence_path=path)
        tracker1.on_retry(mock_retry_state_429, "crossref", "api.crossref.org")
        tracker1.on_retry(mock_retry_state_429, "crossref", "api.crossref.org")
        tracker1.on_retry(mock_retry_state_429, "crossref", "api.crossref.org")
        # Should have -10% reduction
        assert tracker1.get_effective_limit("crossref", "api.crossref.org", 10) == 9
        
        # Second run: load and verify persistence
        tracker2 = ProviderBehaviorTracker(persistence_path=path)
        assert tracker2.get_effective_limit("crossref", "api.crossref.org", 10) == 9
```

---

## 🚀 Deployment Checklist

- [ ] CLI args added and wired
- [ ] Policy selection logic in orchestrator
- [ ] Rate limiter integration complete
- [ ] HTTP client retry hook updated
- [ ] All integration tests passing
- [ ] Backward compatibility verified (flags OFF)
- [ ] Feature gates work in isolation
- [ ] Feature gates work together
- [ ] Performance testing (< 2% overhead)
- [ ] Documentation updated

---

## 📊 Success Criteria

- ✅ No changes to existing behavior when flags OFF
- ✅ Contextual retry policy applies when flag ON
- ✅ Provider learning tracks and reduces limits
- ✅ Persistence survives process restart
- ✅ All tests passing (100% existing + new)
- ✅ < 2% performance overhead
- ✅ Zero breaking changes

---

## 🎯 Next Steps

1. **Find main CLI entry point** (search for `@app.command()` or `typer.Typer`)
2. **Add CLI arguments** (3 new options)
3. **Create setup functions** in orchestrator
4. **Wire in HTTP client retry**
5. **Add integration tests**
6. **Run full test suite**
7. **Verify backward compatibility**

---

## 📝 Implementation Effort

- **CLI wiring:** 15 min
- **Orchestrator setup:** 35 min
- **HTTP client hook:** 20 min
- **Integration tests:** 40 min
- **Total Hour 4-5:** ~2 hours

---

## ⚠️ Important Notes

1. **Existing policy fallback:** When flags are OFF, use existing `create_http_retry_policy()` (don't change it)
2. **Operation type context:** Only use `OperationType.DOWNLOAD` at the HTTP client level (critical path)
3. **Tracker lifecycle:** Create tracker once per run, not per-request
4. **Persistence directory:** Create parent directory if it doesn't exist
5. **Error handling:** If learning persistence fails, log warning but continue (non-blocking)

---

## 🔗 Related Files

- Production: `src/DocsToKG/ContentDownload/errors/tenacity_policies.py`
- Production: `src/DocsToKG/ContentDownload/ratelimit/tenacity_learning.py`
- Config: `src/DocsToKG/ContentDownload/config/models.py` (FeatureGatesConfig already added)
- Tests: `tests/content_download/test_contextual_retry_policy.py`

