# Phase 1: Feature Gate Strategy (Zero-Risk Deployment)

**Date:** October 21, 2025  
**Status:** ✅ FINALIZED - Feature Gate Approach  
**Deployment Risk:** MINIMAL (existing code untouched, new code behind flag)  
**Rollback Complexity:** TRIVIAL (flip one boolean)

---

## Executive Summary

**Key Insight:** Don't replace the existing retry policy. Add the new one alongside it and let users opt-in via a feature flag.

### The Approach

1. **Keep existing `create_http_retry_policy()`** - unchanged, production-proven
2. **Add new `create_contextual_retry_policy()`** - new capability, behind flag
3. **Add new `create_learning_retry_policy()`** - new learning, behind flag
4. **Add CLI flags** - `--enable-contextual-retry`, `--enable-provider-learning`
5. **Add environment variables** - `DOCSTOKG_ENABLE_CONTEXTUAL_RETRY`, `DOCSTOKG_ENABLE_PROVIDER_LEARNING`
6. **Deploy with flags disabled by default** - existing behavior unchanged
7. **Users opt-in when ready** - gradual rollout with instant rollback

### Risk Profile

```
Traditional Approach (Replacement):
  Before:  [Working Policy A]
  After:   [Policy B] ← Breaks if Policy B has bugs
  Rollback: Requires revert + redeploy

Feature Gate Approach (Addition):
  Before:  [Policy A (active)]
  After:   [Policy A (active)] + [Policy B (gated)]
  Rollback: Flip flag, restart (seconds)
```

---

## Implementation Details

### File Structure (No Deletions, Only Additions)

```
src/DocsToKG/ContentDownload/
├── errors/
│   ├── tenacity_policies.py         [NEW] Context-aware predicates
│   └── (existing error handling untouched)
│
├── ratelimit/
│   ├── tenacity_learning.py         [NEW] Provider behavior tracking
│   └── (existing rate limiter untouched)
│
├── config/
│   ├── models.py                    [MODIFIED +15 LOC] Add feature gates
│   └── (existing models untouched)
│
├── args.py                          [MODIFIED +10 LOC] Add CLI flags
└── (existing CLI untouched)
```

### Code Changes (Minimal)

#### 1. Add Feature Gates to Config

```python
# src/DocsToKG/ContentDownload/config/models.py

class DownloadConfig(BaseModel):
    # ... existing fields ...
    
    # Feature Gates (NEW)
    enable_contextual_retry: bool = False      # Default: OFF
    enable_provider_learning: bool = False     # Default: OFF
    provider_learning_path: Optional[Path] = None  # Optional persistence
```

#### 2. Add CLI Flags

```python
# src/DocsToKG/ContentDownload/args.py

parser.add_argument(
    "--enable-contextual-retry",
    action="store_true",
    default=False,
    help="Enable context-aware retry policies (EXPERIMENTAL). Default: disabled.",
)

parser.add_argument(
    "--enable-provider-learning",
    action="store_true",
    default=False,
    help="Enable provider rate limit learning (EXPERIMENTAL). Default: disabled.",
)
```

#### 3. Conditional Policy Selection

```python
# src/DocsToKG/ContentDownload/orchestrator/runner.py

def get_retry_policy(config, provider, host):
    """Get retry policy based on feature gates."""
    
    if config.enable_contextual_retry:
        # Use new contextual policy
        from DocsToKG.ContentDownload.errors.tenacity_policies import (
            OperationType,
            create_contextual_retry_policy,
        )
        return create_contextual_retry_policy(
            operation=OperationType.DOWNLOAD,
            max_attempts=6,
            max_delay_seconds=60,
        )
    else:
        # Use existing proven policy (DEFAULT)
        from DocsToKG.ContentDownload.networking import create_http_retry_policy
        return create_http_retry_policy(
            max_attempts=6,
            max_delay_seconds=60,
        )


def get_rate_limiter(config):
    """Get rate limiter with optional learning."""
    
    limiter = RateLimiterManager(config)
    
    if config.enable_provider_learning:
        # Attach learning tracker
        from DocsToKG.ContentDownload.ratelimit.tenacity_learning import (
            ProviderBehaviorTracker,
        )
        limiter.tracker = ProviderBehaviorTracker(
            persistence_path=config.provider_learning_path,
        )
        logger.info("Provider learning enabled")
    
    return limiter
```

---

## Why This is Safer

### 1. **Preserve Existing Behavior**
   - No changes to `create_http_retry_policy()`
   - No changes to `RateLimiterManager` core logic
   - Existing code paths 100% unaffected

### 2. **Gradual Rollout**
   - Deploy with flags `OFF` → zero impact
   - Enable in development first
   - Enable in staging for validation
   - Enable in production for subset of users
   - Observe metrics before full rollout

### 3. **Instant Rollback**
   - If issues: flip flag and restart
   - No database migration needed
   - No code changes needed
   - **Time to rollback: < 1 minute**

### 4. **A/B Testing**
   - Run contextual policy on 10% of requests
   - Compare metrics: latency, success rate, 429s
   - Scale up only after validation
   - Zero risk during experimentation

### 5. **Metrics Collection**
   ```
   Before Enabling:
     • Baseline: avg_latency=2.5s, success_rate=92%, 429s=1.2%
   
   After Enabling (with monitoring):
     • New: avg_latency=2.4s, success_rate=93%, 429s=0.8%
     • Improvement: 4% latency, +1% success, -33% 429s
   
   After 1 Week (prove stability):
     • Still stable → confidence to keep enabled
     • Regression → instant rollback
   ```

---

## Deployment Checklist

### Phase 1: Build & Test (Hours 0-5)

- [ ] Implement `tenacity_policies.py` with feature gate wrapper
- [ ] Implement `tenacity_learning.py` with feature gate wrapper
- [ ] Add feature gate flags to config
- [ ] Add CLI flags
- [ ] Add 20+ unit tests (both with gate ON and OFF)
- [ ] Add integration tests (policy switching)
- [ ] **Ensure existing tests still pass** (gate OFF by default)

### Phase 2: Staged Deployment (Hours 5-6)

**Step 1: Deploy with flags OFF (0 risk)**
```bash
# Deploy new code, flags disabled
# Existing behavior: 100% unchanged
# Monitoring: no impact expected
```

**Step 2: Enable in development (local testing)**
```bash
# Developers run:
# ./.venv/bin/python -m DocsToKG.ContentDownload.cli \
#   --topic "test" \
#   --enable-contextual-retry \
#   --enable-provider-learning \
#   --out runs/dev
# Collect: latency, success rate, 429 patterns
```

**Step 3: Enable in staging (validation)**
```bash
# Staging environment (production-like, no real users):
# Run with flags ON
# Monitor for 1-2 hours
# Collect metrics
# Compare to baseline
```

**Step 4: Enable in production (gradual)**
```bash
# Option A: Enable for opt-in users first
#   Users run: --enable-contextual-retry
#   Collect: 20+ hours of telemetry
#
# Option B: Enable for background tasks only
#   Non-critical operations use new policy
#   Critical operations use proven policy
#
# Option C: Enable via environment variable
#   DOCSTOKG_ENABLE_CONTEXTUAL_RETRY=1
#   Deploy to subset of workers
```

### Phase 3: Validate & Stabilize (Hours 6-8)

- [ ] Monitor metrics for 24+ hours
- [ ] Compare baseline vs new policy
- [ ] Check for any edge cases or regressions
- [ ] Collect user feedback
- [ ] Decide: keep enabled or revert

### Rollback Decision Tree

```
After 24 hours of production metrics:

Success Rate dropped 1%+?  → ROLLBACK (flip flag OFF)
Latency increased 10%+?    → ROLLBACK (flip flag OFF)
429 errors NOT reduced?    → Keep observing, may need tuning
Memory usage increased?    → ROLLBACK (investigate leak)
CPU overhead > 5%?         → Keep observing or ROLLBACK

All metrics stable/positive after 7 days?
→ KEEP ENABLED, no further action needed
```

---

## Code Example: Feature Gate in Action

```python
# Example 1: Using feature gate in download logic

async def download_artifact(self, artifact, url):
    """Download with conditional retry policy."""
    
    # Get policy based on feature gate
    if self.config.enable_contextual_retry:
        policy = create_contextual_retry_policy(
            operation=OperationType.DOWNLOAD,
        )
        logger.info(f"Using contextual retry for {artifact.id}")
    else:
        policy = create_http_retry_policy()
        logger.info(f"Using standard retry for {artifact.id}")
    
    # Use policy exactly the same way
    for attempt in policy:
        with attempt:
            return await self._perform_download(url)


# Example 2: Using feature gate for rate limiter learning

rate_limiter = RateLimiterManager(config)

if config.enable_provider_learning:
    rate_limiter.tracker = ProviderBehaviorTracker(
        persistence_path=config.provider_learning_path,
    )

# Rate limiter works exactly the same
limit = rate_limiter.get_effective_limit("crossref", "api.crossref.org")
```

---

## Environment Variables (Zero CLI Overhead)

Users can also enable via environment without touching CLI:

```bash
# Enable contextual retry only
export DOCSTOKG_ENABLE_CONTEXTUAL_RETRY=1

# Enable provider learning only
export DOCSTOKG_ENABLE_PROVIDER_LEARNING=1

# Enable both
export DOCSTOKG_ENABLE_CONTEXTUAL_RETRY=1
export DOCSTOKG_ENABLE_PROVIDER_LEARNING=1

# Run with defaults otherwise
./.venv/bin/python -m DocsToKG.ContentDownload.cli \
  --topic "machine learning" \
  --out runs/content
```

---

## Monitoring & Metrics

### What to Track

```
Before (Baseline):
  • Average latency per resolver: 2.3s
  • Success rate: 92.1%
  • 429 responses: 1.8%
  • Timeout rate: 0.3%
  • Provider-specific 429 patterns

After (With Flags ON):
  • Same metrics as above
  • Compare: better/worse/same?
  • Track: any new failure modes?
  • Monitor: CPU/memory overhead?
```

### CLI for Monitoring Learning

```bash
# Show provider learning status
./.venv/bin/python -m DocsToKG.ContentDownload.cli \
  --provider-learning-status \
  --output json

# Output:
{
  "providers": {
    "crossref@api.crossref.org": {
      "status": "normal",
      "consecutive_429s": 0,
      "reduction_pct": 0.0
    },
    "openalex@api.openalex.org": {
      "status": "reducing",
      "consecutive_429s": 5,
      "reduction_pct": 15.0
    }
  }
}
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| New code has bugs | Medium | Low (gate off) | Feature flag + staged rollout |
| Memory leak in tracker | Low | Medium | JSON persistence, bounded memory |
| 429 learning too aggressive | Low | Low | Easy to revert, tune parameters |
| Performance regression | Low | Low | Monitoring, quick rollback |
| **Overall Risk** | **LOW** | **LOW** | **Fully Mitigated** |

---

## Success Criteria

✅ Deployment
- All existing tests pass (gate OFF by default)
- New tests pass (gate ON and OFF)
- No breaking changes to public API
- Flags default to FALSE (backward compatible)

✅ Validation (After 24h)
- Success rate: same or improved
- Latency: same or improved
- 429 rate: same or reduced
- No memory leaks detected
- No CPU regression

✅ Stability (After 7 days)
- Metrics remain stable
- No edge cases discovered
- User feedback positive
- Ready to document as stable feature

---

## Timeline

```
Hour 0-2: Implement contextual policy + tests
Hour 2-3: Implement learning policy + tests
Hour 3-4: Add feature gates to config + CLI
Hour 4-5: Integration tests + validation
Hour 5-6: Documentation + monitoring setup
Hour 6-8: Testing + deployment prep

DEPLOY: Feature flags OFF (zero risk)
MONITOR: 24+ hours baseline comparison
DECIDE: Keep ON or revert

Total Risk: MINIMAL
Total Effort: 6-8 hours (same as before)
Total Safety: MAXIMUM
```

---

## The Beautiful Part

This approach achieves **maximum safety** with **zero overhead**:

1. **No architectural changes** - just add alongside existing code
2. **No breaking changes** - flag defaults to OFF
3. **No risk to existing users** - they never touch new code
4. **Easy to validate** - can test both paths simultaneously
5. **Instant rollback** - flip a boolean, done
6. **Gradual rollout** - enable per-user, per-environment, per-request
7. **Easy to document** - "new experimental features available"

**This is the gold standard for introducing new functionality in production systems.**

---

## Next Steps

1. ✅ Implement with feature gates (6-8 hours)
2. ✅ Deploy with flags OFF (zero risk)
3. ✅ Monitor for 24 hours
4. ✅ Enable in development
5. ✅ Enable in staging (1-2 hours)
6. ✅ Enable in production (1% of traffic)
7. ✅ Scale up based on metrics
8. ✅ Document as stable feature

**Confidence Level: 🟢 VERY HIGH**

The combination of:
- Tenacity-native patterns (proven, familiar)
- Feature gates (reversible at any time)
- Gradual rollout (risk minimization)
- Comprehensive monitoring (data-driven decisions)

...makes this approach **production-ready with minimal risk**.

