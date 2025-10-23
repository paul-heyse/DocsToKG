# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.orchestrator.feature_flags",
#   "purpose": "Feature flags for PR #8 Work Orchestration optimizations.",
#   "sections": [
#     {
#       "id": "is-enabled",
#       "name": "is_enabled",
#       "anchor": "function-is-enabled",
#       "kind": "function"
#     },
#     {
#       "id": "disable-feature",
#       "name": "disable_feature",
#       "anchor": "function-disable-feature",
#       "kind": "function"
#     },
#     {
#       "id": "enable-feature",
#       "name": "enable_feature",
#       "anchor": "function-enable-feature",
#       "kind": "function"
#     },
#     {
#       "id": "get-ttl",
#       "name": "get_ttl",
#       "anchor": "function-get-ttl",
#       "kind": "function"
#     },
#     {
#       "id": "get-batch-size",
#       "name": "get_batch_size",
#       "anchor": "function-get-batch-size",
#       "kind": "function"
#     },
#     {
#       "id": "get-all-flags",
#       "name": "get_all_flags",
#       "anchor": "function-get-all-flags",
#       "kind": "function"
#     },
#     {
#       "id": "log-feature-flags",
#       "name": "log_feature_flags",
#       "anchor": "function-log-feature-flags",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Feature flags for PR #8 Work Orchestration optimizations.

This module provides centralized control over all performance optimizations,
allowing them to be toggled via environment variables or configuration without
modifying the core logic. Each optimization can be independently enabled/disabled
for safe rollouts and quick reversions if needed.

**Optimizations:**

1. Connection Pooling: Per-thread SQLite connection keep-alive
2. Heartbeat Sync: Config-aware lease extension with TTL parameter
3. Job Batching: Batch lease requests to reduce DB contention
4. Semaphore Recycling: TTL-based cleanup of unused semaphores
5. Stats Optimization: Single GROUP BY query vs multiple COUNT(*)

**Usage:**

```python
from DocsToKG.ContentDownload.orchestrator.feature_flags import is_enabled, get_ttl

# Check if a feature is enabled
if is_enabled('connection_pooling'):
    # Use per-thread pooling
    ...

# Get configuration for a feature
ttl = get_ttl('semaphore_recycling')
if ttl is not None:
    # Use TTL-based eviction
    ...
```

**Environment Variables:**

- `DOCSTOKG_ENABLE_CONNECTION_POOLING` (default: true)
- `DOCSTOKG_ENABLE_HEARTBEAT_SYNC` (default: true)
- `DOCSTOKG_ENABLE_JOB_BATCHING` (default: true)
- `DOCSTOKG_ENABLE_SEMAPHORE_RECYCLING` (default: true)
- `DOCSTOKG_ENABLE_STATS_OPTIMIZATION` (default: true)
- `DOCSTOKG_SEMAPHORE_TTL_SECONDS` (default: 3600 = 1 hour)
- `DOCSTOKG_JOB_BATCH_SIZE` (default: 10)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# Feature flag definitions with defaults (all enabled by default)
_FEATURE_FLAGS = {
    "connection_pooling": os.getenv("DOCSTOKG_ENABLE_CONNECTION_POOLING", "true").lower() == "true",
    "heartbeat_sync": os.getenv("DOCSTOKG_ENABLE_HEARTBEAT_SYNC", "true").lower() == "true",
    "job_batching": os.getenv("DOCSTOKG_ENABLE_JOB_BATCHING", "true").lower() == "true",
    "semaphore_recycling": os.getenv("DOCSTOKG_ENABLE_SEMAPHORE_RECYCLING", "true").lower()
    == "true",
    "stats_optimization": os.getenv("DOCSTOKG_ENABLE_STATS_OPTIMIZATION", "true").lower() == "true",
}

# Configuration parameters
_SEMAPHORE_TTL_SECONDS: Optional[int] = (
    int(os.getenv("DOCSTOKG_SEMAPHORE_TTL_SECONDS", "3600"))
    if os.getenv("DOCSTOKG_SEMAPHORE_TTL_SECONDS")
    else 3600
)

_JOB_BATCH_SIZE: int = int(os.getenv("DOCSTOKG_JOB_BATCH_SIZE", "10"))


def is_enabled(feature: str) -> bool:
    """Check if a feature is enabled.

    Args:
        feature: Feature name (e.g., 'connection_pooling', 'heartbeat_sync')

    Returns:
        True if feature is enabled, False otherwise

    Raises:
        ValueError: If feature is unknown
    """
    if feature not in _FEATURE_FLAGS:
        raise ValueError(f"Unknown feature: {feature}. Available: {list(_FEATURE_FLAGS.keys())}")
    return _FEATURE_FLAGS[feature]


def disable_feature(feature: str) -> None:
    """Disable a feature (useful for debugging/testing).

    Args:
        feature: Feature name

    Raises:
        ValueError: If feature is unknown
    """
    if feature not in _FEATURE_FLAGS:
        raise ValueError(f"Unknown feature: {feature}. Available: {list(_FEATURE_FLAGS.keys())}")
    _FEATURE_FLAGS[feature] = False
    logger.info(f"Feature disabled: {feature}")


def enable_feature(feature: str) -> None:
    """Enable a feature.

    Args:
        feature: Feature name

    Raises:
        ValueError: If feature is unknown
    """
    if feature not in _FEATURE_FLAGS:
        raise ValueError(f"Unknown feature: {feature}. Available: {list(_FEATURE_FLAGS.keys())}")
    _FEATURE_FLAGS[feature] = True
    logger.info(f"Feature enabled: {feature}")


def get_ttl(feature: str) -> Optional[int]:
    """Get TTL configuration for a feature (semaphore recycling).

    Args:
        feature: Feature name (typically 'semaphore_recycling')

    Returns:
        TTL in seconds, or None to disable TTL-based eviction
    """
    if feature == "semaphore_recycling":
        return _SEMAPHORE_TTL_SECONDS
    raise ValueError(f"Feature {feature} does not have a TTL configuration")


def get_batch_size() -> int:
    """Get batch size for job batching.

    Returns:
        Batch size (default: 10)
    """
    return _JOB_BATCH_SIZE


def get_all_flags() -> dict[str, bool]:
    """Get all feature flag statuses.

    Returns:
        Dict mapping feature names to enabled/disabled status
    """
    return _FEATURE_FLAGS.copy()


def log_feature_flags() -> None:
    """Log all feature flag statuses at startup."""
    logger.info("=" * 80)
    logger.info("ORCHESTRATION FEATURE FLAGS")
    logger.info("=" * 80)
    for feature, enabled in get_all_flags().items():
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        logger.info(f"  {feature:30s} {status}")
    logger.info(f"  semaphore_ttl_sec:              {_SEMAPHORE_TTL_SECONDS}")
    logger.info(f"  job_batch_size:                 {_JOB_BATCH_SIZE}")
    logger.info("=" * 80)
