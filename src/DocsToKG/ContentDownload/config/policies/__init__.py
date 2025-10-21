"""Configuration policy models.

Provides focused, single-responsibility Pydantic v2 models for each subsystem:
- Retry policies (exponential backoff, max attempts, etc)
- Rate limiting (token bucket, capacity, refill rate)
- Robot rules (enable/disable, TTL)
- Download policies (atomic writes, verification)
- HTTP client configuration (timeouts, pooling)
"""

from __future__ import annotations

# Re-export for backward compatibility
from DocsToKG.ContentDownload.config.policies.retry import (
    BackoffPolicy,
    RetryPolicy,
)
from DocsToKG.ContentDownload.config.policies.ratelimit import RateLimitPolicy
from DocsToKG.ContentDownload.config.policies.robots import RobotsPolicy
from DocsToKG.ContentDownload.config.policies.download import DownloadPolicy
from DocsToKG.ContentDownload.config.policies.http import HttpClientConfig

__all__ = [
    "RetryPolicy",
    "BackoffPolicy",
    "RateLimitPolicy",
    "RobotsPolicy",
    "DownloadPolicy",
    "HttpClientConfig",
]
