# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.config.policies.__init__",
#   "purpose": "Configuration policy models.",
#   "sections": []
# }
# === /NAVMAP ===

"""Configuration policy models.

Provides focused, single-responsibility Pydantic v2 models for each subsystem:
- Retry policies (exponential backoff, max attempts, etc)
- Rate limiting (token bucket, capacity, refill rate)
- Robot rules (enable/disable, TTL)
- Download policies (atomic writes, verification)
- HTTP client configuration (timeouts, pooling)
"""

from __future__ import annotations

from DocsToKG.ContentDownload.config.policies.download import DownloadPolicy
from DocsToKG.ContentDownload.config.policies.http import HttpClientConfig
from DocsToKG.ContentDownload.config.policies.ratelimit import RateLimitPolicy

# Re-export for backward compatibility
from DocsToKG.ContentDownload.config.policies.retry import (
    BackoffPolicy,
    RetryPolicy,
)
from DocsToKG.ContentDownload.config.policies.robots import RobotsPolicy

__all__ = [
    "RetryPolicy",
    "BackoffPolicy",
    "RateLimitPolicy",
    "RobotsPolicy",
    "DownloadPolicy",
    "HttpClientConfig",
]
