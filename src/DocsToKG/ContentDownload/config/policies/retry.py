# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.config.policies.retry",
#   "purpose": "Retry policy configuration models.",
#   "sections": [
#     {
#       "id": "backoffpolicy",
#       "name": "BackoffPolicy",
#       "anchor": "class-backoffpolicy",
#       "kind": "class"
#     },
#     {
#       "id": "retrypolicy",
#       "name": "RetryPolicy",
#       "anchor": "class-retrypolicy",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Retry policy configuration models.

Defines exponential backoff and retry strategies for transient failures.
Used by resolvers to handle network errors, rate limits, and server errors.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BackoffPolicy(BaseModel):
    """Exponential backoff configuration."""

    model_config = ConfigDict(extra="forbid")

    base_delay_ms: int = Field(default=200, ge=0, description="Initial backoff in ms")
    max_delay_ms: int = Field(default=4000, ge=0, description="Maximum backoff in ms")
    multiplier: float = Field(default=2.0, gt=1.0, description="Exponential multiplier")
    jitter_ms: int = Field(default=100, ge=0, description="Jitter range in ms")

    @field_validator("max_delay_ms")
    @classmethod
    def max_gte_base(cls, v: int, info) -> int:
        """Ensure max_delay is >= base_delay."""
        base = info.data.get("base_delay_ms", 0)
        if v < base:
            raise ValueError("max_delay_ms must be >= base_delay_ms")
        return v


class RetryPolicy(BaseModel):
    """Retry policy with exponential backoff."""

    model_config = ConfigDict(extra="forbid")

    retry_statuses: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
        description="HTTP status codes to retry on",
    )
    max_attempts: int = Field(default=4, ge=1, description="Maximum retry attempts")
    base_delay_ms: int = Field(default=200, ge=0, description="Initial backoff in ms")
    max_delay_ms: int = Field(default=4000, ge=0, description="Maximum backoff in ms")
    jitter_ms: int = Field(default=100, ge=0, description="Jitter in ms")

    @field_validator("retry_statuses")
    @classmethod
    def validate_statuses(cls, v: list[int]) -> list[int]:
        """Validate HTTP status codes."""
        if not v:
            raise ValueError("retry_statuses must not be empty")
        for status in v:
            if not (100 <= status < 600):
                raise ValueError(f"Invalid HTTP status code: {status}")
        return v

    @field_validator("max_delay_ms")
    @classmethod
    def validate_max_delay(cls, v: int, info) -> int:
        """Ensure max_delay >= base_delay."""
        base = info.data.get("base_delay_ms", 0)
        if v < base:
            raise ValueError("max_delay_ms must be >= base_delay_ms")
        return v


__all__ = ["BackoffPolicy", "RetryPolicy"]
