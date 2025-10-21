"""Rate limiting policy configuration.

Defines token bucket rate limiting for per-resolver and per-host throttling.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RateLimitPolicy(BaseModel):
    """Token bucket rate limiting configuration."""

    model_config = ConfigDict(extra="forbid")

    capacity: int = Field(default=5, ge=1, description="Bucket capacity (max tokens)")
    refill_per_sec: float = Field(default=1.0, gt=0, description="Tokens per second")
    burst: int = Field(default=2, ge=1, description="Burst allowance (extra tokens)")

    @model_validator(mode="after")
    def validate_capacity_vs_burst(self) -> RateLimitPolicy:
        """Ensure capacity is greater than or equal to burst."""
        if self.capacity < self.burst:
            raise ValueError("Capacity must be >= burst")
        return self


__all__ = ["RateLimitPolicy"]
