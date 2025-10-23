# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.config.policies.http",
#   "purpose": "HTTP client configuration.",
#   "sections": [
#     {
#       "id": "httpclientconfig",
#       "name": "HttpClientConfig",
#       "anchor": "class-httpclientconfig",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""HTTP client configuration.

Controls HTTPX client behavior, timeouts, pooling, caching, and TLS settings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HttpClientConfig(BaseModel):
    """HTTP client configuration for HTTPX singleton."""

    model_config = ConfigDict(extra="forbid")

    http2: bool = Field(default=True, description="Enable HTTP/2")
    user_agent: str = Field(
        default="DocsToKG/ContentDownload (+mailto:research@example.com)",
        description="User-Agent string",
    )
    trust_env: bool = Field(default=True, description="Honor HTTP(S)_PROXY and NO_PROXY")

    # Timeout settings
    timeout_connect_s: float = Field(default=5.0, ge=0.1, description="Connect timeout (seconds)")
    timeout_read_s: float = Field(default=30.0, ge=0.1, description="Read timeout (seconds)")
    timeout_write_s: float = Field(default=30.0, ge=0.1, description="Write timeout (seconds)")
    timeout_pool_s: float = Field(default=5.0, ge=0.1, description="Pool acquire timeout (seconds)")

    # Pool settings
    max_connections: int = Field(default=64, ge=1, description="Max connections")
    max_keepalive_connections: int = Field(
        default=20, ge=1, description="Max keepalive connections"
    )
    keepalive_expiry_s: float = Field(default=30.0, ge=0, description="Keepalive expiry (seconds)")

    # Cache settings (Hishel)
    cache_enabled: bool = Field(default=True, description="Enable Hishel HTTP cache")
    cache_dir: str = Field(
        default="~/.cache/docstokg/http", description="Cache directory (expands ~)"
    )
    cache_bypass: bool = Field(default=False, description="Bypass cache (useful for testing)")

    # Retry settings (transport-level)
    connect_retries: int = Field(default=2, ge=0, description="Transport-level connect retries")
    retry_backoff_base: float = Field(default=0.1, ge=0, description="Backoff base (seconds)")
    retry_backoff_max: float = Field(default=2.0, ge=0, description="Max backoff (seconds)")

    # Security settings
    verify_tls: bool = Field(default=True, description="Verify TLS certificates")
    proxies: dict[str, str] = Field(default_factory=dict, description="Proxy dict (deprecated)")
    polite_headers: dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "DocsToKG/ContentDownload (+mailto:research@example.com)"
        },
        description="Polite HTTP headers (deprecated; use user_agent instead)",
    )
    mailto: str | None = Field(default=None, description="Email for robots.txt compliance")

    @field_validator("timeout_connect_s", "timeout_read_s", "timeout_write_s", "timeout_pool_s")
    @classmethod
    def validate_timeouts(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("All timeouts must be > 0")
        return v

    @field_validator("max_connections", "max_keepalive_connections")
    @classmethod
    def validate_pool_limits(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Pool limits must be > 0")
        return v

    @field_validator("keepalive_expiry_s")
    @classmethod
    def validate_keepalive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Keepalive expiry must be >= 0")
        return v

    @field_validator("connect_retries")
    @classmethod
    def validate_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("connect_retries must be >= 0")
        return v

    @field_validator("retry_backoff_base", "retry_backoff_max")
    @classmethod
    def validate_backoff(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Backoff values must be >= 0")
        return v

    @model_validator(mode="after")
    def validate_read_timeout_vs_write_timeout(self) -> HttpClientConfig:
        """Ensure timeout_read_s is at least as large as timeout_write_s."""
        if self.timeout_read_s < self.timeout_write_s:
            raise ValueError("timeout_read_s must be >= timeout_write_s")
        return self


__all__ = ["HttpClientConfig"]
