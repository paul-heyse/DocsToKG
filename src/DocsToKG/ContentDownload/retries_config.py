# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.retries_config",
#   "purpose": "Tenacity retries configuration loading and management.",
#   "sections": [
#     {
#       "id": "backoffconfig",
#       "name": "BackoffConfig",
#       "anchor": "class-backoffconfig",
#       "kind": "class"
#     },
#     {
#       "id": "retriesconfig",
#       "name": "RetriesConfig",
#       "anchor": "class-retriesconfig",
#       "kind": "class"
#     },
#     {
#       "id": "load-retries-config",
#       "name": "load_retries_config",
#       "anchor": "function-load-retries-config",
#       "kind": "function"
#     },
#     {
#       "id": "get-retries-config",
#       "name": "get_retries_config",
#       "anchor": "function-get-retries-config",
#       "kind": "function"
#     },
#     {
#       "id": "set-retries-config",
#       "name": "set_retries_config",
#       "anchor": "function-set-retries-config",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Tenacity retries configuration loading and management.

Provides RFC-compliant retry configuration with support for:
- Retry-After header parsing (seconds and HTTP-date formats)
- Configurable backoff strategies (exponential with jitter)
- Per-status and per-exception retry policies
- Environment variable and CLI argument overrides
- Per-method gating (GET/HEAD vs POST with idempotency check)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackoffConfig:
    """Exponential backoff configuration."""

    multiplier: float = 0.75  # base for wait_random_exponential
    max_s: float = 60  # maximum backoff wait


@dataclass(frozen=True)
class RetriesConfig:
    """Complete retries configuration."""

    max_attempts: int = 7  # total attempts (initial + retries)
    max_total_s: float = 120  # wall-clock ceiling in seconds
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
    statuses: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    methods: List[str] = field(default_factory=lambda: ["GET", "HEAD"])
    retry_408: bool = False  # optionally retry 408 (Request Timeout)
    retry_on_timeout: bool = True  # retry httpx.TimeoutException
    retry_on_remote_protocol: bool = True  # retry httpx.RemoteProtocolError
    retry_after_cap_s: float = 900  # cap for Retry-After parsing (15 minutes)
    allow_post_if_idempotent: bool = True  # allow POST retry if marked idempotent

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.max_total_s < 0:
            raise ValueError(f"max_total_s must be >= 0, got {self.max_total_s}")
        if self.backoff.multiplier < 0:
            raise ValueError(f"backoff.multiplier must be >= 0, got {self.backoff.multiplier}")
        if self.backoff.max_s < 0:
            raise ValueError(f"backoff.max_s must be >= 0, got {self.backoff.max_s}")
        if self.retry_after_cap_s < 0:
            raise ValueError(f"retry_after_cap_s must be >= 0, got {self.retry_after_cap_s}")


def load_retries_config(
    yaml_dict: Optional[Dict[str, Any]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> RetriesConfig:
    """Load retries configuration from YAML, environment, and CLI.

    Precedence (highest to lowest):
    1. CLI argument overrides
    2. Environment variable overrides
    3. YAML configuration
    4. Built-in defaults

    Args:
        yaml_dict: Dictionary from parsed YAML (retries section)
        env: Environment variables (default: os.environ)
        cli_overrides: CLI argument overrides

    Returns:
        Validated RetriesConfig instance
    """
    import os

    if env is None:
        env = os.environ

    # Start with YAML
    config_dict = dict(yaml_dict or {})

    # Apply environment variable overlays
    env_max_attempts = env.get("DOCSTOKG_RETRIES_MAX_ATTEMPTS")
    if env_max_attempts:
        config_dict["max_attempts"] = int(env_max_attempts)

    env_max_total_s = env.get("DOCSTOKG_RETRIES_MAX_TOTAL_S")
    if env_max_total_s:
        config_dict["max_total_s"] = float(env_max_total_s)

    env_retry_408 = env.get("DOCSTOKG_RETRIES_RETRY_408")
    if env_retry_408:
        config_dict["retry_408"] = env_retry_408.lower() in ("true", "1", "yes")

    env_retry_timeout = env.get("DOCSTOKG_RETRIES_RETRY_ON_TIMEOUT")
    if env_retry_timeout:
        config_dict["retry_on_timeout"] = env_retry_timeout.lower() in ("true", "1", "yes")

    env_retry_remote = env.get("DOCSTOKG_RETRIES_RETRY_ON_REMOTE_PROTOCOL")
    if env_retry_remote:
        config_dict["retry_on_remote_protocol"] = env_retry_remote.lower() in (
            "true",
            "1",
            "yes",
        )

    env_statuses = env.get("DOCSTOKG_RETRIES_STATUSES")
    if env_statuses:
        config_dict["statuses"] = [int(s.strip()) for s in env_statuses.split(",")]

    env_methods = env.get("DOCSTOKG_RETRIES_METHODS")
    if env_methods:
        config_dict["methods"] = [m.strip().upper() for m in env_methods.split(",")]

    # Apply CLI overrides
    if cli_overrides:
        config_dict.update(cli_overrides)

    # Extract backoff config if nested
    backoff_dict = config_dict.pop("backoff", {})
    if backoff_dict:
        backoff = BackoffConfig(**backoff_dict)
        config_dict["backoff"] = backoff
    else:
        config_dict["backoff"] = BackoffConfig()

    # Create and validate
    return RetriesConfig(**config_dict)


# Global retries configuration instance
_GLOBAL_RETRIES_CONFIG: Optional[RetriesConfig] = None


def get_retries_config() -> RetriesConfig:
    """Get global retries configuration instance.

    Returns:
        Global RetriesConfig instance
    """
    global _GLOBAL_RETRIES_CONFIG
    if _GLOBAL_RETRIES_CONFIG is None:
        _GLOBAL_RETRIES_CONFIG = RetriesConfig()
    return _GLOBAL_RETRIES_CONFIG


def set_retries_config(config: RetriesConfig) -> None:
    """Set global retries configuration.

    Args:
        config: RetriesConfig instance to use globally
    """
    global _GLOBAL_RETRIES_CONFIG
    _GLOBAL_RETRIES_CONFIG = config
    LOGGER.info(
        "Retries configuration updated: max_attempts=%d, max_total_s=%.1f",
        config.max_attempts,
        config.max_total_s,
    )
