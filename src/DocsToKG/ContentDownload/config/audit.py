"""Configuration audit trail tracking.

Tracks how configuration was loaded and what sources applied overrides.
Useful for debugging precedence issues and correlating configs with telemetry.

Example:
    cfg, audit = load_config_with_audit("config.yaml")
    if audit.env_overrides:
        logger.info(f"Environment overrides: {list(audit.env_overrides.keys())}")
    logger.info(f"Config hash: {audit.config_hash}")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from DocsToKG.ContentDownload.config.models import ContentDownloadConfig


@dataclass
class ConfigAuditLog:
    """Track how configuration was loaded and what overrides were applied."""

    loaded_from_file: bool = False
    file_path: Optional[str] = None
    env_overrides: Dict[str, str] = field(default_factory=dict)
    cli_overrides: Dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.now)
    schema_version: int = 1
    config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "loaded_from_file": self.loaded_from_file,
            "file_path": self.file_path,
            "env_overrides": self.env_overrides,
            "cli_overrides": self.cli_overrides,
            "loaded_at": self.loaded_at.isoformat(),
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "sources_used": self._sources_used(),
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def _sources_used(self) -> list[str]:
        """List which config sources were used."""
        sources = []
        if self.loaded_from_file:
            sources.append("file")
        if self.env_overrides:
            sources.append("env")
        if self.cli_overrides:
            sources.append("cli")
        return sources or ["defaults"]


def load_config_with_audit(
    path: Optional[str] = None,
    env_prefix: str = "DTKG_",
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[ContentDownloadConfig, ConfigAuditLog]:
    """
    Load configuration and track audit information.

    Tracks which sources (file, env, CLI) were used and what overrides applied.

    Args:
        path: Config file path (YAML/JSON)
        env_prefix: Environment variable prefix (default: DTKG_)
        cli_overrides: CLI argument overrides

    Returns:
        Tuple of (ContentDownloadConfig, ConfigAuditLog)

    Example:
        >>> cfg, audit = load_config_with_audit("config.yaml")
        >>> print(f"Loaded from {audit._sources_used()}")
        >>> print(f"Config hash: {audit.config_hash}")
    """
    from DocsToKG.ContentDownload.config.loader import load_config
    import os

    # Initialize audit log
    audit = ConfigAuditLog(
        loaded_from_file=path is not None,
        file_path=path,
        schema_version=1,
    )

    # Load configuration using standard loader
    cfg = load_config(path=path, env_prefix=env_prefix, cli_overrides=cli_overrides)

    # Collect environment overrides
    tracked_env = {}
    for k, v in os.environ.items():
        if k.startswith(env_prefix):
            tracked_env[k] = v
    audit.env_overrides = tracked_env

    # Track CLI overrides
    if cli_overrides:
        audit.cli_overrides = cli_overrides

    # Compute config hash for provenance tracking
    config_json = cfg.model_dump_json(sort_keys=True)
    audit.config_hash = hashlib.sha256(config_json.encode()).hexdigest()

    return cfg, audit


def compute_config_hash(cfg: ContentDownloadConfig) -> str:
    """
    Compute deterministic hash of configuration.

    Useful for provenance tracking and correlating configs with telemetry.

    Args:
        cfg: Configuration to hash

    Returns:
        SHA256 hex digest of config JSON
    """
    config_json = cfg.model_dump_json(sort_keys=True)
    return hashlib.sha256(config_json.encode()).hexdigest()


__all__ = [
    "ConfigAuditLog",
    "load_config_with_audit",
    "compute_config_hash",
]
