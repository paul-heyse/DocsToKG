"""
ContentDownload Configuration Package

Public API for loading, validating, and introspecting ContentDownload configuration.

Example:
    from DocsToKG.ContentDownload.config import load_config, ContentDownloadConfig

    # Load from file with env/CLI overrides
    config = load_config(
        path="contentdownload.yaml",
        cli_overrides={"resolvers": {"order": ["arxiv", "landing"]}}
    )

    # Get config hash for reproducibility
    config_id = config.config_hash()

    # Export schema for documentation
    from DocsToKG.ContentDownload.config import export_config_schema
    schema = export_config_schema()
"""

from .loader import (
    export_config_schema,
    load_config,
    validate_config_file,
)
from .models import (
    BackoffPolicy,
    ContentDownloadConfig,
    CrossrefConfig,
    DownloadPolicy,
    HttpClientConfig,
    RateLimitPolicy,
    ResolverCommonConfig,
    ResolversConfig,
    RetryPolicy,
    RobotsPolicy,
    TelemetryConfig,
    UnpaywallConfig,
)

__all__ = [
    # Models
    "ContentDownloadConfig",
    "ResolversConfig",
    "HttpClientConfig",
    "RetryPolicy",
    "BackoffPolicy",
    "RateLimitPolicy",
    "RobotsPolicy",
    "DownloadPolicy",
    "TelemetryConfig",
    "ResolverCommonConfig",
    "UnpaywallConfig",
    "CrossrefConfig",
    # Loading/validation
    "load_config",
    "validate_config_file",
    "export_config_schema",
]
