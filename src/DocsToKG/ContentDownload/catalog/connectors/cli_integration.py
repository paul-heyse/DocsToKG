# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.connectors.cli_integration",
#   "purpose": "CLI Integration for CatalogConnector.",
#   "sections": [
#     {
#       "id": "environmentdetector",
#       "name": "EnvironmentDetector",
#       "anchor": "class-environmentdetector",
#       "kind": "class"
#     },
#     {
#       "id": "cliconfigbuilder",
#       "name": "CLIConfigBuilder",
#       "anchor": "class-cliconfigbuilder",
#       "kind": "class"
#     },
#     {
#       "id": "create-connector-from-cli",
#       "name": "create_connector_from_cli",
#       "anchor": "function-create-connector-from-cli",
#       "kind": "function"
#     },
#     {
#       "id": "get-cli-help-text",
#       "name": "get_cli_help_text",
#       "anchor": "function-get-cli-help-text",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
CLI Integration for CatalogConnector

Provides utilities for integrating CatalogConnector with command-line applications,
including:
- Configuration file and provider discovery
- CLI argument parsing helpers
- Environment variable detection
- Credential validation
- Deployment environment detection
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from .config import load_catalog_config
from .connector import CatalogConnector

logger = logging.getLogger(__name__)


class EnvironmentDetector:
    """Detect deployment environment and available credentials."""

    @staticmethod
    def detect_provider() -> Literal["development", "enterprise", "cloud"] | None:
        """Detect provider from environment variables.

        Detection order:
        1. CATALOG_PROVIDER environment variable
        2. CATALOG_CLOUD_* variables → cloud provider
        3. CATALOG_ENTERPRISE_* variables → enterprise provider
        4. CATALOG_DEV_* variables → development provider
        5. Cloud credential detection → cloud provider
        6. None if no clear provider detected

        Returns:
            Provider type or None if not detected
        """
        # Explicit override
        if provider := os.getenv("CATALOG_PROVIDER"):
            if provider in ("development", "enterprise", "cloud"):
                return provider  # type: ignore[return-value]
            logger.warning(f"Unknown provider: {provider}")

        # Detect from environment variables
        env_keys = set(os.environ.keys())

        if any(k.startswith("CATALOG_CLOUD_") for k in env_keys):
            return "cloud"

        if any(k.startswith("CATALOG_ENTERPRISE_") for k in env_keys):
            return "enterprise"

        if any(k.startswith("CATALOG_DEV_") for k in env_keys):
            return "development"

        # Detect from AWS credentials
        if EnvironmentDetector._has_aws_credentials():
            logger.debug("AWS credentials detected, suggesting cloud provider")
            return "cloud"

        return None

    @staticmethod
    def _has_aws_credentials() -> bool:
        """Check if AWS credentials are available.

        Checks:
        - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
        - AWS_PROFILE environment variable
        - ~/.aws/credentials file exists
        """
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            return True

        if os.getenv("AWS_PROFILE"):
            return True

        aws_credentials = Path.home() / ".aws" / "credentials"
        return aws_credentials.exists()

    @staticmethod
    def find_config_file(
        start_dir: str | None = None,
        max_depth: int = 3,
    ) -> Path | None:
        """Find configuration file in standard locations.

        Search order:
        1. CATALOG_CONFIG environment variable
        2. Current directory and parents (up to max_depth)
        3. ~/.config/catalog/config.{yaml,json}
        4. /etc/catalog/config.{yaml,json}

        Args:
            start_dir: Starting directory for search (default: current directory)
            max_depth: Maximum depth to search up directory tree

        Returns:
            Path to configuration file or None if not found
        """
        # Check explicit override
        if config_path := os.getenv("CATALOG_CONFIG"):
            path = Path(config_path).expanduser()
            if path.exists():
                logger.info(f"Config file from CATALOG_CONFIG: {path}")
                return path
            logger.warning(f"CATALOG_CONFIG path not found: {config_path}")

        # Search up directory tree
        start = Path(start_dir or ".").resolve()
        for i in range(max_depth):
            current = start.parent if i > 0 else start

            for filename in ["catalog.yaml", "catalog.yml", "catalog.json"]:
                candidate = current / filename
                if candidate.exists():
                    logger.debug(f"Found config file: {candidate}")
                    return candidate

            start = current.parent

        # Check user config directory
        user_config = Path.home() / ".config" / "catalog"
        for filename in ["config.yaml", "config.yml", "config.json"]:
            candidate = user_config / filename
            if candidate.exists():
                logger.debug(f"Found user config file: {candidate}")
                return candidate

        # Check system config directory
        sys_config = Path("/etc/catalog")
        for filename in ["config.yaml", "config.yml", "config.json"]:
            candidate = sys_config / filename
            if candidate.exists():
                logger.debug(f"Found system config file: {candidate}")
                return candidate

        return None

    @staticmethod
    def suggest_provider_config() -> dict[str, str]:
        """Suggest provider and configuration based on environment.

        Returns:
            Dict with 'provider' and 'reason' keys
        """
        provider = EnvironmentDetector.detect_provider()

        if provider == "cloud":
            return {
                "provider": "cloud",
                "reason": "AWS credentials or CATALOG_CLOUD_* variables detected",
            }

        if provider == "enterprise":
            return {
                "provider": "enterprise",
                "reason": "CATALOG_ENTERPRISE_* variables detected",
            }

        if provider == "development":
            return {
                "provider": "development",
                "reason": "CATALOG_DEV_* variables detected or running locally",
            }

        return {
            "provider": "development",
            "reason": "No specific config detected, defaulting to development",
        }


class CLIConfigBuilder:
    """Build connector configuration from CLI arguments and environment."""

    def __init__(
        self,
        provider: Literal["development", "enterprise", "cloud"] | None = None,
        config_file: str | None = None,
        auto_discover: bool = True,
    ):
        """Initialize CLI config builder.

        Args:
            provider: Explicit provider type (overrides detection)
            config_file: Path to configuration file (overrides discovery)
            auto_discover: Whether to auto-discover provider and config file
        """
        self.explicit_provider = provider
        self.explicit_config_file = config_file
        self.auto_discover = auto_discover

    def build_connector(
        self,
        cli_overrides: dict[str, Any] | None = None,
    ) -> CatalogConnector:
        """Build and return a CatalogConnector instance.

        Configuration precedence:
        1. Explicit CLI overrides
        2. Environment variables
        3. Configuration file
        4. Provider defaults

        Args:
            cli_overrides: CLI arguments to override configuration

        Returns:
            Initialized CatalogConnector

        Raises:
            ValueError: If provider cannot be determined
            RuntimeError: If configuration is invalid
        """
        # Determine provider
        provider = self._get_provider()
        if not provider:
            raise ValueError(
                "Cannot determine catalog provider. "
                "Set CATALOG_PROVIDER or CATALOG_*_* environment variables, "
                "or provide --provider flag."
            )

        # Find configuration file
        config_file = self.explicit_config_file
        if not config_file and self.auto_discover:
            found_config = EnvironmentDetector.find_config_file()
            config_file = str(found_config) if found_config else None

        # Load configuration
        try:
            config = load_catalog_config(
                provider_type=provider,
                config_file=config_file,
                config_dict=cli_overrides,
                use_env=True,
            )
        except ValueError as e:
            raise RuntimeError(f"Configuration error: {e}") from e

        # Create and return connector
        connector = CatalogConnector(provider, config)
        logger.info(
            f"Catalog connector initialized: provider={provider}, "
            f"config_file={config_file or 'none'}"
        )

        return connector

    def get_config_dict(
        self,
        cli_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get configuration dictionary without creating connector.

        Args:
            cli_overrides: CLI arguments to override

        Returns:
            Configuration dictionary
        """
        provider = self._get_provider()
        if not provider:
            raise ValueError("Cannot determine catalog provider")

        config_file = self.explicit_config_file
        if not config_file and self.auto_discover:
            found_config = EnvironmentDetector.find_config_file()
            config_file = str(found_config) if found_config else None

        return load_catalog_config(
            provider_type=provider,
            config_file=config_file,
            config_dict=cli_overrides,
            use_env=True,
        )

    def _get_provider(
        self,
    ) -> Literal["development", "enterprise", "cloud"] | None:
        """Get provider type with auto-detection.

        Returns:
            Provider type or None
        """
        if self.explicit_provider:
            return self.explicit_provider

        if not self.auto_discover:
            return None

        return EnvironmentDetector.detect_provider()


def create_connector_from_cli(
    provider: Literal["development", "enterprise", "cloud"] | None = None,
    config_file: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    auto_discover: bool = True,
) -> CatalogConnector:
    """Convenience function to create connector from CLI context.

    This is the primary entry point for CLI integration.

    Example:
        import click
        from catalog.connectors.cli_integration import create_connector_from_cli

        @click.command()
        @click.option("--provider", type=click.Choice(["dev", "enterprise", "cloud"]))
        @click.option("--config", type=click.Path(exists=True))
        def my_command(provider, config):
            connector = create_connector_from_cli(
                provider=provider,
                config_file=config,
                auto_discover=True,
            )

            with connector as cat:
                # Use connector
                stats = cat.stats()
                print(f"Catalog has {stats['total_records']} records")

    Args:
        provider: Explicit provider (overrides detection)
        config_file: Path to config file (overrides discovery)
        cli_overrides: CLI argument overrides
        auto_discover: Whether to auto-detect provider/config

    Returns:
        Initialized CatalogConnector

    Raises:
        ValueError: If provider cannot be determined
        RuntimeError: If configuration is invalid
    """
    builder = CLIConfigBuilder(
        provider=provider,
        config_file=config_file,
        auto_discover=auto_discover,
    )

    return builder.build_connector(cli_overrides=cli_overrides)


def get_cli_help_text(
    include_examples: bool = True,
) -> str:
    """Generate help text for CLI users about catalog configuration.

    Args:
        include_examples: Whether to include usage examples

    Returns:
        Formatted help text
    """
    help_text = """
CATALOG CONNECTOR CONFIGURATION
===============================

The catalog connector can be configured via:

1. ENVIRONMENT VARIABLES (highest priority)
   CATALOG_PROVIDER=cloud
   CATALOG_CLOUD_CONNECTION_URL=postgresql://...
   CATALOG_CLOUD_S3_BUCKET=my-bucket
   CATALOG_ENTERPRISE_POOL_SIZE=20

2. CONFIGURATION FILES
   catalog.yaml or catalog.json in current directory
   ~/.config/catalog/config.yaml
   /etc/catalog/config.yaml

3. CLI FLAGS
   --provider development|enterprise|cloud
   --config /path/to/config.yaml

4. DEFAULTS
   Development: SQLite in-memory
   Enterprise: Connection pooling (10+20)
   Cloud: AWS US-East-1, S3 artifacts prefix

PRECEDENCE: Environment variables > CLI flags > Config file > Defaults

AUTO-DETECTION:
- Provider is auto-detected from environment or AWS credentials
- Config file is auto-discovered from standard locations
- Can be overridden with explicit --provider and --config flags
"""

    if include_examples:
        help_text += """

EXAMPLES
========

Development (local):
  $ catalog-cli show

Cloud (with AWS credentials):
  $ export CATALOG_CLOUD_S3_BUCKET=my-artifacts
  $ catalog-cli show

Enterprise (with config file):
  $ catalog-cli --config /etc/catalog/config.yaml show

Mixed (file + env override):
  $ export CATALOG_ENTERPRISE_POOL_SIZE=30
  $ catalog-cli --config enterprise.yaml show
"""

    return help_text
