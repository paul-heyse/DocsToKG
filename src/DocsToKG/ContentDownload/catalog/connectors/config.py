# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.connectors.config",
#   "purpose": "Configuration management for CatalogConnector.",
#   "sections": [
#     {
#       "id": "catalogconfig",
#       "name": "CatalogConfig",
#       "anchor": "class-catalogconfig",
#       "kind": "class"
#     },
#     {
#       "id": "load-catalog-config",
#       "name": "load_catalog_config",
#       "anchor": "function-load-catalog-config",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Configuration management for CatalogConnector

Supports multiple configuration sources with precedence:
1. Environment variables (highest priority)
2. Programmatic config (dict)
3. YAML/JSON files
4. Defaults (lowest priority)

Configuration sources are merged in order, allowing flexible deployment patterns.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


class CatalogConfig:
    """Catalog connector configuration loader and validator."""

    # Environment variable mappings
    ENV_PREFIXES = {
        "development": "CATALOG_DEV_",
        "enterprise": "CATALOG_ENTERPRISE_",
        "cloud": "CATALOG_CLOUD_",
    }

    # Required fields per provider
    REQUIRED_FIELDS = {
        "development": [],  # Optional - in-memory by default
        "enterprise": ["connection_url"],
        "cloud": ["connection_url", "s3_bucket"],
    }

    # Default configurations
    DEFAULTS = {
        "development": {
            "db_path": ":memory:",
        },
        "enterprise": {
            "pool_size": 10,
            "max_overflow": 20,
            "echo_sql": False,
        },
        "cloud": {
            "s3_region": "us-east-1",
            "s3_prefix": "artifacts/",
            "pool_size": 10,
            "max_overflow": 20,
            "echo_sql": False,
        },
    }

    def __init__(
        self,
        provider_type: Literal["development", "enterprise", "cloud"],
        config_file: str | None = None,
        config_dict: dict[str, Any] | None = None,
        use_env: bool = True,
    ):
        """Initialize configuration loader.

        Args:
            provider_type: Type of provider (development, enterprise, cloud)
            config_file: Path to YAML or JSON configuration file
            config_dict: Programmatic configuration dictionary
            use_env: Whether to load from environment variables
        """
        self.provider_type = provider_type
        self.config_file = config_file
        self.config_dict = config_dict or {}
        self.use_env = use_env
        self._config: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load and merge configuration from all sources.

        Precedence (highest to lowest):
        1. Environment variables
        2. Programmatic config_dict
        3. Configuration file (YAML/JSON)
        4. Defaults

        Returns:
            Merged configuration dictionary
        """
        # Start with defaults
        defaults = self.DEFAULTS.get(self.provider_type, {})
        self._config = dict(defaults) if defaults else {}  # type: ignore[arg-type]

        # Load from file
        if self.config_file:
            self._config.update(self._load_file(self.config_file))
            logger.debug(f"Loaded config from file: {self.config_file}")

        # Merge programmatic config
        if self.config_dict:
            self._config.update(self.config_dict)
            logger.debug("Merged programmatic config")

        # Load from environment variables
        if self.use_env:
            env_config = self._load_env()
            if env_config:
                self._config.update(env_config)
                logger.debug("Merged environment variables")

        # Validate
        self._validate()

        logger.info(f"Configuration loaded for provider: {self.provider_type}")
        return self._config

    def _load_file(self, path: str) -> dict[str, Any]:
        """Load configuration from YAML or JSON file.

        Args:
            path: File path to YAML or JSON configuration

        Returns:
            Configuration dictionary from file
        """
        config_path = Path(path).expanduser().resolve()

        if not config_path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return {}

        try:
            content = config_path.read_text()

            # JSON files
            if config_path.suffix == ".json":
                return json.loads(content)

            # YAML files
            if config_path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml  # type: ignore[import-untyped]

                    return yaml.safe_load(content) or {}
                except ImportError:
                    logger.error("PyYAML not installed. Install with: pip install pyyaml")
                    raise

            logger.warning(f"Unsupported config file format: {config_path.suffix}")
            return {}

        except Exception as e:
            logger.error(f"Failed to load config file {path}: {e}")
            raise

    def _load_env(self) -> dict[str, Any]:
        """Load configuration from environment variables.

        Environment variable format:
        - CATALOG_{PROVIDER}_KEY=value

        Examples:
        - CATALOG_CLOUD_CONNECTION_URL=postgresql://...
        - CATALOG_CLOUD_S3_BUCKET=my-bucket
        - CATALOG_ENTERPRISE_POOL_SIZE=20

        Returns:
            Configuration dictionary from environment
        """
        config: dict[str, Any] = {}
        prefix = self.ENV_PREFIXES.get(self.provider_type, "CATALOG_")

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Extract config key (remove prefix and convert to lowercase)
            config_key = key[len(prefix) :].lower()

            # Try to parse value as number
            if value.isdigit():
                config[config_key] = int(value)
            elif value.lower() in ("true", "false"):
                config[config_key] = value.lower() == "true"
            else:
                config[config_key] = value

        return config

    def _validate(self) -> None:
        """Validate configuration has all required fields.

        Raises:
            ValueError: If required fields are missing
        """
        required = self.REQUIRED_FIELDS.get(self.provider_type, [])

        missing = [field for field in required if field not in self._config]

        if missing:
            raise ValueError(
                f"Missing required configuration fields for {self.provider_type}: "
                f"{', '.join(missing)}"
            )

        # Provider-specific validation
        if self.provider_type == "cloud":
            self._validate_cloud()
        elif self.provider_type == "enterprise":
            self._validate_enterprise()

    def _validate_cloud(self) -> None:
        """Validate cloud provider configuration."""
        # S3 bucket must be valid
        s3_bucket = self._config.get("s3_bucket", "")
        if s3_bucket and not self._is_valid_bucket_name(s3_bucket):
            raise ValueError(
                f"Invalid S3 bucket name: {s3_bucket}. "
                "Bucket names must be 3-63 chars, lowercase, numbers, hyphens."
            )

        # S3 region should be valid AWS region
        s3_region = self._config.get("s3_region", "us-east-1")
        if not self._is_valid_aws_region(s3_region):
            logger.warning(f"Potentially invalid AWS region: {s3_region}")

        # Connection URL should be PostgreSQL
        conn_url = self._config.get("connection_url", "")
        if conn_url and not conn_url.startswith("postgresql://"):
            logger.warning(f"Connection URL does not appear to be PostgreSQL: {conn_url[:30]}...")

    def _validate_enterprise(self) -> None:
        """Validate enterprise provider configuration."""
        # Connection URL should be valid
        conn_url = self._config.get("connection_url", "")
        if conn_url and not (
            conn_url.startswith("postgresql://") or conn_url.startswith("postgres://")
        ):
            logger.warning(f"Connection URL does not appear to be PostgreSQL: {conn_url[:30]}...")

        # Pool size should be reasonable
        pool_size = self._config.get("pool_size", 10)
        if pool_size < 1 or pool_size > 100:
            logger.warning(f"Unusual pool_size: {pool_size}")

    @staticmethod
    def _is_valid_bucket_name(name: str) -> bool:
        """Check if S3 bucket name is valid.

        S3 bucket naming rules:
        - 3-63 characters
        - Lowercase letters, numbers, hyphens
        - Must start and end with lowercase or number
        - No consecutive hyphens
        """
        if len(name) < 3 or len(name) > 63:
            return False

        if not name[0].isalnum() or not name[-1].isalnum():
            return False

        if "--" in name:
            return False

        return all(c.islower() or c.isdigit() or c == "-" for c in name)

    @staticmethod
    def _is_valid_aws_region(region: str) -> bool:
        """Check if AWS region name looks valid.

        Valid regions are typically: us-east-1, eu-west-1, ap-southeast-1, etc.
        """
        parts = region.split("-")
        if len(parts) < 2:
            return False

        # Check basic format (e.g., us-east-1)
        return all(part.isalnum() or part == "-" for part in parts)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation.

        Args:
            key: Configuration key

        Returns:
            Configuration value

        Raises:
            KeyError: If key not found
        """
        return self._config[key]

    def __repr__(self) -> str:
        """String representation (sanitized for security)."""
        safe_config = self._config.copy()

        # Mask sensitive values
        for sensitive_key in ["connection_url", "password", "secret", "token"]:
            if sensitive_key in safe_config:
                safe_config[sensitive_key] = "***MASKED***"

        return f"CatalogConfig({self.provider_type}, {safe_config})"


def load_catalog_config(
    provider_type: Literal["development", "enterprise", "cloud"],
    config_file: str | None = None,
    config_dict: dict[str, Any] | None = None,
    use_env: bool = True,
) -> dict[str, Any]:
    """Load catalog configuration.

    This is a convenience function for quickly loading catalog config.

    Example:
        config = load_catalog_config(
            "cloud",
            config_file="catalog.yaml",
            config_dict={"pool_size": 20},
        )
        connector = CatalogConnector(provider_type, config)

    Args:
        provider_type: Type of provider
        config_file: Optional path to config file
        config_dict: Optional programmatic config
        use_env: Whether to load from environment

    Returns:
        Loaded configuration dictionary
    """
    loader = CatalogConfig(
        provider_type=provider_type,
        config_file=config_file,
        config_dict=config_dict,
        use_env=use_env,
    )
    return loader.load()
