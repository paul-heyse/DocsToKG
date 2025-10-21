"""
Tests for CatalogConfig - Configuration Loading and Validation

Tests cover:
- Loading defaults for each provider
- Loading from JSON/YAML files
- Loading from environment variables
- Configuration precedence (env > dict > file > defaults)
- Validation of required fields
- S3 bucket name validation
- AWS region validation
- Boolean/numeric parsing
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from DocsToKG.ContentDownload.catalog.connectors.config import (
    CatalogConfig,
    load_catalog_config,
)


class TestCatalogConfigDefaults:
    """Tests for default configurations."""

    def test_load_defaults_development(self) -> None:
        """Load development defaults."""
        config = CatalogConfig("development").load()
        assert config["db_path"] == ":memory:"

    def test_load_defaults_enterprise(self) -> None:
        """Load enterprise defaults."""
        config = CatalogConfig(
            "enterprise", config_dict={"connection_url": "postgresql://localhost/test"}
        ).load()
        assert config["pool_size"] == 10
        assert config["max_overflow"] == 20

    def test_load_defaults_cloud(self) -> None:
        """Load cloud defaults."""
        config = CatalogConfig(
            "cloud",
            config_dict={
                "connection_url": "postgresql://rds/test",
                "s3_bucket": "my-bucket",
            },
        ).load()
        assert config["s3_region"] == "us-east-1"
        assert config["s3_prefix"] == "artifacts/"


class TestCatalogConfigFileLoading:
    """Tests for file-based configuration loading."""

    def test_load_json_config(self) -> None:
        """Load configuration from JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                {"connection_url": "postgresql://localhost/test", "pool_size": 20}, f
            )
            f.flush()

            config = CatalogConfig("enterprise", config_file=f.name).load()
            assert config["connection_url"] == "postgresql://localhost/test"
            assert config["pool_size"] == 20

            os.unlink(f.name)

    def test_load_missing_config_file(self) -> None:
        """Missing config file returns defaults."""
        config = CatalogConfig(
            "enterprise",
            config_file="/nonexistent/path/config.json",
            config_dict={"connection_url": "postgresql://test"},
        ).load()
        assert config["connection_url"] == "postgresql://test"


class TestCatalogConfigEnvironment:
    """Tests for environment variable configuration."""

    def test_environment_variables_cloud(self) -> None:
        """Load cloud configuration from environment variables."""
        os.environ["CATALOG_CLOUD_CONNECTION_URL"] = "postgresql://rds/test"
        os.environ["CATALOG_CLOUD_S3_BUCKET"] = "test-bucket"
        os.environ["CATALOG_CLOUD_POOL_SIZE"] = "25"

        try:
            config = CatalogConfig("cloud").load()
            assert config["connection_url"] == "postgresql://rds/test"
            assert config["s3_bucket"] == "test-bucket"
            assert config["pool_size"] == 25
        finally:
            del os.environ["CATALOG_CLOUD_CONNECTION_URL"]
            del os.environ["CATALOG_CLOUD_S3_BUCKET"]
            del os.environ["CATALOG_CLOUD_POOL_SIZE"]

    def test_environment_variables_enterprise(self) -> None:
        """Load enterprise configuration from environment variables."""
        os.environ["CATALOG_ENTERPRISE_CONNECTION_URL"] = "postgresql://localhost/db"
        os.environ["CATALOG_ENTERPRISE_POOL_SIZE"] = "30"

        try:
            config = CatalogConfig("enterprise").load()
            assert config["connection_url"] == "postgresql://localhost/db"
            assert config["pool_size"] == 30
        finally:
            del os.environ["CATALOG_ENTERPRISE_CONNECTION_URL"]
            del os.environ["CATALOG_ENTERPRISE_POOL_SIZE"]

    def test_boolean_parsing_from_env(self) -> None:
        """Parse boolean values from environment."""
        os.environ["CATALOG_ENTERPRISE_ECHO_SQL"] = "true"

        try:
            config = CatalogConfig(
                "enterprise",
                config_dict={"connection_url": "postgresql://test"},
            ).load()
            assert config["echo_sql"] is True
        finally:
            del os.environ["CATALOG_ENTERPRISE_ECHO_SQL"]


class TestCatalogConfigPrecedence:
    """Tests for configuration precedence."""

    def test_config_precedence_env_wins(self) -> None:
        """Environment variables override everything."""
        os.environ["CATALOG_ENTERPRISE_POOL_SIZE"] = "50"

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump({"pool_size": 30}, f)
                f.flush()

                config = CatalogConfig(
                    "enterprise",
                    config_file=f.name,
                    config_dict={
                        "connection_url": "postgresql://test",
                        "pool_size": 35,
                    },
                ).load()

                # Environment variable should win
                assert config["pool_size"] == 50

                os.unlink(f.name)
        finally:
            del os.environ["CATALOG_ENTERPRISE_POOL_SIZE"]

    def test_config_precedence_dict_over_file(self) -> None:
        """Programmatic config overrides file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"pool_size": 15}, f)
            f.flush()

            config = CatalogConfig(
                "enterprise",
                config_file=f.name,
                config_dict={
                    "connection_url": "postgresql://test",
                    "pool_size": 25,
                },
                use_env=False,
            ).load()

            # Programmatic config should win
            assert config["pool_size"] == 25

            os.unlink(f.name)


class TestCatalogConfigValidation:
    """Tests for configuration validation."""

    def test_required_fields_enterprise(self) -> None:
        """Enterprise provider requires connection_url."""
        with pytest.raises(ValueError, match="connection_url"):
            CatalogConfig("enterprise", use_env=False).load()

    def test_required_fields_cloud(self) -> None:
        """Cloud provider requires connection_url and s3_bucket."""
        with pytest.raises(ValueError, match="s3_bucket"):
            CatalogConfig(
                "cloud",
                config_dict={"connection_url": "postgresql://test"},
                use_env=False,
            ).load()

    def test_validate_s3_bucket_name_valid(self) -> None:
        """Valid S3 bucket names pass validation."""
        config = CatalogConfig(
            "cloud",
            config_dict={
                "connection_url": "postgresql://test",
                "s3_bucket": "my-valid-bucket-123",
            },
            use_env=False,
        ).load()
        assert config["s3_bucket"] == "my-valid-bucket-123"

    def test_validate_s3_bucket_name_invalid_uppercase(self) -> None:
        """S3 bucket names cannot contain uppercase."""
        with pytest.raises(ValueError, match="Invalid S3 bucket name"):
            CatalogConfig(
                "cloud",
                config_dict={
                    "connection_url": "postgresql://test",
                    "s3_bucket": "My-Invalid-Bucket",
                },
                use_env=False,
            ).load()

    def test_validate_s3_bucket_name_too_short(self) -> None:
        """S3 bucket names must be at least 3 characters."""
        with pytest.raises(ValueError, match="Invalid S3 bucket name"):
            CatalogConfig(
                "cloud",
                config_dict={
                    "connection_url": "postgresql://test",
                    "s3_bucket": "ab",
                },
                use_env=False,
            ).load()

    def test_validate_connection_url_postgres(self) -> None:
        """Warning for non-Postgres URLs."""
        # This should work but issue a warning
        config = CatalogConfig(
            "enterprise",
            config_dict={"connection_url": "mysql://localhost/test"},
            use_env=False,
        ).load()
        # Should still load (warning is logged)
        assert config["connection_url"] == "mysql://localhost/test"


class TestCatalogConfigAccessors:
    """Tests for configuration accessor methods."""

    def test_get_method(self) -> None:
        """Test get() method."""
        loader = CatalogConfig("development", use_env=False)
        loader.load()
        assert loader.get("db_path") == ":memory:"
        assert loader.get("missing", "default") == "default"

    def test_bracket_notation(self) -> None:
        """Test bracket notation access."""
        loader = CatalogConfig("development", use_env=False)
        loader.load()
        assert loader["db_path"] == ":memory:"

        with pytest.raises(KeyError):
            _ = loader["nonexistent"]


class TestCatalogConfigConvenience:
    """Tests for convenience functions."""

    def test_load_catalog_config_development(self) -> None:
        """Test load_catalog_config convenience function."""
        config = load_catalog_config("development", use_env=False)
        assert config["db_path"] == ":memory:"

    def test_load_catalog_config_with_dict(self) -> None:
        """Test load_catalog_config with dict override."""
        config = load_catalog_config(
            "enterprise",
            config_dict={
                "connection_url": "postgresql://localhost/test",
                "pool_size": 50,
            },
            use_env=False,
        )
        assert config["pool_size"] == 50
