"""Tests for breaker configuration loader."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

try:
    import yaml
except ImportError:
    yaml = None
    pytest.skip("PyYAML not available", allow_module_level=True)

from DocsToKG.ContentDownload.breakers_loader import (
    load_breaker_config,
    _normalize_host_key,
    _parse_kv_overrides,
    _role_from_str,
)
from DocsToKG.ContentDownload.breakers import RequestRole


class TestBreakerLoader:
    """Test breaker configuration loader functionality."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_breaker_config(
            yaml_path=None,
            env={},
            cli_host_overrides=None,
            cli_role_overrides=None,
            cli_resolver_overrides=None,
        )

        assert config.defaults.fail_max == 5
        assert config.defaults.reset_timeout_s == 60
        assert config.defaults.retry_after_cap_s == 900

    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
defaults:
  fail_max: 3
  reset_timeout_s: 120
  retry_after_cap_s: 600

hosts:
  api.crossref.org:
    fail_max: 2
    reset_timeout_s: 180
  api.openalex.org:
    fail_max: 4
    reset_timeout_s: 90

resolvers:
  crossref:
    fail_max: 1
    reset_timeout_s: 300
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = load_breaker_config(
                yaml_path=yaml_path,
                env={},
                cli_host_overrides=None,
                cli_role_overrides=None,
                cli_resolver_overrides=None,
            )

            # Check defaults
            assert config.defaults.fail_max == 3
            assert config.defaults.reset_timeout_s == 120

            # Check host-specific configs
            assert "api.crossref.org" in config.hosts
            assert config.hosts["api.crossref.org"].fail_max == 2
            assert config.hosts["api.crossref.org"].reset_timeout_s == 180

            assert "api.openalex.org" in config.hosts
            assert config.hosts["api.openalex.org"].fail_max == 4

            # Check resolver configs
            assert "crossref" in config.resolvers
            assert config.resolvers["crossref"].fail_max == 1
            assert config.resolvers["crossref"].reset_timeout_s == 300

        finally:
            Path(yaml_path).unlink()

    def test_env_overrides(self):
        """Test environment variable overrides."""
        env = {
            "DOCSTOKG_BREAKER_DEFAULTS": "fail_max:2,reset:30",
            "DOCSTOKG_BREAKER__api.crossref.org": "fail_max:1,reset:60",
            "DOCSTOKG_BREAKER_ROLE__api.crossref.org__artifact": "fail_max:3,trial_calls:2",
            "DOCSTOKG_BREAKER_RESOLVER__crossref": "fail_max:4,reset:120",
            "DOCSTOKG_BREAKER_CLASSIFY": "failure=500,502,503 neutral=401,403",
            "DOCSTOKG_BREAKER_ROLLING": "enabled:true,window:20,thresh:4,cooldown:30",
        }

        config = load_breaker_config(
            yaml_path=None,
            env=env,
            cli_host_overrides=None,
            cli_role_overrides=None,
            cli_resolver_overrides=None,
        )

        # Check defaults override
        assert config.defaults.fail_max == 2
        assert config.defaults.reset_timeout_s == 30

        # Check host override
        assert "api.crossref.org" in config.hosts
        assert config.hosts["api.crossref.org"].fail_max == 1
        assert config.hosts["api.crossref.org"].reset_timeout_s == 60

        # Check role override
        artifact_role = config.hosts["api.crossref.org"].roles.get(RequestRole.ARTIFACT)
        assert artifact_role is not None
        assert artifact_role.fail_max == 3
        assert artifact_role.trial_calls == 2

        # Check resolver override
        assert "crossref" in config.resolvers
        assert config.resolvers["crossref"].fail_max == 4
        assert config.resolvers["crossref"].reset_timeout_s == 120

        # Check classification override
        assert 500 in config.classify.failure_statuses
        assert 503 in config.classify.failure_statuses
        assert 401 in config.classify.neutral_statuses
        assert 403 in config.classify.neutral_statuses

        # Check rolling window override
        assert config.rolling.enabled is True
        assert config.rolling.window_s == 20
        assert config.rolling.threshold_failures == 4
        assert config.rolling.cooldown_s == 30

    def test_cli_overrides(self):
        """Test CLI argument overrides."""
        config = load_breaker_config(
            yaml_path=None,
            env={},
            cli_host_overrides=["api.crossref.org=fail_max:1,reset:60"],
            cli_role_overrides=["api.crossref.org:artifact=fail_max:2,trial_calls:3"],
            cli_resolver_overrides=["crossref=fail_max:4,reset:120"],
            cli_defaults_override="fail_max:2,reset:30",
            cli_classify_override="failure=500,502,503 neutral=401,403",
            cli_rolling_override="enabled:true,window:20,thresh:4,cooldown:30",
        )

        # Check defaults override
        assert config.defaults.fail_max == 2
        assert config.defaults.reset_timeout_s == 30

        # Check host override
        assert "api.crossref.org" in config.hosts
        assert config.hosts["api.crossref.org"].fail_max == 1
        assert config.hosts["api.crossref.org"].reset_timeout_s == 60

        # Check role override
        artifact_role = config.hosts["api.crossref.org"].roles.get(RequestRole.ARTIFACT)
        assert artifact_role is not None
        assert artifact_role.fail_max == 2
        assert artifact_role.trial_calls == 3

        # Check resolver override
        assert "crossref" in config.resolvers
        assert config.resolvers["crossref"].fail_max == 4
        assert config.resolvers["crossref"].reset_timeout_s == 120

    def test_precedence_order(self):
        """Test that CLI overrides take precedence over env overrides."""
        env = {
            "DOCSTOKG_BREAKER_DEFAULTS": "fail_max:2,reset:30",
        }

        config = load_breaker_config(
            yaml_path=None,
            env=env,
            cli_defaults_override="fail_max:3,reset:45",
        )

        # CLI should override env
        assert config.defaults.fail_max == 3
        assert config.defaults.reset_timeout_s == 45

    def test_invalid_yaml(self):
        """Test handling of invalid YAML files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            yaml_path = f.name

        try:
            with pytest.raises(Exception):  # Should raise YAML parsing error
                load_breaker_config(
                    yaml_path=yaml_path,
                    env={},
                    cli_host_overrides=None,
                    cli_role_overrides=None,
                    cli_resolver_overrides=None,
                )
        finally:
            Path(yaml_path).unlink()

    def test_missing_yaml_file(self):
        """Test handling of missing YAML file."""
        with pytest.raises(FileNotFoundError):
            load_breaker_config(
                yaml_path="/nonexistent/path.yaml",
                env={},
                cli_host_overrides=None,
                cli_role_overrides=None,
                cli_resolver_overrides=None,
            )

    def test_validation_errors(self):
        """Test configuration validation."""
        # Test invalid fail_max
        with pytest.raises(ValueError, match="fail_max must be >=1"):
            load_breaker_config(
                yaml_path=None,
                env={"DOCSTOKG_BREAKER_DEFAULTS": "fail_max:0"},
                cli_host_overrides=None,
                cli_role_overrides=None,
                cli_resolver_overrides=None,
            )

        # Test invalid reset_timeout_s
        with pytest.raises(ValueError, match="reset_timeout_s must be >0"):
            load_breaker_config(
                yaml_path=None,
                env={"DOCSTOKG_BREAKER_DEFAULTS": "reset:0"},
                cli_host_overrides=None,
                cli_role_overrides=None,
                cli_resolver_overrides=None,
            )


class TestHelperFunctions:
    """Test helper functions."""

    def test_normalize_host_key(self):
        """Test host key normalization."""
        assert _normalize_host_key("EXAMPLE.COM") == "example.com"
        assert _normalize_host_key("Example.Com.") == "example.com"
        assert _normalize_host_key("  example.com  ") == "example.com"
        assert _normalize_host_key("") == ""

    def test_parse_kv_overrides(self):
        """Test key-value override parsing."""
        result = _parse_kv_overrides("fail_max:5,reset:60,retry_after_cap:900")
        assert result == {"fail_max": "5", "reset": "60", "retry_after_cap": "900"}

        result = _parse_kv_overrides("enabled=true,window=30")
        assert result == {"enabled": "true", "window": "30"}

        result = _parse_kv_overrides("bare_flag")
        assert result == {"bare_flag": "true"}

    def test_role_from_str(self):
        """Test role string parsing."""
        assert _role_from_str("metadata") == RequestRole.METADATA
        assert _role_from_str("meta") == RequestRole.METADATA
        assert _role_from_str("landing") == RequestRole.LANDING
        assert _role_from_str("artifact") == RequestRole.ARTIFACT

        with pytest.raises(ValueError, match="Unknown role"):
            _role_from_str("invalid")
