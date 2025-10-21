"""
Tests for ContentDownloadConfig immutability (frozen).

Verifies that ContentDownloadConfig cannot be modified after creation,
ensuring reproducibility and preventing accidental runtime mutations.
"""

import pytest
from pydantic import ValidationError

from DocsToKG.ContentDownload.config import ContentDownloadConfig


class TestConfigFrozen:
    """Tests for frozen configuration."""

    def test_config_cannot_be_modified_after_creation(self):
        """Verify that config fields cannot be modified after instantiation."""
        config = ContentDownloadConfig()

        with pytest.raises((ValidationError, TypeError)):
            config.run_id = "modified"  # type: ignore

    def test_config_http_subconfig_frozen(self):
        """Verify that nested http config is also frozen."""
        config = ContentDownloadConfig()

        # Attempt to modify nested config
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            config.http.user_agent = "modified"  # type: ignore

    def test_config_resolvers_subconfig_frozen(self):
        """Verify that nested resolvers config is also frozen."""
        config = ContentDownloadConfig()

        # Attempt to modify nested config
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            config.resolvers.order = ["modified"]  # type: ignore

    def test_config_hash_deterministic(self):
        """Verify config hash is deterministic (same config = same hash)."""
        config1 = ContentDownloadConfig()
        config2 = ContentDownloadConfig()

        assert config1.config_hash() == config2.config_hash()

    def test_config_hash_differs_with_different_run_id(self):
        """Verify config hash differs when config differs."""
        config1 = ContentDownloadConfig(run_id="run1")
        config2 = ContentDownloadConfig(run_id="run2")

        assert config1.config_hash() != config2.config_hash()

    def test_verify_immutable_returns_true(self):
        """Verify that verify_immutable() returns True for frozen config."""
        config = ContentDownloadConfig()
        assert config.verify_immutable() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
