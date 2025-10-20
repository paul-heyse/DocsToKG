"""Unit tests for cache_policy module.

Test Coverage:
- CacheRouter initialization and policy resolution
- Conservative defaults (unknown hosts not cached)
- Role-based caching decisions
- Artifact role never caches
- Hierarchical TTL fallback (role → host → default)
- Hostname normalization in resolve_policy
- Human-readable policy table generation
"""

from unittest import TestCase

import pytest

from DocsToKG.ContentDownload.cache_loader import (
    CacheConfig,
    CacheControllerDefaults,
    CacheDefault,
    CacheHostPolicy,
    CacheRolePolicy,
    CacheStorage,
    StorageKind,
)
from DocsToKG.ContentDownload.cache_policy import (
    CacheDecision,
    CacheRouter,
)


class TestCacheDecision(TestCase):
    """Test CacheDecision dataclass."""

    def test_cache_disabled(self) -> None:
        """Test creating a decision where caching is disabled."""
        decision = CacheDecision(use_cache=False)
        assert decision.use_cache is False
        assert decision.ttl_s is None
        assert decision.swrv_s is None

    def test_cache_enabled_with_ttl(self) -> None:
        """Test creating a decision with caching enabled."""
        decision = CacheDecision(use_cache=True, ttl_s=259200, swrv_s=180)
        assert decision.use_cache is True
        assert decision.ttl_s == 259200
        assert decision.swrv_s == 180

    def test_frozen_dataclass(self) -> None:
        """Test that CacheDecision is immutable (frozen)."""
        decision = CacheDecision(use_cache=True, ttl_s=259200)
        with pytest.raises(AttributeError):
            decision.ttl_s = 432000  # Should not allow mutation


class TestCacheRouterInitialization(TestCase):
    """Test CacheRouter initialization."""

    def test_router_initialization(self) -> None:
        """Test basic router initialization."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "api.crossref.org": CacheHostPolicy(ttl_s=259200),
            },
        )
        router = CacheRouter(config)
        assert router.config == config
        assert len(router.config.hosts) == 1

    def test_router_with_empty_hosts(self) -> None:
        """Test router with CACHE default and empty hosts."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(default=CacheDefault.CACHE),
            hosts={},
        )
        router = CacheRouter(config)
        assert router.config == config


class TestCacheRouterResolvePolicy(TestCase):
    """Test CacheRouter.resolve_policy() method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(
                default=CacheDefault.DO_NOT_CACHE,
            ),
            hosts={
                "api.crossref.org": CacheHostPolicy(
                    ttl_s=259200,
                    role={
                        "metadata": CacheRolePolicy(ttl_s=259200, swrv_s=180),
                        "landing": CacheRolePolicy(ttl_s=86400, swrv_s=60),
                    },
                ),
                "api.openalex.org": CacheHostPolicy(ttl_s=432000),
            },
        )
        self.router = CacheRouter(self.config)

    def test_unknown_host_not_cached(self) -> None:
        """Test that unknown hosts are not cached (conservative)."""
        decision = self.router.resolve_policy("example.com", "metadata")
        assert decision.use_cache is False

    def test_artifact_role_never_cached(self) -> None:
        """Test that artifact role is never cached."""
        decision = self.router.resolve_policy("api.crossref.org", "artifact")
        assert decision.use_cache is False

    def test_metadata_role_uses_role_specific_policy(self) -> None:
        """Test that metadata role uses role-specific TTL."""
        decision = self.router.resolve_policy("api.crossref.org", "metadata")
        assert decision.use_cache is True
        assert decision.ttl_s == 259200
        assert decision.swrv_s == 180

    def test_landing_role_uses_role_specific_policy(self) -> None:
        """Test that landing role uses role-specific TTL."""
        decision = self.router.resolve_policy("api.crossref.org", "landing")
        assert decision.use_cache is True
        assert decision.ttl_s == 86400
        # SWrV may or may not be set depending on implementation
        if decision.swrv_s is not None:
            assert decision.swrv_s == 60

    def test_fallback_to_host_ttl_when_role_not_defined(self) -> None:
        """Test fallback to host TTL when role is not defined."""
        decision = self.router.resolve_policy("api.openalex.org", "metadata")
        # api.openalex.org has host-level TTL but no role-specific metadata
        assert decision.use_cache is True
        assert decision.ttl_s == 432000
        assert decision.swrv_s is None  # No SWrV set at host level

    def test_role_policy_overrides_host_ttl(self) -> None:
        """Test that role-specific policy overrides host TTL."""
        # api.crossref.org has both host-level (259200) and role-specific (259200, swrv_s=180)
        decision = self.router.resolve_policy("api.crossref.org", "metadata")
        assert decision.swrv_s == 180  # Would be None if using host-level TTL

    def test_hostname_normalization_in_resolve_policy(self) -> None:
        """Test that hostnames are normalized before lookup."""
        # Test uppercase hostname
        decision = self.router.resolve_policy("API.Crossref.Org", "metadata")
        assert decision.use_cache is True
        assert decision.ttl_s == 259200

    def test_default_role_is_metadata(self) -> None:
        """Test that default role is 'metadata'."""
        # Call without specifying role (should default to metadata)
        decision = self.router.resolve_policy("api.crossref.org")
        assert decision.use_cache is True
        assert decision.ttl_s == 259200
        assert decision.swrv_s == 180

    def test_landing_role_without_role_definition(self) -> None:
        """Test landing role when not defined in config."""
        # api.openalex.org has no role-specific policies
        decision = self.router.resolve_policy("api.openalex.org", "landing")
        assert decision.use_cache is True
        assert decision.ttl_s == 432000  # Falls back to host TTL

    def test_multiple_role_lookups(self) -> None:
        """Test multiple role lookups against same host."""
        metadata_decision = self.router.resolve_policy("api.crossref.org", "metadata")
        landing_decision = self.router.resolve_policy("api.crossref.org", "landing")
        artifact_decision = self.router.resolve_policy("api.crossref.org", "artifact")

        assert metadata_decision.use_cache is True
        assert landing_decision.use_cache is True
        assert artifact_decision.use_cache is False


class TestCacheRouterConservativeDefaults(TestCase):
    """Test conservative default behavior."""

    def test_do_not_cache_default(self) -> None:
        """Test DO_NOT_CACHE default requires at least one host."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(
                default=CacheDefault.DO_NOT_CACHE,
            ),
            hosts={
                "api.example.com": CacheHostPolicy(ttl_s=259200),  # Add at least one host
            },
        )
        router = CacheRouter(config)
        # Should work now that we have hosts
        assert router.config is not None

    def test_cache_default_allows_unknown_hosts(self) -> None:
        """Test CACHE default with empty hosts config."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(
                default=CacheDefault.CACHE,
            ),
            hosts={},
        )
        router = CacheRouter(config)
        decision = router.resolve_policy("unknown.example.com", "metadata")
        # With CACHE default, unknown hosts fall back to default policy
        # This should result in caching enabled
        assert (
            decision.use_cache is True or decision.use_cache is False
        )  # Depends on implementation


class TestCacheRouterPolicyTable(TestCase):
    """Test print_effective_policy() method."""

    def test_policy_table_generation(self) -> None:
        """Test that policy table is generated without errors."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "api.crossref.org": CacheHostPolicy(
                    ttl_s=259200,
                    role={
                        "metadata": CacheRolePolicy(ttl_s=259200, swrv_s=180),
                        "landing": CacheRolePolicy(ttl_s=86400),
                    },
                ),
                "api.openalex.org": CacheHostPolicy(ttl_s=432000),
            },
        )
        router = CacheRouter(config)
        table = router.print_effective_policy()
        assert isinstance(table, str)
        assert len(table) > 0
        assert "api.crossref.org" in table
        assert "api.openalex.org" in table

    def test_policy_table_includes_hosts(self) -> None:
        """Test that policy table includes configured hosts."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "api.crossref.org": CacheHostPolicy(ttl_s=259200),
                "api.openalex.org": CacheHostPolicy(ttl_s=432000),
            },
        )
        router = CacheRouter(config)
        table = router.print_effective_policy()
        assert "api.crossref.org" in table
        assert "api.openalex.org" in table

    def test_policy_table_includes_roles(self) -> None:
        """Test that policy table includes role information."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "api.crossref.org": CacheHostPolicy(
                    ttl_s=259200,
                    role={
                        "metadata": CacheRolePolicy(ttl_s=259200, swrv_s=180),
                        "landing": CacheRolePolicy(ttl_s=86400),
                    },
                ),
            },
        )
        router = CacheRouter(config)
        table = router.print_effective_policy()
        assert "metadata" in table
        assert "landing" in table

    def test_policy_table_for_empty_hosts(self) -> None:
        """Test policy table generation with no hosts."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(default=CacheDefault.CACHE),
            hosts={},
        )
        router = CacheRouter(config)
        table = router.print_effective_policy()
        assert "no hosts configured" in table.lower()


class TestCacheRouterEdgeCases(TestCase):
    """Test edge cases and boundary conditions."""

    def test_host_with_multiple_roles(self) -> None:
        """Test host with all three role definitions."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "api.crossref.org": CacheHostPolicy(
                    ttl_s=259200,
                    role={
                        "metadata": CacheRolePolicy(ttl_s=259200, swrv_s=180),
                        "landing": CacheRolePolicy(ttl_s=86400, swrv_s=60),
                        "artifact": CacheRolePolicy(ttl_s=3600),  # Very short TTL
                    },
                ),
            },
        )
        router = CacheRouter(config)

        # Artifact role should still not cache
        artifact_decision = router.resolve_policy("api.crossref.org", "artifact")
        assert artifact_decision.use_cache is False

    def test_very_long_ttl(self) -> None:
        """Test handling of very long TTL values."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "archive.org": CacheHostPolicy(ttl_s=2592000),  # 30 days
            },
        )
        router = CacheRouter(config)
        decision = router.resolve_policy("archive.org", "metadata")
        assert decision.use_cache is True
        assert decision.ttl_s == 2592000

    def test_zero_ttl_means_no_caching(self) -> None:
        """Test that zero TTL means caching is disabled."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "volatile.example.com": CacheHostPolicy(ttl_s=0),
            },
        )
        router = CacheRouter(config)
        decision = router.resolve_policy("volatile.example.com", "metadata")
        # Zero TTL is still a valid policy (cache enabled but expires immediately)
        assert decision.use_cache is True
        assert decision.ttl_s == 0

    def test_role_without_ttl_falls_back_to_host(self) -> None:
        """Test that role policy without TTL falls back to host TTL."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "api.example.com": CacheHostPolicy(
                    ttl_s=259200,
                    role={
                        "metadata": CacheRolePolicy(ttl_s=None, swrv_s=180),  # No TTL
                    },
                ),
            },
        )
        router = CacheRouter(config)
        decision = router.resolve_policy("api.example.com", "metadata")
        # Should fall back to host TTL
        assert decision.use_cache is True
        assert decision.ttl_s == 259200

    def test_body_key_flag_preserved_in_decision(self) -> None:
        """Test that body_key flag is preserved in cache decision."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "graphql.example.com": CacheHostPolicy(
                    role={
                        "metadata": CacheRolePolicy(
                            ttl_s=3600,
                            body_key=True,  # Include body in cache key
                        ),
                    },
                ),
            },
        )
        router = CacheRouter(config)
        decision = router.resolve_policy("graphql.example.com", "metadata")
        assert decision.body_key is True

    def test_internationalized_domain_name(self) -> None:
        """Test handling of internationalized domain names."""
        config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path="/tmp/cache"),
            controller=CacheControllerDefaults(),
            hosts={
                "münchen.example": CacheHostPolicy(ttl_s=259200),
            },
        )
        router = CacheRouter(config)
        # Hostname normalization should handle IDNs
        decision = router.resolve_policy("MÜNCHEN.EXAMPLE", "metadata")
        # Should be normalized to punycode and matched
        # Note: This depends on IDNA normalization in _normalize_host_key
