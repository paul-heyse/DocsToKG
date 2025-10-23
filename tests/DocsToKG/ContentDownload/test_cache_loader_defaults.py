"""Regression tests for cache loader defaults."""

from DocsToKG.ContentDownload.cache_loader import CacheDefault, load_cache_config


def test_load_cache_config_allows_empty_hosts_when_default_do_not_cache() -> None:
    """Ensure loading without YAML/env input produces a usable configuration."""

    config = load_cache_config(None, env={})

    assert config.controller.default is CacheDefault.DO_NOT_CACHE
    assert config.hosts == {}
