"""Smoke tests for resolver imports and registry wiring."""

from __future__ import annotations

import importlib

import pytest

from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
from DocsToKG.ContentDownload.resolvers import __all__ as resolver_exports
from DocsToKG.ContentDownload.resolvers.registry_v2 import build_resolvers


MODULES = [
    "DocsToKG.ContentDownload.resolvers.arxiv",
    "DocsToKG.ContentDownload.resolvers.core",
    "DocsToKG.ContentDownload.resolvers.crossref",
    "DocsToKG.ContentDownload.resolvers.doaj",
    "DocsToKG.ContentDownload.resolvers.europe_pmc",
    "DocsToKG.ContentDownload.resolvers.figshare",
    "DocsToKG.ContentDownload.resolvers.hal",
    "DocsToKG.ContentDownload.resolvers.landing_page",
    "DocsToKG.ContentDownload.resolvers.openalex",
    "DocsToKG.ContentDownload.resolvers.openaire",
    "DocsToKG.ContentDownload.resolvers.osf",
    "DocsToKG.ContentDownload.resolvers.semantic_scholar",
    "DocsToKG.ContentDownload.resolvers.unpaywall",
    "DocsToKG.ContentDownload.resolvers.wayback",
    "DocsToKG.ContentDownload.resolvers.zenodo",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_resolver_modules_import(module_name: str) -> None:
    """Ensure resolver modules import without raising exceptions."""

    importlib.import_module(module_name)


def test_build_resolvers_from_default_config() -> None:
    """Building resolver instances should succeed with default configuration."""

    for module_name in MODULES:
        importlib.import_module(module_name)

    config = ContentDownloadConfig()
    instances = build_resolvers(config)

    assert instances, "Expected at least one resolver to be constructed"
    exported_names = set(resolver_exports)
    names = {getattr(resolver, "name", resolver.__class__.__name__.lower()) for resolver in instances}
    assert names <= set(config.resolvers.order), "Registry returned unexpected resolver names"
    assert "unpaywall" in names, "Expected default unpaywall resolver to be built"
