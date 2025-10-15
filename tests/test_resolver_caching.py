"""
Resolver Caching Tests

This module confirms resolver implementations reuse cached HTTP
responses to avoid redundant traffic across repeated DOI lookups while
still allowing caches to be reset between test cases.

Key Scenarios:
- Ensures Unpaywall, Crossref, and Semantic Scholar resolvers share caches
- Verifies cache clearing restores original network behaviour

Dependencies:
- pytest: Fixtures and monkeypatching
- DocsToKG.ContentDownload.resolvers: Resolver implementations under test

Usage:
    pytest tests/test_resolver_caching.py
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("requests")

from DocsToKG.ContentDownload.resolvers import ResolverConfig, clear_resolver_caches
from DocsToKG.ContentDownload.resolvers.providers.crossref import CrossrefResolver
from DocsToKG.ContentDownload.resolvers.providers.semantic_scholar import (
    SemanticScholarResolver,
)
from DocsToKG.ContentDownload.resolvers.providers.unpaywall import UnpaywallResolver


@pytest.fixture(autouse=True)
def clear_cache_between_tests():
    clear_resolver_caches()
    yield
    clear_resolver_caches()


def test_resolver_caches_prevent_duplicate_requests(monkeypatch):
    calls: list[str] = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                if "unpaywall" in url:
                    return {
                        "best_oa_location": {"url_for_pdf": "https://example.org/unpaywall.pdf"}
                    }
                if "crossref" in url:
                    return {
                        "message": {
                            "link": [
                                {
                                    "URL": "https://example.org/crossref.pdf",
                                    "content-type": "application/pdf",
                                }
                            ]
                        }
                    }
                if "semanticscholar" in url:
                    return {"openAccessPdf": {"url": "https://example.org/s2.pdf"}}
                raise AssertionError(f"unexpected URL {url}")

        return _Resp()

    monkeypatch.setattr("DocsToKG.ContentDownload.resolvers.requests.get", fake_get)

    config = ResolverConfig()
    config.unpaywall_email = "test@example.org"
    config.mailto = "test@example.org"
    config.polite_headers = {"User-Agent": "Test"}
    config.semantic_scholar_api_key = "key"

    artifact = SimpleNamespace(doi="10.1234/test")

    session = None

    unpaywall = UnpaywallResolver()
    list(unpaywall.iter_urls(session, config, artifact))
    list(unpaywall.iter_urls(session, config, artifact))
    crossref = CrossrefResolver()
    list(crossref.iter_urls(session, config, artifact))
    list(crossref.iter_urls(session, config, artifact))
    s2 = SemanticScholarResolver()
    list(s2.iter_urls(session, config, artifact))
    list(s2.iter_urls(session, config, artifact))

    assert len(calls) == 3

    clear_resolver_caches()
    list(unpaywall.iter_urls(session, config, artifact))
    assert len(calls) == 4
