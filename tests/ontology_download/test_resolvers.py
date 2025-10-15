"""
Ontology Resolver Tests

This module covers individual ontology resolver implementations, ensuring
they derive download plans with correct URLs, versions, licensing, and
authentication behaviour across supported providers.

Key Scenarios:
- Prefers requested formats across OBO, OLS, BioPortal, SKOS, and XBRL
- Handles API credential loading and retries on transient errors
- Validates configuration error paths for missing metadata

Dependencies:
- pytest: Fixtures and assertions
- DocsToKG.OntologyDownload.resolvers: Resolver implementations under test

Usage:
    pytest tests/ontology_download/test_resolvers.py
"""

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("requests")

import requests
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import resolvers
from DocsToKG.OntologyDownload.config import DefaultsConfiguration, ResolvedConfig
from DocsToKG.OntologyDownload.core import FetchSpec


@pytest.fixture()
def resolved_config():
    return ResolvedConfig(defaults=DefaultsConfiguration(), specs=())


def test_obo_resolver_prefers_requested_format(monkeypatch, resolved_config):
    monkeypatch.setattr(
        resolvers, "get_owl_download", lambda prefix: f"https://example.org/{prefix}.owl"
    )
    monkeypatch.setattr(
        resolvers, "get_obo_download", lambda prefix: f"https://example.org/{prefix}.obo"
    )
    monkeypatch.setattr(resolvers, "get_rdf_download", lambda prefix: None)
    spec = FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl", "obo"])
    plan = resolvers.OBOResolver().plan(spec, resolved_config, logging.getLogger(__name__))
    assert plan.url.endswith("hp.owl")


def test_ols_resolver_uses_download_link(monkeypatch, resolved_config):
    monkeypatch.setattr(
        resolvers.BaseResolver, "_execute_with_retry", lambda self, func, **kwargs: func()
    )
    record = {
        "download": "https://example.org/efo.owl",
        "version": "2024-01-01",
        "license": "CC-BY-4.0",
        "preferredPrefix": "EFO",
    }
    client = SimpleNamespace(get_ontology=lambda _: record, get_ontology_versions=lambda _: [])
    monkeypatch.setattr(resolvers, "OlsClient", lambda: client)
    resolver = resolvers.OLSResolver()
    spec = FetchSpec(id="efo", resolver="ols", extras={}, target_formats=["owl"])
    plan = resolver.plan(spec, resolved_config, logging.getLogger(__name__))
    assert plan.url == "https://example.org/efo.owl"
    assert plan.version == "2024-01-01"
    assert plan.license == "CC-BY-4.0"


def test_ols_resolver_applies_polite_headers(monkeypatch, resolved_config):
    session_headers = {}

    class StubSession:
        def __init__(self):
            self.headers = session_headers

    def get_ontology(_):
        return {"download": "https://example.org/efo.owl"}

    client = SimpleNamespace(
        get_ontology=get_ontology,
        get_ontology_versions=lambda _: [],
        session=StubSession(),
    )
    monkeypatch.setattr(resolvers, "OlsClient", lambda: client)

    resolver = resolvers.OLSResolver()
    logger = logging.LoggerAdapter(logging.getLogger(__name__), {"correlation_id": "corr123"})
    spec = FetchSpec(id="efo", resolver="ols", extras={}, target_formats=["owl"])
    resolver.plan(spec, resolved_config, logger)

    assert "User-Agent" in session_headers
    assert session_headers["X-Request-ID"].startswith("corr123-")


def test_bioportal_resolver_includes_api_key(monkeypatch, resolved_config, tmp_path):
    monkeypatch.setattr(
        resolvers.BaseResolver, "_execute_with_retry", lambda self, func, **kwargs: func()
    )
    ontology = {"license": "CC-BY"}
    submission = {"download": "https://example.org/ncit.owl", "version": "2024"}
    client = SimpleNamespace(
        get_ontology=lambda _: ontology, get_latest_submission=lambda _: submission
    )
    monkeypatch.setattr(resolvers, "BioPortalClient", lambda: client)
    api_key_path = tmp_path / "bioportal_api_key.txt"
    api_key_path.write_text("secret")
    monkeypatch.setattr(resolvers.pystow, "join", lambda *parts: tmp_path)
    resolver = resolvers.BioPortalResolver()
    spec = FetchSpec(
        id="ncit", resolver="bioportal", extras={"acronym": "NCIT"}, target_formats=["owl"]
    )
    plan = resolver.plan(spec, resolved_config, logging.getLogger(__name__))
    assert plan.url == "https://example.org/ncit.owl"
    assert plan.headers["Authorization"] == "apikey secret"


def test_bioportal_resolver_applies_polite_headers(monkeypatch, resolved_config, tmp_path):
    session_headers = {}

    class StubSession:
        def __init__(self):
            self.headers = session_headers

    ontology = {"license": "CC-BY"}
    submission = {"download": "https://example.org/ncit.owl", "version": "2024"}
    client = SimpleNamespace(
        get_ontology=lambda _: ontology,
        get_latest_submission=lambda _: submission,
        session=StubSession(),
    )
    monkeypatch.setattr(resolvers, "BioPortalClient", lambda: client)
    monkeypatch.setattr(resolvers.pystow, "join", lambda *parts: tmp_path)

    resolver = resolvers.BioPortalResolver()
    resolver.client = client
    resolver.api_key_path = tmp_path / "bioportal_api_key.txt"
    logger = logging.LoggerAdapter(logging.getLogger(__name__), {"correlation_id": "abc456"})
    spec = FetchSpec(id="ncit", resolver="bioportal", extras={}, target_formats=["owl"])
    resolver.plan(spec, resolved_config, logger)

    assert "User-Agent" in session_headers
    assert session_headers["X-Request-ID"].startswith("abc456-")



def test_skos_resolver_requires_url(resolved_config):
    resolver = resolvers.SKOSResolver()
    spec = FetchSpec(id="eurovoc", resolver="skos", extras={}, target_formats=["ttl"])
    with pytest.raises(resolvers.ConfigError):
        resolver.plan(spec, resolved_config, logging.getLogger(__name__))


def test_xbrl_resolver_success(resolved_config):
    resolver = resolvers.XBRLResolver()
    spec = FetchSpec(
        id="ifrs",
        resolver="xbrl",
        extras={"url": "https://example.org/ifrs.zip"},
        target_formats=[],
    )
    plan = resolver.plan(spec, resolved_config, logging.getLogger(__name__))
    assert plan.media_type == "application/zip"


def test_bioportal_resolver_auth_error(monkeypatch, resolved_config):
    response = requests.Response()
    response.status_code = 401
    error = requests.HTTPError("unauthorized", response=response)

    def failing_call():
        raise error

    monkeypatch.setattr(
        resolvers.BaseResolver, "_execute_with_retry", lambda self, func, **kwargs: func()
    )
    resolver = resolvers.BioPortalResolver()
    resolver.client = SimpleNamespace(
        get_ontology=lambda acronym: failing_call(),
        get_latest_submission=lambda acronym: None,
    )
    spec = FetchSpec(id="ncit", resolver="bioportal", extras={}, target_formats=["owl"])
    with pytest.raises(resolvers.ConfigError) as exc_info:
        resolver.plan(spec, resolved_config, logging.getLogger(__name__))
    assert "bioportal".lower() in str(exc_info.value).lower()
    assert str(resolver.api_key_path) in str(exc_info.value)


def test_ols_resolver_timeout_retry(monkeypatch, resolved_config):
    attempts = {"count": 0}

    def get_ontology(_):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise requests.Timeout("timeout")
        return {"download": "https://example.org/bfo.owl"}

    resolver = resolvers.OLSResolver()
    resolver.client = SimpleNamespace(get_ontology=get_ontology, get_ontology_versions=lambda _: [])
    spec = FetchSpec(id="bfo", resolver="ols", extras={}, target_formats=["owl"])
    plan = resolver.plan(spec, resolved_config, logging.getLogger(__name__))
    assert plan.url == "https://example.org/bfo.owl"
    assert attempts["count"] == 2


def test_normalize_license_to_spdx_variants():
    assert resolvers.normalize_license_to_spdx("CC BY 4.0") == "CC-BY-4.0"
    assert resolvers.normalize_license_to_spdx("public domain") == "CC0-1.0"
    assert resolvers.normalize_license_to_spdx("Apache License 2.0") == "Apache-2.0"
    assert resolvers.normalize_license_to_spdx("Custom License") == "Custom License"


def test_lov_resolver_parses_metadata(resolved_config):
    payload = {
        "vocabulary": {
            "downloadURL": "https://example.org/voaf.ttl",
            "license": "Creative Commons Attribution 4.0",
            "version": "2024-01-01",
            "mediaType": "text/turtle",
        }
    }

    class StubResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    class StubSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, params=None, timeout=None):
            assert params["uri"] == "http://purl.org/vocommons/voaf"
            return StubResponse(payload)

    resolver = resolvers.LOVResolver(session=StubSession())
    logger = logging.LoggerAdapter(logging.getLogger(__name__), {"correlation_id": "lov123"})
    spec = FetchSpec(
        id="voaf",
        resolver="lov",
        extras={"uri": "http://purl.org/vocommons/voaf"},
        target_formats=["ttl"],
    )
    plan = resolver.plan(spec, resolved_config, logger)

    assert plan.url == "https://example.org/voaf.ttl"
    assert plan.media_type == "text/turtle"
    assert plan.license == "CC-BY-4.0"
    session_headers = resolver.session.headers
    assert session_headers["X-Request-ID"].startswith("lov123-")


def test_lov_resolver_requires_uri(resolved_config):
    resolver = resolvers.LOVResolver(session=SimpleNamespace(headers={}, get=lambda *args, **kwargs: None))
    spec = FetchSpec(id="voaf", resolver="lov", extras={}, target_formats=["ttl"])
    with pytest.raises(resolvers.ConfigError):
        resolver.plan(spec, resolved_config, logging.getLogger(__name__))


def test_ontobee_resolver_prefers_format(resolved_config):
    resolver = resolvers.OntobeeResolver()
    logger = logging.getLogger(__name__)
    spec = FetchSpec(id="HP", resolver="ontobee", extras={}, target_formats=["obo", "owl"])
    plan = resolver.plan(spec, resolved_config, logger)

    assert plan.url == "https://purl.obolibrary.org/obo/hp.obo"
    assert plan.media_type == "text/plain"


def test_ontobee_resolver_validates_identifier(resolved_config):
    resolver = resolvers.OntobeeResolver()
    spec = FetchSpec(id="invalid-id", resolver="ontobee", extras={}, target_formats=["owl"])
    with pytest.raises(resolvers.ConfigError):
        resolver.plan(spec, resolved_config, logging.getLogger(__name__))


def test_resolver_registry_includes_new_entries():
    assert "lov" in resolvers.RESOLVERS
    assert "ontobee" in resolvers.RESOLVERS
