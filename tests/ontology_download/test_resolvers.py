"""Ontology resolver tests validating resolver contracts."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import core, resolvers
from DocsToKG.OntologyDownload.config import ConfigError, DefaultsConfig, ResolvedConfig
from DocsToKG.OntologyDownload.core import FetchSpec


@pytest.fixture()
def resolved_config():
    return ResolvedConfig(defaults=DefaultsConfig(), specs=[])


@pytest.fixture()
def load_cassette():
    base = Path("tests/ontology_download/fixtures/cassettes")

    def _load(name: str) -> dict:
        path = base / f"{name}.json"
        return json.loads(path.read_text())

    return _load


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
    assert plan.service == "obo"


def test_obo_resolver_contract(load_cassette, monkeypatch, resolved_config):
    cassette = load_cassette("obo_chebi")

    monkeypatch.setattr(resolvers, "get_owl_download", lambda prefix: cassette.get("owl"))
    monkeypatch.setattr(resolvers, "get_obo_download", lambda prefix: cassette.get("obo"))
    monkeypatch.setattr(resolvers, "get_rdf_download", lambda prefix: None)

    spec = FetchSpec(id=cassette["id"], resolver="obo", extras={}, target_formats=["owl"])
    plan = resolvers.OBOResolver().plan(spec, resolved_config, logging.getLogger(__name__))

    assert plan.url == cassette["owl"]
    assert plan.media_type == "application/rdf+xml"
    assert plan.service == "obo"


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
    assert plan.service == "ols"


def test_ols_resolver_contract(load_cassette, monkeypatch, resolved_config):
    payload = load_cassette("ols_hp")

    monkeypatch.setattr(
        resolvers.BaseResolver, "_execute_with_retry", lambda self, func, **kwargs: func()
    )

    class StubClient:
        def __init__(self, data):
            self.data = data
            self.session = SimpleNamespace(headers={})

        def get_ontology(self, ontology_id):
            return self.data["get_ontology"]

        def get_ontology_versions(self, ontology_id):
            return self.data.get("versions", [])

    client = StubClient(payload)
    monkeypatch.setattr(resolvers, "OlsClient", lambda: client)

    resolver = resolvers.OLSResolver()
    spec = FetchSpec(id="hp", resolver="ols", extras={}, target_formats=["owl"])
    plan = resolver.plan(spec, resolved_config, logging.getLogger(__name__))

    assert plan.url == payload["get_ontology"]["download"]
    assert plan.version == payload["get_ontology"]["version"]
    assert plan.license == "CC-BY-4.0"
    assert plan.service == "ols"


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
    assert plan.service == "bioportal"


def test_bioportal_resolver_contract(load_cassette, monkeypatch, resolved_config, tmp_path):
    cassette = load_cassette("bioportal_ncit")
    monkeypatch.setattr(
        resolvers.BaseResolver, "_execute_with_retry", lambda self, func, **kwargs: func()
    )
    client = SimpleNamespace(
        get_ontology=lambda acronym: cassette["ontology"],
        get_latest_submission=lambda acronym: cassette["submission"],
        session=SimpleNamespace(headers={}),
    )
    monkeypatch.setattr(resolvers, "BioPortalClient", lambda: client)
    monkeypatch.setattr(resolvers.pystow, "join", lambda *parts: tmp_path)
    api_key_path = tmp_path / "bioportal_api_key.txt"
    api_key_path.write_text(cassette["api_key"])

    resolver = resolvers.BioPortalResolver()
    spec = FetchSpec(id="NCIT", resolver="bioportal", extras={"acronym": "NCIT"}, target_formats=["owl"])
    plan = resolver.plan(spec, resolved_config, logging.getLogger(__name__))

    assert plan.url == cassette["submission"]["download"]
    assert plan.version == cassette["submission"]["version"]
    assert plan.headers["Authorization"] == f"apikey {cassette['api_key']}"
    assert plan.license == "CC-BY-4.0"


def test_bioportal_resolver_applies_polite_headers(monkeypatch, resolved_config, tmp_path):
    session_headers = {}

    class StubSession:
        def __init__(self):
            self.headers = session_headers

    resolved_config.defaults.http.polite_headers["mailto"] = "team@example.org"

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
    assert session_headers["From"] == "team@example.org"


def test_lov_resolver_contract(load_cassette, monkeypatch, resolved_config):
    cassette = load_cassette("lov_schema")
    response_payload = cassette["response"]

    class StubResponse:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                error = requests.HTTPError("lov error")
                error.response = SimpleNamespace(status_code=self.status_code)
                raise error

        def json(self):
            return self._payload

    session = SimpleNamespace(
        headers={},
        get=lambda url, params=None, timeout=0: StubResponse(response_payload),
    )
    monkeypatch.setattr(
        resolvers.BaseResolver, "_execute_with_retry", lambda self, func, **kwargs: func()
    )
    resolver = resolvers.LOVResolver(session=session)
    spec = FetchSpec(
        id="schema", resolver="lov", extras={"uri": cassette["uri"]}, target_formats=["ttl"]
    )
    plan = resolver.plan(spec, resolved_config, logging.getLogger(__name__))

    assert plan.url == response_payload["downloadURL"]
    assert plan.license == "CC-BY-4.0"
    assert plan.version == response_payload["version"]
    assert plan.media_type == "text/turtle"
    assert plan.service == "lov"


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
    assert plan.service == "xbrl"


def test_bioportal_resolver_auth_error(load_cassette, monkeypatch, resolved_config):
    failure = load_cassette("bioportal_auth_failure")
    response = requests.Response()
    response.status_code = failure["status"]
    error = requests.HTTPError(failure["message"], response=response)

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
    assert "bioportal" in str(exc_info.value).lower()
    assert "status 401" in str(exc_info.value).lower()


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


def test_ontobee_resolver_contract(load_cassette, resolved_config):
    cassette = load_cassette("ontobee_hp")
    spec = FetchSpec(
        id=cassette["id"],
        resolver="ontobee",
        extras={},
        target_formats=cassette["formats"],
    )
    plan = resolvers.OntobeeResolver().plan(spec, resolved_config, logging.getLogger(__name__))
    assert plan.url.endswith("hp.obo")
    assert plan.service == "ontobee"


def test_resolver_uses_service_rate_limit(monkeypatch, resolved_config):
    calls = {"count": 0}

    class DummyBucket:
        def consume(self, tokens: float = 1.0) -> None:
            calls["count"] += 1

    monkeypatch.setattr(resolvers, "_get_service_bucket", lambda service, config: DummyBucket())
    monkeypatch.setattr(resolvers, "retry_with_backoff", lambda func, **kwargs: func())

    record = {"download": "https://example.org/efo.owl"}
    client = SimpleNamespace(get_ontology=lambda _: record, get_ontology_versions=lambda _: [])
    monkeypatch.setattr(resolvers, "OlsClient", lambda: client)

    resolver = resolvers.OLSResolver()
    spec = FetchSpec(id="efo", resolver="obo", extras={}, target_formats=["owl"])
    plan = resolver.plan(spec, resolved_config, logging.getLogger(__name__))

    assert plan.service == "ols"
    assert calls["count"] >= 1


def test_lov_resolver_respects_timeout_and_rate_limit(monkeypatch, resolved_config):
    captured = {"timeout": None, "service": None, "consumes": 0}

    class StubResponse:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "downloadURL": "https://example.org/schema.ttl",
                "license": "CC-BY-4.0",
                "version": "2024-01-01",
                "mediaType": "text/turtle",
            }

    class StubSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, params=None, timeout=None):
            captured["timeout"] = timeout
            return StubResponse()

    class StubBucket:
        def consume(self, tokens: float = 1.0) -> None:
            captured["consumes"] += 1

    def _fake_get_bucket(service, config):
        captured["service"] = service
        return StubBucket()

    monkeypatch.setattr(resolvers, "_get_service_bucket", _fake_get_bucket)
    monkeypatch.setattr(resolvers, "retry_with_backoff", lambda func, **kwargs: func())

    resolver = resolvers.LOVResolver(session=StubSession())
    resolved_config.defaults.http.timeout_sec = 9
    spec = FetchSpec(
        id="schema",
        resolver="lov",
        extras={"uri": "http://example.org/schema"},
        target_formats=["ttl"],
    )

    plan = resolver.plan(spec, resolved_config, logging.getLogger(__name__))

    assert captured["timeout"] == 9
    assert captured["service"] == "lov"
    assert captured["consumes"] == 1
    assert plan.url == "https://example.org/schema.ttl"


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
    assert plan.service == "lov"
    session_headers = resolver.session.headers
    assert session_headers["X-Request-ID"].startswith("lov123-")


def test_lov_resolver_requires_uri(resolved_config):
    resolver = resolvers.LOVResolver(
        session=SimpleNamespace(headers={}, get=lambda *args, **kwargs: None)
    )
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
    assert plan.service == "ontobee"


def test_ontobee_resolver_validates_identifier(resolved_config):
    resolver = resolvers.OntobeeResolver()
    spec = FetchSpec(id="invalid-id", resolver="ontobee", extras={}, target_formats=["owl"])
    with pytest.raises(resolvers.ConfigError):
        resolver.plan(spec, resolved_config, logging.getLogger(__name__))


def test_resolver_registry_includes_new_entries():
    assert "lov" in resolvers.RESOLVERS
    assert "ontobee" in resolvers.RESOLVERS


def test_resolver_fallback_chain_on_failure(monkeypatch, resolved_config):
    class PrimaryResolver:
        def plan(self, spec, config, logger):
            raise ConfigError("primary unavailable")

    fallback_plan = resolvers.FetchPlan(
        url="https://fallback.example.org/hp.owl",
        headers={},
        filename_hint="hp.owl",
        version="2024-01-01",
        license="CC-BY-4.0",
        media_type="application/rdf+xml",
        service="ontobee",
    )

    class SecondaryResolver:
        def plan(self, spec, config, logger):
            return fallback_plan

    monkeypatch.setitem(resolvers.RESOLVERS, "primary", PrimaryResolver())
    monkeypatch.setitem(resolvers.RESOLVERS, "secondary", SecondaryResolver())

    resolved_config.defaults.prefer_source = ["primary", "secondary"]
    spec = FetchSpec(id="hp", resolver="primary", extras={}, target_formats=["owl"])

    monkeypatch.setattr(
        core,
        "setup_logging",
        lambda level=None, retention_days=None, max_log_size_mb=None: logging.getLogger(
            "resolver-test"
        ),
    )

    planned = core.plan_one(spec, config=resolved_config)

    assert planned.resolver == "secondary"
    assert [candidate.resolver for candidate in planned.candidates] == ["secondary"]
    assert planned.plan.url == "https://fallback.example.org/hp.owl"
