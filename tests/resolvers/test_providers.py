# === NAVMAP v1 ===
# {
#   "module": "tests.resolvers.test_providers",
#   "purpose": "Pytest coverage for resolvers providers scenarios",
#   "sections": [
#     {
#       "id": "load-fixture",
#       "name": "load_fixture",
#       "anchor": "function-load-fixture",
#       "kind": "function"
#     },
#     {
#       "id": "make-artifact",
#       "name": "make_artifact",
#       "anchor": "function-make-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "test-figshare-resolver-success",
#       "name": "test_figshare_resolver_success",
#       "anchor": "function-test-figshare-resolver-success",
#       "kind": "function"
#     },
#     {
#       "id": "test-figshare-resolver-filters-non-pdf-files",
#       "name": "test_figshare_resolver_filters_non_pdf_files",
#       "anchor": "function-test-figshare-resolver-filters-non-pdf-files",
#       "kind": "function"
#     },
#     {
#       "id": "test-figshare-resolver-multiple-pdfs",
#       "name": "test_figshare_resolver_multiple_pdfs",
#       "anchor": "function-test-figshare-resolver-multiple-pdfs",
#       "kind": "function"
#     },
#     {
#       "id": "test-figshare-resolver-no-matches",
#       "name": "test_figshare_resolver_no_matches",
#       "anchor": "function-test-figshare-resolver-no-matches",
#       "kind": "function"
#     },
#     {
#       "id": "test-figshare-resolver-http-error",
#       "name": "test_figshare_resolver_http_error",
#       "anchor": "function-test-figshare-resolver-http-error",
#       "kind": "function"
#     },
#     {
#       "id": "test-figshare-resolver-json-error",
#       "name": "test_figshare_resolver_json_error",
#       "anchor": "function-test-figshare-resolver-json-error",
#       "kind": "function"
#     },
#     {
#       "id": "test-figshare-resolver-network-error",
#       "name": "test_figshare_resolver_network_error",
#       "anchor": "function-test-figshare-resolver-network-error",
#       "kind": "function"
#     },
#     {
#       "id": "test-figshare-resolver-disabled-without-doi",
#       "name": "test_figshare_resolver_disabled_without_doi",
#       "anchor": "function-test-figshare-resolver-disabled-without-doi",
#       "kind": "function"
#     },
#     {
#       "id": "load-fixture",
#       "name": "load_fixture",
#       "anchor": "function-load-fixture",
#       "kind": "function"
#     },
#     {
#       "id": "make-artifact",
#       "name": "make_artifact",
#       "anchor": "function-make-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "test-zenodo-resolver-success",
#       "name": "test_zenodo_resolver_success",
#       "anchor": "function-test-zenodo-resolver-success",
#       "kind": "function"
#     },
#     {
#       "id": "test-zenodo-resolver-filters-non-pdf-files",
#       "name": "test_zenodo_resolver_filters_non_pdf_files",
#       "anchor": "function-test-zenodo-resolver-filters-non-pdf-files",
#       "kind": "function"
#     },
#     {
#       "id": "test-zenodo-resolver-no-matches",
#       "name": "test_zenodo_resolver_no_matches",
#       "anchor": "function-test-zenodo-resolver-no-matches",
#       "kind": "function"
#     },
#     {
#       "id": "test-zenodo-resolver-http-error",
#       "name": "test_zenodo_resolver_http_error",
#       "anchor": "function-test-zenodo-resolver-http-error",
#       "kind": "function"
#     },
#     {
#       "id": "test-zenodo-resolver-json-error",
#       "name": "test_zenodo_resolver_json_error",
#       "anchor": "function-test-zenodo-resolver-json-error",
#       "kind": "function"
#     },
#     {
#       "id": "test-zenodo-resolver-network-error",
#       "name": "test_zenodo_resolver_network_error",
#       "anchor": "function-test-zenodo-resolver-network-error",
#       "kind": "function"
#     },
#     {
#       "id": "test-zenodo-resolver-disabled-without-doi",
#       "name": "test_zenodo_resolver_disabled_without_doi",
#       "anchor": "function-test-zenodo-resolver-disabled-without-doi",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Consolidated provider-specific resolver tests."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload.resolvers import FigshareResolver, ResolverConfig, ZenodoResolver

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# --- test_figshare_resolver.py ---

if "pyalex" not in sys.modules:
    pyalex_stub = types.ModuleType("pyalex")
    pyalex_stub.Topics = object
    pyalex_stub.Works = object
    config_stub = types.ModuleType("pyalex.config")
    config_stub.mailto = None
    pyalex_stub.config = config_stub
    sys.modules["pyalex"] = pyalex_stub
    sys.modules["pyalex.config"] = config_stub

# --- test_figshare_resolver.py ---

pytest.importorskip("pyalex")

# --- test_figshare_resolver.py ---

requests = pytest.importorskip("requests")

# --- test_figshare_resolver.py ---

responses = pytest.importorskip("responses")


# --- test_figshare_resolver.py ---

def load_fixture(name: str):
    fixture_path = DATA_DIR / name
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# --- test_figshare_resolver.py ---

def make_artifact(tmp_path: Path, **overrides):
    base_kwargs = dict(
        work_id="W-FIGSHARE",
        title="Figshare Example",
        publication_year=2024,
        doi="10.6084/m9.figshare.123456",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="figshare-example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    base_kwargs.update(overrides)
    return downloader.WorkArtifact(**base_kwargs)


# --- test_figshare_resolver.py ---

@responses.activate
def test_figshare_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.POST,
        "https://api.figshare.com/v2/articles/search",
        json=load_fixture("figshare_response_sample.json"),
        status=200,
    )

    results = [
        result
        for result in FigshareResolver().iter_urls(session, config, artifact)
        if not result.is_event
    ]

    assert len(results) == 1
    assert results[0].url == "https://figshare.com/ndownloader/files/45678"
    assert results[0].metadata["source"] == "figshare"
    assert results[0].metadata["article_id"] == 123456


# --- test_figshare_resolver.py ---

@responses.activate
def test_figshare_resolver_filters_non_pdf_files(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    payload = load_fixture("figshare_response_sample.json")
    responses.add(
        responses.POST,
        "https://api.figshare.com/v2/articles/search",
        json=payload,
        status=200,
    )

    urls = [
        result.url
        for result in FigshareResolver().iter_urls(session, config, artifact)
        if not result.is_event
    ]

    assert urls == ["https://figshare.com/ndownloader/files/45678"]


# --- test_figshare_resolver.py ---

@responses.activate
def test_figshare_resolver_multiple_pdfs(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.POST,
        "https://api.figshare.com/v2/articles/search",
        json=load_fixture("figshare_response_multiple_pdf.json"),
        status=200,
    )

    urls = [
        result.url
        for result in FigshareResolver().iter_urls(session, config, artifact)
        if not result.is_event
    ]

    assert urls == [
        "https://figshare.com/ndownloader/files/1",
        "https://figshare.com/ndownloader/files/2",
    ]


# --- test_figshare_resolver.py ---

@responses.activate
def test_figshare_resolver_no_matches(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.POST,
        "https://api.figshare.com/v2/articles/search",
        json=load_fixture("figshare_response_empty.json"),
        status=200,
    )

    results = list(FigshareResolver().iter_urls(session, config, artifact))
    assert results == []


# --- test_figshare_resolver.py ---

@responses.activate
def test_figshare_resolver_http_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.POST,
        "https://api.figshare.com/v2/articles/search",
        status=404,
    )

    events = [
        result
        for result in FigshareResolver().iter_urls(session, config, artifact)
        if result.is_event
    ]

    assert events[0].event == "error"
    assert events[0].event_reason == "http-error"
    assert events[0].http_status == 404
    assert "Figshare API returned" in events[0].metadata["error_detail"]


# --- test_figshare_resolver.py ---

@responses.activate
def test_figshare_resolver_json_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.POST,
        "https://api.figshare.com/v2/articles/search",
        body="not-json",
        status=200,
    )

    events = [
        result
        for result in FigshareResolver().iter_urls(session, config, artifact)
        if result.is_event
    ]

    assert events[0].event == "error"
    assert events[0].event_reason == "json-error"
    assert "content_preview" in events[0].metadata


# --- test_figshare_resolver.py ---

@responses.activate
def test_figshare_resolver_network_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.POST,
        "https://api.figshare.com/v2/articles/search",
        body=requests.ConnectionError("boom"),
    )

    events = [
        result
        for result in FigshareResolver().iter_urls(session, config, artifact)
        if result.is_event
    ]

    assert events[0].event == "error"
    assert events[0].event_reason == "request-error"
    assert "boom" in events[0].metadata["error"]


# --- test_figshare_resolver.py ---

def test_figshare_resolver_disabled_without_doi(tmp_path):
    artifact = make_artifact(tmp_path, doi=None)
    config = ResolverConfig()

    assert not FigshareResolver().is_enabled(config, artifact)

    session = requests.Session()
    results = list(FigshareResolver().iter_urls(session, config, artifact))
    assert len(results) == 1
    assert results[0].event == "skipped"
    assert results[0].event_reason == "no-doi"


# --- test_zenodo_resolver.py ---

def load_fixture(name: str) -> dict:  # noqa: F811
    """Load JSON fixture data from the shared tests/data directory."""

    fixture_path = DATA_DIR / name
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# --- test_zenodo_resolver.py ---

def make_artifact(tmp_path: Path, **overrides: object) -> downloader.WorkArtifact:  # noqa: F811
    """Create a WorkArtifact populated with sensible defaults for testing."""

    base_kwargs = dict(
        work_id="W-ZENODO",
        title="Zenodo Example",
        publication_year=2024,
        doi="10.5281/zenodo.1234567",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="zenodo-example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    base_kwargs.update(overrides)
    return downloader.WorkArtifact(**base_kwargs)


# --- test_zenodo_resolver.py ---

@responses.activate
def test_zenodo_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.GET,
        "https://zenodo.org/api/records/",
        json=load_fixture("zenodo_response_sample.json"),
        status=200,
        match=[
            responses.matchers.query_param_matcher(
                {"q": 'doi:"10.5281/zenodo.1234567"', "size": "3", "sort": "mostrecent"}
            )
        ],
    )

    results = [
        result
        for result in ZenodoResolver().iter_urls(session, config, artifact)
        if not result.is_event
    ]

    assert len(results) == 1
    assert results[0].url == "https://zenodo.org/api/files/abc123/paper.pdf"
    assert results[0].metadata["source"] == "zenodo"
    assert results[0].metadata["record_id"] == "1234567"


# --- test_zenodo_resolver.py ---

@responses.activate
def test_zenodo_resolver_filters_non_pdf_files(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    payload = load_fixture("zenodo_response_sample.json")
    responses.add(
        responses.GET,
        "https://zenodo.org/api/records/",
        json=payload,
        status=200,
    )

    urls = [
        result.url
        for result in ZenodoResolver().iter_urls(session, config, artifact)
        if not result.is_event
    ]

    assert urls == ["https://zenodo.org/api/files/abc123/paper.pdf"]


# --- test_zenodo_resolver.py ---

@responses.activate
def test_zenodo_resolver_no_matches(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.GET,
        "https://zenodo.org/api/records/",
        json=load_fixture("zenodo_response_empty.json"),
        status=200,
    )

    results = list(ZenodoResolver().iter_urls(session, config, artifact))
    assert results == []


# --- test_zenodo_resolver.py ---

@responses.activate
def test_zenodo_resolver_http_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.GET,
        "https://zenodo.org/api/records/",
        status=404,
    )

    events = [
        result
        for result in ZenodoResolver().iter_urls(session, config, artifact)
        if result.is_event
    ]

    assert events[0].event == "error"
    assert events[0].event_reason == "http-error"
    assert events[0].http_status == 404
    assert "Zenodo API returned" in events[0].metadata["error_detail"]


# --- test_zenodo_resolver.py ---

@responses.activate
def test_zenodo_resolver_json_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.GET,
        "https://zenodo.org/api/records/",
        body="not-json",
        status=200,
    )

    events = [
        result
        for result in ZenodoResolver().iter_urls(session, config, artifact)
        if result.is_event
    ]

    assert events[0].event == "error"
    assert events[0].event_reason == "json-error"
    assert "error_detail" in events[0].metadata


# --- test_zenodo_resolver.py ---

@responses.activate
def test_zenodo_resolver_network_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = ResolverConfig()

    responses.add(
        responses.GET,
        "https://zenodo.org/api/records/",
        body=requests.ConnectionError("boom"),
    )

    events = [
        result
        for result in ZenodoResolver().iter_urls(session, config, artifact)
        if result.is_event
    ]

    assert events[0].event == "error"
    assert events[0].event_reason == "request-error"
    assert "boom" in events[0].metadata["error"]


# --- test_zenodo_resolver.py ---

def test_zenodo_resolver_disabled_without_doi(tmp_path):
    artifact = make_artifact(tmp_path, doi=None)
    config = ResolverConfig()

    assert not ZenodoResolver().is_enabled(config, artifact)

    session = requests.Session()
    results = list(ZenodoResolver().iter_urls(session, config, artifact))
    assert len(results) == 1
    assert results[0].event == "skipped"
    assert results[0].event_reason == "no-doi"
