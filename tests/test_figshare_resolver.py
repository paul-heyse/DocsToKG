import json
import sys
import types
from pathlib import Path

import pytest

if "pyalex" not in sys.modules:
    pyalex_stub = types.ModuleType("pyalex")
    pyalex_stub.Topics = object
    pyalex_stub.Works = object
    config_stub = types.ModuleType("pyalex.config")
    config_stub.mailto = None
    pyalex_stub.config = config_stub
    sys.modules["pyalex"] = pyalex_stub
    sys.modules["pyalex.config"] = config_stub

pytest.importorskip("pyalex")

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload.resolvers import ResolverConfig
from DocsToKG.ContentDownload.resolvers.providers.figshare import FigshareResolver

requests = pytest.importorskip("requests")
responses = pytest.importorskip("responses")


def load_fixture(name: str):
    fixture_path = Path(__file__).parent / "data" / name
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        match=[
            responses.matchers.json_params_matcher(
                {"search_for": ':doi: "10.6084/m9.figshare.123456"', "page": 1, "page_size": 3}
            )
        ],
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
    assert "boom" in events[0].metadata["message"]


def test_figshare_resolver_disabled_without_doi(tmp_path):
    artifact = make_artifact(tmp_path, doi=None)
    config = ResolverConfig()

    assert not FigshareResolver().is_enabled(config, artifact)

    session = requests.Session()
    results = list(FigshareResolver().iter_urls(session, config, artifact))
    assert len(results) == 1
    assert results[0].event == "skipped"
    assert results[0].event_reason == "no-doi"
