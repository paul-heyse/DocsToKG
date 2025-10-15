from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("requests")
import requests

responses = pytest.importorskip("responses")
from responses import matchers

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload import resolvers


def make_artifact(tmp_path: Path, **overrides: object) -> downloader.WorkArtifact:
    base_kwargs = dict(
        work_id="W1",
        title="Example",
        publication_year=2024,
        doi="10.1000/example",
        pmid="12345",
        pmcid="PMC67890",
        arxiv_id="2101.00001",
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    base_kwargs.update(overrides)
    return downloader.WorkArtifact(**base_kwargs)


def build_config(**overrides: object) -> resolvers.ResolverConfig:
    config = resolvers.ResolverConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@responses.activate
def test_unpaywall_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(unpaywall_email="tester@example.org")
    responses.add(
        responses.GET,
        "https://api.unpaywall.org/v2/10.1000/example",
        json={
            "best_oa_location": {"url_for_pdf": "https://oa.example/best.pdf"},
            "oa_locations": [{"url_for_pdf": "https://oa.example/extra.pdf"}],
        },
        status=200,
    )

    urls = [result.url for result in resolvers.UnpaywallResolver().iter_urls(session, config, artifact) if not result.is_event]
    assert urls == ["https://oa.example/best.pdf", "https://oa.example/extra.pdf"]


@responses.activate
def test_unpaywall_resolver_http_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(unpaywall_email="tester@example.org")
    responses.add(
        responses.GET,
        "https://api.unpaywall.org/v2/10.1000/example",
        status=503,
    )
    results = list(resolvers.UnpaywallResolver().iter_urls(session, config, artifact))
    assert results[0].event == "error"
    assert results[0].event_reason == "http-error"


@responses.activate
def test_crossref_resolver_includes_mailto(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(unpaywall_email="tester@example.org", mailto="tester@example.org")
    responses.add(
        responses.GET,
        "https://api.crossref.org/works/10.1000/example",
        json={
            "message": {
                "link": [
                    {
                        "URL": "https://publisher.example/file.pdf",
                        "content-type": "application/pdf",
                    }
                ]
            }
        },
        status=200,
    )

    urls = [r.url for r in resolvers.CrossrefResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://publisher.example/file.pdf"]
    assert "mailto=tester%40example.org" in responses.calls[0].request.url


@responses.activate
def test_crossref_resolver_handles_json_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(unpaywall_email="tester@example.org")
    responses.add(
        responses.GET,
        "https://api.crossref.org/works/10.1000/example",
        body="not-json",
        status=200,
    )
    results = list(resolvers.CrossrefResolver().iter_urls(session, config, artifact))
    assert results[0].event == "error"
    assert results[0].event_reason == "json-error"


@responses.activate
def test_landing_page_resolver_patterns(tmp_path):
    pytest.importorskip("bs4")
    artifact = make_artifact(tmp_path, landing_urls=["https://site.example/article"])
    config = build_config()
    html = """
    <html><head>
    <meta name="citation_pdf_url" content="/files/paper.pdf">
    </head></html>
    """
    responses.add(responses.GET, "https://site.example/article", body=html, status=200)
    session = requests.Session()
    results = [r for r in resolvers.LandingPageResolver().iter_urls(session, config, artifact) if not r.is_event]
    assert results[0].url == "https://site.example/files/paper.pdf"

    html_anchor = """
    <html><body><a href="/download/paper.pdf">Get PDF</a></body></html>
    """
    responses.add(responses.GET, "https://site.example/anchor", body=html_anchor, status=200)
    artifact.landing_urls = ["https://site.example/anchor"]
    results = [r for r in resolvers.LandingPageResolver().iter_urls(session, config, artifact) if not r.is_event]
    assert results[0].metadata["pattern"] == "anchor"


@responses.activate
def test_landing_page_resolver_http_error(tmp_path):
    pytest.importorskip("bs4")
    artifact = make_artifact(tmp_path, landing_urls=["https://site.example/error"])
    config = build_config()
    responses.add(responses.GET, "https://site.example/error", status=500)
    session = requests.Session()
    events = [r for r in resolvers.LandingPageResolver().iter_urls(session, config, artifact) if r.is_event]
    assert events[0].event_reason == "http-error"


@responses.activate
def test_pmc_resolver_uses_id_converter(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path, pmcid=None)
    config = build_config(unpaywall_email="tester@example.org")
    responses.add(
        responses.GET,
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
        match=[matchers.query_param_matcher({"ids": "10.1000/example", "format": "json", "tool": "docs-to-kg", "email": "tester@example.org"})],
        json={"records": [{"pmcid": "PMC777"}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC777",
        body='href="/articles/PMC777/pdf/foo.pdf"',
        status=200,
    )
    results = [r.url for r in resolvers.PmcResolver().iter_urls(session, config, artifact)]
    assert "https://www.ncbi.nlm.nih.gov/articles/PMC777/pdf/foo.pdf" in results
    assert "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC777/pdf/" in results


@responses.activate
def test_pmc_resolver_handles_request_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path, pmcid="PMC123")
    config = build_config()
    responses.add(
        responses.GET,
        "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC123",
        body=responses.ConnectionError("boom"),
    )
    results = list(resolvers.PmcResolver().iter_urls(session, config, artifact))
    assert results[-1].url.endswith("PMC123/pdf/")


@responses.activate
def test_europe_pmc_resolver_filters_pdf(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json={
            "resultList": {
                "result": [
                    {
                        "fullTextUrlList": {
                            "fullTextUrl": [
                                {"documentStyle": "pdf", "url": "https://epmc.org/pdf1"},
                                {"documentStyle": "html", "url": "https://epmc.org/html"},
                            ]
                        }
                    }
                ]
            }
        },
        status=200,
    )
    urls = [r.url for r in resolvers.EuropePmcResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://epmc.org/pdf1"]


@responses.activate
def test_europe_pmc_resolver_http_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        status=500,
    )
    urls = list(resolvers.EuropePmcResolver().iter_urls(session, config, artifact))
    assert urls == []


@responses.activate
def test_openaire_resolver_collects_pdf_candidates(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.openaire.eu/search/publications",
        json={
            "response": {
                "results": {
                    "result": [
                        {
                            "metadata": {
                                "instance": {
                                    "url": "https://openaire.example/paper.pdf",
                                    "extra": [
                                        "https://openaire.example/ignore.txt",
                                        "https://openaire.example/paper.pdf",
                                    ],
                                }
                            }
                        }
                    ]
                }
            }
        },
        status=200,
    )

    urls = [r.url for r in resolvers.OpenAireResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://openaire.example/paper.pdf"]


@responses.activate
def test_hal_resolver_uses_file_fields(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.archives-ouvertes.fr/search/",
        json={
            "response": {
                "docs": [
                    {
                        "fileMain_s": "https://hal.archives-ouvertes.fr/fileMain.pdf",
                        "file_s": [
                            "https://hal.archives-ouvertes.fr/alternate.pdf",
                            "https://hal.archives-ouvertes.fr/fileMain.pdf",
                        ],
                    }
                ]
            }
        },
        status=200,
    )

    urls = [r.url for r in resolvers.HalResolver().iter_urls(session, config, artifact)]
    assert urls == [
        "https://hal.archives-ouvertes.fr/fileMain.pdf",
        "https://hal.archives-ouvertes.fr/alternate.pdf",
    ]


@responses.activate
def test_osf_resolver_merges_download_links(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.osf.io/v2/preprints/",
        json={
            "data": [
                {
                    "links": {"download": "https://osf.io/download1"},
                    "attributes": {
                        "primary_file": {
                            "links": {"download": "https://osf.io/download2"}
                        }
                    },
                }
            ]
        },
        status=200,
    )

    urls = [r.url for r in resolvers.OsfResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://osf.io/download1", "https://osf.io/download2"]


@responses.activate
def test_core_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(core_api_key="abc123")
    responses.add(
        responses.GET,
        "https://api.core.ac.uk/v3/search/works",
        json={
            "results": [
                {"downloadUrl": "https://core.org/paper.pdf"},
                {"fullTextLinks": [{"url": "https://core.org/extra.pdf"}]},
            ]
        },
        status=200,
    )
    urls = [r.url for r in resolvers.CoreResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://core.org/paper.pdf", "https://core.org/extra.pdf"]


@responses.activate
def test_core_resolver_handles_failure(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(core_api_key="abc123")
    responses.add(responses.GET, "https://api.core.ac.uk/v3/search/works", status=500)
    urls = list(resolvers.CoreResolver().iter_urls(session, config, artifact))
    assert urls == []


@responses.activate
def test_doaj_resolver_filters_pdf(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://doaj.org/api/v2/search/articles/",
        json={
            "results": [
                {
                    "bibjson": {
                        "link": [
                            {"type": "fulltext", "url": "https://doaj.org/paper.pdf"},
                            {"type": "landing", "url": "https://doaj.org/landing"},
                        ]
                    }
                }
            ]
        },
        status=200,
    )
    urls = [r.url for r in resolvers.DoajResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://doaj.org/paper.pdf"]


@responses.activate
def test_doaj_resolver_handles_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(responses.GET, "https://doaj.org/api/v2/search/articles/", status=429)
    urls = list(resolvers.DoajResolver().iter_urls(session, config, artifact))
    assert urls == []


@responses.activate
def test_semantic_scholar_resolver_handles_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1000/example",
        status=500,
    )
    results = list(resolvers.SemanticScholarResolver().iter_urls(session, config, artifact))
    assert results == []


@responses.activate
def test_semantic_scholar_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1000/example",
        json={"openAccessPdf": {"url": "https://s2.org/paper.pdf"}},
        status=200,
    )
    urls = [r.url for r in resolvers.SemanticScholarResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://s2.org/paper.pdf"]


@responses.activate
def test_wayback_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path, failed_pdf_urls=["https://dead.example/file.pdf"])
    config = build_config()
    responses.add(
        responses.GET,
        "https://archive.org/wayback/available",
        json={
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "url": "https://web.archive.org/web/20200101/https://dead.example/file.pdf",
                    "timestamp": "20200101000000",
                }
            }
        },
        status=200,
    )
    urls = [r.url for r in resolvers.WaybackResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://web.archive.org/web/20200101/https://dead.example/file.pdf"]


@responses.activate
def test_wayback_resolver_handles_missing_snapshot(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path, failed_pdf_urls=["https://dead.example/file.pdf"])
    config = build_config()
    responses.add(
        responses.GET,
        "https://archive.org/wayback/available",
        json={"archived_snapshots": {"closest": {"available": False}}},
        status=200,
    )
    urls = list(resolvers.WaybackResolver().iter_urls(session, config, artifact))
    assert urls == []
