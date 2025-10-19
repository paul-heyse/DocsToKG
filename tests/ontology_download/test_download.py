import logging

import pytest

from DocsToKG.OntologyDownload.errors import PolicyError
from DocsToKG.OntologyDownload.io import network as network_mod
from DocsToKG.OntologyDownload.io.rate_limit import TokenBucket
from DocsToKG.OntologyDownload.testing import ResponseSpec


def _logger() -> logging.Logger:
    logger = logging.getLogger("ontology-download-redirect-test")
    logger.setLevel(logging.INFO)
    return logger


def test_head_redirect_chain_to_disallowed_host(ontology_env, tmp_path):
    """HEAD redirect chains must be rejected when they exit the allowlist."""

    start_path = "fixtures/redirect-head.owl"
    intermediate_path = "fixtures/redirect-head-intermediate.owl"
    disallowed_url = "https://malicious.example/ontology.owl"

    ontology_env.queue_response(
        start_path,
        ResponseSpec(
            method="HEAD",
            status=302,
            headers={"Location": f"/{intermediate_path}"},
        ),
    )
    ontology_env.queue_response(
        intermediate_path,
        ResponseSpec(
            method="HEAD",
            status=302,
            headers={"Location": disallowed_url},
        ),
    )

    url = ontology_env.http_url(start_path)
    destination = tmp_path / "redirect-head.owl"
    config = ontology_env.build_download_config()

    with pytest.raises(PolicyError):
        network_mod.download_stream(
            url=url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=ontology_env.cache_dir,
            logger=_logger(),
            expected_media_type="application/rdf+xml",
            service="test",
        )

    assert not destination.exists()
    methods_paths = [(record.method, record.path) for record in ontology_env.requests]
    assert methods_paths == [
        ("HEAD", "/" + start_path),
        ("HEAD", "/" + intermediate_path),
    ]


def test_get_redirect_to_disallowed_host(ontology_env, tmp_path):
    """Streaming should abort when a GET redirect leaves the configured allowlist."""

    target_path = "fixtures/get-redirect.owl"
    disallowed_url = "https://malicious.example/redirected.owl"

    ontology_env.queue_response(
        target_path,
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": "123",
                "ETag": '"head-etag"',
                "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT",
            },
        ),
    )
    ontology_env.queue_response(
        target_path,
        ResponseSpec(
            method="GET",
            status=302,
            headers={"Location": disallowed_url},
        ),
    )

    url = ontology_env.http_url(target_path)
    destination = tmp_path / "get-redirect.owl"
    config = ontology_env.build_download_config()

    with pytest.raises(PolicyError):
        network_mod.download_stream(
            url=url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=ontology_env.cache_dir,
            logger=_logger(),
            expected_media_type="application/rdf+xml",
            service="test",
        )

    assert not destination.exists()
    assert all(record.path.startswith("/fixtures/") for record in ontology_env.requests)
    assert [record.method for record in ontology_env.requests] == ["HEAD", "GET"]


class _BucketSpy(TokenBucket):
    """Deterministic token bucket that records consumption without sleeping."""

    def __init__(self) -> None:
        super().__init__(rate_per_sec=1000.0, capacity=1000.0)
        self.calls: list[float] = []

    def consume(self, tokens: float = 1.0) -> None:  # type: ignore[override]
        self.calls.append(tokens)


def test_download_stream_consumes_tokens_for_head_and_get(ontology_env, tmp_path):
    """Both HEAD and GET requests must consume rate-limit tokens when available."""

    bucket = _BucketSpy()
    config = ontology_env.build_download_config()
    config.set_bucket_provider(lambda service, http_config, host: bucket)

    fixture_url = ontology_env.register_fixture(
        "bucket-consumption.owl",
        b"rate-limit-body",
        media_type="application/rdf+xml",
    )

    destination = tmp_path / "bucket-consumption.owl"

    network_mod.download_stream(
        url=fixture_url,
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=config,
        cache_dir=ontology_env.cache_dir,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="test-service",
    )

    assert destination.exists()
    assert destination.read_bytes() == b"rate-limit-body"
    assert bucket.calls == [1.0, 1.0]
    assert [record.method for record in ontology_env.requests] == ["HEAD", "GET"]
