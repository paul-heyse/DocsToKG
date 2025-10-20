# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_download",
#   "purpose": "Edge-case coverage for the streaming downloader and redirect handling.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Edge-case coverage for the streaming downloader and redirect handling.

Focuses on HTTP semantics like redirect chains, retry-after handling, and
rate-limited token bucket resets to ensure ``io.network`` enforces security
policies during real downloads."""

import logging
import time
from contextlib import contextmanager

import httpx
import pytest

from DocsToKG.OntologyDownload.errors import DownloadFailure, PolicyError
from DocsToKG.OntologyDownload.io import network as network_mod
from DocsToKG.OntologyDownload.settings import DownloadConfiguration
from DocsToKG.OntologyDownload.testing import ResponseSpec
from tests.conftest import PatchManager


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


def test_get_redirect_follows_validated_target(ontology_env, tmp_path):
    """Allowed redirects should be followed and deliver the final payload."""

    start_path = "fixtures/redirect-valid.owl"
    final_path = "fixtures/redirect-valid-final.owl"
    payload = b"<rdf>redirected</rdf>"

    ontology_env.queue_response(
        start_path,
        ResponseSpec(
            method="HEAD",
            status=302,
            headers={"Location": f"/{final_path}"},
        ),
    )
    ontology_env.queue_response(
        final_path,
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": str(len(payload)),
            },
        ),
    )
    ontology_env.queue_response(
        start_path,
        ResponseSpec(
            method="GET",
            status=302,
            headers={"Location": f"/{final_path}"},
        ),
    )
    ontology_env.queue_response(
        final_path,
        ResponseSpec(
            method="GET",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": str(len(payload)),
            },
            body=payload,
        ),
    )

    url = ontology_env.http_url(start_path)
    destination = tmp_path / "redirect-valid.owl"
    config = ontology_env.build_download_config()

    result = network_mod.download_stream(
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

    assert destination.read_bytes() == payload
    assert result.status == "fresh"
    methods_paths = [(record.method, record.path) for record in ontology_env.requests]
    assert methods_paths == [
        ("HEAD", "/" + start_path),
        ("HEAD", "/" + final_path),
        ("GET", "/" + start_path),
        ("GET", "/" + final_path),
    ]


class _BucketSpy:
    """Deterministic token bucket that records consumption without sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def consume(self, tokens: float = 1.0) -> None:
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


def test_download_timeout_aborts_retries(ontology_env, tmp_path, caplog):
    """Downloads exceeding the configured timeout must fail without retrying."""

    slow_path = "fixtures/slow-timeout.owl"
    ontology_env.queue_response(
        slow_path,
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": "5",
            },
        ),
    )
    ontology_env.queue_response(
        slow_path,
        ResponseSpec(
            method="GET",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": "5",
            },
            body="hello",
            delay_sec=2.5,
        ),
    )

    url = ontology_env.http_url(slow_path)
    destination = tmp_path / "slow-timeout.owl"
    config = ontology_env.build_download_config()
    config.download_timeout_sec = 1
    config.max_retries = 3

    logger = _logger()

    with caplog.at_level(logging.INFO):
        with pytest.raises(DownloadFailure) as exc_info:
            network_mod.download_stream(
                url=url,
                destination=destination,
                headers={},
                previous_manifest=None,
                http_config=config,
                cache_dir=ontology_env.cache_dir,
                logger=logger,
                expected_media_type="application/rdf+xml",
                service="test",
            )

    assert "timeout" in str(exc_info.value).lower()
    assert not exc_info.value.retryable
    assert not destination.exists()

    methods_paths = [(record.method, record.path) for record in ontology_env.requests]
    assert methods_paths == [
        ("HEAD", "/" + slow_path),
        ("GET", "/" + slow_path),
    ]

    timeout_records = [
        record
        for record in caplog.records
        if getattr(record, "stage", None) == "download"
        and getattr(record, "error", None) == "timeout"
        and record.getMessage() == "download timeout"
    ]
    assert timeout_records


def test_head_retry_after_timeout_fails_fast(ontology_env, tmp_path):
    """HEAD 429 responses with long Retry-After should trip the timeout budget immediately."""

    retry_path = "fixtures/head-retry-after.owl"
    ontology_env.queue_response(
        retry_path,
        ResponseSpec(
            method="HEAD",
            status=429,
            headers={"Retry-After": "5"},
        ),
    )

    url = ontology_env.http_url(retry_path)
    destination = tmp_path / "head-retry-after.owl"
    config = ontology_env.build_download_config()
    config.download_timeout_sec = 1

    start = time.monotonic()
    with pytest.raises(DownloadFailure) as exc_info:
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

    elapsed = time.monotonic() - start

    assert elapsed < 0.5
    assert "timeout" in str(exc_info.value).lower()
    assert not destination.exists()
    assert [record.method for record in ontology_env.requests] == ["HEAD"]


def test_stream_timeout_removes_partial_files(ontology_env, tmp_path):
    """Slow streaming responses exceeding the timeout must clean up partial files."""

    slow_path = "fixtures/slow-stream-timeout.owl"
    ontology_env.queue_response(
        slow_path,
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": "2",
            },
        ),
    )

    def slow_stream():
        time.sleep(1.2)
        yield b"a"
        yield b"b"

    ontology_env.queue_response(
        slow_path,
        ResponseSpec(
            method="GET",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": "2",
            },
            stream=slow_stream(),
        ),
    )

    url = ontology_env.http_url(slow_path)
    destination = tmp_path / "slow-stream-timeout.owl"
    config = ontology_env.build_download_config()
    config.download_timeout_sec = 1

    with pytest.raises(DownloadFailure) as exc_info:
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

    assert "timeout" in str(exc_info.value).lower()
    assert not exc_info.value.retryable
    assert not destination.exists()
    destination_part = destination.parent / (destination.name + ".part")
    assert not destination_part.exists()
    assert not list(ontology_env.cache_dir.rglob("*.part"))

def test_retry_after_triggers_sleep(tmp_path):
    delays: list[float] = []

    def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    original_retry = network_mod.retry_with_backoff

    def wrapped_retry(*args, **kwargs):
        kwargs.setdefault("sleep", fake_sleep)
        return original_retry(*args, **kwargs)

    patcher = PatchManager()
    patcher.setattr(network_mod, "retry_with_backoff", wrapped_retry)

    url = "https://example.org/retry-after"
    head_response = httpx.Response(
        200,
        headers={
            "Content-Type": "application/rdf+xml",
            "Content-Length": "5",
        },
        request=httpx.Request("HEAD", url),
    )
    retry_head = httpx.Response(
        200,
        headers={
            "Content-Type": "application/rdf+xml",
            "Content-Length": "5",
        },
        request=httpx.Request("HEAD", url),
    )
    first_get = httpx.Response(
        429,
        headers={
            "Retry-After": "0.2",
            "Content-Type": "application/rdf+xml",
            "Content-Length": "5",
        },
        request=httpx.Request("GET", url),
    )
    second_get = httpx.Response(
        200,
        headers={
            "Content-Type": "application/rdf+xml",
            "Content-Length": "5",
        },
        content=b"hello",
        request=httpx.Request("GET", url),
    )

    responses = iter([head_response, first_get, retry_head, second_get])

    @contextmanager
    def fake_request_with_redirect_audit(**_kwargs):
        yield next(responses)

    patcher.setattr(network_mod, "request_with_redirect_audit", fake_request_with_redirect_audit)

    bucket = _BucketSpy()
    config = DownloadConfiguration()
    config.allowed_hosts = ["example.org"]
    config.set_bucket_provider(lambda service, http_config, host: bucket)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / "retry-after.owl"

    try:
        result = network_mod.download_stream(
            url=url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=cache_dir,
            logger=_logger(),
            expected_media_type="application/rdf+xml",
            service="test",
        )
    finally:
        patcher.close()

    assert result.status == "fresh"
    assert delays, "Expected retry logic to invoke sleep callback"
    assert delays[0] >= 0.2
    assert len(bucket.calls) >= 3
