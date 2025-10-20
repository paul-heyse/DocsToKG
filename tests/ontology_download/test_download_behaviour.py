# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_download_behaviour",
#   "purpose": "Streaming download behaviour against the harness HTTP server.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Streaming download behaviour against the harness HTTP server.

Validates retry/backoff semantics, checksum enforcement, archive extraction
limits, filename sanitisation, and conditional GET handling to ensure the
downloader hardens against adversarial responses."""

from __future__ import annotations

import hashlib
import io
import logging
import tarfile
import time
import zipfile
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

import pytest

from DocsToKG.OntologyDownload.cancellation import CancellationToken
from DocsToKG.OntologyDownload.errors import ConfigError, DownloadFailure, PolicyError
from DocsToKG.OntologyDownload.io import filesystem as fs_mod
from DocsToKG.OntologyDownload.io import get_http_client
from DocsToKG.OntologyDownload.io import network as network_mod
from DocsToKG.OntologyDownload.io import rate_limit as rate_mod
from DocsToKG.OntologyDownload.testing import ResponseSpec


def _logger() -> logging.Logger:
    logger = logging.getLogger("ontology-download-test")
    logger.setLevel(logging.INFO)
    return logger


def test_download_stream_fetches_fixture(ontology_env, tmp_path):
    """Ensure the downloader streams a registered fixture end-to-end."""

    payload = b"@prefix : <http://example.org/> .\n:hp a :Ontology .\n"
    url = ontology_env.register_fixture(
        "hp.owl",
        payload,
        media_type="application/rdf+xml",
        repeats=2,
    )
    config = ontology_env.build_download_config()
    destination = tmp_path / "hp.owl"
    expected_user_agent = config.polite_http_headers().get("User-Agent")

    result = network_mod.download_stream(
        url=url,
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=config,
        cache_dir=ontology_env.cache_dir,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
    )

    assert destination.read_bytes() == payload
    assert result.status == "fresh"
    assert result.content_type == "application/rdf+xml"
    methods = [request.method for request in ontology_env.requests]
    assert methods.count("HEAD") == 1
    assert methods.count("GET") == 1
    head_requests = [request for request in ontology_env.requests if request.method == "HEAD"]
    assert head_requests, "Expected at least one HEAD request to be recorded"
    if expected_user_agent is not None:
        assert all(
            request.headers.get("User-Agent") == expected_user_agent for request in head_requests
        )


def test_download_stream_recovers_from_range_not_satisfiable(ontology_env, tmp_path):
    """A 416 response should trigger a clean retry without a Range header."""

    payload = b"range retry payload\n"
    fixture_name = "range-retry.owl"
    url = ontology_env.register_fixture(
        fixture_name,
        payload,
        media_type="application/rdf+xml",
        repeats=2,
    )

    get_key = ("GET", f"/fixtures/{fixture_name}")
    get_queue = ontology_env._responses[get_key]
    assert len(get_queue) >= 2, "Expected at least two GET responses for retry simulation"
    get_queue[0] = ResponseSpec(
        method="GET",
        status=416,
        headers={
            "Content-Type": "application/rdf+xml",
            "Content-Range": f"bytes */{len(payload)}",
        },
        body=b"",
    )

    destination = tmp_path / fixture_name
    destination_part = Path(f"{destination}.part")
    partial_seed = b"partial-bytes"
    destination_part.write_bytes(partial_seed)
    partial_size = len(partial_seed)

    config = ontology_env.build_download_config()
    config.max_retries = 3

    result = network_mod.download_stream(
        url=url,
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=config,
        cache_dir=ontology_env.cache_dir,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
    )

    assert destination.read_bytes() == payload
    assert result.status in {"fresh", "updated"}
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert not destination_part.exists(), "Expected stale partial files to be removed"

    methods = [record.method for record in ontology_env.requests]
    assert methods.count("HEAD") == 2
    assert methods.count("GET") == 2

    get_records = [record for record in ontology_env.requests if record.method == "GET"]
    assert get_records[0].headers.get("Range") == f"bytes={partial_size}-"

    assert result.content_type == "application/rdf+xml"


def test_head_get_connections_remain_bounded(ontology_env, tmp_path):
    """Back-to-back HEAD/GET cycles should not leak pooled sessions."""

    payload = b"@prefix : <http://example.org/> .\n:onto a :Ontology .\n"
    url = ontology_env.register_fixture(
        "bounded.owl",
        payload,
        media_type="application/rdf+xml",
        repeats=8,
    )
    config = ontology_env.build_download_config()

    for attempt in range(4):
        destination = tmp_path / f"bounded-{attempt}.owl"
        result = network_mod.download_stream(
            url=url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=ontology_env.cache_dir,
            logger=_logger(),
            expected_media_type="application/rdf+xml",
            service="obo",
        )
        assert destination.read_bytes() == payload
        assert result.status == "fresh"

    methods = [request.method for request in ontology_env.requests]
    assert methods.count("HEAD") == 4
    assert methods.count("GET") == 4

    client_a = get_http_client()
    client_b = get_http_client()
    assert client_a is client_b


def test_download_stream_head_precheck_handles_malformed_content_length(ontology_env, tmp_path):
    """Malformed Content-Length headers should be ignored by the downloader."""

    payload = b"@prefix : <http://example.org/> .\n:hp a :Ontology .\n"
    url = ontology_env.register_fixture(
        "hp-malformed-length.owl",
        payload,
        media_type="application/rdf+xml",
    )
    ontology_env.queue_response(
        "fixtures/hp-malformed-length.owl",
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": "not-an-integer",
            },
        ),
    )

    config = ontology_env.build_download_config()
    destination = tmp_path / "hp-malformed-length.owl"

    result = network_mod.download_stream(
        url=url,
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=config,
        cache_dir=ontology_env.cache_dir,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
    )

    assert result.status == "fresh"
    assert destination.read_bytes() == payload
    assert result.content_type == "application/rdf+xml"
    assert result.content_length == len(payload)

    head_requests = [request for request in ontology_env.requests if request.method == "HEAD"]
    assert head_requests, "Expected a HEAD request to be issued"
    head_request = head_requests[-1]
    assert head_request.path.endswith("hp-malformed-length.owl")
    expected_user_agent = config.polite_http_headers().get("User-Agent")
    if expected_user_agent is not None:
        assert head_request.headers.get("User-Agent") == expected_user_agent


def test_head_request_includes_polite_and_conditional_headers(ontology_env, tmp_path):
    """HEAD should forward polite headers plus conditional validators."""

    payload = b"@prefix : <http://example.org/> .\n:hp a :Ontology .\n"
    url = ontology_env.register_fixture(
        "hp-conditional.owl",
        payload,
        media_type="application/rdf+xml",
    )

    config = ontology_env.build_download_config()
    destination = tmp_path / "hp-conditional.owl"
    expected_headers = config.polite_http_headers()
    previous_manifest = {
        "etag": 'W/"conditional-etag"',
        "last_modified": "Wed, 21 Oct 2015 07:28:00 GMT",
    }

    network_mod.download_stream(
        url=url,
        destination=destination,
        headers={},
        previous_manifest=previous_manifest,
        http_config=config,
        cache_dir=ontology_env.cache_dir,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
    )


def test_download_stream_cancellation_breaks_retry_after_sleep(ontology_env, tmp_path):
    """Cancellation during Retry-After backoff should abort promptly."""

    config = ontology_env.build_download_config()
    url = ontology_env.register_fixture(
        "cancelled.owl",
        b"@prefix : <http://example.org/> .\n:hp a :Ontology .\n",
        media_type="application/rdf+xml",
    )
    ontology_env.queue_response(
        "fixtures/cancelled.owl",
        ResponseSpec(
            method="HEAD",
            status=429,
            headers={"Retry-After": "3"},
        ),
    )
    destination = tmp_path / "cancelled.owl"
    previous_manifest = {
        "etag": 'W/"conditional-etag"',
        "last_modified": "Wed, 21 Oct 2015 07:28:00 GMT",
    }
    token = CancellationToken()
    sleep_calls: list[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)
        token.cancel()

    original_retry = network_mod.retry_with_backoff

    def wrapped_retry(func, **kwargs):
        kwargs["sleep"] = fake_sleep
        return original_retry(func, **kwargs)

    with mock.patch.object(network_mod, "retry_with_backoff", side_effect=wrapped_retry):
        with pytest.raises(DownloadFailure):
            network_mod.download_stream(
                url=url,
                destination=destination,
                headers={},
                previous_manifest=previous_manifest,
                http_config=config,
                cache_dir=ontology_env.cache_dir,
                logger=_logger(),
                expected_media_type="application/rdf+xml",
                service="obo",
                cancellation_token=token,
            )

    assert sleep_calls, "Expected the retry loop to invoke sleep"
    assert sleep_calls[0] == pytest.approx(3.0, abs=0.5)
    assert token.is_cancelled(), "Cancellation token should be triggered by the fake sleep"

    head_requests = [request for request in ontology_env.requests if request.method == "HEAD"]
    assert head_requests, "Expected a HEAD request to be issued"
    head_headers = head_requests[-1].headers
    expected_headers = config.polite_http_headers()
    user_agent = expected_headers.get("User-Agent")
    if user_agent is not None:
        assert head_headers.get("User-Agent") == user_agent
    assert head_headers.get("If-None-Match") == previous_manifest["etag"]
    assert head_headers.get("If-Modified-Since") == previous_manifest["last_modified"]


def test_head_retry_after_honours_delay_before_get(ontology_env, tmp_path):
    """A Retry-After header on the HEAD response should delay the subsequent GET."""

    payload = b"@prefix : <http://example.org/> .\n:hp a :Ontology .\n"
    retry_after_sec = 0.35
    url = ontology_env.register_fixture(
        "hp-retry-after.owl",
        payload,
        media_type="application/rdf+xml",
        repeats=1,
    )
    head_key = ("HEAD", "/fixtures/hp-retry-after.owl")
    head_queue = ontology_env._responses.setdefault(head_key, [])
    head_queue.insert(0, ResponseSpec(
        method="HEAD",
        status=429,
        headers={"Retry-After": f"{retry_after_sec:.2f}"},
    ))
    head_queue.insert(1, ResponseSpec(
        method="HEAD",
        status=200,
        headers={
            "Content-Type": "application/rdf+xml",
            "Content-Length": str(len(payload)),
        },
    ))

    config = ontology_env.build_download_config()
    config.perform_head_precheck = True
    destination = tmp_path / "hp-retry-after.owl"

    sleeps: list[float] = []

    def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    original_retry = network_mod.retry_with_backoff
    original_apply_retry_after = network_mod.apply_retry_after

    def wrapped_retry(func, **kwargs):
        kwargs["sleep"] = fake_sleep
        return original_retry(func, **kwargs)

    recorded_delays: list[float] = []

    def fake_apply_retry_after(*, http_config, service, host, delay):
        recorded_delays.append(delay)
        return original_apply_retry_after(
            http_config=http_config,
            service=service,
            host=host,
            delay=delay,
        )

    with mock.patch("DocsToKG.OntologyDownload.io.network.apply_retry_after", side_effect=fake_apply_retry_after):
        with mock.patch.object(network_mod, "retry_with_backoff", side_effect=wrapped_retry):
            result = network_mod.download_stream(
                url=url,
                destination=destination,
                headers={},
                previous_manifest=None,
                http_config=config,
                cache_dir=ontology_env.cache_dir,
                logger=_logger(),
                expected_media_type="application/rdf+xml",
                service="obo",
            )

    assert result.status == "fresh"
    assert destination.read_bytes() == payload
    methods = [record.method for record in ontology_env.requests]
    assert methods.count("HEAD") == 2
    assert recorded_delays and recorded_delays[0] == pytest.approx(retry_after_sec, abs=0.1)

    methods = [request.method for request in ontology_env.requests]
    assert methods.count("HEAD") == 2
    assert methods.count("GET") == 1


def test_download_stream_retries_consume_bucket(ontology_env, tmp_path):
    """A transient failure should consume bucket tokens for each retry."""

    payload = b"@prefix : <http://example.org/> .\n:hp a :Ontology .\n"
    url = ontology_env.register_fixture(
        "hp-retry.owl",
        payload,
        media_type="application/rdf+xml",
        repeats=1,
    )
    head_key = ("HEAD", "/fixtures/hp-retry.owl")
    head_queue = ontology_env._responses.setdefault(head_key, [])
    head_queue.append(
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": str(len(payload)),
            },
        )
    )
    head_queue.append(
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": str(len(payload)),
            },
        )
    )
    get_key = ("GET", "/fixtures/hp-retry.owl")
    get_queue = ontology_env._responses[get_key]
    get_queue[0] = ResponseSpec(
        method="GET",
        status=503,
        headers={
            "Content-Type": "application/rdf+xml",
            "Content-Length": str(len(payload)),
        },
        body=b"",
    )
    get_queue.insert(
        1,
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

    config = ontology_env.build_download_config()
    destination = tmp_path / "hp-retry.owl"

    class RecordingBucket:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def consume(self, tokens: float = 1.0) -> None:  # pragma: no cover - simple recorder
            self.calls.append(tokens)

    recording_bucket = RecordingBucket()

    with mock.patch.object(network_mod, "get_bucket", return_value=recording_bucket):
        result = network_mod.download_stream(
            url=url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=ontology_env.cache_dir,
            logger=_logger(),
            expected_media_type="application/rdf+xml",
            service="obo",
        )

    methods = [record.method for record in ontology_env.requests]
    assert methods.count("HEAD") == 2
    assert methods.count("GET") == 2

    assert destination.read_bytes() == payload
    assert result.status == "fresh"
    assert recording_bucket.calls == [1.0, 1.0, 1.0, 1.0]


def test_download_stream_uses_cached_manifest(ontology_env, tmp_path):
    """A 304 response should produce a cached result without re-downloading."""

    payload = b"ontology-content"
    url = ontology_env.register_fixture(
        "hp-cache.owl",
        payload,
        media_type="application/rdf+xml",
        repeats=2,
    )
    config = ontology_env.build_download_config()
    destination = tmp_path / "hp-cache.owl"

    initial = network_mod.download_stream(
        url=url,
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=config,
        cache_dir=ontology_env.cache_dir,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
    )
    initial_mtime = destination.stat().st_mtime

    previous_manifest = {
        "etag": initial.etag,
        "last_modified": initial.last_modified,
        "content_type": initial.content_type,
        "content_length": initial.content_length,
        "sha256": initial.sha256,
    }
    headers = {
        "ETag": initial.etag or '"cached-etag"',
        "Last-Modified": initial.last_modified or "Wed, 01 Jan 2025 00:00:00 GMT",
        "Content-Type": "application/rdf+xml",
    }

    ontology_env.queue_response(
        "fixtures/hp-cache.owl",
        ResponseSpec(method="HEAD", status=200, headers=headers),
    )
    ontology_env.queue_response(
        "fixtures/hp-cache.owl",
        ResponseSpec(method="GET", status=304, headers=headers),
    )

    cached = network_mod.download_stream(
        url=url,
        destination=destination,
        headers={},
        previous_manifest=previous_manifest,
        http_config=config,
        cache_dir=ontology_env.cache_dir,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
        expected_hash=f"sha256:{initial.sha256}",
    )

    assert cached.status == "cached"
    assert destination.read_bytes() == payload
    assert cached.sha256 == initial.sha256
    assert cached.content_type == initial.content_type
    assert destination.stat().st_mtime == pytest.approx(initial_mtime)


def test_download_stream_logs_progress(ontology_env, tmp_path, caplog):
    """Streaming should emit progress telemetry based on configured cadence."""

    fixture_url = ontology_env.register_fixture(
        "progress.owl",
        b"progress-body",
        media_type="application/rdf+xml",
        repeats=2,
    )
    config = ontology_env.build_download_config()
    config.progress_log_bytes_threshold = 1
    config.progress_log_percent_step = 0.0

    destination = tmp_path / "progress.owl"
    with caplog.at_level(logging.INFO):
        network_mod.download_stream(
            url=fixture_url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=ontology_env.cache_dir,
            logger=_logger(),
            expected_media_type="application/rdf+xml",
            service="obo",
        )

    progress_entries = [record for record in caplog.records if record.getMessage() == "download progress"]
    assert progress_entries, "Expected download progress logs to be emitted"


def test_download_stream_enforces_size_limit(ontology_env, tmp_path):
    """Downloads exceeding the max uncompressed size should raise PolicyError."""

    large_payload = b"x" * 4096
    fixture_url = ontology_env.register_fixture(
        "oversized.owl",
        large_payload,
        media_type="application/rdf+xml",
        repeats=1,
    )
    config = ontology_env.build_download_config()
    config.max_uncompressed_size_gb = 1e-6  # ~1 KB

    destination = tmp_path / "oversized.owl"
    with pytest.raises(PolicyError):
        network_mod.download_stream(
            url=fixture_url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=ontology_env.cache_dir,
            logger=_logger(),
            expected_media_type="application/rdf+xml",
            service="obo",
        )

    assert not destination.exists(), "Oversized download should be cleaned up"


def test_download_stream_resumes_and_streams_hash(ontology_env, tmp_path):
    """Resume downloads should stitch content and report streamed hashes."""

    payload = b"@prefix : <http://example.org/> .\n:hp a :Ontology .\n"
    resume_offset = 16
    destination = tmp_path / "resume.owl"
    destination_part = Path(str(destination) + ".part")
    destination_part.write_bytes(payload[:resume_offset])

    config = ontology_env.build_download_config()
    url = ontology_env.http_url("fixtures/resume.owl")
    headers = {
        "Content-Type": "application/rdf+xml",
        "Content-Length": str(len(payload)),
    }
    ontology_env.queue_response(
        "fixtures/resume.owl",
        ResponseSpec(method="HEAD", status=200, headers=headers),
    )
    ontology_env.queue_response(
        "fixtures/resume.owl",
        ResponseSpec(
            method="GET",
            status=206,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Range": f"bytes {resume_offset}-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(payload) - resume_offset),
            },
            body=payload[resume_offset:],
        ),
    )

    result = network_mod.download_stream(
        url=url,
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=config,
        cache_dir=ontology_env.cache_dir,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
    )

    assert result.status == "updated"
    assert destination.read_bytes() == payload
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    get_requests = [request for request in ontology_env.requests if request.method == "GET"]
    assert get_requests[-1].headers.get("Range") == f"bytes={resume_offset}-"


def test_download_stream_rejects_disallowed_redirect(ontology_env, tmp_path):
    """Redirect targets outside the allowlist should raise a policy error."""

    start_path = "fixtures/disallowed-redirect.owl"
    start_url = ontology_env.http_url(start_path)
    config = ontology_env.build_download_config()
    destination = tmp_path / "disallowed-redirect.owl"

    ontology_env._responses[("HEAD", "/" + start_path)].append(
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={"Content-Type": "application/rdf+xml", "Content-Length": "12"},
        )
    )
    ontology_env._responses[("GET", "/" + start_path)].append(
        ResponseSpec(
            method="GET",
            status=302,
            headers={"Location": "https://malicious.example/ontology.owl"},
        )
    )

    initial_len = len(ontology_env.requests)
    with pytest.raises(PolicyError):
        network_mod.download_stream(
            url=start_url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=ontology_env.cache_dir,
            logger=_logger(),
            expected_media_type="application/rdf+xml",
            service="obo",
        )

    new_requests = ontology_env.requests[initial_len:]
    methods_paths = [(record.method, record.path) for record in new_requests]
    assert ("GET", "/" + start_path) in methods_paths
    assert not destination.exists()


def test_download_stream_follows_valid_redirect_chain(ontology_env, tmp_path):
    """Downloader should follow validated redirects and persist the final payload."""

    payload = b"redirect-payload"
    target_name = "redirect-target.owl"
    target_rel_path = f"fixtures/{target_name}"
    ontology_env.register_fixture(
        target_name,
        payload,
        media_type="application/rdf+xml",
        repeats=1,
    )

    start_path = "fixtures/redirect-source.owl"
    start_url = ontology_env.http_url(start_path)

    ontology_env._responses[("HEAD", "/" + start_path)].append(
        ResponseSpec(
            method="HEAD",
            status=302,
            headers={"Location": f"/{target_rel_path}"},
        )
    )
    ontology_env._responses[("GET", "/" + start_path)].append(
        ResponseSpec(
            method="GET",
            status=302,
            headers={"Location": f"/{target_rel_path}"},
        )
    )

    config = ontology_env.build_download_config()
    destination = tmp_path / target_name
    initial_len = len(ontology_env.requests)

    result = network_mod.download_stream(
        url=start_url,
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=config,
        cache_dir=ontology_env.cache_dir,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
    )

    assert destination.read_bytes() == payload
    assert result.status == "fresh"
    new_requests = ontology_env.requests[initial_len:]
    get_paths = [record.path for record in new_requests if record.method == "GET"]
    assert get_paths == ["/" + start_path, "/" + target_rel_path]


def test_download_stream_respects_validated_url_hint(ontology_env, tmp_path):
    """Passing the validated flag should suppress redundant URL validations."""

    payload = b"@prefix : <http://example.org/> .\n:validated a :Ontology .\n"
    original_url = ontology_env.register_fixture(
        "validated.owl",
        payload,
        media_type="application/rdf+xml",
    )
    config = ontology_env.build_download_config()
    destination = tmp_path / "validated.owl"

    original_validate = network_mod.validate_url_security
    secure_url = original_validate(original_url, config)

    with mock.patch.object(
        network_mod,
        "validate_url_security",
        wraps=original_validate,
    ) as validate_spy:
        network_mod.download_stream(
            url=secure_url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=ontology_env.cache_dir,
            logger=_logger(),
            expected_media_type="application/rdf+xml",
            service="obo",
            url_already_validated=True,
        )

    assert validate_spy.call_count == 0
    assert destination.read_bytes() == payload


def test_extract_zip_rejects_traversal(tmp_path):
    """Zip extraction should guard against traversal attacks."""

    archive = tmp_path / "traversal.zip"
    with zipfile.ZipFile(archive, "w") as zipf:
        info = zipfile.ZipInfo("../evil.txt")
        zipf.writestr(info, "oops")

    with pytest.raises(ConfigError):
        fs_mod.extract_zip_safe(archive, tmp_path / "output", logger=_logger())


def test_extract_tar_rejects_symlink(tmp_path):
    """Tar extraction should reject symlinks inside the archive."""

    archive = tmp_path / "symlink.tar"
    with tarfile.open(archive, "w") as tar:
        data = io.BytesIO(b"content")
        info = tarfile.TarInfo("data.txt")
        info.size = len(data.getvalue())
        tar.addfile(info, data)

        link_info = tarfile.TarInfo("link")
        link_info.type = tarfile.SYMTYPE
        link_info.linkname = "data.txt"
        tar.addfile(link_info)

    with pytest.raises(ConfigError):
        fs_mod.extract_tar_safe(archive, tmp_path / "output", logger=_logger())


def test_sanitize_filename_normalises(tmp_path):
    """Sanitization should strip traversal components and prohibited characters."""

    assert fs_mod.sanitize_filename("../evil.owl") == "evil.owl"
    assert fs_mod.sanitize_filename("..\\..\\windows?.owl") == "windows_.owl"

def test_testing_environment_legacy_mode(ontology_env):
    ontology_env.use_legacy_rate_limiter()
    config = ontology_env.build_download_config()
    assert config.rate_limiter == "legacy"
    provider = config.get_bucket_provider()
    assert provider is not None
    bucket = provider("test", config, "legacy.example")
    assert isinstance(bucket, getattr(rate_mod, "_LegacyTokenBucket"))
