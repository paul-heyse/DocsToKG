"""Streaming download behaviour against the harness HTTP server.

Validates retry/backoff semantics, checksum enforcement, archive extraction
limits, filename sanitisation, and conditional GET handling to ensure the
downloader hardens against adversarial responses.
"""

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
from DocsToKG.OntologyDownload.errors import ConfigError, DownloadFailure
from DocsToKG.OntologyDownload.io import filesystem as fs_mod
from DocsToKG.OntologyDownload.io import network as network_mod
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
    assert methods.count("HEAD") == 1
    assert methods.count("GET") == 2

    get_records = [record for record in ontology_env.requests if record.method == "GET"]
    assert get_records[0].headers.get("Range") == f"bytes={partial_size}-"
    assert "Range" not in get_records[1].headers

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

    parsed = urlparse(url)
    pool = getattr(network_mod.SESSION_POOL, "_pool")
    lock = getattr(network_mod.SESSION_POOL, "_lock")
    max_per_key = getattr(network_mod.SESSION_POOL, "_max_per_key")
    normalize = getattr(network_mod.SESSION_POOL, "_normalize")

    with lock:
        pool_snapshot = {key: list(stack) for key, stack in pool.items()}

    assert all(len(stack) <= max_per_key for stack in pool_snapshot.values())
    normalized_key = normalize("obo", parsed.hostname)
    assert len(pool_snapshot.get(normalized_key, [])) <= max_per_key


def test_preliminary_head_check_handles_malformed_content_length(ontology_env, tmp_path):
    """Malformed Content-Length headers should be ignored by the downloader."""

    payload = b"@prefix : <http://example.org/> .\n:hp a :Ontology .\n"
    url = ontology_env.register_fixture(
        "hp-malformed-length.owl",
        payload,
        media_type="application/rdf+xml",
    )
    parsed_url = urlparse(url)
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
    expected_user_agent = config.polite_http_headers().get("User-Agent")
    downloader = network_mod.StreamingDownloader(
        destination=destination,
        headers={},
        http_config=config,
        previous_manifest=None,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
        origin_host=parsed_url.hostname,
    )

    with network_mod.SESSION_POOL.lease(
        service="obo",
        host=parsed_url.hostname,
        http_config=config,
    ) as session:
        content_type, content_length = downloader._preliminary_head_check(url, session)

    assert content_type == "application/rdf+xml"
    assert content_length is None
    assert ontology_env.requests[-1].method == "HEAD"
    assert ontology_env.requests[-1].path.endswith("hp-malformed-length.owl")
    if expected_user_agent is not None:
        assert ontology_env.requests[-1].headers.get("User-Agent") == expected_user_agent


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


def test_preliminary_head_check_cancels_retry_sleep(ontology_env, tmp_path):
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
    expected_headers = config.polite_http_headers()
    previous_manifest = {
        "etag": 'W/"conditional-etag"',
        "last_modified": "Wed, 21 Oct 2015 07:28:00 GMT",
    }
    request_headers = dict(expected_headers)
    request_headers.update(
        {
            "If-None-Match": previous_manifest["etag"],
            "If-Modified-Since": previous_manifest["last_modified"],
        }
    )
    token = CancellationToken()
    parsed_url = urlparse(url)
    downloader = network_mod.StreamingDownloader(
        destination=destination,
        headers=expected_headers,
        http_config=config,
        previous_manifest=previous_manifest,
        logger=_logger(),
        service="obo",
        origin_host=parsed_url.hostname,
        cancellation_token=token,
    )

    remaining_budget = mock.Mock(return_value=10.0)
    timeout_callback = mock.Mock()
    sleep_calls: list[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)
        token.cancel()

    with network_mod.SESSION_POOL.lease(
        service="obo",
        host=parsed_url.hostname,
        http_config=config,
    ) as session, mock.patch(
        "DocsToKG.OntologyDownload.io.network.time.sleep", side_effect=fake_sleep
    ):
        with pytest.raises(DownloadFailure):
            downloader._preliminary_head_check(
                url,
                session,
                headers=request_headers,
                remaining_budget=remaining_budget,
                timeout_callback=timeout_callback,
            )

    assert sleep_calls, "Expected the retry loop to invoke sleep"
    assert sleep_calls[0] < 1.5, "Sleep loop should use short increments"
    assert token.is_cancelled(), "Cancellation token should be triggered by the fake sleep"
    assert remaining_budget.call_count >= 2
    timeout_callback.assert_not_called()

    head_requests = [request for request in ontology_env.requests if request.method == "HEAD"]
    assert head_requests, "Expected a HEAD request to be issued"
    head_headers = head_requests[-1].headers

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
    ontology_env.queue_response(
        "fixtures/hp-retry-after.owl",
        ResponseSpec(
            method="HEAD",
            status=429,
            headers={"Retry-After": f"{retry_after_sec:.2f}"},
        ),
    )

    config = ontology_env.build_download_config()
    destination = tmp_path / "hp-retry-after.owl"

    start = time.monotonic()
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
    elapsed = time.monotonic() - start

    assert result.status == "fresh"
    assert destination.read_bytes() == payload
    assert elapsed >= retry_after_sec - 0.05

    methods = [request.method for request in ontology_env.requests]
    assert methods.count("HEAD") == 1
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
    parsed_url = urlparse(url)
    ontology_env.queue_response(
        "fixtures/hp-retry.owl",
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
    destination = tmp_path / "hp-retry.owl"
    downloader = network_mod.StreamingDownloader(
        destination=destination,
        headers={},
        http_config=config,
        previous_manifest=None,
        logger=_logger(),
        expected_media_type="application/rdf+xml",
        service="obo",
        origin_host=parsed_url.hostname,
    )

    with network_mod.SESSION_POOL.lease(
        service="obo",
        host=parsed_url.hostname,
        http_config=config,
    ) as session:
        content_type, content_length = downloader._preliminary_head_check(url, session)

    assert content_type == "application/rdf+xml"
    assert content_length is None
    assert ontology_env.requests[-1].method == "HEAD"
    assert ontology_env.requests[-1].path.endswith("hp-retry.owl")


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
