"""Download behaviour tests exercising the harness-backed HTTP server."""

from __future__ import annotations

import io
import logging
import tarfile
import zipfile

import pytest

from DocsToKG.OntologyDownload.errors import ConfigError
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


def test_download_stream_ignores_non_numeric_head_length(ontology_env, tmp_path):
    """HEAD responses with non-numeric Content-Length values should not abort downloads."""

    payload = b"rdf"
    path = "fixtures/non-numeric-head.owl"
    common_headers = {
        "Content-Type": "application/rdf+xml",
        "ETag": '"non-numeric"',
        "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT",
    }

    ontology_env.queue_response(
        path,
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={**common_headers, "Content-Length": "not-a-number"},
        ),
    )
    ontology_env.queue_response(
        path,
        ResponseSpec(
            method="GET",
            status=200,
            headers={**common_headers, "Content-Length": str(len(payload))},
            body=payload,
        ),
    )

    config = ontology_env.build_download_config()
    destination = tmp_path / "non-numeric-head.owl"
    url = ontology_env.http_url(path)

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
    assert result.content_length == len(payload)
    methods = [request.method for request in ontology_env.requests]
    assert methods.count("HEAD") == 1
    assert methods.count("GET") == 1


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
    )

    assert cached.status == "cached"
    assert destination.read_bytes() == payload
    assert cached.sha256 == initial.sha256
    assert cached.content_type == initial.content_type


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
