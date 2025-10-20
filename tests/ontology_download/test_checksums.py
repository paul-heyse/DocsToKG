"""Regression matrix for checksum parsing, normalization, and resolution.

These tests cover how ``DocsToKG.OntologyDownload.checksums`` pulls digests
from configuration extras, resolver metadata, and remote checksum manifests.
They verify algorithm coercion, precedence rules between spec and plan hints,
error propagation when downloads fail, and caching behaviour used by the
planner while constructing manifests.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional
from urllib.parse import urlparse

import pytest
import httpx

from DocsToKG.OntologyDownload.checksums import (
    ExpectedChecksum,
    _fetch_checksum_from_url,
    parse_checksum_extra,
    parse_checksum_url_extra,
    resolve_expected_checksum,
)
from DocsToKG.OntologyDownload.errors import OntologyDownloadError
from DocsToKG.OntologyDownload.planning import FetchSpec
from DocsToKG.OntologyDownload.resolvers import FetchPlan


@pytest.mark.parametrize(
    ("plan_kwargs", "spec_extras", "fetched_digest", "expected"),
    [
        ({}, {"checksum": "a" * 64}, None, ("sha256", "a" * 64)),
        (
            {},
            {"checksum": {"algorithm": "sha512", "value": "b" * 128}},
            None,
            ("sha512", "b" * 128),
        ),
        (
            {"checksum": "c" * 32, "checksum_algorithm": "md5"},
            {},
            None,
            ("md5", "c" * 32),
        ),
        (
            {},
            {"checksum_url": "https://example.org/checks.txt"},
            "d" * 64,
            ("sha256", "d" * 64),
        ),
        (
            {},
            {
                "checksum_url": {
                    "url": "https://example.org/checks-sha512.txt",
                    "algorithm": "sha512",
                }
            },
            "e" * 128,
            ("sha512", "e" * 128),
        ),
        (
            {"checksum_url": "https://example.org/plan.txt", "checksum_algorithm": "sha512"},
            {},
            "f" * 128,
            ("sha512", "f" * 128),
        ),
    ],
)
def test_resolve_expected_checksum_matrix(
    plan_kwargs: Dict[str, object],
    spec_extras: Dict[str, object],
    fetched_digest: Optional[str],
    expected: Optional[tuple[str, str]],
    ontology_env,
    download_config,
) -> None:
    """Ensure checksum parsing/resolution is consistent across planner surfaces."""

    config = download_config.model_copy()

    plan_kwargs = dict(plan_kwargs)
    spec_extras = dict(spec_extras)

    if fetched_digest is not None:
        body = f"{fetched_digest} ontology.owl\n".encode("utf-8")
        checksum_url = ontology_env.register_fixture(
            "checksum.txt",
            body,
            media_type="text/plain",
            repeats=3,
        )
        parsed = urlparse(checksum_url)
        host = parsed.hostname
        if host:
            allowed_entries = [host]
            if parsed.port is not None:
                allowed_entries.append(f"{host}:{parsed.port}")
            config.allowed_hosts = allowed_entries

        if "checksum_url" in spec_extras:
            value = spec_extras["checksum_url"]
            if isinstance(value, dict):
                spec_extras["checksum_url"] = {**value, "url": checksum_url}
            else:
                spec_extras["checksum_url"] = checksum_url

        if "checksum_url" in plan_kwargs:
            value = plan_kwargs["checksum_url"]
            if isinstance(value, dict):
                plan_kwargs["checksum_url"] = {**value, "url": checksum_url}
            else:
                plan_kwargs["checksum_url"] = checksum_url

    spec = FetchSpec(
        id="hp",
        resolver="obo",
        extras=spec_extras,
        target_formats=("owl",),
    )
    plan_args = dict(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint="hp.owl",
        version=None,
        license=None,
        media_type="application/rdf+xml",
    )
    plan_args.update(plan_kwargs)
    plan = FetchPlan(**plan_args)

    result = resolve_expected_checksum(
        spec=spec,
        plan=plan,
        download_config=config,
        logger=logging.getLogger("checksum-test"),
    )
    if fetched_digest is None:
        assert ontology_env.requests == []
    if expected is None:
        assert result is None
        if fetched_digest is not None:
            assert any(req.method == "GET" for req in ontology_env.requests)
        return

    assert isinstance(result, ExpectedChecksum)
    assert (result.algorithm, result.value) == expected
    if fetched_digest is not None:
        assert any(req.method == "GET" for req in ontology_env.requests)

    checksum_extra = spec_extras.get("checksum") if isinstance(spec_extras, dict) else None
    if checksum_extra is not None:
        assert parse_checksum_extra(checksum_extra, context="spec") == expected

    checksum_url_extra = spec_extras.get("checksum_url") if isinstance(spec_extras, dict) else None
    if checksum_url_extra is not None:
        url_value, algorithm = parse_checksum_url_extra(checksum_url_extra, context="spec")
        if isinstance(checksum_url_extra, str):
            assert algorithm is None
        else:
            assert algorithm == expected[0]
        assert url_value.startswith("http://") or url_value.startswith("https://")


def test_fetch_checksum_retries_consume_bucket(ontology_env, download_config) -> None:
    """Verify checksum fetch retries consume bucket tokens on each attempt."""

    config = download_config.model_copy()

    class RecordingBucket:
        def __init__(self) -> None:
            self.calls = 0

        def consume(self, tokens: float = 1.0) -> None:
            self.calls += 1
            if self.calls == 1:
                raise httpx.TransportError("transient bucket failure")

    bucket = RecordingBucket()
    config.set_bucket_provider(lambda service, cfg, host: bucket)

    digest = "abc123".ljust(64, "0")
    body = f"preface\n{digest}\n".encode("utf-8")
    checksum_url = ontology_env.register_fixture(
        "retry-checksum.txt",
        body,
        media_type="text/plain",
        repeats=2,
    )

    result = _fetch_checksum_from_url(
        url=checksum_url,
        algorithm="sha256",
        http_config=config,
        logger=logging.getLogger("checksum-retry-test"),
    )

    assert result == digest
    assert bucket.calls == 2
    # Only one HTTP request should have been issued because the first attempt
    # failed before the network call.
    get_requests = [req for req in ontology_env.requests if req.method == "GET"]
    assert len(get_requests) == 1


def test_fetch_checksum_aborts_when_limit_exceeded(ontology_env, download_config, caplog) -> None:
    """Ensure oversized checksum payloads are aborted and logged."""

    config = download_config.model_copy()
    config.max_checksum_response_bytes = 1_024

    digest = "deadbeef".ljust(64, "f")
    body = ("x" * 2048 + digest).encode("utf-8")
    checksum_url = ontology_env.register_fixture(
        "overflow-checksum.txt",
        body,
        media_type="text/plain",
        repeats=1,
    )

    logger = logging.getLogger("checksum-limit-test")
    caplog.set_level(logging.ERROR, logger="checksum-limit-test")

    with pytest.raises(OntologyDownloadError) as excinfo:
        _fetch_checksum_from_url(
            url=checksum_url,
            algorithm="sha256",
            http_config=config,
            logger=logger,
        )

    assert "exceeded" in str(excinfo.value)
    assert any("checksum response exceeded limit" in record.message for record in caplog.records)
    # The request should still be recorded even though it was aborted mid-stream.
    assert any(req.method == "GET" for req in ontology_env.requests)
