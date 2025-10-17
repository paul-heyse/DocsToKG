"""Tests for checksum parsing and resolution helpers."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pytest

from DocsToKG.OntologyDownload.checksums import (
    ExpectedChecksum,
    parse_checksum_extra,
    parse_checksum_url_extra,
    resolve_expected_checksum,
)
from DocsToKG.OntologyDownload.planning import FetchSpec
from DocsToKG.OntologyDownload.resolvers import FetchPlan
from DocsToKG.OntologyDownload.settings import DownloadConfiguration


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure checksum parsing/resolution is consistent across planner surfaces."""

    import DocsToKG.OntologyDownload.checksums as checksums_mod

    # Prevent network and URL security side effects during testing.
    monkeypatch.setattr(checksums_mod, "validate_url_security", lambda url, config=None: url)

    if fetched_digest is not None:

        class _DummyResponse:
            text = f"{fetched_digest} ontology.owl"

            @staticmethod
            def raise_for_status() -> None:
                return None

        monkeypatch.setattr(checksums_mod.requests, "get", lambda url, timeout: _DummyResponse())
    else:
        monkeypatch.setattr(
            checksums_mod.requests,
            "get",
            lambda url, timeout: pytest.fail("unexpected network call for checksum fetch"),
        )

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
        download_config=DownloadConfiguration(),
        logger=logging.getLogger("checksum-test"),
    )
    if expected is None:
        assert result is None
        return

    assert isinstance(result, ExpectedChecksum)
    assert (result.algorithm, result.value) == expected

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
        assert url_value.startswith("https://")
