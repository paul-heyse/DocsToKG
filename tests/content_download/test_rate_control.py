"""Rate limiter integration tests for DocsToKG.ContentDownload."""

from __future__ import annotations

import httpx
import pytest
from pathlib import Path
from pyrate_limiter import Duration, Rate

from DocsToKG.ContentDownload.core import Classification, WorkArtifact
from DocsToKG.ContentDownload.errors import RateLimitError
from DocsToKG.ContentDownload.download import build_download_outcome
from DocsToKG.ContentDownload.ratelimit import (
    BackendConfig,
    LimiterManager,
    RolePolicy,
    serialize_policy,
)


def _policy_for(rates: list[Rate]) -> RolePolicy:
    return RolePolicy(
        rates={"metadata": list(rates)},
        max_delay_ms={"metadata": 0},
        mode={"metadata": "raise"},
        count_head={"metadata": False},
        weight={"metadata": 1},
    )


def test_limiter_metrics_record_acquire_and_block():
    policy = _policy_for([Rate(1, Duration.SECOND)])
    manager = LimiterManager(policies={"example.com": policy}, backend_config=BackendConfig())

    # First acquire succeeds and records wait statistics.
    result = manager.acquire(host="example.com", role="metadata", method="GET")
    assert result.wait_ms >= 0

    snapshot = manager.metrics_snapshot()
    assert snapshot["example.com"]["metadata"]["acquire_total"] == 1
    assert snapshot["example.com"]["metadata"]["blocked_total"] == 0

    # Second acquire exceeds the single-per-second allowance and raises.
    with pytest.raises(RateLimitError) as exc_info:
        manager.acquire(host="example.com", role="metadata", method="GET")

    error = exc_info.value
    assert error.host == "example.com"
    assert error.role == "metadata"
    assert error.mode == "raise"

    snapshot = manager.metrics_snapshot()
    assert snapshot["example.com"]["metadata"]["blocked_total"] == 1
    # Serialisation helper should include metadata for reporting.
    serialised = serialize_policy(policy)
    assert "metadata" in serialised


def test_build_download_outcome_includes_rate_limiter_metadata():
    artifact = WorkArtifact(
        work_id="W1",
        title="Example",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="example",
        pdf_dir=Path("/tmp"),
        html_dir=Path("/tmp"),
        xml_dir=Path("/tmp"),
    )
    request = httpx.Request(
        "GET",
        "https://example.org/paper.pdf",
        extensions={
            "docs_network_meta": {
                "rate_limiter_wait_ms": 42,
                "rate_limiter_backend": "memory",
                "rate_limiter_mode": "wait",
                "rate_limiter_role": "artifact",
                "from_cache": False,
            }
        },
    )
    response = httpx.Response(200, request=request)
    outcome = build_download_outcome(
        artifact=artifact,
        classification=Classification.PDF,
        dest_path=None,
        response=response,
        elapsed_ms=10.0,
        flagged_unknown=False,
        sha256=None,
        content_length=1024,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        dry_run=False,
        tail_bytes=None,
        head_precheck_passed=True,
        min_pdf_bytes=0,
        tail_check_bytes=0,
        retry_after=None,
        options=None,
    )

    assert outcome.metadata["network"]["rate_limiter_wait_ms"] == 42
    assert outcome.metadata["rate_limiter"]["backend"] == "memory"
    assert outcome.metadata["rate_limiter"]["mode"] == "wait"
    assert outcome.metadata["rate_limiter"]["role"] == "artifact"
