"""Rate limiter integration tests for DocsToKG.ContentDownload."""

from __future__ import annotations

import os
import types
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
from pyrate_limiter import Duration, Rate
from pyrate_limiter.buckets import InMemoryBucket

from DocsToKG.ContentDownload import httpx_transport
from DocsToKG.ContentDownload.core import Classification, WorkArtifact
from DocsToKG.ContentDownload.download import build_download_outcome
from DocsToKG.ContentDownload.errors import RateLimitError
from DocsToKG.ContentDownload.networking import request_with_retries
from DocsToKG.ContentDownload.ratelimit import (
    BackendConfig,
    LimiterManager,
    RateLimitedTransport,
    RolePolicy,
    clone_policies,
    configure_rate_limits,
    get_rate_limiter_manager,
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


def _clone_backend_config(manager) -> BackendConfig:
    options = manager.backend.options
    return BackendConfig(
        backend=manager.backend.backend,
        options=dict(options) if isinstance(options, dict) else dict(options or {}),
    )


def _restore_manager(manager, policies, backend: BackendConfig) -> None:
    manager.configure_backend(backend)
    manager.configure_policies(clone_policies(policies))
    manager.reset_metrics()


def test_cache_hit_skips_rate_limiter_tokens(tmp_path: Path, monkeypatch) -> None:
    manager = get_rate_limiter_manager()
    original_policies = clone_policies(manager.policies())
    original_backend = _clone_backend_config(manager)

    try:
        configure_rate_limits(
            policies={
                "example.org": RolePolicy(
                    rates={"metadata": [Rate(5, Duration.SECOND)]},
                    max_delay_ms={"metadata": 250},
                    mode={"metadata": "wait"},
                    count_head={"metadata": False},
                    weight={"metadata": 1},
                )
            },
            backend=BackendConfig(backend="memory", options={}),
        )

        acquire_calls: list[tuple[str, str, str]] = []
        original_acquire = manager.acquire

        def _spy_acquire(*, host: str, role: str, method: str):
            acquire_calls.append((host, role, method))
            return original_acquire(host=host, role=role, method=method)

        monkeypatch.setattr(manager, "acquire", _spy_acquire)

        request_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            now = datetime.now(timezone.utc)
            return httpx.Response(
                200,
                headers={
                    "Cache-Control": "public, max-age=60",
                    "Content-Type": "text/plain",
                    "Date": format_datetime(now),
                    "Expires": format_datetime(now + timedelta(seconds=60)),
                },
                content=b"cached",
                request=request,
            )

        monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))
        httpx_transport.reset_http_client_for_tests()
        httpx_transport.configure_http_client(transport=httpx.MockTransport(handler))
        client = httpx_transport.get_http_client()

        response1 = client.get("https://example.org/resource")
        assert response1.status_code == 200
        response1.close()

        response2 = client.get(
            "https://example.org/resource",
            headers={"Cache-Control": "only-if-cached"},
        )
        assert response2.status_code == 200
        response2.close()

        assert request_count == 1
        assert len(acquire_calls) == 1
        assert acquire_calls[0] == ("example.org", "metadata", "GET")
    finally:
        httpx_transport.reset_http_client_for_tests()
        _restore_manager(manager, original_policies, original_backend)


def test_multi_window_waits_when_fast_window_exhausted() -> None:
    policy = RolePolicy(
        rates={"metadata": [Rate(2, 200), Rate(5, Duration.SECOND)]},
        max_delay_ms={"metadata": 1000},
        mode={"metadata": "wait"},
        count_head={"metadata": False},
        weight={"metadata": 1},
    )
    manager = LimiterManager(policies={"example.com": policy}, backend_config=BackendConfig())

    first = manager.acquire(host="example.com", role="metadata", method="GET")
    assert first.wait_ms >= 0

    second = manager.acquire(host="example.com", role="metadata", method="GET")
    assert second.wait_ms >= 0

    third = manager.acquire(host="example.com", role="metadata", method="GET")
    assert third.wait_ms > 0
    assert third.wait_ms <= policy.max_delay_ms["metadata"]

    snapshot = manager.metrics_snapshot()
    stats = snapshot["example.com"]["metadata"]
    assert stats["acquire_total"] == 3
    assert stats["wait_ms_count"] >= 1


def test_role_specific_max_delay_behaviour(tmp_path: Path, monkeypatch) -> None:
    manager = get_rate_limiter_manager()
    original_policies = clone_policies(manager.policies())
    original_backend = _clone_backend_config(manager)

    try:
        policy = RolePolicy(
            rates={
                "metadata": [Rate(1, 200)],
                "artifact": [Rate(1, 200)],
            },
            max_delay_ms={
                "metadata": 0,
                "artifact": 500,
            },
            mode={
                "metadata": "raise",
                "artifact": "wait",
            },
            count_head={
                "metadata": False,
                "artifact": False,
            },
            weight={
                "metadata": 1,
                "artifact": 1,
            },
        )
        configure_rate_limits(
            policies={"example.org": policy},
            backend=BackendConfig(backend="memory", options={}),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.extensions.get("role") == "artifact":
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    stream=iter([b"%PDF-1.4\n"]),
                    request=request,
                )
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                content=b"{}",
                request=request,
            )

        monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))
        httpx_transport.reset_http_client_for_tests()
        httpx_transport.configure_http_client(transport=httpx.MockTransport(handler))

        # Metadata request succeeds once then raises on immediate retry due to raise mode.
        response = request_with_retries(
            None,
            "GET",
            "https://example.org/meta",
            role="metadata",
            max_retries=0,
        )
        response.close()

        with pytest.raises(RateLimitError) as exc_info:
            request_with_retries(
                None,
                "GET",
                "https://example.org/meta",
                role="metadata",
                max_retries=0,
            )

        assert exc_info.value.mode == "raise"
        assert exc_info.value.waited_ms >= 0

        # Artifact streaming request waits within allowance on second attempt.
        with request_with_retries(
            None,
            "GET",
            "https://example.org/artifact",
            role="artifact",
            stream=True,
            max_retries=0,
        ) as stream_response:
            list(stream_response.iter_bytes())

        with request_with_retries(
            None,
            "GET",
            "https://example.org/artifact",
            role="artifact",
            stream=True,
            max_retries=0,
        ) as stream_response:
            meta = stream_response.request.extensions.get("docs_network_meta", {})
            wait_ms = meta.get("rate_limiter_wait_ms", 0)
            assert wait_ms > 0
            assert wait_ms <= policy.max_delay_ms["artifact"]
            list(stream_response.iter_bytes())
    finally:
        httpx_transport.reset_http_client_for_tests()
        _restore_manager(manager, original_policies, original_backend)


@pytest.mark.parametrize(
    "backend_name, options_factory",
    [
        ("memory", lambda _tmp: {}),
        ("multiprocess", lambda _tmp: {}),
        ("sqlite", lambda tmp: {"path": str(tmp / "rate-limit.sqlite"), "use_file_lock": False}),
    ],
)
def test_rate_limited_transport_smoke_backends(
    tmp_path: Path, backend_name: str, options_factory
) -> None:
    policy = RolePolicy(
        rates={"metadata": [Rate(5, Duration.SECOND)]},
        max_delay_ms={"metadata": 250},
        mode={"metadata": "wait"},
        count_head={"metadata": False},
        weight={"metadata": 1},
    )
    manager = LimiterManager(
        policies={"example.com": policy},
        backend_config=BackendConfig(backend=backend_name, options=options_factory(tmp_path)),
    )

    transport = RateLimitedTransport(
        httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        manager=manager,
    )
    request = httpx.Request("GET", "https://example.com/resource")
    response = transport.handle_request(request)
    assert response.status_code == 200
    response.close()
    transport.close()


@pytest.mark.skipif(
    os.environ.get("DOCSTOKG_TEST_REDIS") != "1",
    reason="Set DOCSTOKG_TEST_REDIS=1 to enable Redis backend smoke test.",
)
def test_rate_limited_transport_smoke_redis_backend(monkeypatch) -> None:
    pytest.importorskip("redis")

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.ratelimit.RedisBucket.init",
        lambda rates, client, key: InMemoryBucket(list(rates)),
        raising=False,
    )

    class _StubRedis:
        @staticmethod
        def from_url(_url: str) -> object:
            return object()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.ratelimit.Redis",
        _StubRedis,
        raising=False,
    )

    policy = RolePolicy(
        rates={"metadata": [Rate(5, Duration.SECOND)]},
        max_delay_ms={"metadata": 250},
        mode={"metadata": "wait"},
        count_head={"metadata": False},
        weight={"metadata": 1},
    )
    manager = LimiterManager(
        policies={"example.com": policy},
        backend_config=BackendConfig(
            backend="redis",
            options={"url": "redis://localhost:6379/0", "namespace": "docstokg:test"},
        ),
    )
    transport = RateLimitedTransport(
        httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        manager=manager,
    )
    request = httpx.Request("GET", "https://example.com/resource")
    response = transport.handle_request(request)
    assert response.status_code == 200
    response.close()
    transport.close()


@pytest.mark.skipif(
    os.environ.get("DOCSTOKG_TEST_POSTGRES") != "1",
    reason="Set DOCSTOKG_TEST_POSTGRES=1 to enable Postgres backend smoke test.",
)
def test_rate_limited_transport_smoke_postgres_backend(monkeypatch) -> None:
    pytest.importorskip("psycopg")

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.ratelimit.PostgresBucket.init",
        lambda conn, rates, table=None: InMemoryBucket(list(rates)),
        raising=False,
    )
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.ratelimit.connect",
        lambda dsn: types.SimpleNamespace(close=lambda: None),
        raising=False,
    )

    policy = RolePolicy(
        rates={"metadata": [Rate(5, Duration.SECOND)]},
        max_delay_ms={"metadata": 250},
        mode={"metadata": "wait"},
        count_head={"metadata": False},
        weight={"metadata": 1},
    )
    manager = LimiterManager(
        policies={"example.com": policy},
        backend_config=BackendConfig(
            backend="postgres",
            options={"dsn": "postgresql://user:secret@localhost:5432/ratelimit", "table": "rl"},
        ),
    )
    transport = RateLimitedTransport(
        httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        manager=manager,
    )
    request = httpx.Request("GET", "https://example.com/resource")
    response = transport.handle_request(request)
    assert response.status_code == 200
    response.close()
    transport.close()
