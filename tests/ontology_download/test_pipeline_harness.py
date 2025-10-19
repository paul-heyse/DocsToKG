"""Pipeline integration tests driven by the ontology download harness."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import pytest
from DocsToKG.OntologyDownload import api as core
from DocsToKG.OntologyDownload.planning import (
    BatchFetchError,
    BatchPlanningError,
    FetchSpec,
    fetch_all,
    plan_all,
)
from DocsToKG.OntologyDownload.cancellation import CancellationTokenGroup
from DocsToKG.OntologyDownload.resolvers import BaseResolver, FetchPlan
from DocsToKG.OntologyDownload.testing import ResponseSpec, temporary_resolver


def _logger() -> logging.Logger:
    logger = logging.getLogger("ontology-pipeline-test")
    logger.setLevel(logging.INFO)
    return logger


def _resolved_config(ontology_env):
    resolved = ontology_env.build_resolved_config()
    resolved.defaults.http = ontology_env.build_download_config()
    return resolved


def _static_resolver_for(env, *, name: str, filename: str) -> tuple[str, object]:
    fixture_url = env.register_fixture(
        filename,
        (f"{name} content\n").encode("utf-8"),
        media_type="application/rdf+xml",
        repeats=3,
    )
    resolver_name = f"{name}-static"
    resolver = env.static_resolver(
        name=resolver_name,
        fixture_url=fixture_url,
        filename=filename,
        media_type="application/rdf+xml",
        service="obo",
    )
    return resolver_name, resolver


def test_fetch_all_downloads_and_writes_manifest(ontology_env, tmp_path):
    """End-to-end fetch using temporary resolvers should write manifests."""

    hp_resolver_name, hp_resolver = _static_resolver_for(ontology_env, name="hp", filename="hp.owl")
    go_resolver_name, go_resolver = _static_resolver_for(ontology_env, name="go", filename="go.owl")
    hp_spec = FetchSpec(id="hp", resolver=hp_resolver_name, extras={}, target_formats=("owl",))
    go_spec = FetchSpec(id="go", resolver=go_resolver_name, extras={}, target_formats=("owl",))

    config = _resolved_config(ontology_env)

    with (
        temporary_resolver(hp_resolver_name, hp_resolver),
        temporary_resolver(go_resolver_name, go_resolver),
    ):
        results = core.fetch_all(
            [hp_spec, go_spec],
            config=config,
            logger=_logger(),
            force=True,
        )

    assert {result.spec.id for result in results} == {"hp", "go"}
    for result in results:
        assert result.local_path.exists()
        assert result.status in {"fresh", "updated"}
        manifest_payload = json.loads(result.manifest_path.read_text())
        assert manifest_payload["id"] == result.spec.id
        assert manifest_payload["status"] == result.status
        assert Path(manifest_payload["filename"]).name.endswith(".owl")


def test_fetch_all_second_run_uses_cache(ontology_env):
    """Subsequent fetches should return cached results when no changes occur."""

    resolver_name, resolver = _static_resolver_for(
        ontology_env, name="cached", filename="cached.owl"
    )
    spec = FetchSpec(id="cached", resolver=resolver_name, extras={}, target_formats=("owl",))
    config = _resolved_config(ontology_env)

    with temporary_resolver(resolver_name, resolver):
        initial = core.fetch_all([spec], config=config, logger=_logger(), force=True)
        assert initial[0].status in {"fresh", "updated"}
        cached = core.fetch_all([spec], config=config, logger=_logger(), force=False)

    assert cached[0].status == "cached"
    assert cached[0].local_path.read_bytes() == initial[0].local_path.read_bytes()


def test_plan_all_uses_temporary_resolver(ontology_env):
    """Planning should honour temporary resolver registrations."""

    resolver_name, resolver = _static_resolver_for(ontology_env, name="plan", filename="plan.owl")
    spec = FetchSpec(id="plan", resolver=resolver_name, extras={}, target_formats=("owl",))
    config = _resolved_config(ontology_env)

    with temporary_resolver(resolver_name, resolver):
        plans = plan_all([spec], config=config, logger=_logger())

    assert len(plans) == 1
    planned = plans[0]
    assert planned.spec.id == "plan"
    assert planned.plan.url.startswith("http://127.0.0.1")
    assert planned.plan.filename_hint == "plan.owl"


def test_plan_all_batch_failure_cancels_blocking_workers(ontology_env):
    """Batch planning errors should surface without waiting for hung workers."""

    blocking_started = threading.Event()
    release_blocking = threading.Event()

    class _BlockingPlanResolver(BaseResolver):
        NAME = "blocking-plan"

        def plan(self, spec, config, logger, *, cancellation_token=None):  # noqa: D401 - simple blocking stub
            blocking_started.set()
            # Check for cancellation periodically while waiting
            while not release_blocking.is_set():
                if cancellation_token and cancellation_token.is_cancelled():
                    raise RuntimeError("blocking resolver was cancelled")
                release_blocking.wait(timeout=0.1)  # Check every 100ms
            raise RuntimeError("blocking resolver should have been cancelled")

    class _FailingPlanResolver(BaseResolver):
        NAME = "failing-plan"

        def plan(self, spec, config, logger, *, cancellation_token=None):  # noqa: D401 - simple failure stub
            time.sleep(0.1)
            raise RuntimeError("planner boom")

    blocking_spec = FetchSpec(
        id="blocking",
        resolver=_BlockingPlanResolver.NAME,
        extras={},
        target_formats=("owl",),
    )
    failing_spec = FetchSpec(
        id="failing",
        resolver=_FailingPlanResolver.NAME,
        extras={},
        target_formats=("owl",),
    )
    config = _resolved_config(ontology_env)
    config.defaults.http.concurrent_plans = 2
    config.defaults.resolver_fallback_enabled = False
    config.defaults.prefer_source = []
    timer = threading.Timer(0.5, release_blocking.set)
    try:
        with (
            temporary_resolver(_BlockingPlanResolver.NAME, _BlockingPlanResolver()),
            temporary_resolver(_FailingPlanResolver.NAME, _FailingPlanResolver()),
        ):
            timer.start()
            start = time.monotonic()
            with pytest.raises(BatchPlanningError):
                # Create a cancellation token group for the test
                token_group = CancellationTokenGroup()
                plan_all(
                    [blocking_spec, failing_spec],
                    config=config,
                    logger=_logger(),
                    cancellation_token_group=token_group,
                )
            elapsed = time.monotonic() - start
    finally:
        release_blocking.set()
        timer.cancel()

    assert blocking_started.is_set()
    assert elapsed < 1.0


def test_fetch_all_batch_failure_cancels_blocking_workers(ontology_env):
    """Batch fetch errors should not wait for in-flight downloads to finish."""

    blocking_path = "blocking.owl"
    failing_path = "failing.owl"
    headers = {
        "Content-Type": "application/rdf+xml",
        "ETag": "test-etag",
        "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT",
    }

    ontology_env.queue_response(
        blocking_path,
        ResponseSpec(method="HEAD", status=200, headers=headers),
    )
    ontology_env.queue_response(
        blocking_path,
        ResponseSpec(method="GET", status=200, headers=headers, body=b"blocking", delay_sec=0.1),
    )
    ontology_env.queue_response(
        failing_path,
        ResponseSpec(method="HEAD", status=200, headers=headers),
    )
    ontology_env.queue_response(
        failing_path,
        ResponseSpec(method="GET", status=500, headers=headers, body=b"boom"),
    )

    blocking_url = ontology_env.http_url(blocking_path)
    failing_url = ontology_env.http_url(failing_path)

    class _BlockingFetchResolver(BaseResolver):
        NAME = "blocking-fetch"

        def plan(self, spec, config, logger, *, cancellation_token=None):  # noqa: D401 - simple blocking plan
            return FetchPlan(
                url=blocking_url,
                headers={"Accept": "application/rdf+xml"},
                filename_hint="blocking.owl",
                version="test-version",
                license="CC0-1.0",
                media_type="application/rdf+xml",
                service="test",
            )

    class _FailingFetchResolver(BaseResolver):
        NAME = "failing-fetch"

        def plan(self, spec, config, logger, *, cancellation_token=None):  # noqa: D401 - simple failure plan
            time.sleep(0.1)
            return FetchPlan(
                url=failing_url,
                headers={"Accept": "application/rdf+xml"},
                filename_hint="failing.owl",
                version="test-version",
                license="CC0-1.0",
                media_type="application/rdf+xml",
                service="test",
            )

    blocking_spec = FetchSpec(
        id="blocking",
        resolver=_BlockingFetchResolver.NAME,
        extras={},
        target_formats=("owl",),
    )
    failing_spec = FetchSpec(
        id="failing",
        resolver=_FailingFetchResolver.NAME,
        extras={},
        target_formats=("owl",),
    )

    config = _resolved_config(ontology_env)
    config.defaults.http.concurrent_downloads = 2
    config.defaults.resolver_fallback_enabled = False
    config.defaults.prefer_source = []

    with (
        temporary_resolver(_BlockingFetchResolver.NAME, _BlockingFetchResolver()),
        temporary_resolver(_FailingFetchResolver.NAME, _FailingFetchResolver()),
    ):
        start = time.monotonic()
        with pytest.raises(BatchFetchError):
            # Create a cancellation token group for the test
            token_group = CancellationTokenGroup()
            fetch_all(
                [blocking_spec, failing_spec],
                config=config,
                logger=_logger(),
                force=True,
                cancellation_token_group=token_group,
            )
        elapsed = time.monotonic() - start

    assert any(
        request.method == "GET" and request.path == f"/{blocking_path}"
        for request in ontology_env.requests
    )
    assert elapsed < 1.0


def test_plan_all_releases_tokens_after_heavy_batch(ontology_env):
    """Heavy planning batches should release tokens when complete."""

    token_group = CancellationTokenGroup()
    config = _resolved_config(ontology_env)
    config.defaults.http.concurrent_plans = 4

    specs: list[FetchSpec] = []
    with ExitStack() as stack:
        for index in range(8):
            resolver_name, resolver = _static_resolver_for(
                ontology_env,
                name=f"plan-heavy-{index}",
                filename=f"plan-heavy-{index}.owl",
            )
            stack.enter_context(temporary_resolver(resolver_name, resolver))
            specs.append(
                FetchSpec(
                    id=f"plan-heavy-{index}",
                    resolver=resolver_name,
                    extras={},
                    target_formats=("owl",),
                )
            )

        plans = plan_all(
            specs,
            config=config,
            logger=_logger(),
            cancellation_token_group=token_group,
        )

    assert len(plans) == len(specs)
    assert len(token_group) == 0


def test_fetch_all_releases_tokens_after_heavy_batch(ontology_env):
    """Heavy fetch batches should release tokens when complete."""

    token_group = CancellationTokenGroup()
    config = _resolved_config(ontology_env)
    config.defaults.http.concurrent_downloads = 4

    specs: list[FetchSpec] = []
    with ExitStack() as stack:
        for index in range(6):
            resolver_name, resolver = _static_resolver_for(
                ontology_env,
                name=f"fetch-heavy-{index}",
                filename=f"fetch-heavy-{index}.owl",
            )
            stack.enter_context(temporary_resolver(resolver_name, resolver))
            specs.append(
                FetchSpec(
                    id=f"fetch-heavy-{index}",
                    resolver=resolver_name,
                    extras={},
                    target_formats=("owl",),
                )
            )

        results = fetch_all(
            specs,
            config=config,
            logger=_logger(),
            force=True,
            cancellation_token_group=token_group,
        )

    assert len(results) == len(specs)
    assert len(token_group) == 0
