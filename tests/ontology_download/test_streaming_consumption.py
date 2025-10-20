# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_streaming_consumption",
#   "purpose": "Streaming planner/fetcher behaviour tests.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Streaming planner/fetcher behaviour tests.

Ensures ``plan_all`` and ``fetch_all`` consume iterators lazily, honour
cancellation semantics, and propagate validation results while running under
multi-threaded harness scenarios."""

from __future__ import annotations

import logging
import threading
from contextlib import ExitStack
from typing import Iterator

from DocsToKG.OntologyDownload.planning import FetchSpec, fetch_all, plan_all
from DocsToKG.OntologyDownload.resolvers import BaseResolver, FetchPlan
from DocsToKG.OntologyDownload.testing import temporary_resolver, temporary_validator
from DocsToKG.OntologyDownload.validation import ValidationResult


def _logger() -> logging.Logger:
    logger = logging.getLogger("ontology-streaming-test")
    logger.setLevel(logging.INFO)
    return logger


def test_plan_all_consumes_generator_lazily(ontology_env, resolved_config):
    """Ensure ``plan_all`` does not eagerly exhaust generator inputs."""

    total_specs = 5
    concurrency = 3
    resolved_config.defaults.http.concurrent_plans = concurrency

    fixture_url = ontology_env.register_fixture(
        "lazy-plan.owl",
        b"plan-streaming",
        media_type="application/rdf+xml",
        repeats=total_specs,
    )

    resolver_name = "stream-plan"

    produced = 0
    completed = 0
    max_inflight = 0
    lock = threading.Lock()
    first_completion = threading.Event()

    class _StreamingResolver(BaseResolver):
        NAME = resolver_name

        def plan(self_inner, spec, config, logger, *, cancellation_token=None):  # type: ignore[override]
            nonlocal completed, max_inflight
            with lock:
                completed += 1
                inflight = produced - completed
                if inflight > max_inflight:
                    max_inflight = inflight
            first_completion.set()
            return FetchPlan(
                url=fixture_url,
                headers={"Accept": "application/rdf+xml"},
                filename_hint="lazy-plan.owl",
                version="v1",
                license="CC0-1.0",
                media_type="application/rdf+xml",
                service="obo",
            )

    def _specs() -> Iterator[FetchSpec]:
        nonlocal produced, max_inflight
        for index in range(total_specs):
            if not first_completion.is_set() and produced >= concurrency:
                raise AssertionError("generator consumed eagerly before any plan completed")
            spec = FetchSpec(
                id=f"lazy-plan-{index}",
                resolver=resolver_name,
                extras={},
                target_formats=("owl",),
            )
            with lock:
                produced += 1
                inflight = produced - completed
                if inflight > max_inflight:
                    max_inflight = inflight
            yield spec

    resolver = _StreamingResolver()

    with temporary_resolver(resolver_name, resolver):
        plans = plan_all(_specs(), config=resolved_config, logger=_logger())

    assert len(plans) == total_specs
    assert produced == total_specs
    assert completed == total_specs
    assert first_completion.is_set()
    assert max_inflight <= concurrency


def test_fetch_all_consumes_generator_lazily(ontology_env, resolved_config):
    """Ensure ``fetch_all`` drives generators without materialising them eagerly."""

    total_specs = 2
    concurrency = 1
    resolved_config.defaults.http.concurrent_downloads = concurrency
    resolved_config.defaults.http.concurrent_plans = concurrency

    fixture_url = ontology_env.register_fixture(
        "lazy-fetch.owl",
        b"fetch-streaming",
        media_type="application/rdf+xml",
        repeats=total_specs * 2,
    )

    resolver_name = "stream-fetch"

    produced = 0
    completed = 0
    max_inflight = 0
    lock = threading.Lock()
    first_completion = threading.Event()

    class _StreamingResolver(BaseResolver):
        NAME = resolver_name

        def plan(self_inner, spec, config, logger, *, cancellation_token=None):  # type: ignore[override]
            nonlocal completed, max_inflight
            with lock:
                completed += 1
                inflight = produced - completed
                if inflight > max_inflight:
                    max_inflight = inflight
            first_completion.set()
            return FetchPlan(
                url=fixture_url,
                headers={"Accept": "application/rdf+xml"},
                filename_hint="lazy-fetch.owl",
                version="v1",
                license="CC0-1.0",
                media_type="application/rdf+xml",
                service="obo",
            )

    def _specs() -> Iterator[FetchSpec]:
        nonlocal produced, max_inflight
        for index in range(total_specs):
            if not first_completion.is_set() and produced >= concurrency:
                raise AssertionError("generator consumed eagerly before any fetch completed")
            spec = FetchSpec(
                id=f"lazy-fetch-{index}",
                resolver=resolver_name,
                extras={},
                target_formats=("owl",),
            )
            with lock:
                produced += 1
                inflight = produced - completed
                if inflight > max_inflight:
                    max_inflight = inflight
            yield spec

    resolver = _StreamingResolver()

    validator_names = ("rdflib", "pronto", "owlready2", "robot", "arelle")

    def _stub_validator(request, logger):  # type: ignore[override]
        return ValidationResult(ok=True, details={}, output_files=[])

    with ExitStack() as stack:
        for name in validator_names:
            stack.enter_context(temporary_validator(name, _stub_validator))
        stack.enter_context(temporary_resolver(resolver_name, resolver))
        results = fetch_all(
            _specs(),
            config=resolved_config,
            logger=_logger(),
            force=True,
        )

    assert len(results) == total_specs
    assert produced == total_specs
    assert completed == total_specs
    assert first_completion.is_set()
    assert max_inflight <= concurrency
