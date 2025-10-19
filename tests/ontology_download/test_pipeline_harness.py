"""Pipeline integration tests driven by the ontology download harness."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload import api as core
from DocsToKG.OntologyDownload.planning import FetchSpec, plan_all
from DocsToKG.OntologyDownload.testing import temporary_resolver


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

    with temporary_resolver(hp_resolver_name, hp_resolver), temporary_resolver(
        go_resolver_name, go_resolver
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

    resolver_name, resolver = _static_resolver_for(ontology_env, name="cached", filename="cached.owl")
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
