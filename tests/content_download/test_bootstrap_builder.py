"""Tests for mapping Pydantic config into BootstrapConfig."""

from __future__ import annotations

import types
from importlib import import_module
from pathlib import Path
from typing import Iterable

import pytest

from DocsToKG.ContentDownload.api import ResolverResult
from DocsToKG.ContentDownload.bootstrap import build_bootstrap_config
from DocsToKG.ContentDownload.config import ContentDownloadConfig
from DocsToKG.ContentDownload.resolvers import registry_v2
from DocsToKG.ContentDownload.runner import run


@pytest.fixture
def stubbed_arxiv_resolver() -> Iterable[type]:
    """Register a lightweight resolver for tests and restore the original."""

    import_module("DocsToKG.ContentDownload.resolvers.arxiv")
    original = registry_v2.get_resolver_class("arxiv")

    @registry_v2.register_v2("arxiv")
    class StubArxivResolver:  # type: ignore[misc]
        name = "arxiv"

        @classmethod
        def from_config(cls, *_args, **_kwargs) -> "StubArxivResolver":
            return cls()

        def resolve(self, *_args, **_kwargs) -> ResolverResult:
            return ResolverResult(plans=())

    try:
        yield StubArxivResolver
    finally:
        registry_v2.register_v2("arxiv")(original)


def test_build_bootstrap_config_translates_config(tmp_path, stubbed_arxiv_resolver):
    base = ContentDownloadConfig()

    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    telemetry_cfg = base.telemetry.model_copy(
        update={
            "sinks": ["csv", "jsonl"],
            "csv_path": str(telemetry_dir / "attempts.csv"),
            "manifest_path": str(telemetry_dir / "manifest.jsonl"),
        }
    )
    resolvers_cfg = base.resolvers.model_copy(update={"order": ["arxiv"]})
    config = base.model_copy(update={"telemetry": telemetry_cfg, "resolvers": resolvers_cfg})

    bootstrap_cfg = build_bootstrap_config(config)

    assert bootstrap_cfg.http.user_agent == config.http.user_agent
    assert bootstrap_cfg.telemetry_paths["csv"] == Path(telemetry_dir / "attempts.csv")
    assert (
        bootstrap_cfg.telemetry_paths["last_attempt"]
        == Path(telemetry_dir / "attempts.csv").with_name("last.csv")
    )
    assert bootstrap_cfg.telemetry_paths["jsonl"] == Path(telemetry_dir / "manifest.jsonl")
    assert bootstrap_cfg.telemetry_paths["manifest_index"] == Path(
        telemetry_dir / "manifest.jsonl"
    ).with_name("index.json")
    assert bootstrap_cfg.telemetry_paths["summary"] == Path(
        telemetry_dir / "manifest.jsonl"
    ).with_name("summary.json")
    assert bootstrap_cfg.telemetry_paths["sqlite"] == Path(
        telemetry_dir / "manifest.jsonl"
    ).with_name("manifest.sqlite")
    assert "arxiv" in bootstrap_cfg.resolver_registry
    assert isinstance(bootstrap_cfg.resolver_registry["arxiv"], stubbed_arxiv_resolver)
    retry_cfg = bootstrap_cfg.resolver_retry_configs["arxiv"]
    assert retry_cfg.max_attempts == config.resolvers.arxiv.retry.max_attempts
    assert retry_cfg.rate_capacity == float(config.resolvers.arxiv.rate_limit.capacity)
    assert bootstrap_cfg.policy_knobs["download_policy"] == config.download
    assert bootstrap_cfg.policy_knobs["robots_policy"] == config.robots


def test_run_processes_artifact_with_builder(monkeypatch, tmp_path, stubbed_arxiv_resolver):
    base = ContentDownloadConfig()
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    telemetry_cfg = base.telemetry.model_copy(
        update={"sinks": ["csv"], "csv_path": str(telemetry_dir / "attempts.csv")}
    )
    resolvers_cfg = base.resolvers.model_copy(update={"order": ["arxiv"]})
    config = base.model_copy(update={"telemetry": telemetry_cfg, "resolvers": resolvers_cfg})

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.runner.load_config",
        lambda *args, **kwargs: config,
    )

    artifact = types.SimpleNamespace(work_id="artifact-1")
    result = run(artifacts=[artifact])

    assert result.total_processed == 1
    assert result.failed == 1

