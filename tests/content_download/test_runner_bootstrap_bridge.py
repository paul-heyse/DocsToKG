"""Regression tests for DownloadRun bootstrap integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from DocsToKG.ContentDownload.config import ContentDownloadConfig
from DocsToKG.ContentDownload.config.models import ResolversConfig, TelemetryConfig
from DocsToKG.ContentDownload.runner import DownloadRun


class _StubResolver:
    """Simple resolver that yields no plans (safe for tests)."""

    name = "unpaywall"
    _registry_name = "unpaywall"

    def resolve(self, artifact, session, ctx, telemetry, run_id):  # noqa: D401,ARG002
        from DocsToKG.ContentDownload.api import ResolverResult

        return ResolverResult(plans=())


@pytest.fixture()
def stub_resolver() -> _StubResolver:
    return _StubResolver()


def test_process_artifacts_threads_bootstrap_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_resolver: _StubResolver
) -> None:
    """process_artifacts should translate config into BootstrapConfig."""

    csv_path = tmp_path / "telemetry" / "attempts.csv"
    manifest_path = tmp_path / "telemetry" / "manifest.jsonl"

    config = ContentDownloadConfig(
        run_id="test-run-id",
        telemetry=TelemetryConfig(
            sinks=["csv"],
            csv_path=str(csv_path),
            manifest_path=str(manifest_path),
        ),
        resolvers=ResolversConfig(order=["unpaywall"]),
    )

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.runner.build_resolvers",
        lambda cfg: [stub_resolver],
    )

    from DocsToKG.ContentDownload.bootstrap import run_from_config as bootstrap_run_from_config

    captured_configs = []

    def _capture_run_from_config(config, *args, **kwargs):
        captured_configs.append(config)
        return bootstrap_run_from_config(config, *args, **kwargs)

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.runner.run_from_config",
        _capture_run_from_config,
    )

    artifact = SimpleNamespace(work_id="A-001")

    download_run = DownloadRun(config)
    with download_run:
        result = download_run.process_artifacts([artifact])

    assert result.run_id == "test-run-id"
    assert captured_configs, "BootstrapConfig should be provided to run_from_config"

    bootstrap_config = captured_configs[0]
    assert bootstrap_config.run_id == "test-run-id"
    assert bootstrap_config.http.user_agent == config.http.user_agent
    assert bootstrap_config.telemetry_paths["csv"] == csv_path
    assert bootstrap_config.telemetry_paths["last_attempt"] == csv_path.with_name("last.csv")
    assert bootstrap_config.resolver_registry["unpaywall"] is stub_resolver
    assert (
        bootstrap_config.resolver_retry_configs["unpaywall"].max_attempts
        == config.resolvers.unpaywall.retry.max_attempts
    )
    assert (
        bootstrap_config.policy_knobs["chunk_size_bytes"]
        == config.download.chunk_size_bytes
    )
    assert bootstrap_config.policy_knobs["timeout_s"] == config.http.timeout_read_s

    # Telemetry sinks should be created/touched during the run
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0
    assert csv_path.with_name("last.csv").exists()
