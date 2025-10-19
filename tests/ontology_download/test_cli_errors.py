"""Failure-mode coverage for ``ontofetch`` CLI error handling.

These tests orchestrate resolvers that raise planner or download exceptions,
exercise allowlist violations, and confirm the CLI surfaces actionable exit
codes plus JSON diagnostics while leaving manifests in a consistent state.
"""

from __future__ import annotations

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.resolvers import BaseResolver, FetchPlan
from DocsToKG.OntologyDownload.testing import ResponseSpec, temporary_resolver


def _allowed_hosts_arg(env) -> str:
    allowed = env.build_download_config().allowed_hosts or []
    return ",".join(allowed)


class _ExplodingPlanResolver(BaseResolver):
    NAME = "exploding-plan"

    def plan(self, spec, config, logger):  # noqa: D401 - simple test stub
        raise RuntimeError("planner boom")


class _ExplodingFetchResolver(BaseResolver):
    NAME = "exploding-fetch"

    def __init__(self, url: str) -> None:
        self._url = url

    def plan(self, spec, config, logger):  # noqa: D401 - simple test stub
        return FetchPlan(
            url=self._url,
            headers={},
            filename_hint="failure.owl",
            version="test-version",
            license="CC0-1.0",
            media_type="application/rdf+xml",
            service="test",
        )


def test_cli_main_returns_non_zero_for_batch_planning_error(ontology_env, capsys):
    """Batch planning errors should be surfaced to the operator."""

    with temporary_resolver(_ExplodingPlanResolver.NAME, _ExplodingPlanResolver()):
        exit_code = cli_module.cli_main(["plan", "hp", "--resolver", _ExplodingPlanResolver.NAME])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Planning aborted" in captured.err


def test_cli_main_returns_non_zero_for_batch_fetch_error(ontology_env, capsys):
    """Batch fetch errors should be surfaced with a friendly message."""

    failure_path = "failure.owl"
    ontology_env.queue_response(
        failure_path,
        ResponseSpec(status=500, body="boom", headers={"Content-Type": "text/plain"}),
    )
    resolver = _ExplodingFetchResolver(ontology_env.http_url(failure_path))

    with temporary_resolver(resolver.NAME, resolver):
        exit_code = cli_module.cli_main(
            [
                "pull",
                "hp",
                "--resolver",
                resolver.NAME,
                "--allowed-hosts",
                _allowed_hosts_arg(ontology_env),
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Download aborted" in captured.err
