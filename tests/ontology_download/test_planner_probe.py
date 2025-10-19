"""Planner HTTP probe guardrail tests.

Checks that the lightweight HEAD probe enforcing allowlists, header policies,
and retry behaviour raises `PolicyError` or respects caching when encountering
redirect chains, slow responses, or content-length anomalies.
"""

import logging

import pytest

from DocsToKG.OntologyDownload.errors import PolicyError
from DocsToKG.OntologyDownload.planning import planner_http_probe
from DocsToKG.OntologyDownload.testing import ResponseSpec


def _logger() -> logging.Logger:
    logger = logging.getLogger("ontology-planner-probe-test")
    logger.setLevel(logging.INFO)
    return logger


def test_planner_probe_redirect_chain_to_disallowed_host(ontology_env):
    """Planner probes must reject redirect chains that leave the allowlist."""

    start_path = "fixtures/planner-redirect.owl"
    intermediate_path = "fixtures/planner-redirect-intermediate.owl"
    disallowed_url = "https://malicious.example/ontology.owl"

    ontology_env.queue_response(
        start_path,
        ResponseSpec(
            method="HEAD",
            status=302,
            headers={"Location": f"/{intermediate_path}"},
        ),
    )
    ontology_env.queue_response(
        intermediate_path,
        ResponseSpec(
            method="HEAD",
            status=302,
            headers={"Location": disallowed_url},
        ),
    )

    config = ontology_env.build_download_config()
    url = ontology_env.http_url(start_path)

    with pytest.raises(PolicyError):
        planner_http_probe(
            url=url,
            http_config=config,
            logger=_logger(),
            service="test",
            context={"ontology_id": "test-ontology"},
        )

    methods_paths = [(record.method, record.path) for record in ontology_env.requests]
    assert methods_paths == [
        ("HEAD", "/" + start_path),
        ("HEAD", "/" + intermediate_path),
    ]


def test_planner_probe_redirect_to_disallowed_scheme(ontology_env):
    """Planner probes must reject redirects targeting unsupported schemes."""

    start_path = "fixtures/planner-scheme-redirect.owl"
    disallowed_url = "ftp://malicious.example/ontology.owl"

    ontology_env.queue_response(
        start_path,
        ResponseSpec(
            method="HEAD",
            status=302,
            headers={"Location": disallowed_url},
        ),
    )

    config = ontology_env.build_download_config()
    url = ontology_env.http_url(start_path)

    with pytest.raises(PolicyError):
        planner_http_probe(
            url=url,
            http_config=config,
            logger=_logger(),
            service="test",
            context={"ontology_id": "test-ontology"},
        )

    methods_paths = [(record.method, record.path) for record in ontology_env.requests]
    assert methods_paths == [("HEAD", "/" + start_path)]
