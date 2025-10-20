"""Planner HTTP probe guardrail tests.

Checks that the lightweight HEAD probe enforcing allowlists, header policies,
and retry behaviour raises `PolicyError` or respects caching when encountering
redirect chains, slow responses, or content-length anomalies.
"""

import logging
from collections import deque
from urllib.parse import urlparse

import pytest

from DocsToKG.OntologyDownload.errors import PolicyError
from DocsToKG.OntologyDownload.planning import planner_http_probe
from DocsToKG.OntologyDownload.settings import PlannerConfig
from DocsToKG.OntologyDownload.testing import ResponseSpec
from tests.conftest import PatchManager


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
    host = urlparse(url).hostname
    planner_cfg = PlannerConfig(head_precheck_hosts=[host] if host else [])

    with pytest.raises(PolicyError):
        planner_http_probe(
            url=url,
            http_config=config,
            logger=_logger(),
            service="test",
            context={"ontology_id": "test-ontology"},
            planner_config=planner_cfg,
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
    host = urlparse(url).hostname
    planner_cfg = PlannerConfig(head_precheck_hosts=[host] if host else [])

    with pytest.raises(PolicyError):
        planner_http_probe(
            url=url,
            http_config=config,
            logger=_logger(),
            service="test",
            context={"ontology_id": "test-ontology"},
            planner_config=planner_cfg,
        )

    methods_paths = [(record.method, record.path) for record in ontology_env.requests]
    assert methods_paths == [("HEAD", "/" + start_path)]


def test_planner_probe_applies_retry_after_delay(ontology_env, caplog):
    """Retry-After headers should reduce token availability for subsequent attempts."""

    caplog.set_level(logging.INFO)

    recorded_delays = []

    def _record_retry_after(*, http_config, service, host, delay):
        recorded_delays.append((service, host, delay))

    patcher = PatchManager()
    patcher.setattr(
        "DocsToKG.OntologyDownload.planning.apply_retry_after",
        _record_retry_after,
    )

    target_path = "fixtures/probe-retry-after.owl"
    head_key = ("HEAD", f"/{target_path}")
    existing = ontology_env._responses.get(head_key, [])
    head_queue = deque([
        ResponseSpec(
            method="HEAD",
            status=429,
            headers={
                "Retry-After": "3",
                "Content-Type": "application/rdf+xml",
                "Content-Length": "123",
            },
        ),
        ResponseSpec(
            method="HEAD",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": "123",
            },
        ),
        *list(existing),
    ])
    ontology_env._responses[head_key] = head_queue

    config = ontology_env.build_download_config()
    url = ontology_env.http_url(target_path)
    host = urlparse(url).hostname
    planner_cfg = PlannerConfig(head_precheck_hosts=[host] if host else [])

    try:
        result = planner_http_probe(
            url=url,
            http_config=config,
            logger=_logger(),
            service="test",
            context={"ontology_id": "retry-after"},
            planner_config=planner_cfg,
        )
    finally:
        patcher.close()

    if result is None:
        errors = [getattr(record, "error", None) for record in caplog.records]
        pytest.fail(f"planner probe failed: {errors}")

    assert result is not None
    assert result.status_code == 200
    assert recorded_delays
    service, host, delay = recorded_delays[0]
    assert service == "test"
    assert delay == pytest.approx(3.0)
    assert host
    methods = [record.method for record in ontology_env.requests]
    assert methods == ["HEAD", "HEAD"]


def test_planner_probe_defaults_to_get_with_range_header(ontology_env):
    """Planner probes should default to GET and request a single byte."""

    payload = b"@prefix : <http://example.org/> .\n:hp a :Ontology .\n"
    path = "fixtures/planner-default-get.owl"
    ontology_env.queue_response(
        path,
        ResponseSpec(
            method="GET",
            status=200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": str(len(payload)),
            },
            body=payload,
        ),
    )

    config = ontology_env.build_download_config()
    url = ontology_env.http_url(path)

    result = planner_http_probe(
        url=url,
        http_config=config,
        logger=_logger(),
        service="test",
        context={"ontology_id": "default-get"},
    )

    assert result is not None
    assert result.method == "GET"
    methods = [record.method for record in ontology_env.requests]
    assert methods == ["GET"]
    last_request = ontology_env.requests[-1]
    assert last_request.headers.get("Range") == "bytes=0-0"
