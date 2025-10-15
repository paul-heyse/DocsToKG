from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyalex")

from DocsToKG.ContentDownload.download_pyalex_pdfs import load_resolver_config


def test_deprecated_resolver_rate_limits_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{" "\"resolver_rate_limits\": {\"unpaywall\": 2.0}" "}")
    args = Namespace(
        resolver_config=str(config_path),
        unpaywall_email=None,
        core_api_key=None,
        semantic_scholar_api_key=None,
        doaj_api_key=None,
        mailto=None,
        max_resolver_attempts=None,
        resolver_timeout=None,
        disable_resolver=[],
        resolver_order=None,
        log_jsonl=None,
        log_format="jsonl",
        resume_from=None,
    )
    caplog.set_level("WARNING")
    config = load_resolver_config(args, ["unpaywall"], None)
    assert config.resolver_min_interval_s["unpaywall"] == 2.0
    assert any("resolver_rate_limits deprecated" in record.message for record in caplog.records)


def test_user_agent_includes_mailto(tmp_path: Path) -> None:
    args = Namespace(
        resolver_config=None,
        unpaywall_email=None,
        core_api_key=None,
        semantic_scholar_api_key=None,
        doaj_api_key=None,
        mailto="ua-tester@example.org",
        max_resolver_attempts=None,
        resolver_timeout=None,
        disable_resolver=[],
        resolver_order=None,
        log_jsonl=None,
        log_format="jsonl",
        resume_from=None,
    )

    config = load_resolver_config(args, ["unpaywall", "crossref"], None)
    user_agent = config.polite_headers.get("User-Agent")
    assert user_agent == "DocsToKGDownloader/1.0 (+ua-tester@example.org; mailto:ua-tester@example.org)"
