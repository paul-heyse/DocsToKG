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
