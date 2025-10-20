# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_doctor",
#   "purpose": "Deep coverage for the ``ontofetch doctor`` remediation workflow.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Deep coverage for the ``ontofetch doctor`` remediation workflow.

Validates log rotation, dependency reporting, cache inspection, and error exit
codes when the operator requests automated fixes. Guards against regressions in
the interactive troubleshooting experience."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from unittest.mock import patch

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.testing import TestingEnvironment
from tests.conftest import PatchManager


class _DoctorResponse:
    def __init__(self, status: int = 200, reason: str = "OK") -> None:
        self.status_code = status
        self.reason_phrase = reason

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 400


class _DoctorHttpClient:
    def __init__(self) -> None:
        self._response = _DoctorResponse()

    def head(self, *_args, **_kwargs):
        return self._response

    def get(self, *_args, **_kwargs):
        return self._response


def test_doctor_fix_rotates_jsonl(ontology_env, capsys):
    """`ontofetch doctor --fix` should rotate old JSONL logs in the temp environment."""

    log_dir = ontology_env.log_dir
    stale_log = log_dir / "ontofetch-20240101.jsonl"
    stale_log.write_text('{"message": "stale"}\n')
    expired_time = datetime.now(timezone.utc) - timedelta(days=31)
    os.utime(stale_log, (expired_time.timestamp(), expired_time.timestamp()))

    with patch.object(cli_module.net, "get_http_client", return_value=_DoctorHttpClient()):
        exit_code = cli_module.cli_main(["doctor", "--fix", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    # Note: Log cleanup happens during setup_logging() at CLI startup,
    # not during _apply_doctor_fixes(), so no fixes are reported
    assert payload.get("fixes", []) == []
    assert not stale_log.exists(), "stale JSONL log should be rotated"
    compressed = log_dir / "ontofetch-20240101.jsonl.gz"
    assert compressed.exists(), "compressed JSONL log should exist after rotation"


def test_doctor_fix_reports_invalid_rate_limit_override(capsys):
    """``doctor --fix`` should continue when defaults are invalid."""

    patcher = PatchManager()
    try:
        with TestingEnvironment(), patch.object(
            cli_module.net, "get_http_client", return_value=_DoctorHttpClient()
        ):
            patcher.setenv("ONTOFETCH_PER_HOST_RATE_LIMIT", "not-a-limit")
            exit_code = cli_module.cli_main(["doctor", "--fix", "--json"])
    finally:
        patcher.close()

    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    fixes = payload.get("fixes", [])
    assert any("Skipped log rotation" in fix for fix in fixes), fixes
    combined_errors = payload.get("rate_limits", {}).get("error", "")
    assert "Failed to load default rate limits" in combined_errors
