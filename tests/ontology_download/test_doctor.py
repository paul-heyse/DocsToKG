"""Doctor command regression tests."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from DocsToKG.OntologyDownload import cli as cli_module


def test_doctor_fix_rotates_jsonl(ontology_env, capsys):
    """`ontofetch doctor --fix` should rotate old JSONL logs in the temp environment."""

    log_dir = ontology_env.log_dir
    stale_log = log_dir / "ontofetch-20240101.jsonl"
    stale_log.write_text("{\"message\": \"stale\"}\n")
    expired_time = datetime.now(timezone.utc) - timedelta(days=31)
    os.utime(stale_log, (expired_time.timestamp(), expired_time.timestamp()))

    exit_code = cli_module.cli_main(["doctor", "--fix", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    # Note: Log cleanup happens during setup_logging() at CLI startup,
    # not during _apply_doctor_fixes(), so no fixes are reported
    assert payload.get("fixes", []) == []
    assert not stale_log.exists(), "stale JSONL log should be rotated"
    compressed = log_dir / "ontofetch-20240101.jsonl.gz"
    assert compressed.exists(), "compressed JSONL log should exist after rotation"
