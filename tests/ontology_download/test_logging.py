# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_logging",
#   "purpose": "Pytest coverage for ontology download logging scenarios",
#   "sections": [
#     {
#       "id": "test-mask-sensitive-data-masks-tokens",
#       "name": "test_mask_sensitive_data_masks_tokens",
#       "anchor": "function-test-mask-sensitive-data-masks-tokens",
#       "kind": "function"
#     },
#     {
#       "id": "test-setup-logging-emits-structured-json",
#       "name": "test_setup_logging_emits_structured_json",
#       "anchor": "function-test-setup-logging-emits-structured-json",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Ontology Logging Tests

This module validates logging helpers for the ontology download service,
ensuring sensitive fields are masked and structured JSON logs include
expected metadata.

Key Scenarios:
- Redacts API keys across header and payload fields
- Writes JSON log entries with contextual attributes for observability

Dependencies:
- DocsToKG.OntologyDownload: Logging utilities under test

Usage:
    pytest tests/ontology_download/test_logging.py
"""

import json

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import (
    LoggingConfiguration,
    mask_sensitive_data,
    setup_logging,
)
# --- Test Cases ---


def test_mask_sensitive_data_masks_tokens():
    payload = {
        "Authorization": "apikey ABC123",
        "nested": "value",
        "api_key": "secret",
    }
    masked = mask_sensitive_data(payload)
    assert masked["Authorization"] == "***masked***"
    assert masked["api_key"] == "***masked***"
    assert masked["nested"] == "value"


def test_setup_logging_emits_structured_json(tmp_path):
    config = LoggingConfiguration(level="INFO", max_log_size_mb=1, retention_days=1)
    logger = setup_logging(
        level=config.level,
        retention_days=config.retention_days,
        max_log_size_mb=config.max_log_size_mb,
        log_dir=tmp_path,
    )
    try:
        logger.info(
            "download complete",
            extra={
                "correlation_id": "abc123",
                "ontology_id": "hp",
                "stage": "download",
                "extra_fields": {"token": "secret", "status": "ok"},
            },
        )
        for handler in logger.handlers:
            handler.flush()
        log_files = sorted(tmp_path.glob("*.jsonl"))
        assert log_files, "expected a JSON log file"
        payload = log_files[0].read_text().strip()
        assert payload, "log file should contain an entry"
        record = json.loads(payload)
        assert record["message"] == "download complete"
        assert record["level"] == "INFO"
        assert record["correlation_id"] == "abc123"
        assert record["ontology_id"] == "hp"
        assert record["stage"] == "download"
        assert record["token"] == "***masked***"
        assert record["status"] == "ok"
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
