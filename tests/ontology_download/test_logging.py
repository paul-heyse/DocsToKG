import json

from DocsToKG.OntologyDownload.config import LoggingConfig
from DocsToKG.OntologyDownload.logging_config import mask_sensitive_data, setup_logging


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
    config = LoggingConfig(level="INFO", max_log_size_mb=1, retention_days=1)
    logger = setup_logging(config, log_dir=tmp_path)
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
