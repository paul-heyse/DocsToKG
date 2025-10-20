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
#     },
#     {
#       "id": "test-setup-logging-expands-env-log-dir",
#       "name": "test_setup_logging_expands_env_log_dir",
#       "anchor": "function-test-setup-logging-expands-env-log-dir",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Logging helper regression coverage.

Validates masking of sensitive fields, structured JSON formatting, log
rotation/compression, and convenience functions to ensure operational telemetry
remains safe and consumable by downstream systems.
"""

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
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import logging_utils as logging_utils_mod
from DocsToKG.OntologyDownload.io import mask_sensitive_data
from DocsToKG.OntologyDownload.logging_utils import setup_logging
from DocsToKG.OntologyDownload.settings import LoggingConfiguration

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


def test_mask_sensitive_data_masks_nested_structures():
    payload = {
        "Authorization": "ZXhhbXBsZVRva2VuZXhhbXBsZVRva2VuZXhhbXBsZVRva2Vu",
        "headers": {"Authorization": "Bearer secret-token"},
        "items": [
            {"token": "nested-secret"},
            "Bearer inline",
        ],
    }
    masked = mask_sensitive_data(payload)
    assert masked["Authorization"] == "***masked***"
    assert masked["headers"]["Authorization"] == "***masked***"
    assert masked["items"][0]["token"] == "***masked***"
    assert masked["items"][1] == "***masked***"


def test_mask_sensitive_data_masks_authorization_variants():
    payload = {
        "headers": {
            "Authorization": "Basic short-token",
            "X-Other": "value",
        },
        "header_list": [
            ("Authorization", "Bearer token"),
            ("X-Other", "still-visible"),
        ],
        "tuples": (
            "Authorization",
            "ignored",
            ("Authorization", "tiny"),
        ),
    }
    masked = mask_sensitive_data(payload)
    assert masked["headers"]["Authorization"] == "***masked***"
    assert masked["headers"]["X-Other"] == "value"
    header_list = masked["header_list"]
    assert header_list[0][0] == "Authorization"
    assert header_list[0][1] == "***masked***"
    assert header_list[1] == ("X-Other", "still-visible")
    tuples = masked["tuples"]
    assert tuples[0] == "Authorization"
    assert tuples[1] == "ignored"
    assert tuples[2][0] == "Authorization"
    assert tuples[2][1] == "***masked***"


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
                "extra_fields": {
                    "token": "secret",
                    "status": "ok",
                    "Authorization": "Basic short",
                    "headers": [("Authorization", "Bearer foo"), ("X-Test", "allowed")],
                },
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
        assert record["Authorization"] == "***masked***"
        assert record["headers"][0][0] == "Authorization"
        assert record["headers"][0][1] == "***masked***"
        assert record["headers"][1] == ["X-Test", "allowed"]
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def test_setup_logging_expands_env_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ONTOFETCH_LOG_DIR", "~/ontofetch-logs")

    logger = setup_logging()
    try:
        logger.info("env log directory expansion test")
        for handler in logger.handlers:
            handler.flush()
        file_handler = _get_file_handler(logger)
        log_path = Path(file_handler.baseFilename)
        expected_dir = (tmp_path / "ontofetch-logs").resolve()
        assert log_path.parent == expected_dir
        assert expected_dir.exists()
    finally:
        _cleanup_logger(logger)


def _get_file_handler(logger):
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            return handler
    raise AssertionError("expected a file handler")


def _cleanup_logger(logger):
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


@contextmanager
def temporary_env(**overrides):
    previous = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def temporary_attr(module, name, value):
    sentinel = object()
    previous = getattr(module, name, sentinel)
    setattr(module, name, value)
    try:
        yield
    finally:
        if previous is sentinel:
            delattr(module, name)
        else:
            setattr(module, name, previous)


def test_setup_logging_uses_env_dir_when_provided(ontology_env):
    custom_dir = ontology_env.root / "custom"
    with temporary_env(ONTOFETCH_LOG_DIR=f"  {custom_dir}  "):
        logger = setup_logging(level="INFO", retention_days=1, max_log_size_mb=1)
    try:
        file_handler = _get_file_handler(logger)
        assert Path(file_handler.baseFilename).parent == custom_dir
    finally:
        _cleanup_logger(logger)


def test_setup_logging_falls_back_when_env_blank(ontology_env):
    with temporary_env(ONTOFETCH_LOG_DIR="   "):
        logger = setup_logging(level="INFO", retention_days=1, max_log_size_mb=1)
    try:
        file_handler = _get_file_handler(logger)
        assert Path(file_handler.baseFilename).parent == ontology_env.log_dir
    finally:
        _cleanup_logger(logger)


def test_setup_logging_falls_back_when_env_missing(ontology_env):
    with temporary_env(ONTOFETCH_LOG_DIR=None):
        with temporary_attr(logging_utils_mod, "LOG_DIR", ontology_env.log_dir):
            logger = setup_logging(level="INFO", retention_days=1, max_log_size_mb=1)
    try:
        file_handler = _get_file_handler(logger)
        assert Path(file_handler.baseFilename).parent == ontology_env.log_dir
    finally:
        _cleanup_logger(logger)
