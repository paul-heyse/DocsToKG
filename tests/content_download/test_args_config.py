"""Tests covering configuration resolution purity helpers."""

import logging
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import requests

from DocsToKG.ContentDownload import args as args_module
from DocsToKG.ContentDownload.args import (
    ResolvedConfig,
    bootstrap_run_environment,
    build_parser,
    parse_args,
    resolve_config,
)


def _build_args(tmp_path: Path):
    parser = build_parser()
    argv = [
        "--topic-id",
        "https://openalex.org/T12345",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(tmp_path / "pdfs"),
    ]
    args = parse_args(parser, argv)
    return parser, args


def test_resolve_config_is_pure(tmp_path: Path) -> None:
    parser, args = _build_args(tmp_path)
    snapshot = {key: getattr(args, key) for key in vars(args)}

    resolved = resolve_config(args, parser)

    assert isinstance(resolved, ResolvedConfig)
    assert not resolved.pdf_dir.exists()
    assert not resolved.html_dir.exists()
    assert not resolved.xml_dir.exists()

    current = {key: getattr(args, key) for key in vars(args)}
    assert current == snapshot
    assert not hasattr(args, "extract_html_text")
    assert resolved.extract_html_text is False

    with pytest.raises(FrozenInstanceError):
        resolved.run_id = "override"  # type: ignore[misc]


def test_bootstrap_run_environment_creates_directories(tmp_path: Path) -> None:
    parser, args = _build_args(tmp_path)
    resolved = resolve_config(args, parser)

    assert not resolved.pdf_dir.exists()
    assert not resolved.html_dir.exists()
    assert not resolved.xml_dir.exists()

    bootstrap_run_environment(resolved)

    assert resolved.pdf_dir.exists()
    assert resolved.html_dir.exists()
    assert resolved.xml_dir.exists()

    # Idempotent behaviour should not raise when repeated.
    bootstrap_run_environment(resolved)


def test_lookup_topic_id_requests_minimal_payload(monkeypatch, caplog) -> None:
    args_module._lookup_topic_id.cache_clear()

    class FakeTopics:
        def __init__(self) -> None:
            self.calls = []

        def search(self, text: str):
            self.calls.append(("search", text))
            return self

        def select(self, fields):
            self.calls.append(("select", tuple(fields)))
            return self

        def per_page(self, value: int):
            self.calls.append(("per_page", value))
            return self

        def get(self):
            self.calls.append(("get",))
            return [{"id": "https://openalex.org/T999"}]

    fake = FakeTopics()
    monkeypatch.setattr(args_module, "Topics", lambda: fake)
    caplog.set_level(logging.INFO)
    caplog.clear()

    resolved = args_module._lookup_topic_id("quantum photonics")

    assert resolved == "https://openalex.org/T999"
    assert fake.calls == [
        ("search", "quantum photonics"),
        ("select", ("id",)),
        ("per_page", 1),
        ("get",),
    ]
    assert (
        "DocsToKG.ContentDownload",
        logging.INFO,
        "Resolved topic 'quantum photonics' -> https://openalex.org/T999",
    ) in caplog.record_tuples


def test_lookup_topic_id_handles_empty_results(monkeypatch, caplog) -> None:
    args_module._lookup_topic_id.cache_clear()

    class EmptyTopics:
        def __init__(self) -> None:
            self.calls = []

        def search(self, text: str):
            self.calls.append(("search", text))
            return self

        def select(self, fields):
            self.calls.append(("select", tuple(fields)))
            return self

        def per_page(self, value: int):
            self.calls.append(("per_page", value))
            return self

        def get(self):
            self.calls.append(("get",))
            return []

    fake = EmptyTopics()
    monkeypatch.setattr(args_module, "Topics", lambda: fake)
    caplog.set_level(logging.INFO)
    caplog.clear()

    resolved = args_module._lookup_topic_id("obscure field")

    assert resolved is None
    assert fake.calls == [
        ("search", "obscure field"),
        ("select", ("id",)),
        ("per_page", 1),
        ("get",),
    ]
    assert not any(
        record[0] == "DocsToKG.ContentDownload" and record[1] == logging.INFO
        for record in caplog.record_tuples
    )


def test_lookup_topic_id_handles_request_exception(monkeypatch, caplog) -> None:
    args_module._lookup_topic_id.cache_clear()

    class ErrorTopics:
        def search(self, text: str):
            raise requests.RequestException("boom")

    monkeypatch.setattr(args_module, "Topics", lambda: ErrorTopics())
    caplog.set_level(logging.WARNING)
    caplog.clear()

    resolved = args_module._lookup_topic_id("failing topic")

    assert resolved is None
    assert any(
        record[0] == "DocsToKG.ContentDownload" and record[1] == logging.WARNING
        for record in caplog.record_tuples
    )
    assert any("Topic lookup failed" in record[2] for record in caplog.record_tuples)
