"""Tests for Wayback resolver telemetry integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from DocsToKG.ContentDownload.resolvers.wayback import (
    ResolverEvent,
    ResolverEventReason,
    WaybackResolver,
)


class _DummyConfig:
    def __init__(self) -> None:
        self.wayback_config = {}
        self.polite_headers = {}
        self.retry_after_cap = None

    def get_timeout(self, _name: str) -> int:
        return 10


class _DummyArtifact:
    def __init__(self, failed_pdf_urls: list[str], publication_year: int | None = None) -> None:
        self.failed_pdf_urls = failed_pdf_urls
        self.publication_year = publication_year


def _build_context() -> SimpleNamespace:
    return SimpleNamespace()


def test_wayback_iter_urls_emits_success_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = WaybackResolver()
    config = _DummyConfig()
    artifact = _DummyArtifact(["https://example.org/paper.pdf"], publication_year=2020)
    ctx = _build_context()

    snapshot_metadata = {"discovery_method": "availability", "mimetype": "application/pdf"}

    def fake_discover(*_args, **_kwargs):
        return "https://web.archive.org/web/20200101120000/http://example.org/paper.pdf", snapshot_metadata

    monkeypatch.setattr(resolver, "_discover_snapshots", fake_discover)

    results = list(resolver.iter_urls(object(), config, artifact, ctx=ctx))

    assert len(results) == 1
    result = results[0]
    assert result.url.startswith("https://web.archive.org/web/20200101120000/")
    assert result.url.endswith("example.org/paper.pdf")
    assert "telemetry" in result.metadata
    telemetry = result.metadata["telemetry"]
    assert telemetry["status"] == "success"

    events = telemetry["events"]
    assert {event["event"] for event in events} >= {"start", "discovery", "candidate", "complete"}
    complete_event = next(event for event in events if event["event"] == "complete")
    assert complete_event["status"] == "success"

    assert hasattr(ctx, "telemetry_wayback_events")
    ctx_events = ctx.telemetry_wayback_events
    assert len(ctx_events) >= 4
    assert ctx_events[-1]["status"] == "success"


def test_wayback_iter_urls_emits_skip_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = WaybackResolver()
    config = _DummyConfig()
    artifact = _DummyArtifact(["https://example.org/missing.pdf"], publication_year=2019)
    ctx = _build_context()

    def fake_discover(*_args, **_kwargs):
        return None, {"discovery_method": "cdx", "reason": "no_snapshot"}

    monkeypatch.setattr(resolver, "_discover_snapshots", fake_discover)

    results = list(resolver.iter_urls(object(), config, artifact, ctx=ctx))

    assert len(results) == 1
    result = results[0]
    assert result.event is ResolverEvent.SKIPPED
    assert result.event_reason is ResolverEventReason.NO_WAYBACK_SNAPSHOT
    telemetry = result.metadata["telemetry"]
    assert telemetry["status"] == "skip"
    assert telemetry["candidate_url"] is None

    ctx_events = ctx.telemetry_wayback_events
    complete_event = next(event for event in ctx_events if event["event"] == "complete")
    assert complete_event["status"] == "skip"


def test_wayback_iter_urls_emits_error_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = WaybackResolver()
    config = _DummyConfig()
    artifact = _DummyArtifact(["https://example.org/boom.pdf"])
    ctx = _build_context()

    def fake_discover(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(resolver, "_discover_snapshots", fake_discover)

    results = list(resolver.iter_urls(object(), config, artifact, ctx=ctx))

    assert len(results) == 1
    result = results[0]
    assert result.event is ResolverEvent.ERROR
    assert result.event_reason is ResolverEventReason.UNEXPECTED_ERROR
    telemetry = result.metadata["telemetry"]
    assert telemetry["status"] == "error"
    error_event = next(event for event in telemetry["events"] if event["event"] == "error")
    assert error_event["error"] == "boom"

    ctx_events = ctx.telemetry_wayback_events
    assert ctx_events[-1]["status"] == "error"
