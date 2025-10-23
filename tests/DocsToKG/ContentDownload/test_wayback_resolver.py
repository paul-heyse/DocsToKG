from types import SimpleNamespace

from DocsToKG.ContentDownload.resolvers.base import (
    ResolverEvent,
    ResolverEventReason,
)
from DocsToKG.ContentDownload.resolvers.wayback import WaybackResolver


class _DummyConfig:
    wayback_config = {}
    polite_headers = {}
    retry_after_cap = None

    def get_timeout(self, _resolver: str) -> int:
        return 1


def test_wayback_no_snapshot_emits_openaccess_reason(monkeypatch) -> None:
    resolver = WaybackResolver()
    dummy_client = object()
    artifact = SimpleNamespace(
        failed_pdf_urls=["https://example.org/missing.pdf"],
        publication_year=None,
    )

    def _fake_discover(*args, **kwargs):
        return None, {"discovery_method": "none"}

    monkeypatch.setattr(resolver, "_discover_snapshots", _fake_discover)

    [result] = list(resolver.iter_urls(dummy_client, _DummyConfig(), artifact))

    assert result.event == ResolverEvent.SKIPPED
    assert result.event_reason == ResolverEventReason.NO_OPENACCESS_PDF
