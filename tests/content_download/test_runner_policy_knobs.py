import sys
from types import ModuleType, SimpleNamespace


def test_download_run_propagates_download_policy_knobs(monkeypatch) -> None:
    telemetry_stub = ModuleType("DocsToKG.ContentDownload.telemetry_wayback")
    telemetry_stub.TelemetryWayback = object  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "DocsToKG.ContentDownload.telemetry_wayback",
        telemetry_stub,
    )

    from DocsToKG.ContentDownload.config import ContentDownloadConfig
    from DocsToKG.ContentDownload.runner import DownloadRun

    captured: dict[str, object] = {}

    class _FakeResult:
        def __init__(self) -> None:
            self.run_id = "unit-test"
            self.success_count = 0
            self.skip_count = 0
            self.error_count = 0

    def fake_run_from_config(*, config, artifacts, dry_run):  # type: ignore[override]
        ctx = SimpleNamespace(**(config.policy_knobs or {}))
        captured["chunk_size_bytes"] = getattr(ctx, "chunk_size_bytes", None)
        captured["atomic_write"] = getattr(ctx, "atomic_write", None)
        captured["verify_content_length"] = getattr(
            ctx, "verify_content_length", None
        )
        return _FakeResult()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.runner.run_from_config", fake_run_from_config
    )
    monkeypatch.setattr(DownloadRun, "_build_resolvers", lambda self: ({}, {}))

    base_config = ContentDownloadConfig()
    config = base_config.model_copy(
        update={
            "download": base_config.download.model_copy(
                update={
                    "chunk_size_bytes": 4096,
                    "atomic_write": False,
                    "verify_content_length": False,
                }
            )
        }
    )

    with DownloadRun(config) as runner:
        runner.process_artifacts([])

    assert captured["chunk_size_bytes"] == 4096
    assert captured["atomic_write"] is False
    assert captured["verify_content_length"] is False
