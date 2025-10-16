# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_core_logic",
#   "purpose": "Pytest coverage for ontology download core logic scenarios",
#   "sections": [
#     {
#       "id": "stub_requests_metadata",
#       "name": "stub_requests_metadata",
#       "anchor": "SRM",
#       "kind": "function"
#     },
#     {
#       "id": "_make_plan",
#       "name": "_make_plan",
#       "anchor": "MP",
#       "kind": "function"
#     },
#     {
#       "id": "_run_fetch",
#       "name": "_run_fetch",
#       "anchor": "RF",
#       "kind": "function"
#     },
#     {
#       "id": "test_populate_plan_metadata_uses_http_config",
#       "name": "test_populate_plan_metadata_uses_http_config",
#       "anchor": "TPPMU",
#       "kind": "function"
#     },
#     {
#       "id": "test_plan_all_rejects_disallowed_host",
#       "name": "test_plan_all_rejects_disallowed_host",
#       "anchor": "TPARD",
#       "kind": "function"
#     },
#     {
#       "id": "_immediate_future",
#       "name": "_ImmediateFuture",
#       "anchor": "IMME",
#       "kind": "class"
#     },
#     {
#       "id": "test_plan_all_honors_concurrent_plan_limit",
#       "name": "test_plan_all_honors_concurrent_plan_limit",
#       "anchor": "TPAHC",
#       "kind": "function"
#     },
#     {
#       "id": "test_plan_all_skips_failures_when_configured",
#       "name": "test_plan_all_skips_failures_when_configured",
#       "anchor": "TPASF",
#       "kind": "function"
#     },
#     {
#       "id": "test_plan_one_uses_fallback",
#       "name": "test_plan_one_uses_fallback",
#       "anchor": "TPOUF",
#       "kind": "function"
#     },
#     {
#       "id": "test_plan_one_respects_disabled_fallback",
#       "name": "test_plan_one_respects_disabled_fallback",
#       "anchor": "TPORD",
#       "kind": "function"
#     },
#     {
#       "id": "test_fetch_one_download_fallback",
#       "name": "test_fetch_one_download_fallback",
#       "anchor": "TFODF",
#       "kind": "function"
#     },
#     {
#       "id": "test_manifest_fingerprint_ignores_target_format_order",
#       "name": "test_manifest_fingerprint_ignores_target_format_order",
#       "anchor": "TMFIT",
#       "kind": "function"
#     },
#     {
#       "id": "test_manifest_fingerprint_changes_with_normalization_mode",
#       "name": "test_manifest_fingerprint_changes_with_normalization_mode",
#       "anchor": "TMFCW",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Core module behavior tests."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import (
    ConfigError,
    DefaultsConfig,
    DownloadFailure,
    DownloadResult,
    ResolvedConfig,
    ValidationResult,
)
from DocsToKG.OntologyDownload import ontology_download as core
from DocsToKG.OntologyDownload.resolvers import FetchPlan


@pytest.fixture(autouse=True)
def stub_requests_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real HTTP calls when planning augments metadata."""

    class _Response:
        def __init__(self) -> None:
            self.status_code = 200
            self.ok = True
            self.headers = {
                "Last-Modified": "Wed, 01 Jan 2024 00:00:00 GMT",
                "Content-Length": "256",
            }

        def close(self) -> None:
            pass

    monkeypatch.setattr(core.requests, "head", lambda *args, **kwargs: _Response(), raising=False)
    monkeypatch.setattr(core.requests, "get", lambda *args, **kwargs: _Response(), raising=False)


def _make_plan() -> FetchPlan:
    return FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint=None,
        version="2024-01-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )


def _run_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_formats,
    normalization_mode: str,
    normalized_hash: str,
):
    plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint="hp.owl",
        version="2024-01-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )

    class Resolver:
        def plan(self, spec, config, logger):
            return plan

    monkeypatch.setitem(core.RESOLVERS, "obo", Resolver())

    defaults = DefaultsConfig(prefer_source=["obo"])
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=target_formats)

    cache_dir = tmp_path / "cache"
    ontology_dir = tmp_path / "ontologies"
    monkeypatch.setattr(core, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(core, "ONTOLOGY_DIR", ontology_dir)

    class _StubStorage:
        def finalize_version(self, ontology_id: str, version: str, base_dir: Path) -> None:
            pass

        def ensure_local_version(self, ontology_id: str, version: str) -> Path:
            path = ontology_dir / ontology_id / version
            path.mkdir(parents=True, exist_ok=True)
            return path

    monkeypatch.setattr(core, "STORAGE", _StubStorage())

    def _fake_run_validators(requests, logger):
        results = {}
        for request in requests:
            if request.name == "rdflib":
                details = {
                    "normalized_sha256": normalized_hash,
                    "normalization_mode": normalization_mode,
                }
            else:
                details = {}
            results[request.name] = ValidationResult(ok=True, details=details, output_files=[])
        return results

    monkeypatch.setattr(core, "run_validators", _fake_run_validators)
    monkeypatch.setattr(core, "validate_url_security", lambda url, config=None: url)

    def _fake_download_stream(**kwargs):
        destination: Path = kwargs["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = b"ontology"
        destination.write_bytes(payload)
        sha256 = hashlib.sha256(payload).hexdigest()
        return DownloadResult(
            path=destination,
            status="fresh",
            sha256=sha256,
            etag=None,
            last_modified=None,
        )

    monkeypatch.setattr(core, "download_stream", _fake_download_stream)

    logger = logging.getLogger("ontology-download-test")
    logger.setLevel(logging.INFO)
    result = core.fetch_one(spec, config=config, force=True, logger=logger)
    return json.loads(result.manifest_path.read_text())


def test_populate_plan_metadata_uses_http_config(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = DefaultsConfig()
    defaults.http.allowed_hosts = ["example.org"]
    config = ResolvedConfig(defaults=defaults, specs=[])
    planned = core.PlannedFetch(
        spec=core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=("owl",)),
        resolver="obo",
        plan=_make_plan(),
        candidates=(),
    )
    adapter = logging.LoggerAdapter(logging.getLogger("ontology-test"), extra={})

    enriched = core._populate_plan_metadata(planned, config, adapter)

    assert enriched.last_modified is not None
    assert enriched.last_modified_at is not None
    assert enriched.metadata.get("content_length") == 256


def test_plan_all_rejects_disallowed_host(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = DefaultsConfig()
    defaults.http.allowed_hosts = ["allowed.example"]
    config = ResolvedConfig(defaults=defaults, specs=[])

    class _Resolver:
        def plan(self, spec, config, logger):  # pragma: no cover - trivial stub
            return FetchPlan(
                url="https://blocked.example/hp.owl",
                headers={},
                filename_hint=None,
                version="2024-01-01",
                license="CC-BY",
                media_type="application/rdf+xml",
                service="obo",
            )

    monkeypatch.setitem(core.RESOLVERS, "blocked", _Resolver())

    spec = core.FetchSpec(id="hp", resolver="blocked", extras={}, target_formats=("owl",))

    with pytest.raises(ConfigError) as exc_info:
        core.plan_all([spec], config=config)

    assert "blocked.example" in str(exc_info.value)


class _ImmediateFuture:
    def __init__(self, value=None, error: Exception | None = None) -> None:
        self._value = value
        self._error = error

    def result(self):
        if self._error:
            raise self._error
        return self._value

    def cancel(self) -> None:  # pragma: no cover - compatibility shim
        self._error = RuntimeError("cancelled")


def test_plan_all_honors_concurrent_plan_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """plan_all should use configured concurrent_plans when submitting workers."""

    defaults = DefaultsConfig()
    defaults.http.concurrent_plans = 5
    config = ResolvedConfig(defaults=defaults, specs=[])
    specs = [
        core.FetchSpec(id=f"onto{i}", resolver="obo", extras={}, target_formats=("owl",))
        for i in range(3)
    ]

    recorded_workers: List[int] = []

    def fake_plan_one(spec, **_kwargs):
        return core.PlannedFetch(
            spec=spec,
            resolver="obo",
            plan=_make_plan(),
            candidates=(),
        )

    def fake_as_completed(iterable):
        yield from list(iterable)

    class DummyExecutor:
        def __init__(self, max_workers: int) -> None:
            recorded_workers.append(max_workers)
            self._futures: List[_ImmediateFuture] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            try:
                value = fn(*args, **kwargs)
                future = _ImmediateFuture(value=value)
            except Exception as exc:  # pragma: no cover - defensive
                future = _ImmediateFuture(error=exc)
            self._futures.append(future)
            return future

    monkeypatch.setattr(core, "ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr(core, "as_completed", fake_as_completed)
    monkeypatch.setattr(core, "plan_one", fake_plan_one)

    plans = core.plan_all(specs, config=config)

    assert recorded_workers == [5]
    assert [plan.spec.id for plan in plans] == [spec.id for spec in specs]


def test_plan_all_skips_failures_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """plan_all should continue when continue_on_error is enabled."""

    defaults = DefaultsConfig()
    defaults.http.concurrent_plans = 2
    defaults.continue_on_error = True
    config = ResolvedConfig(defaults=defaults, specs=[])

    specs = [
        core.FetchSpec(id="good", resolver="obo", extras={}, target_formats=("owl",)),
        core.FetchSpec(id="bad", resolver="obo", extras={}, target_formats=("owl",)),
    ]

    def fake_plan_one(spec, **_kwargs):
        if spec.id == "bad":
            raise core.ResolverError("boom")
        return core.PlannedFetch(
            spec=spec,
            resolver="obo",
            plan=_make_plan(),
            candidates=(),
        )

    def fake_as_completed(iterable):
        yield from list(iterable)

    class DummyExecutor:
        def __init__(self, max_workers: int) -> None:
            self._futures: List[_ImmediateFuture] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            try:
                value = fn(*args, **kwargs)
                future = _ImmediateFuture(value=value)
            except Exception as exc:
                future = _ImmediateFuture(error=exc)
            self._futures.append(future)
            return future

    monkeypatch.setattr(core, "ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr(core, "as_completed", fake_as_completed)
    monkeypatch.setattr(core, "plan_one", fake_plan_one)

    plans = core.plan_all(specs, config=config)

    assert len(plans) == 1
    assert plans[0].spec.id == "good"


def test_plan_one_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the primary resolver fails, plan_one should fall back to the next option."""

    class FailingResolver:
        def plan(self, spec, config, logger):
            raise ConfigError("boom")

    class SuccessfulResolver:
        def plan(self, spec, config, logger):
            return _make_plan()

    monkeypatch.setitem(core.RESOLVERS, "obo", FailingResolver())
    monkeypatch.setitem(core.RESOLVERS, "lov", SuccessfulResolver())

    defaults = DefaultsConfig(prefer_source=["obo", "lov"])
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    planned = core.plan_one(spec, config=config)

    assert planned.resolver == "lov"
    assert planned.plan.url.endswith("hp.owl")
    assert [candidate.resolver for candidate in planned.candidates] == ["lov"]


def test_plan_one_respects_disabled_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback should be skipped when resolver_fallback_enabled is False."""

    class FailingResolver:
        def plan(self, spec, config, logger):
            raise ConfigError("boom")

    monkeypatch.setitem(core.RESOLVERS, "obo", FailingResolver())

    defaults = DefaultsConfig(
        prefer_source=["obo", "lov"],
        resolver_fallback_enabled=False,
    )
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    with pytest.raises(core.ResolverError):
        core.plan_one(spec, config=config)


def test_fetch_one_download_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Download should fall back to the next resolver on retryable failure."""

    primary_plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={"Accept": "application/rdf+xml"},
        filename_hint="hp.owl",
        version="2024-01-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )
    fallback_plan = FetchPlan(
        url="https://mirror.example.org/hp.owl",
        headers={"Accept": "application/rdf+xml"},
        filename_hint="hp.owl",
        version="2024-01-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="lov",
    )

    class PrimaryResolver:
        def plan(self, spec, config, logger):
            return primary_plan

    class SecondaryResolver:
        def plan(self, spec, config, logger):
            return fallback_plan

    monkeypatch.setitem(core.RESOLVERS, "obo", PrimaryResolver())
    monkeypatch.setitem(core.RESOLVERS, "lov", SecondaryResolver())

    defaults = DefaultsConfig(prefer_source=["lov"])
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    cache_dir = tmp_path / "cache"
    ontology_dir = tmp_path / "ontologies"
    monkeypatch.setattr(core, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(core, "ONTOLOGY_DIR", ontology_dir)

    class _StubStorage:
        def finalize_version(self, ontology_id: str, version: str, base_dir: Path) -> None:
            pass

        def ensure_local_version(self, ontology_id: str, version: str) -> Path:
            path = ontology_dir / ontology_id / version
            path.mkdir(parents=True, exist_ok=True)
            return path

    monkeypatch.setattr(core, "STORAGE", _StubStorage())

    def _fake_run_validators(requests, logger):
        results = {}
        for request in requests:
            details = (
                {"normalized_sha256": "norm", "normalization_mode": "in-memory"}
                if request.name == "rdflib"
                else {}
            )
            results[request.name] = ValidationResult(ok=True, details=details, output_files=[])
        return results

    monkeypatch.setattr(core, "run_validators", _fake_run_validators)
    monkeypatch.setattr(core, "validate_url_security", lambda url, config=None: url)

    attempts = {"count": 0}

    def _fake_download_stream(**kwargs):
        attempts["count"] += 1
        destination: Path = kwargs["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if attempts["count"] == 1:
            raise DownloadFailure("temporary outage", status_code=503, retryable=True)
        payload = b"ontology"
        destination.write_bytes(payload)
        sha256 = hashlib.sha256(payload).hexdigest()
        return DownloadResult(
            path=destination,
            status="fresh",
            sha256=sha256,
            etag=None,
            last_modified=None,
        )

    monkeypatch.setattr(core, "download_stream", _fake_download_stream)

    logger = logging.getLogger("ontology-download-test")
    logger.setLevel(logging.INFO)

    result = core.fetch_one(spec, config=config, force=True, logger=logger)

    assert attempts["count"] == 2
    assert result.spec.resolver == "lov"
    assert result.local_path.exists()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["schema_version"] == core.MANIFEST_SCHEMA_VERSION
    assert manifest["resolver"] == "lov"
    assert manifest["url"] == "https://mirror.example.org/hp.owl"
    chain = manifest["resolver_attempts"]
    assert len(chain) == 2
    assert chain[0]["resolver"] == "obo"
    assert chain[0]["status"] == "failed"
    assert chain[1]["resolver"] == "lov"
    assert chain[1]["status"] == "success"


def test_manifest_fingerprint_ignores_target_format_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_a = _run_fetch(
        monkeypatch,
        tmp_path,
        ["owl", "obo"],
        normalization_mode="in-memory",
        normalized_hash="hash",
    )
    manifest_b = _run_fetch(
        monkeypatch,
        tmp_path,
        ["obo", "owl"],
        normalization_mode="in-memory",
        normalized_hash="hash",
    )
    assert manifest_a["fingerprint"] == manifest_b["fingerprint"]


def test_manifest_fingerprint_changes_with_normalization_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_in_memory = _run_fetch(
        monkeypatch,
        tmp_path,
        ["owl"],
        normalization_mode="in-memory",
        normalized_hash="hash",
    )
    manifest_streaming = _run_fetch(
        monkeypatch,
        tmp_path,
        ["owl"],
        normalization_mode="streaming",
        normalized_hash="hash",
    )
    assert manifest_in_memory["fingerprint"] != manifest_streaming["fingerprint"]
