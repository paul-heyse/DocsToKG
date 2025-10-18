# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_core_logic",
#   "purpose": "Pytest coverage for ontology download core logic scenarios",
#   "sections": [
#     {
#       "id": "stub-requests-metadata",
#       "name": "stub_requests_metadata",
#       "anchor": "function-stub-requests-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "make-plan",
#       "name": "_make_plan",
#       "anchor": "function-make-plan",
#       "kind": "function"
#     },
#     {
#       "id": "run-fetch",
#       "name": "_run_fetch",
#       "anchor": "function-run-fetch",
#       "kind": "function"
#     },
#     {
#       "id": "test-populate-plan-metadata-uses-http-config",
#       "name": "test_populate_plan_metadata_uses_http_config",
#       "anchor": "function-test-populate-plan-metadata-uses-http-config",
#       "kind": "function"
#     },
#     {
#       "id": "test-plan-all-rejects-disallowed-host",
#       "name": "test_plan_all_rejects_disallowed_host",
#       "anchor": "function-test-plan-all-rejects-disallowed-host",
#       "kind": "function"
#     },
#     {
#       "id": "immediatefuture",
#       "name": "_ImmediateFuture",
#       "anchor": "class-immediatefuture",
#       "kind": "class"
#     },
#     {
#       "id": "test-plan-all-honors-concurrent-plan-limit",
#       "name": "test_plan_all_honors_concurrent_plan_limit",
#       "anchor": "function-test-plan-all-honors-concurrent-plan-limit",
#       "kind": "function"
#     },
#     {
#       "id": "test-plan-all-skips-failures-when-configured",
#       "name": "test_plan_all_skips_failures_when_configured",
#       "anchor": "function-test-plan-all-skips-failures-when-configured",
#       "kind": "function"
#     },
#     {
#       "id": "test-plan-one-uses-fallback",
#       "name": "test_plan_one_uses_fallback",
#       "anchor": "function-test-plan-one-uses-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "test-plan-one-respects-disabled-fallback",
#       "name": "test_plan_one_respects_disabled_fallback",
#       "anchor": "function-test-plan-one-respects-disabled-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "test-fetch-one-download-fallback",
#       "name": "test_fetch_one_download_fallback",
#       "anchor": "function-test-fetch-one-download-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "test-manifest-fingerprint-ignores-target-format-order",
#       "name": "test_manifest_fingerprint_ignores_target_format_order",
#       "anchor": "function-test-manifest-fingerprint-ignores-target-format-order",
#       "kind": "function"
#     },
#     {
#       "id": "test-manifest-fingerprint-changes-with-normalization-mode",
#       "name": "test_manifest_fingerprint_changes_with_normalization_mode",
#       "anchor": "function-test-manifest-fingerprint-changes-with-normalization-mode",
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
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import requests
from requests.structures import CaseInsensitiveDict

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

import DocsToKG.OntologyDownload.planning as pipeline_mod
from DocsToKG.OntologyDownload import api as core
from DocsToKG.OntologyDownload import settings as settings_mod
from DocsToKG.OntologyDownload.errors import PolicyError
from DocsToKG.OntologyDownload.io import DownloadResult
from DocsToKG.OntologyDownload.planning import (
    RESOLVERS,
    FetchPlan,
    PlannerProbeResult,
    ResolverError,
)
from DocsToKG.OntologyDownload.settings import (
    ConfigError,
    DefaultsConfig,
    DownloadConfiguration,
    DownloadFailure,
    ResolvedConfig,
)
from DocsToKG.OntologyDownload.validation import ValidationResult

storage_mod = settings_mod


@pytest.fixture(autouse=True)
def test_select_validators_for_zip_only_arelle():
    assert pipeline_mod._select_validators("application/zip") == ["arelle"]


def test_select_validators_for_non_rdf_skips_rdf_validators():
    selected = pipeline_mod._select_validators("application/json")
    assert "rdflib" not in selected
    assert "robot" not in selected


def test_select_validators_for_rdf_includes_defaults():
    assert pipeline_mod._select_validators("application/rdf+xml") == list(
        pipeline_mod.DEFAULT_VALIDATOR_NAMES
    )


@pytest.fixture(autouse=True)
def stub_requests_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real HTTP calls when planning augments metadata."""

    def _fake_probe(**kwargs) -> PlannerProbeResult:
        return PlannerProbeResult(
            url=kwargs.get("url", "https://example.org/resource.owl"),
            method=kwargs.get("method", "HEAD"),
            status_code=200,
            ok=True,
            headers=CaseInsensitiveDict(
                {
                    "Last-Modified": "Wed, 01 Jan 2024 00:00:00 GMT",
                    "Content-Length": "256",
                }
            ),
        )

    monkeypatch.setattr(pipeline_mod, "planner_http_probe", _fake_probe, raising=False)


# --- Helper Functions ---


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

    monkeypatch.setitem(RESOLVERS, "obo", Resolver())

    defaults = DefaultsConfig(prefer_source=["obo"])
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=target_formats)

    cache_dir = tmp_path / "cache"
    ontology_dir = tmp_path / "ontologies"
    overrides = {
        "CACHE_DIR": cache_dir,
        "LOCAL_ONTOLOGY_DIR": ontology_dir,
    }
    for attr, value in overrides.items():
        monkeypatch.setattr(storage_mod, attr, value, raising=False)
        monkeypatch.setattr(pipeline_mod, attr, value, raising=False)
        monkeypatch.setattr(core, attr, value, raising=False)
    monkeypatch.setattr(core, "ONTOLOGY_DIR", ontology_dir, raising=False)

    class _StubStorage:
        def finalize_version(
            self,
            ontology_id: str,
            version: str,
            base_dir: Path,
            *,
            artifact_path: Optional[Path] = None,
            artifact_sha256: Optional[str] = None,
        ) -> None:
            pass

        def mirror_cas_artifact(self, algorithm: str, digest: str, source: Path) -> Path:
            return source

        def ensure_local_version(self, ontology_id: str, version: str) -> Path:
            path = ontology_dir / ontology_id / version
            path.mkdir(parents=True, exist_ok=True)
            return path

    stub_storage = _StubStorage()
    monkeypatch.setattr(storage_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(pipeline_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(core, "STORAGE", stub_storage, raising=False)

    def _fake_run_validators(requests, logger):
        results = {}
        for request in requests:
            if request.name == "rdflib":
                details = {
                    "normalized_sha256": normalized_hash,
                    "normalization_mode": normalization_mode,
                    "streaming_nt_sha256": f"{normalized_hash}-stream",
                    "streaming_prefix_sha256": f"prefix-{normalization_mode}",
                }
            else:
                details = {}
            results[request.name] = ValidationResult(ok=True, details=details, output_files=[])
        return results

    monkeypatch.setattr(pipeline_mod, "run_validators", _fake_run_validators, raising=False)
    monkeypatch.setattr(core, "run_validators", _fake_run_validators, raising=False)
    monkeypatch.setattr(
        pipeline_mod, "validate_url_security", lambda url, config=None: url, raising=False
    )
    monkeypatch.setattr(core, "validate_url_security", lambda url, config=None: url, raising=False)

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
            content_type="application/rdf+xml",
            content_length=len(payload),
        )

    monkeypatch.setattr(pipeline_mod, "download_stream", _fake_download_stream, raising=False)
    monkeypatch.setattr(core, "download_stream", _fake_download_stream, raising=False)

    logger = logging.getLogger("ontology-download-test")
    logger.setLevel(logging.INFO)
    result = core.fetch_one(spec, config=config, force=True, logger=logger)
    return json.loads(result.manifest_path.read_text())


# --- Test Cases ---


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

    enriched = pipeline_mod._populate_plan_metadata(planned, config, adapter)

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

    monkeypatch.setitem(RESOLVERS, "blocked", _Resolver())

    spec = core.FetchSpec(id="hp", resolver="blocked", extras={}, target_formats=("owl",))

    with pytest.raises(PolicyError) as exc_info:
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

    monkeypatch.setattr(pipeline_mod, "ThreadPoolExecutor", DummyExecutor, raising=False)
    monkeypatch.setattr(core, "ThreadPoolExecutor", DummyExecutor, raising=False)
    monkeypatch.setattr(pipeline_mod, "as_completed", fake_as_completed, raising=False)
    monkeypatch.setattr(core, "as_completed", fake_as_completed, raising=False)
    monkeypatch.setattr(pipeline_mod, "plan_one", fake_plan_one, raising=False)
    monkeypatch.setattr(core, "plan_one", fake_plan_one, raising=False)

    plans = core.plan_all(specs, config=config)

    assert recorded_workers == [5]
    assert [plan.spec.id for plan in plans] == [spec.id for spec in specs]


def test_plan_all_aborts_on_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """plan_all cancels outstanding work when a resolver fails."""

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
            raise ResolverError("boom")
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

    monkeypatch.setattr(pipeline_mod, "ThreadPoolExecutor", DummyExecutor, raising=False)
    monkeypatch.setattr(core, "ThreadPoolExecutor", DummyExecutor, raising=False)
    monkeypatch.setattr(pipeline_mod, "as_completed", fake_as_completed, raising=False)
    monkeypatch.setattr(core, "as_completed", fake_as_completed, raising=False)
    monkeypatch.setattr(pipeline_mod, "plan_one", fake_plan_one, raising=False)
    monkeypatch.setattr(core, "plan_one", fake_plan_one, raising=False)

    with pytest.raises(core.BatchPlanningError) as excinfo:
        core.plan_all(specs, config=config)

    assert excinfo.value.failed_spec.id == "bad"
    assert [plan.spec.id for plan in excinfo.value.completed] == ["good"]


def test_plan_all_since_filters_outdated_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = DefaultsConfig()
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="direct", extras={}, target_formats=("owl",))

    def fake_plan_one(spec, **_kwargs):
        fetch_plan = FetchPlan(
            url="https://example.org/hp.owl",
            headers={},
            filename_hint=None,
            version=None,
            license="CC-BY",
            media_type="application/rdf+xml",
            service="direct",
            last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        )
        return core.PlannedFetch(
            spec=spec,
            resolver="direct",
            plan=fetch_plan,
            candidates=(),
            last_modified=fetch_plan.last_modified,
            last_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )

    def fake_as_completed(iterable):
        yield from list(iterable)

    class DummyExecutor:
        def __init__(self, max_workers: int) -> None:  # pragma: no cover - simple stub
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

    def fake_setup_logging(*, level="INFO", retention_days=30, max_log_size_mb=100, log_dir=None):
        logger = logging.getLogger("plan-since-filter")
        logger.handlers[:] = []
        logger.addHandler(logging.NullHandler())
        logger.setLevel(level)
        logger.propagate = False
        return logger

    monkeypatch.setattr(pipeline_mod, "ThreadPoolExecutor", DummyExecutor, raising=False)
    monkeypatch.setattr(core, "ThreadPoolExecutor", DummyExecutor, raising=False)
    monkeypatch.setattr(pipeline_mod, "as_completed", fake_as_completed, raising=False)
    monkeypatch.setattr(core, "as_completed", fake_as_completed, raising=False)
    monkeypatch.setattr(pipeline_mod, "plan_one", fake_plan_one, raising=False)
    monkeypatch.setattr(core, "plan_one", fake_plan_one, raising=False)
    monkeypatch.setattr(pipeline_mod, "setup_logging", fake_setup_logging, raising=False)
    monkeypatch.setattr(core, "setup_logging", fake_setup_logging, raising=False)
    monkeypatch.setattr(pipeline_mod, "ensure_python_version", lambda: None, raising=False)
    monkeypatch.setattr(core, "ensure_python_version", lambda: None, raising=False)
    monkeypatch.setattr(
        pipeline_mod, "generate_correlation_id", lambda: "fixed-correlation", raising=False
    )
    monkeypatch.setattr(core, "generate_correlation_id", lambda: "fixed-correlation", raising=False)

    cutoff = datetime(2021, 1, 1, tzinfo=timezone.utc)
    plans = core.plan_all([spec], config=config, since=cutoff)

    assert plans == []


def test_plan_all_since_fetches_missing_last_modified(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = DefaultsConfig()
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="go", resolver="direct", extras={}, target_formats=("owl",))

    def fake_plan_one(spec, **_kwargs):
        fetch_plan = FetchPlan(
            url="https://example.org/go.owl",
            headers={},
            filename_hint=None,
            version=None,
            license="CC-BY",
            media_type="application/rdf+xml",
            service="direct",
            last_modified=None,
        )
        return core.PlannedFetch(
            spec=spec,
            resolver="direct",
            plan=fetch_plan,
            candidates=(),
            last_modified=None,
            last_modified_at=None,
        )

    def fake_as_completed(iterable):
        yield from list(iterable)

    class DummyExecutor:
        def __init__(self, max_workers: int) -> None:  # pragma: no cover - simple stub
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

    fetched = {"count": 0}
    header_value = "Wed, 05 Feb 2025 00:00:00 GMT"

    def fake_fetch_last_modified(plan, _config, _logger):
        fetched["count"] += 1
        return header_value

    def fake_setup_logging(*, level="INFO", retention_days=30, max_log_size_mb=100, log_dir=None):
        logger = logging.getLogger("plan-since-head")
        logger.handlers[:] = []
        logger.addHandler(logging.NullHandler())
        logger.setLevel(level)
        logger.propagate = False
        return logger

    monkeypatch.setattr(pipeline_mod, "ThreadPoolExecutor", DummyExecutor, raising=False)
    monkeypatch.setattr(core, "ThreadPoolExecutor", DummyExecutor, raising=False)
    monkeypatch.setattr(pipeline_mod, "as_completed", fake_as_completed, raising=False)
    monkeypatch.setattr(core, "as_completed", fake_as_completed, raising=False)
    monkeypatch.setattr(pipeline_mod, "plan_one", fake_plan_one, raising=False)
    monkeypatch.setattr(core, "plan_one", fake_plan_one, raising=False)
    monkeypatch.setattr(
        pipeline_mod, "_fetch_last_modified", fake_fetch_last_modified, raising=False
    )
    monkeypatch.setattr(core, "_fetch_last_modified", fake_fetch_last_modified, raising=False)
    monkeypatch.setattr(pipeline_mod, "setup_logging", fake_setup_logging, raising=False)
    monkeypatch.setattr(core, "setup_logging", fake_setup_logging, raising=False)
    monkeypatch.setattr(pipeline_mod, "ensure_python_version", lambda: None, raising=False)
    monkeypatch.setattr(core, "ensure_python_version", lambda: None, raising=False)
    monkeypatch.setattr(
        pipeline_mod, "generate_correlation_id", lambda: "fixed-correlation", raising=False
    )
    monkeypatch.setattr(core, "generate_correlation_id", lambda: "fixed-correlation", raising=False)

    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plans = core.plan_all([spec], config=config, since=cutoff)

    assert fetched["count"] == 1
    assert len(plans) == 1
    plan = plans[0]
    assert plan.last_modified == header_value
    assert plan.plan.last_modified == header_value
    assert plan.last_modified_at == datetime(2025, 2, 5, tzinfo=timezone.utc)


def test_plan_one_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the primary resolver fails, plan_one should fall back to the next option."""

    class FailingResolver:
        def plan(self, spec, config, logger):
            raise ConfigError("boom")

    class SuccessfulResolver:
        def plan(self, spec, config, logger):
            return _make_plan()

    monkeypatch.setitem(RESOLVERS, "obo", FailingResolver())
    monkeypatch.setitem(RESOLVERS, "lov", SuccessfulResolver())

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

    monkeypatch.setitem(RESOLVERS, "obo", FailingResolver())

    defaults = DefaultsConfig(
        prefer_source=["obo", "lov"],
        resolver_fallback_enabled=False,
    )
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    with pytest.raises(ResolverError):
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

    monkeypatch.setitem(RESOLVERS, "obo", PrimaryResolver())
    monkeypatch.setitem(RESOLVERS, "lov", SecondaryResolver())

    defaults = DefaultsConfig(prefer_source=["lov"])
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    cache_dir = tmp_path / "cache"
    ontology_dir = tmp_path / "ontologies"
    overrides = {
        "CACHE_DIR": cache_dir,
        "LOCAL_ONTOLOGY_DIR": ontology_dir,
    }
    for attr, value in overrides.items():
        monkeypatch.setattr(storage_mod, attr, value, raising=False)
        monkeypatch.setattr(pipeline_mod, attr, value, raising=False)
        monkeypatch.setattr(core, attr, value, raising=False)
    monkeypatch.setattr(core, "ONTOLOGY_DIR", ontology_dir, raising=False)

    class _StubStorage:
        def finalize_version(
            self,
            ontology_id: str,
            version: str,
            base_dir: Path,
            *,
            artifact_path: Optional[Path] = None,
            artifact_sha256: Optional[str] = None,
        ) -> None:
            pass

        def ensure_local_version(self, ontology_id: str, version: str) -> Path:
            path = ontology_dir / ontology_id / version
            path.mkdir(parents=True, exist_ok=True)
            return path

        def mirror_cas_artifact(self, algorithm: str, digest: str, source: Path) -> Path:
            return source

    stub_storage = _StubStorage()
    monkeypatch.setattr(storage_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(pipeline_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(core, "STORAGE", stub_storage, raising=False)

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

    monkeypatch.setattr(pipeline_mod, "run_validators", _fake_run_validators, raising=False)
    monkeypatch.setattr(core, "run_validators", _fake_run_validators, raising=False)
    monkeypatch.setattr(
        pipeline_mod, "validate_url_security", lambda url, config=None: url, raising=False
    )
    monkeypatch.setattr(core, "validate_url_security", lambda url, config=None: url, raising=False)

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
            content_type="application/rdf+xml",
            content_length=len(payload),
        )

    monkeypatch.setattr(pipeline_mod, "download_stream", _fake_download_stream, raising=False)
    monkeypatch.setattr(core, "download_stream", _fake_download_stream, raising=False)

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


def test_fetch_records_expected_checksum_and_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"ontology-data"
    expected_digest = hashlib.sha256(payload).hexdigest()

    checksum_plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint="hp.owl",
        version="2024-03-01",
        license="CC0-1.0",
        media_type="application/rdf+xml",
        service="obo",
        checksum=expected_digest,
        checksum_algorithm="sha256",
    )

    class ChecksumResolver:
        def plan(self, spec, config, logger):
            return checksum_plan

    monkeypatch.setitem(RESOLVERS, "obo", ChecksumResolver())

    defaults = DefaultsConfig(prefer_source=["obo"])
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    cache_dir = tmp_path / "cache"
    ontology_dir = tmp_path / "ontologies"
    for attr, value in {
        "CACHE_DIR": cache_dir,
        "LOCAL_ONTOLOGY_DIR": ontology_dir,
    }.items():
        monkeypatch.setattr(storage_mod, attr, value, raising=False)
        monkeypatch.setattr(pipeline_mod, attr, value, raising=False)
        monkeypatch.setattr(core, attr, value, raising=False)
    monkeypatch.setattr(core, "ONTOLOGY_DIR", ontology_dir, raising=False)

    class _StubStorage:
        def finalize_version(
            self,
            ontology_id: str,
            version: str,
            base_dir: Path,
            *,
            artifact_path: Optional[Path] = None,
            artifact_sha256: Optional[str] = None,
        ) -> None:
            pass

        def ensure_local_version(self, ontology_id: str, version: str) -> Path:
            path = ontology_dir / ontology_id / version
            path.mkdir(parents=True, exist_ok=True)
            return path

        def mirror_cas_artifact(self, algorithm: str, digest: str, source: Path) -> Path:
            return source

    stub_storage = _StubStorage()
    monkeypatch.setattr(storage_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(pipeline_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(core, "STORAGE", stub_storage, raising=False)

    captured: Dict[str, Optional[str]] = {"expected_hash": None}

    def _fake_download_stream(**kwargs):
        captured["expected_hash"] = kwargs.get("expected_hash")
        destination: Path = kwargs["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return DownloadResult(
            path=destination,
            status="fresh",
            sha256=expected_digest,
            etag="etag",
            last_modified="today",
            content_type="application/rdf+xml",
            content_length=len(payload),
        )

    monkeypatch.setattr(pipeline_mod, "download_stream", _fake_download_stream, raising=False)
    monkeypatch.setattr(core, "download_stream", _fake_download_stream, raising=False)

    def _fake_run_validators(requests, logger):
        return {
            request.name: ValidationResult(
                ok=True,
                details={"normalized_sha256": expected_digest, "normalization_mode": "in-memory"},
                output_files=[],
            )
            for request in requests
        }

    monkeypatch.setattr(pipeline_mod, "run_validators", _fake_run_validators, raising=False)
    monkeypatch.setattr(core, "run_validators", _fake_run_validators, raising=False)
    monkeypatch.setattr(
        pipeline_mod, "validate_url_security", lambda url, config=None: url, raising=False
    )
    monkeypatch.setattr(core, "validate_url_security", lambda url, config=None: url, raising=False)

    results = core.fetch_all([spec], config=config, force=True)
    assert captured["expected_hash"] == f"sha256:{expected_digest}"

    manifest = json.loads(results[0].manifest_path.read_text())
    attempt = manifest["resolver_attempts"][0]
    assert attempt["expected_checksum"] == {
        "algorithm": "sha256",
        "value": expected_digest,
    }
    assert manifest["expected_checksum"] == {
        "algorithm": "sha256",
        "value": expected_digest,
    }

    index_path = ontology_dir / spec.id / "index.json"
    index_payload = json.loads(index_path.read_text())
    assert isinstance(index_payload, list) and index_payload
    first_entry = index_payload[0]
    assert first_entry["sha256"] == expected_digest
    assert first_entry["expected_checksum"] == {
        "algorithm": "sha256",
        "value": expected_digest,
    }
    assert first_entry["size"] == len(payload)


def test_fetch_mirrors_cas_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = b"ontology"
    digest = hashlib.sha256(payload).hexdigest()

    plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint="hp.owl",
        version="2024-04-01",
        license="CC0-1.0",
        media_type="application/rdf+xml",
        service="direct",
    )

    class DirectResolver:
        def plan(self, spec, config, logger):
            return plan

    original_direct = RESOLVERS.get("direct")
    RESOLVERS["direct"] = DirectResolver()

    defaults = DefaultsConfig(prefer_source=["direct"], enable_cas_mirror=True)
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="direct", extras={}, target_formats=["owl"])

    cache_dir = tmp_path / "cache"
    ontology_dir = tmp_path / "ontologies"
    cas_dir = tmp_path / "cas"
    overrides = {"CACHE_DIR": cache_dir, "LOCAL_ONTOLOGY_DIR": ontology_dir}
    for attr, value in overrides.items():
        monkeypatch.setattr(storage_mod, attr, value, raising=False)
        monkeypatch.setattr(pipeline_mod, attr, value, raising=False)
        monkeypatch.setattr(core, attr, value, raising=False)
    monkeypatch.setattr(core, "ONTOLOGY_DIR", ontology_dir, raising=False)

    class _StubStorage:
        def finalize_version(
            self,
            ontology_id: str,
            version: str,
            base_dir: Path,
            *,
            artifact_path: Optional[Path] = None,
            artifact_sha256: Optional[str] = None,
        ) -> None:
            pass

        def ensure_local_version(self, ontology_id: str, version: str) -> Path:
            path = ontology_dir / ontology_id / version
            path.mkdir(parents=True, exist_ok=True)
            return path

        def mirror_cas_artifact(self, algorithm: str, checksum: str, source: Path) -> Path:
            cas_path = (
                cas_dir / f"by-{algorithm.lower()}" / checksum[:2] / f"{checksum}{source.suffix}"
            )
            cas_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, cas_path)
            return cas_path

    stub_storage = _StubStorage()
    monkeypatch.setattr(storage_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(pipeline_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(core, "STORAGE", stub_storage, raising=False)

    def _fake_download_stream(**kwargs):
        destination: Path = kwargs["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return DownloadResult(
            path=destination,
            status="fresh",
            sha256=digest,
            etag=None,
            last_modified=None,
            content_type="application/rdf+xml",
            content_length=len(payload),
        )

    monkeypatch.setattr(pipeline_mod, "download_stream", _fake_download_stream, raising=False)
    monkeypatch.setattr(core, "download_stream", _fake_download_stream, raising=False)
    monkeypatch.setattr(pipeline_mod, "run_validators", lambda requests, logger: {}, raising=False)
    monkeypatch.setattr(core, "run_validators", lambda requests, logger: {}, raising=False)
    monkeypatch.setattr(
        pipeline_mod, "validate_url_security", lambda url, config=None: url, raising=False
    )
    monkeypatch.setattr(core, "validate_url_security", lambda url, config=None: url, raising=False)

    try:
        results = core.fetch_all([spec], config=config, force=True)
    finally:
        if original_direct is not None:
            RESOLVERS["direct"] = original_direct
        else:
            RESOLVERS.pop("direct", None)

    manifest_data = json.loads(results[0].manifest_path.read_text())
    cas_artifacts = [
        path for path in manifest_data["artifacts"] if path.endswith(".owl") and "by-" in path
    ]
    assert cas_artifacts, "expected CAS artifact entry"

    index_path = ontology_dir / spec.id / "index.json"
    index_payload = json.loads(index_path.read_text())
    assert index_payload[0]["cas_path"].endswith(f"{digest}.owl")


def test_resolve_expected_checksum_fetches_url(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_digest = "a" * 64
    spec = core.FetchSpec(
        id="hp",
        resolver="obo",
        extras={
            "checksum_url": {"url": "https://example.org/checksums.txt", "algorithm": "sha256"}
        },
        target_formats=["owl"],
    )
    plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint="hp.owl",
        version="2024-03-01",
        license="CC0-1.0",
        media_type="application/rdf+xml",
    )

    class DummyResponse:
        text = f"{expected_digest} hp.owl"

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(requests, "get", lambda url, timeout: DummyResponse())
    monkeypatch.setattr(
        pipeline_mod,
        "validate_url_security",
        lambda url, config=None: url,
        raising=False,
    )

    checksum = pipeline_mod._resolve_expected_checksum(
        spec=spec,
        plan=plan,
        download_config=DownloadConfiguration(),
        logger=logging.getLogger("checksum-test"),
    )

    assert checksum is not None
    assert checksum.algorithm == "sha256"
    assert checksum.value == expected_digest
    assert checksum.to_known_hash() == f"sha256:{expected_digest}"


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
    assert manifest_a["streaming_content_sha256"] == "hash-stream"
    assert manifest_a["streaming_content_sha256"] == manifest_b["streaming_content_sha256"]
    assert manifest_a["streaming_prefix_sha256"] == "prefix-in-memory"


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
    assert manifest_streaming["streaming_prefix_sha256"] == "prefix-streaming"
    assert manifest_streaming["streaming_content_sha256"] == "hash-stream"
