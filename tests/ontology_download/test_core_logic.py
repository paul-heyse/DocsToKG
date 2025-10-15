"""Core module behavior tests."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import core
from DocsToKG.OntologyDownload.config import ConfigError, DefaultsConfig, ResolvedConfig
from DocsToKG.OntologyDownload.resolvers import FetchPlan


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
