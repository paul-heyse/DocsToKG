# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_plugins_registration",
#   "purpose": "Resolver/validator plugin registration regression tests.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Resolver/validator plugin registration regression tests.

Exercises entry-point loading, manual registration/unregistration, registry
caching, and metadata exposure so that dynamic extensions behave predictably
across threads and repeated CLI invocations."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable

from DocsToKG.OntologyDownload import plugins as plugins_mod
from DocsToKG.OntologyDownload.testing import temporary_resolver


class _DummyResolver:
    NAME = "registration-test"

    def plan(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - protocol compliance
        raise NotImplementedError("stub resolver")


@contextmanager
def _register_and_restore(kind: str, name: str, factory: Callable[[], object]):
    """Register a plugin and ensure the previous value is restored."""

    registry = (
        plugins_mod.get_resolver_registry()
        if kind == "resolver"
        else plugins_mod.get_validator_registry()
    )
    previous = registry.get(name)
    try:
        candidate = factory()
        if kind == "resolver":
            plugins_mod.register_resolver(name, candidate, overwrite=True)
        else:
            plugins_mod.register_validator(name, candidate, overwrite=True)
        yield candidate
    finally:
        try:
            if kind == "resolver":
                plugins_mod.unregister_resolver(name)
            else:
                plugins_mod.unregister_validator(name)
        except KeyError:
            pass
        if previous is not None:
            if kind == "resolver":
                plugins_mod.register_resolver(name, previous, overwrite=True)
            else:
                plugins_mod.register_validator(name, previous, overwrite=True)


def test_register_resolver_updates_metadata():
    """Registering a resolver should update the registry and metadata tables."""

    name = "test-harness-resolver"

    with _register_and_restore("resolver", name, lambda: _DummyResolver()) as resolver:
        registry = plugins_mod.get_resolver_registry()
        assert registry[name] is resolver

        meta = plugins_mod.get_registered_plugin_meta("resolver")
        assert name in meta
        assert meta[name]["qualified"].endswith("test_plugins_registration._DummyResolver")
        assert meta[name]["version"] == "local"

        replacement = _DummyResolver()
        plugins_mod.register_resolver(name, replacement, overwrite=True)
        assert plugins_mod.get_resolver_registry()[name] is replacement

    meta_after = plugins_mod.get_registered_plugin_meta("resolver")
    assert name not in meta_after


def test_register_validator_restores_previous_callable():
    """Validators should be replaceable while preserving previous instances."""

    name = "test-harness-validator"

    def validator(*args: Any, **kwargs: Any) -> str:
        return "ok"

    with _register_and_restore("validator", name, lambda: validator):
        registry = plugins_mod.get_validator_registry()
        assert registry[name] is validator

        meta = plugins_mod.get_registered_plugin_meta("validator")
        assert meta[name]["qualified"].endswith("test_plugins_registration.validator")
        assert meta[name]["version"] == "local"

    assert name not in plugins_mod.get_validator_registry()


def test_concurrent_registration_is_thread_safe():
    """Concurrent register/unregister cycles should not raise race conditions."""

    errors: list[BaseException] = []

    def worker(index: int) -> None:
        name = f"thread-resolver-{index}"
        resolver = type(
            f"ThreadResolver{index}", (), {"NAME": name, "plan": lambda self, *a, **k: None}
        )()
        try:
            plugins_mod.register_resolver(name, resolver, overwrite=True)
            assert plugins_mod.get_resolver_registry()[name] is resolver
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)
        finally:
            try:
                plugins_mod.unregister_resolver(name)
            except KeyError:
                pass

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"Encountered registration errors: {errors}"


def test_temporary_resolver_restores_entry(ontology_env):
    """temporary_resolver should replace and then restore the registry entry."""

    original = plugins_mod.get_resolver_registry().get("obo")
    resolver = ontology_env.static_resolver(
        name="obo",
        fixture_url=ontology_env.register_fixture(
            "obo-temp.owl", b"content", media_type="application/rdf+xml"
        ),
        filename="obo-temp.owl",
        media_type="application/rdf+xml",
        service="obo",
    )

    with temporary_resolver("obo", resolver):
        assert plugins_mod.get_resolver_registry()["obo"] is resolver

    restored = plugins_mod.get_resolver_registry().get("obo")
    if original is None:
        assert restored is None
    else:
        assert restored is original
