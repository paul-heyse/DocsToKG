"""Regression tests for the backwards-compatible resolver namespace."""

import importlib
import warnings

import DocsToKG.ContentDownload.resolvers as resolvers


def test_legacy_time_alias_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        module_time = resolvers.time  # type: ignore[attr-defined]

    assert module_time is importlib.import_module("time")
    assert any(w.category is DeprecationWarning for w in captured)


def test_reloading_resolvers_preserves_deprecation_behaviour():
    module = importlib.reload(resolvers)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        module.requests  # type: ignore[attr-defined]

    assert any(w.category is DeprecationWarning for w in captured)
