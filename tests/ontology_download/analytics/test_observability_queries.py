"""Tests for the observability query registry helpers."""

from __future__ import annotations

import pytest

from DocsToKG.OntologyDownload.observability import queries


ALL_QUERY_NAMES = [
    "extract_summary",
    "extract_zip_bombs",
    "net_cache_hit_ratio",
    "net_latency_distribution",
    "policy_heatmap",
    "policy_rejections",
    "ratelimit_cooldowns",
    "ratelimit_pressure",
]

CATEGORY_EXPECTATIONS = {
    "extraction": ["extract_summary", "extract_zip_bombs"],
    "policy": ["policy_heatmap", "policy_rejections"],
    "ratelimit": ["ratelimit_cooldowns", "ratelimit_pressure"],
    "slo": ["net_cache_hit_ratio", "net_latency_distribution"],
}


def test_list_queries_returns_all_sorted_names() -> None:
    """All registered queries should be listed in alphabetical order."""

    assert queries.list_queries() == ALL_QUERY_NAMES


@pytest.mark.parametrize(
    ("category", "expected"),
    sorted(CATEGORY_EXPECTATIONS.items()),
)
def test_list_queries_filters_by_category(category: str, expected: list[str]) -> None:
    """Category filtering should return sorted names for the category."""

    assert queries.list_queries(category) == expected


def test_list_queries_unknown_category_is_empty() -> None:
    """An unknown category should return an empty list."""

    assert queries.list_queries("missing") == []


def test_get_query_returns_sql() -> None:
    """Named queries should resolve to their SQL definitions."""

    sql = queries.get_query("net_latency_distribution")
    assert "PERCENTILE_CONT" in sql


def test_get_query_unknown_raises_key_error() -> None:
    """Unknown query names should raise ``KeyError``."""

    with pytest.raises(KeyError):
        queries.get_query("does_not_exist")


def test_query_summary_matches_registry() -> None:
    """Summary output should contain descriptions keyed by query name."""

    summary = queries.query_summary()
    assert sorted(summary) == ALL_QUERY_NAMES
    for name, details in summary.items():
        definition = queries._QUERY_DEFINITIONS[name]
        assert details.endswith(f"(category: {definition.category})")


@pytest.mark.parametrize(
    ("category", "expected_names"),
    sorted(CATEGORY_EXPECTATIONS.items()),
)
def test_query_summary_filters_by_category(
    category: str, expected_names: list[str]
) -> None:
    """Summaries should filter by category consistently."""

    summary = queries.query_summary(category)
    assert sorted(summary) == expected_names
    assert all(f"category: {category}" in text for text in summary.values())


@pytest.mark.parametrize(
    ("helper", "metric", "expected_name"),
    [
        (queries.get_slo_query, "network", "net_latency_distribution"),
        (queries.get_slo_query, "cache", "net_cache_hit_ratio"),
        (queries.get_rate_limit_query, "pressure", "ratelimit_pressure"),
        (queries.get_rate_limit_query, "cooldowns", "ratelimit_cooldowns"),
        (queries.get_safety_query, "rejections", "policy_rejections"),
        (queries.get_safety_query, "heatmap", "policy_heatmap"),
        (queries.get_extraction_query, "bombs", "extract_zip_bombs"),
        (queries.get_extraction_query, "stats", "extract_summary"),
    ],
)
def test_legacy_helpers_dispatch_to_registered_queries(
    helper, metric: str, expected_name: str
) -> None:
    """Legacy helper wrappers should delegate to the main registry."""

    assert helper(metric) == queries.get_query(expected_name)


@pytest.mark.parametrize(
    ("helper", "metric"),
    [
        (queries.get_slo_query, "unknown"),
        (queries.get_rate_limit_query, "unknown"),
        (queries.get_safety_query, "unknown"),
        (queries.get_extraction_query, "unknown"),
    ],
)
def test_legacy_helpers_return_none_for_unknown_metrics(helper, metric: str) -> None:
    """Helpers should fall back to ``None`` for unsupported metrics."""

    assert helper(metric) is None


def test_registry_definitions_are_unique_and_categories_sorted() -> None:
    """Definitions should remain unique with well-categorised groupings."""

    definitions = list(queries._QUERY_DEFINITIONS.values())
    names = [definition.name for definition in definitions]
    assert len(names) == len(set(names)), "duplicate query names detected"

    categories = sorted({definition.category for definition in definitions})
    assert categories == sorted(CATEGORY_EXPECTATIONS), "unexpected category labels"

    for category, expected_names in CATEGORY_EXPECTATIONS.items():
        registered = sorted(
            definition.name
            for definition in definitions
            if definition.category == category
        )
        assert registered == expected_names
