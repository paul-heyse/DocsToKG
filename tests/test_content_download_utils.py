"""
Content Download Utility Regression Tests

This module provides narrow regression coverage for lightweight helper
functions such as normalization, prefix stripping, and deduplication that
support resolver metadata preparation.

Key Scenarios:
- Normalizes DOI and PMC identifiers regardless of input formatting
- Strips case-insensitive prefixes and deduplicates ordered collections

Dependencies:
- DocsToKG.ContentDownload.utils: Utility functions under test

Usage:
    pytest tests/test_content_download_utils.py
"""

from typing import List

import pytest

try:
    import hypothesis
    from hypothesis import strategies as st  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis is required for these tests", allow_module_level=True)

from DocsToKG.ContentDownload.utils import dedupe, normalize_doi, normalize_pmcid, strip_prefix

given = hypothesis.given


def test_normalize_doi_with_https_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"


def test_normalize_doi_without_prefix() -> None:
    assert normalize_doi("10.1234/abc") == "10.1234/abc"


def test_normalize_doi_with_whitespace() -> None:
    assert normalize_doi("  10.1234/abc  ") == "10.1234/abc"


def test_normalize_doi_none() -> None:
    assert normalize_doi(None) is None


def test_normalize_pmcid_with_pmc_prefix() -> None:
    assert normalize_pmcid("PMC123456") == "PMC123456"


def test_normalize_pmcid_without_prefix_adds_prefix() -> None:
    assert normalize_pmcid("123456") == "PMC123456"


def test_normalize_pmcid_lowercase() -> None:
    assert normalize_pmcid("pmc123456") == "PMC123456"


def test_strip_prefix_case_insensitive() -> None:
    assert strip_prefix("ARXIV:2301.12345", "arxiv:") == "2301.12345"


def test_dedupe_preserves_order() -> None:
    assert dedupe(["b", "a", "b", "c"]) == ["b", "a", "c"]


def test_dedupe_filters_falsey_values() -> None:
    assert dedupe(["a", "", None, "a"]) == ["a"]


@given(st.lists(st.text()))
def test_dedupe_property(values: List[str]) -> None:
    expected = []
    seen = set()
    for item in values:
        if item and item not in seen:
            expected.append(item)
            seen.add(item)

    assert dedupe(values) == expected
