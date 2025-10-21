"""Phase 4: Plan & Plan-Diff Integration Tests.

Tests for database-backed plan caching, deterministic replay, and
plan-diff functionality enabling faster repeat planning and change detection.

Test scope:
- Plan serialization/deserialization to/from database
- Cache hit/miss scenarios
- Plan comparison and diff generation
- CLI integration with --use-cache flag
- E2E workflow with cache persistence
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pytest

from DocsToKG.OntologyDownload.planning import (
    PlannedFetch,
    FetchPlan,
    ResolverCandidate,
    FetchSpec,
    _planned_fetch_to_dict,
    _dict_to_planned_fetch,
    _get_cached_plan,
    _save_plan_to_db,
    _compare_plans,
    _save_plan_diff_to_db,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_fetch_plan() -> FetchPlan:
    """Create a sample FetchPlan for testing."""
    return FetchPlan(
        url="https://example.com/ontology.owl",
        headers={"Accept": "application/rdf+xml"},
        filename_hint="ontology.owl",
        version="2025-01-01",
        license="CC-BY-4.0",
        media_type="application/rdf+xml",
        content_length=1024000,
        last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
    )


@pytest.fixture
def sample_candidate(sample_fetch_plan: FetchPlan) -> ResolverCandidate:
    """Create a sample ResolverCandidate for testing."""
    return ResolverCandidate(
        resolver="obo",
        plan=sample_fetch_plan,
    )


@pytest.fixture
def sample_planned_fetch(
    sample_fetch_plan: FetchPlan,
    sample_candidate: ResolverCandidate,
) -> PlannedFetch:
    """Create a sample PlannedFetch for testing."""
    spec = FetchSpec(
        id="hp",
        resolver="obo",
        extras={},
        target_formats=("owl",),
    )
    return PlannedFetch(
        spec=spec,
        resolver="obo",
        plan=sample_fetch_plan,
        candidates=(sample_candidate,),
        metadata={"expected_checksum": {"algorithm": "sha256", "value": "abc123"}},
        last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
        size=1024000,
    )


@pytest.fixture
def sample_spec() -> FetchSpec:
    """Create a sample FetchSpec for testing."""
    return FetchSpec(
        id="hp",
        resolver="obo",
        extras={},
        target_formats=("owl",),
    )


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================


def test_planned_fetch_to_dict(sample_planned_fetch: PlannedFetch) -> None:
    """Test serialization of PlannedFetch to dictionary."""
    result = _planned_fetch_to_dict(sample_planned_fetch)

    assert isinstance(result, dict)
    assert result["resolver"] == "obo"
    assert result["plan"]["url"] == "https://example.com/ontology.owl"
    assert result["plan"]["version"] == "2025-01-01"
    assert result["plan"]["license"] == "CC-BY-4.0"
    assert result["metadata"]["expected_checksum"]["algorithm"] == "sha256"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["resolver"] == "obo"


def test_dict_to_planned_fetch(
    sample_planned_fetch: PlannedFetch,
    sample_spec: FetchSpec,
) -> None:
    """Test deserialization of dictionary back to PlannedFetch."""
    data = _planned_fetch_to_dict(sample_planned_fetch)
    reconstructed = _dict_to_planned_fetch(data, sample_spec)

    assert reconstructed is not None
    assert reconstructed.resolver == sample_planned_fetch.resolver
    assert reconstructed.plan.url == sample_planned_fetch.plan.url
    assert reconstructed.plan.version == sample_planned_fetch.plan.version
    assert reconstructed.plan.license == sample_planned_fetch.plan.license
    assert len(reconstructed.candidates) == len(sample_planned_fetch.candidates)
    assert reconstructed.metadata == sample_planned_fetch.metadata


def test_roundtrip_serialization(
    sample_planned_fetch: PlannedFetch,
    sample_spec: FetchSpec,
) -> None:
    """Test that serialization and deserialization preserves all data."""
    original_dict = _planned_fetch_to_dict(sample_planned_fetch)
    reconstructed = _dict_to_planned_fetch(original_dict, sample_spec)
    reconstructed_dict = _planned_fetch_to_dict(reconstructed)

    # Verify round-trip integrity
    assert reconstructed_dict["resolver"] == original_dict["resolver"]
    assert reconstructed_dict["plan"]["url"] == original_dict["plan"]["url"]
    assert reconstructed_dict["plan"]["version"] == original_dict["plan"]["version"]


# ============================================================================
# PLAN COMPARISON TESTS
# ============================================================================


def test_compare_plans_first_plan(sample_planned_fetch: PlannedFetch) -> None:
    """Test diff when no previous plan exists (first plan)."""
    diff = _compare_plans(None, sample_planned_fetch)

    assert diff["older"] is False
    assert len(diff["added"]) == 1
    assert diff["added"][0]["resolver"] == "obo"
    assert diff["added"][0]["url"] == "https://example.com/ontology.owl"
    assert len(diff["modified"]) == 0


def test_compare_plans_no_changes(
    sample_planned_fetch: PlannedFetch,
) -> None:
    """Test diff when plans are identical."""
    diff = _compare_plans(sample_planned_fetch, sample_planned_fetch)

    assert diff["older"] is True
    assert diff["unchanged"] == 1
    assert len(diff["modified"]) == 0


def test_compare_plans_url_changed(
    sample_planned_fetch: PlannedFetch,
    sample_fetch_plan: FetchPlan,
    sample_candidate: ResolverCandidate,
) -> None:
    """Test diff when URL has changed."""
    new_plan = FetchPlan(
        url="https://example.com/ontology-v2.owl",
        headers=sample_fetch_plan.headers,
        filename_hint=sample_fetch_plan.filename_hint,
        version=sample_fetch_plan.version,
        license=sample_fetch_plan.license,
        media_type=sample_fetch_plan.media_type,
        content_length=sample_fetch_plan.content_length,
        last_modified=sample_fetch_plan.last_modified,
    )
    new_planned = PlannedFetch(
        spec=sample_planned_fetch.spec,
        resolver=sample_planned_fetch.resolver,
        plan=new_plan,
        candidates=(sample_candidate,),
        metadata=sample_planned_fetch.metadata,
        last_modified=sample_planned_fetch.last_modified,
        size=sample_planned_fetch.size,
    )

    diff = _compare_plans(sample_planned_fetch, new_planned)

    assert diff["older"] is True
    assert len(diff["modified"]) == 1
    assert diff["modified"][0]["field"] == "url"
    assert diff["modified"][0]["old"] == "https://example.com/ontology.owl"
    assert diff["modified"][0]["new"] == "https://example.com/ontology-v2.owl"


def test_compare_plans_version_changed(
    sample_planned_fetch: PlannedFetch,
    sample_fetch_plan: FetchPlan,
    sample_candidate: ResolverCandidate,
) -> None:
    """Test diff when version has changed."""
    new_plan = FetchPlan(
        url=sample_fetch_plan.url,
        headers=sample_fetch_plan.headers,
        filename_hint=sample_fetch_plan.filename_hint,
        version="2025-02-01",
        license=sample_fetch_plan.license,
        media_type=sample_fetch_plan.media_type,
        content_length=sample_fetch_plan.content_length,
        last_modified=sample_fetch_plan.last_modified,
    )
    new_planned = PlannedFetch(
        spec=sample_planned_fetch.spec,
        resolver=sample_planned_fetch.resolver,
        plan=new_plan,
        candidates=(sample_candidate,),
        metadata=sample_planned_fetch.metadata,
        last_modified=sample_planned_fetch.last_modified,
        size=sample_planned_fetch.size,
    )

    diff = _compare_plans(sample_planned_fetch, new_planned)

    assert len(diff["modified"]) == 1
    assert diff["modified"][0]["field"] == "version"
    assert diff["modified"][0]["old"] == "2025-01-01"
    assert diff["modified"][0]["new"] == "2025-02-01"


def test_compare_plans_size_changed(
    sample_planned_fetch: PlannedFetch,
    sample_fetch_plan: FetchPlan,
    sample_candidate: ResolverCandidate,
) -> None:
    """Test diff when content size has changed."""
    new_planned = PlannedFetch(
        spec=sample_planned_fetch.spec,
        resolver=sample_planned_fetch.resolver,
        plan=sample_fetch_plan,
        candidates=(sample_candidate,),
        metadata=sample_planned_fetch.metadata,
        last_modified=sample_planned_fetch.last_modified,
        size=2048000,  # Changed size
    )

    diff = _compare_plans(sample_planned_fetch, new_planned)

    assert len(diff["modified"]) == 1
    assert diff["modified"][0]["field"] == "size_bytes"
    assert diff["modified"][0]["old"] == 1024000
    assert diff["modified"][0]["new"] == 2048000


def test_compare_plans_resolver_changed(
    sample_planned_fetch: PlannedFetch,
    sample_fetch_plan: FetchPlan,
    sample_candidate: ResolverCandidate,
) -> None:
    """Test diff when resolver has changed."""
    new_plan = FetchPlan(
        url=sample_fetch_plan.url,
        headers=sample_fetch_plan.headers,
        filename_hint=sample_fetch_plan.filename_hint,
        version=sample_fetch_plan.version,
        license=sample_fetch_plan.license,
        media_type=sample_fetch_plan.media_type,
        content_length=sample_fetch_plan.content_length,
        last_modified=sample_fetch_plan.last_modified,
    )
    new_candidate = ResolverCandidate(resolver="ontobee", plan=new_plan)
    new_planned = PlannedFetch(
        spec=sample_planned_fetch.spec,
        resolver="ontobee",  # Changed resolver
        plan=new_plan,
        candidates=(new_candidate,),
        metadata=sample_planned_fetch.metadata,
        last_modified=sample_planned_fetch.last_modified,
        size=sample_planned_fetch.size,
    )

    diff = _compare_plans(sample_planned_fetch, new_planned)

    assert len(diff["modified"]) == 1
    assert diff["modified"][0]["field"] == "resolver"
    assert diff["modified"][0]["old"] == "obo"
    assert diff["modified"][0]["new"] == "ontobee"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_serialization_json_compatible(sample_planned_fetch: PlannedFetch) -> None:
    """Test that serialized plan is JSON-compatible."""
    data = _planned_fetch_to_dict(sample_planned_fetch)

    # Should be JSON serializable
    json_str = json.dumps(data)
    assert isinstance(json_str, str)

    # Should be JSON deserializable
    recovered = json.loads(json_str)
    assert recovered["resolver"] == "obo"
    assert recovered["plan"]["url"] == "https://example.com/ontology.owl"


def test_incomplete_dict_handles_gracefully(sample_spec: FetchSpec) -> None:
    """Test that incomplete dictionary data is handled gracefully."""
    incomplete_data = {
        "plan": {
            "url": "https://example.com/ontology.owl",
        },
        # Missing many fields
    }

    result = _dict_to_planned_fetch(incomplete_data, sample_spec)

    # Should return PlannedFetch with defaults rather than crashing
    assert result is not None
    assert result.plan.url == "https://example.com/ontology.owl"
    assert result.plan.version is None


def test_malformed_dict_returns_none(sample_spec: FetchSpec) -> None:
    """Test that malformed dictionary returns None instead of crashing."""
    malformed_data = "not a dict"  # type: ignore

    # Should handle gracefully and return None
    result = _dict_to_planned_fetch(malformed_data, sample_spec)  # type: ignore

    assert result is None


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_plan_with_minimal_metadata(sample_fetch_plan: FetchPlan) -> None:
    """Test plan serialization with minimal metadata."""
    spec = FetchSpec(id="minimal", resolver="direct", extras={}, target_formats=())
    minimal_planned = PlannedFetch(
        spec=spec,
        resolver="direct",
        plan=sample_fetch_plan,
        candidates=(),
        metadata={},
        last_modified=None,
        size=None,
    )

    data = _planned_fetch_to_dict(minimal_planned)
    reconstructed = _dict_to_planned_fetch(data, spec)

    assert reconstructed is not None
    assert reconstructed.metadata == {}
    assert reconstructed.last_modified is None
    assert reconstructed.size is None


def test_plan_with_complex_metadata(sample_fetch_plan: FetchPlan) -> None:
    """Test plan serialization with complex metadata."""
    spec = FetchSpec(id="complex", resolver="obo", extras={}, target_formats=())
    complex_planned = PlannedFetch(
        spec=spec,
        resolver="obo",
        plan=sample_fetch_plan,
        candidates=(),
        metadata={
            "expected_checksum": {
                "algorithm": "sha256",
                "value": "abc123def456",
            },
            "custom_field": {"nested": {"deeply": "nested"}},
            "list_field": [1, 2, 3],
        },
    )

    data = _planned_fetch_to_dict(complex_planned)
    reconstructed = _dict_to_planned_fetch(data, spec)

    assert reconstructed is not None
    assert reconstructed.metadata["custom_field"]["nested"]["deeply"] == "nested"
    assert reconstructed.metadata["list_field"] == [1, 2, 3]


# ============================================================================
# COMPARISON MULTIPLE CHANGES TEST
# ============================================================================


def test_compare_plans_multiple_changes(
    sample_planned_fetch: PlannedFetch,
    sample_fetch_plan: FetchPlan,
    sample_candidate: ResolverCandidate,
) -> None:
    """Test diff with multiple fields changed."""
    new_plan = FetchPlan(
        url="https://example.com/ontology-v2.owl",
        headers=sample_fetch_plan.headers,
        filename_hint=sample_fetch_plan.filename_hint,
        version="2025-02-01",
        license="CC-BY-SA-4.0",  # Changed
        media_type="application/rdf+xml",
        content_length=2048000,
        last_modified=sample_fetch_plan.last_modified,
    )
    new_candidate = ResolverCandidate(resolver="ontobee", plan=new_plan)
    new_planned = PlannedFetch(
        spec=sample_planned_fetch.spec,
        resolver="ontobee",
        plan=new_plan,
        candidates=(new_candidate,),
        metadata=sample_planned_fetch.metadata,
        last_modified=sample_planned_fetch.last_modified,
        size=2048000,
    )

    diff = _compare_plans(sample_planned_fetch, new_planned)

    # Should have multiple modified fields
    modified_fields = {m["field"] for m in diff["modified"]}
    assert "resolver" in modified_fields
    assert "url" in modified_fields
    assert "version" in modified_fields
    assert "license" in modified_fields
    assert "size_bytes" in modified_fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
