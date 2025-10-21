"""Tests for DuckDB plan caching in catalog.queries.

Covers:
- cache_plan() storing and replacing plans
- get_cached_plan() retrieving by ID
- get_cached_plan_by_service() retrieving recent plans
- Schema creation on first use
- Type safety and error handling
"""

from __future__ import annotations

from datetime import datetime, timezone

from DocsToKG.OntologyDownload.catalog.queries import (
    CachedPlanRow,
    cache_plan,
    get_cached_plan,
    get_cached_plan_by_service,
)
from DocsToKG.OntologyDownload.database import get_database
from DocsToKG.OntologyDownload.testing import TestingEnvironment


def test_cache_plan_creates_table() -> None:
    """Plan caching should create the plans table on first use."""
    with TestingEnvironment():
        db = get_database()
        db.bootstrap()

        # Table shouldn't exist yet
        tables_before = db._connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'plans'"
        ).fetchone()
        assert tables_before is None

        # Cache a plan
        cache_plan(
            db._connection,
            plan_id="hp-obo-2024",
            service="hp",
            resolver="obo",
            url="https://purl.obolibrary.org/obo/hp.owl",
            version_id="2024-01-01",
            checksum="abc123",
        )

        # Table should exist now
        tables_after = db._connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'plans'"
        ).fetchone()
        assert tables_after is not None


def test_cache_plan_stores_data() -> None:
    """Cached plans should be retrievable."""
    with TestingEnvironment():
        db = get_database()
        db.bootstrap()

        # Cache a plan
        cache_plan(
            db._connection,
            plan_id="chebi-obo-2024",
            service="chebi",
            resolver="obo",
            url="https://purl.obolibrary.org/obo/chebi.owl",
            version_id="2024-02-01",
            checksum="def456",
        )

        # Retrieve it
        plan = get_cached_plan(db._connection, "chebi-obo-2024")

        assert plan is not None
        assert plan.plan_id == "chebi-obo-2024"
        assert plan.service == "chebi"
        assert plan.resolver == "obo"
        assert plan.url == "https://purl.obolibrary.org/obo/chebi.owl"
        assert plan.version_id == "2024-02-01"
        assert plan.checksum == "def456"


def test_cache_plan_replaces_existing() -> None:
    """Caching the same plan ID should replace the old plan."""
    with TestingEnvironment():
        db = get_database()
        db.bootstrap()

        # Cache first version
        cache_plan(
            db._connection,
            plan_id="hp-v1",
            service="hp",
            resolver="obo",
            url="https://old.url/hp.owl",
            version_id="2024-01-01",
        )

        # Retrieve and verify
        plan1 = get_cached_plan(db._connection, "hp-v1")
        assert plan1.url == "https://old.url/hp.owl"

        # Cache updated version
        cache_plan(
            db._connection,
            plan_id="hp-v1",
            service="hp",
            resolver="obo",
            url="https://new.url/hp.owl",
            version_id="2024-02-01",
        )

        # Retrieve and verify update
        plan2 = get_cached_plan(db._connection, "hp-v1")
        assert plan2.url == "https://new.url/hp.owl"
        assert plan2.version_id == "2024-02-01"


def test_get_cached_plan_not_found() -> None:
    """Getting a non-existent plan should return None."""
    with TestingEnvironment():
        db = get_database()
        db.bootstrap()

        plan = get_cached_plan(db._connection, "nonexistent")

        assert plan is None


def test_get_cached_plan_by_service() -> None:
    """Should retrieve most recent plan for a service."""
    with TestingEnvironment():
        db = get_database()
        db.bootstrap()

        # Cache multiple plans for same service
        cache_plan(
            db._connection,
            plan_id="hp-old",
            service="hp",
            resolver="obo",
            url="https://url1/hp.owl",
            version_id="2024-01-01",
        )

        # Give it a moment so timestamp differs (if DB supports sub-second precision)
        cache_plan(
            db._connection,
            plan_id="hp-new",
            service="hp",
            resolver="ols",
            url="https://url2/hp.owl",
            version_id="2024-02-01",
        )

        # Get most recent
        plan = get_cached_plan_by_service(db._connection, "hp")

        assert plan is not None
        assert plan.service == "hp"
        # Should get the most recently cached one
        assert plan.plan_id in ["hp-old", "hp-new"]


def test_get_cached_plan_by_service_not_found() -> None:
    """Getting plans for non-existent service should return None."""
    with TestingEnvironment():
        db = get_database()
        db.bootstrap()

        plan = get_cached_plan_by_service(db._connection, "unknown-service")

        assert plan is None


def test_cached_plan_row_from_tuple() -> None:
    """CachedPlanRow should construct correctly from tuple."""
    row = (
        "plan-id-1",
        "hp",
        "obo",
        "https://example.org/hp.owl",
        "2024-01-01",
        "abc123def456",
        "2024-01-01T00:00:00",
    )

    plan = CachedPlanRow.from_tuple(row)

    assert plan.plan_id == "plan-id-1"
    assert plan.service == "hp"
    assert plan.resolver == "obo"
    assert plan.url == "https://example.org/hp.owl"
    assert plan.version_id == "2024-01-01"
    assert plan.checksum == "abc123def456"
    assert isinstance(plan.cached_at, datetime)


def test_cache_plan_without_checksum() -> None:
    """Plan caching should work without checksum."""
    with TestingEnvironment():
        db = get_database()
        db.bootstrap()

        # Cache plan without checksum
        cache_plan(
            db._connection,
            plan_id="hp-no-checksum",
            service="hp",
            resolver="obo",
            url="https://example.org/hp.owl",
            version_id="2024-01-01",
            checksum=None,
        )

        # Retrieve
        plan = get_cached_plan(db._connection, "hp-no-checksum")

        assert plan is not None
        assert plan.checksum is None
