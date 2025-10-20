"""Tests for Phase 4: Plan Caching & Comparison in DuckDB database."""

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.database import Database, DatabaseConfiguration, PlanDiffRow, PlanRow


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database file path."""
    db_path = tmp_path / "test_phase4.duckdb"
    return db_path


@pytest.fixture
def db(temp_db_path: Path) -> Database:
    """Create and bootstrap a test database."""
    config = DatabaseConfiguration(db_path=temp_db_path, readonly=False)
    database = Database(config)
    database.bootstrap()
    yield database
    database.close()


class TestPhase4PlanCaching:
    """Test plan caching functionality."""

    def test_upsert_plan_creates_new_plan(self, db: Database) -> None:
        """Verify upsert_plan creates a new plan entry."""
        plan_id = sha256(b"hp_2025-01-01").hexdigest()
        plan_json = {
            "id": "hp",
            "resolver": "obo",
            "url": "https://example.org/hp.owl",
            "version": "2025-01-01",
            "license": "CC-BY-4.0",
            "service": "obo",
            "media_type": "application/rdf+xml",
        }

        db.upsert_plan(
            plan_id=plan_id,
            ontology_id="hp",
            resolver="obo",
            plan_json=plan_json,
            is_current=True,
        )

        plan = db.get_current_plan("hp")
        assert plan is not None
        assert plan.plan_id == plan_id
        assert plan.ontology_id == "hp"
        assert plan.resolver == "obo"
        assert plan.is_current is True

    def test_upsert_plan_idempotence(self, db: Database) -> None:
        """Verify upsert_plan is idempotent."""
        plan_id = sha256(b"hp_2025-01-01").hexdigest()
        plan_json = {
            "id": "hp",
            "resolver": "obo",
            "url": "https://example.org/hp.owl",
            "version": "2025-01-01",
        }

        # Insert twice
        db.upsert_plan(plan_id, "hp", "obo", plan_json, is_current=True)
        db.upsert_plan(plan_id, "hp", "obo", plan_json, is_current=True)

        # Should still be one entry
        plans = db.list_plans("hp")
        assert len(plans) == 1

    def test_upsert_plan_marks_previous_as_non_current(self, db: Database) -> None:
        """Verify that marking a new plan as current unmarks the previous one."""
        plan_id_1 = sha256(b"hp_v1").hexdigest()
        plan_id_2 = sha256(b"hp_v2").hexdigest()

        plan_json_1 = {
            "id": "hp",
            "resolver": "obo",
            "url": "https://v1.example.org/hp.owl",
            "version": "2025-01-01",
        }
        plan_json_2 = {
            "id": "hp",
            "resolver": "obo",
            "url": "https://v2.example.org/hp.owl",
            "version": "2025-01-02",
        }

        db.upsert_plan(plan_id_1, "hp", "obo", plan_json_1, is_current=True)
        db.upsert_plan(plan_id_2, "hp", "obo", plan_json_2, is_current=True)

        current = db.get_current_plan("hp")
        assert current.plan_id == plan_id_2

        # Verify first plan is no longer current
        plans = db.list_plans("hp")
        assert len(plans) == 2
        assert plans[0].is_current is True  # v2
        assert plans[1].is_current is False  # v1

    def test_get_current_plan_returns_none_for_missing(self, db: Database) -> None:
        """Verify get_current_plan returns None for non-existent ontology."""
        plan = db.get_current_plan("nonexistent")
        assert plan is None

    def test_list_plans_empty(self, db: Database) -> None:
        """Verify list_plans returns empty list when no plans exist."""
        plans = db.list_plans("hp")
        assert plans == []

    def test_list_plans_with_multiple(self, db: Database) -> None:
        """Verify list_plans returns all plans for an ontology."""
        plan_ids = [sha256(f"hp_{i}".encode()).hexdigest() for i in range(3)]

        for i, pid in enumerate(plan_ids):
            plan_json = {
                "id": "hp",
                "resolver": "obo",
                "url": f"https://example.org/hp_v{i}.owl",
                "version": f"2025-0{i + 1}-01",
            }
            db.upsert_plan(pid, "hp", "obo", plan_json, is_current=(i == 2))

        plans = db.list_plans("hp")
        assert len(plans) == 3
        assert plans[0].is_current is True  # Most recent, marked current

    def test_list_plans_respects_limit(self, db: Database) -> None:
        """Verify list_plans respects the limit parameter."""
        for i in range(10):
            plan_id = sha256(f"hp_{i}".encode()).hexdigest()
            plan_json = {"id": "hp", "resolver": "obo", "url": f"https://v{i}.org"}
            db.upsert_plan(plan_id, "hp", "obo", plan_json)

        plans = db.list_plans("hp", limit=5)
        assert len(plans) == 5

    def test_plan_json_roundtrip(self, db: Database) -> None:
        """Verify plan_json survives storage and retrieval intact."""
        original_json = {
            "id": "hp",
            "resolver": "obo",
            "url": "https://example.org/hp.owl",
            "version": "2025-01-01",
            "license": "CC-BY-4.0",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        plan_id = sha256(json.dumps(original_json, sort_keys=True).encode()).hexdigest()
        db.upsert_plan(plan_id, "hp", "obo", original_json, is_current=True)

        retrieved = db.get_current_plan("hp")
        assert retrieved.plan_json == original_json


class TestPhase4PlanDiff:
    """Test plan diff storage and retrieval."""

    def test_insert_plan_diff_stores_metadata(self, db: Database) -> None:
        """Verify insert_plan_diff stores diff comparison results."""
        diff_result = {
            "added": [{"id": "new_ontology", "resolver": "ols"}],
            "removed": [{"id": "old_ontology", "resolver": "obo"}],
            "modified": [{"id": "hp", "changes": {"url": {"before": "old", "after": "new"}}}],
        }

        diff_id = sha256(b"diff_1").hexdigest()
        plan_id_1 = sha256(b"plan_1").hexdigest()
        plan_id_2 = sha256(b"plan_2").hexdigest()

        db.insert_plan_diff(
            diff_id=diff_id,
            older_plan_id=plan_id_1,
            newer_plan_id=plan_id_2,
            ontology_id="hp",
            diff_result=diff_result,
        )

        diffs = db.get_plan_diff_history("hp", limit=10)
        assert len(diffs) == 1
        diff = diffs[0]
        assert diff.diff_id == diff_id
        assert diff.added_count == 1
        assert diff.removed_count == 1
        assert diff.modified_count == 1

    def test_insert_plan_diff_preserves_details(self, db: Database) -> None:
        """Verify full diff JSON is preserved through storage."""
        diff_result = {
            "added": [{"id": "new"}],
            "removed": [],
            "modified": [
                {
                    "id": "hp",
                    "changes": {
                        "url": {"before": "old_url", "after": "new_url"},
                        "version": {"before": "2024-01-01", "after": "2024-02-01"},
                    },
                }
            ],
        }

        diff_id = sha256(b"complex_diff").hexdigest()
        db.insert_plan_diff(
            diff_id=diff_id,
            older_plan_id="plan_1",
            newer_plan_id="plan_2",
            ontology_id="hp",
            diff_result=diff_result,
        )

        diffs = db.get_plan_diff_history("hp")
        retrieved_diff = diffs[0].diff_json
        assert retrieved_diff == diff_result

    def test_get_plan_diff_history_orders_by_recency(self, db: Database) -> None:
        """Verify plan diffs are returned in reverse chronological order."""
        for i in range(3):
            diff_id = sha256(f"diff_{i}".encode()).hexdigest()
            db.insert_plan_diff(
                diff_id=diff_id,
                older_plan_id=f"plan_{i}",
                newer_plan_id=f"plan_{i + 1}",
                ontology_id="hp",
                diff_result={"added": [], "removed": [], "modified": []},
            )

        diffs = db.get_plan_diff_history("hp", limit=10)
        assert len(diffs) == 3
        # Should be in descending order by comparison_at
        assert diffs[0].comparison_at >= diffs[1].comparison_at >= diffs[2].comparison_at

    def test_get_plan_diff_history_empty_for_missing_ontology(self, db: Database) -> None:
        """Verify get_plan_diff_history returns empty list for non-existent ontology."""
        diffs = db.get_plan_diff_history("nonexistent")
        assert diffs == []

    def test_get_plan_diff_history_respects_limit(self, db: Database) -> None:
        """Verify get_plan_diff_history respects the limit parameter."""
        for i in range(10):
            diff_id = sha256(f"diff_{i}".encode()).hexdigest()
            db.insert_plan_diff(
                diff_id=diff_id,
                older_plan_id=f"plan_{i}",
                newer_plan_id=f"plan_{i + 1}",
                ontology_id="hp",
                diff_result={"added": [], "removed": [], "modified": []},
            )

        diffs = db.get_plan_diff_history("hp", limit=3)
        assert len(diffs) == 3


class TestPhase4Integration:
    """Integration tests for Phase 4 plan caching workflow."""

    def test_plan_caching_workflow(self, db: Database) -> None:
        """Test a realistic plan caching workflow."""
        # Step 1: Cache a plan
        plan_1_id = sha256(b"hp_run_1").hexdigest()
        plan_1_json = {
            "id": "hp",
            "resolver": "obo",
            "url": "https://v1.org/hp.owl",
            "version": "2025-01-01",
            "license": "CC-BY-4.0",
        }
        db.upsert_plan(plan_1_id, "hp", "obo", plan_1_json, is_current=True)

        # Step 2: Verify current plan
        current = db.get_current_plan("hp")
        assert current is not None
        assert current.plan_json == plan_1_json

        # Step 3: Cache a new plan
        plan_2_id = sha256(b"hp_run_2").hexdigest()
        plan_2_json = {
            "id": "hp",
            "resolver": "obo",
            "url": "https://v2.org/hp.owl",
            "version": "2025-01-02",
            "license": "CC-BY-4.0",
        }
        db.upsert_plan(plan_2_id, "hp", "obo", plan_2_json, is_current=True)

        # Step 4: Store diff between them
        diff_result = {
            "added": [],
            "removed": [],
            "modified": [
                {
                    "id": "hp",
                    "changes": {
                        "url": {"before": plan_1_json["url"], "after": plan_2_json["url"]},
                        "version": {
                            "before": plan_1_json["version"],
                            "after": plan_2_json["version"],
                        },
                    },
                }
            ],
        }
        diff_id = sha256(f"{plan_1_id}:{plan_2_id}".encode()).hexdigest()
        db.insert_plan_diff(diff_id, plan_1_id, plan_2_id, "hp", diff_result)

        # Step 5: Verify history
        plans = db.list_plans("hp")
        assert len(plans) == 2

        diffs = db.get_plan_diff_history("hp")
        assert len(diffs) == 1
        assert diffs[0].added_count == 0
        assert diffs[0].modified_count == 1

    def test_multiple_ontologies_independence(self, db: Database) -> None:
        """Verify plans and diffs for different ontologies don't interfere."""
        # Cache plans for two ontologies
        for ont_id in ["hp", "chebi"]:
            plan_id = sha256(f"{ont_id}_plan".encode()).hexdigest()
            plan_json = {"id": ont_id, "resolver": "obo", "url": f"https://{ont_id}.org"}
            db.upsert_plan(plan_id, ont_id, "obo", plan_json, is_current=True)

        # Store diff for only one
        diff_id = sha256(b"hp_diff").hexdigest()
        db.insert_plan_diff(
            diff_id, "plan_1", "plan_2", "hp", {"added": [], "removed": [], "modified": []}
        )

        # Verify independence
        hp_plans = db.list_plans("hp")
        chebi_plans = db.list_plans("chebi")
        assert len(hp_plans) == 1
        assert len(chebi_plans) == 1

        hp_diffs = db.get_plan_diff_history("hp")
        chebi_diffs = db.get_plan_diff_history("chebi")
        assert len(hp_diffs) == 1
        assert len(chebi_diffs) == 0
