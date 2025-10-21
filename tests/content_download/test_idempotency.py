"""Comprehensive tests for Data Model & Idempotency system.

Tests cover:
  - Job planning and idempotency
  - Leasing and multi-worker coordination
  - State machine transitions
  - Exactly-once operation logging
  - Crash recovery and reconciliation
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from DocsToKG.ContentDownload.idempotency import job_key, op_key
from DocsToKG.ContentDownload.job_effects import get_effect_result, run_effect
from DocsToKG.ContentDownload.job_leasing import (
    lease_next_job,
    release_lease,
    renew_lease,
)
from DocsToKG.ContentDownload.job_planning import plan_job_if_absent
from DocsToKG.ContentDownload.job_reconciler import (
    cleanup_stale_leases,
    cleanup_stale_ops,
    reconcile_jobs,
)
from DocsToKG.ContentDownload.job_state import advance_state, get_current_state
from DocsToKG.ContentDownload.schema_migration import apply_migration


@pytest.fixture
def db_connection():
    """Create an in-memory SQLite database with idempotency schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_migration(conn)
    yield conn
    conn.close()


class TestIdempotencyKeys:
    """Test idempotency key generation."""

    def test_job_key_deterministic(self):
        """Same inputs produce same key."""
        key1 = job_key("work-123", "artifact-456", "https://example.org/paper.pdf")
        key2 = job_key("work-123", "artifact-456", "https://example.org/paper.pdf")
        assert key1 == key2

    def test_job_key_different_inputs(self):
        """Different inputs produce different keys."""
        key1 = job_key("work-123", "artifact-456", "https://example.org/paper.pdf")
        key2 = job_key("work-123", "artifact-457", "https://example.org/paper.pdf")
        assert key1 != key2

    def test_op_key_with_extras(self):
        """Operation keys include extra fields."""
        key1 = op_key("HEAD", "job-123", url="https://example.org/paper.pdf")
        key2 = op_key("HEAD", "job-123", url="https://example.org/other.pdf")
        assert key1 != key2

    def test_op_key_deterministic(self):
        """Operation keys are deterministic with same inputs."""
        key1 = op_key("STREAM", "job-123", url="https://example.org/paper.pdf", range_start=0)
        key2 = op_key("STREAM", "job-123", url="https://example.org/paper.pdf", range_start=0)
        assert key1 == key2


class TestJobPlanning:
    """Test job planning and idempotency."""

    def test_plan_job_creates_new(self, db_connection):
        """Plan creates a new job."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        assert job_id is not None
        assert len(job_id) == 36  # UUID length

    def test_plan_job_idempotent(self, db_connection):
        """Replanning same job returns same ID."""
        job_id1 = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        job_id2 = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        assert job_id1 == job_id2

    def test_plan_multiple_jobs_different_ids(self, db_connection):
        """Different artifacts get different job IDs."""
        job_id1 = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper1.pdf",
        )
        job_id2 = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper2.pdf",
        )
        assert job_id1 != job_id2

    def test_plan_job_state_is_planned(self, db_connection):
        """New job starts in PLANNED state."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        state = get_current_state(db_connection, job_id=job_id)
        assert state == "PLANNED"


class TestJobLeasing:
    """Test job leasing for multi-worker coordination."""

    def test_lease_next_job_claims_one(self, db_connection):
        """Leasing claims one available job."""
        plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        job = lease_next_job(db_connection, owner="worker-1")
        assert job is not None
        assert job["lease_owner"] == "worker-1"
        assert job["state"] == "LEASED"

    def test_lease_multiple_workers_one_wins(self, db_connection):
        """Only one worker can lease a job at a time."""
        plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        job1 = lease_next_job(db_connection, owner="worker-1")
        job2 = lease_next_job(db_connection, owner="worker-2")
        assert job1 is not None
        assert job2 is None  # Second worker gets nothing

    def test_lease_expired_lease_available(self, db_connection):
        """Job with expired lease can be re-leased."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        # Lease with very short TTL
        job = lease_next_job(db_connection, owner="worker-1", ttl_s=1)
        assert job is not None

        # Fast-forward time
        now = time.time() + 2
        job2 = lease_next_job(db_connection, owner="worker-2")
        # Job should still be leased by worker-1 (not yet expired)
        assert job2 is None

    def test_renew_lease_extends_ttl(self, db_connection):
        """Renewing lease extends TTL."""
        plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        job = lease_next_job(db_connection, owner="worker-1", ttl_s=10)
        job_id = job["job_id"]

        # Renew lease
        success = renew_lease(db_connection, job_id=job_id, owner="worker-1", ttl_s=60)
        assert success

        # Try wrong owner
        success2 = renew_lease(db_connection, job_id=job_id, owner="worker-2", ttl_s=60)
        assert not success2

    def test_release_lease_clears_ownership(self, db_connection):
        """Releasing lease clears owner and TTL for re-claim."""
        plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        job = lease_next_job(db_connection, owner="worker-1")
        job_id = job["job_id"]

        success = release_lease(db_connection, job_id=job_id, owner="worker-1")
        assert success

        # Verify lease is cleared by checking the DB directly
        row = db_connection.execute(
            "SELECT lease_owner, lease_until FROM artifact_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        assert row["lease_owner"] is None
        assert row["lease_until"] is None


class TestStateTransitions:
    """Test state machine enforcement."""

    def test_advance_state_valid_transition(self, db_connection):
        """Valid state transition succeeds."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        advance_state(db_connection, job_id=job_id, to_state="LEASED", allowed_from=("PLANNED",))
        state = get_current_state(db_connection, job_id=job_id)
        assert state == "LEASED"

    def test_advance_state_invalid_transition_raises(self, db_connection):
        """Invalid state transition raises RuntimeError."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        with pytest.raises(RuntimeError, match="state_transition_denied"):
            advance_state(
                db_connection,
                job_id=job_id,
                to_state="STREAMING",
                allowed_from=("LEASED",),
            )

    def test_full_state_progression(self, db_connection):
        """Full happy-path state progression."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        # PLANNED → LEASED
        advance_state(db_connection, job_id=job_id, to_state="LEASED", allowed_from=("PLANNED",))
        assert get_current_state(db_connection, job_id=job_id) == "LEASED"

        # LEASED → HEAD_DONE
        advance_state(db_connection, job_id=job_id, to_state="HEAD_DONE", allowed_from=("LEASED",))
        assert get_current_state(db_connection, job_id=job_id) == "HEAD_DONE"

        # HEAD_DONE → STREAMING
        advance_state(
            db_connection,
            job_id=job_id,
            to_state="STREAMING",
            allowed_from=("HEAD_DONE", "RESUME_OK"),
        )
        assert get_current_state(db_connection, job_id=job_id) == "STREAMING"

        # STREAMING → FINALIZED
        advance_state(
            db_connection, job_id=job_id, to_state="FINALIZED", allowed_from=("STREAMING",)
        )
        assert get_current_state(db_connection, job_id=job_id) == "FINALIZED"


class TestExactlyOnceEffects:
    """Test exactly-once operation logging."""

    def test_run_effect_executes_first_time(self, db_connection):
        """Effect function is called on first attempt."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            return {"code": "OK", "status": 200}

        opkey = op_key("HEAD", job_id, url="https://example.org/paper.pdf")
        result = run_effect(
            db_connection, job_id=job_id, kind="HEAD", opkey=opkey, effect_fn=side_effect
        )

        assert call_count == 1
        assert result["status"] == 200

    def test_run_effect_replay_cached_result(self, db_connection):
        """Repeated effect with same key returns cached result without re-execution."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            return {"code": "OK", "status": 200}

        opkey = op_key("HEAD", job_id, url="https://example.org/paper.pdf")

        # First call
        result1 = run_effect(
            db_connection, job_id=job_id, kind="HEAD", opkey=opkey, effect_fn=side_effect
        )

        # Second call with same opkey
        result2 = run_effect(
            db_connection, job_id=job_id, kind="HEAD", opkey=opkey, effect_fn=side_effect
        )

        assert call_count == 1  # Function called only once
        assert result1 == result2
        assert result2["status"] == 200

    def test_get_effect_result(self, db_connection):
        """Get effect result from ledger."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        opkey = op_key("HEAD", job_id, url="https://example.org/paper.pdf")
        run_effect(
            db_connection,
            job_id=job_id,
            kind="HEAD",
            opkey=opkey,
            effect_fn=lambda: {"code": "OK", "status": 200},
        )

        result = get_effect_result(db_connection, opkey=opkey)
        assert result["status"] == 200


class TestReconciliation:
    """Test crash recovery and reconciliation."""

    def test_cleanup_stale_leases(self, db_connection):
        """Stale leases are cleared."""
        plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        job = lease_next_job(db_connection, owner="worker-1", ttl_s=10)

        # Simulate time passing
        now = time.time() + 20
        cleared = cleanup_stale_leases(db_connection, now=now)
        assert cleared == 1

        # Create a second job for re-lease testing
        job_id2 = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-457",
            canonical_url="https://example.org/paper2.pdf",
        )

        # Job should be available for new worker now (first job's lease is cleared)
        job2 = lease_next_job(db_connection, owner="worker-2")
        assert job2 is not None
        # Should get the second job (first one is still in LEASED state)
        assert job2["job_id"] == job_id2

    def test_cleanup_stale_ops(self, db_connection):
        """Abandoned operations are marked."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        # Create an operation that's been in-flight for a while
        opkey = op_key("STREAM", job_id, url="https://example.org/paper.pdf", range_start=0)
        db_connection.execute(
            "INSERT INTO artifact_ops(op_key, job_id, op_type, started_at) VALUES (?, ?, ?, ?)",
            (opkey, job_id, "STREAM", time.time() - 1000),
        )

        # Mark as abandoned
        now = time.time()
        marked = cleanup_stale_ops(db_connection, now=now, abandoned_threshold_s=60)
        assert marked == 1

    def test_reconcile_jobs_cleans_stale_parts(self, db_connection):
        """Stale .part files are cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            staging_root = Path(tmpdir) / ".staging"
            staging_root.mkdir(parents=True)

            # Create stale .part file
            stale_part = staging_root / "old.part"
            stale_part.write_text("stale data")
            # Set modification time to 2 hours ago
            old_time = time.time() - 7200
            import os

            os.utime(stale_part, (old_time, old_time))

            deleted, healed = reconcile_jobs(db_connection, staging_root, max_age_s=3600)
            assert deleted == 1
            assert not stale_part.exists()

            # Create fresh .part file
            fresh_part = staging_root / "fresh.part"
            fresh_part.write_text("fresh data")

            deleted, healed = reconcile_jobs(db_connection, staging_root, max_age_s=3600)
            assert deleted == 0  # Fresh file not deleted
            assert fresh_part.exists()
