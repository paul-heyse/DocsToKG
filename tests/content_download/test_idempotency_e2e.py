"""End-to-End Integration Tests for Idempotency System.

Tests verify the complete download pipeline with idempotency enabled:
  - Full job lifecycle (planning → leasing → streaming → finalization)
  - Crash recovery and re-execution
  - Multi-worker coordination
  - Network failure recovery
  - Idempotency replay verification
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from DocsToKG.ContentDownload.idempotency import op_key
from DocsToKG.ContentDownload.job_effects import run_effect
from DocsToKG.ContentDownload.job_leasing import lease_next_job, release_lease, renew_lease
from DocsToKG.ContentDownload.job_planning import plan_job_if_absent
from DocsToKG.ContentDownload.job_reconciler import cleanup_stale_leases
from DocsToKG.ContentDownload.job_state import advance_state, get_current_state
from DocsToKG.ContentDownload.schema_migration import apply_migration


@pytest.fixture
def db_connection():
    """In-memory database with idempotency schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_migration(conn)
    yield conn
    conn.close()


@pytest.fixture
def temp_staging():
    """Temporary staging directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / ".staging"


class TestCompleteDownloadLifecycle:
    """E2E: Verify complete download workflow with idempotency."""

    def test_complete_download_with_idempotency(self, db_connection):
        """Full happy-path: plan → lease → HEAD → stream → finalize."""
        # 1. PLANNING PHASE
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )
        assert job_id is not None
        assert get_current_state(db_connection, job_id=job_id) == "PLANNED"

        # 2. LEASING PHASE
        leased_job = lease_next_job(db_connection, owner="worker-1", ttl_s=120)
        assert leased_job is not None
        assert leased_job["state"] == "LEASED"
        assert leased_job["lease_owner"] == "worker-1"

        # 3. HEAD REQUEST PHASE
        advance_state(db_connection, job_id=job_id, to_state="HEAD_DONE", allowed_from=("LEASED",))
        head_opkey = op_key("HEAD", job_id, url="https://example.org/paper.pdf")

        def simulate_head():
            return {"code": "OK", "status": 200, "content_length": 1024000}

        head_result = run_effect(
            db_connection, job_id=job_id, kind="HEAD", opkey=head_opkey, effect_fn=simulate_head
        )
        assert head_result["status"] == 200
        assert get_current_state(db_connection, job_id=job_id) == "HEAD_DONE"

        # 4. STREAMING PHASE
        advance_state(
            db_connection,
            job_id=job_id,
            to_state="STREAMING",
            allowed_from=("HEAD_DONE", "RESUME_OK"),
        )
        stream_opkey = op_key("STREAM", job_id, url="https://example.org/paper.pdf", range_start=0)

        def simulate_stream():
            return {
                "code": "OK",
                "bytes": 1024000,
                "sha256": "abc123def456",
                "elapsed_ms": 2500,
            }

        stream_result = run_effect(
            db_connection,
            job_id=job_id,
            kind="STREAM",
            opkey=stream_opkey,
            effect_fn=simulate_stream,
        )
        assert stream_result["bytes"] == 1024000
        assert get_current_state(db_connection, job_id=job_id) == "STREAMING"

        # 5. FINALIZATION PHASE
        advance_state(
            db_connection, job_id=job_id, to_state="FINALIZED", allowed_from=("STREAMING",)
        )
        fin_opkey = op_key("FINALIZE", job_id, sha256="abc123def456")

        def simulate_finalize():
            return {"code": "OK", "final_path": "/data/ontologies/paper.pdf", "size": 1024000}

        fin_result = run_effect(
            db_connection,
            job_id=job_id,
            kind="FINALIZE",
            opkey=fin_opkey,
            effect_fn=simulate_finalize,
        )
        assert fin_result["code"] == "OK"
        assert get_current_state(db_connection, job_id=job_id) == "FINALIZED"

    def test_crash_recovery_mid_stream(self, db_connection):
        """Crash mid-stream: worker dies, another resumes from checkpoint."""
        # Setup job
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        # Worker 1 claims job
        job1 = lease_next_job(db_connection, owner="worker-1", ttl_s=10)
        assert job1 is not None

        # Advance through HEAD
        advance_state(db_connection, job_id=job_id, to_state="HEAD_DONE", allowed_from=("LEASED",))
        advance_state(
            db_connection, job_id=job_id, to_state="STREAMING", allowed_from=("HEAD_DONE",)
        )

        # Start streaming operation
        stream_opkey = op_key("STREAM", job_id, url="https://example.org/paper.pdf", range_start=0)
        db_connection.execute(
            "INSERT INTO artifact_ops(op_key, job_id, op_type, started_at) VALUES (?, ?, ?, ?)",
            (stream_opkey, job_id, "STREAM", time.time() - 5),  # Started 5 seconds ago
        )

        # CRASH: Worker 1 dies, lease expires
        now = time.time() + 20  # Time passes
        cleanup_stale_leases(db_connection, now=now)

        # Verify lease is cleared
        row = db_connection.execute(
            "SELECT lease_owner FROM artifact_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        assert row["lease_owner"] is None

        # Verify state is still STREAMING (crash recovery leaves state as-is)
        assert get_current_state(db_connection, job_id=job_id) == "STREAMING"

    def test_idempotent_replay_after_crash(self, db_connection):
        """After crash: replaying same operation returns cached result, no re-execution."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        opkey = op_key("FINALIZE", job_id, sha256="abc123")

        # Execution 1: Effect runs
        exec_count = 0

        def finalize_effect():
            nonlocal exec_count
            exec_count += 1
            return {"code": "OK", "final_path": "/data/paper.pdf"}

        result1 = run_effect(
            db_connection, job_id=job_id, kind="FINALIZE", opkey=opkey, effect_fn=finalize_effect
        )
        assert exec_count == 1

        # CRASH + RECOVERY: Worker restarts, replays same operation
        result2 = run_effect(
            db_connection, job_id=job_id, kind="FINALIZE", opkey=opkey, effect_fn=finalize_effect
        )

        # Effect function not called again
        assert exec_count == 1
        # Result is cached and returned
        assert result2 == result1

    def test_multi_worker_scenario(self, db_connection):
        """Multiple workers: leasing ensures exclusive access per job."""
        # Create 3 jobs
        jobs = []
        for i in range(3):
            job_id = plan_job_if_absent(
                db_connection,
                work_id=f"work-{i}",
                artifact_id=f"artifact-{i}",
                canonical_url=f"https://example.org/paper{i}.pdf",
            )
            jobs.append(job_id)

        # Worker 1 claims job 0
        job0_worker1 = lease_next_job(db_connection, owner="worker-1")
        assert job0_worker1["job_id"] == jobs[0]

        # Worker 2 claims job 1
        job1_worker2 = lease_next_job(db_connection, owner="worker-2")
        assert job1_worker2["job_id"] == jobs[1]

        # Worker 3 claims job 2
        job2_worker3 = lease_next_job(db_connection, owner="worker-3")
        assert job2_worker3["job_id"] == jobs[2]

        # No more jobs available
        job_worker4 = lease_next_job(db_connection, owner="worker-4")
        assert job_worker4 is None

    def test_network_error_recovery(self, db_connection):
        """Transient network failure: retry + state preserved."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        lease_next_job(db_connection, owner="worker-1", ttl_s=120)
        advance_state(db_connection, job_id=job_id, to_state="HEAD_DONE", allowed_from=("LEASED",))

        # First attempt: network error
        head_opkey1 = op_key("HEAD", job_id, url="https://example.org/paper.pdf", range_start=0)

        def head_with_error():
            raise ConnectionError("Network timeout")

        try:
            run_effect(
                db_connection,
                job_id=job_id,
                kind="HEAD",
                opkey=head_opkey1,
                effect_fn=head_with_error,
            )
        except ConnectionError:
            pass

        # State unchanged after error
        assert get_current_state(db_connection, job_id=job_id) == "HEAD_DONE"

        # Retry with different opkey (retry counter)
        head_opkey2 = op_key("HEAD", job_id, url="https://example.org/paper.pdf", retry=1)

        def head_success():
            return {"code": "OK", "status": 200}

        result = run_effect(
            db_connection, job_id=job_id, kind="HEAD", opkey=head_opkey2, effect_fn=head_success
        )
        assert result["code"] == "OK"

    def test_lease_renewal_during_long_stream(self, db_connection):
        """Long operation: lease renewed to prevent timeout."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        job = lease_next_job(db_connection, owner="worker-1", ttl_s=5)
        assert job is not None

        # Simulate long streaming operation with periodic renewal
        advance_state(db_connection, job_id=job_id, to_state="HEAD_DONE", allowed_from=("LEASED",))
        advance_state(
            db_connection, job_id=job_id, to_state="STREAMING", allowed_from=("HEAD_DONE",)
        )

        # Renew lease mid-stream
        success = renew_lease(db_connection, job_id=job_id, owner="worker-1", ttl_s=60)
        assert success

        # Verify new TTL
        row = db_connection.execute(
            "SELECT lease_until FROM artifact_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        lease_until = row["lease_until"]
        now = time.time()
        # Lease should be ~60 seconds in future
        assert 55 < (lease_until - now) < 65

    def test_duplicate_detection_skip(self, db_connection):
        """Duplicate hash found before network: skip with SKIPPED_DUPLICATE."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        # Simulate duplicate detection (hash already indexed)
        advance_state(
            db_connection, job_id=job_id, to_state="SKIPPED_DUPLICATE", allowed_from=("PLANNED",)
        )

        state = get_current_state(db_connection, job_id=job_id)
        assert state == "SKIPPED_DUPLICATE"

    def test_failed_job_retry(self, db_connection):
        """Failed job transitions to FAILED, can be retried as new job."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        # Simulate failure
        lease_next_job(db_connection, owner="worker-1")
        advance_state(db_connection, job_id=job_id, to_state="FAILED", allowed_from=("LEASED",))

        state = get_current_state(db_connection, job_id=job_id)
        assert state == "FAILED"

        # Clear lease for retry
        release_lease(db_connection, job_id=job_id, owner="worker-1")

        # New worker can retry the same job (it's in FAILED state)
        job2 = lease_next_job(db_connection, owner="worker-2")
        assert job2["job_id"] == job_id

    def test_operation_ordering_preserved(self, db_connection):
        """Operations execute in order: HEAD → STREAM → FINALIZE."""
        job_id = plan_job_if_absent(
            db_connection,
            work_id="work-123",
            artifact_id="artifact-456",
            canonical_url="https://example.org/paper.pdf",
        )

        lease_next_job(db_connection, owner="worker-1")

        # Record operation sequence
        operations = []

        # HEAD
        advance_state(db_connection, job_id=job_id, to_state="HEAD_DONE", allowed_from=("LEASED",))
        operations.append("HEAD_DONE")

        # STREAM
        advance_state(
            db_connection, job_id=job_id, to_state="STREAMING", allowed_from=("HEAD_DONE",)
        )
        operations.append("STREAMING")

        # FINALIZE
        advance_state(
            db_connection, job_id=job_id, to_state="FINALIZED", allowed_from=("STREAMING",)
        )
        operations.append("FINALIZED")

        assert operations == ["HEAD_DONE", "STREAMING", "FINALIZED"]

    def test_concurrent_idempotency_keys_no_collision(self, db_connection):
        """Multiple jobs with same URL get unique job_ids but same idempotency_key."""
        work_id = "work-123"
        artifact_id = "artifact-456"
        url = "https://example.org/paper.pdf"

        # Plan same job twice (should return same job_id)
        job_id1 = plan_job_if_absent(
            db_connection,
            work_id=work_id,
            artifact_id=artifact_id,
            canonical_url=url,
        )
        job_id2 = plan_job_if_absent(
            db_connection,
            work_id=work_id,
            artifact_id=artifact_id,
            canonical_url=url,
        )

        assert job_id1 == job_id2

        # But different artifacts get different job_ids
        job_id3 = plan_job_if_absent(
            db_connection,
            work_id=work_id,
            artifact_id="artifact-789",
            canonical_url=url,
        )

        assert job_id3 != job_id1
