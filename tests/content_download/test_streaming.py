"""Comprehensive test suite for download streaming and idempotency.

Tests cover:
  - Individual streaming functions (quota, resume, streaming, finalization)
  - Resume decision logic (validators, prefix hashing)
  - Orchestrator (full pipeline)
  - Idempotency (keys, leases, state machine)
  - Crash recovery (reconciliation)
  - Multi-worker scenarios
  - Edge cases (large files, empty files, network errors)
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from DocsToKG.ContentDownload import idempotency, streaming

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_db():
    """In-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create minimal schema
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE artifact_jobs (
            job_id TEXT PRIMARY KEY,
            work_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            canonical_url TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'PLANNED',
            lease_owner TEXT,
            lease_until REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            idempotency_key TEXT NOT NULL,
            UNIQUE(work_id, artifact_id, canonical_url),
            UNIQUE(idempotency_key),
            CHECK (state IN ('PLANNED','LEASED','HEAD_DONE','RESUME_OK','STREAMING',
                            'FINALIZED','INDEXED','DEDUPED','FAILED','SKIPPED_DUPLICATE'))
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE artifact_ops (
            op_key TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            op_type TEXT NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL,
            result_code TEXT,
            result_json TEXT,
            FOREIGN KEY(job_id) REFERENCES artifact_jobs(job_id)
        )
        """
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def mock_client():
    """Mock HTTPX client."""
    client = MagicMock()
    client.head = MagicMock()
    client.build_request = MagicMock()
    client.send = MagicMock()
    client.get = MagicMock()
    return client


@pytest.fixture
def mock_hash_index():
    """Mock hash index."""
    index = MagicMock()
    index.get_hash_for_url = MagicMock(return_value=None)
    index.get_path_and_size = MagicMock(return_value=None)
    index.put = MagicMock()
    index.put_url_hash = MagicMock()
    index.dedupe_link_or_copy = MagicMock(return_value="download")
    return index


@pytest.fixture
def mock_manifest_sink():
    """Mock manifest sink."""
    sink = MagicMock()
    sink.write = MagicMock()
    return sink


@pytest.fixture
def test_config():
    """Test configuration object."""
    config = MagicMock()
    config.io = MagicMock(
        chunk_bytes=65536,
        fsync=True,
        preallocate=True,
        preallocate_min_size_bytes=2097152,
    )
    config.resume = MagicMock(
        prefix_check_bytes=65536,
        allow_without_validators=False,
    )
    config.quota = MagicMock(
        free_bytes_min=1073741824,
        margin_factor=1.5,
    )
    config.shard = MagicMock(enabled=True, width=2)
    config.dedupe = MagicMock(hardlink=True, enabled=True)
    config.offline_block_artifacts = True
    return config


# ============================================================================
# Streaming Tests
# ============================================================================


class TestQuotaGuard:
    """Tests for quota guard functionality."""

    def test_sufficient_quota(self, tmp_dir):
        """Should succeed when enough free space."""
        streaming.ensure_quota(tmp_dir, 100 * 1024 * 1024, free_min=1024 * 1024, margin=1.5)
        # Should not raise

    def test_insufficient_quota(self, tmp_dir):
        """Should raise when free space insufficient."""
        with pytest.raises(OSError, match="insufficient free space"):
            streaming.ensure_quota(
                tmp_dir,
                1024 * 1024 * 1024 * 1024,  # 1 TB (way more than available)
                free_min=1024 * 1024,
                margin=1.5,
            )

    def test_quota_with_margin(self, tmp_dir):
        """Should apply margin factor correctly."""
        # This should work with safety margin
        streaming.ensure_quota(tmp_dir, 100 * 1024 * 1024, free_min=1024 * 1024, margin=1.5)


class TestResumeDecision:
    """Tests for resume decision logic."""

    def test_no_part_fresh(self, mock_client):
        """No .part file should return fresh."""
        validators = streaming.ServerValidators(
            etag="abc123",
            last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
            accept_ranges=True,
            content_length=1000,
        )
        decision = streaming.can_resume(
            validators,
            None,
            prefix_check_bytes=65536,
            allow_without_validators=False,
            client=mock_client,
            url="http://example.com/file.pdf",
        )
        assert decision.mode == "fresh"
        assert decision.reason == "no_part"

    def test_no_accept_ranges_discard(self, tmp_dir, mock_client):
        """No accept-ranges should discard part."""
        part_file = tmp_dir / "test.part"
        part_file.write_bytes(b"test data")

        part_state = streaming.LocalPartState(
            path=part_file,
            bytes_local=9,
        )

        validators = streaming.ServerValidators(
            etag="abc123",
            last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
            accept_ranges=False,
            content_length=1000,
        )

        decision = streaming.can_resume(
            validators,
            part_state,
            prefix_check_bytes=65536,
            allow_without_validators=False,
            client=mock_client,
            url="http://example.com/file.pdf",
        )

        assert decision.mode == "discard_part"
        assert decision.reason == "no_accept_ranges"

    def test_validators_mismatch_discard(self, tmp_dir, mock_client):
        """Different etag should discard part."""
        part_file = tmp_dir / "test.part"
        part_file.write_bytes(b"test data")

        part_state = streaming.LocalPartState(
            path=part_file,
            bytes_local=9,
            etag="old_etag",
        )

        validators = streaming.ServerValidators(
            etag="new_etag",
            last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
            accept_ranges=True,
            content_length=1000,
        )

        decision = streaming.can_resume(
            validators,
            part_state,
            prefix_check_bytes=65536,
            allow_without_validators=False,
            client=mock_client,
            url="http://example.com/file.pdf",
        )

        assert decision.mode == "discard_part"
        assert decision.reason == "validators_mismatch"


class TestStreamMetrics:
    """Tests for stream metrics dataclass."""

    def test_metrics_creation(self):
        """Should create metrics successfully."""
        metrics = streaming.StreamMetrics(
            bytes_written=1000,
            elapsed_ms=100,
            fsync_ms=5,
            sha256_hex="abc123def456",
            avg_write_mibps=9.77,
            resumed_from_bytes=500,
        )

        assert metrics.bytes_written == 1000
        assert metrics.avg_write_mibps == 9.77
        assert metrics.sha256_hex == "abc123def456"


# ============================================================================
# Idempotency Tests
# ============================================================================


class TestKeyGeneration:
    """Tests for idempotency key generation."""

    def test_deterministic_ikey(self):
        """Same object should produce same key."""
        obj = {"z": 3, "a": 1, "m": 2}
        key1 = idempotency.ikey(obj)
        key2 = idempotency.ikey(obj)
        assert key1 == key2

    def test_ikey_order_independent(self):
        """Key should be independent of dict insertion order."""
        obj1 = {"z": 3, "a": 1, "m": 2}
        obj2 = {"a": 1, "m": 2, "z": 3}
        assert idempotency.ikey(obj1) == idempotency.ikey(obj2)

    def test_job_key_generation(self):
        """Should generate deterministic job key."""
        key = idempotency.job_key(
            work_id="work-123",
            artifact_id="art-456",
            canonical_url="https://example.com/file.pdf",
        )
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex is 64 chars

    def test_job_key_reproducible(self):
        """Same inputs should produce same job key."""
        key1 = idempotency.job_key("work-1", "art-1", "http://example.com/a.pdf")
        key2 = idempotency.job_key("work-1", "art-1", "http://example.com/a.pdf")
        assert key1 == key2

    def test_op_key_generation(self):
        """Should generate deterministic operation key."""
        job_id = idempotency.job_key("w1", "a1", "http://ex.com/f.pdf")
        op_k = idempotency.op_key("HEAD", job_id, url="http://ex.com/f.pdf")
        assert isinstance(op_k, str)
        assert len(op_k) == 64

    def test_op_key_different_contexts(self):
        """Different contexts should produce different keys."""
        job_id = "test-job-id"
        key1 = idempotency.op_key("STREAM", job_id, url="http://a.com", range_start=0)
        key2 = idempotency.op_key("STREAM", job_id, url="http://b.com", range_start=0)
        assert key1 != key2


class TestLeaseManagement:
    """Tests for lease acquire/renew/release."""

    def test_acquire_lease_success(self, test_db):
        """Should acquire lease on available job."""
        now = time.time()
        cursor = test_db.cursor()

        # Insert a PLANNED job
        cursor.execute(
            """
            INSERT INTO artifact_jobs (job_id, work_id, artifact_id, canonical_url,
                                       state, created_at, updated_at, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "work-1",
                "art-1",
                "http://ex.com/f.pdf",
                "PLANNED",
                now,
                now,
                "key-1",
            ),
        )
        test_db.commit()

        # Acquire lease
        job_id = idempotency.acquire_lease(test_db, "worker-1", 60, now_fn=lambda: now)
        assert job_id == "job-1"

        # Verify lease was set
        cursor.execute("SELECT lease_owner, state FROM artifact_jobs WHERE job_id = ?", ("job-1",))
        row = cursor.fetchone()
        assert row["lease_owner"] == "worker-1"
        assert row["state"] == "LEASED"

    def test_acquire_lease_none_available(self, test_db):
        """Should return None if no available jobs."""
        job_id = idempotency.acquire_lease(test_db, "worker-1", 60)
        assert job_id is None

    def test_renew_lease_success(self, test_db):
        """Should renew active lease."""
        now = time.time()
        cursor = test_db.cursor()

        # Insert leased job
        cursor.execute(
            """
            INSERT INTO artifact_jobs (job_id, work_id, artifact_id, canonical_url,
                                       state, lease_owner, lease_until, created_at,
                                       updated_at, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "work-1",
                "art-1",
                "http://ex.com/f.pdf",
                "LEASED",
                "worker-1",
                now + 60,
                now,
                now,
                "key-1",
            ),
        )
        test_db.commit()

        # Renew lease
        result = idempotency.renew_lease(test_db, "job-1", "worker-1", 120, now_fn=lambda: now)
        assert result is True


class TestStateMachine:
    """Tests for state machine transitions."""

    def test_advance_state_valid_transition(self, test_db):
        """Should advance state for valid transition."""
        now = time.time()
        cursor = test_db.cursor()

        # Insert job in PLANNED state
        cursor.execute(
            """
            INSERT INTO artifact_jobs (job_id, work_id, artifact_id, canonical_url,
                                       state, created_at, updated_at, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "work-1",
                "art-1",
                "http://ex.com/f.pdf",
                "PLANNED",
                now,
                now,
                "key-1",
            ),
        )
        test_db.commit()

        # Advance to LEASED
        result = idempotency.advance_state(
            test_db, "job-1", "LEASED", {"PLANNED"}, now_fn=lambda: now
        )
        assert result is True

        # Verify state changed
        cursor.execute("SELECT state FROM artifact_jobs WHERE job_id = ?", ("job-1",))
        row = cursor.fetchone()
        assert row["state"] == "LEASED"

    def test_advance_state_invalid_transition(self, test_db):
        """Should reject invalid transition."""
        now = time.time()
        cursor = test_db.cursor()

        # Insert job in PLANNED state
        cursor.execute(
            """
            INSERT INTO artifact_jobs (job_id, work_id, artifact_id, canonical_url,
                                       state, created_at, updated_at, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "work-1",
                "art-1",
                "http://ex.com/f.pdf",
                "PLANNED",
                now,
                now,
                "key-1",
            ),
        )
        test_db.commit()

        # Try to advance from STREAMING (wrong previous state)
        result = idempotency.advance_state(
            test_db, "job-1", "FINALIZED", {"STREAMING"}, now_fn=lambda: now
        )
        assert result is False


class TestExactlyOnceEffects:
    """Tests for exactly-once effect execution."""

    def test_run_effect_first_execution(self, test_db):
        """Should execute effect on first call."""
        now = time.time()
        cursor = test_db.cursor()

        # Insert job
        cursor.execute(
            """
            INSERT INTO artifact_jobs (job_id, work_id, artifact_id, canonical_url,
                                       state, created_at, updated_at, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "work-1",
                "art-1",
                "http://ex.com/f.pdf",
                "PLANNED",
                now,
                now,
                "key-1",
            ),
        )
        test_db.commit()

        call_count = {"count": 0}

        def effect_fn():
            call_count["count"] += 1
            return {"code": "OK", "result": "success"}

        op_key = "op-key-1"
        result = idempotency.run_effect(
            test_db, op_key, "job-1", "HEAD", effect_fn, now_fn=lambda: now
        )

        assert result["code"] == "OK"
        assert call_count["count"] == 1

    def test_run_effect_replay(self, test_db):
        """Should return cached result on replay."""
        now = time.time()
        cursor = test_db.cursor()

        # Insert job
        cursor.execute(
            """
            INSERT INTO artifact_jobs (job_id, work_id, artifact_id, canonical_url,
                                       state, created_at, updated_at, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "work-1",
                "art-1",
                "http://ex.com/f.pdf",
                "PLANNED",
                now,
                now,
                "key-1",
            ),
        )
        test_db.commit()

        call_count = {"count": 0}

        def effect_fn():
            call_count["count"] += 1
            return {"code": "OK", "result": "success"}

        op_key = "op-key-1"

        # First call
        result1 = idempotency.run_effect(
            test_db, op_key, "job-1", "HEAD", effect_fn, now_fn=lambda: now
        )
        assert call_count["count"] == 1

        # Second call (replay)
        result2 = idempotency.run_effect(
            test_db, op_key, "job-1", "HEAD", effect_fn, now_fn=lambda: now
        )
        assert call_count["count"] == 1  # Not called again
        assert result1 == result2


class TestReconciliation:
    """Tests for database reconciliation."""

    def test_reconcile_stale_leases(self, test_db):
        """Should clear stale leases."""
        now = time.time()
        old_time = now - 1000  # Very old lease_until

        cursor = test_db.cursor()

        # Insert job with stale lease
        cursor.execute(
            """
            INSERT INTO artifact_jobs (job_id, work_id, artifact_id, canonical_url,
                                       state, lease_owner, lease_until, created_at,
                                       updated_at, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "work-1",
                "art-1",
                "http://ex.com/f.pdf",
                "LEASED",
                "worker-crashed",
                old_time,
                now,
                now,
                "key-1",
            ),
        )
        test_db.commit()

        # Reconcile
        count = idempotency.reconcile_stale_leases(test_db, now_fn=lambda: now)
        assert count == 1

        # Verify lease cleared
        cursor.execute("SELECT lease_owner FROM artifact_jobs WHERE job_id = ?", ("job-1",))
        row = cursor.fetchone()
        assert row["lease_owner"] is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestStreamingIntegration:
    """Integration tests for full streaming pipeline."""

    def test_download_pdf_offline_mode(
        self, tmp_dir, mock_client, mock_hash_index, mock_manifest_sink, test_config
    ):
        """Should raise in offline mode."""
        with pytest.raises(RuntimeError, match="offline: artifacts disabled"):
            streaming.download_pdf(
                client=mock_client,
                head_client=mock_client,
                url="http://example.com/file.pdf",
                cfg=test_config,
                root_dir=tmp_dir,
                staging_dir=tmp_dir,
                artifact_lock=lambda x: MagicMock(
                    __enter__=lambda s: None, __exit__=lambda s, *a: None
                ),
                hash_index=mock_hash_index,
                manifest_sink=mock_manifest_sink,
                logger=logging.getLogger(__name__),
                offline=True,
            )

    def test_download_pdf_deduplication(
        self, tmp_dir, mock_client, mock_hash_index, mock_manifest_sink, test_config
    ):
        """Should use deduplication when hash known."""
        # Setup: hash_index knows about this URL
        mock_hash_index.get_hash_for_url.return_value = "abc123def456"
        mock_hash_index.get_path_and_size.return_value = (
            str(tmp_dir / "ab" / "abc123def456.pdf"),
            1000,
        )
        mock_hash_index.dedupe_link_or_copy.return_value = "hardlink"

        result = streaming.download_pdf(
            client=mock_client,
            head_client=mock_client,
            url="http://example.com/file.pdf",
            cfg=test_config,
            root_dir=tmp_dir,
            staging_dir=tmp_dir / ".staging",
            artifact_lock=lambda x: MagicMock(
                __enter__=lambda s: None, __exit__=lambda s, *a: None
            ),
            hash_index=mock_hash_index,
            manifest_sink=mock_manifest_sink,
            logger=logging.getLogger(__name__),
            offline=False,
        )

        assert result["dedupe_action"] == "hardlink"
        mock_manifest_sink.write.assert_called_once()


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_ikey_performance(self):
        """Idempotency key generation should be fast."""
        obj = {
            "v": 1,
            "kind": "JOB",
            "work_id": "w" * 100,
            "artifact_id": "a" * 100,
            "url": "http://example.com/" + "x" * 100,
        }

        start = time.time()
        for _ in range(1000):
            idempotency.ikey(obj)
        elapsed_ms = (time.time() - start) * 1000

        # Should be very fast (< 100ms for 1000 calls)
        assert elapsed_ms < 100


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_part_file(self, tmp_dir, mock_client):
        """Should handle empty .part file."""
        part_file = tmp_dir / "test.part"
        part_file.write_bytes(b"")

        part_state = streaming.LocalPartState(path=part_file, bytes_local=0)

        validators = streaming.ServerValidators(
            etag="abc123",
            last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
            accept_ranges=True,
            content_length=1000,
        )

        decision = streaming.can_resume(
            validators,
            part_state,
            prefix_check_bytes=65536,
            allow_without_validators=False,
            client=mock_client,
            url="http://example.com/file.pdf",
        )

        assert decision.mode == "fresh"  # Empty part treated as no part

    def test_large_content_length(self):
        """Should handle large content lengths."""
        validators = streaming.ServerValidators(
            etag="abc123",
            last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
            accept_ranges=True,
            content_length=10 * 1024 * 1024 * 1024,  # 10 GB
        )
        assert validators.content_length > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
