"""Tests for SLO metric computation.

Tests verify:
  - Job completion rate calculation
  - Time to complete percentile computation
  - Crash recovery metrics
  - Lease acquisition latency
  - SLO report generation
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from DocsToKG.ContentDownload import slo_compute
from DocsToKG.ContentDownload.schema_migration import apply_migration


@pytest.fixture
def slo_db():
    """Create test SLO database with schema."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_slo.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        apply_migration(conn)
        yield conn
        conn.close()


class TestJobCompletionRate:
    """Test job completion rate computation."""

    def test_completion_rate_no_jobs(self, slo_db):
        """Completion rate is 1.0 when no jobs exist."""
        rate = slo_compute.compute_job_completion_rate(slo_db)
        assert rate == 1.0

    def test_completion_rate_all_completed(self, slo_db):
        """Completion rate is 1.0 when all jobs are finalized."""
        now = time.time()
        slo_db.execute(
            """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                         state, created_at, updated_at, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("j1", "w1", "a1", "https://example.com", "FINALIZED", now, now, "key1"),
        )
        slo_db.commit()
        rate = slo_compute.compute_job_completion_rate(slo_db)
        assert rate == 1.0

    def test_completion_rate_partial_failure(self, slo_db):
        """Completion rate reflects mix of completed and failed jobs."""
        now = time.time()
        slo_db.execute(
            """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                         state, created_at, updated_at, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("j1", "w1", "a1", "https://example.com/1", "FINALIZED", now, now, "key1"),
        )
        slo_db.execute(
            """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                         state, created_at, updated_at, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("j2", "w2", "a2", "https://example.com/2", "FAILED", now, now, "key2"),
        )
        slo_db.commit()
        rate = slo_compute.compute_job_completion_rate(slo_db)
        assert rate == 0.5


class TestTimeToCompletePercentiles:
    """Test time-to-completion percentile computation."""

    def test_percentiles_no_jobs(self, slo_db):
        """Percentiles default to 0 when no jobs exist."""
        percentiles = slo_compute.compute_time_to_complete_percentiles(slo_db)
        assert percentiles == {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_percentiles_single_job(self, slo_db):
        """Percentiles are consistent with single job."""
        now = time.time()
        slo_db.execute(
            """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                         state, created_at, updated_at, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("j1", "w1", "a1", "https://example.com", "FINALIZED", now, now + 10, "key1"),
        )
        slo_db.commit()
        percentiles = slo_compute.compute_time_to_complete_percentiles(slo_db)
        # 10 seconds = 10000 ms
        assert percentiles["p50"] == 10000.0
        assert percentiles["p95"] == 10000.0
        assert percentiles["p99"] == 10000.0

    def test_percentiles_multiple_jobs(self, slo_db):
        """Percentiles are computed correctly across multiple jobs."""
        now = time.time()
        for i in range(100):
            duration = i + 1  # 1-100 seconds
            slo_db.execute(
                """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                             state, created_at, updated_at, idempotency_key)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    f"j{i}",
                    f"w{i}",
                    f"a{i}",
                    f"https://example.com/{i}",
                    "FINALIZED",
                    now,
                    now + duration,
                    f"key{i}",
                ),
            )
        slo_db.commit()
        percentiles = slo_compute.compute_time_to_complete_percentiles(slo_db)
        # p50 should be ~50 seconds, p95 should be ~95 seconds
        assert 40000 < percentiles["p50"] < 60000
        assert 80000 < percentiles["p95"] < 100000


class TestCrashRecoveryRate:
    """Test crash recovery rate computation."""

    def test_recovery_rate_no_jobs(self, slo_db):
        """Recovery rate is 1.0 when no jobs exist."""
        rate = slo_compute.compute_crash_recovery_success_rate(slo_db)
        assert rate == 1.0

    def test_recovery_rate_all_recovered(self, slo_db):
        """Recovery rate is 1.0 when all jobs finalized."""
        now = time.time()
        slo_db.execute(
            """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                         state, created_at, updated_at, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("j1", "w1", "a1", "https://example.com", "FINALIZED", now, now, "key1"),
        )
        slo_db.commit()
        rate = slo_compute.compute_crash_recovery_success_rate(slo_db)
        assert rate == 1.0


class TestLeaseAcquisitionLatency:
    """Test lease acquisition latency computation."""

    def test_latency_no_jobs(self, slo_db):
        """Latency defaults to 0 when no jobs exist."""
        latencies = slo_compute.compute_lease_acquisition_latency(slo_db)
        assert latencies == {"p50": 0.0, "p99": 0.0}

    def test_latency_with_leased_jobs(self, slo_db):
        """Latency is computed from leased jobs."""
        now = time.time()
        slo_db.execute(
            """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                         state, lease_owner, lease_until, created_at, updated_at,
                                         idempotency_key)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                "j1",
                "w1",
                "a1",
                "https://example.com",
                "LEASED",
                "owner1",
                now + 120,
                now,
                now,
                "key1",
            ),
        )
        slo_db.commit()
        latencies = slo_compute.compute_lease_acquisition_latency(slo_db)
        # Both p50 and p99 should be 120 seconds = 120000 ms
        assert latencies["p50"] == 120000.0
        assert latencies["p99"] == 120000.0


class TestOperationReplayRate:
    """Test operation replay rate computation."""

    def test_replay_rate_no_operations(self, slo_db):
        """Replay rate is 0 when no operations exist."""
        rate = slo_compute.compute_operation_replay_rate(slo_db)
        assert rate == 0.0

    def test_replay_rate_with_operations(self, slo_db):
        """Replay rate is computed from operations."""
        now = time.time()
        # Create a job first
        slo_db.execute(
            """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                         state, created_at, updated_at, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("j1", "w1", "a1", "https://example.com", "FINALIZED", now, now, "key1"),
        )
        # Create operations
        slo_db.execute(
            """INSERT INTO artifact_ops(op_key, job_id, op_type, started_at, finished_at,
                                        result_code, result_json)
               VALUES (?,?,?,?,?,?,?)""",
            ("op1", "j1", "HEAD", now, now + 1, "OK", '{"status":200}'),
        )
        slo_db.commit()
        rate = slo_compute.compute_operation_replay_rate(slo_db)
        # With operations, replay rate should be ~1%
        assert 0.0 <= rate <= 0.02


class TestComputeAllMetrics:
    """Test all-metrics computation."""

    def test_compute_all_metrics(self, slo_db):
        """Compute all SLO metrics returns dict with 7 metrics."""
        metrics = slo_compute.compute_all_slo_metrics(slo_db)
        assert len(metrics) == 7
        assert "job_completion_rate" in metrics
        assert "time_to_complete_p50" in metrics
        assert "time_to_complete_p95" in metrics
        assert "time_to_complete_p99" in metrics
        assert "crash_recovery_success_rate" in metrics
        assert "lease_acquisition_latency" in metrics
        assert "operation_replay_rate" in metrics

    def test_all_metrics_have_required_fields(self, slo_db):
        """All computed metrics have required fields."""
        metrics = slo_compute.compute_all_slo_metrics(slo_db)
        for metric_name, metric in metrics.items():
            assert metric.name
            assert metric.actual_value >= 0
            assert metric.target_value >= 0
            assert metric.error_budget >= 0
            assert metric.status in ("pass", "warning", "fail")
            assert isinstance(metric.details, dict)


class TestSLOReport:
    """Test SLO report generation."""

    def test_generate_report(self, slo_db):
        """Generate SLO report returns formatted string."""
        report = slo_compute.generate_slo_report(slo_db)
        assert "SLO REPORT" in report
        assert "Job Completion Rate" in report
        assert "Status:" in report
        assert "Target:" in report
        assert "Actual:" in report

    def test_report_contains_all_metrics(self, slo_db):
        """Report contains all metric names."""
        report = slo_compute.generate_slo_report(slo_db)
        assert "Mean Time to Complete" in report
        assert "Crash Recovery Success Rate" in report
        assert "Lease Acquisition Latency" in report
        assert "Operation Replay Rate" in report
