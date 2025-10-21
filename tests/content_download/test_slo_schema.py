"""Tests for SLO schema and threshold evaluation.

Tests verify:
  - SLO definitions are complete and valid
  - Status evaluation logic (pass/warning/fail)
  - Error budget calculation
  - Alert triggering rules
"""

from __future__ import annotations

import pytest

from DocsToKG.ContentDownload import slo_schema


class TestSLODefinitions:
    """Test SLO definitions."""

    def test_get_slo_definitions(self):
        """Get all SLO definitions."""
        definitions = slo_schema.get_slo_definitions()
        assert len(definitions) == 6
        assert all(isinstance(d, slo_schema.SLOThreshold) for d in definitions)

    def test_slo_names(self):
        """Verify all SLO definitions have unique names."""
        definitions = slo_schema.get_slo_definitions()
        names = [d.name for d in definitions]
        assert len(names) == len(set(names))
        assert "Job Completion Rate" in names
        assert "Crash Recovery Success Rate" in names

    def test_slo_metrics(self):
        """Verify all SLO definitions have unique metric identifiers."""
        definitions = slo_schema.get_slo_definitions()
        metrics = [d.metric for d in definitions]
        assert len(metrics) == len(set(metrics))


class TestStatusEvaluation:
    """Test SLO status evaluation."""

    def test_status_pass_exact_target(self):
        """Status is pass when actual equals target."""
        status = slo_schema.evaluate_slo_status(
            actual=100.0,
            target=100.0,
            error_budget=5.0,
        )
        assert status == "pass"

    def test_status_pass_within_budget(self):
        """Status is pass when within error budget."""
        status = slo_schema.evaluate_slo_status(
            actual=103.0,
            target=100.0,
            error_budget=5.0,
        )
        assert status == "pass"

    def test_status_warning_outside_budget(self):
        """Status is warning when slightly outside error budget."""
        status = slo_schema.evaluate_slo_status(
            actual=112.0,
            target=100.0,
            error_budget=5.0,
        )
        assert status == "warning"

    def test_status_fail_far_outside_budget(self):
        """Status is fail when far outside error budget."""
        status = slo_schema.evaluate_slo_status(
            actual=130.0,
            target=100.0,
            error_budget=5.0,
        )
        assert status == "fail"

    def test_status_symmetry(self):
        """Status evaluation is symmetric around target."""
        status_above = slo_schema.evaluate_slo_status(
            actual=105.0,
            target=100.0,
            error_budget=5.0,
        )
        status_below = slo_schema.evaluate_slo_status(
            actual=95.0,
            target=100.0,
            error_budget=5.0,
        )
        assert status_above == status_below == "pass"


class TestErrorBudget:
    """Test error budget calculation."""

    def test_error_budget_at_target(self):
        """Error budget is 100% when actual equals target."""
        remaining = slo_schema.calculate_error_budget_remaining(
            actual=100.0,
            target=100.0,
            error_budget=5.0,
        )
        assert remaining == 100.0

    def test_error_budget_half_consumed(self):
        """Error budget reflects consumption."""
        remaining = slo_schema.calculate_error_budget_remaining(
            actual=102.5,
            target=100.0,
            error_budget=5.0,
        )
        assert remaining == 50.0

    def test_error_budget_fully_consumed(self):
        """Error budget is 0% when fully consumed."""
        remaining = slo_schema.calculate_error_budget_remaining(
            actual=105.0,
            target=100.0,
            error_budget=5.0,
        )
        assert remaining == 0.0

    def test_error_budget_beyond_limit(self):
        """Error budget stays at 0% when exceeded."""
        remaining = slo_schema.calculate_error_budget_remaining(
            actual=110.0,
            target=100.0,
            error_budget=5.0,
        )
        assert remaining == 0.0


class TestAlertTriggering:
    """Test alert triggering logic."""

    def test_alert_on_fail(self):
        """Alert triggered on fail status."""
        should_alert = slo_schema.should_alert(
            status="fail",
            error_budget_remaining=50.0,
        )
        assert should_alert is True

    def test_alert_on_low_budget(self):
        """Alert triggered when budget < 25%."""
        should_alert = slo_schema.should_alert(
            status="warning",
            error_budget_remaining=20.0,
        )
        assert should_alert is True

    def test_no_alert_on_pass(self):
        """No alert on pass with sufficient budget."""
        should_alert = slo_schema.should_alert(
            status="pass",
            error_budget_remaining=75.0,
        )
        assert should_alert is False

    def test_no_alert_on_warning_with_budget(self):
        """No alert on warning with > 25% budget."""
        should_alert = slo_schema.should_alert(
            status="warning",
            error_budget_remaining=50.0,
        )
        assert should_alert is False


class TestJobCompletionSLO:
    """Test job completion rate SLO."""

    def test_job_completion_target(self):
        """Job completion SLO target is 99.5%."""
        assert slo_schema.JOB_COMPLETION_RATE_TARGET == 0.995

    def test_job_completion_error_budget(self):
        """Job completion error budget is 0.5%."""
        assert slo_schema.JOB_COMPLETION_ERROR_BUDGET == 0.005

    def test_job_completion_failure_states(self):
        """Job completion identifies correct failure states."""
        assert "FAILED" in slo_schema.JOB_COMPLETION_FAILURE_STATES
        assert "SKIPPED_DUPLICATE" in slo_schema.JOB_COMPLETION_FAILURE_STATES


class TestTimingsSLO:
    """Test timing-based SLOs."""

    def test_p50_target(self):
        """Mean time to complete p50 target is 30 seconds."""
        assert slo_schema.TIME_TO_COMPLETE_P50_TARGET_MS == 30000

    def test_p95_target(self):
        """Mean time to complete p95 target is 2 minutes."""
        assert slo_schema.TIME_TO_COMPLETE_P95_TARGET_MS == 120000

    def test_p99_target(self):
        """Mean time to complete p99 target is 5 minutes."""
        assert slo_schema.TIME_TO_COMPLETE_P99_TARGET_MS == 300000

    def test_lease_acquisition_p99_target(self):
        """Lease acquisition latency p99 target is 100ms."""
        assert slo_schema.LEASE_ACQUISITION_P99_TARGET_MS == 100


class TestRecoverySLO:
    """Test crash recovery SLO."""

    def test_recovery_success_target(self):
        """Crash recovery success target is 99.9%."""
        assert slo_schema.CRASH_RECOVERY_SUCCESS_TARGET == 0.999

    def test_recovery_error_budget(self):
        """Crash recovery error budget is 0.1%."""
        assert slo_schema.CRASH_RECOVERY_ERROR_BUDGET == 0.001


class TestReplaySLO:
    """Test operation replay rate SLO."""

    def test_replay_rate_target(self):
        """Operation replay rate target is 5%."""
        assert slo_schema.OPERATION_REPLAY_RATE_TARGET == 0.05

    def test_replay_error_budget(self):
        """Operation replay error budget is 10%."""
        assert slo_schema.OPERATION_REPLAY_ERROR_BUDGET == 0.10


class TestSLOWindows:
    """Test SLO observation windows."""

    def test_window_1day(self):
        """1-day window is 86,400 seconds."""
        assert slo_schema.SLO_WINDOW_1DAY_SECONDS == 86400

    def test_window_7day(self):
        """7-day window is 604,800 seconds."""
        assert slo_schema.SLO_WINDOW_7DAY_SECONDS == 604800

    def test_window_30day(self):
        """30-day window is 2,592,000 seconds."""
        assert slo_schema.SLO_WINDOW_30DAY_SECONDS == 2592000
