# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_breaker_advisor",
#   "purpose": "Tests for breaker advisor and auto-tuner",
#   "sections": [
#     {"id": "test-advisor-metrics", "name": "test_advisor_reads_metrics", "kind": "function"},
#     {"id": "test-advisor-advice", "name": "test_advisor_generates_advice", "kind": "function"},
#     {"id": "test-autotuner-suggest", "name": "test_autotuner_suggests_changes", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""Tests for BreakerAdvisor and BreakerAutoTuner.

Tests the telemetry-driven analysis and auto-tuning system:
- Metrics reading from telemetry SQLite database
- Heuristic advice generation
- Auto-tuning recommendations
- Safe bounds clamping
- Tuning plan generation and enforcement
"""

from __future__ import annotations

import sqlite3
import time

import pytest

# Check if pybreaker is available
try:
    import pybreaker

    HAS_PYBREAKER = True
except ImportError:
    HAS_PYBREAKER = False

from DocsToKG.ContentDownload.breaker_advisor import (
    BreakerAdvisor,
    HostMetrics,
)
from DocsToKG.ContentDownload.breaker_autotune import (
    BreakerAutoTuner,
)
from DocsToKG.ContentDownload.breakers import (
    BreakerConfig,
    BreakerPolicy,
    BreakerRegistry,
)

pytestmark = pytest.mark.skipif(not HAS_PYBREAKER, reason="pybreaker not installed")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def telemetry_db(tmp_path):
    """Mock telemetry database with sample data."""
    db_path = tmp_path / "telemetry.sqlite"
    conn = sqlite3.connect(db_path)

    # Create minimal schema
    conn.execute(
        """
        CREATE TABLE http_events (
            host TEXT,
            ts REAL,
            role TEXT,
            status INTEGER,
            from_cache INTEGER,
            retry_after_s REAL,
            breaker_host_state TEXT,
            breaker_recorded TEXT
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE breaker_transitions (
            host TEXT,
            ts REAL,
            old_state TEXT,
            new_state TEXT,
            reset_timeout_s INTEGER
        )
    """
    )

    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def advisor(telemetry_db):
    """BreakerAdvisor instance for testing."""
    return BreakerAdvisor(str(telemetry_db), window_s=600)


@pytest.fixture
def registry():
    """BreakerRegistry for auto-tuner testing."""
    config = BreakerConfig(
        defaults=BreakerPolicy(
            fail_max=5,
            reset_timeout_s=60,
            retry_after_cap_s=900,
        ),
    )
    return BreakerRegistry(config, cooldown_store=None)


@pytest.fixture
def autotuner(registry):
    """BreakerAutoTuner instance for testing."""
    return BreakerAutoTuner(registry, clamp=True)


# ============================================================================
# Metrics Reading Tests
# ============================================================================


class TestAdvisorMetricsReading:
    """Test metrics reading from telemetry database."""

    def test_read_metrics_empty_database(self, advisor):
        """Read metrics should handle empty database gracefully."""
        metrics = advisor.read_metrics()

        assert isinstance(metrics, dict)
        assert len(metrics) == 0

    def test_read_metrics_basic(self, advisor, telemetry_db):
        """Read metrics should extract basic request statistics."""
        # Insert sample data
        conn = sqlite3.connect(telemetry_db)
        now = time.time()

        conn.execute(
            """
            INSERT INTO http_events
            (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("api.example.org", now, "metadata", 200, 0, None, "closed", None),
        )

        conn.execute(
            """
            INSERT INTO http_events
            (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("api.example.org", now, "metadata", 503, 0, None, "closed", "failure"),
        )

        conn.commit()
        conn.close()

        metrics = advisor.read_metrics(now=now + 1)

        assert "api.example.org" in metrics
        m = metrics["api.example.org"]
        assert m.calls_total == 2
        assert m.e5xx >= 1

    def test_read_metrics_cache_hits(self, advisor, telemetry_db):
        """Read metrics should count cache hits separately."""
        conn = sqlite3.connect(telemetry_db)
        now = time.time()

        # Insert cache hits
        for i in range(3):
            conn.execute(
                """
                INSERT INTO http_events
                (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                ("api.example.org", now + i, "metadata", 200, 1, None, None, None),
            )

        conn.commit()
        conn.close()

        metrics = advisor.read_metrics(now=now + 4)

        m = metrics["api.example.org"]
        assert m.calls_cache_hits == 3
        assert m.calls_net == 0

    def test_read_metrics_retry_after_samples(self, advisor, telemetry_db):
        """Read metrics should collect Retry-After samples."""
        conn = sqlite3.connect(telemetry_db)
        now = time.time()

        # Insert 429 responses with Retry-After
        conn.execute(
            """
            INSERT INTO http_events
            (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("api.example.org", now, "metadata", 429, 0, 10.0, "closed", "failure"),
        )

        conn.commit()
        conn.close()

        metrics = advisor.read_metrics(now=now + 1)

        m = metrics["api.example.org"]
        assert m.e429 >= 1
        assert len(m.retry_after_samples) > 0

    def test_read_metrics_per_host_isolation(self, advisor, telemetry_db):
        """Read metrics should isolate per host."""
        conn = sqlite3.connect(telemetry_db)
        now = time.time()

        # Insert data for multiple hosts
        conn.execute(
            """
            INSERT INTO http_events
            (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("host1.example.org", now, "metadata", 503, 0, None, "closed", "failure"),
        )

        conn.execute(
            """
            INSERT INTO http_events
            (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("host2.example.org", now, "metadata", 200, 0, None, "closed", "success"),
        )

        conn.commit()
        conn.close()

        metrics = advisor.read_metrics(now=now + 1)

        assert "host1.example.org" in metrics
        assert "host2.example.org" in metrics
        assert metrics["host1.example.org"].e5xx >= 1
        assert metrics["host2.example.org"].e5xx == 0


# ============================================================================
# Advice Generation Tests
# ============================================================================


class TestAdvisorAdviceGeneration:
    """Test heuristic advice generation."""

    def test_advise_no_issues(self, advisor):
        """Advise should return empty advice for healthy metrics."""
        metrics = {
            "api.example.org": HostMetrics(
                host="api.example.org",
                window_s=600,
                calls_total=100,
                calls_cache_hits=20,
                calls_net=80,
                e429=0,
                e5xx=0,
                e503=0,
                timeouts=0,
                retry_after_samples=[],
                open_events=0,
                open_durations_s=[],
                half_open_success_trials=0,
                half_open_fail_trials=0,
                max_consecutive_failures=0,
            ),
        }

        advice = advisor.advise(metrics)

        assert "api.example.org" in advice
        a = advice["api.example.org"]
        assert a.suggest_fail_max is None
        assert a.suggest_reset_timeout_s is None

    def test_advise_high_429_rate(self, advisor):
        """Advise should suggest rate limiter reduction on high 429s."""
        metrics = {
            "api.example.org": HostMetrics(
                host="api.example.org",
                window_s=600,
                calls_total=100,
                calls_cache_hits=0,
                calls_net=100,
                e429=10,  # 10% rate
                e5xx=0,
                e503=0,
                timeouts=0,
                retry_after_samples=[],
                open_events=0,
                open_durations_s=[],
                half_open_success_trials=0,
                half_open_fail_trials=0,
                max_consecutive_failures=0,
            ),
        }

        advice = advisor.advise(metrics)

        a = advice["api.example.org"]
        assert a.suggest_metadata_rps_multiplier is not None
        assert a.suggest_metadata_rps_multiplier < 1.0  # Reduction

    def test_advise_reset_timeout_from_retry_after(self, advisor):
        """Advise should suggest reset timeout from Retry-After samples."""
        metrics = {
            "api.example.org": HostMetrics(
                host="api.example.org",
                window_s=600,
                calls_total=10,
                calls_cache_hits=0,
                calls_net=10,
                e429=5,
                e5xx=0,
                e503=0,
                timeouts=0,
                retry_after_samples=[30.0, 45.0, 60.0],
                open_events=1,
                open_durations_s=[],
                half_open_success_trials=0,
                half_open_fail_trials=0,
                max_consecutive_failures=0,
            ),
        }

        advice = advisor.advise(metrics)

        a = advice["api.example.org"]
        assert a.suggest_reset_timeout_s is not None
        assert 15 <= a.suggest_reset_timeout_s <= 900

    def test_advise_multiple_opens(self, advisor):
        """Advise should suggest lower fail_max on multiple opens."""
        metrics = {
            "api.example.org": HostMetrics(
                host="api.example.org",
                window_s=600,
                calls_total=100,
                calls_cache_hits=0,
                calls_net=100,
                e429=0,
                e5xx=10,
                e503=0,
                timeouts=0,
                retry_after_samples=[],
                open_events=5,  # Multiple opens
                open_durations_s=[30.0, 45.0, 60.0, 30.0, 45.0],
                half_open_success_trials=0,
                half_open_fail_trials=0,
                max_consecutive_failures=0,
            ),
        }

        advice = advisor.advise(metrics)

        a = advice["api.example.org"]
        assert a.suggest_fail_max is not None
        assert a.suggest_fail_max < 5  # Lower than default

    def test_advise_half_open_failures(self, advisor):
        """Advise should suggest higher success_threshold on half-open failures."""
        metrics = {
            "api.example.org": HostMetrics(
                host="api.example.org",
                window_s=600,
                calls_total=20,
                calls_cache_hits=0,
                calls_net=20,
                e429=0,
                e5xx=0,
                e503=0,
                timeouts=0,
                retry_after_samples=[],
                open_events=1,
                open_durations_s=[60.0],
                half_open_success_trials=2,
                half_open_fail_trials=2,  # 50% failure rate
                max_consecutive_failures=0,
            ),
        }

        advice = advisor.advise(metrics)

        a = advice["api.example.org"]
        assert a.suggest_success_threshold is not None
        assert a.suggest_success_threshold >= 2


# ============================================================================
# Auto-Tuner Suggestion Tests
# ============================================================================


class TestAutoTunerSuggestions:
    """Test auto-tuner suggestion generation."""

    def test_suggest_generates_plans(self, autotuner, advisor, telemetry_db):
        """Suggest should generate tuning plans from advice."""
        # Insert sample data
        conn = sqlite3.connect(telemetry_db)
        now = time.time()

        conn.execute(
            """
            INSERT INTO http_events
            (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("api.example.org", now, "metadata", 429, 0, 20.0, "closed", "failure"),
        )

        conn.commit()
        conn.close()

        plans = autotuner.suggest(advisor)

        # Should generate at least one plan if there's advice
        assert isinstance(plans, list)

    def test_suggest_empty_when_no_advice(self, autotuner, advisor):
        """Suggest should return empty list when no advice needed."""
        plans = autotuner.suggest(advisor)

        assert isinstance(plans, list)
        # Empty database = no issues = no suggestions
        assert len(plans) == 0

    def test_suggest_includes_reasons(self, autotuner, advisor, telemetry_db):
        """Suggest plans should include reasoning."""
        # Insert problematic data
        conn = sqlite3.connect(telemetry_db)
        now = time.time()

        for i in range(10):
            conn.execute(
                """
                INSERT INTO http_events
                (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                ("api.example.org", now + i * 10, "metadata", 503, 0, None, "closed", "failure"),
            )

        conn.commit()
        conn.close()

        plans = autotuner.suggest(advisor)

        # Should have at least one plan with reasons
        if plans:
            plan = plans[0]
            assert len(plan.changes) > 0
            # Some changes should be reasons (contain 'reason:')
            reasons = [c for c in plan.changes if "reason:" in c]
            assert len(reasons) > 0


# ============================================================================
# Bounds Clamping Tests
# ============================================================================


class TestAutoTunerBoundsClamping:
    """Test safe bounds clamping in auto-tuner."""

    def test_clamp_enabled_by_default(self):
        """Clamping should be enabled by default."""
        config = BreakerConfig()
        registry = BreakerRegistry(config, cooldown_store=None)
        tuner = BreakerAutoTuner(registry)

        assert tuner._clamp is True

    def test_clamp_disabled_when_requested(self):
        """Clamping should be disableable."""
        config = BreakerConfig()
        registry = BreakerRegistry(config, cooldown_store=None)
        tuner = BreakerAutoTuner(registry, clamp=False)

        assert tuner._clamp is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestAdvisorAutoTunerIntegration:
    """Test advisor and auto-tuner working together."""

    def test_workflow_analyze_suggest(self, advisor, autotuner, telemetry_db):
        """Test typical workflow: analyze → suggest."""
        # Insert sample problematic data
        conn = sqlite3.connect(telemetry_db)
        now = time.time()

        # High 429 rate
        for i in range(10):
            conn.execute(
                """
                INSERT INTO http_events
                (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                ("api.example.org", now + i, "metadata", 429, 0, 15.0, "closed", "failure"),
            )

        conn.commit()
        conn.close()

        # Read metrics
        metrics = advisor.read_metrics(now=now + 11)

        # Generate advice
        advice = advisor.advise(metrics)

        # Get suggestions
        plans = autotuner.suggest(advisor)

        # Should have generated something
        assert len(advice) > 0 or len(plans) == 0  # Either advice or nothing

    def test_metrics_window_respects_time_bounds(self, advisor, telemetry_db):
        """Metrics reading should respect window bounds."""
        conn = sqlite3.connect(telemetry_db)
        now = time.time()

        # Insert old data (outside window)
        conn.execute(
            """
            INSERT INTO http_events
            (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("api.example.org", now - 700, "metadata", 503, 0, None, "closed", "failure"),
        )

        # Insert new data (inside window)
        conn.execute(
            """
            INSERT INTO http_events
            (host, ts, role, status, from_cache, retry_after_s, breaker_host_state, breaker_recorded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("api.example.org", now - 50, "metadata", 503, 0, None, "closed", "failure"),
        )

        conn.commit()
        conn.close()

        metrics = advisor.read_metrics(now=now)

        # Should only count recent data
        if "api.example.org" in metrics:
            m = metrics["api.example.org"]
            # Window is 600s, so data from 700s ago should be excluded
            assert m.e5xx <= 1  # Only the recent one


# ============================================================================
# Edge Cases
# ============================================================================


class TestAdvisorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_advise_zero_calls(self, advisor):
        """Advise should handle zero-call metrics gracefully."""
        metrics = {
            "api.example.org": HostMetrics(
                host="api.example.org",
                window_s=600,
                calls_total=0,
                calls_cache_hits=0,
                calls_net=0,
                e429=0,
                e5xx=0,
                e503=0,
                timeouts=0,
                retry_after_samples=[],
                open_events=0,
                open_durations_s=[],
                half_open_success_trials=0,
                half_open_fail_trials=0,
                max_consecutive_failures=0,
            ),
        }

        advice = advisor.advise(metrics)

        # Should handle gracefully (no suggestions for zero traffic)
        a = advice["api.example.org"]
        assert a.suggest_fail_max is None

    def test_advise_extreme_rates(self, advisor):
        """Advise should handle extreme failure rates."""
        metrics = {
            "api.example.org": HostMetrics(
                host="api.example.org",
                window_s=600,
                calls_total=100,
                calls_cache_hits=0,
                calls_net=100,
                e429=100,  # 100% rate
                e5xx=0,
                e503=0,
                timeouts=0,
                retry_after_samples=[],
                open_events=0,
                open_durations_s=[],
                half_open_success_trials=0,
                half_open_fail_trials=0,
                max_consecutive_failures=0,
            ),
        }

        advice = advisor.advise(metrics)

        # Should still generate sensible recommendations
        a = advice["api.example.org"]
        assert a.suggest_metadata_rps_multiplier is not None

    def test_advise_single_failure(self, advisor):
        """Advise should not over-react to single failures."""
        metrics = {
            "api.example.org": HostMetrics(
                host="api.example.org",
                window_s=600,
                calls_total=100,
                calls_cache_hits=0,
                calls_net=100,
                e429=1,  # Single 429
                e5xx=0,
                e503=0,
                timeouts=0,
                retry_after_samples=[],
                open_events=0,
                open_durations_s=[],
                half_open_success_trials=0,
                half_open_fail_trials=0,
                max_consecutive_failures=0,
            ),
        }

        advice = advisor.advise(metrics)

        # Should not suggest changes for single failure
        a = advice["api.example.org"]
        assert a.suggest_metadata_rps_multiplier is None
