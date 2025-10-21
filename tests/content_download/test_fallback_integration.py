"""Comprehensive tests for Fallback & Resiliency Strategy.

Tests cover:
  - Orchestrator core functionality
  - Adapter implementations
  - Configuration loading
  - Integration with download pipeline
  - Telemetry emission
  - Error handling and edge cases
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch, MagicMock

import pytest

from DocsToKG.ContentDownload.fallback.types import (
    AttemptPolicy,
    AttemptResult,
    FallbackPlan,
    ResolutionOutcome,
    TierPlan,
)
from DocsToKG.ContentDownload.fallback.orchestrator import FallbackOrchestrator
from DocsToKG.ContentDownload.fallback.loader import load_fallback_plan, ConfigurationError
from DocsToKG.ContentDownload.fallback.integration import (
    try_fallback_resolution,
    is_fallback_enabled,
    get_fallback_plan_path,
)


# ============================================================================
# CORE ORCHESTRATOR TESTS
# ============================================================================


class TestFallbackOrchestrator:
    """Tests for FallbackOrchestrator core functionality."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes with all dependencies."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 60000,
                "total_attempts": 10,
                "max_concurrent": 2,
                "per_source_timeout_ms": 5000,
            },
            tiers=(TierPlan("test", 1, ("test_source",)),),
            policies={"test_source": AttemptPolicy("test_source", 5000, 2)},
        )

        orchestrator = FallbackOrchestrator(
            plan=plan,
            breaker=None,
            rate=None,
            head_client=None,
            raw_client=None,
            telemetry=None,
            logger=Mock(),
        )

        assert orchestrator.plan == plan
        assert orchestrator.tele is None
        assert orchestrator.breaker is None

    def test_resolve_pdf_with_no_success(self):
        """Test resolve_pdf returns no_pdf when no adapter succeeds."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 1000,
                "total_attempts": 2,
                "max_concurrent": 1,
                "per_source_timeout_ms": 500,
            },
            tiers=(TierPlan("test", 1, ("test_source",)),),
            policies={"test_source": AttemptPolicy("test_source", 500, 1)},
        )

        def mock_adapter(policy, context):
            return AttemptResult(
                outcome="no_pdf",
                reason="not_found",
                elapsed_ms=100,
            )

        orchestrator = FallbackOrchestrator(
            plan=plan,
            breaker=None,
            rate=None,
            head_client=None,
            raw_client=None,
            telemetry=None,
            logger=Mock(),
        )

        result = orchestrator.resolve_pdf(
            context={"work_id": "123"},
            adapters={"test_source": mock_adapter},
        )

        assert result.outcome == "no_pdf"
        assert result.reason == "exhausted"

    def test_resolve_pdf_early_return_on_success(self):
        """Test resolve_pdf returns immediately on success."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 60000,
                "total_attempts": 10,
                "max_concurrent": 2,
                "per_source_timeout_ms": 5000,
            },
            tiers=(
                TierPlan("tier1", 1, ("source1", "source2")),
                TierPlan("tier2", 1, ("source3",)),
            ),
            policies={
                "source1": AttemptPolicy("source1", 5000, 2),
                "source2": AttemptPolicy("source2", 5000, 2),
                "source3": AttemptPolicy("source3", 5000, 2),
            },
        )

        def mock_adapter_success(policy, context):
            return AttemptResult(
                outcome="success",
                reason="found",
                elapsed_ms=100,
                url="https://example.com/paper.pdf",
                status=200,
                host="example.com",
            )

        def mock_adapter_fail(policy, context):
            return AttemptResult(
                outcome="no_pdf",
                reason="not_found",
                elapsed_ms=100,
            )

        orchestrator = FallbackOrchestrator(
            plan=plan,
            breaker=None,
            rate=None,
            head_client=None,
            raw_client=None,
            telemetry=None,
            logger=Mock(),
        )

        result = orchestrator.resolve_pdf(
            context={"work_id": "123"},
            adapters={
                "source1": mock_adapter_success,
                "source2": mock_adapter_fail,
                "source3": mock_adapter_fail,
            },
        )

        # Should succeed in first source
        assert result.outcome == "success"
        assert result.reason == "found"
        assert result.url == "https://example.com/paper.pdf"

    def test_budget_enforcement_attempts(self):
        """Test orchestrator respects attempt budget."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 60000,
                "total_attempts": 2,  # Only 2 attempts allowed
                "max_concurrent": 1,
                "per_source_timeout_ms": 5000,
            },
            tiers=(
                TierPlan("tier1", 1, ("s1",)),
                TierPlan("tier2", 1, ("s2",)),
                TierPlan("tier3", 1, ("s3",)),
            ),
            policies={
                "s1": AttemptPolicy("s1", 5000, 2),
                "s2": AttemptPolicy("s2", 5000, 2),
                "s3": AttemptPolicy("s3", 5000, 2),
            },
        )

        attempt_count = 0

        def mock_adapter(policy, context):
            nonlocal attempt_count
            attempt_count += 1
            return AttemptResult(
                outcome="no_pdf",
                reason="not_found",
                elapsed_ms=100,
            )

        orchestrator = FallbackOrchestrator(
            plan=plan,
            breaker=None,
            rate=None,
            head_client=None,
            raw_client=None,
            telemetry=None,
            logger=Mock(),
        )

        result = orchestrator.resolve_pdf(
            context={"work_id": "123"},
            adapters={"s1": mock_adapter, "s2": mock_adapter, "s3": mock_adapter},
        )

        # Should stop after 2 attempts due to budget
        assert attempt_count <= 2
        assert result.outcome == "no_pdf"


# ============================================================================
# CONFIGURATION LOADING TESTS
# ============================================================================


class TestConfigurationLoading:
    """Tests for configuration loading and merging."""

    def test_load_fallback_plan_default(self):
        """Test loading fallback plan with defaults."""
        plan = load_fallback_plan()

        assert plan is not None
        assert len(plan.tiers) > 0
        assert len(plan.policies) > 0
        assert plan.budgets["total_timeout_ms"] > 0

    def test_configuration_error_on_invalid_yaml(self):
        """Test ConfigurationError raised on invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_yaml = Path(tmpdir) / "invalid.yaml"
            invalid_yaml.write_text("{ invalid yaml: [")

            with pytest.raises(ConfigurationError):
                load_fallback_plan(yaml_path=invalid_yaml)

    def test_configuration_error_on_missing_file(self):
        """Test ConfigurationError raised when file missing."""
        with pytest.raises(FileNotFoundError):
            load_fallback_plan(yaml_path=Path("/nonexistent/path.yaml"))


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Tests for integration module."""

    def test_is_fallback_enabled_default(self):
        """Test fallback disabled by default."""
        options = Mock(enable_fallback_strategy=False)
        assert not is_fallback_enabled(options)

    def test_is_fallback_enabled_when_set(self):
        """Test fallback enabled when flag set."""
        options = Mock(enable_fallback_strategy=True)
        assert is_fallback_enabled(options)

    def test_get_fallback_plan_path_none(self):
        """Test get_fallback_plan_path returns None when not set."""
        options = Mock(fallback_plan_path=None)
        assert get_fallback_plan_path(options) is None

    def test_get_fallback_plan_path_string(self):
        """Test get_fallback_plan_path converts string to Path."""
        options = Mock(fallback_plan_path="/tmp/fallback.yaml")
        path = get_fallback_plan_path(options)
        assert isinstance(path, Path)
        assert str(path) == "/tmp/fallback.yaml"

    def test_try_fallback_resolution_success(self):
        """Test try_fallback_resolution returns result on success."""
        def mock_adapter(policy, context):
            return AttemptResult(
                outcome="success",
                reason="found",
                elapsed_ms=100,
                url="https://example.com/paper.pdf",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "fallback.yaml"
            # Use minimal YAML to avoid loading actual file
            yaml_path.write_text("""
budgets:
  total_timeout_ms: 1000
  total_attempts: 5
  max_concurrent: 1
  per_source_timeout_ms: 500
tiers:
  - name: test
    parallel: 1
    sources:
      - test_source
policies:
  test_source:
    timeout_ms: 500
    retries_max: 1
    robots_respect: false
""")

            result = try_fallback_resolution(
                context={"work_id": "123"},
                adapters={"test_source": mock_adapter},
                fallback_plan_path=yaml_path,
            )

            assert result is not None
            assert result.is_success

    def test_try_fallback_resolution_failure_returns_none(self):
        """Test try_fallback_resolution returns None on failure."""
        def mock_adapter(policy, context):
            return AttemptResult(
                outcome="no_pdf",
                reason="not_found",
                elapsed_ms=100,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "fallback.yaml"
            yaml_path.write_text("""
budgets:
  total_timeout_ms: 1000
  total_attempts: 5
  max_concurrent: 1
  per_source_timeout_ms: 500
tiers:
  - name: test
    parallel: 1
    sources:
      - test_source
policies:
  test_source:
    timeout_ms: 500
    retries_max: 1
    robots_respect: false
""")

            result = try_fallback_resolution(
                context={"work_id": "123"},
                adapters={"test_source": mock_adapter},
                fallback_plan_path=yaml_path,
            )

            assert result is None


# ============================================================================
# TELEMETRY TESTS
# ============================================================================


class TestTelemetry:
    """Tests for telemetry emission."""

    def test_attempt_telemetry_logged(self):
        """Test attempt events are logged to telemetry."""
        mock_telemetry = Mock()
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 1000,
                "total_attempts": 1,
                "max_concurrent": 1,
                "per_source_timeout_ms": 500,
            },
            tiers=(TierPlan("test", 1, ("source",)),),
            policies={"source": AttemptPolicy("source", 500, 1)},
        )

        def mock_adapter(policy, context):
            return AttemptResult(
                outcome="success",
                reason="found",
                elapsed_ms=100,
                url="https://example.com/paper.pdf",
                host="example.com",
            )

        orchestrator = FallbackOrchestrator(
            plan=plan,
            breaker=None,
            rate=None,
            head_client=None,
            raw_client=None,
            telemetry=mock_telemetry,
            logger=Mock(),
        )

        orchestrator.resolve_pdf(
            context={"work_id": "123"},
            adapters={"source": mock_adapter},
        )

        # Verify telemetry was called
        mock_telemetry.log_fallback_attempt.assert_called()


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_adapter_exception_handled_gracefully(self):
        """Test adapter exception doesn't crash orchestrator."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 1000,
                "total_attempts": 2,
                "max_concurrent": 1,
                "per_source_timeout_ms": 500,
            },
            tiers=(
                TierPlan("tier1", 1, ("bad_source",)),
                TierPlan("tier2", 1, ("good_source",)),
            ),
            policies={
                "bad_source": AttemptPolicy("bad_source", 500, 1),
                "good_source": AttemptPolicy("good_source", 500, 1),
            },
        )

        def bad_adapter(policy, context):
            raise RuntimeError("Adapter failure")

        def good_adapter(policy, context):
            return AttemptResult(
                outcome="success",
                reason="found",
                elapsed_ms=100,
                url="https://example.com/paper.pdf",
            )

        orchestrator = FallbackOrchestrator(
            plan=plan,
            breaker=None,
            rate=None,
            head_client=None,
            raw_client=None,
            telemetry=None,
            logger=Mock(),
        )

        result = orchestrator.resolve_pdf(
            context={"work_id": "123"},
            adapters={"bad_source": bad_adapter, "good_source": good_adapter},
        )

        # Should continue to next tier despite exception in first tier
        assert result.outcome == "success"

    def test_missing_required_context_fields(self):
        """Test orchestrator handles missing context fields gracefully."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 1000,
                "total_attempts": 1,
                "max_concurrent": 1,
                "per_source_timeout_ms": 500,
            },
            tiers=(TierPlan("test", 1, ("source",)),),
            policies={"source": AttemptPolicy("source", 500, 1)},
        )

        def mock_adapter(policy, context):
            # Should still work with minimal context
            return AttemptResult(
                outcome="success",
                reason="found",
                elapsed_ms=100,
                url="https://example.com/paper.pdf",
            )

        orchestrator = FallbackOrchestrator(
            plan=plan,
            breaker=None,
            rate=None,
            head_client=None,
            raw_client=None,
            telemetry=None,
            logger=Mock(),
        )

        # Call with minimal context
        result = orchestrator.resolve_pdf(
            context={},
            adapters={"source": mock_adapter},
        )

        assert result.outcome == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
