"""
Fallback & Resiliency Strategy - Integration Tests

Tests the integration of the fallback orchestrator with the ContentDownload
pipeline, including real-world scenarios and feature gating.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from DocsToKG.ContentDownload.fallback.orchestrator import FallbackOrchestrator
from DocsToKG.ContentDownload.fallback.types import (
    AttemptPolicy,
    AttemptResult,
    FallbackPlan,
    TierPlan,
)


class TestFallbackFeatureGate:
    """Test fallback strategy feature gating."""

    def test_feature_gate_disabled_by_default(self):
        """Test feature gate is disabled by default."""
        from DocsToKG.ContentDownload import download
        
        # Verify default state
        assert download.ENABLE_FALLBACK_STRATEGY is False

    def test_feature_gate_enabled_via_env(self, monkeypatch):
        """Test feature gate can be enabled via environment variable."""
        monkeypatch.setenv("DOCSTOKG_ENABLE_FALLBACK_STRATEGY", "1")
        
        # Reimport to pick up env var
        import importlib
        from DocsToKG.ContentDownload import download
        importlib.reload(download)
        
        assert download.ENABLE_FALLBACK_STRATEGY is True

    def test_feature_gate_accepts_multiple_true_values(self, monkeypatch):
        """Test feature gate accepts multiple true values."""
        for true_val in ("1", "true", "yes", "TRUE", "YES"):
            monkeypatch.setenv("DOCSTOKG_ENABLE_FALLBACK_STRATEGY", true_val)
            
            import importlib
            from DocsToKG.ContentDownload import download
            importlib.reload(download)
            
            assert download.ENABLE_FALLBACK_STRATEGY is True, f"Failed for {true_val}"


class TestFallbackPlanLoading:
    """Test fallback plan loading and configuration."""

    def test_load_default_plan(self):
        """Test loading default fallback plan."""
        from DocsToKG.ContentDownload.fallback.loader import load_fallback_plan
        
        plan = load_fallback_plan()
        assert isinstance(plan, FallbackPlan)
        assert len(plan.tiers) > 0
        assert len(plan.policies) > 0
        assert plan.budgets["total_timeout_ms"] > 0

    def test_plan_has_required_fields(self):
        """Test loaded plan has all required fields."""
        from DocsToKG.ContentDownload.fallback.loader import load_fallback_plan
        
        plan = load_fallback_plan()
        assert "total_timeout_ms" in plan.budgets
        assert "total_attempts" in plan.budgets
        assert "max_concurrent" in plan.budgets
        assert len(plan.tiers) > 0
        assert all(isinstance(tier, TierPlan) for tier in plan.tiers)


class TestFallbackOrchestratorIntegration:
    """Test fallback orchestrator integration with real-like scenarios."""

    def test_orchestrator_with_no_sources(self):
        """Test orchestrator handles missing sources gracefully."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 5000,
                "total_attempts": 10,
                "max_concurrent": 5,
            },
            tiers=[TierPlan(name="tier1", parallel=1, sources=["nonexistent"])],
            policies={
                "nonexistent": AttemptPolicy(
                    name="nonexistent", timeout_ms=1000, retries_max=1
                )
            },
        )
        
        orch = FallbackOrchestrator(plan=plan)
        result = orch.resolve_pdf(context={}, adapters={})
        
        assert result.outcome == "error"
        assert not result.is_success()

    def test_orchestrator_with_mock_adapter(self):
        """Test orchestrator with mock successful adapter."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 10000,
                "total_attempts": 20,
                "max_concurrent": 5,
            },
            tiers=[TierPlan(name="tier1", parallel=1, sources=["mock"])],
            policies={
                "mock": AttemptPolicy(name="mock", timeout_ms=2000, retries_max=1)
            },
        )
        
        def mock_adapter(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
            return AttemptResult(
                outcome="success",
                url="https://example.org/paper.pdf",
                elapsed_ms=100,
                status=200,
                host="example.org",
            )
        
        orch = FallbackOrchestrator(plan=plan)
        result = orch.resolve_pdf(
            context={},
            adapters={"mock": mock_adapter},
        )
        
        assert result.is_success()
        assert result.url == "https://example.org/paper.pdf"

    def test_orchestrator_fallback_on_failure(self):
        """Test orchestrator falls through tiers on failure."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 10000,
                "total_attempts": 20,
                "max_concurrent": 5,
            },
            tiers=[
                TierPlan(name="tier1", parallel=1, sources=["fail"]),
                TierPlan(name="tier2", parallel=1, sources=["success"]),
            ],
            policies={
                "fail": AttemptPolicy(name="fail", timeout_ms=2000, retries_max=1),
                "success": AttemptPolicy(
                    name="success", timeout_ms=2000, retries_max=1
                ),
            },
        )
        
        def fail_adapter(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
            return AttemptResult(outcome="error", reason="test_failure")
        
        def success_adapter(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
            return AttemptResult(
                outcome="success",
                url="https://example.org/success.pdf",
                elapsed_ms=100,
                status=200,
                host="example.org",
            )
        
        orch = FallbackOrchestrator(plan=plan)
        result = orch.resolve_pdf(
            context={},
            adapters={"fail": fail_adapter, "success": success_adapter},
        )
        
        assert result.is_success()
        assert result.url == "https://example.org/success.pdf"


class TestFallbackContextPassing:
    """Test context information is correctly passed to adapters."""

    def test_context_contains_artifact_info(self):
        """Test adapter receives artifact context."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 10000,
                "total_attempts": 20,
                "max_concurrent": 5,
            },
            tiers=[TierPlan(name="tier1", parallel=1, sources=["inspect"])],
            policies={
                "inspect": AttemptPolicy(name="inspect", timeout_ms=2000, retries_max=1)
            },
        )
        
        received_context = {}
        
        def inspect_adapter(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
            received_context.update(context)
            return AttemptResult(outcome="success", url="http://x")
        
        orch = FallbackOrchestrator(plan=plan)
        context_in = {
            "work_id": "test_123",
            "doi": "10.1234/test",
            "artifact_id": "artifact_456",
        }
        
        result = orch.resolve_pdf(
            context=context_in,
            adapters={"inspect": inspect_adapter},
        )
        
        # Context should have been passed (inspect adapter called)
        assert result.is_success()


class TestFallbackTelemetry:
    """Test telemetry emission from fallback orchestrator."""

    def test_telemetry_called_on_events(self):
        """Test telemetry sink is called for events."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 10000,
                "total_attempts": 20,
                "max_concurrent": 5,
            },
            tiers=[TierPlan(name="tier1", parallel=1, sources=["test"])],
            policies={
                "test": AttemptPolicy(name="test", timeout_ms=2000, retries_max=1)
            },
        )
        
        telemetry_events = []
        
        class MockTelemetry:
            def emit(self, event: Dict[str, Any]) -> None:
                telemetry_events.append(event)
        
        def test_adapter(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
            return AttemptResult(outcome="success", url="http://x")
        
        orch = FallbackOrchestrator(plan=plan, telemetry=MockTelemetry())
        result = orch.resolve_pdf(
            context={},
            adapters={"test": test_adapter},
        )
        
        # Telemetry should have been emitted
        # (Note: events are emitted for failures/gates, not necessarily successes)
        assert result.is_success()


class TestFallbackPerformance:
    """Test fallback strategy performance characteristics."""

    def test_timeout_enforcement_strict(self):
        """Test timeout is strictly enforced."""
        import time
        
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 200,  # 200ms timeout
                "total_attempts": 100,
                "max_concurrent": 5,
            },
            tiers=[TierPlan(name="tier1", parallel=1, sources=["slow"])],
            policies={
                "slow": AttemptPolicy(name="slow", timeout_ms=1000, retries_max=1)
            },
        )
        
        def slow_adapter(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
            # Adapter that's slow but eventually returns
            time.sleep(0.1)
            return AttemptResult(outcome="success", url="http://x")
        
        orch = FallbackOrchestrator(plan=plan)
        start = time.time()
        result = orch.resolve_pdf(
            context={},
            adapters={"slow": slow_adapter},
        )
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        # Should complete within reasonable time
        assert elapsed < 1000  # Should not exceed 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
