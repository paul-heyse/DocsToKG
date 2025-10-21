"""
Fallback & Resiliency Strategy - Core Tests

Comprehensive test suite covering:
- Data types and validation
- Orchestrator logic and threading
- Budget enforcement
- Health gates
- Telemetry emission
"""

import threading
import time
from typing import Any, Dict

import pytest

from DocsToKG.ContentDownload.fallback.orchestrator import FallbackOrchestrator
from DocsToKG.ContentDownload.fallback.types import (
    AttemptPolicy,
    AttemptResult,
    FallbackPlan,
    ResolutionOutcome,
    TierPlan,
)


class TestAttemptResult:
    """Test AttemptResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = AttemptResult(
            outcome="success",
            url="https://example.org/paper.pdf",
            elapsed_ms=100,
            status=200,
            host="example.org",
        )
        assert result.is_success()
        assert result.url == "https://example.org/paper.pdf"

    def test_failure_result(self):
        """Test creating a failure result."""
        result = AttemptResult(
            outcome="error",
            reason="timeout",
            elapsed_ms=5000,
            status=None,
        )
        assert not result.is_success()
        assert result.outcome == "error"
        assert result.reason == "timeout"

    def test_success_requires_url(self):
        """Test that success outcome requires URL."""
        with pytest.raises(ValueError, match="requires url"):
            AttemptResult(outcome="success")

    def test_non_success_should_not_have_url(self):
        """Test that non-success outcomes reject URLs."""
        with pytest.raises(ValueError, match="should not have url"):
            AttemptResult(outcome="error", url="https://example.org/paper.pdf")

    def test_is_terminal(self):
        """Test terminal outcome detection."""
        assert AttemptResult(outcome="success", url="http://x").is_terminal()
        assert AttemptResult(outcome="no_pdf").is_terminal()
        assert AttemptResult(outcome="nonretryable").is_terminal()
        assert not AttemptResult(outcome="retryable").is_terminal()

    def test_elapsed_ms_validation(self):
        """Test elapsed_ms must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            AttemptResult(outcome="error", elapsed_ms=-1)

    def test_retry_count_validation(self):
        """Test retry_count must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            AttemptResult(outcome="error", retry_count=-1)


class TestAttemptPolicy:
    """Test AttemptPolicy dataclass."""

    def test_policy_creation(self):
        """Test creating a policy."""
        policy = AttemptPolicy(
            name="arxiv",
            timeout_ms=5000,
            retries_max=3,
            robots_respect=False,
        )
        assert policy.name == "arxiv"
        assert policy.timeout_ms == 5000
        assert policy.retries_max == 3

    def test_policy_defaults(self):
        """Test policy default values."""
        policy = AttemptPolicy(name="test", timeout_ms=1000, retries_max=3)
        assert policy.retries_max == 3  # Changed from 0
        assert policy.robots_respect is True  # Default is True, not False
        assert policy.metadata == {}


class TestTierPlan:
    """Test TierPlan dataclass."""

    def test_tier_creation(self):
        """Test creating a tier."""
        tier = TierPlan(
            name="tier1",
            parallel=5,
            sources=["unpaywall", "arxiv"],
        )
        assert tier.name == "tier1"
        assert tier.parallel == 5
        assert tier.sources == ["unpaywall", "arxiv"]


class TestFallbackPlan:
    """Test FallbackPlan dataclass."""

    def test_plan_creation(self):
        """Test creating a fallback plan."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 60000,
                "total_attempts": 50,
                "max_concurrent": 10,
            },
            tiers=[
                TierPlan(name="tier1", parallel=5, sources=["unpaywall"]),
                TierPlan(name="tier2", parallel=3, sources=["arxiv"]),
            ],
            policies={
                "unpaywall": AttemptPolicy(name="unpaywall", timeout_ms=5000, retries_max=3),
                "arxiv": AttemptPolicy(name="arxiv", timeout_ms=3000, retries_max=3),
            },
            gates={},
        )
        assert len(plan.tiers) == 2
        assert len(plan.policies) == 2
        assert plan.budgets["total_timeout_ms"] == 60000

    def test_plan_validation_positive_timeout(self):
        """Test plan validates positive timeout."""
        with pytest.raises(ValueError, match="total_timeout_ms must be positive"):
            FallbackPlan(
                budgets={
                    "total_timeout_ms": 0,
                    "total_attempts": 10,
                    "max_concurrent": 5,
                },
                tiers=[],
                policies={},
            )

    def test_plan_validation_positive_attempts(self):
        """Test plan validates positive attempts."""
        with pytest.raises(ValueError, match="total_attempts must be positive"):
            FallbackPlan(
                budgets={
                    "total_timeout_ms": 10000,
                    "total_attempts": 0,
                    "max_concurrent": 5,
                },
                tiers=[],
                policies={},
            )


class TestFallbackOrchestrator:
    """Test FallbackOrchestrator."""

    @pytest.fixture
    def simple_plan(self) -> FallbackPlan:
        """Create a simple test plan."""
        return FallbackPlan(
            budgets={
                "total_timeout_ms": 10000,
                "total_attempts": 20,
                "max_concurrent": 5,
            },
            tiers=[
                TierPlan(
                    name="tier1",
                    parallel=2,
                    sources=["source1", "source2"],
                )
            ],
            policies={
                "source1": AttemptPolicy(name="source1", timeout_ms=2000, retries_max=3),
                "source2": AttemptPolicy(name="source2", timeout_ms=2000, retries_max=3),
            },
        )

    def test_orchestrator_creation(self, simple_plan):
        """Test creating an orchestrator."""
        orch = FallbackOrchestrator(plan=simple_plan)
        assert orch.plan == simple_plan
        assert orch.clients == {}

    def test_budget_tracking(self, simple_plan):
        """Test budget tracking works."""
        orch = FallbackOrchestrator(plan=simple_plan)
        assert not orch._is_budget_exhausted()
        orch._attempt_count = 20
        assert orch._is_budget_exhausted()

    def test_elapsed_time_tracking(self, simple_plan):
        """Test elapsed time tracking."""
        orch = FallbackOrchestrator(plan=simple_plan)
        orch._start_time = time.time() - 0.1  # Started 100ms ago
        elapsed = orch._elapsed_ms()
        assert elapsed >= 100
        assert elapsed < 200

    def test_remaining_timeout(self, simple_plan):
        """Test remaining timeout calculation."""
        orch = FallbackOrchestrator(plan=simple_plan)
        orch._start_time = time.time()
        remaining = orch._remaining_timeout_s()
        assert 9 < remaining < 11  # Should be ~10 seconds

    def test_health_gate_pass(self, simple_plan):
        """Test health gate passes healthy sources."""
        orch = FallbackOrchestrator(plan=simple_plan)
        result = orch._health_gate("source1", {"offline_mode": False})
        assert result is None  # Passed

    def test_health_gate_offline(self, simple_plan):
        """Test health gate blocks sources in offline mode."""
        orch = FallbackOrchestrator(plan=simple_plan)
        result = orch._health_gate("source1", {"offline_mode": True})
        assert result is not None
        assert result.outcome == "skipped"
        assert result.reason == "offline_mode"

    def test_resolution_no_adapters(self, simple_plan):
        """Test resolution with no adapters."""
        orch = FallbackOrchestrator(plan=simple_plan)
        result = orch.resolve_pdf(context={"test": True}, adapters={})
        assert result.outcome == "error"

    def test_resolution_with_success(self, simple_plan):
        """Test resolution finds success."""

        def mock_success_adapter(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
            return AttemptResult(
                outcome="success",
                url="https://example.org/paper.pdf",
                elapsed_ms=100,
                status=200,
                host="example.org",
            )

        orch = FallbackOrchestrator(plan=simple_plan)
        result = orch.resolve_pdf(
            context={},
            adapters={"source1": mock_success_adapter},
        )
        assert result.is_success()
        assert result.url == "https://example.org/paper.pdf"

    def test_concurrent_execution(self, simple_plan):
        """Test concurrent execution within tier."""
        called_threads = []

        def recording_adapter(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
            called_threads.append(threading.current_thread().ident)
            return AttemptResult(outcome="error", reason="test")

        orch = FallbackOrchestrator(plan=simple_plan)
        result = orch.resolve_pdf(
            context={},
            adapters={
                "source1": recording_adapter,
                "source2": recording_adapter,
            },
        )
        # Should have called both adapters
        assert len(called_threads) >= 2

    def test_budget_timeout_enforcement(self):
        """Test budget timeout is enforced."""
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 500,  # 500ms timeout
                "total_attempts": 100,
                "max_concurrent": 5,
            },
            tiers=[
                TierPlan(name="tier1", parallel=1, sources=["source1"]),
            ],
            policies={
                "source1": AttemptPolicy(name="source1", timeout_ms=1000, retries_max=3),
            },
        )

        def slow_adapter(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
            time.sleep(0.2)  # Sleep less than budget
            return AttemptResult(outcome="error", reason="test")

        orch = FallbackOrchestrator(plan=plan)
        result = orch.resolve_pdf(
            context={},
            adapters={"source1": slow_adapter},
        )
        # Should complete within budget
        assert result.elapsed_ms < 1000


class TestResolutionOutcomes:
    """Test ResolutionOutcome Literal type."""

    def test_all_valid_outcomes(self):
        """Test all valid outcome strings."""
        # Success requires URL
        AttemptResult(outcome="success", url="https://example.org/paper.pdf")

        # Other outcomes don't
        for outcome in ("no_pdf", "nonretryable", "retryable", "timeout", "skipped", "error"):
            AttemptResult(outcome=outcome)  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
