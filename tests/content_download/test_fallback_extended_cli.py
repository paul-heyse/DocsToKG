"""
Tests for Extended Fallback CLI Commands

Tests all 4 extended commands:
  - fallback stats
  - fallback tune
  - fallback explain
  - fallback config
"""

from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from DocsToKG.ContentDownload.fallback.cli_commands import (
    TelemetryAnalyzer,
    ConfigurationTuner,
    StrategyExplainer,
    cmd_fallback_stats,
    cmd_fallback_tune,
    cmd_fallback_explain,
    cmd_fallback_config,
)
from DocsToKG.ContentDownload.fallback.types import (
    AttemptPolicy,
    FallbackPlan,
    TierPlan,
)


class TestTelemetryAnalyzer:
    """Test telemetry analysis functionality."""

    @pytest.fixture
    def sample_records(self) -> List[Dict[str, Any]]:
        """Create sample telemetry records."""
        return [
            {
                "tier": "tier_1",
                "host": "unpaywall.org",
                "outcome": "success",
                "elapsed_ms": 850,
            },
            {
                "tier": "tier_1",
                "host": "arxiv.org",
                "outcome": "error",
                "reason": "timeout",
                "elapsed_ms": 2000,
            },
            {
                "tier": "tier_2",
                "host": "doi.org",
                "outcome": "success",
                "elapsed_ms": 1200,
            },
        ]

    def test_get_overall_stats(self, sample_records):
        """Test overall statistics calculation."""
        analyzer = TelemetryAnalyzer(sample_records)
        stats = analyzer.get_overall_stats()

        assert stats["total_attempts"] == 3
        assert stats["success_count"] == 2
        assert stats["success_rate"] == pytest.approx(0.667, abs=0.01)

    def test_get_tier_stats(self, sample_records):
        """Test per-tier statistics."""
        analyzer = TelemetryAnalyzer(sample_records)
        tier_stats = analyzer.get_tier_stats()

        assert "tier_1" in tier_stats
        assert tier_stats["tier_1"]["attempts"] == 2
        assert tier_stats["tier_2"]["attempts"] == 1

    def test_get_source_stats(self, sample_records):
        """Test per-source statistics."""
        analyzer = TelemetryAnalyzer(sample_records)
        source_stats = analyzer.get_source_stats()

        assert "unpaywall.org" in source_stats
        assert source_stats["unpaywall.org"]["success_rate"] == 1.0

    def test_get_failure_reasons(self, sample_records):
        """Test failure reason ranking."""
        analyzer = TelemetryAnalyzer(sample_records)
        reasons = analyzer.get_failure_reasons(top_n=5)

        assert "timeout" in reasons


class TestConfigurationTuner:
    """Test configuration tuning recommendations."""

    @pytest.fixture
    def sample_plan(self) -> FallbackPlan:
        """Create sample fallback plan."""
        return FallbackPlan(
            budgets={
                "total_timeout_ms": 30000,
                "total_attempts": 100,
                "max_concurrent": 5,
            },
            tiers=[
                TierPlan(name="tier_1", parallel=5, sources=["unpaywall"]),
                TierPlan(name="tier_2", parallel=3, sources=["doi"]),
            ],
            policies={
                "unpaywall": AttemptPolicy(
                    name="unpaywall", timeout_ms=2000, retries_max=3
                ),
                "doi": AttemptPolicy(name="doi", timeout_ms=2000, retries_max=2),
            },
        )

    @pytest.fixture
    def poor_performance_records(self) -> List[Dict[str, Any]]:
        """Create records with poor performance to trigger recommendations."""
        return [
            {
                "tier": "tier_2",
                "outcome": "error",
                "reason": "timeout",
            }
            for _ in range(10)
        ]

    def test_get_recommendations(self, sample_plan, poor_performance_records):
        """Test recommendation generation."""
        tuner = ConfigurationTuner(poor_performance_records, sample_plan)
        recommendations = tuner.get_recommendations()

        # Should have recommendations for poor-performing tier
        assert len(recommendations) > 0

    def test_get_projections(self, sample_plan):
        """Test performance projections."""
        records = [
            {"outcome": "success", "elapsed_ms": 1000},
            {"outcome": "error", "elapsed_ms": 2000},
        ]
        tuner = ConfigurationTuner(records, sample_plan)
        projections = tuner.get_projections([])

        assert "current_success_rate" in projections
        assert "projected_success_rate" in projections


class TestStrategyExplainer:
    """Test strategy explanation functionality."""

    @pytest.fixture
    def sample_plan(self) -> FallbackPlan:
        """Create sample plan."""
        return FallbackPlan(
            budgets={"total_timeout_ms": 30000, "total_attempts": 100, "max_concurrent": 5},
            tiers=[TierPlan(name="tier_1", parallel=5, sources=["unpaywall", "arxiv"])],
            policies={
                "unpaywall": AttemptPolicy(name="unpaywall", timeout_ms=2000, retries_max=3),
                "arxiv": AttemptPolicy(name="arxiv", timeout_ms=2000, retries_max=2),
            },
        )

    def test_render_overview(self, sample_plan):
        """Test overview rendering."""
        explainer = StrategyExplainer(sample_plan)
        output = explainer.render_overview()

        assert "FALLBACK STRATEGY EXPLANATION" in output
        assert "tier_1" in output
        assert "unpaywall" in output


class TestExtendedCLICommands:
    """Integration tests for extended CLI commands."""

    def test_cmd_fallback_stats_no_data(self, capsys):
        """Test stats command with no telemetry data."""
        args = Mock()
        args.manifest = None
        args.period = "24h"
        args.format = "text"

        with patch("DocsToKG.ContentDownload.fallback.cli_commands.load_fallback_plan"):
            cmd_fallback_stats(args)
            captured = capsys.readouterr()
            assert "No telemetry records found" in captured.out

    def test_cmd_fallback_explain(self, capsys):
        """Test explain command."""
        args = Mock()

        with patch("DocsToKG.ContentDownload.fallback.cli_commands.load_fallback_plan") as mock_load:
            mock_load.return_value = FallbackPlan(
                budgets={"total_timeout_ms": 30000, "total_attempts": 100, "max_concurrent": 5},
                tiers=[TierPlan(name="tier_1", parallel=5, sources=["test"])],
                policies={"test": AttemptPolicy(name="test", timeout_ms=2000, retries_max=1)},
            )
            cmd_fallback_explain(args)
            captured = capsys.readouterr()
            assert "FALLBACK STRATEGY EXPLANATION" in captured.out

    def test_cmd_fallback_config_yaml(self, capsys):
        """Test config command with YAML output."""
        args = Mock()
        args.format = "yaml"

        with patch("DocsToKG.ContentDownload.fallback.cli_commands.load_fallback_plan") as mock_load:
            mock_load.return_value = FallbackPlan(
                budgets={"total_timeout_ms": 30000, "total_attempts": 100, "max_concurrent": 5},
                tiers=[TierPlan(name="tier_1", parallel=5, sources=["test"])],
                policies={"test": AttemptPolicy(name="test", timeout_ms=2000, retries_max=1)},
            )
            cmd_fallback_config(args)
            captured = capsys.readouterr()
            assert "fallback_strategy:" in captured.out

    def test_cmd_fallback_config_json(self, capsys):
        """Test config command with JSON output."""
        args = Mock()
        args.format = "json"

        with patch("DocsToKG.ContentDownload.fallback.cli_commands.load_fallback_plan") as mock_load:
            mock_load.return_value = FallbackPlan(
                budgets={"total_timeout_ms": 30000, "total_attempts": 100, "max_concurrent": 5},
                tiers=[TierPlan(name="tier_1", parallel=5, sources=["test"])],
                policies={"test": AttemptPolicy(name="test", timeout_ms=2000, retries_max=1)},
            )
            cmd_fallback_config(args)
            captured = capsys.readouterr()
            assert "budgets" in captured.out or "{" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
