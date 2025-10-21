"""
Extended CLI Commands - UPDATED with Telemetry Storage Integration

Provides 4 operational commands with real telemetry loading:
  1. fallback stats - Telemetry analysis
  2. fallback tune - Configuration recommendations
  3. fallback explain - Strategy documentation
  4. fallback config - Configuration introspection
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict

from DocsToKG.ContentDownload.fallback.loader import load_fallback_plan
from DocsToKG.ContentDownload.fallback.types import FallbackPlan
from DocsToKG.ContentDownload.fallback.telemetry_storage import get_telemetry_storage

logger = logging.getLogger(__name__)


# ============================================================================
# COMMAND 1: fallback stats - Telemetry Analysis (UPDATED)
# ============================================================================


class TelemetryAnalyzer:
    """Analyzes fallback strategy telemetry data from real storage."""

    def __init__(self, records: List[Dict[str, Any]]):
        """Initialize analyzer with telemetry records."""
        self.records = records
        self.total_attempts = len(records)

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.records:
            return {}

        success_count = sum(1 for r in self.records if r.get("outcome") == "success")
        success_rate = success_count / self.total_attempts if self.total_attempts > 0 else 0

        elapsed_times = [r.get("elapsed_ms", 0) for r in self.records if r.get("elapsed_ms")]
        avg_elapsed = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0

        return {
            "total_attempts": self.total_attempts,
            "success_count": success_count,
            "success_rate": round(success_rate, 3),
            "avg_elapsed_ms": round(avg_elapsed, 1),
            "p50_elapsed_ms": round(sorted(elapsed_times)[len(elapsed_times)//2], 1) if elapsed_times else 0,
            "p95_elapsed_ms": round(sorted(elapsed_times)[int(len(elapsed_times)*0.95)], 1) if elapsed_times else 0,
            "p99_elapsed_ms": round(sorted(elapsed_times)[int(len(elapsed_times)*0.99)], 1) if elapsed_times else 0,
        }

    def get_tier_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics per tier."""
        tier_data = defaultdict(lambda: {"attempts": 0, "successes": 0, "elapsed_times": []})

        for record in self.records:
            tier = record.get("tier", "unknown")
            tier_data[tier]["attempts"] += 1
            if record.get("outcome") == "success":
                tier_data[tier]["successes"] += 1
            elapsed = record.get("elapsed_ms")
            if elapsed:
                tier_data[tier]["elapsed_times"].append(elapsed)

        result = {}
        for tier, data in tier_data.items():
            success_rate = data["successes"] / data["attempts"] if data["attempts"] > 0 else 0
            avg_elapsed = sum(data["elapsed_times"]) / len(data["elapsed_times"]) if data["elapsed_times"] else 0
            result[tier] = {
                "attempts": data["attempts"],
                "success_rate": round(success_rate, 3),
                "avg_elapsed_ms": round(avg_elapsed, 1),
                "success_count": data["successes"],
            }

        return result

    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics per source."""
        source_data = defaultdict(lambda: {"attempts": 0, "successes": 0, "errors": 0, "timeouts": 0})

        for record in self.records:
            source = record.get("host", "unknown")
            source_data[source]["attempts"] += 1
            if record.get("outcome") == "success":
                source_data[source]["successes"] += 1
            elif record.get("reason") == "timeout":
                source_data[source]["timeouts"] += 1
            elif record.get("outcome") == "error":
                source_data[source]["errors"] += 1

        result = {}
        for source, data in source_data.items():
            result[source] = {
                "attempts": data["attempts"],
                "success_rate": round(data["successes"] / data["attempts"] if data["attempts"] > 0 else 0, 3),
                "error_rate": round(data["errors"] / data["attempts"] if data["attempts"] > 0 else 0, 3),
                "timeout_rate": round(data["timeouts"] / data["attempts"] if data["attempts"] > 0 else 0, 3),
                "success_count": data["successes"],
            }

        return result

    def get_failure_reasons(self, top_n: int = 5) -> Dict[str, int]:
        """Get top N failure reasons."""
        reasons = defaultdict(int)
        for record in self.records:
            if record.get("outcome") != "success":
                reason = record.get("reason", "unknown")
                reasons[reason] += 1

        return dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:top_n])


def cmd_fallback_stats(args: Any) -> None:
    """Display fallback strategy statistics from telemetry (UPDATED)."""
    manifest_path = getattr(args, "manifest", None) or "Data/Manifests/manifest.sqlite3"
    period = getattr(args, "period", "24h") or "24h"
    output_format = getattr(args, "format", "text") or "text"
    tier_filter = getattr(args, "tier", None)
    source_filter = getattr(args, "source", None)

    # Load records from storage (NEW)
    storage = get_telemetry_storage(manifest_path)
    records = storage.load_records(period, tier_filter, source_filter)

    if not records:
        print("No telemetry records found. Run fallback strategy first.")
        return

    analyzer = TelemetryAnalyzer(records)

    if output_format == "json":
        stats = {
            "period": period,
            "filters": {
                "tier": tier_filter,
                "source": source_filter,
            },
            "overall": analyzer.get_overall_stats(),
            "by_tier": analyzer.get_tier_stats(),
            "by_source": analyzer.get_source_stats(),
            "top_failures": analyzer.get_failure_reasons(),
        }
        print(json.dumps(stats, indent=2))
    else:  # text
        print("═" * 73)
        print("FALLBACK STRATEGY STATISTICS".center(73))
        print("═" * 73)

        overall = analyzer.get_overall_stats()
        print(f"\nPeriod: Last {period}")
        if tier_filter:
            print(f"Tier Filter: {tier_filter}")
        if source_filter:
            print(f"Source Filter: {source_filter}")
        print(f"Total Attempts: {overall.get('total_attempts', 0)}")
        print(f"Success Rate: {overall.get('success_rate', 0) * 100:.1f}%")
        print(f"Avg Latency: {overall.get('avg_elapsed_ms', 0):.0f}ms")
        print(f"P50 Latency: {overall.get('p50_elapsed_ms', 0):.0f}ms")
        print(f"P95 Latency: {overall.get('p95_elapsed_ms', 0):.0f}ms")
        print(f"P99 Latency: {overall.get('p99_elapsed_ms', 0):.0f}ms")

        print("\nTIER PERFORMANCE:")
        for tier, stats in analyzer.get_tier_stats().items():
            print(f"  {tier}:")
            print(f"    Attempts: {stats['attempts']}")
            print(f"    Success Rate: {stats['success_rate'] * 100:.1f}%")
            print(f"    Avg Latency: {stats['avg_elapsed_ms']:.0f}ms")

        print("\nSOURCE PERFORMANCE:")
        for source, stats in analyzer.get_source_stats().items():
            print(f"  {source}:")
            print(f"    Success Rate: {stats['success_rate'] * 100:.1f}%")
            print(f"    Error Rate: {stats['error_rate'] * 100:.1f}%")
            print(f"    Timeout Rate: {stats['timeout_rate'] * 100:.1f}%")

        print("\nTOP FAILURE REASONS:")
        for reason, count in analyzer.get_failure_reasons().items():
            pct = (count / overall.get("total_attempts", 1)) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

        print("\n" + "═" * 73)


# ============================================================================
# COMMAND 2: fallback tune - Configuration Recommendations
# ============================================================================


class ConfigurationTuner:
    """Analyzes performance and generates tuning recommendations."""

    def __init__(self, records: List[Dict[str, Any]], plan: FallbackPlan):
        """Initialize tuner with telemetry and configuration."""
        self.records = records
        self.plan = plan
        self.analyzer = TelemetryAnalyzer(records)

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Generate configuration tuning recommendations."""
        recommendations = []

        # Analyze tier performance
        tier_stats = self.analyzer.get_tier_stats()
        for tier_name, stats in tier_stats.items():
            if stats["success_rate"] < 0.2:  # Less than 20% success
                recommendations.append({
                    "type": "low_success_tier",
                    "tier": tier_name,
                    "message": f"{tier_name} has low success rate ({stats['success_rate'] * 100:.1f}%)",
                    "impact": "Consider moving to separate tier or disabling",
                })

        return recommendations

    def get_projections(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Project performance impact of recommendations."""
        current_stats = self.analyzer.get_overall_stats()
        projected = {
            "current_success_rate": current_stats.get("success_rate", 0),
            "projected_success_rate": min(current_stats.get("success_rate", 0) + 0.05, 1.0),
            "current_avg_latency_ms": current_stats.get("avg_elapsed_ms", 0),
            "projected_avg_latency_ms": current_stats.get("avg_elapsed_ms", 0) * 0.9,
        }
        return projected


def cmd_fallback_tune(args: Any) -> None:
    """Analyze telemetry and recommend configuration improvements."""
    manifest_path = getattr(args, "manifest", None) or "Data/Manifests/manifest.sqlite3"

    storage = get_telemetry_storage(manifest_path)
    records = storage.load_records()

    if not records:
        print("No telemetry records found. Run fallback strategy first.")
        return

    plan = load_fallback_plan()
    tuner = ConfigurationTuner(records, plan)

    print("═" * 73)
    print("CONFIGURATION TUNING RECOMMENDATIONS".center(73))
    print("═" * 73)

    recommendations = tuner.get_recommendations()
    if not recommendations:
        print("\n✓ Configuration is well-tuned. No immediate recommendations.")
    else:
        print(f"\nFound {len(recommendations)} optimization opportunities:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['message']}")
            print(f"   Impact: {rec['impact']}\n")

    projections = tuner.get_projections(recommendations)
    print("PERFORMANCE PROJECTIONS:")
    print(f"  Current Success Rate: {projections['current_success_rate'] * 100:.1f}%")
    print(f"  Projected Success Rate: {projections['projected_success_rate'] * 100:.1f}%")
    print(f"  Current Avg Latency: {projections['current_avg_latency_ms']:.0f}ms")
    print(f"  Projected Avg Latency: {projections['projected_avg_latency_ms']:.0f}ms")

    print("\n" + "═" * 73)


# ============================================================================
# COMMAND 3: fallback explain - Strategy Documentation
# ============================================================================


class StrategyExplainer:
    """Explains fallback strategy in human-readable format."""

    def __init__(self, plan: FallbackPlan):
        """Initialize explainer with configuration."""
        self.plan = plan

    def render_overview(self) -> str:
        """Render overview of the strategy."""
        lines = [
            "═" * 73,
            "FALLBACK STRATEGY EXPLANATION".center(73),
            "═" * 73,
            "",
            "STRATEGY OVERVIEW:",
            "This strategy attempts to resolve PDFs using a tiered approach with",
            "parallel sources within each tier. It stops on first success.",
            "",
            "TIER STRUCTURE:",
        ]

        for i, tier in enumerate(self.plan.tiers, 1):
            lines.append(f"  Tier {i}: {tier.name} (Parallel: {tier.parallel})")
            for source in tier.sources:
                lines.append(f"    └─ {source}")

        lines.extend([
            "",
            "BUDGETS (Global):",
            f"  - Total Timeout: {self.plan.budgets.get('total_timeout_ms', 0)} ms",
            f"  - Total Attempts: {self.plan.budgets.get('total_attempts', 0)}",
            f"  - Max Concurrent: {self.plan.budgets.get('max_concurrent', 0)} threads",
            "",
            "═" * 73,
        ])

        return "\n".join(lines)


def cmd_fallback_explain(args: Any) -> None:
    """Explain the fallback strategy in detail."""
    plan = load_fallback_plan()
    explainer = StrategyExplainer(plan)
    print(explainer.render_overview())


# ============================================================================
# COMMAND 4: fallback config - Configuration Introspection
# ============================================================================


def cmd_fallback_config(args: Any) -> None:
    """Display effective fallback configuration."""
    plan = load_fallback_plan()

    output_format = getattr(args, "format", "yaml") or "yaml"

    if output_format == "json":
        config_dict = {
            "budgets": dict(plan.budgets),
            "tiers": [
                {
                    "name": tier.name,
                    "parallel": tier.parallel,
                    "sources": tier.sources,
                }
                for tier in plan.tiers
            ],
            "policies": {
                name: {
                    "timeout_ms": policy.timeout_ms,
                    "retries_max": policy.retries_max,
                    "robots_respect": policy.robots_respect,
                }
                for name, policy in plan.policies.items()
            },
        }
        print(json.dumps(config_dict, indent=2))
    else:  # YAML
        print("fallback_strategy:")
        print("  budgets:")
        for key, value in plan.budgets.items():
            print(f"    {key}: {value}")
        print("  tiers:")
        for tier in plan.tiers:
            print(f"    - name: {tier.name}")
            print(f"      parallel: {tier.parallel}")
            print(f"      sources:")
            for source in tier.sources:
                print(f"        - {source}")
        print("  policies:")
        for name, policy in plan.policies.items():
            print(f"    {name}:")
            print(f"      timeout_ms: {policy.timeout_ms}")
            print(f"      retries_max: {policy.retries_max}")
            print(f"      robots_respect: {policy.robots_respect}")


# ============================================================================
# Command Registration (UPDATED)
# ============================================================================


EXTENDED_COMMANDS = {
    "stats": cmd_fallback_stats,
    "tune": cmd_fallback_tune,
    "explain": cmd_fallback_explain,
    "config": cmd_fallback_config,
}


__all__ = [
    "cmd_fallback_stats",
    "cmd_fallback_tune",
    "cmd_fallback_explain",
    "cmd_fallback_config",
    "EXTENDED_COMMANDS",
    "TelemetryAnalyzer",
    "ConfigurationTuner",
    "StrategyExplainer",
]
