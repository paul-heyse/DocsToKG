# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.cli_fallback",
#   "purpose": "CLI commands for fallback strategy operational control.",
#   "sections": [
#     {
#       "id": "format-plan-table",
#       "name": "format_plan_table",
#       "anchor": "function-format-plan-table",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-fallback-plan",
#       "name": "cmd_fallback_plan",
#       "anchor": "function-cmd-fallback-plan",
#       "kind": "function"
#     },
#     {
#       "id": "mock-adapter",
#       "name": "_mock_adapter",
#       "anchor": "function-mock-adapter",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-fallback-dryrun",
#       "name": "cmd_fallback_dryrun",
#       "anchor": "function-cmd-fallback-dryrun",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-fallback-tune",
#       "name": "cmd_fallback_tune",
#       "anchor": "function-cmd-fallback-tune",
#       "kind": "function"
#     },
#     {
#       "id": "register-fallback-commands",
#       "name": "register_fallback_commands",
#       "anchor": "function-register-fallback-commands",
#       "kind": "function"
#     },
#     {
#       "id": "handle-fallback-command",
#       "name": "_handle_fallback_command",
#       "anchor": "function-handle-fallback-command",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""CLI commands for fallback strategy operational control.

This module provides commands for:
  - Inspecting effective configuration
  - Dry-running resolution strategy
  - Analyzing telemetry and suggesting tuning

Commands:
  python -m DocsToKG.ContentDownload.cli fallback plan
    → Show effective configuration (after merging YAML/env/CLI)

  python -m DocsToKG.ContentDownload.cli fallback dryrun
    → Simulate resolution with mock adapters (no actual fetches)

  python -m DocsToKG.ContentDownload.cli fallback tune
    → Analyze telemetry, suggest configuration improvements
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from DocsToKG.ContentDownload.fallback.loader import load_fallback_plan
from DocsToKG.ContentDownload.fallback.orchestrator import FallbackOrchestrator
from DocsToKG.ContentDownload.fallback.types import AttemptResult

logger = logging.getLogger(__name__)


def format_plan_table(plan: Any) -> str:
    """Format FallbackPlan as readable text table.

    Args:
        plan: FallbackPlan object

    Returns:
        Formatted table string
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("FALLBACK RESOLUTION PLAN")
    lines.append("=" * 80)

    # Budgets section
    lines.append("\nBUDGETS (Global Constraints):")
    lines.append("-" * 80)
    budgets = plan.budgets
    lines.append(f"  Total Timeout:       {budgets['total_timeout_ms']:,} ms")
    lines.append(f"  Total Attempts:      {budgets['total_attempts']}")
    lines.append(f"  Max Concurrent:      {budgets['max_concurrent']}")
    lines.append(f"  Per-Source Timeout:  {budgets['per_source_timeout_ms']:,} ms")

    # Gates section
    lines.append("\nHEALTH GATES:")
    lines.append("-" * 80)
    gates = plan.gates
    lines.append(f"  Skip if Breaker Open: {gates.get('skip_if_breaker_open', True)}")
    lines.append(f"  Offline Behavior:     {gates.get('offline_behavior', 'metadata_only')}")
    lines.append(f"  Skip if Rate Wait >:  {gates.get('skip_if_rate_wait_exceeds_ms', 5000)} ms")

    # Tiers section
    lines.append("\nTIERS (Sequential Resolution Stages):")
    lines.append("-" * 80)
    for i, tier in enumerate(plan.tiers, 1):
        policies = [plan.policies[s] for s in tier.sources]
        timeouts = [p.timeout_ms for p in policies]
        lines.append(
            f"  Tier {i}: {tier.name:20} "
            f"parallel={tier.parallel} sources={len(tier.sources)} "
            f"timeouts={timeouts}"
        )
        for j, source in enumerate(tier.sources, 1):
            policy = plan.policies[source]
            lines.append(
                f"    {j}. {source:28} "
                f"timeout={policy.timeout_ms}ms retries={policy.retries_max} "
                f"robots={policy.robots_respect}"
            )

    # Policies summary
    lines.append("\nPOLICY SUMMARY:")
    lines.append("-" * 80)
    for source in sorted(plan.policies.keys()):
        policy = plan.policies[source]
        lines.append(
            f"  {source:30} {policy.timeout_ms:5}ms "
            f"retries={policy.retries_max} robots={policy.robots_respect}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def cmd_fallback_plan(args: Optional[Dict[str, Any]] = None) -> None:
    """Display effective fallback plan configuration.

    Shows the configuration after merging YAML, environment, and CLI sources.

    Args:
        args: Optional CLI arguments (unused, kept for consistency)
    """
    try:
        plan = load_fallback_plan()
        print(format_plan_table(plan))
        print("\n✅ Configuration loaded successfully")
        return None
    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ Error loading plan: {e}", flush=True)
        return None


def _mock_adapter(policy: Any, context: Dict[str, Any]) -> AttemptResult:
    """Mock adapter for dryrun testing.

    Args:
        policy: AttemptPolicy
        context: Request context

    Returns:
        Mock AttemptResult
    """
    # Simulate different outcomes based on source
    source = context.get("source", "unknown")

    # Unpaywall and arXiv succeed 80% of the time
    if source in ("unpaywall_pdf", "arxiv_pdf"):
        if context.get("attempt_num", 0) < 2:
            return AttemptResult(
                outcome="success",
                reason="mock_success",
                elapsed_ms=1000,
                url=f"https://example.org/{source}.pdf",
                status=200,
                host="example.org",
                meta={"source": source},
            )

    # Landing scrape succeeds 50% of time
    if source == "landing_scrape_pdf":
        if context.get("attempt_num", 0) == 0:
            return AttemptResult(
                outcome="no_pdf",
                reason="mock_no_pdf",
                elapsed_ms=2000,
                status=200,
                meta={"source": source},
            )

    # Wayback is last resort
    if source == "wayback_pdf":
        return AttemptResult(
            outcome="no_pdf",
            reason="mock_exhausted",
            elapsed_ms=5000,
            status=200,
            meta={"source": source},
        )

    return AttemptResult(
        outcome="no_pdf",
        reason="mock_default",
        elapsed_ms=1000,
        status=200,
        meta={"source": source},
    )


def cmd_fallback_dryrun(args: Optional[Dict[str, Any]] = None) -> None:
    """Dry-run the fallback resolution strategy with mock adapters.

    Simulates resolution without making actual network calls.

    Args:
        args: Optional CLI arguments (unused)
    """
    try:
        plan = load_fallback_plan()

        # Create mock adapters
        adapters = {}
        all_sources = set()
        for tier in plan.tiers:
            all_sources.update(tier.sources)

        for source in all_sources:
            adapters[source] = _mock_adapter

        # Create orchestrator
        orchestrator = FallbackOrchestrator(
            plan=plan,
            breaker=None,  # No breaker for dryrun
            rate_limiter=None,  # No rate limiter for dryrun
            clients={},  # No clients needed for dryrun
            telemetry=None,  # No telemetry for dryrun
            logger=logger,
        )

        # Run dryrun
        print("\n" + "=" * 80)
        print("FALLBACK STRATEGY DRY-RUN")
        print("=" * 80)
        print(f"\nConfiguration: {len(plan.tiers)} tiers, {len(all_sources)} sources")
        print(
            f"Budgets: {plan.budgets['total_timeout_ms']}ms, {plan.budgets['total_attempts']} attempts\n"
        )

        context = {
            "work_id": "dry-run-001",
            "artifact_id": "dry-run-artifact",
            "doi": "10.1234/example",
            "url": "https://example.org",
            "offline": False,
        }

        result = orchestrator.resolve_pdf(context=context, adapters=adapters)  # type: ignore[arg-type]

        print("\n📊 DRY-RUN RESULT:")
        print(f"  Outcome: {result.outcome}")
        print(f"  Reason: {result.reason}")
        print(f"  Elapsed: {result.elapsed_ms}ms")
        if result.url:
            print(f"  URL: {result.url}")

        print("\n✅ Dry-run completed successfully")
        return None

    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ Error running dryrun: {e}", flush=True)
        return None


def cmd_fallback_tune(args: Optional[Dict[str, Any]] = None) -> None:
    """Analyze telemetry and suggest configuration tuning.

    This is a placeholder for Phase 7 telemetry integration.
    Currently shows what telemetry would be analyzed.

    Args:
        args: Optional CLI arguments (unused)
    """
    print("\n" + "=" * 80)
    print("FALLBACK STRATEGY AUTO-TUNING")
    print("=" * 80)
    print("\n⚠️  Auto-tuning requires telemetry data from recent runs.")
    print("   This will be implemented in Phase 7 (Telemetry Integration).\n")
    print("   Telemetry analysis will track:")
    print("     • Success rates per source")
    print("     • Average response times")
    print("     • HTTP status distributions (429s, 5xx, etc.)")
    print("     • Budget utilization")
    print("\n   Suggestions will optimize for:")
    print("     • Throughput (reduce timeout per source)")
    print("     • Reliability (increase attempts, extend timeout)")
    print("     • Resource usage (reduce concurrency, skip slow sources)")
    print("\n✅ Tuning infrastructure ready for Phase 7\n")
    return None


# Command registry for integration with main CLI
FALLBACK_COMMANDS = {
    "plan": cmd_fallback_plan,
    "dryrun": cmd_fallback_dryrun,
    "tune": cmd_fallback_tune,
}


def register_fallback_commands(subparsers: Any) -> None:
    """Register fallback subcommands with argparse.

    Args:
        subparsers: argparse subparsers object
    """
    # Main fallback subcommand
    fallback_parser = subparsers.add_parser(
        "fallback",
        help="Fallback strategy operational commands",
    )

    fallback_subparsers = fallback_parser.add_subparsers(dest="fallback_cmd")

    # fallback plan command
    fallback_subparsers.add_parser(
        "plan",
        help="Show effective fallback configuration",
    )

    # fallback dryrun command
    fallback_subparsers.add_parser(
        "dryrun",
        help="Dry-run resolution strategy with mock adapters",
    )

    # fallback tune command
    fallback_subparsers.add_parser(
        "tune",
        help="Analyze telemetry and suggest tuning",
    )

    # Set default handler
    fallback_parser.set_defaults(func=_handle_fallback_command)


def _handle_fallback_command(args: Any) -> None:
    """Route fallback subcommand to appropriate handler.

    Args:
        args: Parsed CLI arguments
    """
    cmd = getattr(args, "fallback_cmd", None)
    if cmd and cmd in FALLBACK_COMMANDS:
        FALLBACK_COMMANDS[cmd](args)
    else:
        print("❌ Unknown fallback command")


__all__ = [
    "cmd_fallback_plan",
    "cmd_fallback_dryrun",
    "cmd_fallback_tune",
    "register_fallback_commands",
    "format_plan_table",
]
