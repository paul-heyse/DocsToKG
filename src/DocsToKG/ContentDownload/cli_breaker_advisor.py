# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.cli_breaker_advisor",
#   "purpose": "CLI subcommand for breaker advisor (metrics analysis and tuning suggestions)",
#   "sections": [
#     {
#       "id": "install-breaker-advisor-cli",
#       "name": "install_breaker_advisor_cli",
#       "anchor": "function-install-breaker-advisor-cli",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-advise",
#       "name": "_cmd_advise",
#       "anchor": "function-cmd-advise",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""CLI subcommand for breaker advisor (noisy host detection and tuning suggestions).

This module provides the `breaker-advise` subcommand to:
- Analyze telemetry metrics in a sliding window
- Detect noisy hosts and failure patterns
- Suggest or enforce safe breaker/rate-limiter tuning

Typical Usage:
    # Install into argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    install_breaker_advisor_cli(subparsers, make_registry, telemetry_db_path)

    # Use from CLI
    cli breaker-advise --window-s 600
    cli breaker-advise --window-s 600 --enforce
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from pathlib import Path

from DocsToKG.ContentDownload.breaker_advisor import BreakerAdvisor
from DocsToKG.ContentDownload.breaker_autotune import BreakerAutoTuner
from DocsToKG.ContentDownload.breakers import BreakerRegistry

RegistryFactory = Callable[[], tuple[BreakerRegistry, Iterable[str]]]


def install_breaker_advisor_cli(
    subparsers: argparse._SubParsersAction,
    make_registry: RegistryFactory,
    telemetry_db_path: Path,
) -> None:
    """Install breaker-advise CLI subcommand.

    Adds:
      - breaker-advise [--window-s N] [--enforce] - analyze metrics and suggest/apply tuning

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Parent subparsers action from ArgumentParser.add_subparsers()
    make_registry : RegistryFactory
        Callable that returns (BreakerRegistry, known_hosts_list)
    telemetry_db_path : Path
        Path to telemetry SQLite database for metrics analysis
    """
    p = subparsers.add_parser("breaker-advise", help="Analyze metrics and suggest breaker tuning")
    p.add_argument(
        "--window-s",
        type=int,
        default=600,
        help="Analysis window in seconds (default 600)",
    )
    p.add_argument(
        "--enforce",
        action="store_true",
        help="Apply safe adjustments in-memory (instead of just suggesting)",
    )
    p.set_defaults(
        func=_cmd_advise,
        make_registry=make_registry,
        telemetry_db_path=telemetry_db_path,
    )


def _cmd_advise(args: argparse.Namespace) -> int:
    """Analyze telemetry metrics and produce tuning advice.

    Metrics analyzed:
    - 429 ratio per host (suggests rate limiter adjustment)
    - 5xx burst frequency (suggests lower fail_max)
    - Retry-After distribution (suggests reset_timeout_s)
    - Half-open trial outcomes (suggests success_threshold)

    With --enforce, applies safe bounded adjustments (e.g., ±25% rate multiplier).
    Without --enforce, prints suggestions only.
    """
    reg, _ = args.make_registry()
    advisor = BreakerAdvisor(str(args.telemetry_db_path), window_s=args.window_s)
    tuner = BreakerAutoTuner(registry=reg)

    # Get suggestions (safe, read-only)
    plans = tuner.enforce(advisor) if args.enforce else tuner.suggest(advisor)

    if not plans:
        print("✓ No tuning suggestions at this time.")
        return 0

    print(f"Circuit Breaker Tuning Recommendations ({args.window_s}s window)")
    print("=" * 80)
    for plan in plans:
        print(f"\n[{plan.host}]")
        for change in plan.changes:
            print(f"  • {change}")

    if args.enforce:
        print("\n✓ Suggestions applied in-memory. Changes will persist for this session.")
    else:
        print("\nℹ  Run with --enforce to apply these suggestions in-memory.")

    return 0
