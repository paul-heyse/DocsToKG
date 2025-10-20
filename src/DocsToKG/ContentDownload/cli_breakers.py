# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.cli_breakers",
#   "purpose": "CLI subcommands for circuit breaker inspection and operational control",
#   "sections": [
#     {
#       "id": "install-breaker-cli",
#       "name": "install_breaker_cli",
#       "anchor": "function-install-breaker-cli",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-show",
#       "name": "_cmd_show",
#       "anchor": "function-cmd-show",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-open",
#       "name": "_cmd_open",
#       "anchor": "function-cmd-open",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-close",
#       "name": "_cmd_close",
#       "anchor": "function-cmd-close",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""CLI subcommands for circuit breaker inspection and operational control.

This module provides operator-friendly commands to:
- Display current breaker state and cooldown timers
- Force-open a breaker for maintenance windows
- Manually reset a breaker after maintenance

Typical Usage:
    # Install into argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    install_breaker_cli(subparsers, make_registry_factory)

    # Use from CLI
    cli breaker show
    cli breaker show --host api.example.com
    cli breaker open api.example.com --seconds 600 --reason maintenance
    cli breaker close api.example.com
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Iterable, Tuple

from DocsToKG.ContentDownload.breakers import BreakerRegistry, RequestRole

# Type for dependency injection: factory returns (registry, known_hosts)
RegistryFactory = Callable[[], Tuple[BreakerRegistry, Iterable[str]]]


def install_breaker_cli(
    subparsers: argparse._SubParsersAction, make_registry: RegistryFactory
) -> None:
    """Install breaker CLI subcommands into argument parser.

    Adds:
      - breaker show [--host HOST] - list breaker states
      - breaker open <host> --seconds N [--reason REASON] - force-open
      - breaker close <host> - reset breaker

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Parent subparsers action from ArgumentParser.add_subparsers()
    make_registry : RegistryFactory
        Callable that returns (BreakerRegistry, known_hosts_list)
    """
    p = subparsers.add_parser("breaker", help="Inspect and operate circuit breakers")
    sp = p.add_subparsers(dest="breaker_cmd", required=True)

    # show
    ps = sp.add_parser("show", help="Display breaker state and cooldown timers")
    ps.add_argument("--host", help="Filter to single host (optional)")
    ps.set_defaults(func=_cmd_show, make_registry=make_registry)

    # open
    po = sp.add_parser("open", help="Force-open a breaker for maintenance")
    po.add_argument("host", help="Hostname to open (lowercase punycode)")
    po.add_argument("--seconds", type=int, required=True, help="Cooldown duration in seconds")
    po.add_argument("--reason", default="cli-open", help="Reason tag (default: cli-open)")
    po.set_defaults(func=_cmd_open, make_registry=make_registry)

    # close
    pc = sp.add_parser("close", help="Reset breaker and clear cooldown overrides")
    pc.add_argument("host", help="Hostname to close (lowercase punycode)")
    pc.set_defaults(func=_cmd_close, make_registry=make_registry)


def _cmd_show(args: argparse.Namespace) -> int:
    """Show current breaker state and cooldown timers.

    Output format (text table):
    HOST               STATE           COOLDOWN_REMAIN_MS
    api.example.com    closed          0
    broken.host        open            45230
    """
    reg, known_hosts = args.make_registry()

    rows = []
    for h in known_hosts:
        if args.host and args.host.lower() != h:
            continue
        state = reg.current_state(h)
        remain_ms = reg.cooldown_remaining_ms(h) or 0
        rows.append((h, state, remain_ms))

    if not rows:
        print("No breakers to show.")
        return 0

    # Pretty print
    print(f"{'HOST':40} {'STATE':20} {'COOLDOWN_REMAIN_MS':>18}")
    print("-" * 80)
    for h, s, r in sorted(rows):
        print(f"{h:40} {s:20} {r:>18}")
    return 0


def _cmd_open(args: argparse.Namespace) -> int:
    """Force-open a breaker for maintenance window.

    Sets a cooldown override that will block pre-flight checks until the timeout expires.
    """
    reg, _ = args.make_registry()
    deadline = time.monotonic() + max(0, int(args.seconds))
    host = args.host.lower()

    reg.cooldowns.set_until(host, deadline, reason=args.reason)
    print(f"Opened {host} for {args.seconds}s (reason={args.reason})")
    return 0


def _cmd_close(args: argparse.Namespace) -> int:
    """Reset breaker and clear cooldown overrides.

    Clears the cooldown override and resets the pybreaker's failure counters.
    """
    reg, _ = args.make_registry()
    host = args.host.lower()

    reg.cooldowns.clear(host)

    # Also reset pybreaker counters by recording a success
    try:
        reg.on_success(host, role=RequestRole.METADATA)
    except Exception:
        pass  # Best-effort; some registries may not support on_success outside request context

    print(f"Closed {host}")
    return 0
