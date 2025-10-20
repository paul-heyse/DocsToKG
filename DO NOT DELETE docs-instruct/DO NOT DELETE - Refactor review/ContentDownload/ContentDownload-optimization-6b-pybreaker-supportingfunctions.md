Awesome — here are two drop-in components to finish the breaker story:

1. **`SQLiteCooldownStore`** (cross-process cooldowns that respect Retry-After & rolling-window opens), and
2. an **argparse mixin** that adds `breaker show|open|close` subcommands for ops.

I’ve also included a short “best-in-class” checklist derived from the official `pybreaker` guidance (naming, excludes, success_threshold, Redis storage, etc.).

---

# 1) Cross-process cooldowns: `SQLiteCooldownStore`

> Paste as `src/DocsToKG/ContentDownload/sqlite_cooldown_store.py`. It implements the `CooldownStore` protocol from your `breakers.py` skeleton and is safe for multi-process use on a shared local filesystem.

```python
# File: src/DocsToKG/ContentDownload/sqlite_cooldown_store.py
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, ContextManager

# Optional lock: if you shipped locks.sqlite_lock(), we’ll use it; else no-op.
try:
    from DocsToKG.ContentDownload.locks import sqlite_lock as default_lock
except Exception:  # pragma: no cover
    @contextmanager
    def default_lock(_: Path):
        yield

# CooldownStore protocol lives in breakers.py; this class implements it.
#   def get_until(host: str) -> Optional[float]                 # returns monotonic deadline
#   def set_until(host: str, until_monotonic: float, reason: str) -> None
#   def clear(host: str) -> None

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=4000;
CREATE TABLE IF NOT EXISTS breaker_cooldowns (
    host TEXT PRIMARY KEY,
    until_wall REAL NOT NULL,    -- UTC epoch seconds
    reason TEXT,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cd_until ON breaker_cooldowns(until_wall);
"""

@dataclass
class SQLiteCooldownStore:
    """
    Cross-process cooldown store backed by SQLite. Stores wall-clock deadlines,
    converts to monotonic on read so callers are immune to wall-clock drift.

    Parameters
    ----------
    db_path : database file location (directories are created if needed)
    lock_ctx : context manager for file locking (defaults to project locks.sqlite_lock)
    now_wall : callable returning time.time()
    now_mono : callable returning time.monotonic()
    """
    db_path: Path
    lock_ctx: Callable[[Path], ContextManager] = default_lock
    now_wall: Callable[[], float] = time.time
    now_mono: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,         # autocommit mode
            check_same_thread=False,
            detect_types=0,
        )
        c = self._conn.cursor()
        for stmt in _DDL.strip().split(";\n"):
            if stmt.strip():
                c.execute(stmt)

    # ---- CooldownStore API ----------------------------------------------------

    def get_until(self, host: str) -> Optional[float]:
        """
        Return monotonic deadline for 'host' if a cooldown exists and is in the future.
        """
        now_w = self.now_wall()
        now_m = self.now_mono()
        with self.lock_ctx(self.db_path):
            row = self._conn.execute(
                "SELECT until_wall FROM breaker_cooldowns WHERE host=?",
                (host,),
            ).fetchone()
        if not row:
            return None
        until_wall = float(row[0])
        if until_wall <= now_w:
            # expired → clean it up (best-effort)
            self.clear(host)
            return None
        # Convert wall → monotonic so callers compare apples-to-apples.
        return now_m + max(0.0, until_wall - now_w)

    def set_until(self, host: str, until_monotonic: float, reason: str) -> None:
        """
        Write/update cooldown for 'host' with a monotonic deadline.
        We store wall time to share across processes reliably.
        """
        now_w = self.now_wall()
        now_m = self.now_mono()
        # Convert monotonic → wall
        until_wall = now_w + max(0.0, until_monotonic - now_m)
        with self.lock_ctx(self.db_path):
            self._conn.execute(
                """
                INSERT INTO breaker_cooldowns(host, until_wall, reason, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(host) DO UPDATE SET
                    until_wall=excluded.until_wall,
                    reason=excluded.reason,
                    updated_at=excluded.updated_at
                """,
                (host, float(until_wall), str(reason)[:128], now_w),
            )

    def clear(self, host: str) -> None:
        with self.lock_ctx(self.db_path):
            self._conn.execute("DELETE FROM breaker_cooldowns WHERE host=?", (host,))

    # ---- Optional maintenance helpers ----------------------------------------

    def prune_expired(self) -> int:
        """
        Delete rows that expired > 1s ago. Returns rows removed.
        """
        now_w = self.now_wall()
        with self.lock_ctx(self.db_path):
            cur = self._conn.execute(
                "DELETE FROM breaker_cooldowns WHERE until_wall < ?",
                (now_w - 1.0,),
            )
            return cur.rowcount or 0

    def close(self) -> None:  # pragma: no cover
        try:
            self._conn.commit()
        finally:
            self._conn.close()
```

### Wire it into the registry

In your `networking` startup (or wherever you build the `BreakerRegistry`):

```python
from pathlib import Path
from DocsToKG.ContentDownload.breakers_loader import load_breaker_config
from DocsToKG.ContentDownload.breakers import BreakerRegistry
from DocsToKG.ContentDownload.sqlite_cooldown_store import SQLiteCooldownStore

cfg = load_breaker_config(yaml_path=os.getenv("DOCSTOKG_BREAKERS_YAML"), env=os.environ, ...)
store = SQLiteCooldownStore(Path(run_dir) / "telemetry/breakers.sqlite")
registry = BreakerRegistry(cfg, cooldown_store=store, listener_factory=make_breaker_listener, now_monotonic=time.monotonic)
```

> If you later scale cross-host, keep this **SQLite store** for a single machine and add a **RedisCooldownStore** for fleet-wide sharing (same wall↔monotonic conversion).

---

# 2) Argparse mixin: `breaker show|open|close`

> Paste as `src/DocsToKG/ContentDownload/cli_breakers.py`. This adds a `breaker` subcommand you can plug into your main CLI. It expects a **factory** that returns a live `BreakerRegistry` and a list of **known hosts** (from your loaded config).

```python
# File: src/DocsToKG/ContentDownload/cli_breakers.py
from __future__ import annotations

import argparse
import time
from typing import Callable, Iterable, Optional, Tuple

from DocsToKG.ContentDownload.breakers import BreakerRegistry, RequestRole, BreakerOpenError

# Types for dependency injection
RegistryFactory = Callable[[], Tuple[BreakerRegistry, Iterable[str]]]

def install_breaker_cli(subparsers: argparse._SubParsersAction, make_registry: RegistryFactory) -> None:
    """
    Adds:
      docstokg breaker show [--host HOST]
      docstokg breaker open <host> --seconds N [--reason REASON]
      docstokg breaker close <host>
    """
    p = subparsers.add_parser("breaker", help="Inspect and operate circuit breakers")
    sp = p.add_subparsers(dest="breaker_cmd", required=True)

    # show
    ps = sp.add_parser("show", help="List breaker state")
    ps.add_argument("--host", help="Filter to a single host")
    ps.set_defaults(func=_cmd_show, make_registry=make_registry)

    # open
    po = sp.add_parser("open", help="Force-open a host for N seconds (cooldown override)")
    po.add_argument("host", help="Host to open (lowercase punycode)")
    po.add_argument("--seconds", type=int, required=True, help="Cooldown seconds")
    po.add_argument("--reason", default="cli-open", help="Reason tag")
    po.set_defaults(func=_cmd_open, make_registry=make_registry)

    # close
    pc = sp.add_parser("close", help="Clear host cooldown and reset counters")
    pc.add_argument("host", help="Host to close (lowercase punycode)")
    pc.set_defaults(func=_cmd_close, make_registry=make_registry)

def _cmd_show(args: argparse.Namespace) -> int:
    reg, known_hosts = args.make_registry()
    now_m = time.monotonic()
    now_w = time.time()

    rows = []
    for h in known_hosts:
        if args.host and args.host != h:
            continue
        state = reg.current_state(h)             # "host:closed" or "host:open,resolver:..."
        until = reg.cooldowns.get_until(h)       # monotonic deadline or None
        remain_ms = int(max(0.0, (until - now_m) * 1000)) if until else 0
        rows.append((h, state, remain_ms))

    if not rows:
        print("No breakers to show.")
        return 0

    # Pretty print
    print(f"{'HOST':40} {'STATE':24} {'COOLDOWN_REMAIN_MS':>18}")
    for h, s, r in sorted(rows):
        print(f"{h:40} {s:24} {r:>18}")
    return 0

def _cmd_open(args: argparse.Namespace) -> int:
    reg, _ = args.make_registry()
    deadline = time.monotonic() + max(0, int(args.seconds))
    reg.cooldowns.set_until(args.host, deadline, reason=args.reason)
    print(f"Opened {args.host} for {args.seconds}s (reason={args.reason})")
    return 0

def _cmd_close(args: argparse.Namespace) -> int:
    reg, _ = args.make_registry()
    reg.cooldowns.clear(args.host)
    # Also reset pybreaker counters to avoid immediate re-open on first call
    try:
        # Touch the breaker by recording a 'success' (lightweight reset hook)
        reg.on_success(args.host, role=RequestRole.METADATA)  # role doesn’t matter for clearing cooldown
    except Exception:
        pass
    print(f"Closed {args.host}")
    return 0
```

### Wiring the mixin into your main CLI

```python
# In your top-level CLI:
from DocsToKG.ContentDownload.cli_breakers import install_breaker_cli
from DocsToKG.ContentDownload.breakers_loader import load_breaker_config
from DocsToKG.ContentDownload.sqlite_cooldown_store import SQLiteCooldownStore
from DocsToKG.ContentDownload.breakers import BreakerRegistry

def _make_registry() -> tuple[BreakerRegistry, list[str]]:
    cfg = load_breaker_config(os.getenv("DOCSTOKG_BREAKERS_YAML"), env=os.environ, ...)
    store = SQLiteCooldownStore(Path(run_dir)/"telemetry/breakers.sqlite")
    reg = BreakerRegistry(cfg, cooldown_store=store, listener_factory=make_breaker_listener)
    known_hosts = sorted(cfg.hosts.keys())
    return reg, known_hosts

# argparse setup
subparsers = parser.add_subparsers(dest="cmd", required=True)
install_breaker_cli(subparsers, _make_registry)
```

---

# 3) Best-in-class breaker tips (from pybreaker docs)

To reduce flapping and keep observability crisp, consider these optional improvements; they are first-class features in `pybreaker` and slot cleanly into your registry.

1. **Name your breakers** (use `name=`) so listener logs/metrics are human-friendly (e.g., `name="api.crossref.org"`).
2. **Exclude business errors** from counting as failures (e.g., HTTP 4xx validations) using **`exclude=[...]`** predicates (e.g., exclude `httpx.HTTPStatusError` with `<500`).
3. **Require multiple successes to truly “recover”** by setting **`success_threshold=2..3`** — in Half-Open, pybreaker will wait for *N* consecutive successes before closing.
4. **Reraise the original exception when opening** using **`throw_new_error_on_trip=False`** if you want the upstream error class to bubble into metrics.
5. **Share breaker state across workers** if you run many processes/hosts, by using **`CircuitRedisStorage`** with a **unique `namespace`** per external dependency; keep the **cooldown store** (from this plan) for Retry-After and rolling-window overrides on top. (Avoid `decode_responses=True` on the Redis client.)
6. **Keep breakers process-global** (singletons) so they accumulate state across all calls.
7. **Attach a listener** (you already did) to emit **`state_change`**, **`failure`**, and **`success`** events to your telemetry sink for dashboards/alerts.

All of the above are explicitly supported in the pybreaker README and API (counters, state methods, listeners, and storages).

---

# 4) Quick validation checklist

* `breaker show` prints state & remaining cooldown; `breaker open/close` work.
* A 429 with `Retry-After` opens an override for that duration (capped).
* ≥N failures in W seconds triggers a rolling-window manual open.
* Half-open allows only `trial_calls` per role, with a tiny jitter (≤150 ms).
* Cross-process workers respect opens via the SQLite store.
* Telemetry shows **opens/hour**, **success-after-open**, **time saved**, and **open reason mix**.

If you want, I can also provide a tiny `RedisCooldownStore` (mirrors this SQLite version) and a one-page admin “Runbook” for breaker operations (how to identify noisy hosts, when to force-open/close, and how to tune `fail_max` vs `reset_timeout` vs `success_threshold`).
