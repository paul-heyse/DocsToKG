Below is a **refactorer‑friendly “complete reference”** to the `pyrate‑limiter` library, organized so an AI agent (or a human) can replace custom limiters with first‑class primitives from the library.

---

## 0) What pyrate‑limiter is (and isn’t)

* **Goal.** Enforce request quotas using a *leaky‑bucket family* algorithm: you record timestamps of recent events, periodically “leak” (evict) old ones, and admit or delay/deny new ones accordingly. This smooths traffic while respecting strict windows. ([Pyrate Limiter][1])
* **Core building blocks.** *Rates*, *Buckets*, *Clocks*, a *Limiter* facade, optional *Factory* for routing and background leaking, and storage backends (in‑memory, Redis, SQLite, Postgres, multi‑process). ([Pyrate Limiter][1])
* **Modern API (v3+).** The public surface centers around `Rate`, `Duration`, bucket classes, and `Limiter` (sync & async). Version 3 introduced decorator API changes, auto‑leaking via factories, and “bucket must be initialized before passing to Limiter”. ([Pyrate Limiter][2])

---

## 1) Data model and terminology

### `Duration` (time units)

Enum containing time constants **in milliseconds** (`SECOND = 1000`, `MINUTE`, `HOUR`, `DAY`, `WEEK`). You can also pass raw millisecond integers. ([Pyrate Limiter][3])

### `Rate`

`Rate(limit: int, interval: int|Duration)` – “allow **limit** events per **interval** (ms)”. Example: `Rate(500, Duration.HOUR)`. ([Pyrate Limiter][3])

**Ordering rule for multiple rates (multi‑window):** when you define several `Rate`s, you must order them:

1. by **interval/limit ascending order** of the intervals and limits (smallest to largest), and
2. by **limit/interval ratio descending order**.
   Use `validate_rate_list(rates)` to assert correctness. ([PyPI][4])

### `RateItem`

Library‑internal wrapper: `(name, timestamp, weight=1)`. `name` is your key (e.g., user id, “endpoint:x”), `timestamp` filled by a clock, `weight` lets a single acquisition consume multiple “slots”. ([Pyrate Limiter][3])

> **Weighted/atomic semantics.** A request with `weight=W` behaves like *W unit events* with the same timestamp. Insertions are atomic: either all W “sub‑events” fit or none do. Implementations check space before ingesting the item. This is important when replacing ad‑hoc token bursts. ([PyPI][4])

---

## 2) Buckets: how limits are enforced

Every bucket stores recent `RateItem` timestamps and answers:

* **`put(item)`** – record an event if capacity remains (returns bool).
* **`leak(now)`** – evict timestamps older than each rate’s window.
* **`waiting(item)`** – how long (ms) until this item would fit.
* **`peek(i)`**, **`count()`**, **`flush()`** – inspection/maintenance.
  These are defined by the abstract base `AbstractBucket`. ([Pyrate Limiter][5])

A background **Leaker** task removes stale entries on a fixed cadence so buckets don’t grow unbounded. Default leak interval is **10,000 ms**; factories schedule this for you. ([Pyrate Limiter][5])

### Built‑in bucket backends

* **InMemoryBucket** – fast, process‑local Python list. Clock can be `time.time` or `time.monotonic`. Good for single‑process jobs. (No persistence). ([Pyrate Limiter][6])
* **MultiprocessBucket** – in‑memory but shared via `multiprocessing.Manager.ListProxy` + a `multiprocessing.Lock`; provides an extra `limiter_lock()` for the Limiter to coordinate across processes. Useful when multiple worker processes must share quotas. ([Pyrate Limiter][7])
* **RedisBucket** – uses a **sorted set** (ZSET) per key (`name`) with scores = timestamps; available in both sync and async via a classmethod initializer (`await RedisBucket.init(...)` for asyncio). Good for cross‑process / cross‑host rate limits. ([PyPI][4])
* **SQLiteBucket** – persistent, single‑node store; also supports **file‑lock** mode for multiprocess safety. You can initialize from a file path (`init_from_file`), and there are helper factory functions. ([Pyrate Limiter][8])
* **PostgresBucket** – persistence with **psycopg3** integration (methods return sync values or awaitables depending on usage). Provides SQL queries for leak/peek/put; useful for distributed jobs where DB is the shared source of truth. ([Pyrate Limiter][9])
* **BucketAsyncWrapper** – wraps any sync bucket with async methods so the **event loop stays non‑blocking** (uses `asyncio` sleeps). Handy if you only have a sync bucket but run under asyncio. ([Pyrate Limiter][10])

**Implementation note.** Internally the library keeps timestamps sorted and uses helpers (e.g., a **binary search** to find the lower bound of a time window) to compute “size‑in‑window” quickly. That’s how “waiting” times are derived, and why periodic leaking matters for memory. ([Pyrate Limiter][11])

---

## 3) Clocks

A **Clock** timestamps items:

* **MonotonicClock** – default in‑process millisecond clock.
* **MonotonicAsyncClock** – async test clock.
* **PostgresClock** – reads time from the database (consistent across workers).
* **SQLiteClock** – used with SQLite backend.
  Use the built‑in clocks unless your refactor needs a custom time source. ([Pyrate Limiter][12])

---

## 4) The `Limiter` façade (your main entry point)

Construct a `Limiter` with either:

* a **bucket instance**, or
* a **BucketFactory**, or
* a single `Rate` or a list of `Rate`s (it will create a default in‑memory bucket).
  `Limiter` has a small buffer (`buffer_ms`, default **50 ms**) to absorb clock skew when delaying. ([Pyrate Limiter][13])

**Constructor options (conceptually):**

* `raise_when_fail: bool = True` – raise exceptions rather than returning `False`.
* `max_delay: Optional[int|Duration] = None` – the most you’re willing to wait (ms) before failing. You can set this globally or even per call; see below.
* `buffer_ms: int = 50` – extra safety margin when sleeping. ([PyPI][4])

**Core methods**

* `try_acquire(name="pyrate", *, weight=1, blocking=True, timeout=-1) -> bool`
  Sync attempt. If `blocking` and capacity is temporarily full, it may **sleep** (if `max_delay` permits). Note: **timeout is not implemented in the sync path**; prefer the async variant if you need a true timeout. ([Pyrate Limiter][13])

* `await try_acquire_async(name="pyrate", *, weight=1, blocking=True, timeout=-1) -> bool`
  Async variant that uses an **async lock** to avoid blocking the event loop. ([Pyrate Limiter][13])

* `as_decorator()` → a decorator factory that requires a **mapping function** turning the wrapped function’s args into a `name` (or `(name, weight)`); works for sync **and** async functions. Example below. ([PyPI][4])

* `buckets()` → list active buckets; `dispose(bucket)` → remove a bucket (and stop leaking tasks if none remain). ([PyPI][4])

**Delays vs. exceptions**

When over capacity you can either raise or wait:

* **Exceptions** (default): raises `BucketFullException` with `meta_info` (`name`, `weight`, and the `rate` hit). ([PyPI][4])
* **Delays**: allow waiting up to `max_delay` (globally or per call). If the computed wait exceeds your allowance, `LimiterDelayException` is raised (or `False` is returned if `raise_when_fail=False`). The actual sleep includes `buffer_ms`. ([PyPI][4])

---

## 5) Factories & background leaking

If you need **routing** (e.g., one bucket per user or endpoint) or you want to **spin up buckets dynamically**, implement a `BucketFactory`:

* **`wrap_item(name, weight)`** → stamp with a clock and return a `RateItem`.
* **`get(item)`** → return the bucket for that item (create on first seen; call `self.create(...)` so the factory schedules leaking).
  The factory owns a **Leaker** thread/task and exposes `schedule_leak` and `leak_interval` (10s default). ([Pyrate Limiter][5])

Utility factory functions:
`create_inmemory_limiter(...)`, `create_sqlite_bucket(...)`, `create_sqlite_limiter(...)`, and `init_global_limiter(...)` for ProcessPool initializers. These package **common patterns** and set sensible defaults (including file locking for SQLite). ([Pyrate Limiter][14])

---

## 6) Backends: when to use which

| Backend                | Use it when                                                                                  | Notes                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **InMemoryBucket**     | single process / simple jobs                                                                 | fastest, ephemeral. ([Pyrate Limiter][6])                                                                                |
| **MultiprocessBucket** | multiple Python **processes** on the same host                                               | uses `multiprocessing.Manager` ListProxy + Lock; provides `limiter_lock()` for extra coordination. ([Pyrate Limiter][7]) |
| **RedisBucket**        | many workers, many hosts                                                                     | sorted‑set based; sync/async `init(...)` initializers; good for distributed keys. ([PyPI][4])                            |
| **SQLiteBucket**       | single host but **persistent** across restarts; can coordinate different programs via a file | supports **file‑lock**; factory helpers available. ([Pyrate Limiter][14])                                                |
| **PostgresBucket**     | distributed persistence without Redis                                                        | psycopg3 backend, methods support sync/async; explicit SQL for leak/peek/put. ([Pyrate Limiter][9])                      |
| **BucketAsyncWrapper** | you only have a sync bucket but run under asyncio                                            | avoids blocking the event loop. ([Pyrate Limiter][10])                                                                   |

---

## 7) Concurrency and correctness notes

* **Threading vs. asyncio.** `Limiter` uses a thread‑safe lock for sync paths and an **async lock** for the async path to avoid blocking your event loop. Use `try_acquire_async` in coroutines. ([Pyrate Limiter][13])
* **Leaking cadence.** Background leaking happens periodically; you can still call `leak()` opportunistically (e.g., before capacity checks). The **Leaker** thread/task is owned by the factory and runs with a ~10s interval by default. ([Pyrate Limiter][5])
* **Multi‑process.** Prefer `MultiprocessBucket` (or `SQLiteBucket` with file locks) and initialize a global limiter in worker processes with `init_global_limiter`. Contention can inflate computed waits; set `retry_until_max_delay=True` so retries continue until your allowance expires. ([Pyrate Limiter][14])

---

## 8) Typical refactors (drop‑in patterns)

### A) Single‑window limiter (in‑memory)

```python
from pyrate_limiter import Limiter, Rate, Duration, BucketFullException

limiter = Limiter(Rate(5, Duration.SECOND * 2))  # 5 events / 2s

for i in range(6):
    try:
        limiter.try_acquire("user:42")
    except BucketFullException as e:
        print("blocked:", e.meta_info)
```

(Quickstart pattern.) ([PyPI][4])

### B) Multi‑window per key (Redis; sync path)

```python
from redis import Redis
from pyrate_limiter import Limiter, Rate, Duration, RedisBucket

rates = [Rate(10, Duration.SECOND), Rate(500, Duration.MINUTE)]
bucket = RedisBucket.init(rates, Redis(), "api-quota")
limiter = Limiter(bucket)

limiter.try_acquire("endpoint:/search", weight=2)
```

(Uses sorted‑set under the hood; `init` also supports redis.asyncio.) ([PyPI][4])

### C) Async client with delay allowance

```python
from redis.asyncio import Redis
from pyrate_limiter import Limiter, Rate, Duration, RedisBucket

rates = [Rate(5, Duration.SECOND)]
bucket = await RedisBucket.init(rates, Redis(), "chat")
limiter = Limiter(bucket, max_delay=Duration.SECOND)  # up to 1s delay

# Later…
ok = await limiter.try_acquire_async("room:123", weight=1)  # may sleep up to 1s
```

(Async initializer + non‑blocking sleeps.) ([PyPI][4])

### D) Use as a decorator (works for sync or async)

```python
from pyrate_limiter import Limiter, Rate, Duration

limiter = Limiter(Rate(3, Duration.SECOND))
dec = limiter.as_decorator()

def mapping(user_id: str):
    # Either return "name" or ("name", weight)
    return f"user:{user_id}"

@dec(mapping)
def fetch_profile(user_id: str):
    ...
```

(Decorator requires a mapping function.) ([PyPI][4])

### E) Per‑tenant buckets via a factory (dynamic routing + scheduled leak)

```python
from pyrate_limiter import BucketFactory, Rate, Duration, InMemoryBucket, Limiter, RateItem
from pyrate_limiter.clocks import MonotonicClock

rates = [Rate(100, Duration.MINUTE)]
clock = MonotonicClock()

class TenantFactory(BucketFactory):
    def __init__(self):
        self._by_tenant = {}

    def wrap_item(self, name: str, weight: int = 1) -> RateItem:
        return RateItem(name, clock.now(), weight)

    def get(self, item: RateItem):
        key = item.name  # tenant id
        if key not in self._by_tenant:
            bucket = self.create(InMemoryBucket, rates)  # schedules leak
            self._by_tenant[key] = bucket
        return self._by_tenant[key]

limiter = Limiter(TenantFactory())
limiter.try_acquire("tenant:acme")
```

(Factory provides `create(...)` and `schedule_leak` under the hood.) ([Pyrate Limiter][5])

### F) Multiprocessing (shared bucket)

Use `MultiprocessBucket` and initialize a global limiter in worker processes:

```python
from concurrent.futures import ProcessPoolExecutor
from pyrate_limiter import Rate, Duration, Limiter
from pyrate_limiter.buckets.mp_bucket import MultiprocessBucket
from pyrate_limiter.limiter_factory import init_global_limiter

rates = [Rate(50, Duration.SECOND)]
shared_bucket = MultiprocessBucket.init(rates)

def init():
    init_global_limiter(shared_bucket)  # called once per worker

def worker(i):
    from pyrate_limiter import limiter  # the global one set by init_global_limiter
    limiter.try_acquire("global")  # shared across processes
    ...

with ProcessPoolExecutor(initializer=init) as ex:
    ex.map(worker, range(1000))
```

(Factory helper designed for ProcessPool initializers.) ([Pyrate Limiter][14])

---

## 9) Exceptions you’ll see

* **`BucketFullException`** – thrown when capacity is exceeded. Includes `meta_info` with `name`, `weight`, and the `rate` hit. (If you set `raise_when_fail=False`, you’ll get `False` instead.) ([PyPI][4])
* **`LimiterDelayException`** – when the *required* delay exceeds your `max_delay` (per‑instance or per‑call). `meta_info` shows `max_delay` vs. `actual_delay`. ([PyPI][4])

---

## 10) Migration guide (from common custom/older code)

1. **Replace “RequestRate” with `Rate`.** Many v2 examples used `RequestRate`; v3’s API uses `Rate(limit, interval_ms)` and `Duration` values. ([Pyrate Limiter][2])
2. **Use `Limiter(Rate(...))` or supply a bucket.** Buckets must be *initialized before* handing to `Limiter` (v3). ([Pyrate Limiter][2])
3. **Decorator API changed.** Prefer `Limiter.as_decorator()` with a mapping function (instead of `ratelimit(...)` signature used in older docs). ([Pyrate Limiter][2])
4. **Switch any bespoke sliding‑window code** to a suitable bucket backend (in‑memory for single process, Redis for distributed, SQLite/Postgres for durable single‑host/distributed persistence). If you relied on “burst tokens,” adapt to **weights** to consume multiple slots atomically. ([PyPI][4])
5. **Background eviction.** Remove any custom cron/cleanup; factories schedule leaking automatically. Tune `leak_interval` only if you have tight memory/latency constraints. ([Pyrate Limiter][5])

---

## 11) Operational tips & pitfalls

* **Pick the right key (`name`).** Rate limits are enforced **per `name`**; encode what you need (per‑user, per‑endpoint, per‑org).
* **Tune `max_delay` consciously.** For strict SLAs use exceptions; for smoothing, allow bounded delays (`Limiter(..., max_delay=...)`). Remember `buffer_ms` adds a little padding to sleeps. ([PyPI][4])
* **Async correctness.** If your bucket isn’t async, wrap it with `BucketAsyncWrapper` and call `try_acquire_async` to avoid blocking the event loop. ([Pyrate Limiter][10])
* **Rate ordering matters.** If you add multi‑rates out of order, validation will fail—fix the ordering or compute rates systematically and call `validate_rate_list`. ([PyPI][4])
* **Python versions.** Supported on Python **3.8+**; v3 dropped 3.7. ([PyPI][4])

---

## 12) API lookup (quick links)

* **Limiter**: `try_acquire`, `try_acquire_async`, `as_decorator`, `buckets`, `dispose`, `buffer_ms`. ([Pyrate Limiter][13])
* **Rates & units**: `Rate`, `Duration`, `RateItem`. ([Pyrate Limiter][3])
* **Buckets**: `InMemoryBucket`, `MultiprocessBucket`, `RedisBucket`, `SQLiteBucket`, `PostgresBucket`, `BucketAsyncWrapper`. ([Pyrate Limiter][6])
* **Factories & helpers**: `create_inmemory_limiter`, `create_sqlite_bucket`, `create_sqlite_limiter`, `init_global_limiter`. ([Pyrate Limiter][14])
* **Abstracts** (for custom backends): `AbstractBucket`, `BucketFactory`, `Leaker`. ([Pyrate Limiter][5])

---

## 13) End‑to‑end example (practical refactor)

**Replace a home‑grown sliding window that uses a dict of lists per user** with persistent, multi‑window limits across processes:

```python
# Requirements:
# - 20 req/s and 2000 req/hour per user
# - multiple processes
# - survive restarts (persist recent windows)
# - non-blocking async usage

from redis.asyncio import Redis
from pyrate_limiter import Rate, Duration, Limiter, BucketFullException
from pyrate_limiter import RedisBucket  # async initializer

user_rates = [Rate(20, Duration.SECOND), Rate(2000, Duration.HOUR)]
redis = Redis.from_url("redis://localhost:6379/0")

# One distributed bucket for all users (name=user_id at acquire time)
bucket = await RedisBucket.init(user_rates, redis, "users:quota")
limiter = Limiter(bucket, max_delay=None, buffer_ms=50)  # hard-fail on overflow

async def guarded_call(user_id: str):
    try:
        await limiter.try_acquire_async(f"user:{user_id}", weight=1)
    except BucketFullException as e:
        # Return 429 with 'Retry-After' based on e.meta_info['rate'] if desired
        raise

# call guarded_call(...) wherever needed
```

* We chose **RedisBucket** to coordinate across processes/hosts.
* We use **async acquire** to keep the loop responsive.
* Multi‑window rates are **ordered** and validated by the bucket.
* Overflow triggers an exception rather than implicit sleeps, matching typical HTTP 429 behaviors. ([PyPI][4])

---

## 14) Why leaky‑bucket vs token‑bucket (for refactors)

`pyrate‑limiter`’s model keeps a log of recent timestamps and continuously evicts old ones (leaking). This enforces **strict windows** and yields smooth pacing. If your custom code implements **token‑bucket** with bursts, translate bursts via `weight` (one heavy acquisition) or by configuring multiple small windows (e.g., per‑second + per‑minute). ([PyPI][4])

---

## 15) Version awareness

* **Current (3.9.x)** adds MultiprocessBucket, delay tuning, lock simplifications; 3.5+ introduced Postgres; 3.8 added SQLite file lock; 3.0 brought the major API overhaul. ([Pyrate Limiter][2])
* Project is MIT-licensed and actively maintained; PyPI and conda-forge packages available. ([PyPI][4])

## 16) DocsToKG OntologyDownload integration

* `DocsToKG.OntologyDownload` now defaults to a pyrate-limiter manager. Each
  limiter key is normalised as `"{service or '_'}:{host or 'default'}"` and
  backed by `Rate` objects parsed from `per_host_rate_limit` and
  `rate_limits[...]` strings. Legacy token buckets have been removed.
* When `defaults.http.shared_rate_limit_dir` is provided, the limiter manager
  persists counters in `<shared_rate_limit_dir>/ratelimit.sqlite` via
  `SQLiteBucket.init_from_file(..., use_file_lock=True)`. Otherwise an
  `InMemoryBucket` keeps per-process quotas.
* `apply_retry_after` now returns the parsed delay (seconds) without mutating
  limiter state; callers sleep for that duration before the next `consume()`.
  Legacy mode retains the previous behaviour so downstream extensions continue
  to function.
* Tests can continue to inject custom throttles through
  `DownloadConfiguration.set_bucket_provider(...)`; the manager respects custom
  providers regardless of the selected mode.

---

### References (select)

* **Docs Index & Quickstart**: overview, features, examples. ([Pyrate Limiter][1])
* **Limiter API**: methods, sync/async notes, decorator, locks. ([Pyrate Limiter][13])
* **Rates & units**: `Rate`, `Duration`, `RateItem`. ([Pyrate Limiter][3])
* **Abstract bucket/factory/leaker**: lifecycle & background tasks. ([Pyrate Limiter][5])
* **Backends**: in‑memory, multi‑process, SQLite, Redis, Postgres, async wrapper. ([Pyrate Limiter][6])
* **Exceptions & delay semantics**: `BucketFullException`, `LimiterDelayException`, `max_delay`, `buffer_ms`. ([PyPI][4])
* **Changelog (v3 differences)**: migration touchpoints. ([Pyrate Limiter][2])

---

If you share a snippet of your custom limiter, I can **map it one‑to‑one** onto the classes above (picking the minimal bucket + limiter wrapper) and produce the exact migration patch.

[1]: https://pyratelimiter.readthedocs.io/ "pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[2]: https://pyratelimiter.readthedocs.io/en/latest/changelog.html "Changelog - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[3]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.abstracts.rate.html "pyrate_limiter.abstracts.rate module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[4]: https://pypi.org/project/pyrate-limiter/ "pyrate-limiter · PyPI"
[5]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.abstracts.bucket.html "pyrate_limiter.abstracts.bucket module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[6]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.buckets.in_memory_bucket.html?utm_source=chatgpt.com "pyrate_limiter.buckets.in_memory_bucket module"
[7]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.buckets.mp_bucket.html "pyrate_limiter.buckets.mp_bucket module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[8]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.buckets.sqlite_bucket.html "pyrate_limiter.buckets.sqlite_bucket module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[9]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.buckets.postgres.html "pyrate_limiter.buckets.postgres module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[10]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.abstracts.wrappers.html "pyrate_limiter.abstracts.wrappers module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[11]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.utils.html "pyrate_limiter.utils module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[12]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.clocks.html "pyrate_limiter.clocks module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[13]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.limiter.html "pyrate_limiter.limiter module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
[14]: https://pyratelimiter.readthedocs.io/en/latest/modules/pyrate_limiter.limiter_factory.html "pyrate_limiter.limiter_factory module - pyrate-limiter 3.9.0.post12.dev0+e6c52c8 documentation"
