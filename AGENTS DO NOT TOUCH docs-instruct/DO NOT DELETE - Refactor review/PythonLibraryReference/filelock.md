Below is a practical, **end‑to‑end reference to the `filelock` library**—aimed at replacing home‑grown “single‑process at a time” logic in AI/automation stacks. It covers the mental model, core API, sync/async usage, platform differences, observability, migration tips, and real‑world footguns.

> **What is `filelock`?**
> `filelock` provides a **cross‑platform file‑based mutex** with a tiny API (`FileLock`, `SoftFileLock`, `Timeout`) and both sync **and async** variants. It is widely used to coordinate work across *processes* on the same machine (or on a shared volume, with caveats). The docs recommend creating a **separate `.lock` file** instead of trying to lock the file you’re writing. ([FileLock][1])

---

## Install, version & compatibility

```bash
pip install filelock
```

* **Latest (Oct 20, 2025):** **3.20.0**, released **Oct 8, 2025**. **Requires Python ≥ 3.10** and is “Production/Stable.” License: **Unlicense**. ([PyPI][2])

---

## Mental model

A `FileLock` (or its async cousin) protects a **critical section** by acquiring an exclusive OS‑level lock on a *separate* lock file (e.g., `resource.lock`) and releasing it afterward. Acquisition can block, time out, or return immediately (non‑blocking). Locks are **re‑entrant** (a process/thread can acquire the same lock multiple times; a counter tracks nested acquisitions). ([FileLock][1])

> **Hard vs Soft locks**
>
> * **`FileLock`**: platform‑specific *hard* lock (Unix: `fcntl.flock`, Windows: `msvcrt.locking`).
> * **`SoftFileLock`**: only checks the **existence** of the lock file (super portable, but more prone to stale locks if a process crashes). The docs suggest using `FileLock` when all instances run on the same platform and `SoftFileLock` otherwise. ([FileLock][3])

---

## Core API (sync)

```python
from filelock import FileLock, SoftFileLock, Timeout

lock = FileLock("data.index.lock", timeout=10)   # default is blocking; timeout in seconds
with lock:                                       # re-entrant, context manager
    build_or_read_index()
```

**Acquire / release details (sync):**

* `BaseFileLock.acquire(timeout=None, poll_interval=0.05, *, blocking=None)`

  * `timeout < 0` → block forever; `timeout == 0` → single try (non‑blocking); `None` → use the lock’s default.
  * `blocking=False` returns immediately (raises `Timeout` if not acquired) and **takes precedence** over `timeout`.
  * `poll_interval` controls how often we retry. The old `poll_intervall` spelling is **deprecated** (renamed in 3.4.0). ([FileLock][3])
* `release(force=False)` lowers the **lock counter** and unlocks when it reaches zero. The **lock file is not auto‑deleted**. `force=True` ignores the counter (use sparingly). ([FileLock][3])
* Exceptions: `Timeout(lock_file)` if we couldn’t acquire within the deadline. ([FileLock][3])

**Platform behavior:**

* **Unix**: `UnixFileLock` uses `fcntl.flock()` on the lock file.
* **Windows**: `WindowsFileLock` uses `msvcrt.locking()`.
* `FileLock` resolves to the appropriate one for your platform. ([FileLock][3])

**Thread behavior (important):**

* Locks are **thread‑local by default** since **3.11.0**; you can pass `thread_local=False` to share the same lock instance across threads (still re‑entrant). This became a flag you can toggle in **3.12.0**. ([FileLock][1])

**Permissions & paths:**

* `mode=` sets the **lockfile permissions** (default 0o644); explicit modes were added in **3.10.0**.
* Path‑like objects (`pathlib.Path`) are accepted since **3.3.2**.
* Acquiring in a **read‑only** or **missing** directory raises (behavior added in **3.2.0**). ([FileLock][4])

**Decorator usage:** `BaseFileLock` is a `ContextDecorator`, so you can do:

```python
lock = FileLock("cache.lock")
@lock
def critical_step(): ...
```

No extra code; the function runs inside the lock. ([FileLock][4])

---

## Async variants

Use the **async** classes when coordinating work in `asyncio` code:

```python
import asyncio
from filelock import AsyncUnixFileLock, Timeout

async def rebuild():
    lock = AsyncUnixFileLock("models.lock", timeout=15)
    async with await lock.acquire():  # returns an AsyncAcquireReturnProxy
        await refresh_models()

asyncio.run(rebuild())
```

* Async types: `AsyncUnixFileLock` / `AsyncWindowsFileLock` / `AsyncSoftFileLock`. They expose `async acquire(..., blocking=None)` which returns an **async context‑manager** proxy, and `async release(...)`. Under the hood they integrate with the event loop (see `loop`, `run_in_executor`, `executor` accessors). ([FileLock][3])
* Async support is explicitly documented since **3.3.0** (“Document asyncio support”). If you find older notes saying “no asyncio support,” prefer the **current API reference**. ([FileLock][4])

---

## Usage patterns you’ll likely replace in an AI/automation stack

### 1) **Single‑writer gate** (prevent duplicate work)

```python
from filelock import FileLock, Timeout
lock = FileLock("embeddings.lock", timeout=30)

try:
    with lock:
        compute_embeddings_once()
except Timeout:
    # Someone else is doing it; read results or back off
    use_existing_embeddings()
```

* Prefer **separate `.lock` files** (e.g., `embeddings.lock`) instead of locking the target file. ([FileLock][1])

### 2) **Non‑blocking “try once”** (agent can skip or enqueue)

```python
from filelock import FileLock, Timeout
lock = FileLock("periodic-sync.lock", timeout=0)
try:
    with lock:
        run_periodic_sync()
except Timeout:
    schedule_later()
```

`timeout=0` (or `blocking=False`) yields one attempt only. ([FileLock][1])

### 3) **Per‑resource lock helper**

```python
from contextlib import contextmanager
from filelock import FileLock

@contextmanager
def resource_lock(name: str):
    lock = FileLock(f"/var/locks/{name}.lock", timeout=10)
    with lock:
        yield
```

### 4) **Async critical sections**

```python
from filelock import AsyncUnixFileLock, Timeout

async def try_refresh():
    lock = AsyncUnixFileLock("refresh.lock", timeout=5)
    try:
        async with await lock.acquire():
            await refresh()
    except Timeout:
        return False
    return True
```

(Async proxy pattern per the API: `await lock.acquire()` returns an **async** context manager.) ([FileLock][3])

---

## Observability & logging

* Library logs at **DEBUG** under the `filelock` logger. Silence or raise the level as needed:
  `logging.getLogger("filelock").setLevel(logging.INFO)`. ([FileLock][1])
* Properties & helpers: `lock.is_locked`, `lock.lock_counter`, `lock.timeout`, `lock.mode`, etc., let you introspect state; `Timeout` includes the `lock_file` path for error messages. ([FileLock][3])

---

## Platform & filesystem caveats (read this!)

* **Do not lock the file you’re writing**; lock a **separate** `.lock` file. This avoids partial‑write interleaving and simplifies cross‑platform behavior. ([FileLock][1])
* **Network filesystems (NFS/SMB)**: OS‑level locking support varies by server, client, and mount options; some Unix filesystems **do not support `flock`** at all—`filelock` added explicit checks for this in `3.10.5`. If you must coordinate across machines, test your storage stack—or use a **distributed lock** (e.g., Redis) instead. ([FileLock][4])

---

## API cheat‑sheet (sync + async)

* **Main classes**: `FileLock` (alias to Unix/Windows), `SoftFileLock`; async: `AsyncUnixFileLock`, `AsyncWindowsFileLock`, `AsyncSoftFileLock`. ([FileLock][3])
* **Acquire**: `acquire(timeout=None, poll_interval=0.05, *, blocking=None)` → returns a **proxy context manager** (use `with lock.acquire(): ...` or just `with lock:`). Async version returns an **async** proxy (use `async with await lock.acquire(): ...`). ([FileLock][3])
* **Release**: `release(force=False)` (sync) / `await release(force=False)` (async). Does **not** delete the lock file. ([FileLock][3])
* **Errors**: `Timeout` (subclass of `TimeoutError`). ([FileLock][3])
* **Options**: `mode=0o600` (tighter perms), `thread_local=False` (share across threads), `blocking=` (instant fail vs wait), `poll_interval=` (retry cadence). ([FileLock][3])

---

## Migration guide (custom code → `filelock`)

| If your custom code does this…                            | Use this in `filelock`                                                                                                                         |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| “Block until only one process is in the critical section” | `with FileLock("X.lock"):`                                                                                                                     |
| “Try once and bail”                                       | `FileLock("X.lock", timeout=0)` or `acquire(blocking=False)`                                                                                   |
| “Wait up to T seconds then give up”                       | `FileLock("X.lock", timeout=T)` or `with lock.acquire(timeout=T):`                                                                             |
| “Different behavior per platform”                         | Prefer `FileLock` (auto‑selects Unix/Windows hard lock) or `SoftFileLock` if you need ultra‑portable presence‑based semantics. ([FileLock][3]) |
| “Lock across threads”                                     | Default is **thread‑local** (since 3.11.0). Pass `thread_local=False` to share across threads. ([FileLock][1])                                 |
| “I want explicit permissions on the lock file”            | Use `mode=...` (added 3.10.0). ([FileLock][4])                                                                                                 |
| “I’m on a read‑only/missing directory”                    | Acquire will **raise** (since 3.2.0); ensure a writable directory exists for locks. ([FileLock][4])                                            |
| “Decorate a function to run under a lock”                 | `@FileLock("X.lock")` (context‑decorator). ([FileLock][4])                                                                                     |
| “Async code needs the same semantics”                     | `AsyncUnixFileLock` / `AsyncWindowsFileLock`, `async with await lock.acquire(): ...` (documented since 3.3.0). ([FileLock][3])                 |

---

## Footguns & best practices

* **Don’t rely on GC** to hold a lock. If you call `FileLock(...).acquire()` and don’t keep a reference, the lock may be released when the temporary is garbage‑collected. Always use `with lock:` or assign to a variable. ([FileLock][1])
* **Prefer `FileLock` over `SoftFileLock`** unless you truly need presence‑based semantics (e.g., very heterogeneous environments). Soft locks can leave **stale lock files** on crashes; you may need a cleanup policy. ([FileLock][1])
* **Use a separate `.lock` file** (again): it’s the project’s explicit guidance. ([FileLock][1])
* **Network storage caution**: Validate on your stack (NFS version, mount options) or use a service‑backed lock; some filesystems don’t support `flock`. ([FileLock][4])
* **Time‑bounded locks**: pick sensible `timeout` and `poll_interval` values for your SLOs; avoid spinning too fast. ([FileLock][3])
* **Logging noise**: raise the `filelock` logger level if DEBUG spam bothers you. ([FileLock][1])

---

## “Production defaults” for an AI agent

```python
from pathlib import Path
from filelock import FileLock, Timeout

LOCK_DIR = Path("/var/locks")  # must exist & be writable
LOCK_DIR.mkdir(parents=True, exist_ok=True)

def run_once(name: str, timeout: float = 20.0):
    lock = FileLock(str(LOCK_DIR / f"{name}.lock"), timeout=timeout)
    try:
        with lock:
            return do_work(name)
    except Timeout:
        # Someone else is working; caller can return stale data or reschedule
        return None
```

* Place locks in a **writable, dedicated directory**.
* If you need **cross‑threads re‑entrance**, set `thread_local=False`. ([FileLock][1])
* For **async** flows, swap in `AsyncUnixFileLock` (or Windows variant) and wrap your section with `async with await lock.acquire(): ...`. ([FileLock][3])

---

## Recent changes you should know

* **3.11.0–3.12.0**: locks became **thread‑local** by default, and then made **configurable** via `thread_local=`. ([FileLock][4])
* **3.10.0**: `mode=` for explicit lockfile permissions. ([FileLock][4])
* **3.4.0**: `poll_interval` spelling added; old `poll_intervall` kept but deprecated. ([FileLock][4])
* **3.3.0**: type hints; docs note **async** support. ([FileLock][4])
* **3.2.0**: acquiring in read‑only/missing folders raises explicitly. ([FileLock][4])

---

## Authoritative sources

* **Docs – Home & Tutorial** (what the library is, examples, `.lock` guidance, re‑entrance, GC warning, logging, thread‑local behavior). ([FileLock][1])
* **API Reference** (sync + async classes, acquire/release signatures, platform notes, properties). ([FileLock][3])
* **Changelog** (thread‑local, mode, poll interval rename, async support note). ([FileLock][4])
* **PyPI** (current version, release date, Python requirement). ([PyPI][2])

---

If you share how your current locking utilities behave (timeouts, retries, network storage, whether you need cross‑thread reentrance or async), I’ll translate them 1‑for‑1 into a small `filelock` wrapper module with your defaults and safety checks.

[1]: https://py-filelock.readthedocs.io/ "filelock"
[2]: https://pypi.org/project/filelock/ "filelock · PyPI"
[3]: https://py-filelock.readthedocs.io/en/latest/api.html "API - filelock"
[4]: https://py-filelock.readthedocs.io/en/latest/changelog.html "Changelog - filelock"
