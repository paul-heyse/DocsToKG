Amazing—since you shared the exact `acquire_lock` location/surface, here’s a **ready-to-apply diff pack** that:

1. Introduces a **lock-aware JSONL writer** (`JsonlWriter`) so telemetry & manifests never hand-roll locks again.
2. **Injects** that writer into `StageTelemetry` and removes any internal lock helpers.
3. **Deprecates** using `acquire_lock` for manifest/attempts (keeps it for general purpose), with a targeted warning when it’s used on `*.jsonl`.

If your tree already contains parts of this (e.g., you implemented a writer), apply only the relevant hunks.

---

## 1) Add a lock-aware writer to `io.py`

```diff
diff --git a/src/DocsToKG/DocParsing/io.py b/src/DocsToKG/DocParsing/io.py
index 0000000..0000000 100644
--- a/src/DocsToKG/DocParsing/io.py
+++ b/src/DocsToKG/DocParsing/io.py
@@
+from __future__ import annotations
+from pathlib import Path
+from typing import Iterable, Callable, Optional, Mapping
+from filelock import FileLock, Timeout
+import json
+import os
+import errno
+import time
+
+# NOTE: we assume jsonl_append_iter(...) already exists in this module and
+# performs a single-shot atomic append with fsync. We reuse it here.
+
+class JsonlWriter:
+    """
+    Lock-aware JSONL append writer.
+
+    Uses a per-file FileLock (path + '.lock') to serialize concurrent writers,
+    then delegates to jsonl_append_iter(..., atomic=True) for the actual write.
+    """
+    def __init__(self, lock_timeout_s: float = 120.0) -> None:
+        self.lock_timeout_s = float(lock_timeout_s)
+
+    def __call__(self, path: Path, rows: Iterable[Mapping]) -> None:
+        lock_path = Path(f"{path}.lock")
+        lock = FileLock(lock_path)
+        try:
+            lock.acquire(timeout=self.lock_timeout_s)
+            # Delegate to the existing atomic append path.
+            # jsonl_append_iter is expected to ensure parent dirs and fsync.
+            jsonl_append_iter(path, rows, atomic=True)
+        except Timeout as e:
+            raise TimeoutError(
+                f"Timed out acquiring lock {lock_path} after {self.lock_timeout_s}s "
+                f"while appending to {path}. Another writer may be stalled."
+            ) from e
+        finally:
+            try:
+                lock.release()
+            except Exception:
+                # Best-effort; FileLock cleans up on process exit as well.
+                pass
+
+# Default instance used by telemetry/manifest sinks
+DEFAULT_JSONL_WRITER: JsonlWriter = JsonlWriter()
+
@@
 # existing exports / helpers stay as-is
 # def jsonl_append_iter(path: Path, rows: Iterable[dict], *, atomic: bool = True) -> None: ...
```

> If your `jsonl_append_iter` lives elsewhere, adjust the import or move `JsonlWriter` into that module.

---

## 2) Inject the writer into `telemetry.py` and remove internal lock usage

```diff
diff --git a/src/DocsToKG/DocParsing/telemetry.py b/src/DocsToKG/DocParsing/telemetry.py
index 0000000..0000000 100644
--- a/src/DocsToKG/DocParsing/telemetry.py
+++ b/src/DocsToKG/DocParsing/telemetry.py
@@
-from pathlib import Path
-from typing import Optional, Mapping, Iterable
-from DocsToKG.DocParsing.core.concurrency import acquire_lock
-from DocsToKG.DocParsing.io import jsonl_append_iter
+from pathlib import Path
+from typing import Optional, Mapping, Iterable, Callable
+from DocsToKG.DocParsing.io import jsonl_append_iter, DEFAULT_JSONL_WRITER
@@
-class StageTelemetry:
-    def __init__(self, manifests_dir: Path, *, lock_timeout_s: float = 60.0) -> None:
-        self.manifests_dir = manifests_dir
-        self.lock_timeout_s = float(lock_timeout_s)
+class StageTelemetry:
+    def __init__(
+        self,
+        manifests_dir: Path,
+        *,
+        writer: Callable[[Path, Iterable[Mapping]], None] = DEFAULT_JSONL_WRITER,
+    ) -> None:
+        """
+        Telemetry sink for manifest/attempts.
+        All appends go through the injected `writer` (FileLock + atomic append).
+        """
+        self.manifests_dir = manifests_dir
+        self._writer = writer
@@
-    def _append_manifest_row(self, path: Path, row: Mapping) -> None:
-        # historical path used acquire_lock(...) around jsonl_append_iter
-        with acquire_lock(path, timeout=self.lock_timeout_s):
-            jsonl_append_iter(path, [row], atomic=True)
+    def _append_manifest_row(self, path: Path, row: Mapping) -> None:
+        # unified, lock-aware append path
+        self._writer(path, [row])
@@
-    def _append_attempt_row(self, path: Path, row: Mapping) -> None:
-        with acquire_lock(path, timeout=self.lock_timeout_s):
-            jsonl_append_iter(path, [row], atomic=True)
+    def _append_attempt_row(self, path: Path, row: Mapping) -> None:
+        self._writer(path, [row])
@@
-    # def _acquire_lock_for(self, path: Path):  # <- remove this helper entirely if present
-    #     ...
+    # NOTE: any legacy _acquire_lock_for helper can be removed; all locking flows through _writer
```

**Also update the constructor callsites** (usually wherever you build the telemetry sink) to pass nothing—the default writer is fine—or to pass a custom writer in tests.

---

## 3) Mark `acquire_lock` as **discouraged for manifest/attempts** (keep for general purpose)

```diff
diff --git a/src/DocsToKG/DocParsing/core/concurrency.py b/src/DocsToKG/DocParsing/core/concurrency.py
index 0000000..0000000 100644
--- a/src/DocsToKG/DocParsing/core/concurrency.py
+++ b/src/DocsToKG/DocParsing/core/concurrency.py
@@
-from __future__ import annotations
+from __future__ import annotations
+import warnings
 from pathlib import Path
 import contextlib
 from typing import Iterator
@@
-@contextlib.contextmanager
-def acquire_lock(path: Path, timeout: float = 60.0) -> Iterator[bool]:
-    """Acquire an advisory lock using :mod:`filelock` primitives."""
+@contextlib.contextmanager
+def acquire_lock(path: Path, timeout: float = 60.0) -> Iterator[bool]:
+    """
+    Acquire an advisory lock using :mod:`filelock`.
+
+    ⚠️ Note: This context manager is **not** used for manifest/attempts writes anymore.
+    For manifest/attempts JSONL appends, use the injected, lock-aware writer
+    (DEFAULT_JSONL_WRITER) in DocsToKG.DocParsing.io.
+    """
+    # Gentle nudge when someone tries to lock a manifest/attempts JSONL
+    if str(path).endswith(".jsonl"):
+        warnings.warn(
+            "acquire_lock(): discouraged for manifest/attempts JSONL writes; "
+            "use DEFAULT_JSONL_WRITER via StageTelemetry instead.",
+            DeprecationWarning,
+            stacklevel=2,
+        )
     lock = FileLock(f\"{path}.lock\")
     try:
         lock.acquire(timeout=timeout)
         yield True
     finally:
         try:
             lock.release()
         except Exception:
             pass
```

> We keep `acquire_lock` for **other** critical sections (e.g., reserving temp resources). The deprecation warning only fires on `*.jsonl` targets to steer authors toward the unified writer path.

---

## 4) (If present) remove direct lock use around manifest appends in stage code

Search for this pattern:

```python
with acquire_lock(manifest_path, timeout=...):
    jsonl_append_iter(manifest_path, [row], atomic=True)
```

Replace with:

```python
telemetry._append_manifest_row(manifest_path, row)
# (or if you don’t have the sink here)
from DocsToKG.DocParsing.io import DEFAULT_JSONL_WRITER
DEFAULT_JSONL_WRITER(manifest_path, [row])
```

If there are “attempts” appenders, apply the same replacement.

*Because file locations vary across repos, this step is easiest as a codemod sweep rather than a single diff hunk.*

---

## 5) (Optional) unify lifecycle entries via helpers

If you have duplicated code writing `__config__` / `__corpus__` rows, centralize it:

```diff
diff --git a/src/DocsToKG/DocParsing/logging.py b/src/DocsToKG/DocParsing/logging.py
index 0000000..0000000 100644
--- a/src/DocsToKG/DocParsing/logging.py
+++ b/src/DocsToKG/DocParsing/logging.py
@@
+def log_config(stage: str, telemetry: StageTelemetry, snapshot: Mapping) -> None:
+    row = {
+        "stage": stage,
+        "doc_id": "__config__",
+        "status": "success",
+        "duration_s": 0.0,
+        **snapshot,
+    }
+    path = resolve_manifest_path(stage, telemetry.manifests_dir)  # existing helper
+    telemetry._append_manifest_row(path, row)
```

> Wire this into each stage’s `before_stage` (Runner) or the stage entrypoint.

---

## 6) Quick sanity checks (after applying diffs)

* `git grep -n "_acquire_lock_for"` → **no results**
* `git grep -n "with acquire_lock("` → only **non-manifest** locations remain
* Run a smoke pipeline and verify:

  * Manifests are written.
  * No deprecation warnings except when a developer-authored manifest write still uses `acquire_lock`.

---

## 7) Minimal tests you can add now

* **Parallel append** (threads/processes) to the **same** manifest path using `StageTelemetry` → lines = `writers * rows_per_writer`, JSON parse OK.
* **Lock timeout**: create a lock and hold it; a writer with tiny `lock_timeout_s` raises `TimeoutError`.
* **Deprecation**: calling `acquire_lock(Path("x.jsonl"))` inside a test triggers `DeprecationWarning`.

---

### Why this is safe

* You already rely on `filelock`; this merely **routes** all manifest/attempts writes through one place.
* `jsonl_append_iter(..., atomic=True)` remains the single point of truth for how appends are performed (fsync, parent dirs, etc.).
* `acquire_lock` stays available for truly custom critical sections—but not for manifests.

---

If you’d like, I can also generate a **small codemod script** (Python or `sed`) to rewrite any lingering patterns:

```bash
# naive, safe-in-context example (review diffs!):
git grep -n 'with acquire_lock(.*manifest' -n \
| cut -d: -f1 | sort -u \
| xargs -I{} sed -n '1,120p' {}
```

…but usually the three diffs above and a quick grep sweep are all you’ll need.
