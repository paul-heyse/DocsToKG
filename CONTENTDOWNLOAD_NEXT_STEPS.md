# ContentDownload Closeout - Next Steps Implementation Guide

## Status Summary

**Completed:** Task 1 (Atomic Writer Integration) ✅

- download_execution.py uses atomic_write_stream
- Content-Length verification wired in
- Size-mismatch token added
- 304 Not Modified handling added
- cache-hit token added to api/types.py

**Next:** Tasks 2-6 (estimated 6-8 hours remaining)

---

## TASK 2: httpx + hishel Universal Wiring (1-2 hours)

### Step 2.1: Verify Token Refund in PerResolverHttpClient

**File:** `src/DocsToKG/ContentDownload/resolver_http_client.py`

**Current State:** The TokenBucket exists but likely doesn't refund on cache-hit.

**Required Change:** In `PerResolverHttpClient._request()` method, after receiving a response, check for hishel extensions and refund:

```python
# After: resp = self.c.request(method, url, **kw)
from_cache = bool(getattr(resp, "extensions", {}).get("from_cache"))
revalidated = bool(getattr(resp, "extensions", {}).get("revalidated"))

# If pure cache hit (not revalidated), refund the token
if from_cache and not revalidated:
    self.bucket.refund(1.0)
```

**Verification:**

```bash
grep -A 10 "def _request" src/DocsToKG/ContentDownload/resolver_http_client.py
```

### Step 2.2: Audit Resolver GETs

**Command:**

```bash
grep -R "\.get(" src/DocsToKG/ContentDownload/resolvers/*.py | grep -v "session\.get\|client\.get\|http.*\.get"
```

**Expected Result:** No direct `requests.get()` or unauthorized GETs. All should go through session/client.

**If Found Issues:**

- Replace `requests.get(...)` with `session.get(...)`
- Replace `self.session = requests.Session()` with injected session parameter
- File to check: `src/DocsToKG/ContentDownload/resolvers/__init__.py` (resolver factory)

### Step 2.3: Create Test for Cache-Hit Refund

**File:** `tests/content_download/test_cache_hit_refund.py` (new)

```python
"""Test that token bucket refunds on hishel cache-hit."""

def test_cache_hit_refunds_token():
    """When hishel serves from cache, token should be refunded."""
    # Setup: mock response with from_cache=True, revalidated=False
    resp = MockResponse(
        status_code=200,
        extensions={"from_cache": True, "revalidated": False}
    )

    # Get initial tokens
    initial_tokens = client.bucket.tokens

    # Make request (would consume token)
    # Response is from_cache, so token should be refunded

    # Verify: tokens should be same or higher than initial
    assert client.bucket.tokens >= initial_tokens - 0.1  # allow for float imprecision
```

### Step 2.4: Verify httpx is the Only HTTP Path

**Command:**

```bash
grep -R "import requests" src/DocsToKG/ContentDownload/*.py | grep -v "# type: ignore"
```

**Expected:** Zero matches (requests should only appear in comments/docstrings)

**If Found:** Move any remaining requests usage to httpx wrappers

---

## TASK 3: Policy Gates Integration (2-3 hours)

### Step 3.1: Create path_gate.py

**File:** `src/DocsToKG/ContentDownload/policy/path_gate.py` (new)

```python
"""Path safety validation gate.

Prevents:
- Path traversal attacks (.., ~, symlink loops)
- Escaping artifact root directory
- Permission issues
"""

import os
from pathlib import Path

def validate_path_safety(
    final_path: str,
    artifact_root: str,
) -> str:
    """
    Validate that final_path is safe to write to.

    Args:
        final_path: Target write path
        artifact_root: Safe root directory (PDFs, HTMLs, XMLs folder)

    Returns:
        Canonicalized final_path if safe

    Raises:
        ValueError: If path is unsafe (traversal, outside root, etc.)
    """
    # Resolve to absolute path
    abs_final = Path(final_path).resolve()
    abs_root = Path(artifact_root).resolve()

    # Check if final_path is under artifact_root
    try:
        abs_final.relative_to(abs_root)
    except ValueError:
        raise ValueError(f"Path escapes artifact root: {final_path}")

    # Check for write permission
    parent = abs_final.parent
    if not os.access(parent, os.W_OK):
        raise ValueError(f"No write permission: {parent}")

    return str(abs_final)
```

**Add Tests:** `tests/content_download/test_path_gate.py`

- ✅ Normal path accepted
- ✅ Traversal (..) rejected
- ✅ Symlink escape rejected
- ✅ Outside root rejected
- ✅ No write permission rejected

### Step 3.2: Integrate url_gate into stream_candidate_payload

**File:** `src/DocsToKG/ContentDownload/download_execution.py`

Add at the start of `stream_candidate_payload()`:

```python
from DocsToKG.ContentDownload.policy.url_gate import validate_url_security, PolicyError

def stream_candidate_payload(...):
    # ... existing code ...

    # Validate URL before streaming
    try:
        url = validate_url_security(plan.url)
    except PolicyError as e:
        _emit(telemetry, ..., status="policy-gate", reason="url-policy-violation",
              error=str(e))
        raise SkipDownload("policy-type", f"URL policy violation: {e}")
```

**Add Tests:** Policy gate emits attempt token on URL violation

### Step 3.3: Integrate path_gate into finalize_candidate_download

**File:** `src/DocsToKG/ContentDownload/download_execution.py`

```python
from DocsToKG.ContentDownload.policy.path_gate import validate_path_safety

def finalize_candidate_download(...):
    # ... existing 304 check ...

    # Validate final path
    try:
        final_path = validate_path_safety(final_path, artifact_root=os.getcwd())
    except ValueError as e:
        _emit(telemetry, ..., status="policy-gate", reason="path-policy-violation",
              error=str(e))
        raise DownloadError("download-error", f"Path policy violation: {e}")
```

---

## TASK 4: Config Unification (1 hour)

### Step 4.1: Audit Legacy DownloadConfig Usage

**Command:**

```bash
grep -R "from.*DownloadConfig\|import.*DownloadConfig" src/DocsToKG/ContentDownload/ | grep -v ContentDownloadConfig
```

**Record all locations**, then migrate each to use ContentDownloadConfig instead.

### Step 4.2: Example Migration

**Before:**

```python
from DocsToKG.ContentDownload.download import DownloadConfig
config = DownloadConfig(chunk_size=1024*1024)
```

**After:**

```python
from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
config = ContentDownloadConfig(download=DownloadPolicy(chunk_size=1024*1024))
```

### Step 4.3: Verify Frozen Dataclasses

**Command:**

```bash
grep -r "frozen=True" src/DocsToKG/ContentDownload/api/types.py \
       src/DocsToKG/ContentDownload/config/models.py
```

**Expected:** All dataclasses have `frozen=True`

---

## TASK 5: Pipeline Decommission (1 hour)

### Step 5.1: Identify Canonical Pipeline

**Current State:**

- `pipeline.py` - new canonical ResolverPipeline
- `download_pipeline.py` - alternative (possibly legacy)

**Command:**

```bash
wc -l src/DocsToKG/ContentDownload/pipeline.py src/DocsToKG/ContentDownload/download_pipeline.py
```

**Determine:** which one is v2-aligned (should be smaller, cleaner API)

### Step 5.2: Extract Data Contracts

If canonical pipeline uses external types, create `api/pipeline_contracts.py`:

```python
"""Stable contracts exported by canonical pipeline."""

from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    """Configuration passed to ResolverPipeline."""
    ...

# Re-export from api/__init__.py for backward compat
```

### Step 5.3: Update All Imports

**Command:**

```bash
grep -R "from.*import.*pipeline\|from.*pipeline.*import" src/DocsToKG/ContentDownload/
```

Ensure all point to canonical pipeline.

### Step 5.4: Delete Legacy

Once all imports updated:

```bash
rm src/DocsToKG/ContentDownload/pipeline_old.py  # or similar
```

---

## TASK 6: CI Guardrails (0.5 hours)

### Step 6.1: Create GitHub Workflow

**File:** `.github/workflows/guard-requests.yml` (new)

```yaml
name: Guard against direct requests usage

on:
  pull_request:
    paths:
      - "src/DocsToKG/ContentDownload/**"

jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Forbid direct requests usage
        run: |
          ! grep -R "requests\.get(" src/DocsToKG/ContentDownload || exit 1
          ! grep -R "requests\.Session" src/DocsToKG/ContentDownload || exit 1
          echo "✓ No direct requests usage found"
```

### Step 6.2: Test Locally

```bash
! grep -R "requests\.get(" src/DocsToKG/ContentDownload
! grep -R "requests\.Session" src/DocsToKG/ContentDownload
```

**Expected:** Exit code 0 (both grep commands find nothing)

---

## Testing Strategy

### Unit Tests (per task)

- Task 2: Token refund on cache-hit
- Task 3: URL gate + path gate validation
- Task 4: Config loading from Pydantic v2
- Task 5: Pipeline imports work
- Task 6: CI guard catches violations

### Integration Tests

```bash
# Full download flow with atomic writer + cache + policies + config
./.venv/bin/pytest tests/content_download/test_e2e_download_flow.py -v
```

### Regression Tests

```bash
# All existing tests still pass
./.venv/bin/pytest tests/content_download/ -v
```

---

## Final Verification

```bash
# 1. Compile check
./.venv/bin/python -m py_compile src/DocsToKG/ContentDownload/*.py

# 2. Type check
./.venv/bin/mypy src/DocsToKG/ContentDownload/ --strict

# 3. Lint
./.venv/bin/ruff check src/DocsToKG/ContentDownload/
./.venv/bin/black --check src/DocsToKG/ContentDownload/

# 4. Test
./.venv/bin/pytest tests/content_download/ -v

# 5. Guard
! grep -R "requests\.get(" src/DocsToKG/ContentDownload
! grep -R "requests\.Session" src/DocsToKG/ContentDownload
```

---

## Commit Strategy

```bash
# After each task:
git add -A
git commit -m "Task X: [description]"

# Example:
git commit -m "Task 2: Add token refund on hishel cache-hit, verify httpx universal routing"
git commit -m "Task 3: Add path_gate validation, integrate URL/path gates into pipeline"
git commit -m "Task 4: Migrate from DownloadConfig to ContentDownloadConfig"
git commit -m "Task 5: Extract pipeline contracts, delete legacy pipeline module"
git commit -m "Task 6: Add CI workflow to guard against direct requests usage"
```

---

## Timeline

Assuming 1 FTE:

- **Morning (3 hours):** Tasks 2 + 3 (httpx wiring + policy gates)
- **Afternoon (3 hours):** Tasks 4 + 5 + 6 (config, pipeline, CI)
- **Buffer (2 hours):** Testing, fixes, documentation

**Total:** ~8 hours = 1 day
