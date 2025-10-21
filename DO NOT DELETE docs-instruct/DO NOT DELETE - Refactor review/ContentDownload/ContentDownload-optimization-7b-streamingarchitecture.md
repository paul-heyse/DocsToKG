Amazing — here’s a **clear A→B architecture** for item **7) Download streaming & file I/O**, plus **fully wired code snippets** you can paste into your project to make the pathways explicit for a junior dev or agent. I’ll show:

* the architectural flow (who calls what, locks, and data passed),
* a single **orchestrator** function that coordinates HEAD → resume decision → streaming → finalize → hash-index → manifest updates,
* the **streamer** that does rolling SHA-256, optional preallocation, fsync, and careful resume verification,
* quota guard & prefix-hash checks,
* how this fits with your limiter/breaker/offline modes.

---

## Architecture (bird’s-eye view)

```
Planner/Runner
  └─ lease_next_job()  (idempotent; state=LEASED)
      └─ download_pdf()  (ORCHESTRATOR)
           1) HEAD precheck → ServerValidators
           2) decide resume/fresh  (ResumeDecision)
           3) quota guard
           4) STREAM (RateLimitedTransport, Tenacity, breaker are in the hub)
                └─ stream_to_part()  (rolling sha256, fsync, optional prealloc)
           5) finalize_artifact()   (atomic rename under artifact_lock)
           6) hash_index.put(hash→path), url_hash_map.put(url→hash)
           7) manifest/telemetry updates (exactly-once op keys)
```

### Locking scope

* **Streaming**: hold a lock keyed to the artifact **identity** so only one writer creates/extends the `.part` at a time (derive a stable lock name from the canonical URL or intended final hash key if known).
* **Finalization**: lock on the **final path** (hash-based) so only one process renames and claims the canonical file for a given hash.

### Filesystem layout

```
<root>/
  .staging/          # transient .part files (same filesystem as <root>)
  ab/                # shard by first 2 hex chars of sha256
    abcdef... .pdf   # canonical final path = hash-named + extension
```

---

## Key invariants (never compromised)

1. **All writes go to `*.part`** → `flush` → `fsync` → **atomic `os.replace`** (rename) to final path.
2. **Resume only if**: `Accept-Ranges` present **and** `206` + correct `Content-Range` **and** “same object” (ETag/Last-Modified match, or prefix-hash matches).
3. **Hash computed end-to-end** across old bytes (if resuming) + new stream → stored in manifest + hash index.
4. **No duplicate work**: before network, check **hash index** for known content and **hardlink**/copy instead of downloading (config-gated).
5. **Quota guard**: don’t start large downloads if disk budget is low.

---

## Orchestrator (HEAD → resume → stream → finalize → index)

> `download_pdf()` glues together everything for a single artifact.

```python
from __future__ import annotations
import os, time, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

# --- Assume these helpers already exist in your repo (or from earlier messages) ---
# - artifact_lock(final_or_lock_key): contextmanager
# - preallocate_fd(fd, length, path_for_cli=...) -> PreallocResult
# - compute_prefix_sha256(path, max_bytes=65536) -> str
# - SQLiteHashIndex: get_path(), get_hash_for_url(), put(), put_url_hash(), dedupe_link_or_copy(...)
# - canonical_for_request(url, role="artifact"), canonical_for_index(url)
# - RateLimitedTransport is already installed in the HTTPX client
# - HEAD precheck lives in your networking hub: head_precheck(url) → (etag, last_modified, accept_ranges, content_length)

@dataclass
class ServerValidators:
    etag: Optional[str]
    last_modified: Optional[str]
    accept_ranges: bool
    content_length: Optional[int]

@dataclass
class LocalPartState:
    path: Path
    bytes_local: int
    prefix_hash_hex: Optional[str]   # persisted from a prior run (optional)
    etag: Optional[str]
    last_modified: Optional[str]

@dataclass
class ResumeDecision:
    mode: Literal["fresh","resume","discard_part"]
    range_start: Optional[int]
    reason: str

# ---------------- Quota guard ----------------
def ensure_quota(root_dir: Path, expected_bytes: Optional[int], *, free_min: int, margin: float) -> None:
    """Raise OSError if free space is below budget for this download."""
    stat = os.statvfs(root_dir)
    free = stat.f_bavail * stat.f_frsize
    need = int((expected_bytes or 0) * margin) if expected_bytes else free_min
    if free - need < free_min:
        raise OSError(f"quota: insufficient free space (free={free}, need~={need}, floor={free_min})")

# ---------------- Resume decision ----------------
def can_resume(valid: ServerValidators, part: Optional[LocalPartState], *, prefix_check_bytes: int, allow_without_validators: bool, client, url: str) -> ResumeDecision:
    if not part or part.bytes_local <= 0:
        return ResumeDecision("fresh", None, "no_part")
    if not valid.accept_ranges:
        return ResumeDecision("discard_part", None, "no_accept_ranges")
    # If validators present, require match
    if (valid.etag and part.etag and valid.etag != part.etag) or (valid.last_modified and part.last_modified and valid.last_modified != part.last_modified):
        return ResumeDecision("discard_part", None, "validators_mismatch")
    # Prefix-check path: fetch first prefix_check_bytes from server and compare to local prefix hash or bytes
    if (not valid.etag and not valid.last_modified) and not allow_without_validators:
        return ResumeDecision("discard_part", None, "validators_missing")
    # Build remote prefix GET (Range: bytes=0-N)
    end = max(0, min(prefix_check_bytes, part.bytes_local) - 1)
    if end >= 0:
        r = client.get(url, headers={"Range": f"bytes=0-{end}"}, timeout=(5, 10), follow_redirects=True)
        if r.status_code not in (206, 200):
            return ResumeDecision("discard_part", None, f"prefix_http_{r.status_code}")
        remote = r.content  # small (<=64KiB)
        local_prefix_hex = part.prefix_hash_hex or compute_prefix_sha256(part.path, max_bytes=len(remote))
        h = hashlib.sha256(remote).hexdigest()
        if h != local_prefix_hex:
            return ResumeDecision("discard_part", None, "prefix_mismatch")
    # At this point, resume is safe. The correct Range start is the local file length.
    return ResumeDecision("resume", part.bytes_local, "ok")

# ---------------- Streaming to .part ----------------
@dataclass
class StreamMetrics:
    bytes_written: int
    elapsed_ms: int
    fsync_ms: int
    sha256_hex: str
    avg_write_mibps: float
    resumed_from_bytes: int

def stream_to_part(*, client, url: str, part_path: Path, range_start: Optional[int], chunk_bytes: int, do_fsync: bool, preallocate_min: int, expected_total: Optional[int], artifact_lock, logger) -> StreamMetrics:
    """
    Stream into 'part_path' with optional resume, rolling sha256, fsync, optional preallocation.
    Preconditions:
      - If range_start is not None, server must support Accept-Ranges and we will receive 206 with correct Content-Range.
    """
    part_path.parent.mkdir(parents=True, exist_ok=True)
    resumed_from = 0
    t0 = time.monotonic()

    with artifact_lock(part_path):  # lock by .part to ensure single writer for this artifact URL
        mode = "r+b" if part_path.exists() else "w+b"
        with open(part_path, mode) as f:
            # Seek to end if resuming; else truncate to 0
            if range_start and range_start > 0:
                f.seek(0, os.SEEK_END)
                resumed_from = f.tell()
                if resumed_from != range_start:
                    raise RuntimeError(f"resume-misaligned: local={resumed_from} remote={range_start}")
            else:
                f.truncate(0)

            # Preallocate if large and supported
            if expected_total and expected_total >= preallocate_min:
                try:
                    from DocsToKG.ContentDownload.download.preallocate import preallocate_fd
                    preallocate_fd(f.fileno(), expected_total, path_for_cli=str(part_path))
                except Exception as e:
                    logger.debug("prealloc_skip", error=str(e))

            # Build request (Range if resuming)
            headers = {}
            if range_start and range_start > 0:
                headers["Range"] = f"bytes={range_start}-"
            r = client.build_request("GET", url, headers=headers)
            # IMPORTANT: network stack should be Tenacity-wrapped already; here we send/stream.
            with client.send(r, stream=True, follow_redirects=True) as resp:
                status = resp.status_code
                if range_start:
                    if status != 206:
                        raise RuntimeError(f"expected 206 on resume, got {status}")
                    cr = resp.headers.get("Content-Range", "")
                    # Content-Range: bytes start-end/total
                    if not cr.startswith(f"bytes {range_start}-"):
                        raise RuntimeError(f"bad Content-Range: {cr}")
                else:
                    if status not in (200, 206):  # some servers send 206 even for full
                        raise RuntimeError(f"unexpected status {status}")

                # Rolling SHA256: seed with existing bytes if resuming (local read)
                h = hashlib.sha256()
                if resumed_from:
                    with open(part_path, "rb") as seed:
                        while True:
                            b = seed.read(2 * 1024 * 1024)
                            if not b:
                                break
                            h.update(b)

                # Stream chunks
                bytes_before = f.tell()
                for chunk in resp.iter_bytes():
                    if not chunk:
                        continue
                    f.write(chunk)
                    h.update(chunk)
                f.flush()

                fsync_ms = 0
                if do_fsync:
                    tfs = time.monotonic()
                    os.fsync(f.fileno())
                    fsync_ms = int((time.monotonic() - tfs) * 1000)

                bytes_after = f.tell()
                bytes_written = bytes_after - bytes_before

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    total_bytes = (resumed_from or 0) + (bytes_written or 0)
    mib = total_bytes / (1024 * 1024)
    avg_mibps = (mib / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0
    return StreamMetrics(
        bytes_written=bytes_written,
        elapsed_ms=elapsed_ms,
        fsync_ms=fsync_ms,
        sha256_hex=h.hexdigest(),
        avg_write_mibps=round(avg_mibps, 3),
        resumed_from_bytes=resumed_from,
    )

# ---------------- Finalization + hash index ----------------
def finalize_artifact(*, root_dir: Path, part_path: Path, sha256_hex: str, shard_width: int, ext: str, artifact_lock, hash_index, prefer_hardlink: bool) -> Path:
    # Determine canonical final path by hash
    shard = sha256_hex[:shard_width] if shard_width > 0 else ""
    final_dir = root_dir / shard if shard else root_dir
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{sha256_hex}{ext}"

    # Atomic move under lock to avoid two finalizers colliding on same hash
    with artifact_lock(final_path):
        os.replace(part_path, final_path)  # atomic within same filesystem
        size = os.path.getsize(final_path)
        # Register in hash index (canonical store)
        hash_index.put(sha256_hex, final_path, size)

    return final_path

# ---------------- Orchestrator ----------------
def download_pdf(
    *,
    client,                 # HTTPX raw client (RateLimitedTransport, Tenacity in networking)
    head_client,            # HTTPX cached client for HEAD
    url: str,
    cfg,                    # object with fields shown in earlier config (io, resume, quota, shard, dedupe)
    root_dir: Path,
    staging_dir: Path,
    artifact_lock,          # ctxmanager
    hash_index,             # SQLiteHashIndex
    manifest_sink,          # your existing manifest writer (JSONL/SQLite)
    logger,
    offline: bool = False,
) -> dict:
    """
    Returns a manifest row dict with final metrics and dedupe_action.
    Raises a descriptive exception on failure.
    """
    canon_url = url  # assume upstream already canonicalized with role="artifact"

    # 0) Offline guard
    if offline and getattr(cfg, "offline_block_artifacts", True):
        raise RuntimeError("offline: artifacts disabled (blocked_offline)")

    # 1) DEDUPE (quick win) — reuse canonical hash file if we already know it for this URL
    if cfg.dedupe.hardlink:
        sha = hash_index.get_hash_for_url(canon_url)
        if sha:
            hit = hash_index.get_path_and_size(sha)
            if hit:
                existing, _ = hit
                shard = sha[: cfg.shard.width] if cfg.shard.enabled else ""
                expected_final = (root_dir / shard / f"{sha}{'.pdf'}")
                action = hash_index.dedupe_link_or_copy(existing, expected_final, prefer_hardlink=True)
                row = {
                    "final_path": str(expected_final),
                    "part_path": None,
                    "sha256": sha,
                    "dedupe_action": action,
                    "resumed": 0,
                    "bytes_written": 0,
                    "elapsed_ms": 0,
                    "fsync_ms": 0,
                    "avg_write_mibps": 0.0,
                    "shard_prefix": shard or None,
                }
                manifest_sink.write(row)
                return row

    # 2) HEAD precheck
    hv = head_client.head(url, follow_redirects=True, timeout=(5, 10))  # through networking hub wrapper
    accept_ranges = "bytes" in hv.headers.get("Accept-Ranges", "").lower()
    content_length = int(hv.headers.get("Content-Length")) if hv.headers.get("Content-Length") else None
    validators = ServerValidators(
        etag=hv.headers.get("ETag"),
        last_modified=hv.headers.get("Last-Modified"),
        accept_ranges=accept_ranges,
        content_length=content_length,
    )

    # 3) Resolve .part path in staging
    safe_slug = hashlib.sha256(canon_url.encode("utf-8")).hexdigest()[:16]
    part_path = staging_dir / f"{safe_slug}.part"
    lp: Optional[LocalPartState] = None
    if part_path.exists():
        lp = LocalPartState(
            path=part_path,
            bytes_local=os.path.getsize(part_path),
            prefix_hash_hex=None,   # load from sidecar/meta if you persist it; else we’ll compute when needed
            etag=None,
            last_modified=None,
        )

    # 4) Resume decision
    dec = can_resume(valid=validators, part=lp, prefix_check_bytes=cfg.resume.prefix_check_bytes, allow_without_validators=cfg.resume.allow_without_validators, client=head_client, url=url)
    if dec.mode == "discard_part" and part_path.exists():
        try: part_path.unlink()
        except Exception: pass
        lp = None
    range_start = dec.range_start if dec.mode == "resume" else None

    # 5) Quota guard
    try:
        ensure_quota(root_dir, validators.content_length, free_min=cfg.quota.free_bytes_min, margin=cfg.quota.margin_factor)
    except OSError as e:
        raise RuntimeError(f"quota_guard: {e}")

    # 6) STREAM
    sm = stream_to_part(
        client=client,
        url=url,
        part_path=part_path,
        range_start=range_start,
        chunk_bytes=cfg.io.chunk_bytes,
        do_fsync=cfg.io.fsync,
        preallocate_min=cfg.io.preallocate_min_size_bytes if cfg.io.preallocate else 1 << 60,
        expected_total=validators.content_length,
        artifact_lock=artifact_lock,
        logger=logger,
    )

    # 7) FINALIZE (hash-based layout) and INDEX
    final_path = finalize_artifact(
        root_dir=root_dir,
        part_path=part_path,
        sha256_hex=sm.sha256_hex,
        shard_width=(cfg.shard.width if cfg.shard.enabled else 0),
        ext=".pdf",
        artifact_lock=artifact_lock,
        hash_index=hash_index,
        prefer_hardlink=True,
    )
    # Map URL→hash
    hash_index.put_url_hash(canon_url, sm.sha256_hex)

    # 8) Manifest row
    row = {
        "final_path": str(final_path),
        "part_path": str(part_path),
        "content_length": validators.content_length,
        "etag": validators.etag,
        "last_modified": validators.last_modified,
        "sha256": sm.sha256_hex,
        "bytes_written": sm.bytes_written,
        "resumed_from_bytes": sm.resumed_from_bytes,
        "resumed": 1 if (sm.resumed_from_bytes or 0) > 0 else 0,
        "fsync_ms": sm.fsync_ms,
        "elapsed_ms": sm.elapsed_ms,
        "avg_write_mibps": sm.avg_write_mibps,
        "dedupe_action": "download",
        "shard_prefix": sm.sha256_hex[: cfg.shard.width] if cfg.shard.enabled else None,
    }
    manifest_sink.write(row)
    return row
```

> **Where retries happen**: the **networking hub** (HTTPX + Tenacity) wraps the request/response so that any connect/timeout and 429/5xx paths are retried conservatively (Retry-After aware). The streamer **does not** implement its own retry loop—if the socket breaks on `iter_bytes()`, let the exception bubble and the job runner reschedules per your idempotency logic.

---

## Practical considerations & guidance

### PDF signature check (belt-and-suspenders)

Right after the first chunk (or first few KB), if `Content-Type` wasn’t clearly `application/pdf`, sniff the first 5 bytes for `%PDF-`. If missing, abort early with a “classifier mismatch” to avoid writing garbage. (Keep this behind a toggle if you see legitimate non-pdf binary types.)

### Preallocation

* Use the **expected total** when known (`Content-Length`).
* Preallocate only when `expected_total >= preallocate_min_size_bytes` (e.g., 2–8 MiB).
* On unsupported FS, `preallocate_fd()` falls back to `ftruncate` (sparse) safely.

### Lock naming

* For **streaming**, lock by a stable key (e.g., `.staging/<slug>.part`); two workers given the same URL will serialize naturally.
* For **finalization**, lock by the **final path** (hash) to prevent double renames if multiple URLs map to the same content.

### Where to compute URL slug

* We used `sha256(canonical_url)[:16]` for the staging `.part` name; it’s deterministic and avoids filesystem-unfriendly characters.

### Error mapping (raise early with reason)

* `resume-misaligned`, `bad Content-Range`, `prefix_mismatch`—these are **non-retryable** within the streaming function; your runner should handle them by deleting `.part` and calling **fresh** once more (idempotency & state machine from “data model & idempotency” scope).

### Offline mode

* For artifacts, **block** by default (no cache for PDFs). That’s why the orchestrator checks `offline` first and raises `blocked_offline`. This matches your “Hishel for metadata only” policy.

---

## Telemetry you should emit (per artifact)

* `resume_decision` & reason (`ok`, `validators_mismatch`, `no_accept_ranges`, `prefix_mismatch`, …).
* `stream_bytes`, `elapsed_ms`, `fsync_ms`, `avg_write_mibps`.
* `dedupe_action` (`download|hardlink|copy|skipped`).
* `quota_start_free`, `quota_need`, decision.
* `final_path`, `sha256`, `shard_prefix`.

These show up in: “partial downloads resumed/succeeded”, “corruption rate (0)”, fsync p95, shard distribution, etc.

---

## Test checklist (minimal but powerful)

1. **Fresh → finalize**: 200 OK, full write, atomic rename; manifest row present.
2. **Resume success**: part exists; HEAD says Accept-Ranges; 206 with correct `Content-Range`; rolling hash matches.
3. **Resume rejects**: validators mismatch or prefix mismatch; `.part` is deleted and fresh starts next time.
4. **Quota guard**: low free space triggers pre-stream error; no `.part` created or it’s tiny with instant cleanup.
5. **Hardlink dedupe**: url→hash present, canonical file exists; no network called; hardlink or copy created at expected path.
6. **Crash mid-stream**: kill process; on restart, `can_resume()` chooses resume; final hash correct.
7. **Atomic rename**: simulate second process also trying to finalize same hash; lock prevents double rename.

---

## Small “gotchas” we’ve already avoided

* **Atomic rename across filesystems**: we keep `.staging` **on the same FS** as the final root.
* **Double-counting bandwidth**: `avg_write_mibps` uses only new bytes (we also report `resumed_from_bytes`).
* **Cache interactions**: the artifact client must be **raw**, not cached. Cached client is only for HEAD/metadata.
* **Limiter & breaker**: placed in the transport below cache; this code doesn’t sleep for rate, it trusts the hub.

---

If you want, I can also drop a small **“staging cleaner”** (deletes `.part` older than N hours) and a **“reconciler”** snippet that fixes DB↔FS mismatches at startup (e.g., FINALIZED in DB but missing file), but the above gets you a production-grade streaming pipeline with the exact behaviors you wanted.
