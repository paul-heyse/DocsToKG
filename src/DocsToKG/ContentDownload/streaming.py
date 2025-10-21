"""Download streaming architecture with resume support and exactly-once semantics.

This module implements the core streaming pipeline:
  1. HEAD precheck → ServerValidators (ETag, Last-Modified, Accept-Ranges)
  2. Resume decision → ResumeDecision (fresh, resume, or discard)
  3. Quota guard → Prevent disk exhaustion
  4. Stream to .part → Rolling SHA-256, optional preallocation, fsync
  5. Finalize → Atomic rename, hash indexing, deduplication
  6. Manifest updates → Telemetry and tracking

All operations are designed for crash recovery and multi-worker safety via
idempotency keys (see idempotency.py).

RFC Compliance:
  - RFC 7232: HTTP Conditional Requests (ETag, Last-Modified)
  - RFC 7233: HTTP Range Requests (206 Partial Content)
  - RFC 3986: URL normalization (canonical URLs)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass(frozen=True)
class ServerValidators:
    """HTTP response headers used for conditional requests and resume decisions.

    Attributes:
        etag: ETag header value (opaque server validator)
        last_modified: Last-Modified header value (HTTP date)
        accept_ranges: Whether server supports Accept-Ranges (206 Partial Content)
        content_length: Content-Length header value in bytes
    """

    etag: Optional[str]
    last_modified: Optional[str]
    accept_ranges: bool
    content_length: Optional[int]


@dataclass(frozen=True)
class LocalPartState:
    """State of a partial `.part` file from a prior download attempt.

    Attributes:
        path: Path to the `.part` file
        bytes_local: Bytes already written to disk
        prefix_hash_hex: SHA-256 of first N bytes (persisted from prior run)
        etag: ETag stored from prior attempt
        last_modified: Last-Modified stored from prior attempt
    """

    path: Path
    bytes_local: int
    prefix_hash_hex: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None


@dataclass(frozen=True)
class ResumeDecision:
    """Decision about whether to resume from a partial download.

    Attributes:
        mode: "fresh" (no prior part), "resume" (safe to resume), or "discard_part"
        range_start: Byte offset to resume from (if mode == "resume")
        reason: Explanation of the decision (for logging/metrics)
    """

    mode: Literal["fresh", "resume", "discard_part"]
    range_start: Optional[int]
    reason: str


@dataclass(frozen=True)
class StreamMetrics:
    """Metrics collected during a streaming download.

    Attributes:
        bytes_written: New bytes written in this stream attempt
        elapsed_ms: Total elapsed time (milliseconds)
        fsync_ms: Time spent in fsync operations
        sha256_hex: Final SHA-256 hash (lowercase hex)
        avg_write_mibps: Average throughput (MiB/s)
        resumed_from_bytes: Bytes already on disk when resumed
    """

    bytes_written: int
    elapsed_ms: int
    fsync_ms: int
    sha256_hex: str
    avg_write_mibps: float
    resumed_from_bytes: int


# ============================================================================
# Quota Guard
# ============================================================================


def ensure_quota(
    root_dir: Path,
    expected_bytes: Optional[int],
    *,
    free_min: int,
    margin: float,
) -> None:
    """Verify disk quota before starting a download.

    Raises:
        OSError: If free space < (expected + free_min * margin)

    Args:
        root_dir: Filesystem root to check
        expected_bytes: Expected download size (None = use free_min only)
        free_min: Minimum free bytes to maintain
        margin: Safety multiplier (e.g., 1.5 for 50% safety margin)
    """
    try:
        stat = os.statvfs(str(root_dir))
    except OSError as e:
        raise OSError(f"quota: failed to stat {root_dir}: {e}") from e

    free = stat.f_bavail * stat.f_frsize
    need = int((expected_bytes or 0) * margin) if expected_bytes else free_min

    if free - need < free_min:
        raise OSError(
            f"quota: insufficient free space (free={free}, need~={need}, floor={free_min})"
        )

    LOGGER.debug(
        "quota_check_ok",
        extra={
            "free_bytes": free,
            "need_bytes": need,
            "expected_bytes": expected_bytes,
        },
    )


# ============================================================================
# Resume Decision
# ============================================================================


def can_resume(
    valid: ServerValidators,
    part: Optional[LocalPartState],
    *,
    prefix_check_bytes: int,
    allow_without_validators: bool,
    client: Any,
    url: str,
) -> ResumeDecision:
    """Decide whether to resume from a partial `.part` file.

    Algorithm:
      1. If no .part exists, return "fresh"
      2. If server doesn't support ranges, discard
      3. If validators differ, discard (object changed)
      4. If no validators and not allowed, discard (safety)
      5. Prefix hash check: fetch first N bytes from server and compare
      6. If all checks pass, return "resume" with byte offset

    Args:
        valid: Server response headers (ETag, Last-Modified, etc.)
        part: Local .part state (if exists)
        prefix_check_bytes: Number of bytes to verify (typically 64 KiB)
        allow_without_validators: If True, resume without ETag/Last-Modified
        client: HTTPX client for prefix fetch
        url: URL to download from

    Returns:
        ResumeDecision with mode ("fresh", "resume", or "discard_part") and reason
    """
    # No partial file: start fresh
    if not part or part.bytes_local <= 0:
        return ResumeDecision("fresh", None, "no_part")

    # Server doesn't support ranges
    if not valid.accept_ranges:
        return ResumeDecision("discard_part", None, "no_accept_ranges")

    # Validators don't match: object has changed
    if (valid.etag and part.etag and valid.etag != part.etag) or (
        valid.last_modified and part.last_modified and valid.last_modified != part.last_modified
    ):
        return ResumeDecision("discard_part", None, "validators_mismatch")

    # No validators: require opt-in
    if (not valid.etag and not valid.last_modified) and not allow_without_validators:
        return ResumeDecision("discard_part", None, "validators_missing")

    # Prefix hash check: fetch first N bytes from server
    try:
        end = max(0, min(prefix_check_bytes, part.bytes_local) - 1)
        if end >= 0:
            resp = client.get(
                url,
                headers={"Range": f"bytes=0-{end}"},
                timeout=(5, 10),
                follow_redirects=True,
            )
            if resp.status_code not in (206, 200):
                return ResumeDecision("discard_part", None, f"prefix_http_{resp.status_code}")

            remote_prefix = resp.content
            local_prefix_hex = part.prefix_hash_hex or _compute_prefix_sha256(
                part.path, max_bytes=len(remote_prefix)
            )
            remote_prefix_hex = hashlib.sha256(remote_prefix).hexdigest()

            if remote_prefix_hex != local_prefix_hex:
                return ResumeDecision("discard_part", None, "prefix_mismatch")
    except Exception as e:
        LOGGER.warning(f"prefix_check_error: {e}")
        return ResumeDecision("discard_part", None, f"prefix_error_{type(e).__name__}")

    # All checks passed: safe to resume
    return ResumeDecision("resume", part.bytes_local, "ok")


def _compute_prefix_sha256(path: Path, max_bytes: int = 65536) -> str:
    """Compute SHA-256 of first N bytes of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            chunk = f.read(max_bytes)
            h.update(chunk)
    except Exception as e:
        LOGGER.warning(f"prefix_compute_error: {e}")
    return h.hexdigest()


# ============================================================================
# Streaming to .part
# ============================================================================


def stream_to_part(
    *,
    client: Any,
    url: str,
    part_path: Path,
    range_start: Optional[int],
    chunk_bytes: int,
    do_fsync: bool,
    preallocate_min: int,
    expected_total: Optional[int],
    artifact_lock: Callable,
    logger: logging.Logger,
    verify_content_length: bool = True,
) -> StreamMetrics:
    """Stream content into a `.part` file with resume support.

    Features:
      - Optional resume from range_start (206 Partial Content)
      - Rolling SHA-256 computation
      - Optional file preallocation
      - fsync for durability
      - Chunked writing for memory efficiency
      - Atomic file state management
      - Content-Length verification (optional)

    Args:
        client: HTTPX client (with Tenacity + rate limiting in stack)
        url: URL to download
        part_path: Destination `.part` file path
        range_start: Byte offset to resume from (None = fresh)
        chunk_bytes: Read chunk size
        do_fsync: If True, fsync after write
        preallocate_min: Min file size to prealloc
        expected_total: Expected Content-Length
        artifact_lock: Lock context manager for atomic operations
        logger: Logger instance
        verify_content_length: If True, verify bytes_written matches expected_total

    Returns:
        StreamMetrics with bytes written, timing, and final hash

    Raises:
        RuntimeError: On HTTP errors, Content-Range mismatch, or Content-Length mismatch.
    """
    from DocsToKG.ContentDownload.io_utils import SizeMismatchError

    part_path.parent.mkdir(parents=True, exist_ok=True)
    resumed_from = 0
    t0 = time.monotonic()

    with artifact_lock(str(part_path)):
        mode = "r+b" if part_path.exists() else "w+b"
        with open(part_path, mode) as f:
            # Position for resume
            if range_start and range_start > 0:
                f.seek(0, os.SEEK_END)
                resumed_from = f.tell()
                if resumed_from != range_start:
                    raise RuntimeError(
                        f"resume-misaligned: local={resumed_from} remote={range_start}"
                    )
            else:
                f.truncate(0)

            # Preallocation (best-effort)
            if expected_total and expected_total >= preallocate_min:
                try:
                    _preallocate_fd(f.fileno(), expected_total)
                except Exception as e:
                    logger.debug(f"prealloc_skip: {e}")

            # Build request with optional Range header
            headers = {}
            if range_start and range_start > 0:
                headers["Range"] = f"bytes={range_start}-"

            resp = client.build_request("GET", url, headers=headers)

            # Stream response
            with client.send(resp, stream=True, follow_redirects=True) as response:
                status = response.status_code

                # Verify status and Content-Range
                if range_start:
                    if status != 206:
                        raise RuntimeError(f"expected 206 on resume, got {status}")
                    cr = response.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {range_start}-"):
                        raise RuntimeError(f"bad Content-Range: {cr}")
                else:
                    if status not in (200, 206):
                        raise RuntimeError(f"unexpected status {status}")

                # Rolling SHA-256: seed with existing bytes if resuming
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
                for chunk in response.iter_bytes(chunk_size=chunk_bytes):
                    if not chunk:
                        continue
                    f.write(chunk)
                    h.update(chunk)
                f.flush()

                # fsync for durability
                fsync_ms = 0
                if do_fsync:
                    tfs = time.monotonic()
                    os.fsync(f.fileno())
                    fsync_ms = int((time.monotonic() - tfs) * 1000)

                bytes_after = f.tell()
                bytes_written = bytes_after - bytes_before

                # P1 Scope: Verify Content-Length matches bytes written
                if verify_content_length and expected_total is not None:
                    total_bytes = (resumed_from or 0) + (bytes_written or 0)
                    if total_bytes != expected_total:
                        raise SizeMismatchError(expected_total, total_bytes)

    # Metrics
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


def _preallocate_fd(fd: int, size: int) -> None:
    """Preallocate disk space using fallocate or ftruncate.

    Args:
        fd: File descriptor
        size: Number of bytes to allocate

    Raises:
        OSError: On allocation failure
    """
    try:
        os.posix_fallocate(fd, 0, size)
    except (AttributeError, OSError):
        # Fallback to ftruncate (sparse allocation)
        os.ftruncate(fd, size)


# ============================================================================
# Finalization & Indexing
# ============================================================================


def finalize_artifact(
    *,
    root_dir: Path,
    part_path: Path,
    sha256_hex: str,
    shard_width: int,
    ext: str,
    artifact_lock: Callable,
    hash_index: Any,
    prefer_hardlink: bool,
) -> Path:
    """Atomically finalize a `.part` file and register in hash index.

    Flow:
      1. Compute final path from hash (sharded)
      2. Atomic rename under lock
      3. Register in hash index
      4. Return final path

    Args:
        root_dir: Root directory for artifacts
        part_path: Source `.part` file
        sha256_hex: SHA-256 hash (lowercase hex)
        shard_width: Number of hex chars for shard prefix (0 = no sharding)
        ext: File extension (e.g., ".pdf")
        artifact_lock: Lock context manager
        hash_index: Hash index object (must have put() method)
        prefer_hardlink: Hint for deduplication strategy (unused here)

    Returns:
        Path to finalized file

    Raises:
        FileNotFoundError: If part_path doesn't exist
        OSError: On filesystem errors
    """
    # Compute final path
    shard = sha256_hex[:shard_width] if shard_width > 0 else ""
    final_dir = root_dir / shard if shard else root_dir
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{sha256_hex}{ext}"

    # Atomic finalization
    with artifact_lock(str(final_path)):
        if not part_path.exists():
            raise FileNotFoundError(f"part file not found: {part_path}")

        os.replace(str(part_path), str(final_path))
        size = os.path.getsize(final_path)

        # Register in hash index
        hash_index.put(sha256_hex, str(final_path), size)

        LOGGER.info(
            "artifact_finalized",
            extra={
                "sha256": sha256_hex,
                "size_bytes": size,
                "final_path": str(final_path),
            },
        )

    return final_path


# ============================================================================
# Orchestrator
# ============================================================================


def download_pdf(
    *,
    client: Any,
    head_client: Any,
    url: str,
    cfg: Any,
    root_dir: Path,
    staging_dir: Path,
    artifact_lock: Callable,
    hash_index: Any,
    manifest_sink: Any,
    logger: logging.Logger,
    offline: bool = False,
) -> Dict[str, Any]:
    """Orchestrate a complete artifact download with resume and deduplication.

    Flow:
      1. Offline check
      2. Deduplication (quick win if hash known)
      3. HEAD precheck → ServerValidators
      4. Resume decision
      5. Quota guard
      6. Stream to .part
      7. Finalize & index
      8. Write manifest
      9. Return metrics

    Args:
        client: HTTPX raw client (Tenacity/rate limiting in stack)
        head_client: HTTPX cached client for HEAD requests
        url: URL to download
        cfg: Configuration object (io, resume, quota, shard, dedupe settings)
        root_dir: Root directory for artifacts
        staging_dir: Temporary .part staging directory
        artifact_lock: Lock context manager
        hash_index: Hash index object
        manifest_sink: Manifest writer
        logger: Logger instance
        offline: If True and offline_block_artifacts=True, raise error

    Returns:
        Manifest row dict with metrics and deduplication action

    Raises:
        RuntimeError: On offline mode, HTTP errors, quota, etc.
    """
    canon_url = url  # Assume canonicalized upstream

    # Offline guard
    if offline and getattr(cfg, "offline_block_artifacts", True):
        raise RuntimeError("offline: artifacts disabled (blocked_offline)")

    # -------- Deduplication (quick win) --------
    if cfg.dedupe.hardlink:
        sha = hash_index.get_hash_for_url(canon_url)
        if sha:
            hit = hash_index.get_path_and_size(sha)
            if hit:
                existing, _size = hit
                shard = sha[: cfg.shard.width] if cfg.shard.enabled else ""
                expected_final = (
                    root_dir / shard / f"{sha}.pdf" if shard else root_dir / f"{sha}.pdf"
                )

                action = hash_index.dedupe_link_or_copy(
                    existing, str(expected_final), prefer_hardlink=True
                )

                row = {
                    "final_path": str(expected_final),
                    "part_path": None,
                    "sha256": sha,
                    "dedupe_action": action,
                    "resumed_from_bytes": 0,
                    "bytes_written": 0,
                    "elapsed_ms": 0,
                    "fsync_ms": 0,
                    "avg_write_mibps": 0.0,
                    "shard_prefix": shard or None,
                }
                manifest_sink.write(row)
                logger.info(f"dedupe_hardlink: {action}")
                return row

    # -------- HEAD precheck --------
    hv = head_client.head(url, follow_redirects=True, timeout=(5, 10))
    accept_ranges = "bytes" in hv.headers.get("Accept-Ranges", "").lower()
    content_length = (
        int(hv.headers.get("Content-Length")) if hv.headers.get("Content-Length") else None
    )

    validators = ServerValidators(
        etag=hv.headers.get("ETag"),
        last_modified=hv.headers.get("Last-Modified"),
        accept_ranges=accept_ranges,
        content_length=content_length,
    )

    # -------- Resolve .part path --------
    safe_slug = hashlib.sha256(canon_url.encode("utf-8")).hexdigest()[:16]
    part_path = staging_dir / f"{safe_slug}.part"

    lp: Optional[LocalPartState] = None
    if part_path.exists():
        lp = LocalPartState(
            path=part_path,
            bytes_local=os.path.getsize(part_path),
            prefix_hash_hex=None,
            etag=None,
            last_modified=None,
        )

    # -------- Resume decision --------
    dec = can_resume(
        valid=validators,
        part=lp,
        prefix_check_bytes=cfg.resume.prefix_check_bytes,
        allow_without_validators=cfg.resume.allow_without_validators,
        client=head_client,
        url=url,
    )

    if dec.mode == "discard_part" and part_path.exists():
        try:
            part_path.unlink()
        except Exception:
            pass
        lp = None

    range_start = dec.range_start if dec.mode == "resume" else None

    # -------- Quota guard --------
    try:
        ensure_quota(
            root_dir,
            validators.content_length,
            free_min=cfg.quota.free_bytes_min,
            margin=cfg.quota.margin_factor,
        )
    except OSError as e:
        raise RuntimeError(f"quota_guard: {e}") from e

    # -------- Stream --------
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
        verify_content_length=cfg.io.verify_content_length,
    )

    # -------- Finalize & Index --------
    final_path = finalize_artifact(
        root_dir=root_dir,
        part_path=part_path,
        sha256_hex=sm.sha256_hex,
        shard_width=cfg.shard.width if cfg.shard.enabled else 0,
        ext=".pdf",
        artifact_lock=artifact_lock,
        hash_index=hash_index,
        prefer_hardlink=True,
    )

    hash_index.put_url_hash(canon_url, sm.sha256_hex)

    # -------- Manifest row --------
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
