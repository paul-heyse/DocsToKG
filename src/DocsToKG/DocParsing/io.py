# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.io",
#   "purpose": "Low-level I/O helpers shared across DocParsing stages.",
#   "sections": [
#     {
#       "id": "jsonlwriter",
#       "name": "JsonlWriter",
#       "anchor": "class-jsonlwriter",
#       "kind": "class"
#     },
#     {
#       "id": "partition-normalisation-buffer",
#       "name": "_partition_normalisation_buffer",
#       "anchor": "function-partition-normalisation-buffer",
#       "kind": "function"
#     },
#     {
#       "id": "iter-normalised-text-chunks",
#       "name": "_iter_normalised_text_chunks",
#       "anchor": "function-iter-normalised-text-chunks",
#       "kind": "function"
#     },
#     {
#       "id": "atomic-write",
#       "name": "atomic_write",
#       "anchor": "function-atomic-write",
#       "kind": "function"
#     },
#     {
#       "id": "iter-jsonl",
#       "name": "iter_jsonl",
#       "anchor": "function-iter-jsonl",
#       "kind": "function"
#     },
#     {
#       "id": "iter-jsonl-batches",
#       "name": "iter_jsonl_batches",
#       "anchor": "function-iter-jsonl-batches",
#       "kind": "function"
#     },
#     {
#       "id": "dedupe-preserve-order",
#       "name": "dedupe_preserve_order",
#       "anchor": "function-dedupe-preserve-order",
#       "kind": "function"
#     },
#     {
#       "id": "jsonl-load",
#       "name": "jsonl_load",
#       "anchor": "function-jsonl-load",
#       "kind": "function"
#     },
#     {
#       "id": "jsonl-save",
#       "name": "jsonl_save",
#       "anchor": "function-jsonl-save",
#       "kind": "function"
#     },
#     {
#       "id": "jsonl-append-iter",
#       "name": "jsonl_append_iter",
#       "anchor": "function-jsonl-append-iter",
#       "kind": "function"
#     },
#     {
#       "id": "build-jsonl-split-map",
#       "name": "build_jsonl_split_map",
#       "anchor": "function-build-jsonl-split-map",
#       "kind": "function"
#     },
#     {
#       "id": "iter-doctags",
#       "name": "iter_doctags",
#       "anchor": "function-iter-doctags",
#       "kind": "function"
#     },
#     {
#       "id": "iter-jsonl-stream",
#       "name": "_iter_jsonl_stream",
#       "anchor": "function-iter-jsonl-stream",
#       "kind": "function"
#     },
#     {
#       "id": "sanitise-stage",
#       "name": "_sanitise_stage",
#       "anchor": "function-sanitise-stage",
#       "kind": "function"
#     },
#     {
#       "id": "telemetry-filename",
#       "name": "_telemetry_filename",
#       "anchor": "function-telemetry-filename",
#       "kind": "function"
#     },
#     {
#       "id": "manifest-filename",
#       "name": "_manifest_filename",
#       "anchor": "function-manifest-filename",
#       "kind": "function"
#     },
#     {
#       "id": "manifest-append",
#       "name": "manifest_append",
#       "anchor": "function-manifest-append",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-manifest-path",
#       "name": "resolve_manifest_path",
#       "anchor": "function-resolve-manifest-path",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-attempts-path",
#       "name": "resolve_attempts_path",
#       "anchor": "function-resolve-attempts-path",
#       "kind": "function"
#     },
#     {
#       "id": "normalise-hash-name",
#       "name": "_normalise_hash_name",
#       "anchor": "function-normalise-hash-name",
#       "kind": "function"
#     },
#     {
#       "id": "hash-algorithms-available",
#       "name": "_hash_algorithms_available",
#       "anchor": "function-hash-algorithms-available",
#       "kind": "function"
#     },
#     {
#       "id": "select-hash-algorithm",
#       "name": "_select_hash_algorithm",
#       "anchor": "function-select-hash-algorithm",
#       "kind": "function"
#     },
#     {
#       "id": "select-hash-algorithm-uncached",
#       "name": "_select_hash_algorithm_uncached",
#       "anchor": "function-select-hash-algorithm-uncached",
#       "kind": "function"
#     },
#     {
#       "id": "clear-hash-algorithm-cache",
#       "name": "_clear_hash_algorithm_cache",
#       "anchor": "function-clear-hash-algorithm-cache",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-hash-algorithm",
#       "name": "resolve_hash_algorithm",
#       "anchor": "function-resolve-hash-algorithm",
#       "kind": "function"
#     },
#     {
#       "id": "make-hasher",
#       "name": "make_hasher",
#       "anchor": "function-make-hasher",
#       "kind": "function"
#     },
#     {
#       "id": "compute-chunk-uuid",
#       "name": "compute_chunk_uuid",
#       "anchor": "function-compute-chunk-uuid",
#       "kind": "function"
#     },
#     {
#       "id": "relative-path",
#       "name": "relative_path",
#       "anchor": "function-relative-path",
#       "kind": "function"
#     },
#     {
#       "id": "quarantine-artifact",
#       "name": "quarantine_artifact",
#       "anchor": "function-quarantine-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "streamingcontenthasher",
#       "name": "StreamingContentHasher",
#       "anchor": "class-streamingcontenthasher",
#       "kind": "class"
#     },
#     {
#       "id": "compute-content-hash",
#       "name": "compute_content_hash",
#       "anchor": "function-compute-content-hash",
#       "kind": "function"
#     },
#     {
#       "id": "load-manifest-index",
#       "name": "load_manifest_index",
#       "anchor": "function-load-manifest-index",
#       "kind": "function"
#     },
#     {
#       "id": "manifestheapkey",
#       "name": "_ManifestHeapKey",
#       "anchor": "class-manifestheapkey",
#       "kind": "class"
#     },
#     {
#       "id": "manifest-timestamp-key",
#       "name": "_manifest_timestamp_key",
#       "anchor": "function-manifest-timestamp-key",
#       "kind": "function"
#     },
#     {
#       "id": "iter-manifest-tail-lines",
#       "name": "_iter_manifest_tail_lines",
#       "anchor": "function-iter-manifest-tail-lines",
#       "kind": "function"
#     },
#     {
#       "id": "iter-manifest-file",
#       "name": "_iter_manifest_file",
#       "anchor": "function-iter-manifest-file",
#       "kind": "function"
#     },
#     {
#       "id": "iter-manifest-entries",
#       "name": "iter_manifest_entries",
#       "anchor": "function-iter-manifest-entries",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Low-level I/O helpers shared across DocParsing stages.

This module provides JSONL streaming utilities, atomic write helpers, and manifest
bookkeeping routines that power the resume and observability infrastructure. It
deliberately avoids importing CLI-facing modules so other packages can depend on
these primitives without pulling in heavy dependencies.

Key Components:
- JsonlWriter: Lock-aware JSONL append writer for concurrent-safe telemetry writes
- atomic_write: Atomic file writing with fsync durability and parent directory creation
- jsonl_append_iter: Streaming JSONL appends with optional atomicity
- Manifest indexing and hash computation for content verification
- Unicode normalization helpers for cross-platform path handling

The JsonlWriter is the recommended interface for appending to shared manifest or
attempt log files, as it serializes concurrent writers using FileLock and ensures
atomic writes even under concurrent access.

Example:
    from DocsToKG.DocParsing.io import DEFAULT_JSONL_WRITER
    from pathlib import Path

    # Atomically append manifest entries using lock-aware writer
    rows = [{"id": "doc1", "status": "completed"}]
    DEFAULT_JSONL_WRITER(Path("manifest.jsonl"), rows)
"""

from __future__ import annotations

import contextlib
import hashlib
import heapq
import io
import json
import logging
import os
import unicodedata
import uuid
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from functools import total_ordering
from itertools import count
from pathlib import Path
from typing import (
    TextIO,
)

import jsonlines
from filelock import FileLock, Timeout

from .env import data_manifests

_SAFE_HASH_ALGORITHM = "sha256"
_HASH_ALG_ENV_VAR = "DOCSTOKG_HASH_ALG"
_TEXT_HASH_READ_SIZE = 65536
_MANIFEST_TAIL_MIN_WINDOW = 64 * 1024
_MANIFEST_TAIL_BYTES_PER_ENTRY = 4096


_HASH_ALGORITHMS_AVAILABLE: frozenset[str] | None = None
_HASH_ALGORITHM_SELECTION_CACHE: dict[tuple[str | None, str | None], tuple[str | None, str]] = {}


class JsonlWriter:
    """Lock-aware JSONL append writer.

    Uses a per-file FileLock (path + '.lock') to serialize concurrent writers,
    then delegates to jsonl_append_iter(..., atomic=True) for the actual write.
    This ensures safe concurrent appends to manifest and attempt telemetry files.
    """

    def __init__(self, lock_timeout_s: float = 120.0) -> None:
        """Initialize the JSONL writer with a lock timeout.

        Args:
            lock_timeout_s: Timeout in seconds for acquiring the lock.
        """
        self.lock_timeout_s = float(lock_timeout_s)

    def __call__(self, path: Path, rows: Iterable[Mapping]) -> int:
        """Append rows to a JSONL file under FileLock.

        Args:
            path: Target JSONL file path.
            rows: Iterable of dictionaries to append.

        Returns:
            Number of rows appended.

        Raises:
            TimeoutError: If lock cannot be acquired within lock_timeout_s.
        """
        lock_path = Path(f"{path}.lock")
        lock = FileLock(str(lock_path))
        try:
            lock.acquire(timeout=self.lock_timeout_s)
            # Delegate to the existing atomic append path.
            return jsonl_append_iter(path, rows, atomic=True)
        except Timeout as e:
            raise TimeoutError(
                f"Timed out acquiring lock {lock_path} after {self.lock_timeout_s}s "
                f"while appending to {path}. Another writer may be stalled."
            ) from e
        finally:
            try:
                lock.release()
            except Exception:
                # Best-effort; FileLock cleans up on process exit as well.
                pass


# Default instance used by telemetry/manifest sinks
DEFAULT_JSONL_WRITER: JsonlWriter = JsonlWriter()


def _partition_normalisation_buffer(buffer: str) -> tuple[str, str]:
    """Split ``buffer`` into a flushable prefix and a retained suffix.

    The suffix preserves the trailing grapheme cluster so that Unicode
    normalisation remains stable when additional combining marks are read from
    subsequent chunks.
    """

    if not buffer:
        return "", ""

    index = len(buffer)
    while index > 0:
        index -= 1
        if unicodedata.combining(buffer[index]) == 0:
            return buffer[:index], buffer[index:]
    # All characters are combining marks; retain them for the next chunk.
    return "", buffer


def _iter_normalised_text_chunks(handle: TextIO) -> Iterator[bytes]:
    """Yield UTF-8 encoded NFKC-normalised chunks from ``handle``."""

    buffer = ""
    for chunk in iter(lambda: handle.read(_TEXT_HASH_READ_SIZE), ""):
        if not chunk:
            break
        buffer += chunk
        prefix, buffer = _partition_normalisation_buffer(buffer)
        if prefix:
            normalised = unicodedata.normalize("NFKC", prefix)
            if normalised:
                yield normalised.encode("utf-8")
    if buffer:
        yield unicodedata.normalize("NFKC", buffer).encode("utf-8")


@contextlib.contextmanager
def atomic_write(path: Path) -> Iterator[TextIO]:
    """Write to a temporary file and atomically replace the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def iter_jsonl(
    path: Path,
    *,
    start: int | None = None,
    end: int | None = None,
    skip_invalid: bool = False,
    max_errors: int = 10,
) -> Iterator[dict]:
    """Stream JSONL records from ``path`` without materialising the full file."""

    yield from _iter_jsonl_stream(
        path,
        start=start,
        end=end,
        skip_invalid=skip_invalid,
        max_errors=max_errors,
    )


def iter_jsonl_batches(
    paths: Iterable[Path],
    batch_size: int = 1000,
    *,
    skip_invalid: bool = False,
    max_errors: int = 10,
) -> Iterator[list[dict]]:
    """Yield JSONL rows from ``paths`` in batches of ``batch_size`` records."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    buffer: list[dict] = []
    for source in paths:
        for record in iter_jsonl(
            source,
            skip_invalid=skip_invalid,
            max_errors=max_errors,
        ):
            buffer.append(record)
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
    if buffer:
        yield buffer


def dedupe_preserve_order(items: Iterable[str]) -> tuple[str, ...]:
    """Return ``items`` without duplicates while preserving encounter order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def jsonl_load(path: Path, skip_invalid: bool = False, max_errors: int = 10) -> list[dict]:
    """Load a JSONL file into memory with optional error tolerance.

    .. deprecated:: 0.2.0
        Use :func:`iter_jsonl` or :func:`iter_jsonl_batches` for streaming access.
    """

    warnings.warn(
        "jsonl_load is deprecated; switch to iter_jsonl for streaming access.",
        DeprecationWarning,
        stacklevel=2,
    )
    return list(
        iter_jsonl(
            path,
            skip_invalid=skip_invalid,
            max_errors=max_errors,
        )
    )


def jsonl_save(
    path: Path, rows: list[dict], validate: Callable[[dict], None] | None = None
) -> None:
    """Persist dictionaries to a JSONL file atomically."""

    with atomic_write(path) as handle:
        for index, row in enumerate(rows):
            if validate is not None:
                try:
                    validate(row)
                except Exception as exc:  # pragma: no cover - error path exercised
                    raise ValueError(f"Validation failed for row {index}: {exc}") from exc
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_append_iter(
    target: Path | TextIO,
    rows: Iterable[Mapping],
    *,
    atomic: bool = True,
) -> int:
    """Append JSON-serialisable rows to a JSONL file."""

    if hasattr(target, "write"):
        handle = target  # type: ignore[assignment]
        count = 0
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")  # type: ignore[arg-type]
            count += 1
        return count

    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not atomic:
        count = 0
        with path.open("ab") as handle:
            for row in rows:
                payload = json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
                handle.write(payload)
                count += 1
        return count

    buffer = bytearray()
    count = 0
    for row in rows:
        buffer.extend(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        buffer.extend(b"\n")
        count += 1

    if count == 0:
        return 0

    data = bytes(buffer)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(path, flags, 0o666)
    try:
        written = os.write(fd, data)
        if written != len(data):
            raise OSError(
                f"Short write while appending to {path}: expected {len(data)}, wrote {written}"
            )
        os.fsync(fd)
    finally:
        os.close(fd)

    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    return count


def build_jsonl_split_map(
    path: Path,
    *,
    chunk_bytes: int = 32 * 1024 * 1024,
    min_chunk_bytes: int = 1 * 1024 * 1024,
) -> list[tuple[int, int]]:
    """Return newline-aligned byte ranges that partition ``path``."""

    size = path.stat().st_size
    if size == 0:
        return [(0, 0)]

    chunk_bytes = max(chunk_bytes, min_chunk_bytes)
    offsets: list[tuple[int, int]] = []
    with path.open("rb") as handle:
        start = 0
        while start < size:
            target = start + chunk_bytes
            if target >= size:
                end = size
            else:
                handle.seek(target)
                handle.readline()
                end = handle.tell()
                if end <= start:
                    end = min(size, start + chunk_bytes)
            offsets.append((start, end))
            start = end
    return offsets


def iter_doctags(directory: Path) -> Iterator[Path]:
    """Yield DocTags files within ``directory`` and subdirectories.

    The returned paths retain their logical location beneath ``directory`` even
    when they are symbolic links. Files that resolve to the same on-disk target
    are emitted once, preferring concrete files over symlinks and ordering the
    results lexicographically by their logical (relative) path.
    """

    extensions = ("*.doctags", "*.doctag")
    resolved_to_logical: dict[Path, tuple[Path, str, tuple[bool, str]]] = {}
    for pattern in extensions:
        for candidate in directory.rglob(pattern):
            if not candidate.is_file() or candidate.name.startswith("."):
                continue

            try:
                logical = candidate.relative_to(directory)
            except ValueError:
                logical = candidate
            logical_key = logical.as_posix()

            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                resolved = candidate

            priority = (candidate.is_symlink(), logical_key)
            existing = resolved_to_logical.get(resolved)
            if existing is None or priority < existing[2]:
                resolved_to_logical[resolved] = (candidate, logical_key, priority)

    for candidate, _logical_key, _priority in sorted(
        resolved_to_logical.values(), key=lambda item: item[1]
    ):
        yield candidate


def _iter_jsonl_stream(
    path: Path,
    *,
    start: int | None,
    end: int | None,
    skip_invalid: bool,
    max_errors: int,
) -> Iterator[dict]:
    """Yield JSON-decoded records between optional byte offsets."""

    logger = logging.getLogger(__name__)
    errors = 0
    with path.open("rb") as raw_handle:
        if start:
            raw_handle.seek(start)
            if start != 0:
                raw_handle.readline()

        text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", newline="")
        reader = jsonlines.Reader(text_handle)

        try:
            while True:
                pos = raw_handle.tell()
                if end is not None and pos >= end:
                    break

                try:
                    record = reader.read()
                except EOFError:
                    break
                except jsonlines.InvalidLineError as exc:
                    if skip_invalid:
                        errors += 1
                        if errors >= max_errors:
                            logger.error(
                                "Too many JSON errors",
                                extra={
                                    "extra_fields": {
                                        "stage": "core",
                                        "doc_id": "__system__",
                                        "input_hash": None,
                                        "error_code": "JSONL_ERROR_LIMIT",
                                        "path": str(path),
                                        "errors": errors,
                                        "start": start,
                                        "end": end,
                                        "exception": str(exc),
                                    }
                                },
                            )
                            break
                        continue
                    raise ValueError(f"Invalid JSON in {path} at byte offset {pos}: {exc}") from exc

                if record is None:
                    break

                yield record
        finally:
            reader.close()
            with contextlib.suppress(Exception):
                text_handle.detach()

    if errors:
        logger.warning(
            "Skipped invalid JSON lines",
            extra={
                "extra_fields": {
                    "stage": "core",
                    "doc_id": "__system__",
                    "input_hash": None,
                    "error_code": "JSONL_SKIPPED_ROWS",
                    "path": str(path),
                    "skipped": errors,
                    "start": start,
                    "end": end,
                }
            },
        )


def _sanitise_stage(stage: str) -> str:
    """Return a filesystem-friendly identifier for ``stage``."""

    safe = stage.strip() or "all"
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "-" for c in safe)


def _telemetry_filename(stage: str, kind: str) -> str:
    """Return a telemetry filename for ``stage`` and ``kind``."""

    return f"docparse.{_sanitise_stage(stage)}.{kind}.jsonl"


def _manifest_filename(stage: str) -> str:
    """Return manifest filename for a given stage."""

    return _telemetry_filename(stage, "manifest")


def manifest_append(
    stage: str,
    doc_id: str,
    status: str,
    *,
    duration_s: float = 0.0,
    warnings: list[str] | None = None,
    error: str | None = None,
    schema_version: str = "",
    atomic: bool = True,
    **metadata,
) -> None:
    """Append a structured entry to the processing manifest."""

    allowed_status = {"success", "failure", "skip"}
    if status not in allowed_status:
        raise ValueError(f"status must be one of {sorted(allowed_status)}")

    manifest_path = resolve_manifest_path(stage)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "doc_id": doc_id,
        "status": status,
        "duration_s": round(duration_s, 3),
        "warnings": warnings or [],
        "schema_version": schema_version,
    }
    if error is not None:
        entry["error"] = str(error)
    entry.update(metadata)

    jsonl_append_iter(manifest_path, [entry], atomic=atomic)


def resolve_manifest_path(stage: str, root: Path | None = None) -> Path:
    """Return the manifest path for ``stage`` relative to ``root``."""

    manifest_dir = data_manifests(root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / _manifest_filename(stage)


def resolve_attempts_path(stage: str, root: Path | None = None) -> Path:
    """Return the attempts log path for ``stage`` relative to ``root``."""

    manifest_dir = data_manifests(root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / _telemetry_filename(stage, "attempts")


def _normalise_hash_name(candidate: str | None) -> str | None:
    """Normalise algorithm names for comparison against hashlib."""

    if candidate is None:
        return None
    text = str(candidate).strip().lower()
    return text or None


def _hash_algorithms_available() -> frozenset[str]:
    """Return the cached set of available hashlib algorithms."""

    global _HASH_ALGORITHMS_AVAILABLE
    if _HASH_ALGORITHMS_AVAILABLE is None:
        available = {name.lower() for name in hashlib.algorithms_available}
        if not available:
            raise RuntimeError("No hash algorithms are available via hashlib.")
        _HASH_ALGORITHMS_AVAILABLE = frozenset(available)
    return _HASH_ALGORITHMS_AVAILABLE


def _select_hash_algorithm(requested: str | None, default: str | None) -> str:
    """Return a supported hash algorithm honouring env overrides and defaults."""

    env_override = os.getenv(_HASH_ALG_ENV_VAR)
    cache_key = (requested, default)
    cached = _HASH_ALGORITHM_SELECTION_CACHE.get(cache_key)
    if cached is not None and cached[0] == env_override:
        return cached[1]

    algorithm = _select_hash_algorithm_uncached(requested, default, env_override)
    _HASH_ALGORITHM_SELECTION_CACHE[cache_key] = (env_override, algorithm)
    return algorithm


def _select_hash_algorithm_uncached(
    requested: str | None,
    default: str | None,
    env_override: str | None,
) -> str:
    """Resolve a hash algorithm without consulting the selection cache."""

    available = _hash_algorithms_available()
    logger = logging.getLogger(__name__)
    candidates = [
        (_HASH_ALG_ENV_VAR, env_override),
        ("algorithm parameter", requested),
        ("default algorithm", default),
    ]

    for source, candidate in candidates:
        normalised = _normalise_hash_name(candidate)
        if normalised and normalised in available:
            return normalised
        if candidate is None:
            continue
        if normalised:
            logger.warning(
                "Unknown hash algorithm %r supplied via %s; ignoring.",
                candidate,
                source,
            )
        else:
            logger.warning(
                "Blank hash algorithm supplied via %s; ignoring.",
                source,
            )

    fallback = _normalise_hash_name(default)
    if not fallback or fallback not in available:
        fallback = (
            _SAFE_HASH_ALGORITHM
            if _SAFE_HASH_ALGORITHM in available
            else next(iter(sorted(available)))
        )
    logger.warning(
        "No valid hash algorithm requested; falling back to %s.",
        fallback,
    )
    return fallback


def _clear_hash_algorithm_cache() -> None:
    """Reset memoized hash algorithm selections (intended for testing)."""

    _HASH_ALGORITHM_SELECTION_CACHE.clear()


def resolve_hash_algorithm(default: str = _SAFE_HASH_ALGORITHM) -> str:
    """Return the active content hash algorithm, guarding invalid overrides."""

    return _select_hash_algorithm(requested=None, default=default)


def make_hasher(name: str | None = None, *, default: str | None = None) -> hashlib._Hash:
    """Return a configured hashlib object with guarded algorithm resolution."""

    algorithm = _select_hash_algorithm(requested=name, default=default)
    return hashlib.new(algorithm)


def compute_chunk_uuid(
    doc_id: str,
    start_offset: int,
    text: str,
    *,
    algorithm: str = _SAFE_HASH_ALGORITHM,
) -> str:
    """Compute a deterministic UUID for a chunk of text."""

    safe_doc_id = str(doc_id)
    try:
        safe_offset = int(start_offset)
    except (TypeError, ValueError):
        safe_offset = 0
    normalised_text = unicodedata.normalize("NFKC", str(text or ""))

    hasher = make_hasher(name=algorithm)
    hasher.update(safe_doc_id.encode("utf-8"))
    hasher.update(bytes((0x1F,)))
    hasher.update(str(safe_offset).encode("utf-8"))
    hasher.update(bytes((0x1F,)))
    hasher.update(normalised_text.encode("utf-8"))
    digest = hasher.digest()
    if len(digest) < 16:
        digest = (digest * ((16 // len(digest)) + 1))[:16]
    raw = bytearray(digest[:16])
    raw[6] = (raw[6] & 0x0F) | 0x50  # set UUID version bits to 5
    raw[8] = (raw[8] & 0x3F) | 0x80  # set UUID variant bits to RFC 4122
    return str(uuid.UUID(bytes=bytes(raw)))


def relative_path(path: Path | str, root: Path | None) -> str:
    """Return ``path`` rendered relative to ``root`` when feasible."""

    candidate = Path(path)
    if root is None:
        return str(candidate)
    try:
        root_path = Path(root).resolve()
        return str(candidate.resolve().relative_to(root_path))
    except Exception:
        return str(candidate)


def quarantine_artifact(
    path: Path,
    reason: str,
    *,
    logger: logging.Logger | None = None,
    create_placeholder: bool = False,
) -> Path:
    """Move ``path`` to a ``.quarantine`` sibling for operator review."""

    original = Path(path)
    parent = original.parent
    parent.mkdir(parents=True, exist_ok=True)
    suffix = ".quarantine"
    candidate = parent / f"{original.name}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = parent / f"{original.name}{suffix}{counter}"
        counter += 1

    try:
        if original.exists():
            original.rename(candidate)
        elif create_placeholder:
            candidate.write_text(reason + "\n", encoding="utf-8")
    except Exception:
        if logger:
            logger.exception(
                "Failed to quarantine artifact",
                extra={
                    "extra_fields": {
                        "stage": "core",
                        "doc_id": "__system__",
                        "input_hash": None,
                        "error_code": "QUARANTINE_FAILURE",
                        "path": str(original),
                        "attempted_quarantine": str(candidate),
                        "reason": reason,
                    }
                },
            )
        raise

    if logger:
        logger.warning(
            "Quarantined artifact",
            extra={
                "extra_fields": {
                    "stage": "core",
                    "doc_id": "__system__",
                    "input_hash": None,
                    "error_code": "QUARANTINE_SUCCESS",
                    "path": str(original),
                    "quarantine_path": str(candidate),
                    "reason": reason,
                }
            },
        )
    return candidate


class StreamingContentHasher:
    """Incrementally compute a content hash that mirrors :func:`compute_content_hash`."""

    def __init__(self, algorithm: str = _SAFE_HASH_ALGORITHM) -> None:
        """Initialise the streaming hasher with the desired digest algorithm."""

        self.algorithm = algorithm
        self._hasher = make_hasher(name=algorithm)
        self._buffer = ""

    def update(self, text: str) -> None:
        """Ingest ``text`` into the hash while preserving Unicode normalisation semantics."""

        if not text:
            return
        self._buffer += text
        prefix, self._buffer = _partition_normalisation_buffer(self._buffer)
        if prefix:
            normalised = unicodedata.normalize("NFKC", prefix)
            if normalised:
                self._hasher.update(normalised.encode("utf-8"))

    def hexdigest(self) -> str:
        """Finalize the digest and return the hexadecimal representation."""

        if self._buffer:
            tail = unicodedata.normalize("NFKC", self._buffer)
            if tail:
                self._hasher.update(tail.encode("utf-8"))
            self._buffer = ""
        return self._hasher.hexdigest()


def compute_content_hash(path: Path, algorithm: str = _SAFE_HASH_ALGORITHM) -> str:
    """Compute a content hash for ``path`` using the requested algorithm."""

    hasher = make_hasher(name=algorithm)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for chunk in _iter_normalised_text_chunks(handle):
                hasher.update(chunk)
        return hasher.hexdigest()
    except UnicodeDecodeError:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_TEXT_HASH_READ_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


def load_manifest_index(stage: str, root: Path | None = None) -> dict[str, dict]:
    """Load the latest manifest entries for a specific pipeline stage."""

    manifest_dir = data_manifests(root, ensure=False)
    stage_path = manifest_dir / _manifest_filename(stage)
    index: dict[str, dict] = {}
    if not stage_path.exists():
        return index

    with stage_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("stage") != stage:
                continue
            doc_id = entry.get("doc_id")
            if not doc_id:
                continue
            index[doc_id] = entry
    return index


@total_ordering
class _ManifestHeapKey:
    """Comparable key that preserves manifest order when timestamps are missing."""

    __slots__ = ("timestamp", "order")

    def __init__(self, timestamp: str | None, order: int) -> None:
        """Initialize a heap key with the original timestamp and insertion order."""
        self.timestamp = timestamp
        self.order = order

    def __lt__(self, other: object) -> bool:
        """Compare heap keys using timestamp when available, falling back to sequence order."""
        if not isinstance(other, _ManifestHeapKey):
            return NotImplemented
        if self.timestamp is not None and other.timestamp is not None:
            if self.timestamp != other.timestamp:
                return self.timestamp < other.timestamp
        return self.order < other.order

    def __eq__(self, other: object) -> bool:
        """Equality comparison that mirrors the ordering semantics."""
        if not isinstance(other, _ManifestHeapKey):
            return NotImplemented
        return self.timestamp == other.timestamp and self.order == other.order


def _manifest_timestamp_key(entry: Mapping[str, object]) -> str | None:
    """Return a sortable timestamp key or ``None`` when unavailable."""

    raw = entry.get("timestamp")
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped or None
    if isinstance(raw, datetime):
        return raw.isoformat()
    return None


def _iter_manifest_tail_lines(path: Path, limit: int) -> Iterator[str]:
    """Yield the newest ``limit`` JSONL lines from ``path`` in chronological order."""

    if limit <= 0:
        return

    try:
        size = path.stat().st_size
    except OSError:
        return

    if size == 0:
        return

    window_bytes = max(_MANIFEST_TAIL_MIN_WINDOW, limit * _MANIFEST_TAIL_BYTES_PER_ENTRY)
    offsets = build_jsonl_split_map(
        path,
        chunk_bytes=window_bytes,
        min_chunk_bytes=window_bytes,
    )
    collected: list[str] = []
    with path.open("rb") as handle:
        for start, end in reversed(offsets):
            handle.seek(start)
            raw = handle.read(end - start)
            if not raw:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
            chunk_lines = text.splitlines()
            for line in reversed(chunk_lines):
                if not line.strip():
                    continue
                collected.append(line)
                if len(collected) >= limit:
                    break
            if len(collected) >= limit:
                break

    for line in reversed(collected):
        yield line


def _iter_manifest_file(path: Path, stage: str, *, limit: int | None = None) -> Iterator[dict]:
    """Yield manifest entries for a single stage file."""

    if limit is not None and limit > 0:
        line_iter: Iterable[str] = _iter_manifest_tail_lines(path, limit)
    else:
        line_iter = path.open("r", encoding="utf-8")

    with contextlib.ExitStack() as stack:
        if hasattr(line_iter, "__enter__"):
            handle = stack.enter_context(line_iter)  # type: ignore[arg-type]
        else:
            handle = line_iter

        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry.setdefault("stage", stage)
            yield entry


def iter_manifest_entries(
    stages: Sequence[str],
    root: Path | None = None,
    *,
    limit: int | None = None,
) -> Iterator[dict]:
    """Yield manifest entries for ``stages`` sorted by timestamp.

    When ``limit`` is provided, only the newest ``limit`` rows per manifest file are
    read using bounded tail windows, reducing the amount of history that needs to
    be scanned.
    """

    manifest_dir = data_manifests(root, ensure=False)
    heap: list[tuple[_ManifestHeapKey, int, dict, Iterator[dict]]] = []
    unique = count()

    def _push_entry(entry: dict, stream: Iterator[dict]) -> None:
        """Insert the next ``entry`` from ``stream`` into the merge heap."""
        order = next(unique)
        key = _ManifestHeapKey(_manifest_timestamp_key(entry), order)
        heapq.heappush(heap, (key, order, entry, stream))

    for stage in stages:
        stage_path = manifest_dir / _manifest_filename(stage)
        if not stage_path.exists():
            continue
        stream = _iter_manifest_file(stage_path, stage, limit=limit)
        try:
            first_entry = next(stream)
        except StopIteration:
            continue
        _push_entry(first_entry, stream)

    while heap:
        _, _, entry, stream = heapq.heappop(heap)
        yield entry
        try:
            next_entry = next(stream)
        except StopIteration:
            continue
        _push_entry(next_entry, stream)


__all__ = [
    "atomic_write",
    "build_jsonl_split_map",
    "compute_chunk_uuid",
    "compute_content_hash",
    "make_hasher",
    "iter_doctags",
    "iter_manifest_entries",
    "iter_jsonl",
    "iter_jsonl_batches",
    "jsonl_append_iter",
    "jsonl_load",
    "jsonl_save",
    "dedupe_preserve_order",
    "resolve_manifest_path",
    "resolve_attempts_path",
    "load_manifest_index",
    "manifest_append",
    "quarantine_artifact",
    "StreamingContentHasher",
    "relative_path",
    "resolve_hash_algorithm",
]
