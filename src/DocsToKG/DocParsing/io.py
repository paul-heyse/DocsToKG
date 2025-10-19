"""
Low-level I/O helpers shared across DocParsing stages.

This module houses JSONL streaming utilities, atomic write helpers, and manifest
bookkeeping routines. It deliberately avoids importing the CLI-facing modules so
that other packages can depend on these primitives without pulling in heavy
dependencies.
"""

from __future__ import annotations

import contextlib
import hashlib
import heapq
import json
import logging
import os
import unicodedata
import uuid
import warnings
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
)

from .env import data_manifests

_SAFE_HASH_ALGORITHM = "sha256"
_HASH_ALG_ENV_VAR = "DOCSTOKG_HASH_ALG"
_TEXT_HASH_READ_SIZE = 65536


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
    start: Optional[int] = None,
    end: Optional[int] = None,
    skip_invalid: bool = False,
    max_errors: int = 10,
) -> Iterator[dict]:
    """Stream JSONL records from ``path`` without materialising the full file."""

    yield from _iter_jsonl_records(
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
) -> Iterator[List[dict]]:
    """Yield JSONL rows from ``paths`` in batches of ``batch_size`` records."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    buffer: List[dict] = []
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


def dedupe_preserve_order(items: Iterable[str]) -> Tuple[str, ...]:
    """Return ``items`` without duplicates while preserving encounter order."""

    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def jsonl_load(path: Path, skip_invalid: bool = False, max_errors: int = 10) -> List[dict]:
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
    path: Path, rows: List[dict], validate: Optional[Callable[[dict], None]] = None
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
) -> List[Tuple[int, int]]:
    """Return newline-aligned byte ranges that partition ``path``."""

    size = path.stat().st_size
    if size == 0:
        return [(0, 0)]

    chunk_bytes = max(chunk_bytes, min_chunk_bytes)
    offsets: List[Tuple[int, int]] = []
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
    resolved_to_logical: Dict[Path, Tuple[Path, str, Tuple[bool, str]]] = {}
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


def _iter_jsonl_records(
    path: Path,
    *,
    start: Optional[int],
    end: Optional[int],
    skip_invalid: bool,
    max_errors: int,
) -> Iterator[dict]:
    """Yield JSON-decoded records between optional byte offsets."""

    logger = logging.getLogger(__name__)
    errors = 0
    with path.open("rb") as handle:
        if start:
            handle.seek(start)
            if start != 0:
                handle.readline()
        while True:
            pos = handle.tell()
            if end is not None and pos >= end:
                break
            raw = handle.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
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
    warnings: Optional[List[str]] = None,
    error: Optional[str] = None,
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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


def resolve_manifest_path(stage: str, root: Optional[Path] = None) -> Path:
    """Return the manifest path for ``stage`` relative to ``root``."""

    manifest_dir = data_manifests(root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / _manifest_filename(stage)


def resolve_attempts_path(stage: str, root: Optional[Path] = None) -> Path:
    """Return the attempts log path for ``stage`` relative to ``root``."""

    manifest_dir = data_manifests(root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / _telemetry_filename(stage, "attempts")


def _normalise_hash_name(candidate: Optional[str]) -> Optional[str]:
    """Normalise algorithm names for comparison against hashlib."""

    if candidate is None:
        return None
    text = str(candidate).strip().lower()
    return text or None


def _select_hash_algorithm(requested: Optional[str], default: Optional[str]) -> str:
    """Return a supported hash algorithm honouring env overrides and defaults."""

    available = {name.lower() for name in hashlib.algorithms_available}
    if not available:
        raise RuntimeError("No hash algorithms are available via hashlib.")

    logger = logging.getLogger(__name__)
    candidates = [
        (_HASH_ALG_ENV_VAR, os.getenv(_HASH_ALG_ENV_VAR)),
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


def resolve_hash_algorithm(default: str = _SAFE_HASH_ALGORITHM) -> str:
    """Return the active content hash algorithm, guarding invalid overrides."""

    return _select_hash_algorithm(requested=None, default=default)


def make_hasher(name: Optional[str] = None, *, default: Optional[str] = None) -> "hashlib._Hash":
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


def relative_path(path: Path | str, root: Optional[Path]) -> str:
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
    logger: Optional[logging.Logger] = None,
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


def load_manifest_index(stage: str, root: Optional[Path] = None) -> Dict[str, dict]:
    """Load the latest manifest entries for a specific pipeline stage."""

    manifest_dir = data_manifests(root, ensure=False)
    stage_path = manifest_dir / _manifest_filename(stage)
    index: Dict[str, dict] = {}
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


def _manifest_timestamp_key(entry: Mapping[str, object]) -> str:
    """Return a sortable timestamp key for manifest entries."""

    raw = entry.get("timestamp")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, datetime):
        return raw.isoformat()
    if raw is None:
        return ""
    return str(raw)


def _iter_manifest_file(path: Path, stage: str) -> Iterator[dict]:
    """Yield manifest entries for a single stage file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry.setdefault("stage", stage)
            yield entry


def iter_manifest_entries(stages: Sequence[str], root: Optional[Path] = None) -> Iterator[dict]:
    """Yield manifest entries for the requested ``stages`` sorted by timestamp."""

    manifest_dir = data_manifests(root, ensure=False)
    heap: List[Tuple[str, int, dict, Iterator[dict]]] = []
    unique = count()

    for stage in stages:
        stage_path = manifest_dir / _manifest_filename(stage)
        if not stage_path.exists():
            continue
        stream = _iter_manifest_file(stage_path, stage)
        try:
            first_entry = next(stream)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (
                _manifest_timestamp_key(first_entry),
                next(unique),
                first_entry,
                stream,
            ),
        )

    while heap:
        _, _, entry, stream = heapq.heappop(heap)
        yield entry
        try:
            next_entry = next(stream)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (
                _manifest_timestamp_key(next_entry),
                next(unique),
                next_entry,
                stream,
            ),
        )


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
