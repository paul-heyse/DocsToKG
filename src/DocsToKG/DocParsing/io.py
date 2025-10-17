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
import json
import logging
import os
import shutil
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, TextIO, Tuple

from .env import data_manifests


@contextlib.contextmanager
def atomic_write(path: Path) -> Iterator[TextIO]:
    """Write to a temporary file and atomically replace the destination."""

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


def jsonl_load(path: Path, skip_invalid: bool = False, max_errors: int = 10) -> List[dict]:
    """Load a JSONL file into memory with optional error tolerance."""

    return list(
        _iter_jsonl_records(
            path,
            start=None,
            end=None,
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
    if not atomic:
        count = 0
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        return count

    count = 0
    with atomic_write(path) as handle:
        if path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as src:
                shutil.copyfileobj(src, handle)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
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


def _iter_jsonl_records(
    path: Path,
    *,
    start: Optional[int],
    end: Optional[int],
    skip_invalid: bool,
    max_errors: int,
) -> Iterator[dict]:
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


def _manifest_filename(stage: str) -> str:
    """Return manifest filename for a given stage."""

    safe = stage.strip() or "all"
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "-" for c in safe)
    return f"docparse.{safe}.manifest.jsonl"


def manifest_append(
    stage: str,
    doc_id: str,
    status: str,
    *,
    duration_s: float = 0.0,
    warnings: Optional[List[str]] = None,
    error: Optional[str] = None,
    schema_version: str = "",
    **metadata,
) -> None:
    """Append a structured entry to the processing manifest."""

    allowed_status = {"success", "failure", "skip"}
    if status not in allowed_status:
        raise ValueError(f"status must be one of {sorted(allowed_status)}")

    manifest_path = data_manifests() / _manifest_filename(stage)
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

    jsonl_append_iter(manifest_path, [entry])


def resolve_hash_algorithm(default: str = "sha1") -> str:
    """Return the active content hash algorithm, honoring env overrides."""

    env_override = os.getenv("DOCSTOKG_HASH_ALG")
    return env_override.strip() if env_override else default


def compute_chunk_uuid(
    doc_id: str,
    start_offset: int,
    text: str,
    *,
    algorithm: str = "sha1",
) -> str:
    """Compute a deterministic UUID for a chunk of text."""

    safe_doc_id = str(doc_id)
    try:
        safe_offset = int(start_offset)
    except (TypeError, ValueError):
        safe_offset = 0
    normalised_text = unicodedata.normalize("NFKC", str(text or ""))

    selected_algorithm = resolve_hash_algorithm(algorithm)
    hasher = hashlib.new(selected_algorithm)
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


def compute_content_hash(path: Path, algorithm: str = "sha1") -> str:
    """Compute a content hash for ``path`` using the requested algorithm."""

    selected_algorithm = resolve_hash_algorithm(algorithm)
    hasher = hashlib.new(selected_algorithm)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    normalised = unicodedata.normalize("NFKC", text)
    hasher.update(normalised.encode("utf-8"))
    return hasher.hexdigest()


def load_manifest_index(stage: str, root: Optional[Path] = None) -> Dict[str, dict]:
    """Load the latest manifest entries for a specific pipeline stage."""

    manifest_dir = data_manifests(root)
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


def iter_manifest_entries(stages: Sequence[str], root: Optional[Path] = None) -> Iterator[dict]:
    """Yield manifest entries for the requested ``stages`` sorted by timestamp."""

    manifest_dir = data_manifests(root)
    combined: List[dict] = []
    for stage in stages:
        stage_path = manifest_dir / _manifest_filename(stage)
        if not stage_path.exists():
            continue
        with stage_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry.setdefault("stage", stage)
                combined.append(entry)

    combined.sort(key=lambda item: item.get("timestamp", ""))
    for entry in combined:
        yield entry


__all__ = [
    "atomic_write",
    "build_jsonl_split_map",
    "compute_chunk_uuid",
    "compute_content_hash",
    "iter_manifest_entries",
    "jsonl_append_iter",
    "jsonl_load",
    "jsonl_save",
    "load_manifest_index",
    "manifest_append",
    "quarantine_artifact",
    "relative_path",
    "resolve_hash_algorithm",
]
