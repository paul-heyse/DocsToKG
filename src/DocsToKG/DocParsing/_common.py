"""
DocParsing Shared Utilities

This module consolidates lightweight helpers that power multiple DocParsing
pipeline stages. Utilities span path discovery, atomic file writes, JSONL
parsing, manifest bookkeeping, and structured logging so that chunking,
embedding, and conversion scripts can share consistent behaviour without an
additional dependency layer.

Key Features:
- Resolve DocsToKG data directories with environment and ancestor discovery
- Stream JSONL inputs and outputs with validation and error tolerance
- Emit structured JSON logs suited for machine ingestion and dashboards
- Manage pipeline manifests, batching helpers, and advisory file locks

Usage:
    from DocsToKG.DocParsing import _common as common_util

    chunks_dir = common_util.data_chunks()
    with common_util.atomic_write(chunks_dir / \"example.jsonl\") as handle:
        handle.write(\"{}\")

Dependencies:
- json, pathlib, logging: Provide standard I/O and diagnostics primitives.
- typing: Supply type hints consumed by Sphinx documentation tooling and API generators.
- pydantic (optional): Some helpers integrate with schema validation routines.

All helpers are safe to import in multiprocessing contexts and avoid heavy
third-party dependencies beyond the standard library.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, TextIO, TypeVar

T = TypeVar("T")

__all__ = [
    "detect_data_root",
    "data_doctags",
    "data_chunks",
    "data_vectors",
    "data_manifests",
    "data_pdfs",
    "data_html",
    "expand_path",
    "resolve_hf_home",
    "resolve_model_root",
    "find_free_port",
    "atomic_write",
    "iter_doctags",
    "iter_chunks",
    "jsonl_load",
    "jsonl_save",
    "get_logger",
    "Batcher",
    "manifest_append",
    "compute_content_hash",
    "resolve_hash_algorithm",
    "load_manifest_index",
    "acquire_lock",
    "set_spawn_or_warn",
]


def expand_path(path: str | Path) -> Path:
    """Return ``path`` expanded to an absolute :class:`Path`.

    Args:
        path: Candidate filesystem path supplied as string or :class:`Path`.

    Returns:
        Absolute path with user home components resolved.
    """

    return Path(path).expanduser().resolve()


def resolve_hf_home() -> Path:
    """Resolve the HuggingFace cache directory respecting ``HF_HOME``."""

    env = os.getenv("HF_HOME")
    if env:
        return expand_path(env)
    return expand_path(Path.home() / ".cache" / "huggingface")


def resolve_model_root(hf_home: Optional[Path] = None) -> Path:
    """Resolve the DocsToKG model root honoring ``DOCSTOKG_MODEL_ROOT``."""

    env = os.getenv("DOCSTOKG_MODEL_ROOT")
    if env:
        return expand_path(env)
    base = hf_home if hf_home is not None else resolve_hf_home()
    return expand_path(base)


def detect_data_root(start: Optional[Path] = None) -> Path:
    """Locate the DocsToKG Data directory via env var or ancestor scan.

    Checks the ``DOCSTOKG_DATA_ROOT`` environment variable first. If not set,
    scans ancestor directories for a ``Data`` folder containing expected
    subdirectories (``PDFs``, ``HTML``, ``DocTagsFiles``, or
    ``ChunkedDocTagFiles``).

    Args:
        start: Starting directory for the ancestor scan. Defaults to the
            current working directory when ``None``.

    Returns:
        Absolute path to the resolved ``Data`` directory. When
        ``DOCSTOKG_DATA_ROOT`` is set but the directory does not yet exist,
        it is created automatically.

    Examples:
        >>> os.environ["DOCSTOKG_DATA_ROOT"] = "/tmp/data"
        >>> (Path("/tmp/data")).mkdir(parents=True, exist_ok=True)
        >>> detect_data_root()
        PosixPath('/tmp/data')

        >>> os.environ.pop("DOCSTOKG_DATA_ROOT")
        >>> detect_data_root(Path("/workspace/DocsToKG/src"))
        PosixPath('/workspace/DocsToKG/Data')
    """

    env_root = os.getenv("DOCSTOKG_DATA_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if not env_path.exists():
            env_path.mkdir(parents=True, exist_ok=True)
        return env_path

    start_path = Path.cwd() if start is None else Path(start).resolve()
    expected_dirs = ["PDFs", "HTML", "DocTagsFiles", "ChunkedDocTagFiles"]
    for ancestor in [start_path, *start_path.parents]:
        candidate = ancestor / "Data"
        if any((candidate / directory).is_dir() for directory in expected_dirs):
            return candidate.resolve()

    return (start_path / "Data").resolve()


def _ensure_dir(path: Path) -> Path:
    """Create ``path`` if needed and return its absolute form.

    Args:
        path: Directory to create when missing.

    Returns:
        Absolute path to the created directory.

    Examples:
        >>> _ensure_dir(Path("./tmp_dir"))
        PosixPath('tmp_dir')
    """

    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def data_doctags(root: Optional[Path] = None) -> Path:
    """Return the DocTags directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the DocTags directory.

    Examples:
        >>> isinstance(data_doctags(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "DocTagsFiles")


def data_chunks(root: Optional[Path] = None) -> Path:
    """Return the chunk directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the chunk directory.

    Examples:
        >>> isinstance(data_chunks(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "ChunkedDocTagFiles")


def data_vectors(root: Optional[Path] = None) -> Path:
    """Return the vectors directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the vectors directory.

    Examples:
        >>> isinstance(data_vectors(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "Embeddings")


def data_manifests(root: Optional[Path] = None) -> Path:
    """Return the manifests directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the manifests directory.

    Examples:
        >>> isinstance(data_manifests(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "Manifests")


def data_pdfs(root: Optional[Path] = None) -> Path:
    """Return the PDFs directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the PDFs directory.

    Examples:
        >>> isinstance(data_pdfs(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "PDFs")


def data_html(root: Optional[Path] = None) -> Path:
    """Return the HTML directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the HTML directory.

    Examples:
        >>> isinstance(data_html(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "HTML")


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a structured JSON logger configured for console output.

    Args:
        name: Name of the logger to create or retrieve.
        level: Logging level (case insensitive). Defaults to ``"INFO"``.

    Returns:
        Configured :class:`logging.Logger` instance.

    Examples:
        >>> logger = get_logger("docparse")
        >>> logger.level == logging.INFO
        True
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()

        class JSONFormatter(logging.Formatter):
            """Emit structured JSON log messages for DocParsing utilities.

            Attributes:
                default_time_format: Timestamp template applied to log records.

            Examples:
                >>> formatter = JSONFormatter()
                >>> hasattr(formatter, "format")
                True
            """

            converter = time.gmtime

            def format(self, record: logging.LogRecord) -> str:
                """Render a log record as a JSON string.

                Args:
                    record: Logging record produced by the DocParsing pipeline.

                Returns:
                    JSON-formatted string containing canonical log fields and optional extras.
                """
                payload = {
                    "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                extra_fields = getattr(record, "extra_fields", None)
                if isinstance(extra_fields, dict):
                    payload.update(extra_fields)
                return json.dumps(payload, ensure_ascii=False)

        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False
    return logger


def find_free_port(start: int = 8000, span: int = 32) -> int:
    """Locate an available TCP port on localhost within a range.

    Args:
        start: Starting port for the scan. Defaults to ``8000``.
        span: Number of sequential ports to check. Defaults to ``32``.

    Returns:
        The first free port number. Falls back to an OS-assigned ephemeral port
        if the requested range is exhausted.

    Examples:
        >>> port = find_free_port(8500, 1)
        >>> isinstance(port, int)
        True
    """

    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port

    logger = get_logger(__name__)
    logger.warning(
        "Port scan exhausted",
        extra={"extra_fields": {"start": start, "span": span, "action": "ephemeral_port"}},
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def atomic_write(path: Path) -> Iterator[TextIO]:
    """Write to a temporary file and atomically replace the destination.

    Args:
        path: Target path to write.

    Returns:
        Context manager yielding a writable text handle.

    Yields:
        Writable text file handle. Caller must write data before context exit.

    Raises:
        Any exception raised while writing or replacing the file is propagated
        after the temporary file is cleaned up.

    Examples:
        >>> target = Path("/tmp/example.txt")
        >>> with atomic_write(target) as handle:
        ...     _ = handle.write("hello")
    """

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def iter_doctags(directory: Path) -> Iterator[Path]:
    """Yield DocTags files within ``directory`` and subdirectories.

    Args:
        directory: Root directory to scan for DocTags artifacts.

    Returns:
        Iterator over absolute ``Path`` objects.

    Yields:
        Absolute paths to discovered ``.doctags`` or ``.doctag`` files sorted
        lexicographically.

    Examples:
        >>> next(iter_doctags(Path(".")), None) is None
        True
    """

    extensions = ("*.doctags", "*.doctag")
    seen = set()
    for pattern in extensions:
        for candidate in directory.rglob(pattern):
            if candidate.is_file() and not candidate.name.startswith("."):
                seen.add(candidate.resolve())
    for path in sorted(seen):
        yield path


def iter_chunks(directory: Path) -> Iterator[Path]:
    """Yield chunk JSONL files from ``directory`` and all descendants.

    Args:
        directory: Directory containing chunk artifacts.

    Returns:
        Iterator over absolute ``Path`` objects.

    Yields:
        Absolute paths to files matching ``*.chunks.jsonl`` sorted
        lexicographically.

    Examples:
        >>> next(iter_chunks(Path(".")), None) is None
        True
    """

    seen = set()
    for candidate in directory.rglob("*.chunks.jsonl"):
        if candidate.is_file() and not candidate.name.startswith("."):
            seen.add(candidate.resolve())
    for path in sorted(seen):
        yield path


def set_spawn_or_warn(logger: Optional[logging.Logger] = None) -> None:
    """Ensure the multiprocessing start method is set to ``spawn``.

    Args:
        logger: Optional logger used to emit diagnostic output.

    The helper enforces CUDA safety guarantees by configuring the
    ``spawn`` start method when possible. If another start method is
    already active, the helper logs a warning describing the current
    method so callers understand the degraded safety state.
    """

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
        if logger is not None:
            logger.debug("Multiprocessing start method set to 'spawn'")
        return
    except RuntimeError:
        current = mp.get_start_method(allow_none=True)
        if current == "spawn":
            if logger is not None:
                logger.debug("Multiprocessing start method already 'spawn'")
            return
        message = (
            "Multiprocessing start method is %s; CUDA workloads require 'spawn'."
            % (current or "unset")
        )
        if logger is not None:
            logger.warning(message)
        else:
            logging.getLogger(__name__).warning(message)


def set_spawn_or_warn(logger: Optional[logging.Logger] = None) -> None:
    """Ensure the multiprocessing start method is set to ``spawn``.

    Args:
        logger: Optional logger used to emit diagnostic output.

    The helper enforces CUDA safety guarantees by configuring the
    ``spawn`` start method when possible. If another start method is
    already active, the helper logs a warning describing the current
    method so callers understand the degraded safety state.
    """

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
        if logger is not None:
            logger.debug("Multiprocessing start method set to 'spawn'")
        return
    except RuntimeError:
        current = mp.get_start_method(allow_none=True)
        if current == "spawn":
            if logger is not None:
                logger.debug("Multiprocessing start method already 'spawn'")
            return
        message = (
            "Multiprocessing start method is %s; CUDA workloads require 'spawn'."
            % (current or "unset")
        )
        if logger is not None:
            logger.warning(message)
        else:
            logging.getLogger(__name__).warning(message)


def jsonl_load(path: Path, skip_invalid: bool = False, max_errors: int = 10) -> List[dict]:
    """Load a JSONL file into memory with optional error tolerance.

    Args:
        path: JSON Lines file to read.
        skip_invalid: When ``True``, invalid lines are skipped up to
            ``max_errors`` occurrences instead of raising immediately.
        max_errors: Maximum number of errors to tolerate when
            ``skip_invalid`` is ``True``.

    Returns:
        List of parsed dictionaries.

    Raises:
        ValueError: If a malformed JSON line is encountered and ``skip_invalid``
            is ``False``.

    Examples:
        >>> tmp = Path("/tmp/example.jsonl")
        >>> _ = tmp.write_text('{"a": 1}\n', encoding="utf-8")
        >>> jsonl_load(tmp)
        [{'a': 1}]
    """

    logger = get_logger(__name__)
    rows: List[dict] = []
    errors: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - exercised via tests
                message = f"Line {line_number}: {exc}"
                if skip_invalid:
                    errors.append(message)
                    if len(errors) >= max_errors:
                        logger.error(
                            "Too many JSON errors",
                            extra={
                                "extra_fields": {
                                    "path": str(path),
                                    "errors": len(errors),
                                }
                            },
                        )
                        break
                else:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc
    if errors:
        logger.warning(
            "Skipped invalid JSON lines",
            extra={"extra_fields": {"path": str(path), "skipped": len(errors)}},
        )
    return rows


def jsonl_save(
    path: Path, rows: List[dict], validate: Optional[Callable[[dict], None]] = None
) -> None:
    """Persist dictionaries to a JSONL file atomically.

    Args:
        path: Destination JSONL file.
        rows: Sequence of dictionaries to serialize.
        validate: Optional callback invoked per row before serialization.

    Returns:
        None: This function performs I/O side effects only.

    Raises:
        ValueError: If ``validate`` raises an exception for any row.

    Examples:
        >>> tmp = Path("/tmp/example.jsonl")
        >>> jsonl_save(tmp, [{"a": 1}])
        >>> tmp.read_text(encoding="utf-8").strip()
        '{"a": 1}'
    """

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                if validate is not None:
                    try:
                        validate(row)
                    except Exception as exc:  # pragma: no cover - error path exercised
                        raise ValueError(f"Validation failed for row {index}: {exc}") from exc
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


class Batcher(Iterable[List[T]]):
    """Yield fixed-size batches from an iterable.

    Args:
        iterable: Source iterable providing items to batch.
        batch_size: Maximum number of elements per yielded batch.

    Examples:
        >>> list(Batcher([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """

    def __init__(self, iterable: Iterable[T], batch_size: int):
        """Initialise the batching iterator.

        Args:
            iterable: Source iterable providing items to batch.
            batch_size: Target size for each yielded batch.

        Returns:
            None

        Raises:
            ValueError: If ``batch_size`` is less than ``1``.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._iterable = iterable
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[List[T]]:
        """Yield successive lists containing up to ``batch_size`` elements.

        Args:
            None: Iteration consumes the iterable supplied to :class:`Batcher`.

        Returns:
            Iterator over lists where each list contains up to ``batch_size`` items.
        """

        batch: List[T] = []
        for item in self._iterable:
            batch.append(item)
            if len(batch) >= self._batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


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
    """Append a structured entry to the processing manifest.

    Args:
        stage: Pipeline stage emitting the entry.
        doc_id: Identifier of the document being processed.
        status: Outcome status (``success``, ``failure``, or ``skip``).
        duration_s: Optional duration in seconds.
        warnings: Optional list of warning labels.
        error: Optional error description.
        schema_version: Schema identifier recorded for the output.
        **metadata: Arbitrary additional fields to include.

    Returns:
        ``None``.

    Raises:
        ValueError: If ``status`` is not recognised.

    Examples:
        >>> manifest_append("chunk", "doc1", "success")
        >>> (data_manifests() / "docparse.manifest.jsonl").exists()
        True
    """

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

    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def resolve_hash_algorithm(default: str = "sha1") -> str:
    """Return the active content hash algorithm, honoring env overrides.

    Args:
        default: Fallback algorithm name to use when no override is present.

    Returns:
        Hash algorithm identifier resolved from ``DOCSTOKG_HASH_ALG`` or ``default``.
    """

    env_override = os.getenv("DOCSTOKG_HASH_ALG")
    return env_override.strip() if env_override else default


def compute_content_hash(path: Path, algorithm: str = "sha1") -> str:
    """Compute a content hash for ``path`` using the requested algorithm.

    Args:
        path: File whose contents should be hashed.
        algorithm: Hash algorithm name supported by :mod:`hashlib`.

    Notes:
        The ``DOCSTOKG_HASH_ALG`` environment variable overrides ``algorithm``
        when set, enabling fleet-wide hash changes without code edits.

    Returns:
        Hex digest string.

    Examples:
        >>> tmp = Path("/tmp/hash.txt")
        >>> _ = tmp.write_text("hello", encoding="utf-8")
        >>> compute_content_hash(tmp) == hashlib.sha1(b"hello").hexdigest()
        True
    """

    selected_algorithm = resolve_hash_algorithm(algorithm)
    hasher = hashlib.new(selected_algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest_index(stage: str, root: Optional[Path] = None) -> Dict[str, dict]:
    """Load the latest manifest entries for a specific pipeline stage.

    Args:
        stage: Manifest stage identifier to filter entries by.
        root: Optional DocsToKG data root used to resolve the manifest path.

    Returns:
        Mapping of ``doc_id`` to the most recent manifest entry for that stage.

    Raises:
        None: Manifest rows that fail to parse are skipped to keep processing resilient.

    Examples:
        >>> index = load_manifest_index("embeddings")  # doctest: +SKIP
        >>> isinstance(index, dict)
        True
    """

    manifest_dir = data_manifests(root)
    stage_path = manifest_dir / _manifest_filename(stage)
    legacy_path = manifest_dir / "docparse.manifest.jsonl"
    index: Dict[str, dict] = {}
    if stage_path.exists():
        candidates = [stage_path]
    elif legacy_path.exists():
        candidates = [legacy_path]
    else:
        return index

    for manifest_path in candidates:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("stage") != stage:
                    if manifest_path is stage_path:
                        continue
                    continue
                doc_id = entry.get("doc_id")
                if not doc_id:
                    continue
                index[doc_id] = entry
    return index


@contextlib.contextmanager
def acquire_lock(path: Path, timeout: float = 60.0) -> Iterator[bool]:
    """Acquire an advisory lock using ``.lock`` sentinel files.

    Args:
        path: Target file path whose lock should be acquired.
        timeout: Maximum time in seconds to wait for the lock.

    Returns:
        Iterator yielding a boolean when the lock is acquired.

    Yields:
        ``True`` once the lock is acquired.

    Raises:
        TimeoutError: If the lock cannot be obtained within ``timeout``.

    Examples:
        >>> target = Path("/tmp/lock.txt")
        >>> with acquire_lock(target):
        ...     pass
    """

    lock_path = path.with_suffix(path.suffix + ".lock")
    start = time.time()
    while lock_path.exists():
        try:
            pid_text = lock_path.read_text(encoding="utf-8").strip()
            existing_pid = int(pid_text) if pid_text else None
        except (OSError, ValueError):
            existing_pid = None

        if existing_pid and not _pid_is_running(existing_pid):
            lock_path.unlink(missing_ok=True)
            continue

        if time.time() - start > timeout:
            raise TimeoutError(f"Could not acquire lock on {path} after {timeout}s")
        time.sleep(0.1)

    try:
        lock_path.write_text(str(os.getpid()), encoding="utf-8")
        yield True
    finally:
        lock_path.unlink(missing_ok=True)


def _pid_is_running(pid: int) -> bool:
    """Return ``True`` if a process with the given PID appears to be alive."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - platform specific
        return True
    except OSError:  # pragma: no cover - defensive guard
        return False
    return True
