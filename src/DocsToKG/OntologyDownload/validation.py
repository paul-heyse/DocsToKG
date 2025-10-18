"""Validation and normalization utilities for ontology downloads."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

from threading import BoundedSemaphore

import psutil

from . import plugins as _plugins
from .plugins import get_validator_registry
from .io import log_memory_usage
from .settings import ResolvedConfig, get_owlready2, get_pronto, get_rdflib

metadata = _plugins.metadata

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .plugins import ResolverPlugin, ValidatorPlugin

_PROCESS = psutil.Process()
_VALIDATOR_SEMAPHORE_CACHE: Dict[int, BoundedSemaphore] = {}
_VALIDATOR_PLUGINS_LOADED = False


def _current_memory_mb() -> float:
    """Return the current resident memory usage in megabytes."""

    return _PROCESS.memory_info().rss / (1024**2)


def _acquire_validator_slot(config: ResolvedConfig) -> BoundedSemaphore:
    """Return a process-wide semaphore guarding validator concurrency."""

    limit = getattr(
        config.defaults.validation,
        "max_concurrent_validators",
        2,
    )
    limit = int(limit)
    limit = max(1, min(8, limit))
    semaphore = _VALIDATOR_SEMAPHORE_CACHE.get(limit)
    if semaphore is None:
        semaphore = BoundedSemaphore(limit)
        _VALIDATOR_SEMAPHORE_CACHE[limit] = semaphore
    return semaphore

rdflib = get_rdflib()
pronto = get_pronto()
owlready2 = get_owlready2()


def load_validator_plugins(
    registry: MutableMapping[str, "ValidatorPlugin"],
    *,
    logger: Optional[logging.Logger] = None,
    reload: bool = False,
) -> None:
    """Load validator plugins while tracking module-level load state."""

    global _VALIDATOR_PLUGINS_LOADED
    should_reload = reload or not _VALIDATOR_PLUGINS_LOADED
    _plugins.load_validator_plugins(registry, logger=logger, reload=should_reload)
    _VALIDATOR_PLUGINS_LOADED = True


@dataclass(slots=True)
class ValidationRequest:
    """Parameters describing a single validation task.

    Attributes:
        name: Identifier of the validator to execute.
        file_path: Path to the ontology document to inspect.
        normalized_dir: Directory used to write normalized artifacts.
        validation_dir: Directory for validator reports and logs.
        config: Resolved configuration that supplies timeout thresholds.

    Examples:
        >>> from pathlib import Path
        >>> from DocsToKG.OntologyDownload import ResolvedConfig
        >>> req = ValidationRequest(
        ...     name="rdflib",
        ...     file_path=Path("ontology.owl"),
        ...     normalized_dir=Path("normalized"),
        ...     validation_dir=Path("validation"),
        ...     config=ResolvedConfig.from_defaults(),
        ... )
        >>> req.name
        'rdflib'
    """

    name: str
    file_path: Path
    normalized_dir: Path
    validation_dir: Path
    config: ResolvedConfig


@dataclass(slots=True)
class ValidationResult:
    """Outcome produced by a validator.

    Attributes:
        ok: Indicates whether the validator succeeded.
        details: Arbitrary metadata describing validator output.
        output_files: Generated files for downstream processing.

    Examples:
        >>> result = ValidationResult(ok=True, details={"triples": 10}, output_files=["ontology.ttl"])
        >>> result.ok
        True
    """

    ok: bool
    details: Dict[str, object]
    output_files: List[str]

    def to_dict(self) -> Dict[str, object]:
        """Represent the validation result as a JSON-serializable dict.

        Args:
            None.

        Returns:
            Dictionary with boolean status, detail payload, and output paths.
        """
        return {
            "ok": self.ok,
            "details": self.details,
            "output_files": self.output_files,
        }


class ValidationTimeout(Exception):
    """Raised when a validation task exceeds the configured timeout.

    Args:
        message: Optional description of the timeout condition.

    Examples:
        >>> raise ValidationTimeout("rdflib exceeded 60s")
        Traceback (most recent call last):
        ...
        ValidationTimeout: rdflib exceeded 60s
    """


def _write_validation_json(path: Path, payload: MutableMapping[str, object]) -> None:
    """Persist structured validation metadata to disk as JSON.

    Args:
        path: Destination path for the JSON payload.
        payload: Mapping containing validation results.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _python_merge_sort(source: Path, destination: Path, *, chunk_size: int = 100_000) -> None:
    """Sort an N-Triples file using a disk-backed merge strategy.

    Args:
        source: Path to the unsorted triple file.
        destination: Output path that receives sorted triples.
        chunk_size: Number of lines loaded into memory per chunk before flushing.
    """

    with tempfile.TemporaryDirectory(prefix="ontology-sort-") as tmp_dir:
        chunk_paths: List[Path] = []
        with source.open("r", encoding="utf-8") as reader:
            while True:
                lines = list(islice(reader, chunk_size))
                if not lines:
                    break
                lines.sort()
                chunk_path = Path(tmp_dir) / f"chunk-{len(chunk_paths)}.nt"
                chunk_path.write_text("".join(lines), encoding="utf-8")
                chunk_paths.append(chunk_path)

        destination.parent.mkdir(parents=True, exist_ok=True)
        if not chunk_paths:
            destination.write_text("", encoding="utf-8")
            return

        with contextlib.ExitStack() as stack:
            iterators: List[Iterator[str]] = []
            for chunk_path in chunk_paths:
                handle = stack.enter_context(chunk_path.open("r", encoding="utf-8"))
                iterators.append(iter(handle))
            with destination.open("w", encoding="utf-8") as writer:
                for line in heapq.merge(*iterators):
                    writer.write(line)


def _term_to_string(term, namespace_manager) -> str:
    """Render an RDF term using the provided namespace manager.

    Args:
        term: RDF term such as a URIRef, BNode, or Literal.
        namespace_manager: Namespace manager responsible for prefix resolution.

    Returns:
        Term rendered in N3 form, falling back to :func:`str` when unavailable.
    """
    formatter = getattr(term, "n3", None)
    if callable(formatter):
        return formatter(namespace_manager)
    return str(term)


def _canonicalize_turtle(graph) -> str:
    """Return canonical Turtle output with sorted prefixes and triples.

    The canonical form mirrors the ontology downloader specification by sorting
    prefixes lexicographically and emitting triples ordered by subject,
    predicate, and object so downstream hashing yields deterministic values.

    Args:
        graph: RDF graph containing triples to canonicalize.

    Returns:
        Canonical Turtle serialization as a string.
    """

    namespace_manager = getattr(graph, "namespace_manager", None)
    if namespace_manager is None or not hasattr(namespace_manager, "namespaces"):
        raise AttributeError("graph lacks namespace manager support")

    try:
        namespace_items = list(namespace_manager.namespaces())
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise AttributeError("unable to iterate namespaces") from exc

    prefix_map: Dict[str, str] = {}
    for prefix, namespace in namespace_items:
        key = prefix or ""
        prefix_map[key] = str(namespace)

    try:
        triples = list(graph)
    except Exception as exc:  # pragma: no cover - stub graphs are not iterable
        raise AttributeError("graph is not iterable") from exc

    triple_lines = [
        f"{_term_to_string(subject, namespace_manager)} {_term_to_string(predicate, namespace_manager)} {_term_to_string(obj, namespace_manager)} ."
        for subject, predicate, obj in sorted(
            ((s, p, o) for s, p, o in triples),
            key=lambda item: (
                _term_to_string(item[0], namespace_manager),
                _term_to_string(item[1], namespace_manager),
                _term_to_string(item[2], namespace_manager),
            ),
        )
    ]

    bnode_map: Dict[str, str] = {}
    triple_lines = [_canonicalize_blank_nodes_line(line, bnode_map) for line in triple_lines]

    prefix_lines = []
    for key in sorted(prefix_map):
        label = f"{key}:" if key else ":"
        prefix_lines.append(f"@prefix {label} <{prefix_map[key]}> .")

    lines: List[str] = []
    lines.extend(prefix_lines)
    if prefix_lines and triple_lines:
        lines.append("")
    lines.extend(triple_lines)
    return "\n".join(lines) + ("\n" if lines else "")


_BNODE_PATTERN = re.compile(r"_:[A-Za-z0-9]+")


def _canonicalize_blank_nodes_line(line: str, mapping: Dict[str, str]) -> str:
    """Replace blank node identifiers with deterministic sequential labels.

    Args:
        line: Serialized triple line containing blank node identifiers.
        mapping: Mutable mapping preserving deterministic blank node assignments.

    Returns:
        Triple line with normalized blank node identifiers.
    """

    def _replace(match: re.Match[str]) -> str:
        key = match.group(0)
        mapped = mapping.get(key)
        if mapped is None:
            mapped = f"_:b{len(mapping)}"
            mapping[key] = mapped
        return mapped

    return _BNODE_PATTERN.sub(_replace, line)


def _sort_triple_file(source: Path, destination: Path) -> None:
    """Sort serialized triple lines using platform sort when available.

    Args:
        source: Path to the unsorted triple file.
        destination: Output path that receives sorted triples.
    """

    sort_binary = shutil.which("sort")
    if sort_binary:
        try:
            with destination.open("w", encoding="utf-8", newline="\n") as handle:
                subprocess.run(  # noqa: PLW1510 - intentional check handling
                    [sort_binary, source.as_posix()],
                    check=True,
                    stdout=handle,
                    text=True,
                )
            return
        except (subprocess.SubprocessError, OSError):
            # Fall back to pure Python sorting when the external command fails.
            pass

    _python_merge_sort(source, destination)


def normalize_streaming(
    source: Path,
    output_path: Optional[Path] = None,
    *,
    graph=None,
    chunk_bytes: int = 1 << 20,
    return_header_hash: bool = False,
) -> Union[str, Tuple[str, str]]:
    """Normalize ontologies using streaming canonical Turtle serialization.

    The streaming path serializes triples to a temporary file, leverages the
    platform ``sort`` command (when available) to order triples lexicographically,
    and streams the canonical Turtle output while computing a SHA-256 digest.
    When ``output_path`` is provided the canonical form is persisted without
    retaining the entire content in memory.

    Args:
        source: Path to the ontology document providing triples.
        output_path: Optional destination for the normalized Turtle document.
        graph: Optional pre-loaded RDF graph re-used instead of reparsing.
        chunk_bytes: Threshold controlling how frequently buffered bytes are flushed.
        return_header_hash: When True, also return the hash of Turtle prefix directives.

    Returns:
        SHA-256 hex digest of the canonical Turtle content, and optionally the hash
        of the serialized prefix header when ``return_header_hash`` is True.
    """

    graph_obj = graph if graph is not None else rdflib.Graph()
    if graph is None:
        graph_obj.parse(source.as_posix())

    namespace_manager = getattr(graph_obj, "namespace_manager", None)
    if namespace_manager is None or not hasattr(namespace_manager, "namespaces"):
        raise AttributeError("graph lacks namespace manager support")

    try:
        namespace_items = list(namespace_manager.namespaces())
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise AttributeError("unable to iterate namespaces") from exc

    prefix_map: Dict[str, str] = {}
    for prefix, namespace in namespace_items:
        key = prefix or ""
        prefix_map[key] = str(namespace)

    prefix_lines = []
    for key in sorted(prefix_map):
        label = f"{key}:" if key else ":"
        prefix_lines.append(f"@prefix {label} <{prefix_map[key]}> .\n")

    header_hash = hashlib.sha256("".join(prefix_lines).encode("utf-8")).hexdigest()

    chunk_limit = max(1, int(chunk_bytes))
    buffer = bytearray()
    sha256 = hashlib.sha256()

    def _flush(writer: Optional[BinaryIO]) -> None:
        if not buffer:
            return
        sha256.update(buffer)
        if writer is not None:
            writer.write(buffer)
        buffer.clear()

    with tempfile.TemporaryDirectory(prefix="ontology-stream-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        unsorted_path = tmp_path / "triples.unsorted"
        with unsorted_path.open("w", encoding="utf-8", newline="\n") as handle:
            for subject, predicate, obj in graph_obj:
                line = (
                    f"{_term_to_string(subject, namespace_manager)} "
                    f"{_term_to_string(predicate, namespace_manager)} "
                    f"{_term_to_string(obj, namespace_manager)} ."
                )
                handle.write(line + "\n")

        sorted_path = tmp_path / "triples.sorted"
        _sort_triple_file(unsorted_path, sorted_path)

        with contextlib.ExitStack() as stack:
            writer: Optional[BinaryIO] = None
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = stack.enter_context(output_path.open("wb"))

            def _emit(text: str) -> None:
                buffer.extend(text.encode("utf-8"))
                if len(buffer) >= chunk_limit:
                    _flush(writer)

            wrote_any = False
            for line in prefix_lines:
                _emit(line)
                wrote_any = True

            bnode_map: Dict[str, str] = {}
            blank_line_pending = bool(prefix_lines)

            with sorted_path.open("r", encoding="utf-8") as reader:
                for raw_line in reader:
                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    if blank_line_pending:
                        _emit("\n")
                        blank_line_pending = False
                    canonical_line = _canonicalize_blank_nodes_line(line, bnode_map) + "\n"
                    _emit(canonical_line)
                    wrote_any = True

            if not wrote_any and writer is not None:
                writer.truncate(0)
                writer.flush()

            _flush(writer)

    content_hash = sha256.hexdigest()
    if return_header_hash:
        return content_hash, header_hash
    return content_hash

class ValidatorSubprocessError(RuntimeError):
    """Raised when a validator subprocess exits unsuccessfully.

    Attributes:
        message: Human-readable description of the underlying subprocess failure.

    Examples:
        >>> raise ValidatorSubprocessError("rdflib validator crashed")
        Traceback (most recent call last):
        ...
        ValidatorSubprocessError: rdflib validator crashed
    """


def _worker_pronto(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Pronto validation logic and emit JSON-friendly results."""

    file_path = Path(payload["file_path"])
    ontology = pronto.Ontology(file_path.as_posix())
    terms = len(list(ontology.terms()))
    result: Dict[str, Any] = {"ok": True, "terms": terms}

    normalized_path = payload.get("normalized_path")
    if normalized_path:
        destination = Path(normalized_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ontology.dump(destination.as_posix(), format="obojson")
        result["normalized_written"] = True

    return result


def _worker_owlready2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Owlready2 validation logic and emit JSON-friendly results."""

    file_path = Path(payload["file_path"])
    ontology = owlready2.get_ontology(file_path.resolve().as_uri()).load()
    entities = len(list(ontology.classes()))
    return {"ok": True, "entities": entities}


_WORKER_DISPATCH = {
    "pronto": _worker_pronto,
    "owlready2": _worker_owlready2,
}


def _run_validator_subprocess(
    name: str, payload: Dict[str, object], *, timeout: int
) -> Dict[str, object]:
    """Execute a validator worker module within a subprocess.

    The subprocess workflow enforces parser timeouts, returns JSON payloads,
    and helps release memory held by heavy libraries such as Pronto and
    Owlready2 after each validation completes.
    """

    command = [sys.executable, "-m", "DocsToKG.OntologyDownload.validation", "worker", name]
    env = os.environ.copy()

    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationTimeout(f"{name} validator exceeded {timeout}s") from exc
    except OSError as exc:
        raise ValidatorSubprocessError(f"Failed to launch {name} validator: {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
        message = stderr or (f"{name} validator subprocess failed with code {completed.returncode}")
        raise ValidatorSubprocessError(message)

    stdout = completed.stdout.decode("utf-8", errors="ignore").strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValidatorSubprocessError(f"{name} validator returned invalid JSON output") from exc


def _run_with_timeout(func, timeout_sec: int) -> None:
    """Execute a callable and raise :class:`ValidationTimeout` on deadline expiry.

    Args:
        func: Callable invoked without arguments.
        timeout_sec: Number of seconds allowed for execution.

    Returns:
        None

    Raises:
        ValidationTimeout: When the callable exceeds the allotted runtime.
    """
    if platform.system() in ("Linux", "Darwin"):
        import signal

        class _Alarm(Exception):
            """Sentinel exception raised when the alarm signal fires.

            Args:
                message: Optional description associated with the exception.

            Attributes:
                message: Optional description associated with the exception.

            Examples:
                >>> try:
                ...     raise _Alarm()
                ... except _Alarm:
                ...     pass
            """

        def _handler(signum, frame):  # pragma: no cover - platform dependent
            """Signal handler converting SIGALRM into :class:`ValidationTimeout`.

            Args:
                signum: Received signal number.
                frame: Current stack frame (unused).
            """
            raise ValidationTimeout()  # pragma: no cover - bridges to outer scope

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_sec)
        try:
            func()
        finally:
            signal.alarm(0)
    else:  # Windows
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                future.result(timeout=timeout_sec)
            except FuturesTimeoutError as exc:  # pragma: no cover - platform specific
                raise ValidationTimeout() from exc


def _prepare_xbrl_package(
    request: ValidationRequest, logger: logging.Logger
) -> tuple[Path, List[str]]:
    """Extract XBRL taxonomy ZIP archives for downstream validation.

    Args:
        request: Validation request describing the ontology package under test.
        logger: Logger used to record extraction telemetry.

    Returns:
        Tuple containing the entrypoint path passed to Arelle and a list of artifacts.

    Raises:
        ValueError: If the archive is malformed or contains unsafe paths.
    """
    package_path = request.file_path
    if package_path.suffix.lower() != ".zip":
        return package_path, []
    if not zipfile.is_zipfile(package_path):
        raise ValueError("XBRL package is not a valid ZIP archive")

    with zipfile.ZipFile(package_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path detected in archive: {member.filename}")
            if member.compress_size == 0 and member.file_size > 0:
                raise ValueError(f"Zip entry {member.filename} has invalid compression size")
            ratio = member.file_size / max(member.compress_size, 1)
            if ratio > 10:
                raise ValueError(
                    f"Zip entry {member.filename} exceeds compression ratio limit (ratio={ratio:.1f})"
                )

    temp_dir = Path(tempfile.mkdtemp(prefix="ontofetch-xbrl-"))
    try:
        with zipfile.ZipFile(package_path) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                target_path = temp_dir / member_path
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    final_dir = request.validation_dir / "arelle" / package_path.stem
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_dir), final_dir)
    logger.info(
        "extracted xbrl package",
        extra={"stage": "validate", "validator": "arelle", "destination": str(final_dir)},
    )

    entrypoint_candidates = sorted(final_dir.rglob("*.xsd")) or sorted(final_dir.rglob("*.xml"))
    entrypoint = entrypoint_candidates[0] if entrypoint_candidates else package_path
    artifacts = [str(path) for path in final_dir.rglob("*") if path.is_file()]
    return entrypoint, artifacts


def validate_rdflib(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Parse ontologies with rdflib, canonicalize Turtle output, and emit hashes.

    Args:
        request: Validation request describing the target ontology and output directories.
        logger: Logger adapter used for structured validation events.

    Returns:
        ValidationResult capturing success state, metadata, canonical hash,
        and generated files.

    Raises:
        ValidationTimeout: Propagated when parsing exceeds configured timeout.
    """
    graph = rdflib.Graph()
    payload: Dict[str, object] = {"ok": False}
    timeout = request.config.defaults.validation.parser_timeout_sec

    def _parse() -> None:
        """Parse the ontology with rdflib to populate the graph object."""
        graph.parse(request.file_path.as_posix())

    try:
        log_memory_usage(logger, stage="validate", event="before", validator="rdflib")
        _run_with_timeout(_parse, timeout)
        log_memory_usage(logger, stage="validate", event="after", validator="rdflib")
        triple_count = len(graph)
        payload = {"ok": True, "triples": triple_count}
        output_files: List[str] = []
        normalization_mode = "in-memory"

        if "ttl" in request.config.defaults.normalize_to:
            request.normalized_dir.mkdir(parents=True, exist_ok=True)
            stem = request.file_path.stem
            normalized_ttl = request.normalized_dir / f"{stem}.ttl"
            threshold_bytes = (
                request.config.defaults.validation.streaming_normalization_threshold_mb
                * 1024
                * 1024
            )
            file_size = request.file_path.stat().st_size
            streaming_hash: Optional[str] = None
            streaming_header_hash: Optional[str] = None
            normalized_sha: Optional[str] = None
            if file_size >= threshold_bytes:
                normalization_mode = "streaming"
                try:
                    streaming_hash, streaming_header_hash = normalize_streaming(
                        request.file_path,
                        output_path=normalized_ttl,
                        graph=graph,
                        return_header_hash=True,
                    )
                    normalized_sha = streaming_hash
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "streaming normalization failed, falling back to in-memory",
                        extra={
                            "stage": "validate",
                            "validator": "rdflib",
                            "error": str(exc),
                        },
                    )
                    normalization_mode = "in-memory"
                    streaming_hash = None

            if normalized_sha is None:
                try:
                    canonical_ttl = _canonicalize_turtle(graph)
                except AttributeError:
                    graph.serialize(destination=normalized_ttl, format="turtle")
                    canonical_ttl = normalized_ttl.read_text(encoding="utf-8")
                else:
                    normalized_ttl.write_text(canonical_ttl, encoding="utf-8")
                normalized_sha = hashlib.sha256(canonical_ttl.encode("utf-8")).hexdigest()

            payload["normalized_sha256"] = normalized_sha
            payload["normalization_mode"] = normalization_mode
            output_files.append(str(normalized_ttl))
            if streaming_hash is not None:
                payload["streaming_nt_sha256"] = streaming_hash
            if streaming_header_hash is not None:
                payload["streaming_prefix_sha256"] = streaming_header_hash
        if (
            "ttl" in request.config.defaults.normalize_to
            and payload.get("streaming_nt_sha256") is None
        ):
            try:
                extra_hash, extra_header = normalize_streaming(
                    request.file_path,
                    graph=graph,
                    return_header_hash=True,
                )
                payload["streaming_nt_sha256"] = extra_hash
                payload.setdefault("streaming_prefix_sha256", extra_header)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "failed to compute streaming normalization hash",
                    extra={
                        "stage": "validate",
                        "validator": "rdflib",
                        "error": str(exc),
                    },
                )

        _write_validation_json(request.validation_dir / "rdflib_parse.json", payload)
        return ValidationResult(ok=True, details=payload, output_files=output_files)
    except ValidationTimeout:
        message = f"Parser timeout after {timeout}s"
        payload = {"ok": False, "error": message}
    except MemoryError as exc:
        payload = {"ok": False, "error": "rdflib memory limit exceeded"}
        logger.warning(
            "rdflib memory error",
            extra={"stage": "validate", "validator": "rdflib", "error": str(exc)},
        )
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "rdflib_parse.json", payload)
    logger.warning(
        "rdflib validation failed",
        extra={"stage": "validate", "validator": "rdflib", "error": payload.get("error")},
    )
    return ValidationResult(ok=False, details=payload, output_files=[])


def validate_pronto(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Execute Pronto validation in an isolated subprocess and emit OBO Graphs when requested.

    Args:
        request: Validation request describing ontology inputs and output directories.
        logger: Structured logger for recording warnings and failures.

    Returns:
        ValidationResult with parsed ontology statistics, subprocess output,
        and any generated artifacts.

    Raises:
        ValidationTimeout: Propagated when Pronto takes longer than allowed.
    """

    try:
        timeout = request.config.defaults.validation.parser_timeout_sec
        payload: Dict[str, object] = {"file_path": str(request.file_path)}
        normalized_path: Optional[Path] = None
        if "obographs" in request.config.defaults.normalize_to:
            request.normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized_path = request.normalized_dir / (request.file_path.stem + ".json")
            payload["normalized_path"] = str(normalized_path)

        log_memory_usage(logger, stage="validate", event="before", validator="pronto")
        result_payload = _run_validator_subprocess("pronto", payload, timeout=timeout)
        log_memory_usage(logger, stage="validate", event="after", validator="pronto")
        result_payload.setdefault("ok", True)
        output_files: List[str] = []
        if normalized_path and result_payload.get("normalized_written"):
            output_files.append(str(normalized_path))
        _write_validation_json(request.validation_dir / "pronto_parse.json", result_payload)
        return ValidationResult(
            ok=bool(result_payload.get("ok")),
            details=result_payload,
            output_files=output_files,
        )
    except ValidationTimeout:
        payload = {"ok": False, "error": f"Parser timeout after {timeout}s"}
    except ValidatorSubprocessError as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive catch
        payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "pronto_parse.json", payload)
    logger.warning(
        "pronto validation failed",
        extra={"stage": "validate", "validator": "pronto", "error": payload.get("error")},
    )
    return ValidationResult(ok=False, details=payload, output_files=[])


def validate_owlready2(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Inspect ontologies with Owlready2 in a subprocess to count entities and catch parsing errors.

    Args:
        request: Validation request referencing the ontology to parse.
        logger: Logger for reporting failures or memory warnings.

    Returns:
        ValidationResult summarizing entity counts or failure details.

    Raises:
        None
    """
    try:
        size_mb = request.file_path.stat().st_size / (1024**2)
        limit = request.config.defaults.validation.skip_reasoning_if_size_mb
        if size_mb > limit:
            reason = f"Skipping reasoning for large file (> {limit} MB)"
            payload = {"ok": True, "skipped": True, "reason": reason}
            _write_validation_json(request.validation_dir / "owlready2_parse.json", payload)
            logger.info(
                "owlready2 reasoning skipped",
                extra={
                    "stage": "validate",
                    "validator": "owlready2",
                    "file_size_mb": round(size_mb, 2),
                    "limit_mb": limit,
                },
            )
            return ValidationResult(ok=True, details=payload, output_files=[])
        timeout = request.config.defaults.validation.parser_timeout_sec
        payload = {"file_path": str(request.file_path)}
        log_memory_usage(logger, stage="validate", event="before", validator="owlready2")
        result_payload = _run_validator_subprocess("owlready2", payload, timeout=timeout)
        log_memory_usage(logger, stage="validate", event="after", validator="owlready2")
        result_payload.setdefault("ok", True)
        _write_validation_json(request.validation_dir / "owlready2_parse.json", result_payload)
        return ValidationResult(
            ok=bool(result_payload.get("ok")), details=result_payload, output_files=[]
        )
    except ValidationTimeout:
        message = f"Parser timeout after {request.config.defaults.validation.parser_timeout_sec}s"
        payload = {"ok": False, "error": message}
    except ValidatorSubprocessError as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive catch
        payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "owlready2_parse.json", payload)
    logger.warning(
        "owlready2 validation failed",
        extra={"stage": "validate", "validator": "owlready2", "error": payload.get("error")},
    )
    return ValidationResult(ok=False, details=payload, output_files=[])


def validate_robot(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Run ROBOT CLI validation and conversion workflows when available.

    Args:
        request: Validation request detailing ontology paths and output locations.
        logger: Logger adapter for reporting warnings and CLI errors.

    Returns:
        ValidationResult describing generated outputs or encountered issues.

    Raises:
        None
    """
    robot_path = shutil.which("robot")
    result_payload: Dict[str, object]
    output_files: List[str] = []
    if not robot_path:
        result_payload = {"ok": True, "skipped": True, "reason": "robot binary not found"}
        _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
        logger.info(
            "robot not installed; skipping",
            extra={"stage": "validate", "validator": "robot", "skip": True},
        )
        return ValidationResult(ok=True, details=result_payload, output_files=[])

    normalized_path = request.normalized_dir / (request.file_path.stem + ".ttl")
    request.normalized_dir.mkdir(parents=True, exist_ok=True)
    report_path = request.validation_dir / "robot_report.tsv"
    try:
        log_memory_usage(logger, stage="validate", event="before", validator="robot")
        convert_cmd = [
            robot_path,
            "convert",
            "-i",
            str(request.file_path),
            "-o",
            str(normalized_path),
        ]
        report_cmd = [robot_path, "report", "-i", str(request.file_path), "-o", str(report_path)]
        subprocess.run(convert_cmd, check=True, capture_output=True)
        subprocess.run(report_cmd, check=True, capture_output=True)
        normalized_sha: Optional[str] = None
        streaming_prefix_hash: Optional[str] = None
        try:
            streaming_hash, streaming_prefix_hash = normalize_streaming(
                normalized_path,
                output_path=normalized_path,
                return_header_hash=True,
            )
            normalized_sha = streaming_hash
        except Exception as exc:  # pragma: no cover - best-effort normalization
            logger.warning(
                "robot normalization failed",
                extra={"stage": "validate", "validator": "robot", "error": str(exc)},
            )
        log_memory_usage(logger, stage="validate", event="after", validator="robot")
        output_files = [str(normalized_path), str(report_path)]
        result_payload = {"ok": True, "outputs": output_files}
        if normalized_sha:
            result_payload["normalized_sha256"] = normalized_sha
            result_payload["streaming_nt_sha256"] = normalized_sha
        if streaming_prefix_hash:
            result_payload["streaming_prefix_sha256"] = streaming_prefix_hash
        _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
        return ValidationResult(ok=True, details=result_payload, output_files=output_files)
    except subprocess.CalledProcessError as exc:
        result_payload = {"ok": False, "error": exc.stderr.decode("utf-8", errors="ignore")}
    except MemoryError as exc:
        result_payload = {"ok": False, "error": "robot memory limit exceeded"}
        _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
        logger.warning(
            "robot memory error",
            extra={"stage": "validate", "validator": "robot", "error": str(exc)},
        )
        return ValidationResult(ok=False, details=result_payload, output_files=output_files)
    except Exception as exc:  # pylint: disable=broad-except
        result_payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
    logger.warning(
        "robot validation failed",
        extra={"stage": "validate", "validator": "robot", "error": result_payload.get("error")},
    )
    return ValidationResult(ok=False, details=result_payload, output_files=output_files)


def validate_arelle(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Validate XBRL ontologies with Arelle CLI if installed.

    Args:
        request: Validation request referencing the ontology under test.
        logger: Logger used to communicate validation progress and failures.

    Returns:
        ValidationResult indicating whether the validation completed and
        referencing any produced log files.

    Raises:
        None
    """
    try:
        from arelle import Cntlr  # type: ignore

        entrypoint, artifacts = _prepare_xbrl_package(request, logger)
        controller = Cntlr.Cntlr(
            logFile=str(request.validation_dir / "arelle.log"), logToBuffer=True
        )
        log_memory_usage(logger, stage="validate", event="before", validator="arelle")
        controller.run(["--file", str(entrypoint)])
        log_memory_usage(logger, stage="validate", event="after", validator="arelle")
        payload = {
            "ok": True,
            "log": str(request.validation_dir / "arelle.log"),
            "entrypoint": str(entrypoint),
        }
        if artifacts:
            payload["artifacts"] = artifacts
        _write_validation_json(request.validation_dir / "arelle_validation.json", payload)
        outputs = [payload["log"], *(artifacts or [])]
        return ValidationResult(ok=True, details=payload, output_files=outputs)
    except ValueError as exc:
        payload = {"ok": False, "error": str(exc)}
        _write_validation_json(request.validation_dir / "arelle_validation.json", payload)
        logger.warning(
            "arelle package validation failed",
            extra={"stage": "validate", "validator": "arelle", "error": str(exc)},
        )
        return ValidationResult(ok=False, details=payload, output_files=[])
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
        _write_validation_json(request.validation_dir / "arelle_validation.json", payload)
        logger.warning(
            "arelle validation failed",
            extra={"stage": "validate", "validator": "arelle", "error": payload.get("error")},
        )
        return ValidationResult(ok=False, details=payload, output_files=[])


VALIDATORS = {
    "rdflib": validate_rdflib,
    "pronto": validate_pronto,
    "owlready2": validate_owlready2,
    "robot": validate_robot,
    "arelle": validate_arelle,
}


_plugins.register_plugin_registry("validator", VALIDATORS)
get_validator_registry()
_VALIDATOR_PLUGINS_LOADED = True


def _run_validator_task(
    validator: Callable[[ValidationRequest, logging.Logger], ValidationResult],
    request: ValidationRequest,
    logger: logging.Logger,
    *,
    use_semaphore: bool = True,
) -> ValidationResult:
    """Execute a single validator with exception guards."""

    start_time = time.perf_counter()
    before_mb = _current_memory_mb()
    log_memory_usage(logger, stage="validate", event="before", validator=request.name)
    semaphore = _acquire_validator_slot(request.config) if use_semaphore else None
    acquired = False
    if semaphore is not None:
        acquired = semaphore.acquire(timeout=request.config.defaults.validation.parser_timeout_sec)
        if not acquired:
            raise ValidationTimeout("validator concurrency limit prevented start")  # pragma: no cover
    try:
        result = validator(request, logger)
    except Exception as exc:  # pylint: disable=broad-except
        duration = time.perf_counter() - start_time
        after_mb = _current_memory_mb()
        log_memory_usage(logger, stage="validate", event="after", validator=request.name)
        payload = {"ok": False, "error": str(exc)}
        payload["metrics"] = {
            "duration_sec": round(duration, 3),
            "rss_mb_before": round(before_mb, 2),
            "rss_mb_after": round(after_mb, 2),
            "rss_mb_delta": round(after_mb - before_mb, 2),
        }
        _write_validation_json(request.validation_dir / f"{request.name}_parse.json", payload)
        logger.error(
            "validator crashed",
            extra={
                "stage": "validate",
                "validator": request.name,
                "error": payload.get("error"),
            },
        )
        return ValidationResult(ok=False, details=payload, output_files=[])
    else:
        duration = time.perf_counter() - start_time
        after_mb = _current_memory_mb()
        log_memory_usage(logger, stage="validate", event="after", validator=request.name)
        metrics = {
            "duration_sec": round(duration, 3),
            "rss_mb_before": round(before_mb, 2),
            "rss_mb_after": round(after_mb, 2),
            "rss_mb_delta": round(after_mb - before_mb, 2),
        }
        if not isinstance(result.details, dict):
            result.details = {"value": result.details}
        result.details.setdefault("metrics", {}).update(metrics)
        return result
    finally:
        if acquired and semaphore is not None:
            semaphore.release()


def _run_validator_in_process(name: str, request: ValidationRequest) -> ValidationResult:
    """Execute a validator inside a worker process."""

    validator = VALIDATORS.get(name)
    if validator is None:
        raise ValueError(f"Unknown validator '{name}'")

    process_logger = logging.getLogger(f"DocsToKG.Validator.{name}")
    if not process_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        process_logger.addHandler(handler)
        process_logger.setLevel(logging.INFO)
        process_logger.propagate = True

    return _run_validator_task(validator, request, process_logger, use_semaphore=False)


def run_validators(
    requests: Iterable[ValidationRequest], logger: logging.Logger
) -> Dict[str, ValidationResult]:
    """Execute registered validators and aggregate their results.

    Args:
        requests: Iterable of validation requests that specify validators to run.
        logger: Logger adapter shared across validation executions.

    Returns:
        Mapping from validator name to the corresponding ValidationResult.
    """

    request_list = list(requests)
    if not request_list:
        return {}

    def _determine_max_workers() -> int:
        for request in request_list:
            validation_config = getattr(request.config.defaults, "validation", None)
            if validation_config is not None and hasattr(
                validation_config, "max_concurrent_validators"
            ):
                value = int(validation_config.max_concurrent_validators)
                return max(1, min(8, value))
        return 2

    max_workers = _determine_max_workers()
    results: Dict[str, ValidationResult] = {}

    process_enabled = False
    process_validator_names: set[str] = set()
    for request in request_list:
        validation_config = getattr(request.config.defaults, "validation", None)
        if validation_config is None:
            continue
        if getattr(validation_config, "use_process_pool", False):
            process_enabled = True
            for name in getattr(validation_config, "process_pool_validators", []):
                if isinstance(name, str):
                    process_validator_names.add(name.strip().lower())

    thread_jobs: List[Tuple[Callable[[ValidationRequest, logging.Logger], ValidationResult], ValidationRequest]] = []
    process_requests: List[ValidationRequest] = []
    for request in request_list:
        validator = VALIDATORS.get(request.name)
        if not validator:
            continue
        normalized = request.name.lower()
        if process_enabled and normalized in process_validator_names:
            process_requests.append(request)
        else:
            thread_jobs.append((validator, request))

    futures: Dict[Any, ValidationRequest] = {}
    executors: List[Any] = []

    try:
        if thread_jobs:
            thread_workers = min(max_workers, len(thread_jobs))
            thread_workers = max(1, thread_workers)
            thread_executor = ThreadPoolExecutor(max_workers=thread_workers)
            executors.append(thread_executor)
            for validator, request in thread_jobs:
                future = thread_executor.submit(
                    _run_validator_task,
                    validator,
                    request,
                    logger,
                    use_semaphore=True,
                )
                futures[future] = request

        if process_requests:
            process_workers = min(max_workers, len(process_requests))
            process_workers = max(1, process_workers)
            process_executor = ProcessPoolExecutor(max_workers=process_workers)
            executors.append(process_executor)
            for request in process_requests:
                future = process_executor.submit(_run_validator_in_process, request.name, request)
                futures[future] = request

        for future in as_completed(futures):
            request = futures[future]
            try:
                results[request.name] = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                payload = {"ok": False, "error": str(exc)}
                _write_validation_json(
                    request.validation_dir / f"{request.name}_parse.json", payload
                )
                logger.error(
                    "validator crashed",
                    extra={
                        "stage": "validate",
                        "validator": request.name,
                        "error": payload.get("error"),
                    },
                )
                results[request.name] = ValidationResult(ok=False, details=payload, output_files=[])
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=False)

    return results


def _run_worker_cli(name: str, stdin_payload: str) -> None:
    """Execute a validator worker handler and emit JSON to stdout."""

    handler = _WORKER_DISPATCH.get(name)
    if handler is None:
        raise SystemExit(f"Unknown validator worker '{name}'")
    payload = json.loads(stdin_payload or "{}")
    result = handler(payload)
    sys.stdout.write(json.dumps(result))


def main() -> None:
    """Entry point for module execution providing validator worker dispatch.

    Args:
        None.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser(description="Ontology validator worker runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker", help="Run a validator worker")
    worker_parser.add_argument("name", choices=sorted(_WORKER_DISPATCH))
    args = parser.parse_args()

    if args.command == "worker":
        payload = sys.stdin.read()
        _run_worker_cli(args.name, payload)
    else:  # pragma: no cover - argparse enforces choices
        parser.error("Unknown command")


__all__ = [
    "ValidationRequest",
    "ValidationResult",
    "ValidationTimeout",
    "ValidatorSubprocessError",
    "load_validator_plugins",
    "normalize_streaming",
    "run_validators",
    "validate_rdflib",
    "validate_pronto",
    "validate_owlready2",
    "validate_robot",
    "validate_arelle",
    "VALIDATORS",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess dispatch
    main()
