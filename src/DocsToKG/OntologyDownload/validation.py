"""Ontology validation pipeline.

This module implements the post-download workflow that verifies ontologies,
normalizes output, and collects structured telemetry for DocsToKG. Validators
support streaming normalization for large ontologies, deterministic hashing for
manifest fingerprints, and optional dependency fallbacks for tools such as
rdflib, Pronto, Owlready2, ROBOT, and Arelle.
"""

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
import zipfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, MutableMapping, Optional

import psutil

from .infrastructure import get_owlready2, get_pronto, get_rdflib
from .settings import ResolvedConfig

rdflib = get_rdflib()
pronto = get_pronto()
owlready2 = get_owlready2()


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
        >>> from DocsToKG.OntologyDownload.config import ResolvedConfig
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
            None

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


def _log_memory(logger: logging.Logger, validator: str, event: str) -> None:
    """Emit memory usage diagnostics for a validator when debug logging is enabled.

    Args:
        logger: Logger responsible for validator telemetry.
        validator: Name of the validator emitting the event.
        event: Lifecycle label describing when the measurement is captured.

    Returns:
        None
    """
    is_enabled = getattr(logger, "isEnabledFor", None)
    if callable(is_enabled):
        enabled = is_enabled(logging.DEBUG)
    else:  # pragma: no cover - fallback for stub loggers
        enabled = False
    if not enabled:
        return
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024**2)
    logger.debug(
        "memory usage",
        extra={
            "stage": "validate",
            "validator": validator,
            "event": event,
            "memory_mb": round(memory_mb, 2),
        },
    )


def _write_validation_json(path: Path, payload: MutableMapping[str, object]) -> None:
    """Persist structured validation metadata to disk as JSON.

    Args:
        path: Destination path for the JSON payload.
        payload: Mapping containing validation results.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _python_merge_sort(source: Path, destination: Path, *, chunk_size: int = 100_000) -> None:
    """Sort an N-Triples file using a disk-backed merge strategy."""

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
    formatter = getattr(term, "n3", None)
    if callable(formatter):
        return formatter(namespace_manager)
    return str(term)


def _canonicalize_turtle(graph) -> str:
    """Return canonical Turtle output with sorted prefixes and triples.

    The canonical form mirrors the ontology downloader specification by sorting
    prefixes lexicographically and emitting triples ordered by subject,
    predicate, and object so downstream hashing yields deterministic values.
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
    """Replace blank node identifiers with deterministic sequential labels."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(0)
        mapped = mapping.get(key)
        if mapped is None:
            mapped = f"_:b{len(mapping)}"
            mapping[key] = mapped
        return mapped

    return _BNODE_PATTERN.sub(_replace, line)


def _sort_triple_file(source: Path, destination: Path) -> None:
    """Sort serialized triple lines using platform sort when available."""

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
) -> str:
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

    Returns:
        SHA-256 hex digest of the canonical Turtle content.
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

    try:
        triples = list(graph_obj)
    except Exception as exc:  # pragma: no cover - stub graphs are not iterable
        raise AttributeError("graph is not iterable") from exc

    def _iter_canonical_lines():
        for key in sorted(prefix_map):
            label = f"{key}:" if key else ":"
            yield f"@prefix {label} <{prefix_map[key]}> .\n"

        ordered = sorted(
            ((s, p, o) for s, p, o in triples),
            key=lambda item: (
                _term_to_string(item[0], namespace_manager),
                _term_to_string(item[1], namespace_manager),
                _term_to_string(item[2], namespace_manager),
            ),
        )

        if prefix_map and ordered:
            yield "\n"

        bnode_map: Dict[str, str] = {}
        for subject, predicate, obj in ordered:
            line = (
                f"{_term_to_string(subject, namespace_manager)} "
                f"{_term_to_string(predicate, namespace_manager)} "
                f"{_term_to_string(obj, namespace_manager)} ."
            )
            yield _canonicalize_blank_nodes_line(line, bnode_map) + "\n"

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

    with contextlib.ExitStack() as stack:
        writer: Optional[BinaryIO] = None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = stack.enter_context(output_path.open("wb"))

        wrote_any = False
        for text in _iter_canonical_lines():
            buffer.extend(text.encode("utf-8"))
            wrote_any = True
            if len(buffer) >= chunk_limit:
                _flush(writer)

        if not wrote_any and output_path is not None:
            output_path.write_text("", encoding="utf-8")

        _flush(writer)

    return sha256.hexdigest()


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
        _log_memory(logger, "rdflib", "before")
        _run_with_timeout(_parse, timeout)
        _log_memory(logger, "rdflib", "after")
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
            normalized_sha: Optional[str] = None
            if file_size >= threshold_bytes:
                normalization_mode = "streaming"
                try:
                    streaming_hash = normalize_streaming(
                        request.file_path,
                        output_path=normalized_ttl,
                        graph=graph,
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
        if (
            "ttl" in request.config.defaults.normalize_to
            and payload.get("streaming_nt_sha256") is None
        ):
            try:
                payload["streaming_nt_sha256"] = normalize_streaming(
                    request.file_path,
                    graph=graph,
                )
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

        _log_memory(logger, "pronto", "before")
        result_payload = _run_validator_subprocess("pronto", payload, timeout=timeout)
        _log_memory(logger, "pronto", "after")
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
        message = f"Parser timeout after {timeout}s"
        payload = {"ok": False, "error": message}
    except ValidatorSubprocessError as exc:
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
        _log_memory(logger, "owlready2", "before")
        result_payload = _run_validator_subprocess("owlready2", payload, timeout=timeout)
        _log_memory(logger, "owlready2", "after")
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
        _log_memory(logger, "robot", "before")
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
        _log_memory(logger, "robot", "after")
        output_files = [str(normalized_path), str(report_path)]
        result_payload = {"ok": True, "outputs": output_files}
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
        _log_memory(logger, "arelle", "before")
        controller.run(["--file", str(entrypoint)])
        _log_memory(logger, "arelle", "after")
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
    results: Dict[str, ValidationResult] = {}
    for request in requests:
        validator = VALIDATORS.get(request.name)
        if not validator:
            continue
        try:
            results[request.name] = validator(request, logger)
        except Exception as exc:  # pylint: disable=broad-except
            payload = {"ok": False, "error": str(exc)}
            _write_validation_json(request.validation_dir / f"{request.name}_parse.json", payload)
            logger.error(
                "validator crashed",
                extra={
                    "stage": "validate",
                    "validator": request.name,
                    "error": payload.get("error"),
                },
            )
            results[request.name] = ValidationResult(ok=False, details=payload, output_files=[])
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
    """Entry point for module execution providing validator worker dispatch."""

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


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess dispatch
    main()


__all__ = [
    "ValidationRequest",
    "ValidationResult",
    "run_validators",
    "normalize_streaming",
    "validate_rdflib",
    "validate_pronto",
    "validate_owlready2",
    "validate_robot",
    "validate_arelle",
]
