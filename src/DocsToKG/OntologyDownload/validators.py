"""
Ontology Validation Pipeline

This module implements the post-download validation workflow that verifies
ontology integrity, generates normalized artifacts, and captures structured
telemetry for DocsToKG. Validators can leverage optional dependencies such as
rdflib, Pronto, Owlready2, ROBOT, and Arelle while falling back gracefully
when utilities are absent.

Key Features:
- Uniform :class:`ValidationRequest` / :class:`ValidationResult` data model
- Timeout and memory instrumentation for resource-intensive validators
- JSON reporting helpers compatible with automated documentation generation
- Pluggable registry enabling selective validator execution
- Canonical Turtle normalization with deterministic SHA-256 hashing
- Subprocess isolation for memory-intensive Pronto and Owlready2 validators

Usage:
    from DocsToKG.OntologyDownload.validators import run_validators

    results = run_validators(requests, logger)
    print(results[\"rdflib\"].details)
"""

from __future__ import annotations

import hashlib
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
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional

import psutil

from .config import ResolvedConfig
from .optdeps import get_owlready2, get_pronto, get_rdflib

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

    with source.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    lines.sort()
    destination.write_text("".join(lines), encoding="utf-8")


def normalize_streaming(
    source: Path,
    *,
    output_path: Optional[Path] = None,
    graph=None,
) -> str:
    """Normalize ontologies using external sort to ensure deterministic output.

    The streaming path serializes triples to a temporary file, leverages the
    platform ``sort`` utility (when available) to order the triples
    lexicographically, and streams the sorted output while computing the
    canonical SHA-256 hash. Callers may supply an ``output_path`` to persist the
    normalized Turtle document without storing the entire content in memory.
    """

    graph_obj = graph if graph is not None else rdflib.Graph()
    if graph is None:
        graph_obj.parse(source.as_posix())

    namespace_manager = getattr(graph_obj, "namespace_manager", None)
    prefix_map: Dict[str, str] = {}
    if namespace_manager and hasattr(namespace_manager, "namespaces"):
        for prefix, namespace in namespace_manager.namespaces():
            key = prefix or ""
            prefix_map[key] = str(namespace)

    with tempfile.TemporaryDirectory() as temp_dir:
        unsorted_path = Path(temp_dir) / "triples.nt"
        with unsorted_path.open("w", encoding="utf-8", newline="\n") as handle:
            for subject, predicate, obj in graph_obj:
                line = (
                    f"{_term_to_string(subject, namespace_manager)} "
                    f"{_term_to_string(predicate, namespace_manager)} "
                    f"{_term_to_string(obj, namespace_manager)} .\n"
                )
                handle.write(line)

        sorted_path = Path(temp_dir) / "triples.sorted"
        _sort_triple_file(unsorted_path, sorted_path)

        writer = None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = output_path.open("wb")

        sha256 = hashlib.sha256()

        def _write(data: bytes) -> None:
            if not data:
                return
            sha256.update(data)
            if writer:
                writer.write(data)

        bnode_map: Dict[str, str] = {}

        try:
            for key in sorted(prefix_map):
                prefix_line = f"@prefix {key + ':' if key else ':'} <{prefix_map[key]}> .\n"
                _write(prefix_line.encode("utf-8"))

            with sorted_path.open("r", encoding="utf-8") as handle:
                first_line = handle.readline()
                has_triples = bool(first_line)
                if prefix_map and has_triples:
                    _write(b"\n")
                if has_triples:
                    canonical_first = _canonicalize_blank_nodes_line(first_line.rstrip("\n"), bnode_map)
                    _write(canonical_first.encode("utf-8") + b"\n")
                    for raw_line in handle:
                        canonical_line = _canonicalize_blank_nodes_line(raw_line.rstrip("\n"), bnode_map)
                        _write(canonical_line.encode("utf-8") + b"\n")
        finally:
            if writer:
                writer.close()

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


def _run_validator_subprocess(
    name: str, payload: Dict[str, object], *, timeout: int
) -> Dict[str, object]:
    """Execute a validator worker module within a subprocess.

    The subprocess workflow enforces parser timeouts, returns JSON payloads,
    and helps release memory held by heavy libraries such as Pronto and
    Owlready2 after each validation completes.
    """

    worker_script = Path(__file__).resolve().with_name("validator_workers.py")
    command = [sys.executable, str(worker_script), name]
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_path}" if existing_path else str(project_root)
    )

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
        normalized_sha: Optional[str] = None
        if "ttl" in request.config.defaults.normalize_to:
            request.normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized_path = request.normalized_dir / (request.file_path.stem + ".ttl")
            threshold_mb = (
                request.config.defaults.validation.streaming_normalization_threshold_mb
            )
            file_size = request.file_path.stat().st_size
            use_streaming = file_size >= threshold_mb * (1024**2)
            if use_streaming:
                try:
                    normalized_sha = normalize_streaming(
                        request.file_path,
                        output_path=normalized_path,
                        graph=graph,
                    )
                    normalization_mode = "streaming"
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "streaming normalization failed, falling back to in-memory",
                        extra={
                            "stage": "validate",
                            "validator": "rdflib",
                            "error": str(exc),
                        },
                    )
            if normalized_sha is None:
                try:
                    canonical_ttl = _canonicalize_turtle(graph)
                    normalized_path.write_text(canonical_ttl, encoding="utf-8")
                    normalized_sha = hashlib.sha256(canonical_ttl.encode("utf-8")).hexdigest()
                except AttributeError:
                    graph.serialize(destination=normalized_path, format="turtle")
                    canonical_ttl = normalized_path.read_text(encoding="utf-8")
                    normalized_sha = hashlib.sha256(canonical_ttl.encode("utf-8")).hexdigest()
            payload["normalized_sha256"] = normalized_sha
            payload["normalization_mode"] = normalization_mode
            output_files.append(str(normalized_path))
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
