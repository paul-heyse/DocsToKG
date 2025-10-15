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

Usage:
    from DocsToKG.OntologyDownload.validators import run_validators

    results = run_validators(requests, logger)
    print(results[\"rdflib\"].details)
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, cast

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
    """Parse ontologies with rdflib and optionally produce Turtle output.

    Args:
        request: Validation request describing the target ontology and output directories.
        logger: Logger adapter used for structured validation events.

    Returns:
        ValidationResult capturing success state, metadata, and generated files.

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
        if "ttl" in request.config.defaults.normalize_to:
            request.normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized_path = request.normalized_dir / (request.file_path.stem + ".ttl")
            graph.serialize(destination=normalized_path, format="turtle")
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
    """Execute Pronto-based validation and emit OBO Graphs when requested.

    Args:
        request: Validation request describing ontology inputs and output directories.
        logger: Structured logger for recording warnings and failures.

    Returns:
        ValidationResult with parsed ontology statistics and generated artifacts.

    Raises:
        ValidationTimeout: Propagated when Pronto takes longer than allowed.
    """

    def _load() -> pronto.Ontology:
        """Load the ontology into memory using Pronto."""
        return pronto.Ontology(request.file_path.as_posix())

    try:
        timeout = request.config.defaults.validation.parser_timeout_sec
        container: Dict[str, object] = {}

        def _execute() -> None:
            """Load the ontology and capture term statistics."""
            ontology_obj = _load()
            container["terms"] = len(list(ontology_obj.terms()))
            container["ontology"] = ontology_obj

        _log_memory(logger, "pronto", "before")
        _run_with_timeout(_execute, timeout)
        _log_memory(logger, "pronto", "after")
        ontology = container.get("ontology")
        if ontology is None:
            raise RuntimeError("Pronto failed to return ontology")
        ontology = cast(pronto.Ontology, ontology)
        payload = {"ok": True, "terms": container["terms"]}
        output_files: List[str] = []
        if "obographs" in request.config.defaults.normalize_to:
            request.normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized_path = request.normalized_dir / (request.file_path.stem + ".json")
            ontology.dump(normalized_path.as_posix(), format="obojson")
            output_files.append(str(normalized_path))
        _write_validation_json(request.validation_dir / "pronto_parse.json", payload)
        return ValidationResult(ok=True, details=payload, output_files=output_files)
    except ValidationTimeout:
        message = f"Parser timeout after {timeout}s"
        payload = {"ok": False, "error": message}
    except MemoryError as exc:
        payload = {"ok": False, "error": "pronto memory limit exceeded"}
        logger.warning(
            "pronto memory error",
            extra={"stage": "validate", "validator": "pronto", "error": str(exc)},
        )
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "pronto_parse.json", payload)
    logger.warning(
        "pronto validation failed",
        extra={"stage": "validate", "validator": "pronto", "error": payload.get("error")},
    )
    return ValidationResult(ok=False, details=payload, output_files=[])


def validate_owlready2(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Inspect ontologies with Owlready2 to count entities and catch parsing errors.

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
        _log_memory(logger, "owlready2", "before")
        ontology = owlready2.get_ontology(request.file_path.resolve().as_uri()).load()
        _log_memory(logger, "owlready2", "after")
        payload = {"ok": True, "entities": len(list(ontology.classes()))}
        _write_validation_json(request.validation_dir / "owlready2_parse.json", payload)
        return ValidationResult(ok=True, details=payload, output_files=[])
    except MemoryError as exc:
        payload = {
            "ok": False,
            "error": f"Memory limit exceeded parsing {request.file_path.name}. Consider skipping reasoning",
        }
        _write_validation_json(request.validation_dir / "owlready2_parse.json", payload)
        logger.warning(
            "owlready2 memory error",
            extra={"stage": "validate", "validator": "owlready2", "error": str(exc)},
        )
        return ValidationResult(ok=False, details=payload, output_files=[])
    except Exception as exc:  # pylint: disable=broad-except
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
    "validate_rdflib",
    "validate_pronto",
    "validate_owlready2",
    "validate_robot",
    "validate_arelle",
]
