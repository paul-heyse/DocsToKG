"""
Ontology Validation Pipeline

This module defines validators that run after ontology documents are fetched.
Each validator parses the downloaded artifact, records structured results, and
optionally emits normalized representations for downstream document processing.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, cast

import rdflib
import pronto
import owlready2

from .config import ResolvedConfig


@dataclass(slots=True)
class ValidationRequest:
    """Parameters describing a single validation task.

    Attributes:
        name: Identifier of the validator to execute.
        file_path: Path to the ontology document to inspect.
        normalized_dir: Directory used to write normalized artifacts.
        validation_dir: Directory for validator reports and logs.
        config: Resolved configuration that supplies timeout thresholds.
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
    """Raised when a validation task exceeds the configured timeout."""


def _write_validation_json(path: Path, payload: MutableMapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _run_with_timeout(func, timeout_sec: int) -> None:
    if platform.system() in ("Linux", "Darwin"):
        import signal

        class _Alarm(Exception):
            pass

        def _handler(signum, frame):  # pragma: no cover - platform dependent
            raise ValidationTimeout()

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_sec)
        try:
            func()
        finally:
            signal.alarm(0)
    else:  # Windows
        exc: List[BaseException] = []

        def _target() -> None:
            try:
                func()
            except BaseException as error:  # pragma: no cover - delegated exception handling
                exc.append(error)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)
        if thread.is_alive():
            raise ValidationTimeout()
        if exc:
            raise exc[0]


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

    def _parse() -> None:
        graph.parse(request.file_path.as_posix())

    try:
        _run_with_timeout(_parse, request.config.defaults.validation.parser_timeout_sec)
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
        payload = {"ok": False, "error": "rdflib parser timeout"}
    except MemoryError as exc:
        payload = {"ok": False, "error": "rdflib memory limit exceeded"}
        logger.warning(
            "rdflib memory error",
            extra={"stage": "validate", "error": str(exc)},
        )
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "rdflib_parse.json", payload)
    logger.warning("rdflib validation failed", extra={"stage": "validate", "error": payload.get("error")})
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
        return pronto.Ontology(request.file_path.as_posix())

    try:
        container: Dict[str, object] = {}

        def _execute() -> None:
            ontology_obj = _load()
            container["terms"] = len(list(ontology_obj.terms()))
            container["ontology"] = ontology_obj

        _run_with_timeout(_execute, request.config.defaults.validation.parser_timeout_sec)
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
        payload = {"ok": False, "error": "pronto parser timeout"}
    except MemoryError as exc:
        payload = {"ok": False, "error": "pronto memory limit exceeded"}
        logger.warning(
            "pronto memory error",
            extra={"stage": "validate", "error": str(exc)},
        )
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "pronto_parse.json", payload)
    logger.warning("pronto validation failed", extra={"stage": "validate", "error": payload.get("error")})
    return ValidationResult(ok=False, details=payload, output_files=[])


def validate_owlready2(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Inspect ontologies with Owlready2 to count entities and catch parsing errors.

    Args:
        request: Validation request referencing the ontology to parse.
        logger: Logger for reporting failures or memory warnings.

    Returns:
        ValidationResult summarizing entity counts or failure details.
    """
    try:
        ontology = owlready2.get_ontology(request.file_path.resolve().as_uri()).load()
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
            extra={"stage": "validate", "error": str(exc)},
        )
        return ValidationResult(ok=False, details=payload, output_files=[])
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
        _write_validation_json(request.validation_dir / "owlready2_parse.json", payload)
        logger.warning("owlready2 validation failed", extra={"stage": "validate", "error": payload.get("error")})
        return ValidationResult(ok=False, details=payload, output_files=[])


def validate_robot(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Run ROBOT CLI validation and conversion workflows when available.

    Args:
        request: Validation request detailing ontology paths and output locations.
        logger: Logger adapter for reporting warnings and CLI errors.

    Returns:
        ValidationResult describing generated outputs or encountered issues.
    """
    robot_path = shutil.which("robot")
    result_payload: Dict[str, object]
    output_files: List[str] = []
    if not robot_path:
        result_payload = {"ok": True, "skipped": True, "reason": "robot binary not found"}
        _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
        logger.info("robot not installed; skipping", extra={"stage": "validate", "skip": True})
        return ValidationResult(ok=True, details=result_payload, output_files=[])

    normalized_path = request.normalized_dir / (request.file_path.stem + ".ttl")
    request.normalized_dir.mkdir(parents=True, exist_ok=True)
    report_path = request.validation_dir / "robot_report.tsv"
    try:
        convert_cmd = [robot_path, "convert", "-i", str(request.file_path), "-o", str(normalized_path)]
        report_cmd = [robot_path, "report", "-i", str(request.file_path), "-o", str(report_path)]
        subprocess.run(convert_cmd, check=True, capture_output=True)
        subprocess.run(report_cmd, check=True, capture_output=True)
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
            extra={"stage": "validate", "error": str(exc)},
        )
        return ValidationResult(ok=False, details=result_payload, output_files=output_files)
    except Exception as exc:  # pylint: disable=broad-except
        result_payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
    logger.warning("robot validation failed", extra={"stage": "validate", "error": result_payload.get("error")})
    return ValidationResult(ok=False, details=result_payload, output_files=output_files)


def validate_arelle(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Validate XBRL ontologies with Arelle CLI if installed.

    Args:
        request: Validation request referencing the ontology under test.
        logger: Logger used to communicate validation progress and failures.

    Returns:
        ValidationResult indicating whether the validation completed and
        referencing any produced log files.
    """
    try:
        from arelle import Cntlr  # type: ignore

        controller = Cntlr.Cntlr(logFile=str(request.validation_dir / "arelle.log"), logToBuffer=True)
        controller.run(["--file", str(request.file_path)])
        payload = {"ok": True, "log": str(request.validation_dir / "arelle.log")}
        _write_validation_json(request.validation_dir / "arelle_validation.json", payload)
        return ValidationResult(ok=True, details=payload, output_files=[payload["log"]])
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
        _write_validation_json(request.validation_dir / "arelle_validation.json", payload)
        logger.warning("arelle validation failed", extra={"stage": "validate", "error": payload.get("error")})
        return ValidationResult(ok=False, details=payload, output_files=[])


VALIDATORS = {
    "rdflib": validate_rdflib,
    "pronto": validate_pronto,
    "owlready2": validate_owlready2,
    "robot": validate_robot,
    "arelle": validate_arelle,
}


def run_validators(requests: Iterable[ValidationRequest], logger: logging.Logger) -> Dict[str, ValidationResult]:
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
                extra={"stage": "validate", "validator": request.name, "error": payload.get("error")},
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
