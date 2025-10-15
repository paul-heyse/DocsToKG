"""Subprocess workers for memory-intensive ontology validators.

These helpers execute within isolated Python interpreters to release
memory promptly after heavy validation steps such as Pronto and
Owlready2 parsing. Each worker reads a JSON payload from ``stdin`` and
emits a JSON document to ``stdout`` describing the validation result.

The module is intentionally lightweight so spawning subprocesses remains
fast. It relies on the optional dependency accessors in
``DocsToKG.OntologyDownload.optdeps`` which transparently provide stub
implementations during testing when the real dependencies are not
installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .optdeps import get_owlready2, get_pronto

pronto = get_pronto()
owlready2 = get_owlready2()


def _run_pronto(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Pronto validation logic in a subprocess context and emit JSON.

    Args:
        payload: Mapping containing ``file_path`` of the ontology and optional
            ``normalized_path`` where serialized output should be written.

    Returns:
        Dictionary describing the validation outcome, including metrics such as
        ``terms`` and an ``ok`` flag.
    """

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


def _run_owlready2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Owlready2 validation logic in a subprocess context and emit JSON.

    Args:
        payload: Mapping containing ``file_path`` that should be parsed by Owlready2.

    Returns:
        Dictionary containing validation status metadata such as entity counts.
    """

    file_path = Path(payload["file_path"])
    ontology = owlready2.get_ontology(file_path.resolve().as_uri()).load()
    entities = len(list(ontology.classes()))
    return {"ok": True, "entities": entities}


def main() -> None:
    """Parse command line arguments and execute the requested worker.

    Args:
        None

    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="Ontology validator worker")
    parser.add_argument("worker", choices={"pronto", "owlready2"})
    args = parser.parse_args()

    payload = json.loads(sys.stdin.read() or "{}")
    if args.worker == "pronto":
        result = _run_pronto(payload)
    else:
        result = _run_owlready2(payload)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    main()
