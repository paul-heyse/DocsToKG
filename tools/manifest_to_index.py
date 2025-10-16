"""Generate a manifest index JSON from the DocsToKG download JSONL log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def convert_manifest_to_index(src: Path, dest: Path) -> None:
    """Build a manifest index mapping work IDs to PDF metadata."""

    index: Dict[str, Dict[str, Optional[str]]] = {}
    with src.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_type") != "manifest":
                continue
            work_id = record.get("work_id")
            if not work_id:
                continue
            classification = record.get("classification")
            path = record.get("path")
            payload: Dict[str, Optional[str]] = {
                "classification": classification,
                "pdf_path": None,
                "sha256": None,
            }
            if (
                path
                and classification
                and (classification.startswith("pdf") or classification == "cached")
            ):
                payload["pdf_path"] = path
                payload["sha256"] = record.get("sha256")
            index[work_id] = payload

    ordered = dict(sorted(index.items(), key=lambda item: item[0]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a DocsToKG manifest JSONL log into a compact index JSON."
    )
    parser.add_argument("src_jsonl", type=Path, help="Path to the manifest JSONL log.")
    parser.add_argument(
        "out_json", type=Path, help="Destination path for the manifest index JSON file."
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    convert_manifest_to_index(args.src_jsonl, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
