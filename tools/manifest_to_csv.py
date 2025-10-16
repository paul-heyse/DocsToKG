from __future__ import annotations

"""Produce a manifest.last.csv file from the DocsToKG manifest JSONL log."""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

HEADER = [
    "work_id",
    "title",
    "publication_year",
    "resolver",
    "url",
    "classification",
    "path",
    "sha256",
    "content_length",
    "etag",
    "last_modified",
]


def convert_manifest_to_csv(src: Path, dest: Path) -> None:
    """Generate a CSV containing the last manifest entry for each work."""

    last_records: "OrderedDict[str, Dict[str, Optional[str]]]" = OrderedDict()
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
            last_records[work_id] = record

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for record in last_records.values():
            row = {
                field: "" if record.get(field) is None else record.get(field, "")
                for field in HEADER
            }
            writer.writerow(row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a DocsToKG manifest JSONL log into a manifest.last.csv file."
    )
    parser.add_argument("src_jsonl", type=Path, help="Path to the manifest JSONL log.")
    parser.add_argument(
        "out_csv", type=Path, help="Destination path for the manifest.last.csv file."
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    convert_manifest_to_csv(args.src_jsonl, args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

