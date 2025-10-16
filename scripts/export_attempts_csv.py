#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "scripts.export_attempts_csv",
#   "purpose": "Utility script for export attempts csv workflows",
#   "sections": [
#     {
#       "id": "globals",
#       "name": "Globals",
#       "anchor": "GLOB",
#       "kind": "infra"
#     },
#     {
#       "id": "private_helpers",
#       "name": "Private Helpers",
#       "anchor": "PH",
#       "kind": "internal"
#     },
#     {
#       "id": "public_functions",
#       "name": "Public Functions",
#       "anchor": "PF",
#       "kind": "api"
#     },
#     {
#       "id": "module_entry_points",
#       "name": "Module Entry Points",
#       "anchor": "MEP",
#       "kind": "cli"
#     }
#   ]
# }
# === /NAVMAP ===

"""Convert ContentDownload JSONL attempt logs into the legacy CSV format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional
# --- Globals ---

CSV_HEADER: List[str] = [
    "timestamp",
    "work_id",
    "resolver_name",
    "resolver_order",
    "url",
    "status",
    "http_status",
    "content_type",
    "elapsed_ms",
    "resolver_wall_time_ms",
    "reason",
    "sha256",
    "content_length",
    "dry_run",
    "metadata",
]
# --- Private Helpers ---


def _iter_attempt_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("record_type") != "attempt":
                continue
            yield data
# --- Public Functions ---


def export_attempts_jsonl_to_csv(input_path: Path, output_path: Path) -> None:
    rows = list(_iter_attempt_records(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            metadata = row.get("metadata")
            if metadata and not isinstance(metadata, str):
                metadata = json.dumps(metadata, sort_keys=True)
            writer.writerow(
                {
                    "timestamp": row.get("timestamp"),
                    "work_id": row.get("work_id"),
                    "resolver_name": row.get("resolver_name"),
                    "resolver_order": row.get("resolver_order"),
                    "url": row.get("url"),
                    "status": row.get("status"),
                    "http_status": row.get("http_status"),
                    "content_type": row.get("content_type"),
                    "elapsed_ms": row.get("elapsed_ms"),
                    "resolver_wall_time_ms": row.get("resolver_wall_time_ms"),
                    "reason": row.get("reason"),
                    "sha256": row.get("sha256"),
                    "content_length": row.get("content_length"),
                    "dry_run": row.get("dry_run"),
                    "metadata": metadata or "",
                }
            )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to attempts JSONL file")
    parser.add_argument("output", type=Path, help="Path to CSV output file")
    return parser.parse_args(argv)
# --- Module Entry Points ---


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    export_attempts_jsonl_to_csv(args.input, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
