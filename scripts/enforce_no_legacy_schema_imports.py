#!/usr/bin/env python3
"""Fail fast when code imports the deprecated DocParsing.schemas shim."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    ROOT / "src" / "DocsToKG" / "DocParsing" / "schemas.py",
    ROOT / "tests" / "docparsing" / "test_import_shim.py",
}
SEARCH_ROOTS = [ROOT / "src", ROOT / "tests"]


def _scan_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:  # pragma: no cover - binary files
        return False
    return "DocParsing.schemas" in text


def main() -> int:
    offenders: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path in ALLOWED:
                continue
            if _scan_file(path):
                offenders.append(path.relative_to(ROOT))

    if offenders:
        joined = "\n".join(str(path) for path in offenders)
        print(
            "Found deprecated imports of DocsToKG.DocParsing.schemas:\n"
            f"{joined}\nUpdate the modules to import from DocsToKG.DocParsing.formats.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
