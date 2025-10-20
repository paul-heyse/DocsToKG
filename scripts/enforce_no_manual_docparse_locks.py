#!/usr/bin/env python3
"""Fail when DocParsing reintroduces bespoke lock sentinels or JSONL loops."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "DocsToKG" / "DocParsing"
FORBIDDEN_PATTERNS = {
    "_iter_jsonl_records": "JSONL iteration must use the library-backed adapter.",
    ".lock": "Lock files should be managed through core.concurrency.acquire_lock (FileLock).",
}
ALLOWED_LOCK_FILES = {
    SRC_ROOT / "core" / "concurrency.py",
}


def _scan_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    issues: list[str] = []
    for pattern, message in FORBIDDEN_PATTERNS.items():
        if pattern == ".lock" and path in ALLOWED_LOCK_FILES:
            continue
        if pattern in text:
            issues.append(f"{path.relative_to(ROOT)} -> {message}")
    return issues


def main() -> int:
    offenders: list[str] = []
    for py_file in SRC_ROOT.rglob("*.py"):
        offenders.extend(_scan_file(py_file))

    if offenders:
        joined = "\n".join(offenders)
        print(
            "Found discouraged DocParsing patterns:\n"
            f"{joined}\n"
            "Use the FileLock helper and the jsonlines adapter instead.",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
