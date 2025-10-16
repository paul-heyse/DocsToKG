#!/usr/bin/env python3
"""Update NAVMAP metadata blocks to enumerate top-level classes and functions.

The Module Organization Guide requires each NAVMAP to list every public item in
source order. This helper script parses Python modules, extracts top-level
``class`` and ``def`` statements, and rewrites the NAVMAP ``sections`` array
accordingly.

Usage:
    python scripts/update_navmaps.py [path ...]

When no paths are supplied, all Python files under ``src/DocsToKG`` are
processed.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List

NAVMAP_START = "# === NAVMAP v1 ==="
NAVMAP_END = "# === /NAVMAP ==="


def slugify(name: str) -> str:
    """Convert ``name`` into a lower-case, hyphen separated slug."""

    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def extract_navmap(lines: List[str]) -> tuple[int, int, str] | None:
    """Locate the NAVMAP block within ``lines``.

    Returns ``(start_index, end_index, json_payload)`` where the indices
    correspond to the lines containing the start/end markers (inclusive), and
    ``json_payload`` is the decoded JSON string without ``#`` prefixes.
    """

    try:
        start = lines.index(NAVMAP_START + "\n")
    except ValueError:
        try:
            start = lines.index(NAVMAP_START)
        except ValueError:
            return None
    try:
        end = lines.index(NAVMAP_END + "\n", start)
    except ValueError:
        try:
            end = lines.index(NAVMAP_END, start)
        except ValueError:
            raise RuntimeError("NAVMAP start marker found without matching end marker")

    payload_lines = []
    for line in lines[start + 1 : end]:
        line = line.rstrip("\n")
        if line.startswith("# "):
            payload_lines.append(line[2:])
        elif line.startswith("#"):
            payload_lines.append(line[1:])
        else:
            payload_lines.append(line)
    payload = "\n".join(payload_lines).strip()
    return start, end, payload


def build_sections(module_ast: ast.Module) -> List[dict]:
    """Generate NAVMAP section entries for top-level classes and functions."""

    sections: List[dict] = []
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            slug = slugify(name)
            sections.append(
                {
                    "id": slug,
                    "name": name,
                    "anchor": f"function-{slug}",
                    "kind": "function",
                }
            )
        elif isinstance(node, ast.AsyncFunctionDef):
            name = node.name
            slug = slugify(name)
            sections.append(
                {
                    "id": slug,
                    "name": name,
                    "anchor": f"function-{slug}",
                    "kind": "function",
                }
            )
        elif isinstance(node, ast.ClassDef):
            name = node.name
            slug = slugify(name)
            sections.append(
                {
                    "id": slug,
                    "name": name,
                    "anchor": f"class-{slug}",
                    "kind": "class",
                }
            )
    return sections


def update_file(path: Path) -> bool:
    """Rewrite the NAVMAP in ``path`` if present. Returns ``True`` on change."""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    navmap_info = extract_navmap(lines)
    if not navmap_info:
        return False

    start, end, payload = navmap_info
    try:
        navmap = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse NAVMAP JSON in {path}: {exc}") from exc

    module_ast = ast.parse(text)
    navmap["sections"] = build_sections(module_ast)

    serialized = json.dumps(navmap, indent=2)
    navmap_lines = [NAVMAP_START + "\n"]
    navmap_lines.extend(f"# {line}\n" for line in serialized.splitlines())
    navmap_lines.append(NAVMAP_END + "\n")

    new_lines = lines[:start] + navmap_lines + lines[end + 1 :]
    new_text = "".join(new_lines)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def iter_targets(args: Iterable[str]) -> Iterable[Path]:
    if args:
        for arg in args:
            p = Path(arg)
            if p.is_dir():
                yield from p.rglob("*.py")
            else:
                yield p
    else:
        root = Path("src/DocsToKG")
        yield from root.rglob("*.py")


def main(argv: List[str]) -> int:
    changed = 0
    for path in iter_targets(argv):
        if update_file(path):
            changed += 1
    print(f"Updated NAVMAPs in {changed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
