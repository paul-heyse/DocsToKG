#!/usr/bin/env python3
"""Update NAVMAP metadata blocks to enumerate top-level classes and functions.

The Module Organization Guide requires each NAVMAP to list every public item in
source order. This helper script parses Python modules, extracts top-level
``class`` and ``def`` statements, and rewrites the NAVMAP ``sections`` array
accordingly. When a module lacks a NAVMAP entirely, the script now injects a
fresh block using the module's dotted path, the first line of its docstring (if
available), and the discovered top-level definitions.

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
ENCODING_PATTERN = re.compile(r"^#.*coding[:=]\s*[-\w.]+")


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


def parse_navmap_json(payload: str, path: Path) -> dict:
    """Parse NAVMAP JSON, tolerating inline ``#`` comments."""

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        cleaned_lines = []
        for line in payload.splitlines():
            if "#" in line:
                line = line.split("#", 1)[0].rstrip()
            cleaned_lines.append(line)
        cleaned_payload = "\n".join(cleaned_lines)
        cleaned_payload = re.sub(r",\s*(?=[}\]])", "", cleaned_payload)
        try:
            return json.loads(cleaned_payload)
        except json.JSONDecodeError as inner_exc:
            raise RuntimeError(f"Failed to parse NAVMAP JSON in {path}: {inner_exc}") from inner_exc


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


def compute_module_name(path: Path) -> str:
    """Return the dotted module path for ``path`` relative to ``src``."""

    try:
        rel = path.with_suffix("").relative_to(Path("src"))
    except ValueError:
        rel = path.with_suffix("")
    return ".".join(rel.parts)


def derive_purpose(module_ast: ast.Module) -> str:
    """Use the first line of the module docstring as the NAVMAP purpose."""

    docstring = ast.get_docstring(module_ast)
    if not docstring:
        return "Describe this module's responsibilities."

    summary = docstring.strip().split("\n\n", 1)[0].strip()
    first_line = summary.splitlines()[0].strip()
    if not first_line.endswith((".", "!", "?")):
        first_line += "."
    return first_line


def render_navmap_block(navmap: dict, include_trailing_blank: bool = False) -> List[str]:
    """Serialise a NAVMAP dictionary into a list of commented lines."""

    serialized = json.dumps(navmap, indent=2)
    navmap_lines = [NAVMAP_START + "\n"]
    navmap_lines.extend(f"# {line}\n" for line in serialized.splitlines())
    navmap_lines.append(NAVMAP_END + "\n")
    if include_trailing_blank:
        navmap_lines.append("\n")
    return navmap_lines


def update_file(path: Path) -> bool:
    """Ensure ``path`` contains an up-to-date NAVMAP. Returns ``True`` on change."""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    module_ast = ast.parse(text)
    navmap_info = extract_navmap(lines)

    module_name = compute_module_name(path)
    purpose_fallback = derive_purpose(module_ast)

    if navmap_info:
        start, end, payload = navmap_info
        navmap = parse_navmap_json(payload, path)

        if navmap.get("module") != module_name:
            navmap["module"] = module_name
        if not navmap.get("purpose"):
            navmap["purpose"] = purpose_fallback
        navmap["sections"] = build_sections(module_ast)

        navmap_lines = render_navmap_block(navmap)
        new_lines = lines[:start] + navmap_lines + lines[end + 1 :]
    else:
        navmap = {
            "module": module_name,
            "purpose": purpose_fallback,
            "sections": build_sections(module_ast),
        }
        navmap_lines = render_navmap_block(navmap, include_trailing_blank=True)

        leading: List[str] = []
        start_idx = 0
        if lines and lines[0].startswith("#!"):
            leading.append(lines[0])
            start_idx = 1
        if len(lines) > start_idx and ENCODING_PATTERN.match(lines[start_idx]):
            leading.append(lines[start_idx])
            start_idx += 1
        remaining = lines[start_idx:]
        new_lines = leading + navmap_lines + remaining

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
