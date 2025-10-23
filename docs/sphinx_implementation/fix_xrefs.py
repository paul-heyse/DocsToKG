#!/usr/bin/env python3
"""
Auto-fix Sphinx cross-reference warnings by consulting local + intersphinx inventories.

- Parses 'WARNING: py:* reference target not found: ...' lines from a Sphinx log
- Loads objects.inv from your build and from any intersphinx URLs you pass
- Picks the correct role and fully-qualified target (class/func/meth/etc.)
- Rewrites .rst and .md sources in-place (use --dry-run to preview)

Usage:
  python tools/fix_xrefs.py \
    --root docs/source \
    --build docs/_build/dirhtml \
    --log build.log \
    --intersphinx https://docs.python.org/3 https://tenacity.readthedocs.io/en/stable/objects.inv \
    --write
"""

from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sphobjinv as soi  # pip install sphobjinv

WARNING_RE = re.compile(
    r"""
    ^(?P<file>[^:\n]+?)          # path
    (?::(?P<line>\d+))?          # optional line
    :\s*WARNING:\s*
    (?P<role>[\w:]+)             # e.g. py:class
    \s+reference\ target\ not\ found:\s*
    (?P<target>.+?)\s*
    \[ref\.\w+\]                 # [ref.class] etc
    $
    """,
    re.VERBOSE,
)

# Map Sphinx roles -> inventory roles
ROLE_MAP = {
    "py:mod": "module",
    "py:module": "module",
    "py:func": "function",
    "py:meth": "method",
    "py:class": "class",
    "py:exc": "exception",
    "py:attr": "attribute",
    "py:data": "data",
    "py:type": "type",
    "py:obj": "*",  # wildcard, match any role
}


# Heuristic role preferences by token shape
def preferred_roles_for(token: str) -> List[str]:
    # Dunder methods ⟹ method
    if token.startswith("__") and token.endswith("__"):
        return ["method", "function", "attribute", "class"]
    # CamelCase ⟹ class/exception by default
    if token[:1].isupper() and "_" not in token:
        return ["class", "exception", "type", "attribute", "module", "function", "method"]
    # snake_case ⟹ function/method
    return ["function", "method", "attribute", "module", "class", "type", "exception"]


NON_CODE_PATTERN = re.compile(r"[^\w\.]")  # things like http://, ?, *, %, spaces, etc.


class InventoryIndex:
    def __init__(self):
        # (domain, role, fullname) -> (uri, dispname)
        self.by_full: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
        # tail -> list of (domain, role, fullname, uri)
        self.by_tail: Dict[str, List[Tuple[str, str, str, str]]] = {}

    def add(self, inv: soi.Inventory):
        for obj in inv.objects:
            dom = obj.domain  # e.g. 'py'
            role = obj.role  # e.g. 'class','function'
            name = obj.name  # fully qualified
            uri = obj.uri
            self.by_full[(dom, role, name)] = (uri, obj.dispname)
            tail = name.split(".")[-1]
            self.by_tail.setdefault(tail, []).append((dom, role, name, uri))

    def find_exact(self, role: Optional[str], target: str) -> Optional[Tuple[str, str, str, str]]:
        """Return a single match for exact fullname (prefer py domain)."""
        candidates = []
        for (dom, r, name), (uri, _disp) in self.by_full.items():
            if name == target and (role in (None, "*", r)):
                candidates.append((dom, r, name, uri))
        # prefer python domain
        candidates.sort(key=lambda x: (0 if x[0] == "py" else 1, x[1], x[2]))
        return candidates[0] if candidates else None

    def find_by_suffix(
        self, role_pref: Iterable[str], target: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """Return a single unique candidate whose fullname endswith the token (or last two parts)."""
        parts = target.split(".")
        tails = [".".join(parts[-2:]), parts[-1]] if len(parts) > 1 else [parts[-1]]
        pool: List[Tuple[str, str, str, str]] = []
        for tail in tails:
            if tail in self.by_tail:
                pool.extend(self.by_tail[tail])

        if not pool:
            return None

        # Filter by preferred role order
        for role in role_pref:
            subset = [c for c in pool if c[1] == role]
            if len(subset) == 1:
                return subset[0]
            if len(subset) > 1:
                # choose the shortest fully-qualified name as tie-breaker
                subset.sort(key=lambda c: (c[2].count("."), len(c[2])))
                return subset[0]

        # Fallback: any unique by full name tail
        pool.sort(key=lambda c: (c[0] != "py", c[2].count("."), len(c[2])))
        return pool[0]


def load_inventory(path_or_url: str) -> soi.Inventory:
    # Accept either .../objects.inv, or a docs root URL (we'll append objects.inv)
    if path_or_url.startswith("http"):
        url = (
            path_or_url
            if path_or_url.endswith("objects.inv")
            else path_or_url.rstrip("/") + "/objects.inv"
        )
        return soi.Inventory(url=url)
    return soi.Inventory(path_or_url)


def parse_warnings(log_path: Path) -> List[dict]:
    items = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = WARNING_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        items.append(
            {
                "file": d["file"],
                "line": int(d["line"]) if d["line"] else None,
                "declared_role": d["role"],  # e.g., 'py:class'
                "target": d["target"].strip(),
            }
        )
    return items


def correct_role(declared_role: str, token: str, inv_roles_available: List[str]) -> str:
    # If declared role maps to inventory role and appears in candidates, keep it.
    mapped = ROLE_MAP.get(declared_role, None)
    if mapped and (mapped == "*" or mapped in inv_roles_available):
        return declared_role

    # Otherwise, choose first preferred role that exists
    for r in preferred_roles_for(token):
        if r in inv_roles_available:
            # map inventory role back to py:* role
            for k, v in ROLE_MAP.items():
                if v == r and k.startswith("py:"):
                    return k
            return declared_role
    return declared_role


def substitute_in_text(
    text: str, old_role: str, old_target: str, new_role: str, new_fullname: str
) -> Tuple[str, int]:
    """
    Replace a single occurrence on a line/file of:
      :old_role:`old_target`
      :old_role:`label <old_target>`
    and MyST forms:
      {old_role}`old_target` or {old_role}`label <old_target>`
    with the corrected role + fully-qualified name, using '~' to shorten display.
    """
    count = 0
    # Escaped target for regex
    t = re.escape(old_target)

    # RST patterns
    patterns = [
        (rf":{re.escape(old_role)}:`\s*{t}\s*`", rf":{new_role}:`~{new_fullname}`"),
        (rf":{re.escape(old_role)}:`[^`<]+?\s*<\s*{t}\s*>`", rf":{new_role}:`~{new_fullname}`"),
    ]
    # MyST patterns
    patterns += [
        (rf"\{{{re.escape(old_role)}\}}`\s*{t}\s*`", rf"{{{new_role}}}`~{new_fullname}`"),
        (
            rf"\{{{re.escape(old_role)}\}}`[^`<]+?\s*<\s*{t}\s*>`",
            rf"{{{new_role}}}`~{new_fullname}`",
        ),
    ]

    out = text
    for pat, repl in patterns:
        out, n = re.subn(pat, repl, out, count=1)
        count += n
        if n:
            break  # one replacement per call is enough
    return out, count


def literalize_non_code(text: str, role: str, target: str) -> Tuple[str, int]:
    """
    For obviously not-a-Python-object tokens, replace :role:`target` with literal ``target``.
    This cleans cases like URLs, query strings, %2A, etc.
    """
    t = re.escape(target)
    if not NON_CODE_PATTERN.search(target):
        return text, 0

    # RST + MyST
    patterns = [
        (rf":{re.escape(role)}:`\s*{t}\s*`", rf"``{target}``"),
        (rf"\{{{re.escape(role)}\}}`\s*{t}\s*`", rf"``{target}``"),
    ]
    out = text
    total = 0
    for pat, repl in patterns:
        out, n = re.subn(pat, repl, out)
        total += n
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/source", help="Docs source root containing .rst/.md")
    ap.add_argument(
        "--build", default="docs/_build/dirhtml", help="HTML build dir containing objects.inv"
    )
    ap.add_argument("--log", required=True, help="Path to sphinx build log with warnings")
    ap.add_argument(
        "--intersphinx", nargs="*", default=[], help="Base doc URLs or direct objects.inv URLs"
    )
    ap.add_argument("--write", action="store_true", help="Actually write fixes to files")
    ap.add_argument("--dry-run", action="store_true", help="Print proposed edits without writing")
    args = ap.parse_args()

    root = Path(args.root)
    build = Path(args.build)
    log_path = Path(args.log)

    # 1) Parse warnings
    issues = parse_warnings(log_path)
    if not issues:
        print("No unresolved xref warnings found.")
        return

    # 2) Build inventory index (local + intersphinx)
    idx = InventoryIndex()
    local_inv = build / "objects.inv"
    if not local_inv.exists():
        raise SystemExit(f"objects.inv not found at {local_inv}. Build HTML first.")
    idx.add(load_inventory(str(local_inv)))

    for url in args.intersphinx:
        try:
            inv = load_inventory(url)
            idx.add(inv)
        except Exception as e:
            print(f"[warn] failed to load intersphinx inventory from {url}: {e}")

    # 3) Process each file once (batch replacements)
    grouped: Dict[Path, List[dict]] = {}
    for it in issues:
        grouped.setdefault(Path(it["file"]), []).append(it)

    total_files = 0
    total_changes = 0

    for file_path, items in grouped.items():
        # Only operate inside source tree
        fpath = file_path if file_path.is_absolute() else root / file_path
        if not fpath.exists():
            # try relative to root directly (when log has full path already)
            fpath = file_path
            if not fpath.exists():
                print(f"[skip] missing file: {file_path}")
                continue

        text = fpath.read_text(encoding="utf-8", errors="ignore")
        orig_text = text
        file_changes = 0

        for it in items:
            decl_role = it["declared_role"]  # e.g. 'py:class'
            target = it["target"]

            # First, literalize obvious non-code refs (URLs/params/etc.)
            if NON_CODE_PATTERN.search(target):
                text, n = literalize_non_code(text, decl_role, target)
                file_changes += n
                if n:
                    continue  # fixed

            # Try to find candidates in the inventory
            inv_role = ROLE_MAP.get(decl_role, "*")
            roles_pref = preferred_roles_for(target) if inv_role in ("*", None) else [inv_role]

            cand = idx.find_exact(inv_role if inv_role != "*" else None, target)
            if not cand:
                cand = idx.find_by_suffix(roles_pref, target)

            if not cand:
                # nothing found: last resort literalize (avoid future warnings)
                text, n = literalize_non_code(text, decl_role, target)
                file_changes += n
                continue

            dom, found_role, fullname, _uri = cand

            # Pick corrected role (map inventory role -> py:* role)
            inv_roles_available = [found_role]
            new_role = correct_role(decl_role, target, inv_roles_available)

            # Substitute on the line/file
            text, n = substitute_in_text(text, decl_role, target, new_role, fullname)
            if n == 0:
                # maybe the markup omitted the domain? try generic roles
                # e.g., :class:`Name` (default domain py)
                text, n = substitute_in_text(
                    text, decl_role.split(":")[-1], target, new_role, fullname
                )
            file_changes += n

        if file_changes and args.write:
            fpath.write_text(text, encoding="utf-8")
            total_files += 1
            total_changes += file_changes
            print(f"[write] {fpath}  (+{file_changes} edits)")
        elif file_changes and args.dry_run:
            print(f"[dry] {fpath}  (+{file_changes} edits)")
        else:
            # no changes
            pass

    print(f"Done. Files changed: {total_files}, edits applied: {total_changes}")


if __name__ == "__main__":
    main()
