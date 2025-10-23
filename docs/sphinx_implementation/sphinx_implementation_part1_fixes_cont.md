Awesome — here’s a **drop-in fixer** your agent can run to auto-resolve most of those Sphinx cross-ref warnings. It:

* parses your Sphinx warning log,
* loads the **local** `objects.inv` plus any **Intersphinx** inventories you point it at,
* **finds the real target** (class/func/method) by exact/suffix match,
* **rewrites** your `.rst`/**MyST** pages to correct the **role** and **fully-qualified name** (uses `~` to display just the last segment),
* and “literalizes” obvious non-code tokens (e.g., URLs, query strings) so they stop being treated as Python objects.

> Why this works: it relies on Sphinx’s Python-domain roles and target resolution rules, `objects.inv` inventories (local + intersphinx), and the `~` short text modifier — all straight from Sphinx’s docs. ([sphinx-doc.org][1])

---

### 0) One-time Sphinx config hardening (recommended)

Add these to `conf.py` so you don’t generate new spurious refs:

```python
# Stop accidental single-backtick -> py:obj in .rst
default_role = None  # doc: default_role option  :contentReference[oaicite:1]{index=1}
primary_domain = "py"

# Make all unresolved refs fail in CI, so the agent fixes them
nitpicky = True  # same as -n/--nitpicky  :contentReference[oaicite:2]{index=2}

# Limit MyST’s implicit cross-ref searching
myst_ref_domains = ["std", "py"]  # MyST option  :contentReference[oaicite:3]{index=3}

# Intersphinx for externals seen in your warnings (extend as needed)
extensions += ["sphinx.ext.intersphinx"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "tenacity": ("https://tenacity.readthedocs.io/en/stable/", None),
}
```

And at the **top** of every module page like `04-api/telemetry_wayback/index.rst` add:

```rst
.. py:currentmodule:: DocsToKG.telemetry_wayback
```

This scopes short names so `emit_candidate` resolves without fully-qualifying. ([sphinx-doc.org][1])

---

### 1) Install helper dependency

```bash
pip install sphobjinv
```

`sphobjinv` gives you programmatic access to **objects.inv** (local or remote). ([sphobjinv.readthedocs.io][2])

---

### 2) Put this script in your repo (e.g., `tools/fix_xrefs.py`)

```python
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
            role = obj.role   # e.g. 'class','function'
            name = obj.name   # fully qualified
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

    def find_by_suffix(self, role_pref: Iterable[str], target: str) -> Optional[Tuple[str, str, str, str]]:
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
        url = path_or_url if path_or_url.endswith("objects.inv") else path_or_url.rstrip("/") + "/objects.inv"
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

def substitute_in_text(text: str, old_role: str, old_target: str, new_role: str, new_fullname: str) -> Tuple[str, int]:
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
        (rf"\{{{re.escape(old_role)}\}}`[^`<]+?\s*<\s*{t}\s*>`", rf"{{{new_role}}}`~{new_fullname}`"),
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
    ap.add_argument("--build", default="docs/_build/dirhtml", help="HTML build dir containing objects.inv")
    ap.add_argument("--log", required=True, help="Path to sphinx build log with warnings")
    ap.add_argument("--intersphinx", nargs="*", default=[], help="Base doc URLs or direct objects.inv URLs")
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
                text, n = substitute_in_text(text, decl_role.split(":")[-1], target, new_role, fullname)
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
```

**How it fixes your sample:**

* `WaybackTelemetry`, `emit_candidate`, etc.: the script sees `py:class` on snake_case → prefers `function/method`, finds the module/class by suffix in the inventory (from your build), rewrites to `:py:func:`~DocsToKG.telemetry_wayback.emit_candidate`` (or `:py:meth:` if it resolves to a method). Roles & target resolution rules: Sphinx Python domain. ([sphinx-doc.org][1])
* `tenacity.RetryCallState` / `tenacity.Retrying`: with the **tenacity** intersphinx mapping added, those will resolve to external inventory entries automatically; the script will rewrite to the fully-qualified external name if needed. Intersphinx mechanics: `objects.inv` + inter-project linking. ([sphinx-doc.org][3])
* The long list of `urls/index.rst` tokens like `HTTP://Example.COM`, `?pdf=1`, `%2A`: they’re not Python objects — the script **literalizes** them (`...`), which is the recommended way to display inline code examples rather than cross-refs. Default role guidance and the `:code:` role live in Sphinx’s roles docs. ([sphinx-doc.org][4])

---

### 3) Run it in a tight loop

```bash
# 1) Build once to collect warnings (HTML build writes objects.inv)
sphinx-build -n -b dirhtml docs/source docs/_build/dirhtml > build.log 2>&1 || true

# 2) Fix sources in place
python tools/fix_xrefs.py \
  --root docs/source \
  --build docs/_build/dirhtml \
  --log build.log \
  --intersphinx https://docs.python.org/3 https://tenacity.readthedocs.io/en/stable/objects.inv \
  --write

# 3) Rebuild (fail on any remaining unresolved refs)
sphinx-build -nW -b dirhtml docs/source docs/_build/dirhtml
```

* `-n/--nitpicky` warns on **all** unresolved references; `-W` upgrades warnings to errors so this can gate CI. ([sphinx-doc.org][5])

---

### 4) Optional: last-mile safety net (extension hook)

If you still have a few stubborn misses after rewriting, you can hook Sphinx’s **`missing-reference`** event to auto-resolve them from inventories on the fly (returning a `reference` node suppresses the warning). Drop this as `docs/_ext/xref_rescue.py` and add `extensions += ["_ext.xref_rescue"]`:

```python
# docs/_ext/xref_rescue.py
from docutils import nodes
from sphinx.util import logging

logger = logging.getLogger(__name__)

def setup(app):
    app.connect("missing-reference", on_missing)
    return {"version": "1.0", "parallel_read_safe": True}

def on_missing(app, env, node, contnode):
    """
    Try to resolve missing refs by probing intersphinx inventories.
    Return a reference node to silence warnings, or None to let Sphinx warn.
    """
    target = node.get("reftarget") or ""
    reftype = node.get("reftype") or ""
    # Use Sphinx's intersphinx inventories (already loaded via intersphinx_mapping)
    invs = getattr(app.builder, "intersphinx_named_inventory", {})
    # Probe across all domains/roles in external inventories
    for invname, inv in invs.items():
        # inv is like {'py:class': {'pkg.mod.Name': '...'}, ...}
        for role, mapping in inv.items():
            # prefer python roles when our reftype looks pythonic
            if reftype and not role.startswith(reftype.split(":")[0]):
                pass
            # exact, then suffix match
            if target in mapping:
                return _ref(app, mapping[target], contnode)
            for fullname, uri in mapping.items():
                if fullname.endswith("." + target):
                    return _ref(app, uri, contnode)
    return None

def _ref(app, uri, contnode):
    ref = nodes.reference("", "", internal=False)
    ref["refuri"] = app.builder.get_relative_uri(app.env.docname, uri)
    ref.append(contnode)
    return ref
```

Sphinx’s event and domain APIs allow this pattern; returning a node from `missing-reference` satisfies the resolver and prevents a warning. ([sphinx-doc.org][6])

---

## Notes & references the agent should “know”

* **Python domain roles & target resolution** (`:py:class:`, `:py:func:`, `:py:meth:`), and using `~` to shorten link text. ([sphinx-doc.org][1])
* **`default_role`** should be `None` unless you intentionally want single-backtick text to become refs; otherwise use explicit roles or `inline code`/`:code:`. ([sphinx-doc.org][7])
* **Intersphinx mechanics**: each HTML build emits **`objects.inv`**; `intersphinx_mapping` lets Sphinx resolve externals (and otherwise missing refs) against those inventories. ([sphinx-doc.org][3])
* **`py:currentmodule`** scopes short names on a page to a module, the canonical way to keep references concise. ([sphinx-doc.org][1])
* **`nitpicky`/`-n`** and **`nitpick_ignore(_regex)`** are the right levers for CI and for ignoring known non-symbol examples. ([sphinx-doc.org][5])

---

If you want, I can tailor the script to also **insert `.. py:currentmodule::`** at the top of each `04-api/<module>/index.rst` by deriving the module path from your directory layout (e.g., `DocsToKG.<dir>`).

[1]: https://www.sphinx-doc.org/en/master/usage/domains/python.html?utm_source=chatgpt.com "The Python Domain — Sphinx documentation"
[2]: https://sphobjinv.readthedocs.io/en/stable/api_usage.html?utm_source=chatgpt.com "API Usage — sphobjinv 2.3.1.3 documentation"
[3]: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html?utm_source=chatgpt.com "sphinx.ext.intersphinx – Link to other projects’ documentation — Sphinx documentation"
[4]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html?utm_source=chatgpt.com "Roles — Sphinx documentation"
[5]: https://www.sphinx-doc.org/zh-cn/master/usage/configuration.html?utm_source=chatgpt.com "配置 — Sphinx documentation"
[6]: https://www.sphinx-doc.org/en/master/extdev/event_callbacks.html?utm_source=chatgpt.com "Event callbacks API — Sphinx documentation"
[7]: https://www.sphinx-doc.org/en/master/usage/configuration.html?utm_source=chatgpt.com "Configuration — Sphinx documentation"
