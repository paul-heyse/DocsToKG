# DocsToKG Code Style & Documentation Guide
**Version:** 2025-10-23 · **Applies to:** ContentDownload, DocParsing, HybridSearch, OntologyDownload

> This is the project‑wide, enforceable coding standard for Python code, docstrings, module navmaps, and the “README‑overview” pattern that appears near the top of important modules. It harmonizes PEPs, Google‑style docstrings (via Sphinx Napoleon), and our own NAVMAP v1 convention.

---

## 0) Quick defaults (tl;dr)
- **Formatting:** Black (line length 100), `isort` with `profile=black`. **No bikeshedding.** citeturn1search1 citeturn1search2
- **Linting:** Ruff as the single linter; enable E/F/I/B/UP/SIM sets; `pydocstyle` rules with `convention=google`. citeturn1search4
- **Types:** Comprehensive type hints (PEP 484/526). Fail CI on new `Any` in changed code. citeturn5search1 citeturn5search0
- **Docstrings:** **Google style** for all public APIs; parsed by **Sphinx Napoleon**. citeturn0search5 citeturn0search1
- **Docs build:** `sphinx-build -b html "docs/sphinx documentation" "docs/sphinx documentation/_build/html"` (use `sphinx-autobuild` for live refresh).
- **GPU installs:** `uv pip install --find-links .wheelhouse -e .[gpu]` (wheelhouse holds prebuilt CUDA wheels for Python 3.13).
- **Navmap:** JSON **NAVMAP v1** comment block at top of key modules; see schema below. (Example in our pipeline module.) fileciteturn9file0
- **Module README‑overview:** Immediately below NAVMAP: module‑level docstring with Summary → Responsibilities → Public API → Examples → Performance → Observability → Gotchas → References.
- **Commits & versioning:** **Conventional Commits** + **SemVer 2.0.0**. citeturn6search0 citeturn6search3
- **Pre‑commit:** mandatory hooks: ruff, black, isort, pyupgrade, codespell, trailing‑ws, end‑of‑file. citeturn1search6

---

## 1) Language level & naming
- Target Python **3.12+** (CPython 3.12 is the baseline; 3.13 is supported for GPU wheels). Use modern syntax (pattern matching, `typing.Self`, `|` unions).
- **Naming:** follow PEP 8 (modules: `snake_case.py`; classes: `CapWords`; functions/vars: `snake_case`; constants: `UPPER_SNAKE`; non‑public: `_underscore`). citeturn0search2
- Prefer explicit `__all__` in public modules to make exported surface clear. (PEP 257 also recommends listing exports in module docs.) citeturn0search0

---

## 2) Formatting, imports, and linting
### Black
- Enforce with Black; set line length to **100** for better docstring readability; avoid project‑specific exceptions. citeturn1search1

### isort
- Sort imports with `profile="black"`; this avoids conflicts between isort and Black. citeturn1search0

### Ruff
- Single linter for everything: enable rule families: `E,F` (pycodestyle/pyflakes), `I` (isort), `B` (bugbear), `UP` (pyupgrade), `SIM` (simplify). Add `pydocstyle` with `convention="google"`. citeturn1search4

---

## 3) Types
- **Function + method signatures must be fully typed** (parameters and return). Prefer `typing` generics and `TypedDict` where appropriate for structured dicts. citeturn5search1 citeturn5search3
- Use **variable annotations** (PEP 526) for module/class attributes; avoid type comments. citeturn5search0
- Prefer forward‑ref friendly annotations; when building with Sphinx, `from __future__ import annotations` or postponed evaluation (per PEP 563 context) can reduce import cycles. citeturn5search2
- **Mypy** (or equivalent) is required in CI. Minimal baseline:
  - `python_version = "3.12"`, `disallow_untyped_defs = true`, `warn_unused_ignores = true`, `warn_redundant_casts = true`, `no_implicit_optional = true`, `strict_equality = true`, `show_error_codes = true`.
  - Configuration lives in `pyproject.toml` under `[tool.mypy]`. citeturn2search4

---

## 4) Docstrings (Google style; PEP 257 semantics)
All public modules, classes, functions, and methods **must** have docstrings. Follow PEP 257 for structure (one‑line summary, blank line, details). Use **Google style sections** (`Args`, `Returns`, `Raises`, `Examples`, `Notes`)—automatically parsed by **Sphinx Napoleon**. citeturn0search0 citeturn0search1 citeturn0search5

Google-style docstrings feed Sphinx builds located under ``docs/sphinx documentation``. Generate HTML locally with ``sphinx-build -b html "docs/sphinx documentation" "docs/sphinx documentation/_build/html"``; the output is ignored by Git.

**Template (function):**
```python
def score_chunks(chunks: Sequence[Chunk], *, k: int = 10) -> list[float]:
    """Compute BM25 scores for `chunks`.

    Args:
        chunks: Pre-tokenized chunks to score.
        k: Number of top features to consider for BM25 tuning.

    Returns:
        A list of scores aligned to `chunks`.

    Raises:
        ValueError: If `k <= 0` or tokenization is inconsistent.

    Examples:
        >>> score_chunks([Chunk("hi")], k=1)
        [0.0]
    """
    ...
```

**Class docstring checklist:**
- One‑line summary of responsibility.
- Public attributes under **Attributes**.
- If subclass‑extension protocol exists, add **See Also** and **Notes**.
- Document constructor params on `__init__` as needed (Google style). citeturn0search5

---

## 5) NAVMAP v1 (module navigation map)
A machine‑readable JSON block at the very top of key modules describing the public surface. We already do this in the HybridSearch pipeline: use it as the canonical example. fileciteturn9file0

**Header comment form (exact):**
```python
# === NAVMAP v1 ===
# {{
#   "module": "package.subpackage.module",
#   "purpose": "Short purpose sentence",
#   "sections": [
#     {{"id": "chunkingestionpipeline", "name": "ChunkIngestionPipeline", "anchor": "class-chunkingestionpipeline", "kind": "class"}},
#     {{"id": "observability", "name": "Observability", "anchor": "class-observability", "kind": "class"}}
#   ]
# }}
# === /NAVMAP ===
```

**Schema (required keys):**
- `module: str`, `purpose: str`, `sections: list[Section]`
- `Section`: `id: str` (kebab), `name: str` (human), `anchor: str` (docs anchor), `kind: "class" | "func" | "type" | "const" | "module"`
- Optional: `private: bool`, `deprecated: bool`, `see: list[str]`

**Conventions:**
- `id` is stable and kebab‑case; `anchor` matches Sphinx/README anchors.
- Keep **only public** items in NAVMAP (private names start with `_`).

---

## 6) Module “README‑overview” (beneath NAVMAP)
Immediately after NAVMAP, include a **module‑level docstring** with the following headings:

1. **Summary** (1–3 lines)  
2. **Responsibilities** (bulleted)  
3. **Public API** (bullet list tied to NAVMAP anchors)  
4. **Examples** (doctest‑style)  
5. **Performance & Limits** (big‑O, memory notes, HW assumptions)  
6. **Observability** (metrics names, span names, log keys)  
7. **Gotchas** (edge cases, invariants)  
8. **References** (internal docs/specs; external standards)

Sphinx **Napoleon** will render sections from Google style cleanly. citeturn0search1

---

## 7) Testing style
- Use **pytest**; fixtures over xUnit, parametrize heavily; keep unit tests fast and deterministic. citeturn3search0 citeturn3search3
- Prefer **doctest‑like** examples in docstrings for API entry points when short.
- Property‑style tests (e.g., vector bijection across ingest/restore) are encouraged.

---

## 8) Repository hygiene
- **Pre‑commit** is mandatory; contributors must run `pre-commit install`. Include hooks for Ruff, Black, isort, pyupgrade, codespell, whitespace/EOF fixes. citeturn1search6
- **EditorConfig** at repo root to normalize indentation, EOF newline, line endings. citeturn4search5
- **Conventional Commits** and **SemVer** govern CHANGELOG and releases. citeturn6search0 citeturn6search3
- Install local tooling with `pip install -e .[dev,docs]` (or via `uv pip install --python .venv -e .[dev,docs]`) before running lint/type/doc jobs.

---

## 9) Reference configurations

### `pyproject.toml` (baseline)
```toml
[tool.black]
line-length = 100
target-version = ["py312"]

[tool.isort]
profile = "black"
line_length = 100

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["E203"]  # handled by Black

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.mypy]
python_version = "3.12"
warn_unused_ignores = true
warn_redundant_casts = true
no_implicit_optional = true
disallow_untyped_defs = true
show_error_codes = true
strict_equality = true

[tool.pytest.ini_options]
addopts = "-q"
xfail_strict = true
```

### `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.5
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format

  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/asottile/pyupgrade
    rev: v3.17.0
    hooks:
      - id: pyupgrade
        args: ["--py311-plus"]

  - repo: https://github.com/codespell-project/codespell
    rev: v2.3.0
    hooks:
      - id: codespell

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
```

### `EDITORCONFIG` (root)
```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4

[*.md]
max_line_length = off
trim_trailing_whitespace = false
```

---

## 10) Examples

### A. NAVMAP header + README‑overview skeleton
```python
# === NAVMAP v1 ===
# {{
#   "module": "DocsToKG.Example.module",
#   "purpose": "Short, active-voice summary of module responsibility",
#   "sections": [
#     {{"id": "manager", "name": "Manager", "anchor": "class-manager", "kind": "class"}},
#     {{"id": "evaluate", "name": "evaluate", "anchor": "func-evaluate", "kind": "func"}}
#   ]
# }}
# === /NAVMAP ===

"""One‑line summary.

Responsibilities:
- …

Public API:
- :class:`Manager` — core orchestration
- :func:`evaluate` — …

Examples:
    >>> Manager().run()

Performance & Limits:
- Assumes …

Observability:
- Metrics: example_seconds, example_total
- Spans: example.step

Gotchas:
- …

References:
- PEP 8, PEP 257; Sphinx Napoleon.
"""
```

### B. Function docstring (Google style)
```python
def ingest(path: Path, *, batch: int = 1000) -> int:
    """Ingest vectors from ``path`` into the active namespace.

    Args:
        path: Parquet or JSONL vectors file.
        batch: Batch size for upserts.

    Returns:
        Number of vectors ingested.

    Raises:
        IngestError: If vector dimension or footer metadata mismatch.
    """
```

---

## Citations
- PEP 257 docstring conventions, module/class/function requirements. citeturn0search0
- Sphinx **Napoleon** parses Google/NumPy‑style docstrings. citeturn0search1
- PEP 8 naming & layout guidance. citeturn0search2
- Black formatter principles. citeturn1search1
- isort `profile="black"`. citeturn1search0
- Ruff linter capabilities. citeturn1search4
- Type hints (PEP 484) and variable annotations (PEP 526). citeturn5search1 citeturn5search0
- Pytest fixtures & parametrization. citeturn3search0 citeturn3search3
- Conventional Commits and SemVer 2.0.0. citeturn6search0 citeturn6search3
