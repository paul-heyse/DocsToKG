Absolutely—here’s a concrete, repo-specific plan that implements those **first three sections** (human+machine outputs, repo-mirrored nav, complete API). I’ve tailored it to your layout (`src/DocsToKG/...`, existing `docs/` content, and your docs automation scripts).

---

# 0) Prep: normalize the docs layout (once)

You already have generated HTML checked in under `docs/html/` and helper scripts under `docs/scripts/` (e.g., `generate_api_docs.py`, `validate_docs.py`, `validate_code_annotations.py`)  . Let’s standardize to the classic Sphinx split:

* **Source**: `docs/source/`
* **Builds**: `docs/build/` (`dirhtml`, `json`, `text`, `linkcheck` subdirs)

### Commands

```bash
# from repo root
git mv docs/html docs/build/dirhtml || true            # if html is present
mkdir -p docs/source docs/build/json docs/build/text docs/build/linkcheck

# keep builds out of git
printf "\n# Sphinx outputs\n/docs/build/\n" >> .gitignore
git rm -r --cached docs/build/dirhtml 2>/dev/null || true
```

> You’ve already established a “documentation-first” flow and script gates (e.g., `generate_api_docs.py`, `validate_docs.py`) — we’ll reuse and wire them into the new build targets. The commit diffs also reference enforcing “NAVMAP ordering” via `validate_code_annotations.py`; we’ll honor that too.  

---

# 1) Ship outputs for humans *and* machines

## 1.1 Dependencies (add if missing)

```bash
# using uv (preferred)
uv pip install -U myst-parser pydata-sphinx-theme sphinx-sitemap sphinx-copybutton sphinx-design
```

## 1.2 `docs/source/conf.py` (drop-in)

Create or update `docs/source/conf.py`:

```python
import os, sys, importlib
from datetime import date

# --- paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)  # so autodoc can import DocsToKG

# --- project ---
project = "DocsToKG"
author = "Paul Heyse"
release = os.getenv("DOCSTOKG_VERSION", "0.x")
html_title = f"{project} {release}"

# --- extensions ---
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode", "sphinx.ext.linkcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",   # .nojekyll for GH Pages
    "sphinx_sitemap",           # sitemap.xml for crawlers
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_typehints = "description"
autosectionlabel_prefix_document = True
nitpicky = True                    # warn on missing refs
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# --- theme ---
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_last_updated_fmt = "%Y-%m-%d"

# --- site base (required for sitemap) ---
# GitHub Pages canonical URL (adjust if you use a custom domain)
html_baseurl = "https://paul-heyse.github.io/DocsToKG/"

# --- intersphinx targets you reference in code/docs ---
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "pydantic": ("https://docs.pydantic.dev/latest/", {}),
    "httpx": ("https://www.python-httpx.org/", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
}

# --- heavy deps mocked so autodoc can import modules cleanly ---
autodoc_mock_imports = [
    "faiss", "faiss_gpu", "torch", "cupy", "vllm", "rmm", "raft", "cuvs", "pyarrow",
    # anything else you know is optional/heavy at import time
]

# --- link back to GitHub lines on each object ---
def linkcode_resolve(domain, info):
    if domain != "py":  # only for Python objects
        return None
    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname:
        return None
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return None
    # best-effort: find source file + lines
    import inspect, os
    try:
        obj = mod
        for part in fullname.split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        try:
            fn = inspect.getsourcefile(mod) or inspect.getfile(mod)
            lineno = 1
        except Exception:
            return None
    # make path relative to repo root
    fn_rel = os.path.relpath(fn, PROJECT_ROOT).replace("\\", "/")
    return f"https://github.com/paul-heyse/DocsToKG/blob/main/{fn_rel}#L{lineno}"
```

## 1.3 Builders & convenience commands

Add a simple **makefile** at `docs/Makefile`:

```make
SPHINXOPTS=-n -W --keep-going
SOURCEDIR=source
BUILDDIR=build

.PHONY: html dirhtml json text linkcheck all clean

html:
\t@sphinx-build $(SPHINXOPTS) -b html $(SOURCEDIR) $(BUILDDIR)/html

dirhtml:
\t@sphinx-build $(SPHINXOPTS) -b dirhtml $(SOURCEDIR) $(BUILDDIR)/dirhtml

json:
\t@sphinx-build $(SPHINXOPTS) -b json $(SOURCEDIR) $(BUILDDIR)/json

text:
\t@sphinx-build $(SPHINXOPTS) -b text $(SOURCEDIR) $(BUILDDIR)/text

linkcheck:
\t@sphinx-build -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck

all: dirhtml json text
clean:
\trm -rf $(BUILDDIR)
```

> `-n -W` forces missing refs to fail the build (great for catching link rot). You’re already running doc validation scripts in CI; keep those and add these build targets to the flow.

## 1.4 GitHub Pages workflow

Create `.github/workflows/docs.yml`:

```yaml
name: docs
on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - name: Install docs deps
        run: |
          uv pip install -U sphinx myst-parser pydata-sphinx-theme \
            sphinx-sitemap sphinx-copybutton sphinx-design
      - name: Build docs (dirhtml + json + text + linkcheck)
        run: |
          make -C docs dirhtml json text
          make -C docs linkcheck || true   # non-blocking linkcheck
      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/build/dirhtml
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

Now your published site (including `objects.inv` for cross-project symbol resolution) will live at:

```
https://paul-heyse.github.io/DocsToKG/
```

---

# 2) Mirror the repo structure (so agents can “walk the tree”)

Your code is organized into clear subpackages: `ContentDownload`, `DocParsing`, `HybridSearch`, `OntologyDownload` (commit docs point people to those exact paths) . We’ll create a **Map of the Project** that mirrors these folders and auto-expands as you add files.

## 2.1 Top-level map page

Create `docs/source/index.md`:

````md
# DocsToKG

```{toctree}
:maxdepth: 2
:caption: Map of the Project

00-map/index
04-api/index
````

```{toctree}
:maxdepth: 1
:caption: Guides & Architecture

../architecture/900-north-star-complete   # keep your long-form docs in place
```

````

Create `docs/source/00-map/index.md`:

```md
# Map of the Project

This section mirrors the repository so agents (and humans) can navigate by package.

```{toctree}
:maxdepth: 2
:glob:

ContentDownload/*
DocParsing/*
HybridSearch/*
OntologyDownload/*
````

````

## 2.2 Per-module landing pages (auto-generated or minimal hand-written)

Create minimal stubs that link to README and key entrypoints. Example: `docs/source/00-map/DocParsing/index.md`:

```md
# DocParsing

- Code root: `src/DocsToKG/DocParsing`
- Purpose: DocTags → chunks → embeddings pipeline
- CLI: `python -m DocsToKG.DocParsing.cli --help`

```{toctree}
:maxdepth: 2
:glob:

*
````

````

Repeat for `ContentDownload/`, `HybridSearch/`, `OntologyDownload/`.

### Optional: generate these from the repo

You already have helper scripts for docs automation; the commit history references a “NAVMAP” validation and multiple doc scripts in `docs/scripts/` :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}. If helpful, add `docs/scripts/generate_navmap.py` that walks `src/DocsToKG/*` and writes/updates those `00-map/*/` pages (keeping headings stable for persistent anchors). Wire it into your existing `generate_all_docs.py`.

---

# 3) Generate a *complete* API surface (no holes)

You already lean on docs automation scripts (the diffs show `generate_api_docs.py` and doc gates in CI) :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}. Let’s finalize the Sphinx side.

## 3.1 Autodoc/autosummary/napoleon

(Already enabled in `conf.py` above.)

Add a landing page `docs/source/04-api/index.md`:

```md
# API Reference

Auto-generated from docstrings. See per-module references below.

```{toctree}
:maxdepth: 2
:glob:

DocsToKG*
````

````

## 3.2 apidoc (stub generation)

Generate stubs (run whenever modules move):

```bash
# from repo root
uv run sphinx-apidoc -o docs/source/04-api src/DocsToKG --force --module-first
````

If imports are **heavy/fragile**, keep using your **generator script** (parsing the source w/o imports) or switch to **AutoAPI**/**autodoc2** later; for now, mock heavy deps via `autodoc_mock_imports` to keep build fast and deterministic.

## 3.3 Quality gates

* Build in nitpicky mode and fail on warnings (already set via `SPHINXOPTS=-n -W`).
* Keep your existing doc validation scripts in the PR checklist (the diffs show you run `generate_api_docs.py` and validation before merging).

---

# 4) One-touch local run

```bash
# Build everything locally
make -C docs all
# Serve dirhtml locally if you like:
python -m http.server -d docs/build/dirhtml 8000
```

---

## Notes specific to **DocsToKG**

* The repo highlights those four core packages as “study key modules” for newcomers; the **00-map/** structure above aligns exactly to that, giving agents a deterministic path hierarchy.
* Your doc automation already references **API generation** and **validation**; keep those scripts but change their output paths to `docs/source/04-api/` and call `make -C docs dirhtml json text` after generation so `objects.inv`, `sitemap.xml`, and machine-readable trees are always published.
* If you want CI to **gate on docs freshness**, add a job step that runs:

  ```bash
  direnv exec . python docs/scripts/generate_api_docs.py
  git diff --quiet -- docs/source/04-api || { echo "API docs out of date"; exit 1; }
  ```

  (Your diffs already show similar “documentation & annotation validation” guidance.)

---

If you want, I can turn this into a ready-to-merge PR (files + workflow + starter map pages) or wire your existing `docs/scripts/generate_all_docs.py` to run the `dirhtml/json/text/linkcheck` targets and push Pages.
