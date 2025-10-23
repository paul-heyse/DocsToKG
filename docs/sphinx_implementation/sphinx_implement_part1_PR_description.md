Perfect — I put together a **ready-to-apply PR patch** and a step-by-step plan an AI programming agent can follow to implement this *exactly*.

[Download the patch](sandbox:/mnt/data/0001-docs-sphinx-agent-nav.patch)

---

# What this PR adds

1. **Ship outputs for humans & machines**

* HTML with **pretty permalinks** via `dirhtml`
* **JSON** and **text** builders for machine ingestion
* **Sitemap** + canonical `html_baseurl`
* GitHub Pages **deploy workflow** (builds on each push to `main`)
* All wired with a simple `docs/Makefile`

2. **Mirror repo structure for crawlable nav**

* `docs/source/00-map/…` mirrors your `src/DocsToKG/*` packages
* Minimal landing pages for: `ContentDownload/`, `DocParsing/`, `HybridSearch/`, `OntologyDownload/`

3. **Complete API surface (no imports required)**

* Uses **sphinx-autoapi** to parse `src/DocsToKG` statically
* Adds stable pages under `04-api/`, linked from the home index
* Mocks heavy imports just in case (`autodoc_mock_imports`)

---

# Files created by the patch

* `.github/workflows/docs.yml` – build & publish docs to GitHub Pages
* `docs/Makefile` – `dirhtml`, `json`, `text`, `linkcheck` targets (with `-n -W`)
* `docs/.gitignore` – ignores `/build/`
* `docs/README.md` – local build/use instructions
* `docs/source/conf.py` – theme, sitemap, AutoAPI, linkcode to GitHub, MyST, etc.
* `docs/source/index.md` – top-level toctrees (Map + API)
* `docs/source/00-map/…` – four per-module landing pages + map index
* `docs/source/_static/.gitkeep`, `docs/source/_templates/.gitkeep`

---

# One-shot instructions (AI agent friendly)

## A) Create the branch, apply the patch, and open the PR

```bash
# from repo root
git checkout -b docs/sphinx-agent-nav
git apply --index  /path/to/0001-docs-sphinx-agent-nav.patch
git commit -m "docs: agent-friendly Sphinx setup (dirhtml/json/text + sitemap + GH Pages + AutoAPI)"
git push -u origin docs/sphinx-agent-nav

# open PR (requires GitHub CLI)
gh pr create --fill --base main --head docs/sphinx-agent-nav
```

> If you prefer, the patch can be applied via GitHub’s web UI by uploading the diff into a new commit on a branch.

## B) Local build & quick smoke test

```bash
# install doc deps (uses uv, which you already use)
uv pip install -U sphinx myst-parser pydata-sphinx-theme sphinx-sitemap sphinx-copybutton sphinx-design sphinx-autoapi

# build all the outputs
make -C docs all

# optional: view HTML locally
python -m http.server -d docs/build/dirhtml 8000
```

**Verify outputs**

* `docs/build/dirhtml/index.html` renders with the **PyData** theme
* `docs/build/json/**` contains per-page JSON structures
* `docs/build/text/**` contains plain-text pages
* `docs/build/dirhtml/objects.inv` exists (symbol inventory)

## C) Merge & publish

* Merge the PR → the **docs** workflow builds and publishes to GitHub Pages.
* Published URL (default): `https://paul-heyse.github.io/DocsToKG/`

---

# Implementation details (for the agent)

* **Theme & UX**: `pydata_sphinx_theme` with right-side page TOC and dark/light toggle.
* **MyST Markdown** enabled (`myst_parser` + common extensions).
* **Sitemap**: `sphinx_sitemap` with `html_baseurl="https://paul-heyse.github.io/DocsToKG/"`.
* **AutoAPI**:

  * `autoapi_dirs = [src/DocsToKG]`
  * `autoapi_root = "04-api"` (stable, predictable URLs)
  * No imports needed; robust to heavy deps.
* **Source deep-links**: `sphinx.ext.viewcode` + `sphinx.ext.linkcode` with a resolver that links to `main` lines on GitHub.
* **Quality gates**: Makefile runs Sphinx with `-n -W --keep-going` to fail on missing refs while compiling as much as possible. CI runs `linkcheck` non-blocking.

---

# “Why these choices”

* **AutoAPI** ensures the API docs are generated even if `DocsToKG` has heavy CUDA or GPU deps that would make `autodoc` brittle.
* **dirhtml + sitemap** produce crawler-friendly, stable URLs for agents and external cross-references.
* **JSON + text** builders provide machine-readable structure and easy ingestion for downstream tooling.
* The **Map of the Project** mirrors `src/DocsToKG/*`, which matches the way you describe the system to agents (ContentDownload → DocParsing → HybridSearch → OntologyDownload).

---

# Follow-ups you might want (optional)

* Add `sphinx-autobuild` dev target:

  ```bash
  uv pip install sphinx-autobuild
  sphinx-autobuild docs/source docs/build/dirhtml --watch src
  ```

* If you later introduce versioned docs, enable the PyData **version switcher** in `conf.py` and add a small `_static/switcher.json`.
* If you want your long-form `docs/architecture/*.md` within the site, either:

  * move them under `docs/source/architecture/` (simplest), or
  * add a symlink `docs/source/architecture -> ../architecture` (Sphinx will follow it during CI builds).

---

If you’d like, I can also open the PR programmatically next time — the GitHub tool here didn’t return a branch SHA we can build on, so I shipped a universal patch instead.
