# DocsToKG Sphinx Documentation

This folder contains the Sphinx configuration used to build agent-friendly documentation.
Outputs include:

- **dirhtml** for human-friendly permalinks (`docs/build/dirhtml`)
- **json** and **text** exports for downstream automation (`docs/build/json`, `docs/build/text`)
- `sitemap.xml`, `objects.inv`, and deep links back to GitHub

## Local build

```bash
uv pip install -U sphinx myst-parser pydata-sphinx-theme \
    sphinx-sitemap sphinx-copybutton sphinx-design sphinx-autoapi \
    sphinxext-opengraph[social-cards] sphinx-notfound-page sphinxext-rediraffe \
    sphinx-codeautolink autodoc-pydantic myst-nb sphinxcontrib-mermaid \
    sphinx-issues sphinxcontrib-spelling pyenchant
make -C docs all

# Optional: serve the HTML locally
python -m http.server -d docs/build/dirhtml 8000
```

## Structure

- `source/conf.py` – project configuration, theme, extensions, AutoAPI setup
- `source/index.md` – site entry point
- `source/00-map/` – mirrors `src/DocsToKG/*` packages for quick navigation
- AutoAPI generates API reference pages under `04-api/` during the build (not checked in)
- `source/spelling-wordlist.txt` – customizable spelling exceptions for the spelling builder
- `source/_ext/xref_rescue.py` – missing-reference hook that rewrites unresolved xrefs using intersphinx inventories at build time

## CI / GitHub Pages

The workflow in `.github/workflows/docs.yml` builds and publishes documentation to GitHub Pages on each push to `main`.
The default published site lives at: `https://paul-heyse.github.io/DocsToKG/`

## Extensions

Enabled Sphinx extensions are tracked in `[tool.docs.extensions]` within `pyproject.toml`. Highlights include:

- Open Graph metadata & social cards (`sphinxext-opengraph`)
- Friendly 404s and redirect management (`sphinx-notfound-page`, `sphinxext-rediraffe`)
- Automatic code reference linking and rich Pydantic model rendering (`sphinx-codeautolink`, `autodoc-pydantic`)
- Mermaid diagrams, external ToC, GitHub issue linking, and optional spelling checks.
- Notebook support (`myst-nb`) is installed but disabled by default due to compatibility with the current MyST parser; flip it on when notebooks are ready.

## Cross-reference cleanup helper

After a strict build (`sphinx-build -n -b dirhtml docs/source docs/_build/dirhtml`), run:

```bash
python docs/sphinx_implementation/fix_xrefs.py \
  --root docs/source \
  --build docs/_build/dirhtml \
  --log build.log \
  --intersphinx https://docs.python.org/3 https://tenacity.readthedocs.io/en/stable/objects.inv \
  --write
```

The script parses unresolved-reference warnings, looks up the true targets in local + intersphinx inventories, and rewrites source roles/targets so future builds stay clean.

## Vendored HTTPX inventory

HTTPX does not publish an intersphinx inventory upstream. Run the helper script to refresh the local copy before rebuilding docs:

```bash
UV_NO_SYNC=1 uv run python docs/scripts/fetch_httpx_inventory.py
```

This writes `objects.inv` to `docs/_static/inventories/httpx/`, which is referenced by `conf.py` for intersphinx resolution.
