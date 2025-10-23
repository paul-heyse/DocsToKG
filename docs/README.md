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
    sphinx-external-toc sphinx-issues sphinxcontrib-spelling pyenchant
make -C docs all

# Optional: serve the HTML locally
python -m http.server -d docs/build/dirhtml 8000
```

## Structure

- `source/conf.py` – project configuration, theme, extensions, AutoAPI setup
- `source/index.md` – site entry point
- `source/00-map/` – mirrors `src/DocsToKG/*` packages for quick navigation
- AutoAPI generates API reference pages under `04-api/` during the build (not checked in)
- `_toc.yml` – externalized navigation tree consumed by `sphinx-external-toc`
- `source/spelling-wordlist.txt` – customizable spelling exceptions for the spelling builder

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
