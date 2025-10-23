# DocsToKG Sphinx Documentation

This folder contains the Sphinx configuration used to build agent-friendly documentation.
Outputs include:

- **dirhtml** for human-friendly permalinks (`docs/build/dirhtml`)
- **json** and **text** exports for downstream automation (`docs/build/json`, `docs/build/text`)
- `sitemap.xml`, `objects.inv`, and deep links back to GitHub

## Local build

```bash
uv pip install -U sphinx myst-parser pydata-sphinx-theme \
    sphinx-sitemap sphinx-copybutton sphinx-design sphinx-autoapi
make -C docs all

# Optional: serve the HTML locally
python -m http.server -d docs/build/dirhtml 8000
```

## Structure

- `source/conf.py` – project configuration, theme, extensions, AutoAPI setup
- `source/index.md` – site entry point
- `source/00-map/` – mirrors `src/DocsToKG/*` packages for quick navigation
- AutoAPI generates API reference pages under `04-api/` during the build (not checked in)

## CI / GitHub Pages

The workflow in `.github/workflows/docs.yml` builds and publishes documentation to GitHub Pages on each push to `main`.
The default published site lives at: `https://paul-heyse.github.io/DocsToKG/`

