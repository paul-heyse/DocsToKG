# Tool Configuration

DocsToKG ships helper scripts and third-party tooling to maintain documentation quality and developer workflows. Configure them as follows.

## Documentation Automation

| Script | Location | Purpose | Usage |
|--------|----------|---------|-------|
| `generate_all_docs.py` | `docs/scripts/` | Orchestrates API generation, Sphinx build, validation, and link checks | `python docs/scripts/generate_all_docs.py` |
| `generate_api_docs.py` | `docs/scripts/` | Extracts docstrings from `src/` into `docs/04-api/` | `python docs/scripts/generate_api_docs.py` |
| `validate_docs.py` | `docs/scripts/` | Style, structure, and internal link validation | `python docs/scripts/validate_docs.py` |
| `check_links.py` | `docs/scripts/` | Asynchronous external link checker | `python docs/scripts/check_links.py --timeout 10` |
| `build_docs.py` | `docs/scripts/` | Wrapper for Sphinx builds and linkcheck targets | `python docs/scripts/build_docs.py --format html` |

Install Sphinx dependencies via:

```bash
pip install -r docs/build/sphinx/requirements.txt
```

## Code Quality

Recommended formatting and linting stack:

- `ruff` for linting and import ordering (`pyproject.toml` enables `extend-select = ["I"]`)
- `black` with line length 100
- `mypy --strict` aligned with project type hints

Configure pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Suggested `.pre-commit-config.yaml` snippets are maintained in the development guide.

## Vale (Optional)

Use [Vale](https://vale.sh/) for prose linting:

```bash
brew install vale  # or download a release binary
vale docs/
```

Rules can be extended by dropping configurations into `docs/.vale/` (see Style Guide for conventions).

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `DOCSTOKG_DATA_ROOT` | Base directory for downloaded artifacts | `~/.cache/docstokg` |
| `ONTOLOGY_FETCHER_CONFIG` | Override path to `sources.yaml` | `~/.data/ontology-fetcher/configs/sources.yaml` |
| `HYBRID_SEARCH_CONFIG` | Path to hybrid search runtime config | `config/hybrid_config.json` |

Keep these variables in `.env` files when scripting to ensure reproducible runs.
