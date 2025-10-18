# 1. Tool Configuration

DocsToKG bundles helper scripts and third-party tooling to keep documentation, code quality, and automation workflows aligned with the current repository structure.

## 2. Documentation Automation

| Script | Location | Purpose | Example |
|--------|----------|---------|---------|
| `generate_all_docs.py` | `docs/scripts/` | Runs API extraction, Sphinx builds, link checks, and validation in a single pass. | `direnv exec . python docs/scripts/generate_all_docs.py` |
| `generate_api_docs.py` | `docs/scripts/` | Pulls docstrings from `src/` into `docs/04-api/`. | `direnv exec . python docs/scripts/generate_api_docs.py` |
| `validate_docs.py` | `docs/scripts/` | Ensures structure, headings, and inline references comply with the style guide. | `direnv exec . python docs/scripts/validate_docs.py` |
| `validate_code_annotations.py` | `docs/scripts/` | Confirms NAVMAP blocks and docstrings follow `CODE_ANNOTATION_STANDARDS.md`. | `direnv exec . python docs/scripts/validate_code_annotations.py` |
| `check_links.py` | `docs/scripts/` | Concurrent HTTP link checker with retry/back-off controls. | `direnv exec . python docs/scripts/check_links.py --timeout 10` |
| `build_docs.py` | `docs/scripts/` | Wrapper around Sphinx builders (`html`, `linkcheck`, `doctest`). | `direnv exec . python docs/scripts/build_docs.py --format html` |

Install documentation dependencies with:

```bash
direnv exec . pip install -r docs/build/sphinx/requirements.txt
```

## 3. Code Quality Stack

- `ruff` – linting, import sorting, and select auto-fixes (`pyproject.toml` enables the `I` import rules and `E`, `F`, `UP`, `RUF` families).
- `black` – formatting with a 100-character line length (referenced in CI configuration).
- `mypy --strict` – type checking aligned with the settings in `pyproject.toml`.
- `pytest` markers:
  - `-m smoke` for deployment spot-checks.
  - `-m hybrid_search` / `-m ontology` for subsystem suites.

Install and wire pre-commit hooks:

```bash
direnv exec . pip install pre-commit
direnv exec . pre-commit install
```

Repository defaults (hook versions, excludes) live in `docs/05-development/index.md`.

## 4. Developer Utilities

| Utility | Location | Description |
|---------|----------|-------------|
| `scripts/bootstrap_env.sh` | `scripts/` | Creates/refreshes `.venv`, installs bundled wheels (`torch`, `faiss`, `cupy`, `vllm`), and installs DocsToKG in editable mode. |
| `scripts/dev.sh` | `scripts/` | Convenience wrapper for running commands inside the project environment when `direnv` is unavailable (`./scripts/dev.sh exec pytest -q`). |
| `openspec` CLI | `openspec/` | Specification management (`openspec validate <change-id> --strict`) used for proposal review and implementation tracking. |

## 5. Vale (Optional)

Use [Vale](https://vale.sh/) for prose linting:

```bash
brew install vale  # or download from Vale releases
direnv exec . vale docs/
```

Rules live under `docs/.vale/` and align with the guidance in `docs/05-development/index.md`.

## 6. Environment Variables

| Variable | Purpose | Typical value |
|----------|---------|---------------|
| `DOCSTOKG_DATA_ROOT` | Root directory for downloads, manifests, and chunk outputs. | `/srv/docstokg` or project-local `./Data` |
| `ONTOLOGY_FETCHER_CONFIG` | Path to ontology `sources.yaml`. | `/etc/docstokg/sources.yaml` |
| `HYBRID_SEARCH_CONFIG` | Hybrid search runtime configuration consumed by `HybridSearchConfigManager`. | `config/hybrid_config.json` |
| `PA_ALEX_KEY` | OpenAlex API key used by content download pipelines. | Retrieved from secrets manager |

Store these variables in an `.env` file or platform-specific secret store, then let `.envrc` surface them during deployment (`direnv exec . …`).
