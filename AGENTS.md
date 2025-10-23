# Repository Guidelines

## Agents

Please read AGENTS.md at the root directory

## Project Structure & Module Organization

- `src/DocsToKG/` hosts production code; extend existing domains (`ContentDownload`, `DocParsing`, `HybridSearch`, `OntologyDownload`) before creating new roots.
- `tests/` should mirror that layout; keep fixtures nearby and store bulky artifacts under `Data/` with README notes.
- `docs/`, `docs/scripts/`, `scripts/`, `ci/`, and `openspec/` contain references, automation, CI wiring, and RFCs—prefer updating them over ad-hoc scripts.

## Build, Test, and Development Commands

- Run `./scripts/bootstrap_env.sh` to create the venv, install the editable package, and verify the CUDA FAISS wheel; follow with `direnv allow`.
- `pip install -e .` refreshes dependencies after editing `pyproject.toml`; commit lockstep with the change it enables.
- `pytest -q` covers the default suite; append `-m real_vectors --real-vectors` or `-m scale_vectors` for GPU smoke tests.
- Use `./scripts/run_precommit.sh` for Black, Isort, Ruff, and static checks; refresh docs via `python docs/scripts/generate_api_docs.py` and `python docs/scripts/check_links.py --timeout 10`.

## Coding Style & Naming Conventions

- Target Python 3.13, four-space indentation, and explicit type hints for exported APIs.
- Black and Ruff enforce a 100-character limit; fix lint locally (`ruff check src/ tests/`) before review.
- Follow `snake_case` for modules, functions, and Typer commands, `PascalCase` for classes, and reserve `ALL_CAPS` for constants; prefer relative imports within a domain.

## Testing Guidelines

- Mirror package layout under `tests/DocsToKG/...` and name tests `test_<module>_<behaviour>`.
- Tag GPU or slow paths with `@pytest.mark.real_vectors` or `@pytest.mark.scale_vectors`; keep the default run fast.
- Run `pytest --cov=DocsToKG --cov-report=term-missing` before shared refactors and rely on deterministic fixtures instead of live services; clean temporary artifacts under `tmp/`.

## Commit & Pull Request Guidelines

- Write commit subjects in imperative mood with an optional scope prefix (`HybridSearch: harden FAISS loader retry`).
- Keep commits focused, avoid mixing formatting-only changes, and include docs/config updates with behaviour changes.
- PRs need a concise summary, linked issue when available, and proof of passing tests or manual checks; flag follow-up tasks in the description.
