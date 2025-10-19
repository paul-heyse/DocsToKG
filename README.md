# 1. DocsToKG

DocsToKG turns raw documents into searchable knowledge artefacts by combining document acquisition, Docling-based parsing, ontology downloads, and a FAISS-backed hybrid search engine.

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 2. Overview

- **Content acquisition**: Fetch PDFs and metadata from external services such as Pyalex.
- **Doc parsing & chunking**: Convert documents into DocTags, chunked Markdown, and embeddings using Docling pipelines.
- **Hybrid search**: Fuse BM25, SPLADE, and FAISS dense retrieval with configurable ranking.
- **Ontology management**: Download and validate ontologies for consistent terminology and enrichment.
- **Documentation tooling**: Automated scripts generate API docs, run validations, and perform link checks.

See `docs/01-overview/` for a high-level introduction and `docs/03-architecture/` for subsystem details.

## 3. Quick Start

```bash
git clone https://github.com/paul-heyse/DocsToKG.git
cd DocsToKG

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
pip install -r requirements.in         # optional GPU / ML stack
pip install -r docs/build/sphinx/requirements.txt  # documentation tooling
```

Or let the bootstrap script handle those steps:

```bash
./scripts/bootstrap_env.sh
direnv allow  # re-load the environment so .envrc activates .venv automatically
```

> ℹ️  The repository ships with a preconfigured `.envrc` that adds `.venv/bin` to `PATH`,
> exports `VIRTUAL_ENV`, and appends `src/` to `PYTHONPATH`. Any shell (or AI agent) that
> runs commands via `direnv exec . …` automatically picks up the project virtual environment
> without additional configuration.

Create a `.env` file (or export variables manually):

```bash
cat > .env <<'EOF'
DOCSTOKG_DATA_ROOT=./Data
HYBRID_SEARCH_CONFIG=./config/hybrid_config.json
ONTOLOGY_FETCHER_CONFIG=./configs/sources.yaml
PA_ALEX_KEY=replace-with-api-key
EOF
```

Verify the environment:

```bash
pytest -q
python docs/scripts/generate_api_docs.py
python docs/scripts/validate_docs.py
```

## 4. Example Usage

### Hybrid Search Service

```python
from DocsToKG.HybridSearch import HybridSearchRequest
from my_project.hybrid import build_hybrid_service  # see docs/06-operations/index.md

request = HybridSearchRequest(query="ontology alignment best practices", page_size=3)
service = build_hybrid_service()
response = service.search(request)
for result in response.results:
    print(result.doc_id, round(result.score, 3), result.highlights)
```

#### CPU snapshot refresh throttling

Dense FAISS indexes now throttle CPU snapshot refreshes to reduce excessive
`serialize()` calls on busy ingest nodes. Tune the behaviour via
`DenseIndexConfig`:

```python
DenseIndexConfig(
    snapshot_refresh_interval_seconds=30.0,  # minimum time between refreshes
    snapshot_refresh_writes=5000,            # or minimum writes before refresh
)
```

Set either value to `0` to disable that threshold; when both are `0` the store
refreshes after every write (legacy behaviour). Operators can still force a
refresh—use `FaissVectorStore.flush_snapshot()` before shutdown or after
maintenance windows, though `HybridSearchService.close()` now flushes a final
snapshot automatically during graceful shutdown.

Observability now reports when snapshots occur: counters
`faiss_snapshot_refresh_total` and `faiss_snapshot_refresh_skipped` track actual
and throttled attempts (labelled by reason and whether the refresh was forced),
while the gauge `faiss_snapshot_age_seconds` surfaces the age of the cached CPU
replica for dashboards and alerts.

### Ontology Download CLI

```bash
python -m DocsToKG.OntologyDownload.cli pull --spec configs/sources.yaml --force --json
python -m DocsToKG.OntologyDownload.cli validate hp latest
```

Planner metadata probes now validate URLs before issuing any network calls and
reuse the polite networking stack (session pooling, headers, and rate limits)
used by the download pipeline. Operators can opt out of planner probes via
`defaults.planner.probing_enabled: false` or the CLI flag `--no-planner-probes`;
URL validation still runs and planner logs record structured events for
blocked, skipped, fallback, and retry scenarios.

## 5. Content Download Enhancements

### 5.1 Additional Open Access Resolvers

The default resolver registry now includes Zenodo and Figshare, expanding open
access coverage without additional configuration. Both providers honour
``ResolverConfig`` timeouts, polite headers, and conditional request metadata.

To opt out, disable them via ``resolver_toggles`` in your configuration file:

```yaml
resolver_toggles:
  zenodo: false
  figshare: false
```

### 5.2 Bounded Concurrency

Use the ``--workers`` flag to enable bounded parallelism when downloading
content via the OpenAlex pipeline. Each worker drives an isolated
``ResolverPipeline`` that honours per-resolver rate limits and maintains
independent HTTP sessions with retry support.

```bash
# Sequential (default, safest)
python -m DocsToKG.ContentDownload.download_pyalex_pdfs --workers 1 --topic "oncology" --year-start 2020 --year-end 2024

# Parallel (2-5x throughput)
python -m DocsToKG.ContentDownload.download_pyalex_pdfs --workers 3 --topic "oncology" --year-start 2020 --year-end 2024
```

Additional operational flags:

- ``--max-concurrent-per-host 2`` keeps simultaneous downloads per domain polite.
- ``--domain-bytes-budget example.com=500MB`` guards against single-host bandwidth drain.
- ``--log-rotate 250MB`` rotates JSONL attempt logs during long-running crawls.
- ``--domain-token-bucket example.org=0.5:capacity=2`` enforces host-specific request rates.

**Concurrency recommendations:**

- Start with ``--workers=3`` for production workloads.
- Monitor rate limit compliance with resolver APIs while scaling.
- Higher values (>5) may overwhelm downstream services despite per-resolver
  throttling.

### 5.3 HEAD Pre-check Filtering

HEAD preflight checks remove obvious HTML landing pages and zero-byte
responses before performing costly ``GET`` downloads. The feature is enabled by
default and can be tuned per resolver:

```yaml
enable_head_precheck: true
resolver_head_precheck:
  landing_page: false
  wayback: false  # opt-out for resolvers that reject HEAD
  zenodo: true    # explicitly keep HEAD preflight enabled
```

When a HEAD request fails (timeout or 5xx), the pipeline automatically falls
back to the original ``GET`` attempt to avoid false negatives.

### 5.4 Additional CLI Flags

- ``--dry-run``: compute resolver coverage without writing files.
- ``--resume-from <manifest.jsonl>``: skip works already recorded as successful. CSV
  attempts logs are also supported when a paired ``*.sqlite3``/``*.sqlite`` cache lives
  alongside the CSV—even outside the active manifest directory. If the
  manifest path is wrong or missing, the downloader now fails fast with a clear
  error so you can fix the flag before rerunning.
- ``--extract-text=html``: save plaintext alongside HTML fallbacks (requires ``trafilatura``).
- ``--enable-resolver openaire`` (and ``hal``/``osf``): opt into additional EU/preprint resolvers.
- ``--resolver-config config.yaml``: load advanced options such as
  ``max_concurrent_resolvers`` and ``resolver_head_precheck`` (see
  ``docs/resolver-configuration.md``).

### Resolver Enhancements

- **Zenodo and Figshare support** – the default resolver order now queries
  Zenodo and Figshare APIs, expanding coverage of institutional repositories.
- **Bounded intra-work concurrency** – configure
  ``max_concurrent_resolvers`` to execute independent resolvers in parallel
  while preserving per-resolver rate limits.
- **HEAD pre-check filtering** – enable ``enable_head_precheck`` (default) to
  perform lightweight HEAD requests that skip HTML and zero-byte responses
  before downloading.
- **Migration resources** – see ``docs/migration-modularize-resolvers.md`` for
  a full list of import path changes, configuration defaults, and testing tips.

### 5.5 Troubleshooting Content Downloads

- **Partial files remain (``*.part``)** – rerun with fewer workers or check network
  stability before retrying.
- **Resolver rate limit warnings** – lower ``--workers`` or increase per-resolver
  ``resolver_min_interval_s``.
- **High memory usage** – reduce ``--workers`` to limit in-flight downloads.

### 5.6 Logging and Exports

- Attempts log to JSONL by default. Convert to CSV with
  ``python scripts/export_attempts_csv.py attempts.jsonl attempts.csv``.
- Alternatively, use ``jq``:
  ``jq -r '[.timestamp,.work_id,.status,.url] | @csv' attempts.jsonl > attempts.csv``.

## 6. Development

### 6.1 Development Workflow

```bash
# Formatting and linting
./scripts/run_precommit.sh  # runs pre-commit hooks with --hook-stage manual
black src/ tests/
isort src/ tests/
ruff check src/ tests/
mypy src/ --strict

# Tests
pytest -q
pytest -m real_vectors --real-vectors  # optional vector-backed suite

# Documentation
python docs/scripts/generate_all_docs.py
python docs/scripts/check_links.py --timeout 10
```

Refer to `docs/05-development/index.md` for contribution guidelines and review checklists.

## 7. Project Structure

```
DocsToKG/
├── src/DocsToKG/                 # Core packages
│   ├── ContentDownload/          # Document acquisition utilities
│   ├── DocParsing/               # Docling pipelines and embedding scripts
│   ├── HybridSearch/             # Hybrid search configuration, storage, retrieval, API
│   └── OntologyDownload/         # Ontology downloader CLI and validators
├── docs/                         # Documentation framework (guides, templates, scripts)
├── tests/                        # Automated tests and fixtures
├── openspec/                     # Spec-driven change proposals and tasks
├── requirements.in               # Optional GPU / ML dependencies
├── pyproject.toml                # Project metadata and tooling configuration
└── docs/scripts/                 # Documentation automation scripts
```

## 8. DocParsing Pipeline CLI

The DocParsing toolkit now exposes first-class module entrypoints for each stage
of the pipeline. The legacy scripts remain available but emit a
``DeprecationWarning`` on direct invocation.

```bash
# Convert HTML or PDF corpora to DocTags (auto-detects mode when possible)
python -m DocsToKG.DocParsing.core.cli doctags --input Data/HTML

# Chunk DocTags with topic-aware coalescence
python -m DocsToKG.DocParsing.core.cli chunk --min-tokens 256 --max-tokens 512

# Generate hybrid embeddings (BM25 + SPLADE + Qwen)
python -m DocsToKG.DocParsing.core.cli embed --resume

```

Use ``--help`` or append ``-- --help`` after the subcommand for the full set of
flags, including data-root overrides, resume/force controls, and tokenizer
configuration.

Synthetic benchmarking now lives in the test suite; run
``pytest tests/docparsing/test_synthetic_benchmark.py`` to exercise the
deterministic model and verify baseline performance.

## 9. Support & Community

- **Issues**: [GitHub Issues](https://github.com/paul-heyse/DocsToKG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/paul-heyse/DocsToKG/discussions)
- **License**: MIT (see [LICENSE](LICENSE))

Read `CONTRIBUTING.md` before submitting changes, and keep documentation in sync with implementation using the provided automation scripts.
