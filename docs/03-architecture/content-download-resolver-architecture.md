# 1. Content Download Resolver Architecture

```mermaid
graph TD
    A[Work Artifact] --> B[ResolverPipeline]
    B --> C{Resolver Execution}
    C -->|OpenAlexResolver| D1[OpenAlex URLs]
    C -->|Provider Modules| D2[Zenodo]
    C -->|Provider Modules| D3[Figshare]
    C -->|Provider Modules| D4[Other Resolvers]
    D1 --> E[Download Function]
    D2 --> E
    D3 --> E
    D4 --> E
    E --> F[ConditionalRequestHelper]
    F -->|304 Cached| G[Manifest Update]
    F -->|200 Modified| H[HTTP Download]
    H --> I[DownloadOutcome]
    I --> J[Attempt Logger]
    J --> K[JSONL / CSV Logs]
    I --> L[ResolverMetrics]
    L --> M[Wall-Time & Rate Statistics]
```

## 1. Module Responsibilities

- **`pipeline.py`** – orchestrates resolver execution, respects rate limits,
  enforces concurrency settings, and performs HEAD pre-check filtering.
- **`types.py`** – houses resolver dataclasses, configuration validation, and
  attempt logging interfaces.
- **`providers/*`** – individual resolver implementations with defensive error
  handling and metadata-rich events.
- **`http.py`** – centralised retry/backoff utilities with Retry-After
  compliance and structured logging.
- **`conditional.py`** – interprets cached vs modified responses and validates
  manifest metadata.

## 2. Related DocsToKG References

- `docs/resolver-configuration.md` – configuration details and YAML snippets.
- `docs/adding-custom-resolvers.md` – extensibility guidance for bespoke
  providers.
