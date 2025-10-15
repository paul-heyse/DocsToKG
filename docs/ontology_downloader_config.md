# Ontology Downloader Configuration

The ontology downloader uses a YAML configuration file (typically `sources.yaml`) to describe
where ontologies should be downloaded from and how the pipeline behaves. The file is divided
into a `defaults` section and an `ontologies` list.

## Installation

Install the downloader and all required dependencies inside your virtual environment:

```bash
pip install -e .[ontology-download]
```

If you prefer to install the minimal dependency set manually, ensure the following packages are
available: `bioregistry`, `oaklib`, `ols-client`, `ontoportal-client`, `rdflib`, `pronto`,
`owlready2`, `arelle`, `pystow`, `pooch`, `pyyaml`, and `requests`.

## Pystow Configuration

The downloader stores cache, configuration, and logs using [pystow](https://pystow.readthedocs.io/)
under `~/.data/ontology-fetcher/` by default. Override locations and credentials via environment
variables:

| Variable | Purpose |
| --- | --- |
| `ONTOFETCH_HOME` | Override the root directory for cache/config storage. |
| `ONTOFETCH_MAX_RETRIES` | Override HTTP retry count. |
| `ONTOFETCH_TIMEOUT_SEC` | Override HTTP timeout. |
| `ONTOFETCH_PER_HOST_RATE_LIMIT` | Override per-host rate limit (e.g., `2/second`). |
| `ONTOFETCH_BACKOFF_FACTOR` | Override exponential backoff factor. |
| `ONTOFETCH_LOG_LEVEL` | Override default log level. |

BioPortal API keys can be stored at `~/.data/ontology-fetcher/configs/bioportal_api_key.txt`. OLS
credentials (if required) can be stored in `~/.data/ontology-fetcher/configs/ols_api_token.txt`.

## Defaults

| Key | Type | Description |
| --- | ---- | ----------- |
| `accept_licenses` | list of strings | Allowed license identifiers. Downloads fail closed if a license is not listed. |
| `normalize_to` | list of strings | Formats to produce in the normalization step (e.g., `ttl`, `obographs`). |
| `prefer_source` | list of strings | Resolver preference order for ad-hoc downloads. |
| `http` | mapping | HTTP behaviour (`max_retries`, `timeout_sec`, `download_timeout_sec`, `backoff_factor`, `per_host_rate_limit`, `max_download_size_gb`, `concurrent_downloads`). |
| `validation` | mapping | Validation timeouts and limits (`parser_timeout_sec`, `max_memory_mb`, `skip_reasoning_if_size_mb`). |
| `logging` | mapping | Structured logging configuration (`level`, `max_log_size_mb`, `retention_days`). |
| `continue_on_error` | bool | Continue batch processing after an error (default `true`). |
| `concurrent_downloads` | int | Future parallelism control (currently informational). |

## Ontology Entries

Each entry in `ontologies` must provide at least an `id`. Optional keys include `resolver`
(one of `obo`, `ols`, `bioportal`, `skos`, `xbrl`), `extras` (resolver-specific parameters), and
`target_formats` (format priority for the resolver).

## Example

```yaml
defaults:
  accept_licenses: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"]
  normalize_to: ["ttl", "obographs"]
  prefer_source: ["obo", "ols", "bioportal", "direct"]
  http:
    max_retries: 5
    timeout_sec: 30
    download_timeout_sec: 300
    backoff_factor: 0.5
    per_host_rate_limit: "4/second"
    max_download_size_gb: 5
    concurrent_downloads: 2
  validation:
    parser_timeout_sec: 60
    max_memory_mb: 2048
    skip_reasoning_if_size_mb: 500
  logging:
    level: "INFO"
    max_log_size_mb: 100
    retention_days: 30
  continue_on_error: true

ontologies:
  - id: hp
    resolver: obo
    target_formats: [owl, obo]
  - id: efo
    resolver: ols
    target_formats: [owl]
  - id: ncit
    resolver: bioportal
    extras:
      acronym: NCIT
  - id: eurovoc
    resolver: skos
    extras:
      url: https://op.europa.eu/o/opportal-service/euvoc-download-handler?cellarURI=http%3A%2F%2Fpublications.europa.eu%2Fresource%2Feurovoc
  - id: ifrs
    resolver: xbrl
    extras:
      url: https://example.org/ifrs-taxonomy.zip
```

When a resolver requires additional hints, populate the `extras` mapping:

| Resolver | Required extras | Example |
| --- | --- | --- |
| `bioportal` | `acronym` (defaults to uppercase `id` if omitted) | `acronym: NCIT` |
| `skos` | `url` pointing directly to the SKOS/RDF download | `url: https://.../eurovoc.ttl` |
| `xbrl` | `url` pointing to the taxonomy ZIP archive | `url: https://example.org/ifrs-taxonomy.zip` |

## CLI Usage

Use the `ontofetch` command to operate the downloader:

```bash
# Download ontologies declared in sources.yaml
ontofetch pull --spec ~/.data/ontology-fetcher/configs/sources.yaml

# Download a single ontology using defaults
ontofetch pull hp --resolver obo --target-formats owl,obo

# Show stored manifest information
ontofetch show hp --versions

# Re-run validators on an existing download
ontofetch validate hp --json --rdflib --pronto

# Validate configuration files (long-form and shortcut)
ontofetch config validate --spec ./sources.yaml --json
ontofetch config-validate --spec ./sources.yaml
```

All subcommands accept `--json` for machine-readable output and `--log-level` to adjust verbosity.

## Resolver Requirements

| Resolver | Requirements |
| --- | --- |
| `obo` / `bioregistry` | Internet access to OBO Library mirrors. |
| `ols` | Public instances require no credentials; private instances may require API tokens stored at `~/.data/ontology-fetcher/configs/ols_api_token.txt`. |
| `bioportal` | BioPortal API key stored at `~/.data/ontology-fetcher/configs/bioportal_api_key.txt`. |
| `skos` | Direct HTTPS URL to RDF/SKOS asset. |
| `xbrl` | HTTPS URL to taxonomy ZIP; Arelle must be installed for validation. |

ROBOT conversions require a Java runtime and the `robot` CLI in `PATH`.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `HTTP 401/403` from BioPortal/OLS | Ensure API keys exist at the paths above and have the correct permissions. |
| Repeated download retries | Increase backoff/timeout via environment variables; check network connectivity. |
| `No space left on device` | Free disk space under the configured cache directory. Partial `.part` files are cleaned automatically. |
| Validation memory errors | Increase available memory or disable heavy validators using CLI flags (e.g., `--no-robot`). |
| Rate-limit warnings | Reduce concurrency or adjust `per_host_rate_limit` in configuration. |
| `Permission denied writing to` | Set `PYSTOW_HOME` to a writable directory and retry. |

## Storage Layout & Manifest Schema

Downloaded ontologies are stored under `~/.data/ontology-fetcher/ontologies/<id>/<version>/` with
sub-directories `original/`, `normalized/`, and `validation/`. The `manifest.json` file includes:

| Field | Description |
| --- | --- |
| `id` | Ontology identifier. |
| `resolver` | Resolver used to obtain the artifact. |
| `url` | Final HTTPS URL of the download. |
| `filename` | Stored filename in `original/`. |
| `version` | Resolver-provided version (or timestamp). |
| `license` | Declared license if available. |
| `status` | `fresh`, `cached`, or `updated`. |
| `sha256` | SHA-256 checksum of the downloaded file. |
| `etag`, `last_modified` | HTTP provenance headers. |
| `downloaded_at` | UTC timestamp of the download. |
| `target_formats` | Requested output formats. |
| `validation` | Validator results keyed by validator name. |
| `artifacts` | List of stored artifact paths (including extracted ZIP contents). |

## Example Workflows

1. **Batch download**: Populate `sources.yaml`, then run `ontofetch pull --spec sources.yaml` to
   download all configured ontologies in one run.
2. **Incremental update**: Re-run `ontofetch pull --spec sources.yaml` periodically; ETag/Last-Modified
   headers prevent redundant downloads. Use `--force` to bypass caches when required.
3. **Validation-only reruns**: For previously downloaded ontologies, execute `ontofetch validate hp@2024-01-01 --robot`
   to regenerate validation reports without redownloading.

## Validation

Validate configuration files using the CLI:

```bash
ontofetch config validate --spec path/to/sources.yaml
```

The command exits with a non-zero status if structural issues are detected. Missing configuration files
emit a clear message and exit with code `2`.
