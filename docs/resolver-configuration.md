# 1. Resolver Configuration Guide

The DocsToKG content download pipeline exposes a flexible
:class:`ResolverConfig` structure that can be customised through configuration
files or command line overrides. This guide documents the most common options
introduced with the modular resolver architecture.

## 1. Enabling Concurrency

To opt into bounded intra-work concurrency, set
``max_concurrent_resolvers`` and ensure each resolver has an appropriate rate
limit:

```yaml
max_concurrent_resolvers: 3
resolver_min_interval_s:
  unpaywall: 1.0
  crossref: 0.5
```

The example above allows up to three resolvers to run simultaneously while
respecting Unpaywall and Crossref rate limits.

At the CLI level the downloader exposes matching controls:

```bash
python -m DocsToKG.ContentDownload.download_pyalex_pdfs \
    --concurrent-resolvers 3 \
    --head-precheck \
    --accept "application/pdf,text/html;q=0.9"
```

## 2. Tuning HEAD Pre-checks

Conditional HEAD requests reduce wasted downloads by filtering obvious HTML
landing pages. Use the following snippet to disable HEAD filtering for
specific providers while leaving the feature enabled globally:

```yaml
enable_head_precheck: true
resolver_head_precheck:
  wayback: false
```

Resolvers omitted from ``resolver_head_precheck`` inherit the global
``enable_head_precheck`` value.

## 3. Enabling Global URL Deduplication

Large topic crawls often surface the same PDF across multiple works. When the
``enable_global_url_dedup`` flag is set the pipeline records URLs in a
process-wide set and skips repeat downloads with a ``duplicate-url-global``
reason. Operators can enable the feature either in configuration files:

```yaml
enable_global_url_dedup: true
```

or ad-hoc via the CLI:

```bash
python -m DocsToKG.ContentDownload.download_pyalex_pdfs \
    --topic "knowledge graphs" \
    --year-start 2020 --year-end 2023 \
    --global-url-dedup
```

The feature remains disabled by default to preserve legacy behaviour.

## 4. Configuring Domain-Level Rate Limiting

Some publishers share infrastructure across multiple resolvers. The
``domain_min_interval_s`` mapping enforces minimum spacing between requests to a
hostname regardless of the originating resolver. Configure the interval in
seconds per domain:

```yaml
domain_min_interval_s:
  example.org: 0.75
  publisher.edu: 1.5
```

CLI runs can supply the same information with repeated
``--domain-min-interval`` flags. Domains are matched case-insensitively:

```bash
python -m DocsToKG.ContentDownload.download_pyalex_pdfs \
    --domain-min-interval example.org=0.5 \
    --domain-min-interval publisher.edu=1.5
```

Domain-level limits complement existing resolver-level throttles and are
particularly helpful when multiple providers resolve to the same host.

## 5. Adjusting Classification Heuristics

The downloader exposes configurable heuristics for classifying streamed
payloads and validating PDFs. Set the knobs in resolver configuration files to
override the defaults (which mirror historical behaviour):

```yaml
sniff_bytes: 65536        # auto-upgrade ambiguous streams to PDF after n bytes
min_pdf_bytes: 1024       # PDFs smaller than this are treated as corrupt
tail_check_bytes: 2048    # number of bytes scanned for the %%EOF marker
```

These values propagate through the resolver pipeline context so that
`download_candidate` applies them consistently across worker threads. Lower
`sniff_bytes` when dealing with very small PDFs, or raise `tail_check_bytes`
for publishers that append trailers after the EOF marker.

## 6. Staging Mode and Derived Logs

Long-running harvests benefit from isolated output folders and richer manifest
artifacts. The downloader exposes a `--staging` flag that creates a timestamped
run directory containing all artifacts for that invocation:

```bash
python -m DocsToKG.ContentDownload.download_pyalex_pdfs \
    --topic "graph embeddings" \
    --year-start 2022 --year-end 2024 \
    --out ./runs \
    --staging
```

With staging enabled the CLI writes:

- `runs/YYYYMMDD_HHMM/PDF/` for downloaded PDFs
- `runs/YYYYMMDD_HHMM/HTML/` for captured HTML fallbacks
- `runs/YYYYMMDD_HHMM/manifest.jsonl` for manifest records
- `runs/YYYYMMDD_HHMM/manifest.metrics.json` for aggregate resolver metrics

Generate the sidecar index and last-attempt CSV after the run completes:

```bash
python tools/manifest_to_index.py runs/YYYYMMDD_HHMM/manifest.jsonl \
  runs/YYYYMMDD_HHMM/manifest.index.json
python tools/manifest_to_csv.py runs/YYYYMMDD_HHMM/manifest.jsonl \
  runs/YYYYMMDD_HHMM/manifest.last.csv
```

When `--log-format csv` is supplied the downloader still emits JSONL manifests
and writes an attempts CSV mirror. Run `tools/manifest_to_csv.py` to produce a
`manifest.last.csv` file that captures the latest manifest record per work for
rapid auditing:

```bash
python -m DocsToKG.ContentDownload.download_pyalex_pdfs \
    --topic "graph embeddings" \
    --year-start 2022 --year-end 2024 \
    --out ./runs \
    --manifest ./runs/manifest.jsonl \
    --log-format csv
```

The CSV header follows the manifest schema (`work_id`, `resolver`, `classification`,
`sha256`, etc.) enabling downstream tooling to consume it without parsing JSONL.
