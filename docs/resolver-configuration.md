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
