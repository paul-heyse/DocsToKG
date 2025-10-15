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
