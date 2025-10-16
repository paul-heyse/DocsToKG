# 1. Content Download Pipeline Migration Guide

This guide summarises the key operational updates introduced by the
``refactor-content-download-pipeline`` change within DocsToKG.

## 2. Config Changes

- ``resolver_rate_limits`` has been replaced with ``resolver_min_interval_s``.
  Update YAML/JSON resolver configurations accordingly.
- Rate limits express the minimum seconds between requests (e.g. ``1.0`` means
  at most one request per second).

## 3. Logging Changes

- Attempts and manifest records now stream to JSON Lines (``.jsonl``) files.
- To export CSV, use the helper script: ``python scripts/export_attempts_csv.py``
  or run ``jq``: ``jq -r '[.timestamp,.work_id,...]|@csv' attempts.jsonl > attempts.csv``.

## 4. CLI Additions

- ``--workers``: controls parallel worker count (default ``1``).
- ``--dry-run``: measures resolver coverage without writing files.
- ``--resume-from``: skips works already logged in an existing manifest JSONL.
- ``--extract-html-text``: extracts plaintext from HTML fallbacks (requires
  ``trafilatura``).

## 5. Ontology Downloader API Migration (``harden-ontology-downloader-core``)

- Replace imports that referenced ``DocsToKG.OntologyDownload.core`` (and other legacy
  module aliases such as ``.config``, ``.validators``, ``.download``) with
  ``from DocsToKG.OntologyDownload import ...`` using the exported names listed in
  ``DocsToKG.OntologyDownload.__all__``. Direct module consumers should import
  ``DocsToKG.OntologyDownload.ontology_download`` or ``.cli`` explicitly.
- Update custom tooling that parsed rate limit strings, directory sizes, or version
  timestamps to call the shared helpers ``parse_rate_limit_to_rps``, ``_directory_size``,
  ``parse_iso_datetime``, and ``parse_version_timestamp`` from the core module.
- Configurations that relied on planning to warn (but proceed) for disallowed hosts now
  receive a ``ConfigError`` during planning; adjust allowlists before upgrading.
