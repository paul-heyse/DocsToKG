# Content Download Pipeline Migration Guide

This guide summarises the key operational updates introduced by the
``refactor-content-download-pipeline`` change.

## Config Changes

- ``resolver_rate_limits`` has been replaced with ``resolver_min_interval_s``.
  Update YAML/JSON resolver configurations accordingly.
- Rate limits express the minimum seconds between requests (e.g. ``1.0`` means
  at most one request per second).

## Logging Changes

- Attempts and manifest records now stream to JSON Lines (``.jsonl``) files.
- To export CSV, use the helper script: ``python scripts/export_attempts_csv.py``
  or run ``jq``: ``jq -r '[.timestamp,.work_id,...]|@csv' attempts.jsonl > attempts.csv``.

## CLI Additions

- ``--workers``: controls parallel worker count (default ``1``).
- ``--dry-run``: measures resolver coverage without writing files.
- ``--resume-from``: skips works already logged in an existing manifest JSONL.
- ``--extract-html-text``: extracts plaintext from HTML fallbacks (requires
  ``trafilatura``).
