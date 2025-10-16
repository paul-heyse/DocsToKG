# 1. Content Download Migration Guide

This brief guide helps downstream consumers migrate to the refactored content
download pipeline introduced in the `refactor-content-download-robustness`
change set.

## 1. Removed Compatibility Shims

- `DocsToKG.ContentDownload.resolvers.time` and
  `DocsToKG.ContentDownload.resolvers.requests` were deleted. Import these
  modules directly from the standard library / requests package.
- `DocsToKG.ContentDownload.resolvers.request_with_retries` proxy was removed.
  Use `from DocsToKG.ContentDownload.network import request_with_retries`.
- Deprecated module aliases (`DocsToKG.ContentDownload.results`, `.similarity`,
  `.retrieval`, `.operations`, `.schema`) are no longer available; import the
  consolidated modules (`ranking`, `vectorstore`, `service`, `storage`)
  instead.

## 2. Resolver Implementations

- New resolvers should inherit from `ApiResolverBase` to leverage shared error
  handling via `_request_json()`.
- Session-less resolver branches and cached `_fetch_*` helpers were removed –
  every resolver must use the provided `requests.Session` instance and the
  shared network helpers.

## 3. CLI Behaviour Changes

- Use `--staging` to create timestamped run directories with separate `PDF/`
  and `HTML/` folders plus `manifest.jsonl`, `manifest.index.json`, and
  `manifest.metrics.json` sidecars.
- The CLI always writes JSONL manifests. When `--log-format csv` is supplied
  it additionally produces `manifest.last.csv` summarising the latest attempt
  per work.
- Domain-level throttling (`--domain-min-interval`) and global URL deduplication
  (`--global-url-dedup`) now apply across worker threads.
- Resume runs tolerate manifests with incomplete metadata; partial entries no
  longer raise errors and conditional requests fall back to full downloads when
  required fields are missing.

## 4. Manifest Consumers

- Manifest entries maintain their historical schema. New sidecar files
  (`.index.json`, `.metrics.json`, `manifest.last.csv`) provide faster lookup
  surfaces for resumption tools and operational dashboards.
- HTTP corruption heuristics now record whether HEAD prechecks succeeded via
  the `head_precheck_passed` manifest flag.

Update any automation, dashboards, or imports to reflect these changes before
upgrading to DocsToKG 0.3.0.
