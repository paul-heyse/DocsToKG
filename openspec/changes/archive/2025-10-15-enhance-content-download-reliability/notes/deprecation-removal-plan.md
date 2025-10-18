# Follow-Up Task: Remove Legacy Resolver Exports

- **Tracking ticket:** OPS-1842 (pasted below for archival)
- **Target release:** DocsToKG 2025.12
- **Owners:** Content Download maintainers

## Removal Steps

1. Delete `time` and `requests` from `DocsToKG.ContentDownload.resolvers.__all__`.
2. Remove both modules from `_LEGACY_EXPORTS` and simplify `__getattr__` to raise
   `AttributeError` for unknown symbols.
3. `grep -R "ContentDownload.resolvers.time" src tests` and update call sites to
   import from the standard library / `requests` directly.
4. Run `pytest tests/test_resolvers_namespace.py` to confirm namespace hygiene.

## Communication

- Deprecation warning introduced in 2025.10 release with removal timeline.
- Mentioned in CHANGELOG under **Deprecated** and in release highlights.
- Downstream partners notified via ops mailing list (2025-10-15).

## Ticket Text (OPS-1842)

> **Summary:** Remove resolver namespace exports for `time` and `requests`.
> **Acceptance:** (a) Import now raises `AttributeError`. (b) Namespace docstring
> updated with removal note. (c) CHANGELOG records breaking change.
> **Rollback:** Reintroduce shim by restoring `_LEGACY_EXPORTS` entries if
> downstream adapters report breakage within 48 h of release.
