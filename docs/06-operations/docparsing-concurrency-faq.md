# DocParsing Concurrency FAQ

## How do I enable forensic lock retention?

Set the environment variable `DOCSTOKG_RETAIN_LOCK_FILES_ON_ERROR=1` (or `true`,
`yes`, `on`) before invoking DocParsing workflows. When enabled, the
`safe_write` helper and `JsonlWriter` manifest sink retain their `.lock`
sentinels whenever a write operation raises an exception. This allows operators
to inspect which paths were mid-write during a crash and correlate them with
stalled processes.

To retain lock sentinels for every write (successes included), use
`DOCSTOKG_RETAIN_LOCK_FILES=1`. Both flags can be combined; the "always on"
flag wins when set.

## Can I retain lock files programmatically?

Yes. Any direct call to `safe_write` may request forensic retention without
changing global environment variables by passing
`retain_lock_on_error=True`. The lock sentinel remains only when the supplied
`write_fn` raises an exception, mirroring the environment variable behaviour.

The manifest and telemetry writers honour the same environment variables via
`DocsToKG.DocParsing.io.DEFAULT_JSONL_WRITER`, so no additional wiring is
required for standard pipeline runs.
