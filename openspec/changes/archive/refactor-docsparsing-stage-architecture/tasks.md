## 1. Package Decomposition
- [ ] 1.1 Map current embedding/chunking imports and define target package graph (config, cli, runtime, workers).
- [ ] 1.2 Move embedding stage code into the new modules and add re-export façades to keep existing imports working.
- [ ] 1.3 Repeat the package split for chunking, updating tests and any shared helpers that assumed a flat module.

## 2. Configuration Layering
- [ ] 2.1 Extend `StageConfigBase` to detect unknown keys and emit actionable errors or warnings.
- [ ] 2.2 Implement explicit clear semantics (CLI flags, empty string/null handling) and cover them with unit tests.
- [ ] 2.3 Update stage configs (`EmbedCfg`, `ChunkerCfg`, others) and fixtures to align with the stricter behavior.

## 3. Runtime Context & CLI Hardening
- [ ] 3.1 Introduce scoped managers for HTTP sessions and stage telemetry; replace `_STAGE_TELEMETRY` and `_HTTP_SESSION`.
- [ ] 3.2 Update CLI entrypoints to use the new context, log bootstrap failures, and propagate non-zero exits.
- [ ] 3.3 Bolster tests to exercise concurrent runs, telemetry cleanup, and CLI bootstrap error reporting.
