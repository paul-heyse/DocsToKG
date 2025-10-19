## 1. Discovery & Configuration Plumbing
- [ ] 1.1 Add `vector_artifact_name(logical: Path, fmt: str) -> Path` to `src/DocsToKG/DocParsing/core/discovery.py`; normalize `fmt`, guard unsupported values, and return the relative vectors filename (`.vectors.jsonl` or `.vectors.parquet`).
- [ ] 1.2 Update `derive_doc_id_and_vectors_path` to accept an explicit `vector_format` argument, call `vector_artifact_name`, and adjust all call sites (`embedding/runtime.py`, `core/planning.py`, `tests/docparsing/**`) to pass the resolved format instead of assuming JSONL.
- [ ] 1.3 Extend `ParsingContext` (`src/DocsToKG/DocParsing/context.py`) to retain `vector_format` with a default of `"jsonl"`; ensure `apply_config` and `to_manifest` continue to surface the field.
- [ ] 1.4 Update `EmbedCfg` (`embedding/config.py`) to validate `vector_format` against `{"jsonl", "parquet"}`, honour `DOCSTOKG_EMBED_VECTOR_FORMAT`, and raise `EmbeddingCLIValidationError` for invalid values with the name of the offending option.
- [ ] 1.5 Ensure `_build_stage_args` in `core/cli.py` forwards `--embed-format` to the embed stage; add logging metadata so run headers include `vector_format`.
- [ ] 1.6 Broaden the embedding CLI (`embedding/cli.py`) to expose both formats in `--vector-format/--format`, update help text, and include the choice in `EMBED_CLI_OPTIONS` defaults. Synchronise the tests that assert available choices (`tests/docparsing/test_embedding_cli_formats.py`).

## 2. Parquet Writer Runtime
- [ ] 2.1 Add a lazy `pyarrow` import helper in `embedding/runtime.py` that either returns the module or raises `EmbeddingCLIValidationError` with remediation instructions (`pip install "DocsToKG[docparse-parquet]"`) when Parquet is requested.
- [ ] 2.2 Define a module-level Arrow schema or builder that maps `VectorRow` dictionaries to Arrow columns (string UUID, struct BM25/SPLADE, dense vector as `list<float32>`, metadata serialised as JSON text).
- [ ] 2.3 Implement `ParquetVectorWriter` mirroring `JsonlVectorWriter` semantics: open via `atomic_write`, batch rows, convert to `pyarrow.Table`, write with `pyarrow.parquet.write_table`, and respect `_crash_after_write` fault injection (raise after `n` rows to reuse durability tests).
- [ ] 2.4 Update `create_vector_writer` to instantiate the new writer. Include format-normalisation, dependency lookup, and actionable errors when `pyarrow` is unavailable or too old.
- [ ] 2.5 Adjust `process_chunk_file_vectors` to pass `args.vector_format` into the writer factory, ensure per-batch payloads remain dictionaries for validation, and record the written path in the manifest using the new helper.
- [ ] 2.6 Emit additional runtime logs when Parquet is selected (e.g., include `writer="parquet"` in "Embedding file written" events) so observability dashboards can distinguish formats.

## 3. Resume, Planning, Validation
- [ ] 3.1 Introduce a format-aware reader utility (e.g., `_iter_vector_rows(path: Path, fmt: str)`) that streams JSONL via `iter_rows_in_batches` or Parquet via `pyarrow.parquet.ParquetFile.iter_batches`, yielding dictionaries for schema validation.
- [ ] 3.2 Modify `_validate_vectors_for_chunks` to accept `vector_format`, call the new reader, and surface consistent error messages (including listing `.parquet` paths) when files are missing or invalid.
- [ ] 3.3 Update resume logic (`ResumeController.should_skip`, `embedding/runtime.py`’s skip branch) to compare both manifest format and output path. If the requested format differs from the manifest’s recorded format, force regeneration.
- [ ] 3.4 Ensure quarantine helpers (`_handle_embedding_quarantine`) preserve the correct extension when renaming corrupted outputs.
- [ ] 3.5 Revise `plan_embed` to thread format through discovery, update plan previews to show the correct suffix, and extend associated planner tests to cover both formats.
- [ ] 3.6 Confirm `compute_content_hash` usage remains correct: hash chunk inputs as today; for resume hash verification that touches vector files (e.g., `plan --verify-hash`), add coverage to show Parquet files are hashed successfully.

## 4. HybridSearch & Downstream Consumers
- [ ] 4.1 Enhance `src/DocsToKG/HybridSearch/pipeline.py` and `service.py` ingestion paths to branch on vector suffix, calling either the current JSONL reader or a new parquet reader backed by `pyarrow` (reuse DocParsing reader where possible).
- [ ] 4.2 Add detection for mixed-format datasets (within a namespace or manifest batch) and raise a descriptive `IngestError` that enumerates conflicting files and the expected format.
- [ ] 4.3 Update helper scripts (`scripts/build_real_hybrid_fixture.py`, `examples/hybrid_search_quickstart.py`) to accept a `--vector-format` option and honour the appropriate suffix when copying or generating fixture data.
- [ ] 4.4 Extend dataset utilities and documentation to note that parquet ingestion requires the HybridSearch environment to install the same `pyarrow` extra.

## 5. Fixtures & Test Data
- [ ] 5.1 Generate a parquet twin of the existing sample vectors (e.g., `tests/data/vectors/sample.vectors.parquet`) using the new writer to guarantee schema fidelity.
- [ ] 5.2 Update `tests/data/hybrid_dataset.jsonl` (and any other manifests) to reference both JSONL and parquet variants where appropriate for test coverage.
- [ ] 5.3 Ensure git attributes or LFS rules accommodate the new parquet artifacts if size exceeds thresholds.

## 6. Automated Tests
- [ ] 6.1 Add unit tests for `ParquetVectorWriter` covering roundtrip serialization, crash rollback, and dependency error handling.
- [ ] 6.2 Parameterize DocParsing runtime tests (`tests/docparsing/test_embedding_resume_behavior.py`, `test_embedding_runtime_validation.py`, `test_core_submodules.py`, `test_docparsing_core.py`, `test_planning_streaming.py`) so they exercise both formats.
- [ ] 6.3 Extend CLI surface tests to assert that `docparse embed --format parquet` executes end-to-end via the CLI harness (e.g., using `_run_stage` helpers).
- [ ] 6.4 Add a regression test that runs `docparse embed --format parquet --validate-only` against generated fixtures and confirms manifests contain `vector_format: "parquet"`.
- [ ] 6.5 Expand HybridSearch test suites (`tests/hybrid_search/test_suite.py`) to ingest parquet vectors, validate feature extraction, and verify mixed-format guardrails.
- [ ] 6.6 Update snapshot/manifest tests to cover parquet path derivation (e.g., planning summaries, manifest log entries).

## 7. Documentation & Developer Experience
- [ ] 7.1 Revise `src/DocsToKG/DocParsing/README.md` (CLI sections, pipeline outputs, quickstart) to mention both formats, include sample commands, and document the `DOCSTOKG_EMBED_VECTOR_FORMAT` environment variable.
- [ ] 7.2 Update `openspec/AGENTS.md` and any other agent-facing runbooks to highlight the Parquet option, required dependencies, and validation workflows.
- [ ] 7.3 Adjust HybridSearch README and operational guides to describe parquet ingestion requirements and troubleshooting steps for mixed datasets.
- [ ] 7.4 Confirm `pyproject.toml` extras list already includes `pyarrow`; if not, add or update an extra (e.g., `docparse-parquet`) and document installation instructions.

## 8. Rollout & QA
- [ ] 8.1 Capture log excerpts and manifest samples from both JSONL and Parquet runs to include in release notes / QA artifacts.
- [ ] 8.2 Confirm that environments without `pyarrow` emit the expected validation error by running the embed stage in a controlled sandbox without the dependency.
- [ ] 8.3 Notify downstream owners (HybridSearch, analytics consumers) about the new capability and provide guidance on opting into parquet while reiterating that JSONL remains the default.
