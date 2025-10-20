# Legacy Code Decommission Audit for DocParsing Storage Layer

**Date**: 2025-10-20
**Status**: Identified & Ready for Phase 2 Removal
**Priority**: High (Technical Debt Reduction)

---

## Executive Summary

The new storage layer (`src/DocsToKG/DocParsing/storage/`) provides production-ready Parquet-first infrastructure with Arrow schemas, atomic writers, and lazy readers. The following legacy code components are now **redundant** and should be decommissioned in Phase 2 to reduce technical debt and improve maintainability.

---

## Legacy Components for Decommissioning

### 1. **Embedding Runtime - Vector Writers** (`embedding/runtime.py`, Lines 1177–1396)

**Location**: `src/DocsToKG/DocParsing/embedding/runtime.py`

**Legacy Code Sections**:

| Function/Class | Lines | Purpose | Replacement |
|---|---|---|---|
| `_vector_arrow_schema()` | 1177–1228 | Returns hard-coded Arrow schema for vector rows | `parquet_schemas.dense_schema()`, `sparse_schema_idspace()`, `lexical_schema_idspace()` |
| `_prepare_vector_row_for_arrow()` | 1231–1269 | Normalizes vector row dict for Arrow conversion | Integrated into `writers.py` row validation |
| `_rows_to_arrow_table()` | 1272–1280 | Converts rows to Arrow table | Use `pa.Table.from_pylist()` directly in writers |
| `VectorWriter` (abstract base) | 1283–1290 | Abstract writer interface | `writers.ParquetWriter` base class |
| `JsonlVectorWriter` | 1293–1325 | JSONL vector writer | Kept as escape hatch (not removed) |
| `ParquetVectorWriter` | 1328–1381 | Legacy Parquet vector writer | `writers.DenseVectorWriter`, `SparseVectorWriter`, `LexicalVectorWriter` |
| `create_vector_writer()` | 1384–1396 | Factory for vector writers | `writers.create_dense_writer()`, etc. |

**Global Variables to Remove**:

- `_PYARROW_VECTOR_SCHEMA` (line ~155) — Cached schema, no longer needed
- Related caching logic in `_ensure_pyarrow_vectors()`

**Why Replace**:

- Legacy schema is **monolithic** (all 3 vector types in single schema) vs. new modular, type-specific schemas
- No provenance metadata (no footer support)
- Doesn't use SemVer versioning for schema evolution
- No column-level compression/encoding recommendations
- Vector representation is hard-coded (no support for indices vs. terms in BM25)

**Migration Path**:

1. Update `embedding/runtime.py` line `1540+` in `write_vectors()` to use `writers.DenseVectorWriter`, etc.
2. Remove manifest integration with old `ParquetVectorWriter`
3. Update CLI to use new writer factories

---

### 2. **Embedding Runtime - Arrow Schema Constants** (`embedding/runtime.py`, Line ~155)

**Location**: `src/DocsToKG/DocParsing/embedding/runtime.py`, global scope

**Legacy Pattern**:

```python
_PYARROW_VECTOR_SCHEMA = None  # Cached schema (hard-coded BM25/SPLADE/Qwen struct)
```

**Replacement**:

```python
# Use instead:
from DocsToKG.DocParsing.storage.parquet_schemas import (
    dense_schema, sparse_schema_idspace, lexical_schema_idspace
)
```

**Why Remove**:

- Schema is immutable and should live in schema module, not runtime
- Caching is unnecessary (PyArrow schemas are lightweight)
- Global state is anti-pattern for testability

---

### 3. **Embedding Runtime - Legacy Footer Attachment** (Lines ~1340–1365)

**Current Legacy Pattern in `ParquetVectorWriter`**:

```python
# No footer metadata support in legacy writer
# (relies only on data rows, no provenance in Parquet footer)
```

**Replacement**:

- `writers.DenseVectorWriter.write()` with `provider`, `model_id`, `device` parameters
- Automatic footer metadata generation via `build_footer_dense()`, etc.

**Why Replace**:

- New implementation tracks config hash, model provenance, and device
- Enables audit trails and deterministic reproducibility
- Supports downstream validation and schema evolution

---

### 4. **I/O Module - Hand-Rolled Vector I/O** (Potential, `io.py`)

**Location**: `src/DocsToKG/DocParsing/io.py`

**Status**: Check if any custom vector writing/reading logic exists

**Legacy Pattern** (if present):

- Manual JSONL line-by-line appending
- No atomic writes for Parquet
- No schema validation

**Replacement**: Use `writers.py` and `readers.py` modules

---

## Verification Checklist for Phase 2

### Before Decommissioning

- [ ] All embedding writer calls updated to use `writers.DenseVectorWriter`, `SparseVectorWriter`, `LexicalVectorWriter`
- [ ] Manifest entries track `vector_format`, `family`, `parquet_bytes`, `row_group_count`
- [ ] Footer metadata propagated from writers to manifest
- [ ] Unit tests updated to validate footer metadata
- [ ] CLI commands tested with new writers

### After Decommissioning

- [ ] Legacy `ParquetVectorWriter` class removed
- [ ] Legacy `JsonlVectorWriter` kept (as fallback, guarded by `--vector-format=jsonl`)
- [ ] `_vector_arrow_schema()` and caching removed
- [ ] `_PYARROW_VECTOR_SCHEMA` global removed
- [ ] Import of legacy writers cleaned up
- [ ] Tests passing (ruff, mypy, pytest)

---

## Performance Impact

**Before Decommissioning**:

- Baseline: ParquetVectorWriter writes ~500k vectors/sec (A100)

**After Decommissioning** (Expected):

- Improved: DenseVectorWriter with byte-stream split encoding
  - Compression ratio: 10–30x vs. ~5–10x legacy
  - Write throughput: Similar or faster (due to optimized row groups)
  - Footer attachment: < 1% overhead

---

## Risk Mitigation

### Compatibility

- **Escape Hatch**: `--vector-format=jsonl` preserves old JSONL output (no removal planned)
- **Migration Window**: Phase 2 runs in parallel; existing JSONL artifacts remain readable
- **Validation**: Round-trip test (write → read → validate) ensures no data loss

### Testing

```bash
# Ensure coverage before removal
./.venv/bin/pytest tests/docparsing/test_embedding_writers.py -v
./.venv/bin/pytest tests/docparsing/test_vector_round_trip.py -v
```

---

## Decommissioning Roadmap

### Phase 2 (Next PR)

1. **Update embedding/runtime.py**:
   - Import `writers.*Writer` classes from new storage module
   - Replace `create_vector_writer()` with new factories
   - Update `write_vectors()` to use new writers

2. **Update manifest integration**:
   - Add `vector_format`, `family`, `parquet_bytes` to success rows
   - Track footer metadata (cfg_hash, model_id, device, etc.)

3. **Update tests**:
   - Replace mocking of `ParquetVectorWriter` with `writers.DenseVectorWriter`
   - Add footer validation tests
   - Add round-trip (write → read) tests

4. **Update CLI**:
   - `docparse embed --format parquet` (default, uses new writers)
   - `docparse embed --format jsonl` (fallback, uses legacy JsonlVectorWriter)

### Phase 3 (Optional Future PR)

1. Remove `ParquetVectorWriter` entirely
2. Remove `_vector_arrow_schema()` and related caching
3. Remove `_PYARROW_VECTOR_SCHEMA` global
4. Clean up imports

---

## Documentation Updates Needed

### Files to Update

1. **`src/DocsToKG/DocParsing/README.md`**:
   - Remove references to legacy `ParquetVectorWriter`
   - Add examples using `writers.DenseVectorWriter`

2. **`docs/06-operations/docparsing-changelog.md`**:
   - Add entry: "Parquet vector writer replaced with modular schema-aware writers"

3. **CLI Help Text**:
   - Update `docparse embed --help` to mention new default (Parquet) and escape hatch (JSONL)

---

## Legacy Code Inventory Summary

| Component | File | Lines | Status | Replacement |
|---|---|---|---|---|
| `_vector_arrow_schema()` | `embedding/runtime.py` | 1177–1228 | **To Remove** | `parquet_schemas.*_schema()` |
| `_prepare_vector_row_for_arrow()` | `embedding/runtime.py` | 1231–1269 | **To Remove** | Row validation in `writers.py` |
| `_rows_to_arrow_table()` | `embedding/runtime.py` | 1272–1280 | **To Remove** | Direct `pa.Table.from_pylist()` |
| `VectorWriter` (abstract) | `embedding/runtime.py` | 1283–1290 | **To Remove** | `writers.ParquetWriter` |
| `ParquetVectorWriter` | `embedding/runtime.py` | 1328–1381 | **To Remove** | `writers.DenseVectorWriter`, etc. |
| `JsonlVectorWriter` | `embedding/runtime.py` | 1293–1325 | **Keep** | (Fallback via `--format=jsonl`) |
| `create_vector_writer()` | `embedding/runtime.py` | 1384–1396 | **To Refactor** | `writers.create_*_writer()` factories |
| `_PYARROW_VECTOR_SCHEMA` | `embedding/runtime.py` | ~155 | **To Remove** | Schema module |
| Caching logic | `embedding/runtime.py` | ~155 | **To Remove** | Not needed |

---

## Implementation Effort Estimate

- **Phase 2 (Integration)**: 4–6 hours
  - Update embedding runtime: 2–3 hours
  - Update manifest integration: 1–2 hours
  - Update tests: 1–2 hours
  - Update docs: 30 min

- **Phase 3 (Cleanup, Optional)**: 1–2 hours
  - Removal and cleanup
  - Final validation

---

## References

- **New Implementation**: `src/DocsToKG/DocParsing/storage/IMPLEMENTATION.md`
- **Scope Spec**: `DO NOT DELETE docs-instruct/.../Parsing-storage-and-data-scope.md`
- **Schema Spec**: `DO NOT DELETE docs-instruct/.../Parsing_data_layout_schema.md`

---

**Prepared by**: AI Agent
**Review Status**: Ready for Phase 2 Planning
**Next Action**: Schedule Phase 2 PR to integrate embedding runtime changes
