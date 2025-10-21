# TASK 2.3: Advanced Features - Query Profiling & Schema Introspection

**Date**: October 21, 2025
**Status**: Implementation Planning
**Estimated Duration**: 1-2 days
**Scope**: ~300 LOC production, 15-20 tests
**Quality Target**: 100/100

---

## 📋 SCOPE

### Query Profiling (Part 1)
Performance analysis and optimization recommendations.

**Profiling Methods** (3 total):
1. `profile_query()` - EXPLAIN ANALYZE for query plans
2. `estimate_query_cost()` - Cost estimation
3. `optimize_suggestions()` - Performance recommendations

**Profile DTO**:
- `QueryProfile` - Query plan, timing, rows, cost metrics

### Schema Introspection (Part 2)
Database schema analysis and metadata.

**Introspection Methods** (4 total):
1. `get_schema()` - Complete schema definition
2. `list_tables()` - Table enumeration
3. `get_table_info()` - Table metadata
4. `get_indexes()` - Index information

**Introspection DTOs** (4 total):
- `TableSchema` - Table structure
- `ColumnInfo` - Column metadata
- `IndexInfo` - Index details
- `SchemaInfo` - Complete schema

### Advanced CLI Commands (Part 3)
High-level operations for schema and profiling.

**CLI Commands** (4 total):
1. `profile` - Profile a query
2. `schema` - Show schema
3. `tables` - List tables
4. `analyze` - Analyze performance

---

## 🏗️ ARCHITECTURE

### Query Profiling Layer

```
CatalogQueries (Existing)
    ↓
CatalogProfiler (New)
    ├── profile_query()
    ├── estimate_query_cost()
    └── optimize_suggestions()
    ↓
Profile DTOs
    ├── QueryProfile
    └── Performance metrics
    ↓
DuckDB EXPLAIN ANALYZE
```

### Schema Introspection Layer

```
CatalogQueries (Existing)
    ↓
CatalogSchema (New)
    ├── get_schema()
    ├── list_tables()
    ├── get_table_info()
    └── get_indexes()
    ↓
Schema DTOs
    ├── SchemaInfo
    ├── TableSchema
    ├── ColumnInfo
    └── IndexInfo
    ↓
DuckDB Information Schema
```

---

## 📊 IMPLEMENTATION SEQUENCE

### Phase 2.3a: Query Profiling DTOs (30 min, 50 LOC)

**File**: `src/DocsToKG/OntologyDownload/catalog/profiling_dto.py`

Tasks:
1. Create `QueryProfile` dataclass
2. Add performance metrics fields
3. Add convenience properties
4. Add NAVMAP header

### Phase 2.3b: Query Profiler Implementation (1 hour, 150 LOC)

**File**: `src/DocsToKG/OntologyDownload/catalog/profiler.py`

Tasks:
1. Create `CatalogProfiler` class
2. Implement `profile_query()` with EXPLAIN ANALYZE
3. Implement `estimate_query_cost()`
4. Implement `optimize_suggestions()`
5. Add error handling

### Phase 2.3c: Schema Introspection DTOs (30 min, 60 LOC)

**File**: `src/DocsToKG/OntologyDownload/catalog/schema_dto.py`

Tasks:
1. Create 4 schema DTOs
2. Add metadata fields
3. Add convenience properties
4. Add NAVMAP header

### Phase 2.3d: Schema Introspection Implementation (1 hour, 100 LOC)

**File**: `src/DocsToKG/OntologyDownload/catalog/schema_inspector.py`

Tasks:
1. Create `CatalogSchema` class
2. Implement `get_schema()`
3. Implement `list_tables()`
4. Implement `get_table_info()`
5. Implement `get_indexes()`

### Phase 2.3e: Comprehensive Testing (1 hour, 200+ LOC)

**File**: `tests/ontology_download/test_advanced_features.py`

Tasks:
1. Test profiling methods (5-7 tests)
2. Test schema introspection (5-7 tests)
3. Test DTOs (3 tests)
4. Test error handling (2-3 tests)

### Phase 2.3f: CLI Integration (30 min)

**File**: `src/DocsToKG/OntologyDownload/cli/db_cmd.py` (updated)

Tasks:
1. Add `profile` command
2. Add `schema` command
3. Add `tables` command
4. Add `analyze` command

---

## 📈 TESTING STRATEGY

### Profiling Tests (5-7 tests)

```
TestQueryProfiler (5 tests)
  - profile_query() success
  - profile_query() with complex query
  - estimate_query_cost()
  - optimize_suggestions()
  - Error handling

TestProfileDTOs (2 tests)
  - QueryProfile creation
  - Metrics calculation
```

### Schema Introspection Tests (5-7 tests)

```
TestSchemaIntrospection (5 tests)
  - get_schema()
  - list_tables()
  - get_table_info()
  - get_indexes()
  - Error handling

TestSchemaDTOs (2 tests)
  - DTO creation
  - Metadata parsing
```

### Error Handling Tests (2-3 tests)
- Invalid queries
- Missing tables
- Database errors

---

## ✅ ACCEPTANCE CRITERIA

- [ ] Query profiling fully implemented
- [ ] Schema introspection fully implemented
- [ ] 4 CLI commands added
- [ ] 4 profiling/schema DTOs defined
- [ ] 15-20 tests passing (100%)
- [ ] 100% type hints
- [ ] Zero linting errors
- [ ] Performance metrics collected
- [ ] All queries optimized
- [ ] Complete documentation
- [ ] NAVMAP headers present

---

## 📚 SUCCESS METRICS

### Code Quality
- ✅ 100% type hints
- ✅ Zero linting errors
- ✅ 15-20 tests (100% passing)
- ✅ Clean architecture
- ✅ Reusable DTOs

### Functionality
- ✅ Full profiling support
- ✅ Complete schema introspection
- ✅ 4 CLI commands
- ✅ Performance recommendations
- ✅ Complete metadata

### Documentation
- ✅ NAVMAP headers
- ✅ Docstrings complete
- ✅ Type hints clear
- ✅ Usage examples

### Integration
- ✅ CLI commands work
- ✅ Well-integrated
- ✅ Error handling
- ✅ All tests passing

---

## 🎯 DELIVERY PLAN

1. **Start**: Profiling DTOs (30 min)
2. **Implement**: Profiler (1 hour)
3. **DTOs**: Schema DTOs (30 min)
4. **Implement**: Schema Inspector (1 hour)
5. **Test**: Comprehensive suite (1 hour)
6. **CLI**: Integrate commands (30 min)
7. **Verify**: Full validation (30 min)

**Total: ~5.5 hours** (3-4 hour dev + 1.5-2 hour testing/integration)

---

## 🎯 CRITICAL SUCCESS FACTORS

1. **Query Profiling**: Must provide actionable optimization suggestions
2. **Schema Introspection**: Must accurately reflect database structure
3. **CLI Integration**: Must be intuitive and user-friendly
4. **Performance**: All operations <100ms
5. **Error Handling**: Graceful handling of all edge cases

---

**Status**: Ready to implement Phase 2.3
**Quality Target**: 100/100
**Test Target**: 100% passing
**Timeline**: 3-4 hours estimated (1 session)

Ready to begin! 🚀
