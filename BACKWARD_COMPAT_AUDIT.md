# 🔍 BACKWARD COMPATIBILITY AUDIT

**Date**: October 21, 2025  
**Status**: Identifying remaining backward compatibility code for removal

## FINDINGS

### CRITICAL (Must Remove)

1. **SQLite Alias Creation** (`telemetry.py:1666, 1974-1990`)
   - `_legacy_alias_path` attribute
   - `_ensure_legacy_alias()` method
   - Creates `.sqlite` symlink/copy for `.sqlite3` files
   - **Reason**: Resume compatibility for old manifest files
   - **Action**: REMOVE - new paths only

2. **Legacy Wire Format Conversion** (`core.py:413-431, 475-499`)
   - `Classification.from_wire()` with `legacy_map`
   - `ReasonCode.from_wire()` with similar pattern
   - Maps old classification strings to new enums
   - **Reason**: Support old manifest data formats
   - **Action**: REMOVE - convert at load time or reject old data

### ACCEPTABLE (Keep for Now)

3. **Optional Dependency Fallbacks** (various files)
   - `try: import X except ImportError` patterns
   - Examples: h2, redis, duckdb, parquet
   - **Reason**: These are optional features, not backward compatibility
   - **Action**: KEEP - these enable/disable features, not maintain legacy API

4. **Documentation References**
   - Comments mentioning "backward compatibility" or "legacy"
   - **Reason**: Documentation, not active code
   - **Action**: KEEP - clear documentation is good

## REMOVAL PLAN

### Phase 1: SQLite Alias Code
- Remove `_legacy_alias_path` attribute initialization
- Remove `_ensure_legacy_alias()` method
- Remove calls to `_ensure_legacy_alias()` in close() method
- Impact: 30 LOC removed
- Risk: LOW (only affects resume of old manifests)

### Phase 2: Wire Format Conversions
- Simplify `Classification.from_wire()` - remove legacy_map, require modern format
- Simplify `ReasonCode.from_wire()` - same approach
- Impact: 40 LOC removed
- Risk: LOW (only affects loading old manifest data)

Total Removal: ~70 LOC of actual backward compatibility code

