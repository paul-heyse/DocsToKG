# 📋 TASK 1.2: CLI COMMANDS - IMPLEMENTATION PLAN

**Date**: October 21, 2025  
**Scope**: Implement 9 CLI commands for DuckDB catalog operations  
**Estimated Time**: 4-6 hours  
**Target LOC**: 800-1,200 production + 200-300 tests  

---

## COMMANDS TO IMPLEMENT

### Phase 1: Foundation & Read Operations (90 min)

1. **db migrate** - Apply pending DuckDB migrations
   - Call `apply_pending_migrations()`
   - Show applied migrations
   - Support `--dry-run` flag

2. **db latest** - Get/set latest version pointer
   - Get mode: `ontofetch db latest get`
   - Set mode: `ontofetch db latest set <version>`
   - Show timestamp and updater

3. **db versions** - List all versions
   - Support `--service` filter
   - Support `--limit` parameter
   - Support `--format json|table`

4. **db files** - List files in a version
   - `ontofetch db files --version <v>`
   - Filter by format (--format)
   - Show size and mtime

### Phase 2: Statistics & Analysis (60 min)

5. **db stats** - Get version statistics
   - File count, total size
   - Format distribution
   - Validation pass/fail counts
   - Display as JSON or table

6. **db delta** - Compare two versions
   - `ontofetch db delta A B`
   - Show new files, deleted files, renamed files
   - Format changes
   - Validation differences

### Phase 3: Maintenance Commands (90 min)

7. **db doctor** - Reconcile DB↔FS inconsistencies
   - Find orphaned files
   - Find missing DB records
   - Offer fix options
   - Support `--fix` for auto-fix
   - Support `--dry-run`

8. **db prune** - Identify/remove orphans
   - Walk filesystem
   - Compare with DB
   - Show what would be deleted
   - Support `--apply` to delete
   - Support `--dry-run`

9. **db backup** - Create timestamped backups
   - Copy DuckDB file
   - Create backup directory
   - Show backup location

---

## IMPLEMENTATION ARCHITECTURE

### File Structure
```
src/DocsToKG/OntologyDownload/cli/
  ├─ __init__.py (already exists - re-export db_cmd)
  └─ db_cmd.py (scaffold exists - complete implementation)
```

### Helper Modules
```
Core APIs:
  - catalog.repo.Repo → Queries
  - catalog.migrations → Migration runner
  - catalog.doctor → Reconciliation logic
  - catalog.gc (prune) → Garbage collection

Output:
  - formatters.py → Table/JSON formatting
  - logging_utils.py → Structured logging

Configuration:
  - settings.py → ResolvedConfig
```

### Typer Integration
Already using Typer in the project. Pattern:

```python
from typer import Typer

app = Typer(help="DuckDB catalog utilities")

@app.command()
def migrate(...):
    pass

@app.command()
def latest(...):
    pass
```

---

## QUALITY REQUIREMENTS

- ✅ **Type Hints**: 100% coverage
- ✅ **Documentation**: Docstrings + help text
- ✅ **Error Handling**: Graceful, informative errors
- ✅ **Testing**: 200-300 LOC tests
- ✅ **Output**: JSON + table formats
- ✅ **Logging**: Structured with adapter
- ✅ **Dry-run**: All write commands support
- ✅ **Exit codes**: Proper for scripting

---

## IMPLEMENTATION SEQUENCE

1. **Setup** (15 min)
   - Review db_cmd.py scaffold
   - Ensure imports correct
   - Create test file

2. **Phase 1** (90 min)
   - db migrate
   - db latest
   - db versions
   - db files

3. **Phase 2** (60 min)
   - db stats
   - db delta

4. **Phase 3** (90 min)
   - db doctor
   - db prune
   - db backup

5. **Testing & Polish** (30 min)
   - Comprehensive tests
   - Help text verification
   - Edge case handling

---

## SUCCESS CRITERIA

- [ ] All 9 commands implemented
- [ ] 200+ LOC tests (>80% passing)
- [ ] Zero linting errors
- [ ] 100% type hints
- [ ] All --format flags work
- [ ] All --dry-run flags work
- [ ] Comprehensive help text
- [ ] Backward compatible
- [ ] Production quality

