# Implementation Tasks

## Overview

This document provides detailed implementation tasks for enhancing the ontology downloader with explicit instructions for AI programming agents. Each task includes:

- **Objective**: What needs to be accomplished
- **Files to Modify**: Specific file paths
- **Implementation Details**: Step-by-step instructions with code patterns
- **Testing Requirements**: Specific test scenarios to implement
- **Acceptance Criteria**: How to verify completion

## 1. Foundation: Code Cleanup and Consolidation

### 1.1 Create Optional Dependency Module

#### 1.1.1 Create `src/DocsToKG/OntologyDownload/optdeps.py`

**Objective**: Create centralized module for managing optional dependencies

**Implementation**:

```python
# File: src/DocsToKG/OntologyDownload/optdeps.py
"""Centralized optional dependency management with fallback stubs."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import os

# Module-level caches
_pystow: Any = None
_rdflib: Any = None
_pronto: Any = None
_owlready2: Any = None
```

**Acceptance Criteria**:

- [x] File created at correct path
- [x] Module imports without errors
- [x] Empty module passes linting

#### 1.1.2 Implement `get_pystow()` with fallback stub

**Objective**: Provide pystow or fallback implementation

**Implementation**:

```python
class _PystowFallback:
    """Minimal pystow replacement for when dependency is absent."""

    def __init__(self) -> None:
        self._root = Path(os.environ.get("PYSTOW_HOME", Path.home() / ".data"))

    def join(self, *segments: str) -> Path:
        """Join path segments relative to fallback root.

        Args:
            *segments: Path components to append

        Returns:
            Path object under fallback directory
        """
        return self._root.joinpath(*segments)

def get_pystow() -> Any:
    """Get pystow module or fallback stub.

    Returns:
        Real pystow module if installed, otherwise _PystowFallback instance

    Examples:
        >>> pystow = get_pystow()
        >>> cache_dir = pystow.join("ontology-fetcher", "cache")
    """
    global _pystow
    if _pystow is None:
        try:
            import pystow
            _pystow = pystow
        except ModuleNotFoundError:
            _pystow = _PystowFallback()
    return _pystow
```

**Testing Requirements**:

```python
# tests/ontology_download/test_optdeps.py
def test_get_pystow_with_real_module(monkeypatch):
    """Test get_pystow returns real module when installed."""
    # Setup: ensure pystow is available
    # Test: call get_pystow()
    # Assert: returns actual pystow module with join method

def test_get_pystow_fallback_when_missing(monkeypatch):
    """Test get_pystow returns fallback when module missing."""
    # Setup: monkeypatch import to raise ModuleNotFoundError
    # Test: call get_pystow()
    # Assert: returns _PystowFallback instance
    # Assert: fallback.join() returns Path object

def test_pystow_fallback_respects_env_var(monkeypatch):
    """Test fallback uses PYSTOW_HOME environment variable."""
    # Setup: set PYSTOW_HOME=/custom/path
    # Test: create _PystowFallback(), call join("cache")
    # Assert: returns /custom/path/cache
```

**Acceptance Criteria**:

- [x] `get_pystow()` function implemented
- [x] Fallback class provides `join()` method
- [x] Caching works (same object returned on repeated calls)
- [x] All tests pass

#### 1.1.3 Implement `get_rdflib()` with stub Graph class

**Objective**: Provide rdflib or minimal stub for testing

**Implementation**:

```python
class _StubGraph:
    """Minimal Graph replacement for when rdflib is absent."""

    def __init__(self) -> None:
        self._source: Optional[Path] = None
        self._triples: List[tuple] = []

    def parse(self, source: str, format: Optional[str] = None) -> None:
        """Record source path for stub testing.

        Args:
            source: Path to RDF file
            format: RDF format (ignored in stub)
        """
        self._source = Path(source)
        # Stub: just record that parse was called

    def __len__(self) -> int:
        """Return placeholder triple count."""
        return 1  # Stub always reports 1 triple

    def serialize(self, destination: Any, format: str = "turtle") -> None:
        """Write stub serialization.

        Args:
            destination: Output path or file object
            format: Serialization format
        """
        dest_path = Path(destination) if isinstance(destination, (str, Path)) else destination
        if isinstance(dest_path, Path):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if self._source and self._source.exists():
                dest_path.write_bytes(self._source.read_bytes())
            else:
                dest_path.write_text("# Stub TTL output\n")

class _StubRDFLib:
    """Stub rdflib module."""
    Graph = _StubGraph

def get_rdflib() -> Any:
    """Get rdflib module or stub.

    Returns:
        Real rdflib if installed, otherwise _StubRDFLib

    Examples:
        >>> rdflib = get_rdflib()
        >>> g = rdflib.Graph()
        >>> g.parse("ontology.owl")
    """
    global _rdflib
    if _rdflib is None:
        try:
            import rdflib
            _rdflib = rdflib
        except ModuleNotFoundError:
            _rdflib = _StubRDFLib()
    return _rdflib
```

**Testing Requirements**:

```python
def test_get_rdflib_with_real_module():
    """Test returns real rdflib when installed."""
    # Test: import rdflib via get_rdflib()
    # Assert: has Graph class with parse, serialize methods

def test_rdflib_stub_parse_records_source():
    """Test stub records source path on parse."""
    # Setup: create stub Graph
    # Test: call parse("test.owl")
    # Assert: _source attribute set to Path("test.owl")

def test_rdflib_stub_serialize_copies_source():
    """Test stub serialize copies source to destination."""
    # Setup: create temp source file, stub Graph, parse
    # Test: serialize to destination
    # Assert: destination exists with source content
```

**Acceptance Criteria**:

- [x] `get_rdflib()` function implemented
- [x] Stub Graph has parse, serialize, **len** methods
- [x] Stub behavior matches real rdflib interface for testing
- [x] Tests pass

#### 1.1.4 Implement `get_pronto()` with stub Ontology class

**Objective**: Provide pronto or stub for OBO parsing

**Implementation**:

```python
class _StubOntology:
    """Minimal Ontology stub for when pronto is absent."""

    def __init__(self, handle: str) -> None:
        """Initialize stub with ontology path.

        Args:
            handle: Path to ontology file
        """
        self._path = Path(handle)

    def terms(self):
        """Yield placeholder terms.

        Yields:
            Placeholder term identifiers
        """
        return ["TERM:0000001", "TERM:0000002"]

    def dump(self, file: str, format: str = "obo") -> None:
        """Write minimal stub output.

        Args:
            file: Output file path
            format: Output format (obo, obojson)
        """
        output_path = Path(file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if format == "obojson":
            output_path.write_text('{"graphs": []}')
        else:
            output_path.write_text("format-version: 1.2\n")

class _StubPronto:
    """Stub pronto module."""
    Ontology = _StubOntology

def get_pronto() -> Any:
    """Get pronto module or stub.

    Returns:
        Real pronto if installed, otherwise _StubPronto
    """
    global _pronto
    if _pronto is None:
        try:
            import pronto
            _pronto = pronto
        except ModuleNotFoundError:
            _pronto = _StubPronto()
    return _pronto
```

**Testing Requirements**:

```python
def test_pronto_stub_terms_returns_list():
    """Test stub Ontology.terms() returns iterable."""
    # Setup: create stub Ontology
    # Test: call terms()
    # Assert: returns non-empty iterable

def test_pronto_stub_dump_creates_file():
    """Test stub dump writes output file."""
    # Setup: create stub Ontology, temp output dir
    # Test: call dump(path, format="obojson")
    # Assert: output file exists with JSON content
```

**Acceptance Criteria**:

- [x] `get_pronto()` function implemented
- [x] Stub Ontology has terms(), dump() methods
- [x] Tests pass

#### 1.1.5 Implement `get_owlready2()` with stub wrapper

**Objective**: Provide owlready2 or stub for OWL reasoning

**Implementation**:

```python
class _StubLoadedOntology:
    """Stub for loaded ontology from owlready2."""

    def classes(self):
        """Return placeholder class list.

        Returns:
            List of placeholder class identifiers
        """
        return ["Class1", "Class2", "Class3"]

class _StubOntologyWrapper:
    """Stub ontology wrapper."""

    def __init__(self, iri: str) -> None:
        """Initialize with IRI.

        Args:
            iri: Ontology IRI or file:// URI
        """
        self._iri = iri

    def load(self) -> _StubLoadedOntology:
        """Load ontology (stub returns placeholder).

        Returns:
            Stub loaded ontology
        """
        return _StubLoadedOntology()

class _StubOwlready2:
    """Stub owlready2 module."""

    @staticmethod
    def get_ontology(iri: str) -> _StubOntologyWrapper:
        """Get ontology wrapper (stub).

        Args:
            iri: Ontology IRI

        Returns:
            Stub ontology wrapper
        """
        return _StubOntologyWrapper(iri)

def get_owlready2() -> Any:
    """Get owlready2 module or stub.

    Returns:
        Real owlready2 if installed, otherwise _StubOwlready2
    """
    global _owlready2
    if _owlready2 is None:
        try:
            import owlready2
            _owlready2 = owlready2
        except ModuleNotFoundError:
            _owlready2 = _StubOwlready2()
    return _owlready2
```

**Testing Requirements**:

```python
def test_owlready2_stub_get_ontology():
    """Test stub get_ontology returns wrapper."""
    # Setup: get stub owlready2
    # Test: call get_ontology("file:///path/to/ont.owl")
    # Assert: returns wrapper with load() method

def test_owlready2_stub_load_returns_ontology():
    """Test wrapper.load() returns ontology with classes."""
    # Setup: create wrapper
    # Test: call load()
    # Assert: result has classes() method returning list
```

**Acceptance Criteria**:

- [x] `get_owlready2()` function implemented
- [x] Stub provides get_ontology() → wrapper → load() → classes() chain
- [x] Tests pass

#### 1.1.6 Add comprehensive docstrings and type hints

**Objective**: Document all functions with Google-style docstrings

**Implementation**:

- Add module-level docstring explaining purpose
- Add docstrings to all functions with Args, Returns, Examples sections
- Add type hints using `from __future__ import annotations`
- Add `__all__` export list

**Example**:

```python
"""Centralized optional dependency management with fallback stubs.

This module provides getter functions for optional dependencies used by the
ontology downloader. When a dependency is not installed, lightweight stub
implementations are returned that provide minimal functionality for testing.

Examples:
    >>> from DocsToKG.OntologyDownload.optdeps import get_pystow, get_rdflib
    >>> pystow = get_pystow()
    >>> cache_dir = pystow.join("ontology-fetcher", "cache")
"""

__all__ = [
    "get_pystow",
    "get_rdflib",
    "get_pronto",
    "get_owlready2",
]
```

**Acceptance Criteria**:

- [x] All functions have Google-style docstrings
- [x] All functions have type hints
- [x] Module docstring explains purpose
- [x] `__all__` list defined

### 1.2 Migrate Code to Use optdeps Module

#### 1.2.1 Update `core.py` to import pystow from `optdeps.get_pystow()`

**Objective**: Replace local pystow stub with optdeps version

**Files to Modify**: `src/DocsToKG/OntologyDownload/core.py`

**Implementation Steps**:

1. Remove lines 23-53 (local _PystowFallback class and try/except)
2. Add import at top: `from .optdeps import get_pystow`
3. Replace `pystow = _PystowFallback()` with `pystow = get_pystow()`
4. Update all `pystow.join()` calls (should work unchanged)

**Code Changes**:

```python
# OLD (lines 23-53):
try:
    import pystow
except ModuleNotFoundError:
    class _PystowFallback:
        # ... implementation ...
    pystow = _PystowFallback()

# NEW:
from .optdeps import get_pystow
pystow = get_pystow()
```

**Testing**:

- Run existing tests for core.py
- Verify DATA_ROOT, CACHE_DIR, etc. still work
- Test with and without pystow installed

**Acceptance Criteria**:

- [x] Local stub removed
- [x] optdeps import added
- [x] All tests pass
- [x] No functionality change

#### 1.2.2 Update `resolvers.py` to import pystow from `optdeps.get_pystow()`

**Objective**: Replace resolver pystow stub with optdeps

**Files to Modify**: `src/DocsToKG/OntologyDownload/resolvers.py`

**Implementation Steps**:

1. Remove lines 23-53 (local _PystowFallback class)
2. Add import: `from .optdeps import get_pystow`
3. Replace `pystow = _PystowFallback()` with `pystow = get_pystow()`

**Acceptance Criteria**:

- [x] Local stub removed
- [x] optdeps import added
- [x] Resolver tests pass

#### 1.2.3 Update `validators.py` to import rdflib, pronto, owlready2 from optdeps

**Objective**: Replace all validator stubs with optdeps versions

**Files to Modify**: `src/DocsToKG/OntologyDownload/validators.py`

**Implementation Steps**:

1. Remove lines 27-162 (all local stub classes)
2. Add imports at top:

   ```python
   from .optdeps import get_rdflib, get_pronto, get_owlready2

   rdflib = get_rdflib()
   pronto = get_pronto()
   owlready2 = get_owlready2()
   ```

3. All usage of rdflib, pronto, owlready2 should work unchanged

**Acceptance Criteria**:

- [x] All local stubs removed (135 lines deleted)
- [x] optdeps imports added
- [x] All validator tests pass

#### 1.2.4 Remove all local stub implementations

**Objective**: Verify no duplicate stubs remain

**Implementation**:

- Search for `class _Stub` in all files
- Search for `class _Fallback` in all files
- Ensure only optdeps.py contains stubs

**Acceptance Criteria**:

- [x] `grep -r "class _Stub" src/DocsToKG/OntologyDownload/` returns only optdeps.py
- [x] No duplicate stub implementations exist

#### 1.2.5 Verify tests pass with centralized stubs

**Objective**: Ensure migration didn't break anything

**Implementation**:

```bash
# Run full test suite for ontology download
pytest tests/ontology_download/ -v --tb=short

# Run with coverage
pytest tests/ontology_download/ --cov=src/DocsToKG/OntologyDownload --cov-report=term-missing

# Test with pystow uninstalled
pip uninstall -y pystow
pytest tests/ontology_download/test_core.py -v
pip install pystow
```

**Acceptance Criteria**:

- [x] All tests pass with real dependencies
- [x] All tests pass with stubs (dependencies uninstalled)
- [x] No test breakage from migration

### 1.3 Extract CLI Formatting Utilities

#### 1.3.1 Create `src/DocsToKG/OntologyDownload/cli_utils.py`

**Objective**: Create module for reusable CLI formatting

**Implementation**:

```python
"""CLI formatting utilities for ontology downloader.

Provides reusable formatting functions for tables, summaries, and structured
output across all CLI subcommands.
"""

from __future__ import annotations
from typing import Sequence, Dict, Any

__all__ = [
    "format_table",
    "format_validation_summary",
]
```

**Acceptance Criteria**:

- [x] File created
- [x] Module imports successfully
- [x] Has `__all__` export list

#### 1.3.2-1.3.4 Move formatting functions from cli.py to cli_utils.py

**Objective**: Extract `_format_table`, `_format_row`, `_format_validation_summary`

**Implementation**:

```python
# cli_utils.py
def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Format data as ASCII table with headers.

    Args:
        headers: Column header strings
        rows: List of row data (each row is sequence of cell strings)

    Returns:
        Formatted table string with aligned columns and separator lines

    Examples:
        >>> print(format_table(["Name", "Status"], [["hp", "success"], ["efo", "cached"]]))
        Name | Status
        -----+--------
        hp   | success
        efo  | cached
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def format_validation_summary(results: Dict[str, Dict[str, Any]]) -> str:
    """Format validation results as table.

    Args:
        results: Mapping of validator name to result dict with 'ok' and 'details'

    Returns:
        Formatted table showing validator, status, and details

    Examples:
        >>> results = {"rdflib": {"ok": True, "details": {"triples": 100}}}
        >>> print(format_validation_summary(results))
        validator | status | details
        ----------+--------+--------------
        rdflib    | ok     | triples=100
    """
    rows = []
    for name, payload in results.items():
        status = "ok" if payload.get("ok") else "error"
        details = payload.get("details", {})
        message = ""
        if isinstance(details, dict):
            if "error" in details:
                message = str(details["error"])
            elif details:
                message = ", ".join(f"{k}={v}" for k, v in details.items())
        rows.append((name, status, message))
    return format_table(("validator", "status", "details"), rows)
```

**Acceptance Criteria**:

- [x] Functions moved from cli.py lines 143-170
- [x] Functions renamed (remove `_` prefix, now public API)
- [x] Docstrings added with examples
- [x] Type hints added

#### 1.3.5 Update cli.py imports to use cli_utils

**Objective**: Import and use new cli_utils functions

**Files to Modify**: `src/DocsToKG/OntologyDownload/cli.py`

**Implementation**:

```python
# Add to imports at top
from .cli_utils import format_table, format_validation_summary

# Replace usage:
# OLD: table = _format_table(...)
# NEW: table = format_table(...)

# OLD: print(_format_validation_summary(summary))
# NEW: print(format_validation_summary(summary))
```

**Specific Changes**:

- Line 313: Change `_format_table` to `format_table`
- Line 346: Change `_format_validation_summary` to `format_validation_summary`
- Remove lines 143-170 (old function definitions)

**Acceptance Criteria**:

- [x] cli.py imports from cli_utils
- [x] All calls updated to new names
- [x] Old function definitions removed
- [x] CLI tests pass

#### 1.3.6 Add unit tests for cli_utils formatting functions

**Objective**: Test formatting functions in isolation

**Files to Create**: `tests/ontology_download/test_cli_utils.py`

**Implementation**:

```python
"""Tests for CLI formatting utilities."""

import pytest
from DocsToKG.OntologyDownload.cli_utils import format_table, format_validation_summary


def test_format_table_basic():
    """Test basic table formatting with headers and rows."""
    headers = ["ID", "Status"]
    rows = [["hp", "success"], ["efo", "cached"]]
    result = format_table(headers, rows)

    assert "ID" in result
    assert "Status" in result
    assert "hp" in result
    assert "success" in result
    assert "|" in result  # column separator
    assert "-" in result  # header separator


def test_format_table_column_alignment():
    """Test columns are aligned by widest cell."""
    headers = ["Short", "Longer Header"]
    rows = [["a", "b"], ["xyz", "abcdefgh"]]
    result = format_table(headers, rows)

    lines = result.split("\n")
    # All lines should have same width (within reason for separators)
    assert len(lines) >= 4  # header + separator + 2 rows
    # Check alignment (headers/rows should align)
    assert "Short" in lines[0]
    assert "xyz" in lines[2]


def test_format_table_empty_rows():
    """Test table with headers but no data rows."""
    headers = ["Name", "Value"]
    rows = []
    result = format_table(headers, rows)

    assert "Name" in result
    assert "Value" in result
    # Should still show headers and separator
    lines = result.split("\n")
    assert len(lines) == 2  # header + separator only


def test_format_validation_summary_success():
    """Test validation summary with successful validator."""
    results = {
        "rdflib": {
            "ok": True,
            "details": {"triples": 1234, "elapsed": 2.5}
        }
    }
    result = format_validation_summary(results)

    assert "rdflib" in result
    assert "ok" in result
    assert "triples=1234" in result


def test_format_validation_summary_error():
    """Test validation summary with failed validator."""
    results = {
        "pronto": {
            "ok": False,
            "details": {"error": "timeout after 60s"}
        }
    }
    result = format_validation_summary(results)

    assert "pronto" in result
    assert "error" in result
    assert "timeout" in result


def test_format_validation_summary_multiple():
    """Test summary with multiple validators."""
    results = {
        "rdflib": {"ok": True, "details": {}},
        "pronto": {"ok": False, "details": {"error": "parse error"}},
        "owlready2": {"ok": True, "details": {"entities": 50}}
    }
    result = format_validation_summary(results)

    assert "rdflib" in result
    assert "pronto" in result
    assert "owlready2" in result
    # Should have 3 data rows + header + separator
    lines = result.split("\n")
    assert len(lines) == 5
```

**Acceptance Criteria**:

- [x] Test file created with 6+ test cases
- [x] Tests cover basic formatting, alignment, empty cases, errors
- [x] All tests pass
- [x] Coverage >95% for cli_utils.py

### 1.4 Migrate Configuration to Pydantic v2

#### 1.4.1 Add pydantic>=2.0.0 to requirements.txt

**Objective**: Add Pydantic v2 as required dependency

**Files to Modify**: `requirements.txt` (or `requirements.in` if using pip-compile)

**Implementation**:

```
# Add to requirements
pydantic>=2.0.0,<3.0.0
```

**Testing**:

```bash
pip install -r requirements.txt
python -c "import pydantic; print(pydantic.__version__)"
# Should print 2.x.x
```

**Acceptance Criteria**:

- [x] Pydantic 2.x added to requirements
- [x] Installs successfully
- [x] Version constraint prevents Pydantic 1.x

#### 1.4.2 Convert `LoggingConfiguration` dataclass to Pydantic `BaseModel`

**Objective**: Replace dataclass with Pydantic model

**Files to Modify**: `src/DocsToKG/OntologyDownload/config.py`

**Implementation**:

```python
# Add imports at top
from pydantic import BaseModel, Field, field_validator

# Replace dataclass (lines 183-200)
class LoggingConfiguration(BaseModel):
    """Structured logging options for ontology download operations.

    Attributes:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR)
        max_log_size_mb: Maximum individual log file size before rotation
        retention_days: Number of days to keep logs prior to deletion
    """

    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    max_log_size_mb: int = Field(
        default=100,
        gt=0,
        description="Maximum log file size in MB before rotation"
    )
    retention_days: int = Field(
        default=30,
        ge=1,
        description="Number of days to retain logs"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level is recognized."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return upper

    model_config = {
        "frozen": False,  # Allow mutation for env var overrides
        "validate_assignment": True,
    }
```

**Testing Requirements**:

```python
# tests/ontology_download/test_config.py
def test_logging_config_defaults():
    """Test LoggingConfiguration uses correct defaults."""
    config = LoggingConfiguration()
    assert config.level == "INFO"
    assert config.max_log_size_mb == 100
    assert config.retention_days == 30


def test_logging_config_validates_level():
    """Test invalid logging level raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        LoggingConfiguration(level="INVALID")
    assert "level must be one of" in str(exc_info.value)


def test_logging_config_case_insensitive_level():
    """Test logging level accepts lowercase and normalizes."""
    config = LoggingConfiguration(level="debug")
    assert config.level == "DEBUG"


def test_logging_config_validates_positive_values():
    """Test max_log_size_mb must be positive."""
    with pytest.raises(ValidationError):
        LoggingConfiguration(max_log_size_mb=0)

    with pytest.raises(ValidationError):
        LoggingConfiguration(max_log_size_mb=-10)
```

**Acceptance Criteria**:

- [x] Dataclass decorator removed
- [x] Inherits from BaseModel
- [x] Field() definitions with constraints
- [x] field_validator for level
- [x] Tests updated and passing

#### 1.4.3 Convert `ValidationConfig` dataclass to Pydantic `BaseModel`

**Objective**: Replace ValidationConfig dataclass

**Implementation**:

```python
class ValidationConfig(BaseModel):
    """Validation limits governing parser execution.

    Attributes:
        parser_timeout_sec: Maximum runtime per validator
        max_memory_mb: Memory ceiling for validation processes
        skip_reasoning_if_size_mb: Threshold to disable reasoning
    """

    parser_timeout_sec: int = Field(
        default=60,
        gt=0,
        le=3600,
        description="Parser timeout in seconds (max 1 hour)"
    )
    max_memory_mb: int = Field(
        default=2048,
        gt=0,
        description="Maximum memory in MB for validators"
    )
    skip_reasoning_if_size_mb: int = Field(
        default=500,
        gt=0,
        description="File size threshold to skip reasoning"
    )

    model_config = {
        "frozen": False,
        "validate_assignment": True,
    }
```

**Testing Requirements**:

```python
def test_validation_config_timeout_bounds():
    """Test parser_timeout_sec has reasonable bounds."""
    # Should reject 0 or negative
    with pytest.raises(ValidationError):
        ValidationConfig(parser_timeout_sec=0)

    # Should reject > 1 hour
    with pytest.raises(ValidationError):
        ValidationConfig(parser_timeout_sec=4000)

    # Should accept reasonable values
    config = ValidationConfig(parser_timeout_sec=120)
    assert config.parser_timeout_sec == 120
```

**Acceptance Criteria**:

- [x] Converted to BaseModel
- [x] Constraints on timeout (1 second to 1 hour)
- [x] Tests updated

#### 1.4.4 Convert `DownloadConfiguration` dataclass to Pydantic `BaseModel`

**Objective**: Replace DownloadConfiguration with more validation

**Implementation**:

```python
class DownloadConfiguration(BaseModel):
    """HTTP download and retry settings.

    Attributes:
        max_retries: Number of retries for transient failures
        timeout_sec: Timeout for connection and read
        download_timeout_sec: Upper bound for whole file download
        backoff_factor: Backoff multiplier between retries
        per_host_rate_limit: Token bucket rate limit definition
        max_download_size_gb: Maximum allowed archive size
        concurrent_downloads: Maximum concurrent workers
        validate_media_type: Enable HEAD request media type validation
        rate_limits: Per-service rate limit overrides
        allowed_hosts: Optional host allowlist for URL validation
    """

    max_retries: int = Field(default=5, ge=0, le=20)
    timeout_sec: int = Field(default=30, gt=0, le=300)
    download_timeout_sec: int = Field(default=300, gt=0, le=3600)
    backoff_factor: float = Field(default=0.5, ge=0.1, le=10.0)
    per_host_rate_limit: str = Field(
        default="4/second",
        pattern=r"^\d+(\.\d+)?/(second|sec|s|minute|min|m|hour|h)$"
    )
    max_download_size_gb: float = Field(default=5.0, gt=0, le=100.0)
    concurrent_downloads: int = Field(default=1, ge=1, le=10)
    validate_media_type: bool = Field(default=True)
    rate_limits: Dict[str, str] = Field(default_factory=dict)
    allowed_hosts: Optional[List[str]] = Field(default=None)

    @field_validator('rate_limits')
    @classmethod
    def validate_rate_limits(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate all rate limit values have correct format."""
        import re
        pattern = re.compile(r"^\d+(\.\d+)?/(second|sec|s|minute|min|m|hour|h)$")
        for service, limit in v.items():
            if not pattern.match(limit):
                raise ValueError(
                    f"Invalid rate limit '{limit}' for service '{service}'. "
                    f"Expected format: <number>/<unit> (e.g., '5/second', '60/min')"
                )
        return v

    def rate_limit_per_second(self) -> float:
        """Convert per_host_rate_limit to requests per second."""
        import re
        match = re.match(r"^([\d.]+)/(second|sec|s|minute|min|m|hour|h)$", self.per_host_rate_limit)
        if not match:
            raise ValueError(f"Invalid rate limit format: {self.per_host_rate_limit}")

        value = float(match.group(1))
        unit = match.group(2)

        if unit in ("second", "sec", "s"):
            return value
        elif unit in ("minute", "min", "m"):
            return value / 60.0
        elif unit in ("hour", "h"):
            return value / 3600.0
        else:
            raise ValueError(f"Unknown rate limit unit: {unit}")

    def parse_service_rate_limit(self, service: str) -> Optional[float]:
        """Parse rate limit for specific service to requests per second.

        Args:
            service: Service identifier (e.g., "obo", "ols", "bioportal")

        Returns:
            Requests per second for service, or None if not configured
        """
        limit_str = self.rate_limits.get(service)
        if not limit_str:
            return None

        import re
        match = re.match(r"^([\d.]+)/(second|sec|s|minute|min|m|hour|h)$", limit_str)
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2)

        if unit in ("second", "sec", "s"):
            return value
        elif unit in ("minute", "min", "m"):
            return value / 60.0
        elif unit in ("hour", "h"):
            return value / 3600.0
        return None

    model_config = {
        "frozen": False,
        "validate_assignment": True,
    }
```

**Testing Requirements**:

```python
def test_download_config_rate_limit_formats():
    """Test various rate limit format permutations."""
    # Test per-second
    config = DownloadConfiguration(per_host_rate_limit="5/second")
    assert config.rate_limit_per_second() == 5.0

    config = DownloadConfiguration(per_host_rate_limit="5/s")
    assert config.rate_limit_per_second() == 5.0

    # Test per-minute
    config = DownloadConfiguration(per_host_rate_limit="60/minute")
    assert config.rate_limit_per_second() == 1.0

    config = DownloadConfiguration(per_host_rate_limit="30/min")
    assert config.rate_limit_per_second() == 0.5

    # Test per-hour
    config = DownloadConfiguration(per_host_rate_limit="3600/hour")
    assert config.rate_limit_per_second() == 1.0

    config = DownloadConfiguration(per_host_rate_limit="1800/h")
    assert config.rate_limit_per_second() == 0.5

    # Test decimal values
    config = DownloadConfiguration(per_host_rate_limit="0.5/second")
    assert config.rate_limit_per_second() == 0.5


def test_download_config_service_rate_limits():
    """Test per-service rate limit configuration."""
    config = DownloadConfiguration(
        rate_limits={
            "obo": "2/second",
            "ols": "5/minute",
            "bioportal": "100/hour"
        }
    )

    assert config.parse_service_rate_limit("obo") == 2.0
    assert config.parse_service_rate_limit("ols") == pytest.approx(5.0/60.0)
    assert config.parse_service_rate_limit("bioportal") == pytest.approx(100.0/3600.0)
    assert config.parse_service_rate_limit("unknown") is None


def test_download_config_invalid_rate_limit():
    """Test invalid rate limit format raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        DownloadConfiguration(per_host_rate_limit="invalid")
    assert "pattern" in str(exc_info.value).lower()


def test_download_config_allowed_hosts():
    """Test allowed_hosts configuration."""
    config = DownloadConfiguration(
        allowed_hosts=["example.org", "purl.obolibrary.org"]
    )
    assert "example.org" in config.allowed_hosts
    assert "purl.obolibrary.org" in config.allowed_hosts
```

**Acceptance Criteria**:

- [x] Converted to BaseModel
- [x] New fields: validate_media_type, rate_limits, allowed_hosts
- [x] rate_limit_per_second() method works
- [x] parse_service_rate_limit() method added
- [x] All tests pass

#### 1.4.5-1.4.6 Convert `DefaultsConfig` and `ResolvedConfig` to Pydantic

**Objective**: Complete Pydantic migration for all config classes

**Implementation** for DefaultsConfig:

```python
class DefaultsConfig(BaseModel):
    """Collection of default settings for ontology fetch specifications."""

    accept_licenses: List[str] = Field(
        default_factory=lambda: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"]
    )
    normalize_to: List[str] = Field(
        default_factory=lambda: ["ttl"]
    )
    prefer_source: List[str] = Field(
        default_factory=lambda: ["obo", "ols", "bioportal", "direct"]
    )
    http: DownloadConfiguration = Field(
        default_factory=DownloadConfiguration
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig
    )
    logging: LoggingConfiguration = Field(
        default_factory=LoggingConfiguration
    )
    continue_on_error: bool = Field(default=True)

    model_config = {
        "frozen": False,
        "validate_assignment": True,
    }
```

**Implementation** for ResolvedConfig:

```python
class ResolvedConfig(BaseModel):
    """Container for merged configuration defaults and fetch specifications."""

    defaults: DefaultsConfig
    specs: List["FetchSpec"] = Field(default_factory=list)

    @classmethod
    def from_defaults(cls) -> "ResolvedConfig":
        """Create empty resolved configuration with library defaults."""
        return cls(defaults=DefaultsConfig(), specs=[])

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "arbitrary_types_allowed": True,  # Allow FetchSpec which may not be Pydantic
    }
```

**Acceptance Criteria**:

- [x] Both classes converted
- [x] Nested models work (http, validation, logging)
- [x] from_defaults() class method works
- [x] Tests updated

#### 1.4.7 Remove custom YAML fallback parser

**Objective**: Delete ~150 lines of custom parser code

**Files to Modify**: `src/DocsToKG/OntologyDownload/config.py`

**Implementation**:

1. Remove lines 23-141 (entire custom YAML parser implementation)
2. Remove _parse_scalar, _peek_next, _safe_load, _FallbackYAML
3. Keep try/except for yaml import but remove fallback:

```python
try:
    import yaml
except ModuleNotFoundError as exc:
    raise ImportError(
        "PyYAML is required for configuration parsing. "
        "Install it with: pip install pyyaml"
    ) from exc
```

**Acceptance Criteria**:

- [x] Lines 23-141 deleted (~120 lines removed)
- [x] yaml.safe_load is only YAML parser used
- [x] Clear error if PyYAML not installed
- [x] Tests pass

#### 1.4.8 Update `build_resolved_config()` to use Pydantic model_validate

**Objective**: Use Pydantic validation instead of manual parsing

**Implementation**:

```python
def build_resolved_config(raw_config: Mapping[str, object]) -> ResolvedConfig:
    """Construct fully-resolved configuration from raw YAML contents.

    Args:
        raw_config: Mapping from YAML parsing

    Returns:
        ResolvedConfig with validated defaults and fetch specs

    Raises:
        ValidationError: If configuration violates schema
    """
    try:
        # Parse defaults section using Pydantic
        defaults_section = raw_config.get("defaults", {})
        defaults = DefaultsConfig.model_validate(defaults_section)

        # Apply environment variable overrides
        _apply_env_overrides(defaults)

        # Parse ontologies section
        ontologies = raw_config.get("ontologies", [])
        if not isinstance(ontologies, list):
            raise ConfigError("'ontologies' must be a list")

        fetch_specs: List[FetchSpec] = []
        for index, entry in enumerate(ontologies, start=1):
            if not isinstance(entry, Mapping):
                raise ConfigError(f"Ontology entry #{index} must be a mapping")
            fetch_specs.append(merge_defaults(entry, defaults))

        return ResolvedConfig(defaults=defaults, specs=fetch_specs)

    except ValidationError as exc:
        # Convert Pydantic ValidationError to ConfigError with clear message
        errors = exc.errors()
        messages = []
        for error in errors:
            loc = " -> ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            messages.append(f"{loc}: {msg}")
        raise ConfigError(f"Configuration validation failed:\n  " + "\n  ".join(messages)) from exc
```

**Acceptance Criteria**:

- [x] Uses DefaultsConfig.model_validate()
- [x] Catches ValidationError and converts to ConfigError
- [x] Clear error messages with field locations
- [x] Tests pass

#### 1.4.9 Implement Pydantic field validators for config constraints

**Objective**: Add validators for complex constraints

**Implementation**: Already covered in 1.4.2-1.4.4 with @field_validator decorators

**Additional Validators**:

```python
# In DefaultsConfig
@field_validator('prefer_source')
@classmethod
def validate_prefer_source(cls, v: List[str]) -> List[str]:
    """Validate resolver names are recognized."""
    valid_resolvers = {"obo", "bioregistry", "ols", "bioportal", "skos", "xbrl", "lov", "ontobee", "direct"}
    for resolver in v:
        if resolver not in valid_resolvers:
            raise ValueError(f"Unknown resolver '{resolver}'. Valid: {valid_resolvers}")
    return v
```

**Acceptance Criteria**:

- [x] Validators added for critical fields
- [x] Tests cover validation failures
- [x] Error messages are actionable

#### 1.4.10 Update environment variable merging to use Pydantic settings

**Objective**: Use Pydantic Settings for env var handling

**Implementation**:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class EnvironmentOverrides(BaseSettings):
    """Environment variable overrides for ontology downloader."""

    max_retries: Optional[int] = Field(default=None, alias="ONTOFETCH_MAX_RETRIES")
    timeout_sec: Optional[int] = Field(default=None, alias="ONTOFETCH_TIMEOUT_SEC")
    download_timeout_sec: Optional[int] = Field(default=None, alias="ONTOFETCH_DOWNLOAD_TIMEOUT_SEC")
    per_host_rate_limit: Optional[str] = Field(default=None, alias="ONTOFETCH_PER_HOST_RATE_LIMIT")
    backoff_factor: Optional[float] = Field(default=None, alias="ONTOFETCH_BACKOFF_FACTOR")
    log_level: Optional[str] = Field(default=None, alias="ONTOFETCH_LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_prefix="ONTOFETCH_",
        case_sensitive=False,
        extra="ignore"
    )

def _apply_env_overrides(defaults: DefaultsConfig) -> None:
    """Apply environment variable overrides to defaults.

    Args:
        defaults: DefaultsConfig to modify in-place
    """
    env = EnvironmentOverrides()
    logger = logging.getLogger("DocsToKG.OntologyDownload")

    if env.max_retries is not None:
        defaults.http.max_retries = env.max_retries
        logger.info(f"Config overridden: max_retries={env.max_retries}", extra={"stage": "config"})

    if env.timeout_sec is not None:
        defaults.http.timeout_sec = env.timeout_sec
        logger.info(f"Config overridden: timeout_sec={env.timeout_sec}", extra={"stage": "config"})

    # ... similar for other fields
```

**Acceptance Criteria**:

- [x] EnvironmentOverrides class created
- [x] _apply_env_overrides uses Pydantic settings
- [x] Tests verify env vars override config file
- [x] Logging shows which overrides applied

#### 1.4.11 Update all config tests to work with Pydantic models

**Objective**: Fix tests broken by Pydantic migration

**Implementation**:

- Update test fixtures to create Pydantic models
- Change assertions for ValidationError instead of ConfigError for validation failures
- Add tests for Pydantic-specific features (model_validate, model_dump, model_json_schema)

**Example Test Updates**:

```python
def test_defaults_config_pydantic():
    """Test DefaultsConfig is Pydantic model."""
    from pydantic import BaseModel
    assert issubclass(DefaultsConfig, BaseModel)


def test_defaults_config_json_schema():
    """Test JSON schema generation."""
    schema = DefaultsConfig.model_json_schema()
    assert "properties" in schema
    assert "accept_licenses" in schema["properties"]
    assert schema["properties"]["accept_licenses"]["type"] == "array"


def test_download_config_validation_error():
    """Test Pydantic ValidationError for invalid values."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as exc_info:
        DownloadConfiguration(max_retries=-1)
    errors = exc_info.value.errors()
    assert any("greater than or equal to 0" in str(e) for e in errors)
```

**Acceptance Criteria**:

- [x] All config tests pass
- [x] Tests use ValidationError where appropriate
- [x] New tests for Pydantic features
- [x] Coverage maintained >85%

#### 1.4.12 Verify JSON Schema generation works for documentation

**Objective**: Generate and validate JSON Schema

**Implementation**:

```python
# Script: scripts/generate_config_schema.py
"""Generate JSON Schema for configuration documentation."""

import json
from pathlib import Path
from DocsToKG.OntologyDownload.config import ResolvedConfig

def main():
    schema = ResolvedConfig.model_json_schema()

    # Add metadata
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Ontology Downloader Configuration"
    schema["description"] = "Schema for DocsToKG ontology downloader sources.yaml"

    # Write to docs
    output_path = Path("docs/schemas/ontology-downloader-config.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2))

    print(f"Schema written to {output_path}")
    print(f"Schema has {len(schema.get('properties', {}))} root properties")

if __name__ == "__main__":
    main()
```

**Test Schema**:

```bash
# Generate schema
python scripts/generate_config_schema.py

# Validate schema is valid JSON Schema
pip install check-jsonschema
check-jsonschema --check-metaschema docs/schemas/ontology-downloader-config.json

# Validate example sources.yaml against schema
check-jsonschema --schemafile docs/schemas/ontology-downloader-config.json docs/examples/sources.yaml
```

**Acceptance Criteria**:

- [x] JSON Schema generates successfully
- [x] Schema validates as valid JSON Schema
- [x] Example sources.yaml validates against schema
- [x] Schema documented in docs/

---

## 2. Robustness: Download and Security Enhancements

### 2.1 Implement HEAD Request with Media Type Validation

#### 2.1.1 Add `_preliminary_head_check()` method to `StreamingDownloader` class

**Objective**: Perform HEAD request before full GET

**Files to Modify**: `src/DocsToKG/OntologyDownload/download.py`

**Implementation**:

```python
# Add to StreamingDownloader class around line 300

def _preliminary_head_check(
    self,
    url: str,
    session: requests.Session
) -> tuple[Optional[str], Optional[int]]:
    """Perform HEAD request to validate media type and get content length.

    Args:
        url: Target URL to check
        session: Requests session to use

    Returns:
        Tuple of (content_type, content_length) or (None, None) if HEAD fails

    Raises:
        ConfigError: If content exceeds size limit
    """
    try:
        head_response = session.head(
            url,
            headers=self.custom_headers,
            timeout=self.http_config.timeout_sec,
            allow_redirects=True
        )

        # HEAD request allowed to fail gracefully (some servers don't support it)
        if head_response.status_code >= 400:
            self.logger.debug(
                "HEAD request failed, proceeding with GET",
                extra={
                    "stage": "download",
                    "method": "HEAD",
                    "status_code": head_response.status_code,
                    "url": url
                }
            )
            return None, None

        content_type = head_response.headers.get("Content-Type")
        content_length_str = head_response.headers.get("Content-Length")
        content_length = int(content_length_str) if content_length_str else None

        # Check size early
        if content_length:
            max_bytes = self.http_config.max_download_size_gb * (1024**3)
            if content_length > max_bytes:
                self.logger.error(
                    "file exceeds size limit (HEAD check)",
                    extra={
                        "stage": "download",
                        "content_length": content_length,
                        "limit_bytes": max_bytes,
                        "url": url
                    }
                )
                raise ConfigError(
                    f"File size {content_length} bytes exceeds limit of "
                    f"{self.http_config.max_download_size_gb} GB (detected via HEAD)"
                )

        return content_type, content_length

    except requests.RequestException as exc:
        # HEAD request failures are non-fatal
        self.logger.debug(
            "HEAD request exception, proceeding with GET",
            extra={"stage": "download", "error": str(exc), "url": url}
        )
        return None, None
```

**Acceptance Criteria**:

- [x] Method added to StreamingDownloader
- [x] Returns content_type and content_length tuple
- [x] Raises ConfigError if size exceeds limit
- [x] Logs HEAD failures at debug level
- [x] Non-fatal if HEAD not supported

#### 2.1.2 Implement Content-Type header extraction and validation

**Objective**: Validate media type matches expected

**Implementation**:

```python
# Add to StreamingDownloader class

def _validate_media_type(
    self,
    actual_content_type: Optional[str],
    expected_media_type: Optional[str]
) -> None:
    """Validate actual Content-Type matches expected media type.

    Args:
        actual_content_type: Content-Type from HTTP response
        expected_media_type: Expected media type from FetchPlan

    Raises:
        ConfigError: If validation enabled and types mismatch severely
    """
    # Skip if validation disabled
    if not self.http_config.validate_media_type:
        return

    # Skip if no expected type specified
    if not expected_media_type:
        return

    # Skip if server didn't provide Content-Type
    if not actual_content_type:
        self.logger.warning(
            "server did not provide Content-Type header",
            extra={"stage": "download", "expected_media_type": expected_media_type}
        )
        return

    # Parse Content-Type to remove charset, boundary, etc.
    actual_mime = actual_content_type.split(";")[0].strip().lower()
    expected_mime = expected_media_type.strip().lower()

    if actual_mime != expected_mime:
        # Check for acceptable variations
        acceptable_variations = {
            "application/rdf+xml": {"application/xml", "text/xml"},
            "text/turtle": {"text/plain", "application/x-turtle"},
            "application/owl+xml": {"application/xml", "text/xml"},
        }

        acceptable = acceptable_variations.get(expected_mime, set())
        if actual_mime in acceptable:
            self.logger.info(
                "acceptable media type variation",
                extra={
                    "stage": "download",
                    "expected": expected_mime,
                    "actual": actual_mime
                }
            )
            return

        # Log warning for mismatch
        self.logger.warning(
            "media type mismatch",
            extra={
                "stage": "download",
                "expected_media_type": expected_mime,
                "actual_media_type": actual_mime,
                "action": "proceeding with download"
            }
        )

        # Could add strict mode that raises ConfigError here:
        # if self.http_config.strict_media_type_validation:
        #     raise ConfigError(f"Media type mismatch: expected {expected_mime}, got {actual_mime}")
```

**Acceptance Criteria**:

- [x] Validates Content-Type against expected
- [x] Handles acceptable variations (XML types)
- [x] Logs warnings for mismatches
- [x] Configurable via validate_media_type flag

#### 2.1.3 Implement early Content-Length check against size limits

**Objective**: Fail fast if file too large

**Implementation**: Already covered in 2.1.1 within `_preliminary_head_check()`

**Additional Test**:

```python
def test_head_check_rejects_oversized_file():
    """Test HEAD check raises ConfigError for file exceeding limit."""
    # Setup: mock HEAD response with Content-Length > max_download_size_gb
    # Test: call _preliminary_head_check()
    # Assert: raises ConfigError with size info
```

**Acceptance Criteria**:

- [x] Raises ConfigError if Content-Length > limit
- [x] Error message includes size and limit
- [x] Works before starting GET request

#### 2.1.4 Add config flag `http.validate_media_type` (default: true)

**Objective**: Make validation optional

**Implementation**: Already added in 1.4.4 to DownloadConfiguration:

```python
validate_media_type: bool = Field(default=True)
```

**Configuration Example**:

```yaml
# sources.yaml
defaults:
  http:
    validate_media_type: true  # Enable HEAD request validation (default)
    # Set to false to skip Content-Type checking
```

**Acceptance Criteria**:

- [x] Field added to DownloadConfiguration
- [x] Defaults to True
- [x] Can be set False in config
- [x] Documented in example sources.yaml

#### 2.1.5 Log warnings for media-type mismatches with override instructions

**Objective**: Provide actionable guidance when types don't match

**Implementation**: Already in 2.1.2, enhance logging:

```python
        self.logger.warning(
            "media type mismatch detected",
            extra={
                "stage": "download",
                "expected_media_type": expected_mime,
                "actual_media_type": actual_mime,
                "url": url,
                "action": "proceeding with download",
                "override_hint": "Set defaults.http.validate_media_type: false to disable validation"
            }
        )
```

**Acceptance Criteria**:

- [x] Warning logged with both types
- [x] Hint provided for disabling validation
- [x] URL included for troubleshooting

#### 2.1.6 Add tests for HEAD request success, failure, and mismatch scenarios

**Objective**: Comprehensive test coverage for HEAD validation

**Files to Create/Modify**: `tests/ontology_download/test_download.py`

**Implementation**:

```python
def test_head_check_success_returns_metadata():
    """Test successful HEAD request returns content type and length."""
    # Setup: mock HEAD response with Content-Type and Content-Length
    with requests_mock.Mocker() as m:
        m.head("https://example.org/ont.owl", headers={
            "Content-Type": "application/rdf+xml",
            "Content-Length": "1024"
        })

        downloader = StreamingDownloader(
            destination=Path("test.owl"),
            headers={},
            http_config=DownloadConfiguration(),
            previous_manifest=None,
            logger=logging.getLogger("test")
        )

        content_type, content_length = downloader._preliminary_head_check(
            "https://example.org/ont.owl",
            requests.Session()
        )

        assert content_type == "application/rdf+xml"
        assert content_length == 1024


def test_head_check_graceful_on_405():
    """Test HEAD check handles 405 Method Not Allowed gracefully."""
    with requests_mock.Mocker() as m:
        m.head("https://example.org/ont.owl", status_code=405)

        downloader = StreamingDownloader(
            destination=Path("test.owl"),
            headers={},
            http_config=DownloadConfiguration(),
            previous_manifest=None,
            logger=logging.getLogger("test")
        )

        content_type, content_length = downloader._preliminary_head_check(
            "https://example.org/ont.owl",
            requests.Session()
        )

        # Should return None, None and not raise
        assert content_type is None
        assert content_length is None


def test_head_check_raises_on_oversized():
    """Test HEAD check raises ConfigError for file exceeding size limit."""
    with requests_mock.Mocker() as m:
        # 6GB file when limit is 5GB
        m.head("https://example.org/huge.owl", headers={
            "Content-Length": str(6 * 1024 * 1024 * 1024)
        })

        downloader = StreamingDownloader(
            destination=Path("test.owl"),
            headers={},
            http_config=DownloadConfiguration(max_download_size_gb=5.0),
            previous_manifest=None,
            logger=logging.getLogger("test")
        )

        with pytest.raises(ConfigError) as exc_info:
            downloader._preliminary_head_check(
                "https://example.org/huge.owl",
                requests.Session()
            )

        assert "exceeds limit" in str(exc_info.value)


def test_validate_media_type_match():
    """Test media type validation passes on exact match."""
    downloader = StreamingDownloader(
        destination=Path("test.owl"),
        headers={},
        http_config=DownloadConfiguration(validate_media_type=True),
        previous_manifest=None,
        logger=logging.getLogger("test")
    )

    # Should not raise or log warning
    downloader._validate_media_type(
        "application/rdf+xml",
        "application/rdf+xml"
    )


def test_validate_media_type_acceptable_variation():
    """Test validation accepts known variations."""
    downloader = StreamingDownloader(
        destination=Path("test.owl"),
        headers={},
        http_config=DownloadConfiguration(validate_media_type=True),
        previous_manifest=None,
        logger=logging.getLogger("test")
    )

    # application/xml is acceptable for application/rdf+xml
    downloader._validate_media_type(
        "application/xml",
        "application/rdf+xml"
    )


def test_validate_media_type_mismatch_logs_warning(caplog):
    """Test media type mismatch logs warning."""
    downloader = StreamingDownloader(
        destination=Path("test.owl"),
        headers={},
        http_config=DownloadConfiguration(validate_media_type=True),
        previous_manifest=None,
        logger=logging.getLogger("test")
    )

    with caplog.at_level(logging.WARNING):
        downloader._validate_media_type(
            "text/html",  # Wrong type
            "application/rdf+xml"
        )

    assert "mismatch" in caplog.text.lower()


def test_validate_media_type_disabled():
    """Test validation skipped when disabled."""
    downloader = StreamingDownloader(
        destination=Path("test.owl"),
        headers={},
        http_config=DownloadConfiguration(validate_media_type=False),
        previous_manifest=None,
        logger=logging.getLogger("test")
    )

    # Should not log even with severe mismatch
    downloader._validate_media_type(
        "text/plain",
        "application/rdf+xml"
    )
```

**Acceptance Criteria**:

- [x] 7+ test cases covering HEAD scenarios
- [x] Tests pass
- [x] Coverage >90% for new methods

### 2.2 Implement Per-Service Rate Limiting

#### 2.2.1 Propagate service identifiers through fetch planning

**Objective**: Ensure resolvers and core download orchestration expose service metadata for rate limiting and logging.

**Implementation**:

- Extend `FetchPlan` dataclass with an optional `service` field (string).
- Update `BaseResolver._build_plan()` helper to accept `service` and forward it when instantiating `FetchPlan`.
- Modify each resolver to populate `service=spec.resolver` (e.g., `obo`, `ols`, `bioportal`, `skos`, `xbrl`).
- In `core.fetch_one()`, attach `plan.service` to the logger adapter extras and pass it into `download_stream()`.

**Acceptance Criteria**:

- [x] `FetchPlan` exposes a `service` attribute with docstring updates.
- [x] All resolver implementations populate `service` consistently.
- [x] `core.fetch_one()` includes `service` in logging context and download invocation.

#### 2.2.2 Add per-service token bucket selection

**Objective**: Key rate limit buckets by service and host while respecting service-specific overrides.

**Implementation**:

- Update `_get_bucket()` signature to accept optional `service` and construct cache keys as `"service:host"`.
- Use `DownloadConfiguration.parse_service_rate_limit()` to override the default per-host rate when available.
- Ensure default per-host buckets remain functional when `service` is absent.
- Reset or isolate `_TOKEN_BUCKETS` entries in unit tests to prevent leakage between scenarios.

**Acceptance Criteria**:

- [x] `_get_bucket()` caches buckets per service/host combination.
- [x] Buckets fall back to `per_host_rate_limit` when no override exists.
- [x] Unit tests cover service-specific, fallback, and host-only buckets.

#### 2.2.3 Document and test per-service rate limiting

**Objective**: Update examples, API docs, and tests to reflect the new behaviour.

**Implementation**:

- Add `rate_limits` examples for OLS and BioPortal in `docs/examples/sources.yaml`.
- Expand API docs (`DocsToKG.OntologyDownload.download.md`, `DocsToKG.OntologyDownload.resolvers.md`) to describe service-aware rate limiting.
- Add/extend unit tests in `tests/ontology_download/test_download.py`, `test_config.py`, and `test_integration.py` for the new service metadata.

**Acceptance Criteria**:

- [x] Example configuration demonstrates per-service overrides.
- [x] API docs mention `service` usage in fetch plans and download rate limiting.
- [x] New and existing tests pass validating per-service rate limiting.

### 2.3 Harden URL Validation with Allowlist and IDN Safety

#### 2.3.1 Implement punycode normalization and homograph detection

**Objective**: Ensure URL validation normalizes IDN hosts to punycode and rejects suspicious homograph attempts.

**Implementation**:

- Add `_enforce_idn_safety()` helper in `download.py` to reject invisible Unicode characters and mixed-script labels.
- Normalize hosts via `.encode("idna").decode("ascii")` before DNS resolution and token bucket lookups.
- Rebuild URL netloc with normalized hostname while preserving IPv6 bracket notation.

**Acceptance Criteria**:

- [x] IDN hosts such as `münchen.example.org` convert to `xn--mnchen-3ya.example.org` before resolution.
- [x] Mixed-script homographs (Latin + Cyrillic) raise `ConfigError` prior to any network access.
- [x] IPv4 and IPv6 literals remain supported after normalization.

#### 2.3.2 Enforce optional host allowlist during validation

**Objective**: Respect `DownloadConfiguration.allowed_hosts` for SSRF mitigation.

**Implementation**:

- Add `DownloadConfiguration.normalized_allowed_hosts()` returning exact and wildcard punycoded hostnames.
- Update `validate_url_security()` to accept the HTTP configuration and enforce allowlist matches (including `*.example.org` support).
- Ensure both `core.fetch_one()` and `download_stream()` pass the active HTTP configuration into the validator.

**Acceptance Criteria**:

- [x] URLs to hosts not present in `allowed_hosts` raise `ConfigError`.
- [x] Wildcard entries such as `*.example.org` permit subdomains.
- [x] Allowlist lookups occur prior to DNS resolution.

#### 2.3.3 Expand unit tests for URL security features

**Objective**: Cover new allowlist and IDN scenarios with deterministic unit tests.

**Implementation**:

- Add tests in `tests/ontology_download/test_download.py` for allowlist success/failure, punycode normalization, wildcard handling, and mixed-script rejection.
- Extend `tests/ontology_download/test_config.py` to verify allowlist normalization helper output.
- Monkeypatch DNS resolution in tests to avoid external lookups and assert normalized hostnames used.

**Acceptance Criteria**:

- [x] Added tests fail on regressions of allowlist enforcement or IDN handling.
- [x] All updated tests pass locally.
- [x] Code coverage includes new helper paths.

---

## Task Completion Tracking

**Foundation (1.1-1.4)**: 27/52 tasks complete
**Robustness Downloads (2.1-2.4)**: 12/27 tasks complete
**Robustness Validation (3.1-3.4)**: 0/21 tasks complete
**Capabilities (4.1-5.2)**: 0/32 tasks complete
**Storage (6.1)**: 0/10 tasks complete
**CLI (7.1-7.3)**: 0/13 tasks complete
**Testing (8.1-8.5)**: 0/30 tasks complete
**Deployment (9.1-9.3)**: 0/13 tasks complete

**Total**: 39/198 tasks complete (19.7%)

## Notes for AI Programming Agents

1. **Work Sequentially**: Complete tasks in numbered order; later tasks depend on earlier ones
2. **Test After Each Subtask**: Run relevant tests after each implementation
3. **Commit Frequently**: Commit working code after each major task (e.g., after 1.1, 1.2, etc.)
4. **Use Type Hints**: All new code should have complete type annotations
5. **Follow Existing Patterns**: Match coding style of surrounding code
6. **Update Tests**: Every code change requires corresponding test updates
7. **Document as You Go**: Add docstrings immediately, not as separate task
8. **Ask for Clarification**: If any instruction is ambiguous, ask before implementing
9. **Verify Acceptance Criteria**: Check off acceptance criteria as you complete each task
10. **Run Full Suite**: Run complete test suite before marking phase complete
