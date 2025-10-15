# Implementation Tasks - Detailed Guide for AI Agents

## Prerequisites & Setup

**Environment Setup:**

```bash
cd /home/paul/DocsToKG
direnv allow  # Or: source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
git checkout main
git pull
```

**Create feature branch:**

```bash
git checkout -b refine-docparsing-phase-1
```

## Task Dependencies Matrix

| Phase | Task | Depends On | Can Run In Parallel |
|-------|------|------------|---------------------|
| 1 | 1.1 | None | - |
| 1 | 1.2 | 1.1 | 1.3 |
| 1 | 1.3 | 1.1 | 1.2 |
| 1 | 1.4 | 1.2, 1.3 | 1.5 |
| 1 | 1.5 | 1.2, 1.3 | 1.4 |
| 2 | 2.1-2.3 | None | All |
| 3 | 3.1-3.2 | None | Both |
| 4 | 4.1 | None | - |
| 5 | 5.1-5.3 | None | All |

---

## Phase 1: Legacy Script Quarantine

- [ ] Task 1.1: Create legacy directory structure
- [ ] Task 1.2: Move HTML converter to legacy with comprehensive shim
- [ ] Task 1.3: Move PDF converter to legacy with comprehensive shim
- [ ] Task 1.4: Create comprehensive unit tests for shims
- [ ] Task 1.5: Update imports in tests that reference moved scripts
- [ ] Task 1.6: Phase 1 completion checklist


**Goal**: Move deprecated scripts to `legacy/` with backward-compatible shims

### Task 1.1: Create legacy directory structure
- [x] Task 1.1: Create legacy directory structure

**What**: Create dedicated `legacy/` subdirectory for deprecated scripts

**Commands**:

```bash
mkdir -p src/DocsToKG/DocParsing/legacy
cat > src/DocsToKG/DocParsing/legacy/__init__.py << 'EOF'
"""Legacy DocParsing scripts (deprecated).

This package contains deprecated direct-invocation scripts maintained only for
backward compatibility. New code should use the unified CLI:
    python -m DocsToKG.DocParsing.cli.doctags_convert

The scripts in this package may be removed in a future release.
"""

__all__ = [
    "run_docling_html_to_doctags_parallel",
    "run_docling_parallel_with_vllm_debug",
]
EOF
```

**Validation**:

```bash
# Verify directory exists
test -d src/DocsToKG/DocParsing/legacy && echo "✓ Directory exists" || (echo "✗ Directory missing" && exit 1)

# Verify __init__.py exists and is non-empty
test -s src/DocsToKG/DocParsing/legacy/__init__.py && echo "✓ __init__.py created" || (echo "✗ __init__.py missing" && exit 1)

# Verify package is importable
python -c "from DocsToKG.DocParsing import legacy; print('✓ Package importable')" || (echo "✗ Import failed" && exit 1)
```

**Expected Output**:

```
✓ Directory exists
✓ __init__.py created
✓ Package importable
```

---

### Task 1.2: Move HTML converter to legacy with comprehensive shim
- [x] Task 1.2: Move HTML converter to legacy with comprehensive shim

**What**: Relocate HTML converter to `legacy/` and create forwarding shim

**Step 1: Move original file**

```bash
# Verify source exists before moving
test -f src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py || \
  (echo "✗ Source file not found" && exit 1)

# Move to legacy
mv src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py \
   src/DocsToKG/DocParsing/legacy/

# Verify move succeeded
test -f src/DocsToKG/DocParsing/legacy/run_docling_html_to_doctags_parallel.py && \
  echo "✓ File moved to legacy/" || (echo "✗ Move failed" && exit 1)
```

**Step 2: Create shim at original location**

```bash
cat > src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py << 'EOF'
#!/usr/bin/env python3
"""
Legacy HTML → DocTags Converter (DEPRECATED)

⚠️ This script is deprecated. Use the unified CLI instead:
    python -m DocsToKG.DocParsing.cli.doctags_convert --mode html

This shim forwards all invocations to the unified CLI for backward compatibility.
It will be removed in a future release.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser (deprecated, forwards to unified CLI)."""
    warnings.warn(
        "run_docling_html_to_doctags_parallel.py is deprecated. "
        "Use: python -m DocsToKG.DocParsing.cli.doctags_convert --mode html",
        DeprecationWarning,
        stacklevel=2,
    )
    from DocsToKG.DocParsing.cli.doctags_convert import build_parser

    return build_parser()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments (deprecated)."""
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Forward to unified CLI with HTML mode forced.

    Args:
        argv: Command-line arguments. None uses sys.argv[1:].

    Returns:
        Exit code from unified CLI.
    """
    warnings.warn(
        "\n"
        "=" * 70 + "\n"
        "DEPRECATION WARNING: run_docling_html_to_doctags_parallel.py\n"
        "=" * 70 + "\n"
        "This script is deprecated. Please update your code to use:\n"
        "  python -m DocsToKG.DocParsing.cli.doctags_convert --mode html\n"
        "\n"
        "This shim will be removed in the next major release.\n"
        "=" * 70,
        DeprecationWarning,
        stacklevel=2,
    )

    from DocsToKG.DocParsing.cli.doctags_convert import main as unified_main

    # Inject --mode html if not already present
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    if "--mode" not in argv:
        argv = ["--mode", "html"] + argv

    return unified_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
EOF
```

**Step 3: Set executable permissions**

```bash
chmod +x src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py
```

**Validation**:

```bash
# Test shim exists and is valid Python
python -m py_compile src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py && \
  echo "✓ Shim syntax valid" || (echo "✗ Syntax error" && exit 1)

# Test shim imports successfully
python -c "from DocsToKG.DocParsing import run_docling_html_to_doctags_parallel; \
  assert hasattr(run_docling_html_to_doctags_parallel, 'main'); \
  print('✓ Shim imports and has main()')" || (echo "✗ Import failed" && exit 1)

# Test deprecation warning appears
python -W all -c "
import warnings
warnings.simplefilter('always')
from unittest.mock import patch
with warnings.catch_warnings(record=True) as w:
    with patch('DocsToKG.DocParsing.cli.doctags_convert.main', return_value=0):
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import main
        main(['--help'])
    assert len(w) >= 1, 'No warnings emitted'
    assert issubclass(w[0].category, DeprecationWarning), 'Wrong warning type'
    assert 'deprecated' in str(w[0].message).lower(), 'Missing deprecation message'
print('✓ Deprecation warning works')
" || (echo "✗ Warning test failed" && exit 1)

# Test legacy module still accessible
python -c "from DocsToKG.DocParsing.legacy import run_docling_html_to_doctags_parallel; \
  print('✓ Legacy module accessible')" || (echo "✗ Legacy import failed" && exit 1)
```

**Expected Output**:

```
✓ Shim syntax valid
✓ Shim imports and has main()
✓ Deprecation warning works
✓ Legacy module accessible
```

---

### Task 1.3: Move PDF converter to legacy with comprehensive shim
- [x] Task 1.3: Move PDF converter to legacy with comprehensive shim

**What**: Relocate PDF converter to `legacy/` and create forwarding shim

**Step 1: Move original file**

```bash
# Verify source exists
test -f src/DocsToKG/DocParsing/run_docling_parallel_with_vllm_debug.py || \
  (echo "✗ Source file not found" && exit 1)

# Move to legacy
mv src/DocsToKG/DocParsing/run_docling_parallel_with_vllm_debug.py \
   src/DocsToKG/DocParsing/legacy/

# Verify move
test -f src/DocsToKG/DocParsing/legacy/run_docling_parallel_with_vllm_debug.py && \
  echo "✓ File moved to legacy/" || (echo "✗ Move failed" && exit 1)
```

**Step 2: Create shim at original location**

```bash
cat > src/DocsToKG/DocParsing/run_docling_parallel_with_vllm_debug.py << 'EOF'
#!/usr/bin/env python3
"""
Legacy PDF → DocTags Converter with vLLM (DEPRECATED)

⚠️ This script is deprecated. Use the unified CLI instead:
    python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf

This shim forwards all invocations to the unified CLI for backward compatibility.
It will be removed in a future release.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser (deprecated, forwards to unified CLI)."""
    warnings.warn(
        "run_docling_parallel_with_vllm_debug.py is deprecated. "
        "Use: python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf",
        DeprecationWarning,
        stacklevel=2,
    )
    from DocsToKG.DocParsing.cli.doctags_convert import build_parser

    return build_parser()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments (deprecated)."""
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Forward to unified CLI with PDF mode forced.

    Args:
        argv: Command-line arguments. None uses sys.argv[1:].

    Returns:
        Exit code from unified CLI.
    """
    warnings.warn(
        "\n"
        "=" * 70 + "\n"
        "DEPRECATION WARNING: run_docling_parallel_with_vllm_debug.py\n"
        "=" * 70 + "\n"
        "This script is deprecated. Please update your code to use:\n"
        "  python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf\n"
        "\n"
        "This shim will be removed in the next major release.\n"
        "=" * 70,
        DeprecationWarning,
        stacklevel=2,
    )

    from DocsToKG.DocParsing.cli.doctags_convert import main as unified_main

    # Inject --mode pdf if not already present
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    if "--mode" not in argv:
        argv = ["--mode", "pdf"] + argv

    return unified_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
EOF
```

**Step 3: Set executable permissions**

```bash
chmod +x src/DocsToKG/DocParsing/run_docling_parallel_with_vllm_debug.py
```

**Validation**: Same as Task 1.2, replace `run_docling_html_to_doctags_parallel` with `run_docling_parallel_with_vllm_debug` and `--mode html` with `--mode pdf`.

---

### Task 1.4: Create comprehensive unit tests for shims
- [x] Task 1.4: Create comprehensive unit tests for shims

**What**: Add test suite verifying shim behavior

**Create test file**:

```bash
cat > tests/test_docparsing_legacy_shims.py << 'EOF'
#!/usr/bin/env python3
"""
Unit tests for legacy DocParsing script shims.

These tests verify that the deprecated scripts still work via their shims
and properly emit deprecation warnings.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest


class TestHTMLShim:
    """Test suite for HTML converter shim."""

    def test_module_imports(self):
        """HTML shim module should import without errors."""
        from DocsToKG.DocParsing import run_docling_html_to_doctags_parallel

        assert hasattr(run_docling_html_to_doctags_parallel, "main")
        assert hasattr(run_docling_html_to_doctags_parallel, "build_parser")
        assert hasattr(run_docling_html_to_doctags_parallel, "parse_args")

    def test_main_callable(self):
        """HTML shim main() should be callable."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import main

        assert callable(main)

    def test_emits_deprecation_warning_on_main(self):
        """HTML shim should emit DeprecationWarning when main() is called."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import main

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            with patch("DocsToKG.DocParsing.cli.doctags_convert.main", return_value=0):
                main(["--help"])

            assert len(warning_list) >= 1, "No warnings captured"
            assert issubclass(
                warning_list[0].category, DeprecationWarning
            ), f"Wrong warning type: {warning_list[0].category}"
            assert "deprecated" in str(warning_list[0].message).lower()

    def test_emits_deprecation_warning_on_build_parser(self):
        """HTML shim should emit warning when build_parser() is called."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import (
            build_parser,
        )

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            parser = build_parser()

            assert len(warning_list) >= 1
            assert issubclass(warning_list[0].category, DeprecationWarning)

    def test_forwards_to_unified_cli_with_html_mode(self):
        """HTML shim should forward to unified CLI with --mode html."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import main

        with patch("DocsToKG.DocParsing.cli.doctags_convert.main") as mock_main:
            mock_main.return_value = 42
            result = main(["--resume", "--workers", "4"])

            # Verify unified CLI was called
            assert mock_main.called, "Unified CLI main() not called"
            assert mock_main.call_count == 1

            # Extract actual arguments passed to unified CLI
            call_args = mock_main.call_args[0][0]
            assert isinstance(call_args, list)
            assert "--mode" in call_args, "Missing --mode flag"
            assert "html" in call_args, "Missing html mode"
            assert "--resume" in call_args, "Missing --resume flag"
            assert "--workers" in call_args, "Missing --workers flag"

            # Verify return value propagated
            assert result == 42

    def test_preserves_existing_mode_flag(self):
        """HTML shim should not duplicate --mode if already present."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import main

        with patch("DocsToKG.DocParsing.cli.doctags_convert.main") as mock_main:
            mock_main.return_value = 0
            main(["--mode", "pdf", "--resume"])  # User explicitly wants pdf

            call_args = mock_main.call_args[0][0]
            mode_count = call_args.count("--mode")
            assert mode_count == 1, f"--mode appears {mode_count} times (should be 1)"


class TestPDFShim:
    """Test suite for PDF converter shim."""

    def test_module_imports(self):
        """PDF shim module should import without errors."""
        from DocsToKG.DocParsing import run_docling_parallel_with_vllm_debug

        assert hasattr(run_docling_parallel_with_vllm_debug, "main")
        assert hasattr(run_docling_parallel_with_vllm_debug, "build_parser")
        assert hasattr(run_docling_parallel_with_vllm_debug, "parse_args")

    def test_main_callable(self):
        """PDF shim main() should be callable."""
        from DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug import main

        assert callable(main)

    def test_emits_deprecation_warning(self):
        """PDF shim should emit DeprecationWarning."""
        from DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug import main

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            with patch("DocsToKG.DocParsing.cli.doctags_convert.main", return_value=0):
                main(["--help"])

            assert len(warning_list) >= 1
            assert issubclass(warning_list[0].category, DeprecationWarning)
            assert "deprecated" in str(warning_list[0].message).lower()

    def test_forwards_to_unified_cli_with_pdf_mode(self):
        """PDF shim should forward to unified CLI with --mode pdf."""
        from DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug import main

        with patch("DocsToKG.DocParsing.cli.doctags_convert.main") as mock_main:
            mock_main.return_value = 0
            main(["--workers", "2"])

            assert mock_main.called
            call_args = mock_main.call_args[0][0]
            assert "--mode" in call_args
            assert "pdf" in call_args
            assert "--workers" in call_args


class TestLegacyModuleAccess:
    """Test that legacy modules are still accessible from legacy package."""

    def test_legacy_html_importable(self):
        """Legacy HTML module should be importable from legacy package."""
        from DocsToKG.DocParsing.legacy import run_docling_html_to_doctags_parallel

        assert hasattr(run_docling_html_to_doctags_parallel, "main")

    def test_legacy_pdf_importable(self):
        """Legacy PDF module should be importable from legacy package."""
        from DocsToKG.DocParsing.legacy import run_docling_parallel_with_vllm_debug

        assert hasattr(run_docling_parallel_with_vllm_debug, "main")

    def test_legacy_package_has_all(self):
        """Legacy package should export expected modules in __all__."""
        from DocsToKG.DocParsing import legacy

        assert hasattr(legacy, "__all__")
        assert "run_docling_html_to_doctags_parallel" in legacy.__all__
        assert "run_docling_parallel_with_vllm_debug" in legacy.__all__
EOF
```

**Run tests**:

```bash
pytest tests/test_docparsing_legacy_shims.py -v --tb=short
```

**Expected output**:

```
tests/test_docparsing_legacy_shims.py::TestHTMLShim::test_module_imports PASSED
tests/test_docparsing_legacy_shims.py::TestHTMLShim::test_main_callable PASSED
tests/test_docparsing_legacy_shims.py::TestHTMLShim::test_emits_deprecation_warning_on_main PASSED
tests/test_docparsing_legacy_shims.py::TestHTMLShim::test_emits_deprecation_warning_on_build_parser PASSED
tests/test_docparsing_legacy_shims.py::TestHTMLShim::test_forwards_to_unified_cli_with_html_mode PASSED
tests/test_docparsing_legacy_shims.py::TestHTMLShim::test_preserves_existing_mode_flag PASSED
tests/test_docparsing_legacy_shims.py::TestPDFShim::test_module_imports PASSED
tests/test_docparsing_legacy_shims.py::TestPDFShim::test_main_callable PASSED
tests/test_docparsing_legacy_shims.py::TestPDFShim::test_emits_deprecation_warning PASSED
tests/test_docparsing_legacy_shims.py::TestPDFShim::test_forwards_to_unified_cli_with_pdf_mode PASSED
tests/test_docparsing_legacy_shims.py::TestLegacyModuleAccess::test_legacy_html_importable PASSED
tests/test_docparsing_legacy_shims.py::TestLegacyModuleAccess::test_legacy_pdf_importable PASSED
tests/test_docparsing_legacy_shims.py::TestLegacyModuleAccess::test_legacy_package_has_all PASSED

===================== 13 passed in 0.15s =======================
```

---

### Task 1.5: Update imports in tests that reference moved scripts
- [x] Task 1.5: Update imports in tests that reference moved scripts

**What**: Fix any existing tests that import the moved scripts

**Step 1: Find affected test files**

```bash
# Create list of files to check
cat > /tmp/check_imports.sh << 'EOF'
#!/bin/bash
echo "Searching for imports that may need updating..."
grep -r "from DocsToKG.DocParsing.run_docling_html" tests/ 2>/dev/null | cut -d: -f1 | sort -u > /tmp/html_imports.txt
grep -r "from DocsToKG.DocParsing.run_docling_parallel_with_vllm" tests/ 2>/dev/null | cut -d: -f1 | sort -u > /tmp/pdf_imports.txt
grep -r "import run_docling_html" tests/ 2>/dev/null | cut -d: -f1 | sort -u >> /tmp/html_imports.txt
grep -r "import run_docling_parallel_with_vllm" tests/ 2>/dev/null | cut -d: -f1 | sort -u >> /tmp/pdf_imports.txt

echo "Files importing HTML converter:"
sort -u /tmp/html_imports.txt
echo ""
echo "Files importing PDF converter:"
sort -u /tmp/pdf_imports.txt
EOF

chmod +x /tmp/check_imports.sh
/tmp/check_imports.sh
```

**Step 2: Update each affected file**

For each file in the lists above, choose appropriate import strategy:

**Strategy A: Use shim (recommended for backward compatibility tests)**

```python
# No change needed - shims are drop-in replacements
from DocsToKG.DocParsing import run_docling_html_to_doctags_parallel
```

**Strategy B: Use legacy package (for tests of legacy-specific behavior)**

```python
# Change to import from legacy package
from DocsToKG.DocParsing.legacy import run_docling_html_to_doctags_parallel
```

**Strategy C: Modernize to unified CLI (for new test patterns)**

```python
# Update to use unified CLI directly
from DocsToKG.DocParsing.cli import doctags_convert

# Then update test calls:
# Old: run_docling_html_to_doctags_parallel.main(['--resume'])
# New: doctags_convert.main(['--mode', 'html', '--resume'])
```

**Recommendation**: Use Strategy A (shims) for most tests unless the test specifically needs to test legacy behavior or should be modernized.

**Step 3: Validate all tests still pass**

```bash
# Run full DocParsing test suite
pytest tests/ -k "docparsing or docling" -v

# If any tests fail, examine the failure and apply appropriate strategy above
```

---

### Task 1.6: Phase 1 completion checklist
- [x] Task 1.6: Phase 1 completion checklist

**Run all validations**:

```bash
#!/bin/bash
set -e

echo "=== Phase 1 Validation ==="
echo ""

# 1. Directory structure
echo "1. Checking directory structure..."
test -d src/DocsToKG/DocParsing/legacy && echo "  ✓ legacy/ directory exists" || exit 1
test -f src/DocsToKG/DocParsing/legacy/__init__.py && echo "  ✓ __init__.py exists" || exit 1

# 2. Shims exist
echo "2. Checking shims..."
test -f src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py && \
  echo "  ✓ HTML shim exists" || exit 1
test -f src/DocsToKG/DocParsing/run_docling_parallel_with_vllm_debug.py && \
  echo "  ✓ PDF shim exists" || exit 1

# 3. Legacy files exist
echo "3. Checking legacy files..."
test -f src/DocsToKG/DocParsing/legacy/run_docling_html_to_doctags_parallel.py && \
  echo "  ✓ Legacy HTML exists" || exit 1
test -f src/DocsToKG/DocParsing/legacy/run_docling_parallel_with_vllm_debug.py && \
  echo "  ✓ Legacy PDF exists" || exit 1

# 4. Imports work
echo "4. Checking imports..."
python -c "from DocsToKG.DocParsing import run_docling_html_to_doctags_parallel" && \
  echo "  ✓ HTML shim imports" || exit 1
python -c "from DocsToKG.DocParsing import run_docling_parallel_with_vllm_debug" && \
  echo "  ✓ PDF shim imports" || exit 1
python -c "from DocsToKG.DocParsing.legacy import run_docling_html_to_doctags_parallel" && \
  echo "  ✓ Legacy HTML imports" || exit 1
python -c "from DocsToKG.DocParsing.legacy import run_docling_parallel_with_vllm_debug" && \
  echo "  ✓ Legacy PDF imports" || exit 1

# 5. Tests pass
echo "5. Running test suite..."
pytest tests/test_docparsing_legacy_shims.py -q && \
  echo "  ✓ All tests pass" || exit 1

# 6. No regressions
echo "6. Checking for regressions..."
pytest tests/ -k "docparsing or docling" -q --tb=line && \
  echo "  ✓ No regressions detected" || exit 1

echo ""
echo "=============================="
echo "✅ Phase 1 Complete!"
echo "=============================="
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Commit: git add -A && git commit -m 'Phase 1: Legacy script quarantine'"
echo "  3. Proceed to Phase 2: Test scaffolding cleanup"
```

**Save validation script**:

```bash
cat > scripts/validate_phase1.sh << 'EOF'
[paste the validation script above]
EOF
chmod +x scripts/validate_phase1.sh
./scripts/validate_phase1.sh
```

**Expected output**: All checks pass with ✓ symbols

---

## Phase 2: Test Scaffolding Cleanup

- [ ] Task 2.1: Remove `_promote_simple_namespace_modules()` from chunker
- [ ] Task 2.2: Move promotion helper to test utilities
- [ ] Task 2.3: Delete duplicate `SOFT_BARRIER_MARGIN` constant
- [ ] Task 2.4: Smoke test chunker functionality
- [ ] Task 2.5: Phase 2 completion checklist


**Goal**: Remove test-only code from production modules

### Task 2.1: Remove `_promote_simple_namespace_modules()` from chunker
- [x] Task 2.1: Remove `_promote_simple_namespace_modules()` from chunker

**What**: Delete Hypothesis test helper from production code

**Current code location**: `src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py`, lines 59-76

**Step 1: Verify current content**

```bash
# Extract lines 59-76 to verify they contain the promotion function
sed -n '59,76p' src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py
```

**Expected output** (should see the promotion function):

```python
def _promote_simple_namespace_modules() -> None:
    """Convert any SimpleNamespace placeholders in sys.modules to real modules.

    Some tests install lightweight SimpleNamespace stubs into sys.modules for
    optional dependencies (for example ``trafilatura``). Hypothesis' internal
    providers assume module objects are hashable, which SimpleNamespace is not.
    Promoting the stubs to ModuleType instances preserves their attributes while
    restoring hashability, preventing spurious test failures.
    """

    for name, module in list(sys.modules.items()):
        if isinstance(module, SimpleNamespace):
            promoted = ModuleType(name)
            promoted.__dict__.update(vars(module))
            sys.modules[name] = promoted


_promote_simple_namespace_modules()
```

**Step 2: Remove function and its invocation**

```bash
# Create backup
cp src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py \
   src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py.backup

# Remove lines 59-76 (function definition and call)
sed -i '59,76d' src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py
```

**Step 3: Remove unused imports**

```bash
# Check if SimpleNamespace and ModuleType are still used elsewhere
grep -n "SimpleNamespace\|ModuleType" src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py

# If no other uses found, remove from imports (around line 29)
# Before:
#   from types import ModuleType, SimpleNamespace
# After:
#   # ModuleType and SimpleNamespace removed - no longer needed

# Apply the change
sed -i 's/from types import ModuleType, SimpleNamespace/# types.ModuleType and SimpleNamespace removed - no longer needed/' \
  src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py
```

**Validation**:

```bash
# Verify function removed
! grep -q "_promote_simple_namespace_modules" \
  src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py && \
  echo "✓ Function removed" || (echo "✗ Function still present" && exit 1)

# Verify imports cleaned
! grep -q "from types import ModuleType, SimpleNamespace" \
  src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py && \
  echo "✓ Imports cleaned" || echo "⚠ Imports may still be needed"

# Verify file still compiles
python -m py_compile src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py && \
  echo "✓ File compiles" || (echo "✗ Syntax error introduced" && exit 1)

# Verify module still imports
python -c "from DocsToKG.DocParsing import DoclingHybridChunkerPipelineWithMin; \
  print('✓ Module imports successfully')" || (echo "✗ Import failed" && exit 1)
```

---

### Task 2.2: Move promotion helper to test utilities
- [x] Task 2.2: Move promotion helper to test utilities

**What**: Extract function to test-only module for Hypothesis compatibility

**Create new test utility file**:

```bash
cat > tests/_stubs.py << 'EOF'
"""
Test Stubs and Utilities for DocParsing Tests

This module provides test-only helpers for managing optional dependencies
and test fixtures. These utilities should NEVER be imported by production code.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, Any


def promote_simple_namespace_modules() -> None:
    """Convert SimpleNamespace test stubs in sys.modules to ModuleType instances.

    Background:
        Some DocParsing tests install lightweight SimpleNamespace objects into
        sys.modules as stubs for optional dependencies (e.g., trafilatura, vllm).
        This allows tests to run without installing heavyweight dependencies.

        However, Hypothesis' internal reflection code assumes all sys.modules
        entries are hashable. SimpleNamespace instances are NOT hashable by
        default, causing Hypothesis to raise TypeError during strategy generation.

    Solution:
        This function scans sys.modules for SimpleNamespace instances and promotes
        them to proper ModuleType objects. The promotion preserves all attributes
        (making them API-compatible with the stubs) while restoring hashability.

    Usage:
        Call this function BEFORE importing any DocParsing modules in tests that
        use Hypothesis and install SimpleNamespace stubs:

        >>> from tests._stubs import promote_simple_namespace_modules
        >>> # Install stubs...
        >>> promote_simple_namespace_modules()  # Make stubs hashable
        >>> from DocsToKG.DocParsing import ...  # Now safe to import

    Warning:
        This function modifies sys.modules globally. Only call it in test fixtures
        or setUp methods, never in production code.

    Examples:
        >>> import sys
        >>> from types import SimpleNamespace
        >>> # Simulate stub installation
        >>> sys.modules['fake_module'] = SimpleNamespace(foo='bar')
        >>> # Promote stubs
        >>> promote_simple_namespace_modules()
        >>> # Verify promotion
        >>> assert isinstance(sys.modules['fake_module'], ModuleType)
        >>> assert sys.modules['fake_module'].foo == 'bar'  # Attributes preserved
        >>> # Clean up
        >>> del sys.modules['fake_module']
    """
    for module_name, module_obj in list(sys.modules.items()):
        if isinstance(module_obj, SimpleNamespace):
            # Create real ModuleType instance
            promoted = ModuleType(module_name)
            # Preserve all attributes from the stub
            promoted.__dict__.update(vars(module_obj))
            # Replace stub with promoted module
            sys.modules[module_name] = promoted


def dependency_stubs(**stubs: Dict[str, Any]) -> None:
    """Install test stubs for optional dependencies.

    Args:
        **stubs: Keyword arguments where key is module name and value is either:
            - SimpleNamespace with stub attributes
            - dict converted to SimpleNamespace
            - callable returning stub

    Examples:
        >>> dependency_stubs(
        ...     vllm=SimpleNamespace(LLM=lambda: None, PoolingParams=lambda: None),
        ...     sentence_transformers=dict(SparseEncoder=None)
        ... )
        >>> promote_simple_namespace_modules()  # Make Hypothesis-safe
    """
    for name, stub in stubs.items():
        if callable(stub):
            sys.modules[name] = stub()
        elif isinstance(stub, dict):
            sys.modules[name] = SimpleNamespace(**stub)
        else:
            sys.modules[name] = stub


__all__ = [
    "promote_simple_namespace_modules",
    "dependency_stubs",
]
EOF
```

**Create tests for the utility**:

```bash
cat > tests/test_stubs.py << 'EOF'
"""Tests for test stub utilities."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests._stubs import promote_simple_namespace_modules, dependency_stubs


class TestPromoteSimpleNamespace:
    """Test suite for promote_simple_namespace_modules()."""

    def test_promotes_simple_namespace_to_module_type(self):
        """SimpleNamespace stubs should be converted to ModuleType."""
        # Install stub
        stub = SimpleNamespace(test_attr="value")
        sys.modules["_test_stub_module"] = stub

        try:
            # Verify it's a SimpleNamespace
            assert isinstance(sys.modules["_test_stub_module"], SimpleNamespace)

            # Promote
            promote_simple_namespace_modules()

            # Verify promotion
            assert isinstance(sys.modules["_test_stub_module"], ModuleType)
            assert sys.modules["_test_stub_module"].test_attr == "value"
        finally:
            # Clean up
            if "_test_stub_module" in sys.modules:
                del sys.modules["_test_stub_module"]

    def test_preserves_attributes(self):
        """Promoted modules should retain all original attributes."""
        stub = SimpleNamespace(
            func=lambda x: x * 2,
            const=42,
            nested=SimpleNamespace(deep="value")
        )
        sys.modules["_test_stub"] = stub

        try:
            promote_simple_namespace_modules()

            mod = sys.modules["_test_stub"]
            assert mod.func(5) == 10
            assert mod.const == 42
            assert mod.nested.deep == "value"
        finally:
            if "_test_stub" in sys.modules:
                del sys.modules["_test_stub"]

    def test_makes_hashable(self):
        """Promoted modules should be hashable (for Hypothesis)."""
        sys.modules["_test_hashable"] = SimpleNamespace()

        try:
            # SimpleNamespace is not hashable
            with pytest.raises(TypeError):
                hash(sys.modules["_test_hashable"])

            promote_simple_namespace_modules()

            # ModuleType is hashable
            hash(sys.modules["_test_hashable"])  # Should not raise
        finally:
            if "_test_hashable" in sys.modules:
                del sys.modules["_test_hashable"]


class TestDependencyStubs:
    """Test suite for dependency_stubs()."""

    def test_installs_stubs(self):
        """dependency_stubs() should install stubs into sys.modules."""
        try:
            dependency_stubs(_test_dep=SimpleNamespace(version="1.0"))
            assert "_test_dep" in sys.modules
            assert sys.modules["_test_dep"].version == "1.0"
        finally:
            if "_test_dep" in sys.modules:
                del sys.modules["_test_dep"]

    def test_converts_dict_to_namespace(self):
        """dict stubs should be converted to SimpleNamespace."""
        try:
            dependency_stubs(_test_dict_dep={"key": "value"})
            assert hasattr(sys.modules["_test_dict_dep"], "key")
            assert sys.modules["_test_dict_dep"].key == "value"
        finally:
            if "_test_dict_dep" in sys.modules:
                del sys.modules["_test_dict_dep"]
EOF
```

**Run tests**:

```bash
pytest tests/test_stubs.py -v
```

**Validation**:

```bash
# Verify test utility file exists
test -f tests/_stubs.py && echo "✓ Test utility created" || exit 1

# Verify it imports
python -c "from tests._stubs import promote_simple_namespace_modules; \
  print('✓ Test utility imports')" || exit 1

# Verify tests pass
pytest tests/test_stubs.py -q && echo "✓ Tests pass" || exit 1
```

---

### Task 2.3: Delete duplicate `SOFT_BARRIER_MARGIN` constant
- [x] Task 2.3: Delete duplicate `SOFT_BARRIER_MARGIN` constant

**What**: Remove second declaration of `SOFT_BARRIER_MARGIN` (line 80)

**Step 1: Verify duplication**

```bash
# Find all occurrences
grep -n "^SOFT_BARRIER_MARGIN" src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py

# Expected: Should show two lines, like:
#   78:SOFT_BARRIER_MARGIN = 64
#   80:SOFT_BARRIER_MARGIN = 64
```

**Step 2: Remove duplicate**

```bash
# Remove second occurrence (line 80)
# Note: Line numbers may have shifted after Task 2.1, so we use pattern matching

# Create a script to remove ONLY the duplicate
python << 'EOF'
from pathlib import Path

file_path = Path("src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py")
lines = file_path.read_text().splitlines(keepends=True)

# Find indices of SOFT_BARRIER_MARGIN declarations
declarations = [i for i, line in enumerate(lines) if line.strip() == "SOFT_BARRIER_MARGIN = 64"]

if len(declarations) < 2:
    print(f"⚠ Expected duplicate, found {len(declarations)} declarations")
elif len(declarations) == 2:
    # Remove second occurrence
    print(f"Removing duplicate at line {declarations[1] + 1}")
    del lines[declarations[1]]
    file_path.write_text("".join(lines))
    print("✓ Duplicate removed")
else:
    print(f"⚠ Found {len(declarations)} declarations (expected 2)")
EOF
```

**Validation**:

```bash
# Count occurrences (should be 1)
count=$(grep -c "^SOFT_BARRIER_MARGIN = 64" src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py)
if [ "$count" -eq 1 ]; then
  echo "✓ Duplicate removed (1 declaration remaining)"
elif [ "$count" -gt 1 ]; then
  echo "✗ Still has $count declarations"
  exit 1
else
  echo "✗ No declarations found"
  exit 1
fi

# Verify file still compiles
python -m py_compile src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py && \
  echo "✓ File compiles" || exit 1
```

---

### Task 2.4: Smoke test chunker functionality
- [x] Task 2.4: Smoke test chunker functionality

**What**: Verify chunker still works after modifications

**Run smoke tests**:

```bash
# Test 1: Help text displays
python -m DocsToKG.DocParsing.cli.chunk_and_coalesce --help && \
  echo "✓ Help displays" || (echo "✗ Help failed" && exit 1)

# Test 2: Module imports
python -c "
from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import main, build_parser
print('✓ Chunker imports successfully')
" || (echo "✗ Import failed" && exit 1)

# Test 3: Parser builds
python -c "
from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import build_parser
parser = build_parser()
args = parser.parse_args(['--help'])
print('✓ Parser builds and parses')
" 2>&1 | grep -q "✓" || echo "✓ Parser works"

# Test 4: SOFT_BARRIER_MARGIN accessible
python -c "
from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import SOFT_BARRIER_MARGIN
assert SOFT_BARRIER_MARGIN == 64, f'Wrong value: {SOFT_BARRIER_MARGIN}'
print('✓ SOFT_BARRIER_MARGIN = 64')
" || (echo "✗ Constant missing or wrong value" && exit 1)
```

**Expected output**:

```
✓ Help displays
✓ Chunker imports successfully
✓ Parser works
✓ SOFT_BARRIER_MARGIN = 64
```

---

### Task 2.5: Phase 2 completion checklist
- [x] Task 2.5: Phase 2 completion checklist

```bash
#!/bin/bash
set -e

echo "=== Phase 2 Validation ==="
echo ""

# 1. Promotion function removed from production
echo "1. Verifying promotion function removed..."
! grep -q "_promote_simple_namespace_modules" \
  src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py && \
  echo "  ✓ Function removed from production" || exit 1

# 2. Test utility exists
echo "2. Checking test utility..."
test -f tests/_stubs.py && echo "  ✓ Test utility created" || exit 1
python -c "from tests._stubs import promote_simple_namespace_modules" && \
  echo "  ✓ Test utility imports" || exit 1

# 3. Duplicate constant removed
echo "3. Verifying SOFT_BARRIER_MARGIN..."
count=$(grep -c "^SOFT_BARRIER_MARGIN = 64" \
  src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py)
[ "$count" -eq 1 ] && echo "  ✓ Single declaration" || exit 1

# 4. Chunker still works
echo "4. Testing chunker functionality..."
python -m DocsToKG.DocParsing.cli.chunk_and_coalesce --help > /dev/null 2>&1 && \
  echo "  ✓ Chunker runs" || exit 1

# 5. Tests pass
echo "5. Running test suite..."
pytest tests/test_stubs.py -q && echo "  ✓ Stub tests pass" || exit 1

echo ""
echo "============================"
echo "✅ Phase 2 Complete!"
echo "============================"
echo ""
echo "Next: Phase 3 - Atomic writes"
```

---

## Phase 3: Atomic Chunk & Vector Writes

- [ ] Task 3.1: Replace chunker direct write with atomic write operation
- [ ] Task 3.2: Replace embeddings direct write with atomic write operation
- [ ] Task 3.3: Add crash recovery integration test suite


**Goal**: Replace direct file writes with atomic write operations to prevent partial file corruption during crashes or interruptions.

### Task 3.1: Replace chunker direct write with atomic write operation
- [x] Task 3.1: Replace chunker direct write with atomic write operation

**Functional Requirements**:

The chunker module currently writes chunk JSONL files by opening the target file directly and writing records sequentially. If the process crashes, is killed, or encounters a disk-full condition mid-write, this leaves behind a partial file containing incomplete JSON records or truncated lines. These partial files appear valid to resume logic because they exist on disk, but they fail when parsers attempt to read them, causing silent data loss or corruption.

You must modify the chunker to use an atomic write pattern that eliminates this failure mode. The atomic write implementation must follow these requirements:

1. **Write to temporary location first**: Instead of opening the final output path directly, the chunker must write all content to a temporary file in the same directory. The temporary file should be named by appending a unique suffix to the intended final filename.

2. **Ensure data is persisted to disk**: After all records are written to the temporary file, the implementation must call operating system flush and sync operations to guarantee that buffered data is written to physical storage. This prevents scenarios where the file appears complete in memory but is lost during a system crash.

3. **Atomic rename operation**: Once the temporary file is complete and synced, the implementation must use the operating system's atomic rename operation to move the temporary file to the final destination path. The rename must be atomic at the filesystem level, meaning it either fully succeeds or fully fails with no intermediate state visible to other processes.

4. **Cleanup on failure**: If any error occurs during writing, flushing, or syncing, the implementation must remove the temporary file before propagating the exception. This ensures that failed writes leave no artifacts behind.

5. **Import the helper function**: The chunker module must import the `atomic_write` context manager from the common utilities module and use it to wrap the existing file write logic.

6. **Preserve existing write logic**: The internal write loop that iterates over chunk records and serializes them to JSONL must remain unchanged. Only the outer file-opening operation should be modified to use the atomic write helper instead of directly opening the output path.

The implementation must maintain backward compatibility with existing file permissions, encoding settings, and error handling behavior. The change should be transparent to callers and require no modifications to manifest logic, resume detection, or downstream consumers.

**Validation Requirements**:

After implementation, verify that:

- The chunker module still compiles and imports without errors
- Running the chunker on a small test corpus produces valid output files
- The output file only appears at its final path after the write completes successfully
- If the write process is interrupted, either a complete valid file exists or no file exists at the target path
- Existing tests continue to pass without modification

---

### Task 3.2: Replace embeddings direct write with atomic write operation
- [x] Task 3.2: Replace embeddings direct write with atomic write operation

**Functional Requirements**:

The embeddings module generates vector JSONL files containing BM25, SPLADE, and dense embeddings for document chunks. Currently, the module writes these files using a temporary file pattern implemented locally within the `write_vectors` function. However, this implementation duplicates logic and does not fully guarantee atomicity because it performs manual cleanup operations that can fail independently.

You must refactor the embeddings writer to use the same atomic write helper used by the chunker. The requirements are:

1. **Identify the existing write pattern**: The `write_vectors` function already creates a temporary file by appending a suffix to the output path, writes all vector records to this temporary file, then attempts to rename it to the final path. This partial implementation must be replaced entirely.

2. **Replace with atomic write helper**: Remove the manual temporary file creation, rename, and cleanup logic. Replace it with a call to the `atomic_write` context manager, passing the final output path as an argument. The context manager handles all temporary file management.

3. **Ensure the helper is imported**: Add `atomic_write` to the imports from the common utilities module if not already present.

4. **Preserve validation logic**: The function currently performs extensive validation on vector dimensions, norms, and sparsity before writing each record. This validation logic must remain in place and execute before any data is written to the temporary file.

5. **Maintain error reporting**: When validation fails, the function writes failure entries to the manifest with specific error messages. This behavior must be preserved exactly, ensuring that manifest entries are written before exceptions propagate.

6. **Keep batch write semantics**: The function writes all vectors for a chunk file in a single atomic operation. This all-or-nothing semantic must be maintained—either all vectors for a file are written successfully, or none appear at the final path.

The refactoring must not change the function's signature, return values, or exception behavior. Callers should see no difference in behavior except improved crash safety.

**Validation Requirements**:

After implementation, verify that:

- The embeddings module imports and compiles without errors
- Running the embeddings pipeline on test data produces valid vector files
- Vector files only appear at final paths after writes complete
- Failed validations still write appropriate manifest entries before raising exceptions
- Memory usage remains unchanged
- Existing tests pass without modification

---

### Task 3.3: Add crash recovery integration test suite
- [x] Task 3.3: Add crash recovery integration test suite

**Functional Requirements**:

Currently, the test suite verifies that the chunker and embeddings modules produce correct output under normal operation, but it does not test behavior during interruption or failure scenarios. You must create a new test module that explicitly verifies atomic write behavior and crash recovery semantics.

The test suite must include these test cases:

**Test Case 1: Chunker interrupted mid-write leaves no partial files**

This test must verify that if the chunker process is interrupted while writing a chunk file, the output directory contains either a complete valid chunk file or no file at all. The test must:

- Create a small test DocTags file as input
- Configure the chunker to write to a temporary directory
- Simulate a write interruption by patching the file write operation to raise an exception after writing some but not all records
- After the exception, verify that the output directory contains no file at the expected output path
- Verify that no temporary files remain in the directory
- Verify that the incomplete data cannot be read by downstream consumers

**Test Case 2: Embeddings interrupted mid-write leaves no partial files**

This test must verify atomic behavior for the embeddings writer. The test must:

- Create a small test chunk file as input
- Configure the embeddings pipeline to write to a temporary directory
- Simulate an interrupt during vector serialization
- Verify no file exists at the expected output path
- Verify no temporary files remain
- Verify that partial vector data is not visible to readers

**Test Case 3: Successful writes produce immediately-readable files**

This test must verify that after an atomic write completes successfully, the output file is immediately visible and readable by other processes. The test must:

- Run the chunker on a test input
- Immediately after the write completes, open the output file and verify all records are present and valid
- Verify the file can be read without errors
- Verify the file is not marked as temporary or partial

**Test Case 4: Resume logic correctly handles atomic write failures**

This test must verify that the pipeline's resume logic correctly identifies incomplete outputs. The test must:

- Run the chunker on an input file
- Record the input file's content hash
- Simulate a failure during write
- Run the chunker again with resume enabled
- Verify that the chunker detects the missing output and attempts to reprocess the input
- Verify that the second run completes successfully
- Verify that the manifest contains entries for both attempts

**Test Case 5: Concurrent writes do not interfere**

This test must verify that multiple chunker processes can write to the same output directory without corrupting each other's files. The test must:

- Launch multiple chunker processes in parallel, each processing different input files
- Verify that all output files are created successfully
- Verify that no output files contain records from the wrong input
- Verify that no temporary files remain after all processes complete

The test module must use pytest fixtures to manage temporary directories, test data, and cleanup. Tests must not require external dependencies beyond those already used by the DocParsing module. Tests must complete in under 10 seconds total to keep CI fast.

**Validation Requirements**:

After creating the test suite:

- All tests must pass on the first run
- Tests must be deterministic and never flake
- Tests must clean up all temporary files
- Tests must be runnable in parallel with other tests

---

## Phase 4: UTC Timestamp Correction

- [ ] Task 4.1: Correct JSONFormatter timestamp behavior in common utilities


**Goal**: Fix the JSON log formatter to emit true UTC timestamps rather than local time with UTC labels

### Task 4.1: Correct JSONFormatter timestamp behavior in common utilities
- [x] Task 4.1: Correct JSONFormatter timestamp behavior in common utilities

**Functional Requirements**:

The DocParsing pipeline emits structured JSON logs for monitoring, debugging, and audit purposes. Each log record includes a timestamp field formatted as an ISO 8601 string with a trailing "Z" character to indicate UTC. However, the current implementation uses the system's local timezone when formatting timestamps, then appends "Z" to the result. This creates falsely-labeled timestamps—the values appear to be UTC but actually represent local time, causing errors when correlating logs across machines in different timezones or when performing time-based analysis.

You must modify the JSON log formatter class to generate true UTC timestamps. The requirements are:

1. **Identify the formatter class**: The `JSONFormatter` class is defined within the `get_logger` function in the common utilities module. This class inherits from Python's standard `logging.Formatter` base class and overrides the `format` method to produce JSON output.

2. **Override the time converter**: The `logging.Formatter` base class uses an attribute called `converter` to determine how to convert timestamps from epoch seconds to struct_time tuples. By default, this attribute points to `time.localtime`, which converts timestamps to the local timezone. You must set this attribute to `time.gmtime` instead, which converts timestamps to UTC.

3. **Add the converter attribute to the class**: You must add a class-level attribute assignment `converter = time.gmtime` immediately after the class definition line and before the first method definition. This attribute must be set at class definition time, not in an instance method, to ensure it applies to all instances of the formatter.

4. **Ensure the time module is imported**: Verify that the `time` module is imported at the top of the file. If it is not already imported, add `import time` to the import statements.

5. **Preserve all other formatter behavior**: The `format` method that constructs the JSON payload must remain unchanged. The method already calls `formatTime` which respects the `converter` attribute. No other modifications to the formatter class are required.

The timestamp format string remains unchanged—it should continue to produce ISO 8601 timestamps with millisecond precision and a trailing "Z" character. The only change is that these timestamps now represent actual UTC time rather than mislabeled local time.

**Validation Requirements**:

After implementation, you must verify that:

**Validation 1: Module still imports without errors**

- Import the common utilities module
- Verify no ImportError or other exception occurs
- Verify the get_logger function is still accessible

**Validation 2: Logger still produces JSON output**

- Call get_logger to create a logger instance
- Emit a test log message
- Verify that a JSON-formatted log line appears on stderr
- Verify the JSON can be parsed without errors

**Validation 3: Timestamps are truly UTC**

- Set the system timezone to a non-UTC value (or verify current timezone is not UTC)
- Create a logger and emit a test message
- Parse the timestamp from the JSON log output
- Calculate the difference between the logged timestamp and the current UTC time
- Verify the difference is less than one second (accounting for execution delay)
- Verify the difference does NOT match the local timezone offset

**Validation 4: Timestamp format unchanged**

- Verify timestamps still follow the pattern: YYYY-MM-DDTHH:MM:SS.ffffffZ
- Verify the millisecond portion is included
- Verify the trailing "Z" character is present

**Validation 5: Existing tests pass**

- Run all tests that use the logger
- Verify no test failures occur due to timestamp format changes
- If any tests compare timestamps to local time expectations, update those tests to expect UTC

---

## Phase 5: Hash Algorithm Tagging

- [ ] Task 5.1: Add environment variable override to content hash computation
- [ ] Task 5.2: Add hash algorithm tag to chunker manifest entries
- [ ] Task 5.3: Add hash algorithm tag to embeddings manifest entries


**Goal**: Enable SHA-256 migration path while maintaining backward compatibility with existing SHA-1 hashes

### Task 5.1: Add environment variable override to content hash computation
- [x] Task 5.1: Add environment variable override to content hash computation

**Functional Requirements**:

The pipeline uses content hashing to detect when input files have changed and need reprocessing. Currently, all hashes use the SHA-1 algorithm. While SHA-1 is sufficient for change detection (collision resistance is not security-critical in this context), some organizations mandate SHA-256 for compliance reasons. The pipeline must support SHA-256 without breaking existing resume logic that relies on SHA-1 hashes in old manifest entries.

You must modify the content hash function to support algorithm selection via environment variable. The requirements are:

1. **Identify the hash function**: The `compute_content_hash` function in the common utilities module takes a file path and an algorithm name as parameters, creates a hasher object using hashlib, reads the file in chunks, and returns the hex digest.

2. **Add environment variable override**: At the very beginning of the function, before creating the hasher object, you must read the `DOCSTOKG_HASH_ALG` environment variable. If this variable is set, use its value as the algorithm name instead of the function's algorithm parameter. If the variable is not set, use the parameter value as before.

3. **Preserve the default parameter**: The function signature must remain unchanged. The `algorithm` parameter must still default to "sha1" to maintain backward compatibility with existing code that calls the function without specifying an algorithm.

4. **Environment variable takes precedence**: When the environment variable is set, it must override both the default parameter value and any algorithm value explicitly passed by callers. This allows operators to force a specific algorithm across the entire pipeline without modifying code in multiple locations.

5. **Validate the algorithm name**: The hashlib module will raise an exception if given an invalid algorithm name. This existing error handling must remain in place—do not add additional validation. Let hashlib's error messages surface directly to operators so they receive clear feedback about supported algorithms.

6. **Document the environment variable**: While not part of the code change itself, you must ensure that documentation mentions the `DOCSTOKG_HASH_ALG` environment variable, its purpose, supported values (sha1, sha256, sha512, etc.), and the fact that changing it invalidates existing resume logic.

The implementation must not change the function's performance characteristics. The environment variable should be read once per function call, not cached globally, to allow runtime changes during long-running processes if needed.

**Validation Requirements**:

After implementation:

**Validation 1: Default behavior unchanged**

- Call the function without setting the environment variable
- Verify it returns SHA-1 hashes
- Verify the output matches the output from the unmodified function

**Validation 2: Environment variable override works**

- Set `DOCSTOKG_HASH_ALG=sha256`
- Call the function with a test file
- Verify the output is a valid SHA-256 hash
- Verify the output matches the result of manually hashing the file with SHA-256

**Validation 3: Invalid algorithms fail clearly**

- Set `DOCSTOKG_HASH_ALG=invalid_algorithm`
- Call the function
- Verify that a clear exception is raised
- Verify the exception message mentions the invalid algorithm name

**Validation 4: Explicit parameter still works**

- Without setting the environment variable
- Call the function with `algorithm="sha256"` explicitly
- Verify it returns a SHA-256 hash

**Validation 5: Environment variable overrides explicit parameter**

- Set `DOCSTOKG_HASH_ALG=sha512`
- Call the function with `algorithm="sha256"` explicitly
- Verify the result is a SHA-512 hash (env var takes precedence)

---

### Task 5.2: Add hash algorithm tag to chunker manifest entries
- [x] Task 5.2: Add hash algorithm tag to chunker manifest entries

**Functional Requirements**:

When the chunker writes manifest entries, it currently records the content hash of each input file to support resume logic. However, the manifest does not record which hash algorithm was used. If an operator changes the hash algorithm, the resume logic compares hashes from different algorithms and incorrectly decides that files have changed, causing unnecessary reprocessing.

You must modify the chunker to record the hash algorithm alongside each content hash in manifest entries. The requirements are:

1. **Identify manifest write locations**: The chunker calls `manifest_append` in several places: when skipping a file due to resume logic, when successfully processing a file, and when a processing error occurs. Each of these calls includes an `input_hash` field. You must add a `hash_alg` field to each of these calls.

2. **Read the algorithm from environment**: At each manifest write location, immediately read the `DOCSTOKG_HASH_ALG` environment variable using the same pattern as in the hash function: `os.getenv("DOCSTOKG_HASH_ALG", "sha1")`. This ensures the recorded algorithm matches the algorithm actually used to compute the hash.

3. **Add hash_alg parameter to manifest calls**: For each `manifest_append` call that includes an `input_hash` parameter, add a `hash_alg` parameter immediately after it. The value should be the result of reading the environment variable. Maintain alphabetical ordering of optional parameters if the codebase follows that convention.

4. **Include algorithm in all manifest entry types**: The hash algorithm must be recorded for success entries, failure entries, and skip entries. This ensures that resume logic has complete information regardless of how the previous run ended.

5. **Preserve manifest append behavior**: The `manifest_append` function accepts arbitrary keyword arguments and includes them in the manifest entry JSON. Adding `hash_alg` as a keyword argument requires no changes to the manifest_append function itself—the new field will automatically appear in the output.

The implementation must add exactly one new field to manifest entries. The field name must be exactly "hash_alg" to match the field name used by other pipeline stages. The default value must be "sha1" to match the default algorithm.

**Validation Requirements**:

After implementation:

**Validation 1: New field appears in manifest**

- Run the chunker on a test input
- Read the resulting manifest entry
- Verify the entry contains a "hash_alg" field
- Verify the field value is "sha1" (default)

**Validation 2: Environment variable controls recorded algorithm**

- Set `DOCSTOKG_HASH_ALG=sha256`
- Run the chunker on a test input
- Read the manifest entry
- Verify "hash_alg" is "sha256"
- Verify "input_hash" is a valid SHA-256 hash

**Validation 3: All entry types include the field**

- Run the chunker in various scenarios (success, failure, skip)
- Verify all resulting manifest entries include "hash_alg"

**Validation 4: Resume logic still functions**

- Run the chunker twice on the same input with the same algorithm
- Verify the second run correctly skips the file
- Change the algorithm
- Run a third time
- Verify the third run reprocesses the file (hash mismatch)

---

### Task 5.3: Add hash algorithm tag to embeddings manifest entries
- [x] Task 5.3: Add hash algorithm tag to embeddings manifest entries

**Functional Requirements**:

The embeddings pipeline must record hash algorithms in its manifest entries using the same approach as the chunker. The requirements are identical to Task 5.2, but applied to the embeddings module instead of the chunker module.

You must locate all `manifest_append` calls in the embeddings module and add `hash_alg` parameters. The embeddings module writes manifest entries in these scenarios:

- When skipping a chunk file during resume
- When successfully processing a chunk file
- When processing fails due to validation errors or other exceptions
- When writing the corpus-level summary entry

The implementation must follow the same patterns as Task 5.2:

- Read the environment variable at each manifest write location
- Use "sha1" as the default value
- Add the parameter immediately after `input_hash`
- Include it in all entry types

Note that the corpus-level summary entry does not have an input_hash field, so it should not have a hash_alg field either. The algorithm field is only meaningful for entries that correspond to specific input files.

**Validation Requirements**:

Same validation steps as Task 5.2, but executed against the embeddings pipeline instead of the chunker.

---

## Phase 6: Simplify CLI Argument Parsing

- [ ] Task 6.1: Script 1: DocParsing/DoclingHybridChunkerPipelineWithMin.py
- [ ] Task 6.2: Script 2: EmbeddingV2.py
- [ ] Task 6.3: Script 3: run_docling_html_to_doctags_parallel.py (legacy)
- [ ] Task 6.4: Script 4: run_docling_parallel_with_vllm_debug.py (legacy)

**Goal**: Remove ~80 lines of argument merging boilerplate code across 4 scripts while preserving exact behavior

### Functional Requirements for Tasks 6.1 through 6.4

**Context and Problem**:

Each of the four main DocParsing scripts (chunker, embeddings, HTML converter, PDF converter) contains nearly identical boilerplate code that handles command-line argument parsing. This boilerplate consists of approximately 20 lines per script that:

1. Creates an argument parser
2. Parses an empty argument list to get default values
3. Parses the actual arguments (either from a provided namespace or from sys.argv)
4. Iterates over the provided arguments
5. Overwrites default values with any non-None provided values
6. Returns the merged result

This pattern was originally introduced to support programmatic invocation where callers provide a pre-populated argument namespace, while also supporting command-line invocation where arguments come from sys.argv. However, this pattern is unnecessarily complex because argparse already handles default values correctly when parsing arguments.

You must replace this boilerplate with a single-line pattern that achieves the same result. The requirements apply identically to all four scripts:

**Requirements**:

1. **Identify the argument parsing location**: Each script has a `main` function that accepts an optional `args` parameter. Near the beginning of this function, look for code that creates a parser, parses default values, parses provided values, and merges them. This code typically spans 15-25 lines.

2. **Understand the intent**: The existing code supports two invocation patterns:
   - **Command-line invocation**: When `args` is None, the script should parse sys.argv using the argument parser
   - **Programmatic invocation**: When `args` is a namespace object, the script should use the provided values while filling in any missing values with defaults

3. **Replace with conditional expression**: Replace the entire boilerplate block with a single line that uses a conditional expression: If `args` is not None, use it as-is. If `args` is None, create the parser and parse sys.argv. This works because:
   - When programmatically invoked, the caller is responsible for providing a complete namespace with all required fields
   - When command-line invoked, argparse automatically applies default values from parser configuration
   - The merging behavior in the old code was redundant because argparse already merges defaults with provided arguments

4. **Preserve argument validation**: The argparse parser performs validation when parsing arguments. This validation must still occur for command-line invocations. For programmatic invocations where validation is bypassed, the caller is responsible for providing valid arguments.

5. **Maintain the variable name**: The result must be assigned to a variable named `args` so that the rest of the function continues to work without modification. The type of `args` remains `argparse.Namespace` in both invocation patterns.

6. **Update the imports if needed**: Some scripts may need to import the `os` module if it's not already imported (for use elsewhere in the script).

7. **Remove unused helper functions**: Some scripts define a `parse_args` helper function that is only used by the boilerplate. If this function is not used elsewhere in the script, it can be removed. However, if it's used by tests or other code, it must be preserved.

**The specific replacement pattern is**:

Replace approximately 20 lines that look like:

- Creating a parser
- Parsing empty arguments to get defaults
- Parsing provided arguments or sys.argv
- Looping over provided arguments
- Merging provided values into defaults

With a single line:

- Use args if provided, otherwise parse sys.argv

**Per-script notes**:

**Script 1: DoclingHybridChunkerPipelineWithMin.py**

- The boilerplate is in the `main` function
- After replacement, verify that data root detection still works correctly
- Verify that both test invocations and CLI invocations continue to function

**Script 2: EmbeddingV2.py**

- The boilerplate is in the `main` function
- After replacement, verify that batch size validation still occurs
- Verify that model configuration objects are built correctly from the simplified args

**Script 3: run_docling_html_to_doctags_parallel.py (legacy)**

- This is a legacy file but still needs the simplification for maintainability
- After replacement, verify that worker count configuration still works
- Verify that the multiprocessing start method is still set correctly

**Script 4: run_docling_parallel_with_vllm_debug.py (legacy)**

- This is a legacy file but still needs the simplification for maintainability
- After replacement, verify that vLLM server configuration still works correctly
- Verify that model name normalization still occurs

**Validation Requirements** (apply to all four scripts):

After modifying each script:

**Validation 1: Script compiles and imports**

- Use py_compile to verify syntax correctness
- Import the module and verify no ImportError occurs

**Validation 2: Help text works**

- Run the script with --help
- Verify the help text appears correctly
- Verify no errors occur during help display

**Validation 3: Command-line invocation works**

- Run the script with various command-line arguments
- Verify arguments are parsed correctly
- Verify the script behaves identically to the previous version

**Validation 4: Programmatic invocation works**

- Import the module's main function
- Create a namespace object with test parameters
- Call main with this namespace
- Verify the script uses the provided values
- Verify the script does not parse sys.argv

**Validation 5: Default values are applied**

- Run the script without providing all arguments
- Verify default values from the parser configuration are used
- Verify the behavior matches the previous version

**Validation 6: Tests still pass**

- Run all tests that invoke the script
- Verify no test failures occur
- Verify test invocation patterns still work

**Validation 7: Line count reduced**

- Count lines in the function before modification
- Count lines in the function after modification
- Verify approximately 15-20 lines were removed

The simplification must not change any observable behavior. The only difference should be shorter, more maintainable code.

---

## Phase 7: Memory Optimization (Drop uuid_to_chunk)

- [x] Task 7.1: Modify Pass A to return only statistics
- [x] Task 7.2: Update Pass A call site in main function
- [x] Task 7.3: Remove uuid_to_chunk parameter from process_chunk_file_vectors
- [x] Task 7.4: Update call sites to process_chunk_file_vectors

**Goal**: Eliminate the corpus-wide text cache to dramatically reduce peak memory usage during embeddings generation

### Context and Problem

The embeddings pipeline currently operates in two passes. Pass A reads all chunk files, assigns UUIDs to chunks that lack them, computes BM25 corpus statistics, and builds a dictionary (`uuid_to_chunk`) that maps every chunk UUID to a Chunk object containing the full text. This dictionary is then held in memory throughout Pass B. Pass B iterates over chunk files again, and for each file, it looks up the chunk texts from the dictionary to generate SPLADE and Qwen embeddings.

This architecture has a critical flaw: The `uuid_to_chunk` dictionary holds the full text of every chunk in the entire corpus simultaneously in memory. For a corpus with 100,000 chunks averaging 300 tokens each, this consumes multiple gigabytes of RAM. This memory usage is unnecessary because the chunk texts are already stored in JSONL files on disk and can be re-read during Pass B without measurable performance impact.

You must refactor the embeddings pipeline to eliminate the `uuid_to_chunk` dictionary entirely, replacing it with direct file reads during Pass B. The refactoring must preserve all existing functionality including UUID management, BM25 statistics, validation, and manifest generation.

### Task 7.1: Modify Pass A to return only statistics
- [x] Task 7.1: Modify Pass A to return only statistics

**Functional Requirements**:

The `process_pass_a` function currently returns two values: the `uuid_to_chunk` dictionary and the `stats` object containing BM25 corpus statistics. You must modify this function to return only the statistics object.

1. **Locate the function signature**: Find the `process_pass_a` function definition. It currently declares a return type hint indicating it returns a tuple of two values.

2. **Change the return type**: Modify the return type hint to indicate the function returns only `BM25Stats`, not a tuple. Remove the `Tuple` wrapper and the `Dict[str, Chunk]` component.

3. **Remove the uuid_to_chunk dictionary**: Inside the function, locate the code that creates and populates the `uuid_to_chunk` dictionary. This code adds an entry to the dictionary for each chunk processed. You must completely remove this dictionary creation, population, and storage.

4. **Preserve UUID assignment**: The function currently assigns UUIDs to chunks that lack them and writes the updated chunk files back to disk. This behavior must be preserved exactly. The function must still call `ensure_uuid` and `jsonl_save` when UUIDs are added.

5. **Preserve BM25 accumulation**: The function currently accumulates BM25 statistics by adding each chunk's text to a `BM25StatsAccumulator`. This accumulation must continue exactly as before. The accumulator still needs the chunk text, so text extraction must remain.

6. **Preserve progress reporting**: The function currently uses tqdm to show progress through chunk files. This progress bar must remain with its current description and formatting.

7. **Preserve logging**: The function currently logs summary information after completing Pass A. This logging must remain unchanged.

8. **Simplify the return statement**: At the end of the function, change the return statement from returning a tuple to returning only the stats object.

The key insight is that Pass A still needs to read chunk texts to compute BM25 statistics, but it does not need to store those texts in memory after processing each file. Extracting text, computing statistics, and then discarding the text is sufficient.

**Validation Requirements**:

After modification:

**Validation 1: Function signature is correct**

- Review the function definition
- Verify the return type indicates only BM25Stats
- Verify no tuple is present in the type hint

**Validation 2: Code compiles**

- Use py_compile on the module
- Verify no syntax errors occur

**Validation 3: Function returns correct type**

- Call the function with test data
- Verify the return value is a BM25Stats object
- Verify attempting to unpack two values raises an error

---

### Task 7.2: Update Pass A call site in main function
- [x] Task 7.2: Update Pass A call site in main function

**Functional Requirements**:

The `main` function currently calls `process_pass_a` and unpacks the result into two variables: `uuid_to_chunk` and `stats`. You must modify this call site to receive only the stats object.

1. **Locate the call site**: Find where `main` calls `process_pass_a`. The call currently looks like an assignment with tuple unpacking.

2. **Remove tuple unpacking**: Change the assignment from unpacking two values to receiving a single value. Assign the result directly to a variable named `stats`.

3. **Remove uuid_to_chunk usage later in main**: Search the rest of the `main` function for any references to `uuid_to_chunk`. Remove these references if they exist. In the current codebase, `uuid_to_chunk` is passed to `process_chunk_file_vectors`, so you'll need to update that call as well (covered in Task 7.3).

4. **Preserve the stats usage**: The stats object is used later when calling `process_chunk_file_vectors`. This usage must be preserved exactly.

**Validation Requirements**:

After modification:

**Validation 1: Code compiles**

- Verify no syntax errors from the assignment change

**Validation 2: Variable exists for later use**

- Verify `stats` is defined and accessible
- Verify it contains the expected BM25Stats object

---

### Task 7.3: Remove uuid_to_chunk parameter from process_chunk_file_vectors
- [x] Task 7.3: Remove uuid_to_chunk parameter from process_chunk_file_vectors

**Functional Requirements**:

The `process_chunk_file_vectors` function currently accepts `uuid_to_chunk` as a parameter and uses it to retrieve chunk texts by UUID. You must modify this function to read chunk texts directly from the input file instead.

1. **Modify the function signature**: Remove the `uuid_to_chunk` parameter from the function definition. The parameter is currently declared between the `chunk_file` parameter and the `stats` parameter.

2. **Locate text retrieval code**: Inside the function, find the code that builds the `texts` list. Currently, this code iterates over UUIDs and looks up each UUID in the `uuid_to_chunk` dictionary to retrieve the text.

3. **Replace with direct file read**: Change the text retrieval to read directly from the chunk file. The function already loads the chunk file rows into a `rows` variable using `jsonl_load`. After loading the rows, extract the text from each row by accessing the "text" key. Build a list of texts in the same order as the rows.

4. **Handle missing text gracefully**: Use the dict `.get` method with a default empty string to handle rows that might lack a "text" key (though this should never happen in valid chunk files).

5. **Preserve UUID extraction**: The function still needs to extract UUIDs from the rows to pair with the generated vectors. This UUID extraction must remain unchanged.

6. **Preserve all validation and error handling**: The function performs extensive validation on vector dimensions and norms. All validation must be preserved exactly.

7. **Preserve manifest writing**: The function writes manifest entries for successes and failures. All manifest logic must remain unchanged.

**Validation Requirements**:

After modification:

**Validation 1: Function signature is correct**

- Verify uuid_to_chunk parameter is removed
- Verify other parameters remain in order

**Validation 2: Texts are retrieved correctly**

- Run the function on test data
- Verify texts list has same length as rows list
- Verify texts contain actual chunk text content

**Validation 3: Vectors are generated correctly**

- Verify SPLADE and Qwen encoders receive the correct texts
- Verify vectors correspond to the correct chunks

---

### Task 7.4: Update call sites to process_chunk_file_vectors
- [x] Task 7.4: Update call sites to process_chunk_file_vectors

**Functional Requirements**:

The `main` function calls `process_chunk_file_vectors` for each chunk file during Pass B. These calls currently pass the `uuid_to_chunk` dictionary as an argument. You must modify these calls to remove this argument.

1. **Locate all call sites**: Find all places where `process_chunk_file_vectors` is called. Typically there is one call site inside the Pass B loop.

2. **Remove the uuid_to_chunk argument**: For each call, identify the position where `uuid_to_chunk` is passed and remove it. Ensure that arguments after this position shift up to fill the gap.

3. **Preserve all other arguments**: The function call includes arguments for chunk_file, stats, args, validator, and logger. All these arguments must remain in their correct positions.

4. **Verify argument order matches signature**: After removal, verify that the argument order at the call site matches the parameter order in the function definition.

**Validation Requirements**:

After modification:

**Validation 1: Calls match signature**

- Review each call site
- Verify arguments align with parameters
- Verify no extra or missing arguments

**Validation 2: Code compiles**

- Compile the module
- Verify no TypeError from mismatched arguments

**Validation 3: Function executes correctly**

- Run the embeddings pipeline on test data
- Verify vectors are generated successfully
- Verify manifest entries are correct

---

### Overall Phase 7 Validation

After completing all tasks:

**Memory usage test**:

- Run the embeddings pipeline on a moderately-sized corpus (1000+ chunks)
- Monitor peak memory usage using tracemalloc or a system monitor
- Compare memory usage to the unmodified version
- Verify memory usage is significantly reduced (should drop by approximately the total size of all chunk texts)

**Functional correctness test**:

- Run the pipeline on the same corpus with both versions
- Compare the output vector files
- Verify vectors are bit-for-bit identical (or functionally equivalent within floating-point tolerance)
- Verify manifest entries contain the same information

**Performance test**:

- Measure wall-clock time for both versions on the same corpus
- Verify the new version is not significantly slower
- Some disk I/O overhead is expected, but should be negligible compared to model inference time

**Resume behavior test**:

- Run the pipeline partway, then interrupt it
- Resume the pipeline
- Verify resume logic still works correctly
- Verify no memory leaks occur across resume cycles

---

## Phase 8: Remaining Enhancements

- [ ] Task 8.1: De-hardcode model and cache directory paths
- [ ] Task 8.2: Implement manifest sharding by stage
- [ ] Task 8.3: Add vLLM service preflight telemetry to manifest
- [ ] Task 8.4: Promote image flags to top-level chunk schema fields
- [ ] Task 8.5: Add offline mode support for model loading
- [ ] Task 8.6: Document SPLADE attention backend fallback behavior


**Goal**: Complete remaining improvements including path configuration, manifest optimizations, schema enhancements, and offline mode support

### Task 8.1: De-hardcode model and cache directory paths
- [x] Task 8.1: De-hardcode model and cache directory paths

**Functional Requirements**:

The embeddings module currently contains hardcoded absolute paths for model caching and storage. These paths are specific to one machine (`/home/paul/hf-cache`) and fail when the pipeline runs on different systems or in CI environments. You must replace these hardcoded paths with environment-aware defaults that work across different systems while still allowing explicit configuration.

**Requirements**:

1. **Identify hardcoded paths**: The embeddings module defines module-level constants `HF_HOME`, `MODEL_ROOT`, `QWEN_DIR`, and `SPLADE_DIR` with hardcoded absolute paths pointing to specific directories in a user's home folder.

2. **Use environment variables with fallbacks**: For each hardcoded path, replace it with a configuration that:
   - First checks for a specific environment variable
   - Falls back to XDG or HuggingFace standard locations if the variable is not set
   - Falls back to sensible defaults in the user's home directory as a last resort

3. **Implement the following hierarchy**:
   - `HF_HOME`: Check `$HF_HOME`, fall back to `~/.cache/huggingface`
   - `MODEL_ROOT`: Check `$DOCSTOKG_MODEL_ROOT`, fall back to `HF_HOME`
   - `QWEN_DIR`: Check `$DOCSTOKG_QWEN_DIR`, fall back to `MODEL_ROOT/Qwen/Qwen3-Embedding-4B`
   - `SPLADE_DIR`: Check `$DOCSTOKG_SPLADE_DIR`, fall back to `MODEL_ROOT/naver/splade-v3`

4. **Add CLI overrides**: Add new command-line arguments to the embeddings parser:
   - `--qwen-model-dir`: Override the Qwen model directory
   - `--splade-model-dir`: Override the SPLADE model directory
   These CLI arguments should take precedence over environment variables when provided.

5. **Update configuration objects**: Modify the `QwenCfg` and `SpladeCfg` dataclasses to accept directory paths from the command-line arguments rather than using the hardcoded module-level constants.

6. **Preserve cache coordination**: Multiple environment variables are set to coordinate caching across HuggingFace libraries (HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE, SENTENCE_TRANSFORMERS_HOME). This coordination must be preserved, updated to reference the new configurable paths.

7. **Document the variables**: The help text for the CLI arguments should clearly explain the precedence order: CLI argument > environment variable > default path.

**Validation Requirements**:

**Validation 1: Default behavior works**

- Unset all related environment variables
- Run the embeddings pipeline
- Verify it creates cache directories in `~/.cache/huggingface`
- Verify models are found

**Validation 2: Environment variables override defaults**

- Set `HF_HOME=/tmp/test-cache`
- Run the pipeline
- Verify cache is created in `/tmp/test-cache`
- Verify no files are created in the default location

**Validation 3: CLI arguments override environment variables**

- Set `DOCSTOKG_QWEN_DIR=/env/qwen`
- Run with `--qwen-model-dir /cli/qwen`
- Verify the CLI path is used, not the environment variable

**Validation 4: Relative paths work**

- Run with `--qwen-model-dir ./local-models/qwen`
- Verify the relative path is correctly resolved
- Verify models are loaded from the relative location

---

### Task 8.2: Implement manifest sharding by stage
- [x] Task 8.2: Implement manifest sharding by stage

**Functional Requirements**:

The pipeline appends all manifest entries to a single file `docparse.manifest.jsonl` regardless of which stage generated them. As this file grows to tens of thousands of entries, the `load_manifest_index` function becomes slow because it must scan the entire file to find relevant entries for a specific stage. You must implement manifest sharding where each stage writes to its own manifest file.

**Requirements**:

1. **Define shard naming convention**: Manifest files must be named according to the pattern `docparse.<stage>.manifest.jsonl` where `<stage>` is the stage identifier (e.g., "chunks", "embeddings", "doctags-html", "doctags-pdf").

2. **Modify manifest_append**: The `manifest_append` function must determine the target manifest file based on the `stage` parameter rather than using a fixed filename. Construct the filename from the stage parameter and write to that file.

3. **Preserve atomicity**: Each append operation must remain atomic. If multiple processes write to the same manifest file, concurrent appends should not corrupt the file. The current line-based append strategy provides this naturally.

4. **Modify load_manifest_index**: Update this function to read from the stage-specific manifest file rather than filtering a monolithic file. The function signature should remain unchanged, but internally it should construct the filename from the stage parameter.

5. **Handle missing files gracefully**: If a stage-specific manifest file does not exist, `load_manifest_index` should return an empty dictionary rather than raising an exception. This maintains backward compatibility with fresh installs.

6. **Provide migration path**: Existing deployments have a monolithic manifest file. The implementation should still work correctly when the monolithic file exists but sharded files do not. Consider: does the old file need to be split, or can new runs simply start using sharded files while leaving old data in place?

7. **Update documentation**: Document the new manifest file naming in docstrings and help text so operators know where to find logs for each stage.

**Validation Requirements**:

**Validation 1: New installations work**

- Run pipeline on a system with no existing manifest
- Verify stage-specific manifest files are created
- Verify each file contains only entries for its stage

**Validation 2: Resume works with sharded manifests**

- Run chunker on test data
- Run again with --resume
- Verify the chunker correctly reads its own manifest shard
- Verify skip decisions are correct

**Validation 3: Concurrent writes don't corrupt files**

- Run multiple chunker processes in parallel writing to the same manifest directory
- Verify all manifest entries are written correctly
- Verify no truncated or corrupted lines appear

**Validation 4: Empty stages return empty indices**

- Call load_manifest_index for a stage that has never run
- Verify it returns an empty dictionary
- Verify no exceptions are raised

---

### Task 8.3: Add vLLM service preflight telemetry to manifest
- [x] Task 8.3: Add vLLM service preflight telemetry to manifest

**Functional Requirements**:

The PDF converter uses a vLLM server to run the Granite-Docling vision model. When PDF conversions fail, operators need to know whether the vLLM server was healthy when processing began. Currently, the converter probes the server, logs the results, but does not persist them to the manifest. You must add a manifest entry that captures vLLM server state at startup.

**Requirements**:

1. **Identify the preflight location**: The PDF converter's `main` function calls `ensure_vllm` to start or reuse a vLLM server. After this function returns and before processing any PDFs, you must write a manifest entry.

2. **Use a sentinel document ID**: The manifest entry must use `doc_id="__service__"` to distinguish it from entries for actual documents. This convention allows queries to separate service telemetry from document processing results.

3. **Capture service metadata**: The manifest entry must include:
   - `served_models`: List of model aliases the server exposes
   - `vllm_version`: Version string from the vLLM package
   - `port`: HTTP port where the server is listening
   - `owns_process`: Boolean indicating whether this run started the server or reused an existing one
   - `status`: Always "success" if execution reaches this point (server startup failures would have raised exceptions earlier)

4. **Write once per run**: The entry should be written once at the beginning of each run, not once per PDF. If the PDF converter runs multiple times with the same server, each run writes its own service entry with the same information.

5. **Include timing**: The entry should include a `duration_s` field indicating how long the server probe took. This helps identify slow startup or networking issues.

6. **Preserve schema consistency**: Use `schema_version="docparse/1.1.0"` to match other entries. Mark the stage as "doctags-pdf" since this is part of the PDF conversion stage.

**Validation Requirements**:

**Validation 1: Entry is written on successful startup**

- Run the PDF converter with a working vLLM server
- Verify a manifest entry with `doc_id="__service__"` exists
- Verify it contains all required fields

**Validation 2: Entry reflects server state accurately**

- Run with a freshly started server
- Verify `owns_process` is true
- Run again with the same server
- Verify `owns_process` is false

**Validation 3: Entry distinguishes from document entries**

- Query the manifest for `doc_id="__service__"`
- Verify only service entries are returned, not document entries

---

### Task 8.4: Promote image flags to top-level chunk schema fields
- [x] Task 8.4: Promote image flags to top-level chunk schema fields

**Functional Requirements**:

Chunk records currently store image metadata in a nested `provenance` object. Downstream scoring and filtering logic must navigate through the provenance structure to access these flags, making queries verbose and error-prone. You must add optional top-level fields that duplicate this information for convenience while preserving the nested fields for full provenance tracking.

**Requirements**:

1. **Add new schema fields**: Extend the `ChunkRow` Pydantic model with three optional top-level fields:
   - `has_image_captions: Optional[bool]` - Indicates presence of image captions
   - `has_image_classification: Optional[bool]` - Indicates presence of image classification labels
   - `num_images: Optional[int]` - Count of images in the chunk

2. **Make fields optional**: These fields must have default value `None` to maintain backward compatibility with existing chunk files that lack these fields. Readers must handle both the presence and absence of these fields gracefully.

3. **Populate during chunking**: Modify the chunker to populate these top-level fields whenever it creates a chunk record. The values should be copies of the corresponding fields from the `ProvenanceMetadata` object.

4. **Preserve provenance fields**: The nested provenance structure must remain unchanged. Both representations must coexist. The provenance provides detailed sourcing information, while the top-level fields provide quick filtering.

5. **Add field documentation**: Add Pydantic Field descriptors with clear descriptions explaining that these fields duplicate provenance data for convenience.

6. **Update serialization**: Verify that `model_dump` includes these fields in the output JSON. The `exclude_none=True` parameter means absent fields won't appear in older records, maintaining storage efficiency.

**Validation Requirements**:

**Validation 1: Schema validates with new fields**

- Create a ChunkRow with the new fields populated
- Call model_dump
- Verify the JSON includes the top-level fields

**Validation 2: Schema validates without new fields**

- Create a ChunkRow without the new fields (use old chunk file data)
- Call model_dump
- Verify no error occurs
- Verify the output JSON omits the absent fields

**Validation 3: Chunker populates fields**

- Run the chunker on a DocTags file containing images
- Read the output chunk file
- Verify chunks with images have the top-level fields populated
- Verify the values match the provenance metadata

**Validation 4: Provenance still present**

- Verify the provenance object still contains the same information
- Verify no data was moved out of provenance, only duplicated

---

### Task 8.5: Add offline mode support for model loading
- [x] Task 8.5: Add offline mode support for model loading

**Functional Requirements**:

The embeddings pipeline loads SPLADE and Qwen models from HuggingFace. By default, HuggingFace libraries check online repositories even when models exist in local cache, causing failures in air-gapped environments. You must add an `--offline` flag that disables all network access during model loading.

**Requirements**:

1. **Add CLI flag**: Add a boolean `--offline` argument to the embeddings argument parser. The help text should explain that this flag prevents network access and requires all models to be present in local cache.

2. **Set environment variable for HuggingFace offline mode**: When `--offline` is True, set the environment variable `TRANSFORMERS_OFFLINE=1` before loading any models. This variable instructs HuggingFace libraries to never access the network.

3. **Update SPLADE configuration**: The `SpladeCfg` dataclass already has a `cache_folder` field. When creating the SPLADE encoder, ensure `local_files_only=True` is passed when offline mode is active. The current code already does this, so verify it remains in place.

4. **Update Qwen configuration**: The Qwen model is loaded via vLLM, which has its own configuration. Add a `--qwen-model-dir` CLI argument that accepts a path to a local model directory. When this argument is provided, vLLM should load from the local path without network access.

5. **Fail fast on missing models**: In offline mode, if a required model is not present in local cache, the pipeline should fail immediately with a clear error message explaining which model is missing and where it was expected. Do not attempt fallback downloads.

6. **Document cache preparation**: Add help text explaining how operators should prepare the cache for offline runs: download models in an online environment first, then transfer the cache directory to the offline environment.

**Validation Requirements**:

**Validation 1: Online mode still works**

- Run without --offline flag
- Verify models are loaded successfully
- Verify pipeline functions normally

**Validation 2: Offline mode works with complete cache**

- Pre-download all required models
- Disconnect from network (or use firewall rules)
- Run with --offline flag
- Verify models load from cache
- Verify no network access attempts occur

**Validation 3: Offline mode fails clearly on missing models**

- Clear a model from cache
- Run with --offline flag
- Verify the error message clearly states which model is missing
- Verify the error message includes the expected cache path

---

### Task 8.6: Document SPLADE attention backend fallback behavior
- [x] Task 8.6: Document SPLADE attention backend fallback behavior

**Functional Requirements**:

The embeddings pipeline accepts a `--splade-attn` argument that controls which attention implementation the SPLADE model uses. When set to "auto", the code maps this to `None` internally, letting HuggingFace/PyTorch choose the best available implementation. However, the help text does not explain this fallback behavior, causing confusion when operators specify "auto" but logs show "sdpa" or "eager" was selected.

**Requirements**:

1. **Update the CLI help text**: Modify the help string for the `--splade-attn` argument to explain the fallback logic:
   - "auto" attempts FlashAttention 2 if available, falls back to SDPA, then to eager
   - "flash_attention_2" requires the Flash Attention 2 package, fails if not installed
   - "sdpa" uses PyTorch's scaled dot-product attention implementation
   - "eager" uses the standard (slower) attention implementation

2. **Add a manifest field**: When writing the corpus summary manifest entry, include a field `splade_attn_backend_used` that records which backend was actually selected. This allows post-run analysis of which implementation ran.

3. **Add SPLADE sparsity threshold to summary**: The current code warns when SPLADE vectors have zero non-zero elements but doesn't record the threshold used for this warning. Add a `sparsity_warn_threshold_pct` field to the summary manifest entry with value 1.0 (indicating 1% of vectors being empty triggers a warning).

4. **Preserve existing behavior**: The actual attention selection logic must not change. Only the documentation and observability are being enhanced.

5. **Consider documenting in module docstring**: Add a note to the module-level docstring explaining the attention backend selection process for future maintainers.

**Validation Requirements**:

**Validation 1: Help text is clear**

- Run `--help` and review the `--splade-attn` help text
- Verify it explains all four options and the fallback behavior

**Validation 2: Summary includes metadata**

- Run the embeddings pipeline
- Read the corpus summary entry from the manifest
- Verify `sparsity_warn_threshold_pct` is present and set to 1.0
- Verify `splade_attn_backend_used` is present (if this information is available to record)

**Validation 3: Behavior unchanged**

- Run with `--splade-attn auto` and verify it works as before
- Run with each other option and verify they work as before

---

## Implementation Phase Summary

| Phase | Tasks | LOC Changed | Complexity | Patterns Used |
|-------|-------|-------------|------------|---------------|
| 1 | 6 | ~200 | Low | File Move, Shim, Test Suite |
| 2 | 5 | ~150 | Low | Code Removal, Test Utility |
| 3 | 3 | ~50 | Medium | Atomic Write, Crash Test |
| 4 | 1 | ~10 | Low | UTC Logger |
| 5 | 3 | ~30 | Low | Env Var Config, Manifest |
| 6 | 4 | -60 | Low | CLI Simplification |
| 7 | 4 | ~100 | High | Streaming Refactor |
| 8 | 6 | ~180 | Medium | Multiple patterns |

**Total**: ~660 LOC changed, ~60 LOC removed

---

## Validation & Testing Strategy

After each phase:

1. **Syntax check**: `python -m py_compile <modified_file>`
2. **Import test**: `python -c "import <module>"`
3. **Unit tests**: `pytest tests/ -k <relevant_tests>`
4. **Integration smoke test**: Run small corpus through pipeline
5. **Phase checklist**: Execute phase-specific validation script

**Final validation before PR**:

```bash
# Run complete test suite
pytest tests/ -v

# Run through small fixture
python -m DocsToKG.DocParsing.cli.doctags_convert --mode html \
  --input Data/HTML --output /tmp/doctags --resume

python -m DocsToKG.DocParsing.cli.chunk_and_coalesce \
  --in-dir /tmp/doctags --out-dir /tmp/chunks

# Check for regressions
git diff main --stat
```

---

## Reference Documents

For detailed implementation patterns, code templates, and troubleshooting:

- **implementation-patterns.md**: Reusable code patterns and recipes
- **design.md**: Architectural decisions and trade-offs
- **proposal.md**: High-level change overview
- **specs/doc-parsing/spec.md**: Specification deltas
