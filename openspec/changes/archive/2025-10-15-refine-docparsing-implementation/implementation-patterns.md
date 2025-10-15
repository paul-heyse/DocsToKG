# Implementation Patterns & Recipes

**Purpose**: This document provides reusable code patterns, scripts, and testing recipes for implementing the DocParsing refinements. Each pattern is self-contained and can be applied across multiple tasks.

**Target Audience**: AI programming agents and human developers implementing the changes

**Organization**: Patterns are grouped by category and reference specific tasks in tasks.md

---

## Table of Contents

1. [File Operation Patterns](#file-operation-patterns)
2. [Testing Patterns](#testing-patterns)
3. [Configuration Patterns](#configuration-patterns)
4. [Refactoring Patterns](#refactoring-patterns)
5. [Validation Patterns](#validation-patterns)
6. [Manifest Patterns](#manifest-patterns)
7. [CLI Patterns](#cli-patterns)
8. [Schema Patterns](#schema-patterns)

---

## File Operation Patterns

### Pattern: Atomic Write Replacement

**Use Case**: Replace direct file writes with crash-safe atomic writes

**Context**: Tasks 3.1, 3.2 - Ensuring no partial files remain after crashes

**Implementation**:

```python
# BEFORE:
with output_path.open("w", encoding="utf-8") as handle:
    for item in data:
        handle.write(json.dumps(item) + "\n")

# AFTER:
from DocsToKG.DocParsing._common import atomic_write

with atomic_write(output_path) as handle:
    for item in data:
        handle.write(json.dumps(item) + "\n")
```

**Script: Apply to file programmatically**:

```python
#!/usr/bin/env python3
"""Apply atomic write pattern to a Python file."""

from pathlib import Path
import re
import sys


def apply_atomic_write_pattern(file_path: Path) -> bool:
    """
    Replace direct file writes with atomic_write() calls.

    Args:
        file_path: Path to Python file to modify

    Returns:
        True if changes were made, False otherwise
    """
    content = file_path.read_text()
    original = content

    # Step 1: Ensure atomic_write is imported
    if "atomic_write" not in content:
        # Find existing _common imports
        import_pattern = r'from DocsToKG\.DocParsing\._common import \((.*?)\)'
        match = re.search(import_pattern, content, re.DOTALL)

        if match:
            # Multi-line import - add to list
            imports = match.group(1)
            if "atomic_write" not in imports:
                new_imports = imports.rstrip() + ",\n    atomic_write"
                content = content.replace(match.group(0),
                    f'from DocsToKG.DocParsing._common import ({new_imports})')
        else:
            # Single-line import - convert to multi-line if needed
            single_import = r'from DocsToKG\.DocParsing\._common import (.+)'
            match = re.search(single_import, content)
            if match:
                existing = match.group(1).strip()
                if "atomic_write" not in existing:
                    content = content.replace(match.group(0),
                        f'from DocsToKG.DocParsing._common import {existing}, atomic_write')

    # Step 2: Replace direct writes with atomic writes
    # Pattern: with path.open("w") as handle:
    patterns = [
        (r'with (\w+)\.open\("w", encoding="utf-8"\) as (\w+):',
         r'with atomic_write(\1) as \2:'),
        (r'with open\((\w+), "w", encoding="utf-8"\) as (\w+):',
         r'with atomic_write(\1) as \2:'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    # Step 3: Write changes if any were made
    if content != original:
        file_path.write_text(content)
        return True
    return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python apply_atomic_write.py <file_path>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if apply_atomic_write_pattern(target):
        print(f"✓ Applied atomic write pattern to {target}")
    else:
        print(f"ℹ No changes needed for {target}")
```

**Validation Template**:

```bash
# Verify syntax
python -m py_compile <modified_file>

# Verify atomic_write is imported
grep -q "atomic_write" <modified_file> && echo "✓ Import present"

# Verify no direct writes remain (adjust pattern for specific use case)
! grep -E 'with .+\.open\("w"' <modified_file> && echo "✓ Direct writes removed"

# Test functionality
python -c "from <module> import main; print('✓ Module imports')"
```

---

### Pattern: File Move with Backup

**Use Case**: Safely move files while preserving originals

**Context**: Task 1.2, 1.3 - Moving scripts to legacy/

**Implementation**:

```bash
#!/bin/bash
# Safe file move with automatic backup

move_with_backup() {
    local source="$1"
    local dest="$2"
    local backup="${source}.backup.$(date +%Y%m%d_%H%M%S)"

    # Verify source exists
    if [ ! -f "$source" ]; then
        echo "✗ Source not found: $source"
        return 1
    fi

    # Create backup
    cp "$source" "$backup"
    echo "✓ Backup created: $backup"

    # Ensure destination directory exists
    mkdir -p "$(dirname "$dest")"

    # Move file
    mv "$source" "$dest"

    # Verify move
    if [ -f "$dest" ] && [ ! -f "$source" ]; then
        echo "✓ Move successful: $source → $dest"
        return 0
    else
        echo "✗ Move failed, restoring from backup"
        mv "$backup" "$source"
        return 1
    fi
}

# Example usage:
# move_with_backup \
#   "src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py" \
#   "src/DocsToKG/DocParsing/legacy/run_docling_html_to_doctags_parallel.py"
```

---

## Testing Patterns

### Pattern: Crash Recovery Test

**Use Case**: Verify atomic writes leave no partial files

**Context**: Task 3.3 - Testing crash-safe behavior

**Complete Test Implementation**:

```python
#!/usr/bin/env python3
"""Test patterns for crash recovery and atomic operations."""

import json
import signal
import subprocess
import time
from pathlib import Path
from typing import List

import pytest


class TestAtomicWrites:
    """Test atomic write behavior across DocParsing pipeline."""

    def test_chunker_atomic_write_on_crash(self, tmp_path):
        """Chunker should leave no partial files when interrupted."""
        # Setup test DocTags
        doctags_dir = tmp_path / "doctags"
        doctags_dir.mkdir()
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()

        # Create sample DocTags file
        test_doctag = doctags_dir / "test.doctags"
        test_doctag.write_text(self._create_sample_doctags())

        # Run chunker in subprocess so we can kill it
        proc = subprocess.Popen(
            [
                "python", "-m",
                "DocsToKG.DocParsing.cli.chunk_and_coalesce",
                "--in-dir", str(doctags_dir),
                "--out-dir", str(chunks_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Let it start processing
        time.sleep(0.5)

        # Kill process mid-execution
        proc.send_signal(signal.SIGKILL)
        proc.wait()

        # Check output directory
        chunk_files = list(chunks_dir.glob("*.chunks.jsonl"))
        temp_files = list(chunks_dir.glob("*.chunks.jsonl.tmp"))

        # Assertions:
        # 1. No temporary files should remain
        assert len(temp_files) == 0, \
            f"Found {len(temp_files)} temp files (should be 0)"

        # 2. If chunk file exists, it should be valid JSONL
        for chunk_file in chunk_files:
            self._assert_valid_jsonl(chunk_file)

        print(f"✓ Crash recovery test passed: {len(chunk_files)} valid chunks")

    def test_embeddings_atomic_write_on_crash(self, tmp_path):
        """Embeddings should leave no partial vector files when interrupted."""
        # Similar pattern to above, but for embeddings
        # Implementation details omitted for brevity
        pass

    def test_resume_after_crash(self, tmp_path):
        """Pipeline should resume correctly after crash."""
        doctags_dir = tmp_path / "doctags"
        doctags_dir.mkdir()
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()

        # Create multiple DocTags
        for i in range(5):
            (doctags_dir / f"doc{i}.doctags").write_text(
                self._create_sample_doctags()
            )

        # Run first pass (let it complete 2-3 files)
        proc = subprocess.Popen(
            [
                "python", "-m",
                "DocsToKG.DocParsing.cli.chunk_and_coalesce",
                "--in-dir", str(doctags_dir),
                "--out-dir", str(chunks_dir),
                "--resume",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1.5)
        proc.send_signal(signal.SIGKILL)
        proc.wait()

        # Count completed files
        completed_first = len(list(chunks_dir.glob("*.chunks.jsonl")))

        # Run second pass with --resume
        proc = subprocess.Popen(
            [
                "python", "-m",
                "DocsToKG.DocParsing.cli.chunk_and_coalesce",
                "--in-dir", str(doctags_dir),
                "--out-dir", str(chunks_dir),
                "--resume",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait()

        # Count all completed files
        completed_final = len(list(chunks_dir.glob("*.chunks.jsonl")))

        # Should have all 5 files now
        assert completed_final == 5, \
            f"Expected 5 files, got {completed_final}"
        assert completed_final > completed_first, \
            "Resume should complete remaining files"

        print(f"✓ Resume test passed: {completed_first} → {completed_final}")

    @staticmethod
    def _create_sample_doctags() -> str:
        """Create minimal valid DocTags for testing."""
        return json.dumps({
            "type": "document",
            "content": [
                {"type": "paragraph", "text": "Sample content for testing."}
            ]
        })

    @staticmethod
    def _assert_valid_jsonl(file_path: Path) -> None:
        """Assert file is valid JSONL with no partial lines."""
        content = file_path.read_text()
        lines = [l for l in content.splitlines() if l.strip()]

        for i, line in enumerate(lines, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Invalid JSON at {file_path}:{i}: {e}\n"
                    f"Line content: {line[:100]}"
                )
```

**Usage**:

```bash
pytest tests/test_atomic_writes.py -v -s
```

---

### Pattern: UTC Validation Test

**Use Case**: Verify logs emit true UTC timestamps

**Context**: Task 4.1 - Validating UTC timestamp fix

**Test Implementation**:

```python
def test_logger_emits_utc_timestamps():
    """Logger should emit UTC timestamps regardless of system timezone."""
    import logging
    import os
    import time
    from datetime import datetime, timezone
    from io import StringIO

    # Save original TZ
    original_tz = os.environ.get('TZ')

    try:
        # Force PST timezone (UTC-8)
        os.environ['TZ'] = 'America/Los_Angeles'
        time.tzset()

        # Get logger (should use UTC)
        from DocsToKG.DocParsing._common import get_logger
        logger = get_logger("test_utc")

        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.handlers.clear()
        logger.addHandler(handler)

        # Log a message
        before = datetime.now(timezone.utc)
        logger.info("Test UTC timestamp")
        after = datetime.now(timezone.utc)

        # Parse logged timestamp
        output = stream.getvalue()
        import json
        log_entry = json.loads(output)

        # Extract and parse timestamp (ISO 8601 format)
        timestamp_str = log_entry["timestamp"]
        logged_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        # Verify it's within expected UTC range (not PST)
        assert before <= logged_time <= after, \
            f"Logged time {logged_time} outside UTC range {before} - {after}"

        print(f"✓ UTC timestamp verified: {timestamp_str}")

    finally:
        # Restore original timezone
        if original_tz:
            os.environ['TZ'] = original_tz
        else:
            os.environ.pop('TZ', None)
        time.tzset()
```

---

## Configuration Patterns

### Pattern: Environment Variable Configuration

**Use Case**: Replace hardcoded paths with env-aware defaults

**Context**: Task 8.1 - De-hardcoding model paths

**Before & After**:

```python
# BEFORE (hardcoded):
HF_HOME = Path("/home/paul/hf-cache")
MODEL_ROOT = HF_HOME
QWEN_DIR = MODEL_ROOT / "Qwen" / "Qwen3-Embedding-4B"
SPLADE_DIR = MODEL_ROOT / "naver" / "splade-v3"

# AFTER (env-aware):
import os
from pathlib import Path

HF_HOME = Path(
    os.getenv("HF_HOME") or
    Path.home() / ".cache" / "huggingface"
)
MODEL_ROOT = Path(
    os.getenv("DOCSTOKG_MODEL_ROOT", str(HF_HOME))
)
QWEN_DIR = Path(
    os.getenv("DOCSTOKG_QWEN_DIR",
    str(MODEL_ROOT / "Qwen" / "Qwen3-Embedding-4B"))
)
SPLADE_DIR = Path(
    os.getenv("DOCSTOKG_SPLADE_DIR",
    str(MODEL_ROOT / "naver" / "splade-v3"))
)

# Also set standard HF env vars
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
```

**Automated Application Script**:

```python
def apply_env_aware_paths(file_path: Path) -> None:
    """Replace hardcoded paths with environment-aware defaults."""
    content = file_path.read_text()

    replacements = [
        # HF_HOME
        (
            'HF_HOME = Path("/home/paul/hf-cache")',
            'HF_HOME = Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface")'
        ),
        # MODEL_ROOT
        (
            'MODEL_ROOT = HF_HOME',
            'MODEL_ROOT = Path(os.getenv("DOCSTOKG_MODEL_ROOT", str(HF_HOME)))'
        ),
        # QWEN_DIR
        (
            'QWEN_DIR = MODEL_ROOT / "Qwen" / "Qwen3-Embedding-4B"',
            'QWEN_DIR = Path(os.getenv("DOCSTOKG_QWEN_DIR", '
            'str(MODEL_ROOT / "Qwen" / "Qwen3-Embedding-4B")))'
        ),
        # SPLADE_DIR
        (
            'SPLADE_DIR = MODEL_ROOT / "naver" / "splade-v3"',
            'SPLADE_DIR = Path(os.getenv("DOCSTOKG_SPLADE_DIR", '
            'str(MODEL_ROOT / "naver" / "splade-v3")))'
        ),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    file_path.write_text(content)
    print(f"✓ Applied env-aware paths to {file_path}")
```

**Validation**:

```bash
# Test with default paths
python -c "
from DocsToKG.DocParsing.EmbeddingV2 import HF_HOME, QWEN_DIR
print(f'HF_HOME: {HF_HOME}')
print(f'QWEN_DIR: {QWEN_DIR}')
assert 'paul' not in str(HF_HOME).lower(), 'Hardcoded path remains'
print('✓ Default paths are generic')
"

# Test with custom env vars
HF_HOME=/tmp/cache DOCSTOKG_QWEN_DIR=/models/qwen python -c "
from DocsToKG.DocParsing.EmbeddingV2 import HF_HOME, QWEN_DIR
assert str(HF_HOME) == '/tmp/cache', f'HF_HOME not respected: {HF_HOME}'
assert str(QWEN_DIR) == '/models/qwen', f'QWEN_DIR not respected: {QWEN_DIR}'
print('✓ Environment overrides work')
"
```

---

### Pattern: Offline Mode Implementation

**Use Case**: Support air-gapped deployments

**Context**: Task 8.5 - Adding offline operation mode

**CLI Integration**:

```python
# In build_parser():
parser.add_argument(
    "--offline",
    action="store_true",
    help=(
        "Enable offline mode: disable network access for model loading. "
        "Requires all models to be pre-cached locally. "
        "Sets TRANSFORMERS_OFFLINE=1 and validates model paths exist."
    )
)

# In main():
def main(args: argparse.Namespace | None = None) -> int:
    parser = build_parser()
    args = args if args is not None else parser.parse_args()

    if args.offline:
        # Set offline environment variables
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        logger.info(
            "Offline mode enabled",
            extra={"extra_fields": {"network_access": "disabled"}}
        )

        # Validate required models exist locally
        _validate_offline_model_availability(args)

    # ... rest of main

def _validate_offline_model_availability(args: argparse.Namespace) -> None:
    """Verify all required models are available locally in offline mode."""
    from DocsToKG.DocParsing.EmbeddingV2 import QWEN_DIR, SPLADE_DIR

    required_paths = [
        ("Qwen model", QWEN_DIR),
        ("SPLADE model", SPLADE_DIR),
    ]

    missing = []
    for name, path in required_paths:
        if not path.exists():
            missing.append(f"{name} ({path})")

    if missing:
        raise FileNotFoundError(
            f"Offline mode requires local models, but the following are missing:\n" +
            "\n".join(f"  - {item}" for item in missing) +
            f"\n\nEither:\n"
            f"  1. Download models and set paths via env vars\n"
            f"  2. Run without --offline to allow network access"
        )
```

---

## Refactoring Patterns

### Pattern: CLI Simplification

**Use Case**: Remove boilerplate argument merging

**Context**: Task 6.1-6.4 - Simplifying CLI parsing

**Bulk Application Script**:

```python
#!/usr/bin/env python3
"""Apply CLI simplification pattern to multiple files."""

import re
from pathlib import Path
from typing import List


def simplify_cli_parsing(file_path: Path) -> bool:
    """
    Replace 'merge defaults + provided' pattern with direct parsing.

    Returns:
        True if changes were made
    """
    content = file_path.read_text()
    original = content

    # Pattern to match the old boilerplate
    old_pattern = re.compile(
        r'parser = build_parser\(\)\s+'
        r'defaults = parser\.parse_args\(\[\]\)\s+'
        r'provided = parse_args\(\) if args is None else args\s+'
        r'for key, value in vars\(provided\)\.items\(\):\s+'
        r'if value is not None:\s+'
        r'setattr\(defaults, key, value\)\s+'
        r'args = defaults',
        re.MULTILINE
    )

    # Replacement
    new_code = 'args = args if args is not None else build_parser().parse_args()'

    content = old_pattern.sub(new_code, content)

    if content != original:
        file_path.write_text(content)
        return True
    return False


def apply_to_multiple_files(file_paths: List[Path]) -> None:
    """Apply CLI simplification to multiple files."""
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        if simplify_cli_parsing(file_path):
            print(f"  ✓ Simplified CLI parsing")

            # Count lines saved
            old_lines = sum(1 for _ in open(file_path))
            # Estimate: removed ~6 lines, added 1 line = net -5
            print(f"  ✓ Saved ~5 lines")
        else:
            print(f"  ℹ No changes needed")


if __name__ == "__main__":
    target_files = [
        Path("src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py"),
        Path("src/DocsToKG/DocParsing/EmbeddingV2.py"),
        Path("src/DocsToKG/DocParsing/legacy/run_docling_html_to_doctags_parallel.py"),
        Path("src/DocsToKG/DocParsing/legacy/run_docling_parallel_with_vllm_debug.py"),
    ]

    apply_to_multiple_files(target_files)
    print("\n✅ CLI simplification complete")
```

---

### Pattern: Streaming Architecture Refactor

**Use Case**: Eliminate memory-hungry data structures

**Context**: Task 7.1-7.4 - Dropping uuid_to_chunk map

**Step-by-Step Refactoring**:

**Step 1: Modify Pass A signature and implementation**

```python
# BEFORE:
def process_pass_a(
    files: Sequence[Path],
    logger
) -> Tuple[Dict[str, Chunk], BM25Stats]:
    """UUID assignment + BM25 stats + text retention."""
    uuid_to_chunk: Dict[str, Chunk] = {}
    accumulator = BM25StatsAccumulator()

    for chunk_file in tqdm(files, desc="Pass A"):
        rows = jsonl_load(chunk_file)
        if ensure_uuid(rows):
            jsonl_save(chunk_file, rows)
        for row in rows:
            text = row.get("text", "")
            uuid_value = row["uuid"]
            doc_id = row.get("doc_id", "unknown")
            # MEMORY HOG: Store full text
            uuid_to_chunk[uuid_value] = Chunk(
                uuid=uuid_value,
                text=text,
                doc_id=doc_id
            )
            accumulator.add_document(text)

    return uuid_to_chunk, accumulator.finalize()

# AFTER:
def process_pass_a(
    files: Sequence[Path],
    logger
) -> BM25Stats:
    """UUID assignment + BM25 stats (text discarded after use)."""
    accumulator = BM25StatsAccumulator()

    for chunk_file in tqdm(files, desc="Pass A: BM25 stats"):
        rows = jsonl_load(chunk_file)
        if ensure_uuid(rows):
            jsonl_save(chunk_file, rows)
        for row in rows:
            text = row.get("text", "")
            # Use text for stats, then discard
            accumulator.add_document(text)
            # No retention!

    return accumulator.finalize()
```

**Step 2: Update call site**

```python
# BEFORE:
uuid_to_chunk, stats = process_pass_a(files, logger)

# AFTER:
stats = process_pass_a(files, logger)
```

**Step 3: Refactor Pass B to stream from disk**

```python
# BEFORE:
def process_chunk_file_vectors(
    chunk_file: Path,
    uuid_to_chunk: Dict[str, Chunk],  # ← Remove this parameter
    stats: BM25Stats,
    args: argparse.Namespace,
    validator: SPLADEValidator,
    logger,
) -> Tuple[int, List[int], List[float]]:
    rows = jsonl_load(chunk_file)
    uuids = [row["uuid"] for row in rows]
    # MEMORY HOG: Look up text from corpus-wide dict
    texts = [uuid_to_chunk[uuid].text for uuid in uuids]

    # ... rest of encoding

# AFTER:
def process_chunk_file_vectors(
    chunk_file: Path,
    # uuid_to_chunk parameter REMOVED
    stats: BM25Stats,
    args: argparse.Namespace,
    validator: SPLADEValidator,
    logger,
) -> Tuple[int, List[int], List[float]]:
    rows = jsonl_load(chunk_file)
    uuids = [row["uuid"] for row in rows]
    # Stream from disk: already have rows loaded
    texts = [row.get("text", "") for row in rows]

    # ... rest unchanged
```

**Step 4: Update all Pass B call sites**

```bash
# Find all calls to process_chunk_file_vectors
grep -n "process_chunk_file_vectors" src/DocsToKG/DocParsing/EmbeddingV2.py

# For each call, remove uuid_to_chunk argument:
# BEFORE:
#   process_chunk_file_vectors(chunk_file, uuid_to_chunk, stats, ...)
# AFTER:
#   process_chunk_file_vectors(chunk_file, stats, ...)
```

**Verification**:

```python
def test_streaming_reduces_memory():
    """Verify streaming architecture reduces peak memory."""
    import tracemalloc

    # Create test corpus
    num_chunks = 1000
    chunk_size = 1024  # 1KB per chunk

    # Measure old approach (simulated)
    tracemalloc.start()
    # Simulate retaining all text
    corpus = ["x" * chunk_size for _ in range(num_chunks)]
    peak_old = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Measure new approach (streaming)
    tracemalloc.start()
    # Process without retention
    for _ in range(num_chunks):
        text = "x" * chunk_size
        # Use text (simulate BM25)
        _ = len(text.split())
        # Text goes out of scope immediately
    peak_new = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Streaming should use <10% of retained memory
    ratio = peak_new / peak_old
    assert ratio < 0.1, f"Streaming uses {ratio*100:.1f}% (should be <10%)"
    print(f"✓ Memory reduction: {peak_old/1024/1024:.1f}MB → {peak_new/1024/1024:.1f}MB")
```

---

## Manifest Patterns

### Pattern: Manifest Sharding

**Use Case**: Split monolithic manifest by stage

**Context**: Task 8.2 - Improving resume performance

**Implementation**:

```python
# UPDATE: manifest_append() function
def manifest_append(
    stage: str,
    doc_id: str,
    status: str,
    *,
    duration_s: float = 0.0,
    warnings: Optional[List[str]] = None,
    error: Optional[str] = None,
    schema_version: str = "",
    **metadata,
) -> None:
    """Append structured entry to stage-specific manifest shard."""
    allowed_status = {"success", "failure", "skip"}
    if status not in allowed_status:
        raise ValueError(f"status must be one of {sorted(allowed_status)}")

    manifest_dir = data_manifests()
    # CHANGE: Use stage-specific file
    manifest_path = manifest_dir / f"docparse.{stage}.manifest.jsonl"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "doc_id": doc_id,
        "status": status,
        "duration_s": round(duration_s, 3),
        "warnings": warnings or [],
        "schema_version": schema_version,
    }
    if error is not None:
        entry["error"] = str(error)
    entry.update(metadata)

    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


# UPDATE: load_manifest_index() function
def load_manifest_index(stage: str, root: Optional[Path] = None) -> Dict[str, dict]:
    """Load manifest index from stage-specific shard with fallback."""
    manifest_dir = data_manifests(root)

    # Try stage-specific shard first
    shard_path = manifest_dir / f"docparse.{stage}.manifest.jsonl"
    if shard_path.exists():
        return _read_manifest_from_file(shard_path, stage)

    # Fallback to monolithic manifest (backward compatibility)
    monolithic_path = manifest_dir / "docparse.manifest.jsonl"
    if monolithic_path.exists():
        return _read_manifest_from_file(monolithic_path, stage, filter_stage=True)

    return {}


def _read_manifest_from_file(
    path: Path,
    stage: str,
    filter_stage: bool = False
) -> Dict[str, dict]:
    """Read manifest entries from file, optionally filtering by stage."""
    index: Dict[str, dict] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip if filtering and stage doesn't match
            if filter_stage and entry.get("stage") != stage:
                continue

            doc_id = entry.get("doc_id")
            if not doc_id:
                continue

            # Keep latest entry for each doc_id
            index[doc_id] = entry

    return index
```

**Migration Note**: No manual migration needed. On first run after update:

- New entries go to sharded manifests
- Old entries in monolithic manifest still accessible via fallback
- Users can optionally split old manifest with this script:

```bash
#!/bin/bash
# Optional: Split existing monolithic manifest into shards

MANIFEST_DIR="Data/Manifests"
MONOLITHIC="$MANIFEST_DIR/docparse.manifest.jsonl"

if [ ! -f "$MONOLITHIC" ]; then
    echo "No monolithic manifest found"
    exit 0
fi

# Extract by stage
for stage in "doctags-html" "doctags-pdf" "chunks" "embeddings"; do
    jq -r "select(.stage == \"$stage\")" "$MONOLITHIC" \
        > "$MANIFEST_DIR/docparse.$stage.manifest.jsonl"

    count=$(wc -l < "$MANIFEST_DIR/docparse.$stage.manifest.jsonl")
    echo "✓ Extracted $count entries for stage: $stage"
done

# Optionally backup and remove monolithic
mv "$MONOLITHIC" "$MONOLITHIC.backup"
echo "✓ Backed up monolithic manifest"
```

---

### Pattern: Manifest Field Addition

**Use Case**: Add new fields to existing manifest calls

**Context**: Task 5.2-5.3 - Adding hash_alg field

**Automated Script**:

```python
#!/usr/bin/env python3
"""Add hash_alg field to all manifest_append() calls."""

import re
from pathlib import Path
from typing import List


def add_hash_alg_field(file_path: Path) -> int:
    """
    Add hash_alg field to manifest_append calls that have input_hash.

    Returns:
        Number of modifications made
    """
    content = file_path.read_text()
    modifications = 0

    # Pattern: manifest_append(..., input_hash=..., ...)
    # We want to add: hash_alg=os.getenv("DOCSTOKG_HASH_ALG", "sha1"),

    # Find all manifest_append calls
    pattern = re.compile(
        r'manifest_append\((.*?)\)',
        re.DOTALL
    )

    def add_field(match):
        nonlocal modifications
        call_content = match.group(1)

        # Only modify if input_hash present but hash_alg not
        if 'input_hash=' in call_content and 'hash_alg=' not in call_content:
            # Find where to insert (after input_hash line)
            lines = call_content.split('\n')
            new_lines = []

            for line in lines:
                new_lines.append(line)
                if 'input_hash=' in line:
                    # Add hash_alg on next line with same indentation
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(
                        ' ' * indent +
                        'hash_alg=os.getenv("DOCSTOKG_HASH_ALG", "sha1"),'
                    )
                    modifications += 1

            return f'manifest_append({chr(10).join(new_lines)})'

        return match.group(0)

    content = pattern.sub(add_field, content)

    if modifications > 0:
        file_path.write_text(content)

    return modifications


if __name__ == "__main__":
    files_to_update = [
        "src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py",
        "src/DocsToKG/DocParsing/EmbeddingV2.py",
    ]

    total = 0
    for file_path in files_to_update:
        path = Path(file_path)
        count = add_hash_alg_field(path)
        if count > 0:
            print(f"✓ Added hash_alg to {count} calls in {path.name}")
            total += count
        else:
            print(f"ℹ No changes needed in {path.name}")

    print(f"\n✅ Total: {total} manifest calls updated")
```

---

## Schema Patterns

### Pattern: Schema Field Addition

**Use Case**: Add optional fields to Pydantic schemas

**Context**: Task 8.4 - Promoting image flags to top-level

**Implementation**:

```python
# File: src/DocsToKG/DocParsing/schemas.py

# BEFORE (fields only in ProvenanceMetadata):
class ChunkRow(BaseModel):
    doc_id: str = Field(...)
    source_path: str = Field(...)
    chunk_id: int = Field(...)
    # ... other fields ...
    provenance: Optional[ProvenanceMetadata] = Field(None)

# AFTER (add top-level optional fields):
class ChunkRow(BaseModel):
    doc_id: str = Field(...)
    source_path: str = Field(...)
    chunk_id: int = Field(...)
    # ... other fields ...

    # NEW: Top-level image metadata (optional for backward compat)
    has_image_captions: Optional[bool] = Field(
        None,
        description="Whether chunk includes image captions"
    )
    has_image_classification: Optional[bool] = Field(
        None,
        description="Whether chunk includes image classification"
    )
    num_images: Optional[int] = Field(
        None,
        ge=0,
        description="Number of images in chunk"
    )

    provenance: Optional[ProvenanceMetadata] = Field(None)
```

**Usage in Chunker**:

```python
# When creating ChunkRow, populate both locations:
row = ChunkRow(
    doc_id=name,
    source_path=str(path),
    chunk_id=cid,
    # ... other fields ...

    # Top-level (for easy access)
    has_image_captions=r.has_image_captions,
    has_image_classification=r.has_image_classification,
    num_images=r.num_images,

    # Provenance (for detailed tracking)
    provenance=ProvenanceMetadata(
        parse_engine=parse_engine,
        docling_version=docling_version,
        has_image_captions=r.has_image_captions,
        has_image_classification=r.has_image_classification,
        num_images=r.num_images,
    ),
)
```

**Backward Compatibility**: Optional fields allow old JSONL files (without these fields) to still validate.

---

## CLI Patterns

### Pattern: CLI Help Text Enhancement

**Use Case**: Improve operator understanding of options

**Context**: Task 8.6 - Documenting SPLADE attention backend

**Before & After**:

```python
# BEFORE:
parser.add_argument(
    "--splade-attn",
    type=str,
    default="auto",
    choices=["auto", "sdpa", "eager", "flash_attention_2"],
    help="Attention backend for SPLADE transformer (default: auto/SDPA)."
)

# AFTER:
parser.add_argument(
    "--splade-attn",
    type=str,
    default="auto",
    choices=["auto", "sdpa", "eager", "flash_attention_2"],
    help=(
        "Attention backend for SPLADE transformer. "
        "'auto' tries backends in order: SDPA → eager → FlashAttention2 (if installed). "
        "Explicitly specify 'sdpa', 'eager', or 'flash_attention_2' to force a backend. "
        "Default: auto"
    )
)
```

---

## Validation Patterns

### Pattern: Comprehensive Phase Validation

**Use Case**: Verify all changes in a phase work correctly

**Template**:

```bash
#!/bin/bash
# Template for phase validation scripts

set -e

PHASE_NAME="Phase X"
PHASE_NUM="X"

echo "=== $PHASE_NAME Validation ==="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

check_passed=0
check_failed=0

run_check() {
    local description="$1"
    local command="$2"

    echo -n "Checking: $description... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((check_passed++))
    else
        echo -e "${RED}✗${NC}"
        ((check_failed++))
        return 1
    fi
}

# Check 1: Files exist
run_check "Required files exist" "test -f path/to/file"

# Check 2: Syntax valid
run_check "Python syntax valid" "python -m py_compile path/to/file.py"

# Check 3: Imports work
run_check "Module imports" "python -c 'import module'"

# Check 4: Tests pass
run_check "Unit tests pass" "pytest tests/test_specific.py -q"

# Check 5: Integration smoke test
run_check "CLI runs" "python -m module.cli --help > /dev/null"

# Summary
echo ""
echo "======================================="
if [ $check_failed -eq 0 ]; then
    echo -e "${GREEN}✅ $PHASE_NAME Complete!${NC}"
    echo "Passed: $check_passed/$((check_passed + check_failed))"
    echo ""
    echo "Next steps:"
    echo "  git add -A"
    echo "  git commit -m 'Phase $PHASE_NUM: <description>'"
    echo "  # Proceed to Phase $((PHASE_NUM + 1))"
    exit 0
else
    echo -e "${RED}❌ $PHASE_NAME Failed${NC}"
    echo "Passed: $check_passed, Failed: $check_failed"
    exit 1
fi
```

---

## Quick Reference

### Pattern Application Checklist

For each pattern:

- [ ] Read pattern description and context
- [ ] Review before/after code examples
- [ ] Run provided validation commands
- [ ] Verify no regressions introduced
- [ ] Update documentation if pattern is customized

### Common Validation Commands

```bash
# Syntax check
python -m py_compile <file>

# Import test
python -c "from <module> import <function>"

# Type check (if mypy configured)
mypy <file> --strict

# Unit tests
pytest tests/ -k <pattern> -v

# Integration test
python -m <cli.module> --help
```

### Debugging Patterns

When a pattern fails:

1. **Check file paths**: Verify all Path objects resolve correctly
2. **Inspect changes**: Use `git diff` to see what actually changed
3. **Test imports**: Run isolated import tests
4. **Review logs**: Check for error messages in structured JSON logs
5. **Rollback**: Restore from `.backup` files if needed

---

**End of Implementation Patterns**

For questions or additional patterns, refer to:

- tasks.md - Detailed task breakdown
- design.md - Architectural decisions
- proposal.md - High-level overview
