# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.fixtures.test_corpus_manifest",
#   "purpose": "Manifest and registry of test corpus fixtures (archives, adversarial cases, edge cases)",
#   "sections": [
#     {"id": "corpus-registry", "name": "CorpusRegistry", "anchor": "class-corpus-registry", "kind": "class"},
#     {"id": "archive-metadata", "name": "ArchiveMetadata", "anchor": "dataclass-archive-metadata", "kind": "dataclass"},
#     {"id": "adversarial-cases", "name": "Adversarial cases", "anchor": "section-adversarial-cases", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Test corpus manifest and registry.

Defines minimal, deterministic archive fixtures for testing extraction,
validation, and error handling paths. All fixtures are hermetic (no network)
and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ArchiveMetadata:
    """Metadata for a single test corpus archive."""

    name: str
    """Archive filename (e.g., 'traversal.zip')."""

    format: Literal["zip", "tar.gz", "tar.bz2", "tar.xz", "tar"]
    """Archive format."""

    size_bytes: int
    """Uncompressed size in bytes."""

    num_entries: int
    """Number of files/directories."""

    category: Literal["benign", "adversarial", "edge_case", "unicode", "performance"]
    """Test category."""

    description: str
    """Human-readable description."""

    purpose: str
    """Test scenario this archive targets."""

    expected_issues: list[str] = field(default_factory=list)
    """Known issues or edge cases in this archive."""

    @property
    def path(self) -> Path:
        """Full path to archive in test corpus."""
        corpus_root = Path(__file__).parent / "corpus"
        return corpus_root / self.name


class CorpusRegistry:
    """Registry of all test corpus archives."""

    # Benign small archives for smoke tests
    EMPTY_ZIP = ArchiveMetadata(
        name="empty.zip",
        format="zip",
        size_bytes=22,  # Minimal ZIP with no files
        num_entries=0,
        category="benign",
        description="Empty ZIP file",
        purpose="Test handling of empty archives",
    )

    SIMPLE_TAR_GZ = ArchiveMetadata(
        name="simple.tar.gz",
        format="tar.gz",
        size_bytes=512,  # Single text file
        num_entries=1,
        category="benign",
        description="Simple TAR.GZ with one text file",
        purpose="Smoke test for tar.gz extraction",
    )

    # Adversarial cases
    PATH_TRAVERSAL_ZIP = ArchiveMetadata(
        name="traversal.zip",
        format="zip",
        size_bytes=200,
        num_entries=3,
        category="adversarial",
        description="Archive with path traversal attempts (../../../etc/passwd)",
        purpose="Verify path traversal protection in extraction",
        expected_issues=["path_traversal"],
    )

    SYMLINK_LOOP_TAR = ArchiveMetadata(
        name="symlink_loop.tar",
        format="tar",
        size_bytes=512,
        num_entries=2,
        category="adversarial",
        description="Archive with circular symlinks",
        purpose="Test handling of circular symlink detection",
        expected_issues=["symlink_loop"],
    )

    ZIP_BOMB = ArchiveMetadata(
        name="zip_bomb.zip",
        format="zip",
        size_bytes=1024,  # 1 KB compressed → 100 MB uncompressed
        num_entries=1,
        category="adversarial",
        description="ZIP bomb (highly compressible payload)",
        purpose="Verify extraction ratio limits are enforced",
        expected_issues=["ratio_limit_exceeded"],
    )

    # Edge cases
    LONG_PATHS_ZIP = ArchiveMetadata(
        name="long_paths.zip",
        format="zip",
        size_bytes=4096,
        num_entries=5,
        category="edge_case",
        description="Archive with paths near 255-char limit",
        purpose="Test long filename handling (POSIX limits)",
        expected_issues=["path_length"],
    )

    RESERVED_NAMES_ZIP = ArchiveMetadata(
        name="reserved_names.zip",
        format="zip",
        size_bytes=1024,
        num_entries=10,
        category="edge_case",
        description="Archive with Windows reserved names (CON, PRN, AUX, etc.)",
        purpose="Test Windows reserved name sanitization",
        expected_issues=["reserved_name"],
    )

    # Unicode and normalization
    UNICODE_NFD_TAR = ArchiveMetadata(
        name="unicode_nfd.tar",
        format="tar",
        size_bytes=2048,
        num_entries=3,
        category="unicode",
        description="Archive with NFD-normalized Unicode filenames",
        purpose="Test NFC/NFD normalization and collision detection",
        expected_issues=["unicode_normalization"],
    )

    UNICODE_BIDI_ZIP = ArchiveMetadata(
        name="unicode_bidi.zip",
        format="zip",
        size_bytes=1024,
        num_entries=2,
        category="unicode",
        description="Archive with bidirectional Unicode (RTL text)",
        purpose="Test Unicode directionality handling",
        expected_issues=["bidi_text"],
    )

    # Performance testing
    LARGE_COMPRESSIBLE_TAR_GZ = ArchiveMetadata(
        name="large_compressible.tar.gz",
        format="tar.gz",
        size_bytes=10 * 1024 * 1024,  # 10 MB compressed
        num_entries=1000,
        category="performance",
        description="Large archive with highly compressible content (patterns)",
        purpose="Benchmark extraction throughput",
        expected_issues=[],
    )

    DEEPLY_NESTED_ZIP = ArchiveMetadata(
        name="deeply_nested.zip",
        format="zip",
        size_bytes=8192,
        num_entries=100,  # 100 levels deep
        category="performance",
        description="Archive with deeply nested directory structure (100 levels)",
        purpose="Test handling of deep directory trees",
        expected_issues=["depth_limit"],
    )

    @classmethod
    def all(cls) -> list[ArchiveMetadata]:
        """Return all registered archive metadata."""
        return [
            cls.EMPTY_ZIP,
            cls.SIMPLE_TAR_GZ,
            cls.PATH_TRAVERSAL_ZIP,
            cls.SYMLINK_LOOP_TAR,
            cls.ZIP_BOMB,
            cls.LONG_PATHS_ZIP,
            cls.RESERVED_NAMES_ZIP,
            cls.UNICODE_NFD_TAR,
            cls.UNICODE_BIDI_ZIP,
            cls.LARGE_COMPRESSIBLE_TAR_GZ,
            cls.DEEPLY_NESTED_ZIP,
        ]

    @classmethod
    def by_category(cls, category: str) -> list[ArchiveMetadata]:
        """Filter archives by category."""
        return [arch for arch in cls.all() if arch.category == category]

    @classmethod
    def adversarial(cls) -> list[ArchiveMetadata]:
        """Return all adversarial test archives."""
        return cls.by_category("adversarial")

    @classmethod
    def benign(cls) -> list[ArchiveMetadata]:
        """Return all benign test archives."""
        return cls.by_category("benign")

    @classmethod
    def edge_cases(cls) -> list[ArchiveMetadata]:
        """Return all edge case archives."""
        return cls.by_category("edge_case")
