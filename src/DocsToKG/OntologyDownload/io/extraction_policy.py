# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.extraction_policy",
#   "purpose": "Configuration and validation for the 10 archive extraction hardening policies",
#   "sections": [
#     {"id": "policy", "name": "Extraction Policy", "anchor": "POL", "kind": "dataclass"},
#     {"id": "defaults", "name": "Defaults & Factory", "anchor": "DEF", "kind": "factory"},
#     {"id": "validation", "name": "Policy Validation", "anchor": "VAL", "kind": "methods"}
#   ]
# }
# === /NAVMAP ===

"""Configuration and validation for the 10 archive extraction hardening policies.

This module provides the ExtractionPolicy dataclass that defines all 10 policies
with sensible, secure defaults. Each policy is independently configurable to support
different use cases while maintaining defense-in-depth security.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ExtractionPolicy:
    """Configuration for all 10 archive extraction hardening policies.

    Default-deny security posture: all policies enabled by default, opt-in for risky features.

    Policies (4 phases):

    **Phase 1: Foundation**
    - encapsulate: Single-root encapsulation (anti-tarbomb)
    - encapsulation_name: Naming policy (sha256 | basename)
    - use_dirfd: DirFD + openat semantics (race-free)

    **Phase 2: Pre-Scan Security**
    - allow_symlinks: Reject symlink entries
    - allow_hardlinks: Reject hardlink entries
    - max_depth: Path depth limit
    - max_components_len: Max bytes per path component
    - max_path_len: Max bytes for full path
    - normalize_unicode: Unicode normalization (NFC | NFD)
    - casefold_collision_policy: Detect case-fold duplicates (reject | allow)

    **Phase 3: Resource Budgets**
    - max_entries: Entry count budget
    - max_file_size_bytes: Per-file size limit
    - max_entry_ratio: Per-entry compression ratio

    **Phase 4: Permissions & Space**
    - preserve_permissions: Preserve file modes (or strip setuid/setgid)
    - dir_mode: Default directory mode
    - file_mode: Default file mode
    - check_disk_space: Verify space before extraction
    """

    # ========================================================================
    # PHASE 1: Foundation (Encapsulation + DirFD)
    # ========================================================================

    encapsulate: bool = True
    """Enable single-root encapsulation (default: True).

    When True, all extracted files go into a deterministic subdirectory
    of `destination`, preventing tar-bomb-style extraction into sibling dirs.
    """

    encapsulation_name: Literal["sha256", "basename"] = "sha256"
    """Encapsulation root naming policy.

    - "sha256": Use first 12 chars of archive SHA256 digest (reproducible, unique)
    - "basename": Use archive filename (human-readable, not unique)
    """

    use_dirfd: bool = True
    """Enable DirFD + openat semantics (default: True).

    When True, uses O_PATH + openat() for all filesystem operations on the
    encapsulation root, eliminating TOCTOU races and symlink-in-parent attacks.
    """

    # ========================================================================
    # PHASE 2: Pre-Scan Security (Links, Specials, Paths)
    # ========================================================================

    allow_symlinks: bool = False
    """Allow symlink entries (default: False).

    When False, symlinks are rejected during pre-scan.
    When True, symlinks are allowed but must resolve within root.
    """

    allow_hardlinks: bool = False
    """Allow hardlink entries (default: False).

    When False, hardlinks are rejected during pre-scan.
    When True, hardlinks are allowed if target is already extracted.
    """

    max_depth: int = 32
    """Maximum path depth (default: 32).

    Number of directory levels deep. Prevents deeply nested path attacks.
    """

    max_components_len: int = 240
    """Maximum bytes per path component (default: 240).

    After UTF-8 encoding. Most filesystems support 255; we reserve margin.
    """

    max_path_len: int = 4096
    """Maximum bytes for full path (default: 4096).

    After UTF-8 encoding. Most systems support 4096; we cap here.
    """

    normalize_unicode: Literal["NFC", "NFD", "none"] = "NFC"
    """Unicode normalization for paths (default: "NFC").

    - "NFC": Normalize to composed form (standard)
    - "NFD": Normalize to decomposed form
    - "none": No normalization
    """

    casefold_collision_policy: Literal["reject", "allow"] = "reject"
    """Policy for case-fold collisions (default: "reject").

    Detects paths that differ only in case (e.g., "file.txt" and "FILE.txt").
    - "reject": Fail extraction if collision detected
    - "allow": Allow both (first wins on case-insensitive filesystems)
    """

    # ========================================================================
    # PHASE 3: Resource Budgets
    # ========================================================================

    max_entries: int = 50_000
    """Maximum entry count (default: 50,000).

    Prevents extraction bombs with millions of tiny files consuming inodes.
    """

    max_file_size_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB
    """Maximum per-file size (default: 2 GiB).

    Enforced both on declared size (pre-scan) and streamed size (extract).
    """

    max_entry_ratio: float = 100.0
    """Maximum per-entry compression ratio (default: 100:1).

    If provided by libarchive, detects entries with extreme compression.
    Prevents zip-bomb attacks at per-file granularity.
    """

    # ========================================================================
    # PHASE 4: Permissions & Space
    # ========================================================================

    preserve_permissions: bool = False
    """Preserve file modes from archive (default: False).

    When False, setuid/setgid/sticky bits are stripped; dir_mode and
    file_mode are applied instead.
    When True, archive modes are preserved (except setuid/setgid).
    """

    dir_mode: int = 0o755
    """Default directory mode when not preserving (default: 0o755)."""

    file_mode: int = 0o644
    """Default file mode when not preserving (default: 0o644)."""

    check_disk_space: bool = True
    """Check disk space before extraction (default: True).

    Verifies `statfs(destination).f_bavail >= total_uncompressed * 1.1`.
    """

    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================

    def validate(self) -> list[str]:
        """Validate policy configuration.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []

        # Phase 1
        if self.encapsulation_name not in ("sha256", "basename"):
            errors.append(
                f"encapsulation_name must be 'sha256' or 'basename', got '{self.encapsulation_name}'"
            )
        if self.use_dirfd and not self.encapsulate:
            errors.append("use_dirfd requires encapsulate=True")

        # Phase 2
        if self.max_depth <= 0:
            errors.append(f"max_depth must be > 0, got {self.max_depth}")
        if self.max_components_len <= 0:
            errors.append(f"max_components_len must be > 0, got {self.max_components_len}")
        if self.max_path_len <= 0:
            errors.append(f"max_path_len must be > 0, got {self.max_path_len}")
        if self.max_path_len < self.max_components_len:
            errors.append(
                f"max_path_len ({self.max_path_len}) must be >= max_components_len ({self.max_components_len})"
            )
        if self.normalize_unicode not in ("NFC", "NFD", "none"):
            errors.append(
                f"normalize_unicode must be 'NFC', 'NFD', or 'none', got '{self.normalize_unicode}'"
            )
        if self.casefold_collision_policy not in ("reject", "allow"):
            errors.append(
                f"casefold_collision_policy must be 'reject' or 'allow', got '{self.casefold_collision_policy}'"
            )

        # Phase 3
        if self.max_entries <= 0:
            errors.append(f"max_entries must be > 0, got {self.max_entries}")
        if self.max_file_size_bytes <= 0:
            errors.append(f"max_file_size_bytes must be > 0, got {self.max_file_size_bytes}")
        if self.max_entry_ratio <= 0:
            errors.append(f"max_entry_ratio must be > 0, got {self.max_entry_ratio}")

        # Phase 4
        if self.dir_mode <= 0 or self.dir_mode > 0o777:
            errors.append(f"dir_mode must be in range [0o001, 0o777], got {oct(self.dir_mode)}")
        if self.file_mode <= 0 or self.file_mode > 0o777:
            errors.append(f"file_mode must be in range [0o001, 0o777], got {oct(self.file_mode)}")

        return errors

    def is_valid(self) -> bool:
        """Check if policy is valid."""
        return len(self.validate()) == 0

    def summary(self) -> dict[str, str]:
        """Get a human-readable summary of all enabled policies.

        Returns:
            Dictionary mapping policy name to status (enabled/disabled).
        """
        return {
            "Phase 1: Encapsulation": f"enabled (policy={self.encapsulation_name})"
            if self.encapsulate
            else "disabled",
            "Phase 1: DirFD": "enabled" if self.use_dirfd else "disabled",
            "Phase 2: Symlinks": "rejected" if not self.allow_symlinks else "allowed",
            "Phase 2: Hardlinks": "rejected" if not self.allow_hardlinks else "allowed",
            "Phase 2: Path Depth": f"max={self.max_depth}",
            "Phase 2: Component Length": f"max={self.max_components_len} bytes",
            "Phase 2: Full Path Length": f"max={self.max_path_len} bytes",
            "Phase 2: Unicode Normalization": self.normalize_unicode,
            "Phase 2: Case-Fold Collisions": self.casefold_collision_policy,
            "Phase 3: Max Entries": f"{self.max_entries:,}",
            "Phase 3: Max File Size": f"{self.max_file_size_bytes / (1024**3):.1f} GiB",
            "Phase 3: Max Entry Ratio": f"{self.max_entry_ratio}:1",
            "Phase 4: Preserve Permissions": "yes" if self.preserve_permissions else "no",
            "Phase 4: Dir Mode": oct(self.dir_mode),
            "Phase 4: File Mode": oct(self.file_mode),
            "Phase 4: Check Disk Space": "yes" if self.check_disk_space else "no",
        }


def safe_defaults() -> ExtractionPolicy:
    """Factory for safe extraction policy defaults.

    Returns a policy with all protections enabled and conservative limits.
    """
    return ExtractionPolicy()


def lenient_defaults() -> ExtractionPolicy:
    """Factory for lenient extraction policy (less restrictive).

    Disables some defenses; use only for trusted archives.
    """
    return ExtractionPolicy(
        allow_symlinks=True,
        allow_hardlinks=True,
        max_entries=1_000_000,
        max_file_size_bytes=100 * 1024 * 1024 * 1024,  # 100 GiB
        max_entry_ratio=1000.0,
    )


def strict_defaults() -> ExtractionPolicy:
    """Factory for strict extraction policy (maximum protection).

    Minimal limits; use for untrusted archives from adversarial sources.
    """
    return ExtractionPolicy(
        allow_symlinks=False,
        allow_hardlinks=False,
        max_depth=16,
        max_components_len=128,
        max_path_len=2048,
        max_entries=10_000,
        max_file_size_bytes=100 * 1024 * 1024,  # 100 MiB
        max_entry_ratio=50.0,
        normalize_unicode="NFC",
        casefold_collision_policy="reject",
        preserve_permissions=False,
    )
