# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.extraction_policy",
#   "purpose": "Configuration and validation for the 10 archive extraction hardening policies",
#   "sections": [
#     {"id": "policy", "name": "Extraction Policy", "anchor": "POL", "kind": "pydantic"},
#     {"id": "defaults", "name": "Defaults & Factory", "anchor": "DEF", "kind": "factory"},
#     {"id": "validation", "name": "Policy Validation", "anchor": "VAL", "kind": "validators"}
#   ]
# }
# === /NAVMAP ===

"""Configuration and validation for the 10 archive extraction hardening policies.

This module provides the ExtractionSettings Pydantic v2 model that defines all 10+ policies
with sensible, secure defaults. Each policy is independently configurable to support
different use cases while maintaining defense-in-depth security.

Benefits of Pydantic v2:
- Automatic field validation and coercion
- JSON schema generation for documentation
- Serialization to dict/JSON with schema conformance
- Type safety with runtime checking
- Provenance tracking via config_hash computation
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractionSettings(BaseModel):
    """Pydantic v2 model for all 10+ archive extraction hardening policies.

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
    - windows_portability_strict: Enforce Windows portability

    **Phase 3: Resource Budgets**
    - max_entries: Entry count budget
    - max_file_size_bytes: Per-file size limit
    - max_entry_ratio: Per-entry compression ratio
    - max_total_ratio: Total compression ratio

    **Phase 4: Permissions & Space**
    - check_disk_space: Verify space before extraction
    - preserve_permissions: Preserve file modes (or strip setuid/setgid)
    - dir_mode: Default directory mode
    - file_mode: Default file mode
    """

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",  # Reject unknown fields
    )

    # ========================================================================
    # PHASE 1: Foundation (Encapsulation + DirFD)
    # ========================================================================

    encapsulate: bool = Field(
        default=True,
        description="Enable single-root encapsulation to prevent tar-bomb attacks",
    )

    encapsulation_name: Literal["sha256", "basename"] = Field(
        default="sha256",
        description="Encapsulation root naming: sha256 (unique) or basename (human-readable)",
    )

    use_dirfd: bool = Field(
        default=True,
        description="Enable DirFD + openat semantics for TOCTOU race prevention",
    )

    # ========================================================================
    # PHASE 2: Pre-Scan Security (Links, Specials, Paths)
    # ========================================================================

    allow_symlinks: bool = Field(
        default=False,
        description="Allow symlink entries (must resolve within root)",
    )

    allow_hardlinks: bool = Field(
        default=False,
        description="Allow hardlink entries (must reference already-extracted file)",
    )

    max_depth: int = Field(
        default=32,
        ge=1,
        le=1000,
        description="Maximum path depth (prevents deeply nested path attacks)",
    )

    max_components_len: int = Field(
        default=240,
        ge=1,
        le=1000,
        description="Maximum bytes per path component (UTF-8 encoded)",
    )

    max_path_len: int = Field(
        default=4096,
        ge=1,
        le=65536,
        description="Maximum bytes for full path (UTF-8 encoded)",
    )

    normalize_unicode: Literal["NFC", "NFD", "none"] = Field(
        default="NFC",
        description="Unicode normalization: NFC (composed), NFD (decomposed), or none",
    )

    casefold_collision_policy: Literal["reject", "allow"] = Field(
        default="reject",
        description="Detect case-fold collisions: reject (fail) or allow (first wins)",
    )

    windows_portability_strict: bool = Field(
        default=True,
        description="Enforce Windows portability checks (reserved names, trailing space/dot)",
    )

    # ========================================================================
    # PHASE 3: Resource Budgets
    # ========================================================================

    max_entries: int = Field(
        default=50_000,
        ge=1,
        le=10_000_000,
        description="Maximum entry count (prevents extraction bombs)",
    )

    max_file_size_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,  # 2 GiB
        ge=1,
        le=1024 * 1024 * 1024 * 1024,  # 1 TB
        description="Maximum per-file size in bytes",
    )

    max_entry_ratio: float = Field(
        default=100.0,
        gt=0.0,
        le=10000.0,
        description="Maximum per-entry compression ratio (prevents zip-bomb at file level)",
    )

    max_total_ratio: float = Field(
        default=10.0,
        gt=0.0,
        le=10000.0,
        description="Maximum total compression ratio (prevents zip-bomb at archive level)",
    )

    # ========================================================================
    # PHASE 4: Permissions & Space
    # ========================================================================

    check_disk_space: bool = Field(
        default=True,
        description="Verify available disk space before extraction",
    )

    space_safety_margin: float = Field(
        default=1.10,
        ge=1.0,
        le=2.0,
        description="Safety margin for disk space checks (1.10 = 10% headroom)",
    )

    # File preallocation
    preallocate: bool = Field(
        default=True,
        description="Preallocate disk space for files (reduces fragmentation)",
    )

    preallocate_strategy: Literal["full", "none"] = Field(
        default="full",
        description="Preallocation strategy: full (posix_fallocate) or none",
    )

    # Adaptive buffer sizing
    copy_buffer_min: int = Field(
        default=64 * 1024,  # 64 KiB
        ge=1024,
        le=1024 * 1024,
        description="Minimum copy buffer size in bytes",
    )

    copy_buffer_max: int = Field(
        default=1024 * 1024,  # 1 MiB
        ge=1024,
        le=100 * 1024 * 1024,
        description="Maximum copy buffer size in bytes",
    )

    # Atomic writes & fsync discipline
    atomic: bool = Field(
        default=True,
        description="Enable atomic write discipline (temp → fsync → rename → dirfsync)",
    )

    group_fsync: int = Field(
        default=32,
        ge=1,
        le=10000,
        description="Directory fsync frequency (every N files)",
    )

    # Inline hashing
    hash_enable: bool = Field(
        default=True,
        description="Enable inline hashing during extraction",
    )

    hash_algorithms: list[str] = Field(
        default=["sha256"],
        description="Hash algorithms to compute (e.g., sha256, sha512)",
    )

    hash_mode: Literal["inline", "parallel"] = Field(
        default="inline",
        description="Hashing mode: inline (during write) or parallel",
    )

    hash_parallel_threads: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Number of threads for parallel hashing",
    )

    # Selective extraction (include/exclude patterns)
    include_globs: list[str] = Field(
        default_factory=list,
        description="Include patterns (glob syntax); empty = include all",
    )

    exclude_globs: list[str] = Field(
        default_factory=list,
        description="Exclude patterns (glob syntax); empty = exclude none",
    )

    report_skipped: bool = Field(
        default=True,
        description="Report skipped entries in telemetry",
    )

    # CPU guard (wall-time limit)
    max_wall_time_seconds: int = Field(
        default=120,
        ge=1,
        le=3600,
        description="Maximum wall-time limit in seconds",
    )

    cpu_guard_action: Literal["abort", "warn"] = Field(
        default="abort",
        description="Action on timeout: abort (fail) or warn (log and continue)",
    )

    # Permissions enforcement (Phase 4)
    preserve_permissions: bool = Field(
        default=False,
        description="Preserve file permissions from archive (or use defaults)",
    )

    file_mode: int = Field(
        default=0o644,
        description="Default file mode (octal, e.g., 0o644)",
    )

    dir_mode: int = Field(
        default=0o755,
        description="Default directory mode (octal, e.g., 0o755)",
    )

    # ========================================================================
    # Correctness & Integrity
    # ========================================================================

    integrity_verify: bool = Field(
        default=True,
        description="Enable CRC/checksum verification",
    )

    integrity_fail_on_mismatch: bool = Field(
        default=True,
        description="Fail extraction if integrity check fails",
    )

    # Timestamp policy
    timestamp_mode: Literal["preserve", "normalize", "source_date_epoch"] = Field(
        default="preserve",
        description="Timestamp handling: preserve (from archive), normalize, or source_date_epoch",
    )

    timestamp_normalize_to: Literal["archive_mtime", "now"] = Field(
        default="archive_mtime",
        description="When normalizing timestamps: use archive mtime or current time",
    )

    timestamp_preserve_dir_mtime: bool = Field(
        default=False,
        description="Preserve directory mtimes (or leave as extraction time)",
    )

    # Format allow-list
    allowed_formats: list[str] = Field(
        default=["zip", "tar", "ustar", "pax", "gnutar"],
        description="Allowed archive formats (empty = no restrictions)",
    )

    allowed_filters: list[str] = Field(
        default=["none", "gzip", "bzip2", "xz", "zstd"],
        description="Allowed compression filters (empty = no restrictions)",
    )

    # Entry ordering
    deterministic_order: Literal["header", "path_asc"] = Field(
        default="header",
        description="Output ordering: header (archive order) or path_asc (lexicographic)",
    )

    # Duplicate entry policy
    duplicate_policy: Literal["reject", "first_wins", "last_wins"] = Field(
        default="reject",
        description="Duplicate handling: reject (fail), first_wins, or last_wins",
    )

    # Provenance manifest
    manifest_emit: bool = Field(
        default=True,
        description="Write .extract.audit.json manifest",
    )

    manifest_filename: str = Field(
        default=".extract.audit.json",
        description="Audit manifest filename",
    )

    manifest_include_digest: bool = Field(
        default=True,
        description="Include file digests in audit manifest",
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @field_validator("max_path_len")
    @classmethod
    def validate_max_path_len(cls, v: int, info) -> int:
        """Ensure max_path_len >= max_components_len."""
        if "max_components_len" in info.data:
            if v < info.data["max_components_len"]:
                raise ValueError(
                    f"max_path_len ({v}) must be >= max_components_len ({info.data['max_components_len']})"
                )
        return v

    @field_validator("copy_buffer_max")
    @classmethod
    def validate_copy_buffer_max(cls, v: int, info) -> int:
        """Ensure copy_buffer_max >= copy_buffer_min."""
        if "copy_buffer_min" in info.data:
            if v < info.data["copy_buffer_min"]:
                raise ValueError(
                    f"copy_buffer_max ({v}) must be >= copy_buffer_min ({info.data['copy_buffer_min']})"
                )
        return v

    @field_validator("use_dirfd")
    @classmethod
    def validate_use_dirfd(cls, v: bool, info) -> bool:
        """use_dirfd requires encapsulate=True."""
        if v and "encapsulate" in info.data and not info.data["encapsulate"]:
            raise ValueError("use_dirfd requires encapsulate=True")
        return v

    @field_validator("file_mode", "dir_mode", mode="before")
    @classmethod
    def validate_mode(cls, v: int | str) -> int:
        """Accept octal strings or int file modes."""
        if isinstance(v, str):
            return int(v, 8)
        if not isinstance(v, int):
            raise ValueError(f"Mode must be int or octal string, got {type(v)}")
        if v <= 0 or v > 0o777:
            raise ValueError(f"Mode must be in range [0o001, 0o777], got {oct(v)}")
        return v

    # ========================================================================
    # METHODS
    # ========================================================================

    def model_dump_minimal(self) -> dict[str, Any]:
        """Dump only non-default fields for config_hash computation.

        Returns:
            Dictionary with only customized fields (for provenance tracking).
        """
        defaults = ExtractionSettings()
        result = {}
        for field_name, field_value in self:
            default_value = getattr(defaults, field_name)
            if field_value != default_value:
                result[field_name] = field_value
        return result

    def summary(self) -> dict[str, str]:
        """Get a human-readable summary of all policies.

        Returns:
            Dictionary mapping policy name to status/value.
        """
        return {
            "Phase 1: Encapsulation": (
                f"enabled (policy={self.encapsulation_name})" if self.encapsulate else "disabled"
            ),
            "Phase 1: DirFD": "enabled" if self.use_dirfd else "disabled",
            "Phase 2: Symlinks": "rejected" if not self.allow_symlinks else "allowed",
            "Phase 2: Hardlinks": "rejected" if not self.allow_hardlinks else "allowed",
            "Phase 2: Path Depth": f"max={self.max_depth}",
            "Phase 2: Component Length": f"max={self.max_components_len} bytes",
            "Phase 2: Full Path Length": f"max={self.max_path_len} bytes",
            "Phase 2: Unicode Normalization": self.normalize_unicode,
            "Phase 2: Case-Fold Collisions": self.casefold_collision_policy,
            "Phase 2: Windows Portability": (
                "strict" if self.windows_portability_strict else "lenient"
            ),
            "Phase 3: Max Entries": f"{self.max_entries:,}",
            "Phase 3: Max File Size": f"{self.max_file_size_bytes / (1024**3):.1f} GiB",
            "Phase 3: Max Entry Ratio": f"{self.max_entry_ratio}:1",
            "Phase 3: Max Total Ratio": f"{self.max_total_ratio}:1",
            "Phase 4: Preserve Permissions": "yes" if self.preserve_permissions else "no",
            "Phase 4: Dir Mode": oct(self.dir_mode),
            "Phase 4: File Mode": oct(self.file_mode),
            "Phase 4: Check Disk Space": "yes" if self.check_disk_space else "no",
            "Deterministic Ordering": self.deterministic_order,
            "Manifest Emission": "yes" if self.manifest_emit else "no",
        }


# For backward compatibility with old code referencing ExtractionPolicy
ExtractionPolicy = ExtractionSettings


def safe_defaults() -> ExtractionSettings:
    """Factory for safe extraction settings defaults.

    Returns a policy with all protections enabled and conservative limits.
    """
    return ExtractionSettings()


def lenient_defaults() -> ExtractionSettings:
    """Factory for lenient extraction settings (less restrictive).

    Disables some defenses; use only for trusted archives.
    """
    return ExtractionSettings(
        allow_symlinks=True,
        allow_hardlinks=True,
        max_entries=1_000_000,
        max_file_size_bytes=100 * 1024 * 1024 * 1024,  # 100 GiB
        max_entry_ratio=1000.0,
        max_total_ratio=100.0,
    )


def strict_defaults() -> ExtractionSettings:
    """Factory for strict extraction settings (maximum protection).

    Minimal limits; use for untrusted archives from adversarial sources.
    """
    return ExtractionSettings(
        allow_symlinks=False,
        allow_hardlinks=False,
        max_depth=16,
        max_components_len=128,
        max_path_len=2048,
        max_entries=10_000,
        max_file_size_bytes=100 * 1024 * 1024,  # 100 MiB
        max_entry_ratio=50.0,
        max_total_ratio=5.0,
        normalize_unicode="NFC",
        casefold_collision_policy="reject",
        preserve_permissions=False,
    )
