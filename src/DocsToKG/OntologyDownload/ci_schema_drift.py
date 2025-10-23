# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.ci_schema_drift",
#   "purpose": "CI schema drift detection for OntologyDownloadSettings.",
#   "sections": [
#     {
#       "id": "normalize-schema",
#       "name": "_normalize_schema",
#       "anchor": "function-normalize-schema",
#       "kind": "function"
#     },
#     {
#       "id": "check-schema-drift",
#       "name": "check_schema_drift",
#       "anchor": "function-check-schema-drift",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""CI schema drift detection for OntologyDownloadSettings.

This module provides utilities for detecting schema drift in CI/CD pipelines:
- Generate schemas and compare against committed versions
- Detect breaking vs non-breaking changes
- Provide detailed change reports
- Fail build on unexpected drift

Used in GitHub Actions to ensure schema consistency across commits.

Example:
    >>> from DocsToKG.OntologyDownload.ci_schema_drift import check_schema_drift
    >>> exit_code = check_schema_drift(
    ...     expected_dir=Path("docs/schemas/"),
    ...     fail_on_drift=True
    ... )
"""

import json
import sys
from difflib import unified_diff
from pathlib import Path

from DocsToKG.OntologyDownload.settings_schema import (
    generate_settings_schema,
    generate_submodel_schemas,
)


def _normalize_schema(schema: dict) -> str:
    """Normalize schema to deterministic JSON string.

    Args:
        schema: JSON schema dictionary

    Returns:
        Normalized JSON string with sorted keys
    """
    return json.dumps(schema, indent=2, sort_keys=True)


def check_schema_drift(
    expected_dir: Path = None,
    fail_on_drift: bool = True,
    verbose: bool = False,
) -> int:
    """Check for schema drift between generated and committed schemas.

    Compares newly generated schemas against committed versions in the
    expected directory. Reports any changes and can optionally fail the
    build if unexpected drift is detected.

    Args:
        expected_dir: Directory containing committed schemas (docs/schemas/)
        fail_on_drift: If True, exit with 1 on any drift detected
        verbose: Print detailed diff output

    Returns:
        Exit code: 0 if no drift, 1 if drift detected and fail_on_drift=True

    Example:
        >>> exit_code = check_schema_drift(
        ...     expected_dir=Path("docs/schemas/"),
        ...     fail_on_drift=True,
        ...     verbose=True
        ... )
        >>> sys.exit(exit_code)
    """
    if expected_dir is None:
        # Use docs/schemas/ relative to workspace root
        expected_dir = Path.cwd() / "docs" / "schemas"

    expected_dir = expected_dir.resolve()

    print("🔍 Checking schema drift...")
    print(f"   Expected dir: {expected_dir}")
    print()

    # Generate current schemas
    print("📝 Generating current schemas...")
    current_top = generate_settings_schema()
    current_sub = generate_submodel_schemas()

    # Load expected schemas
    print("📋 Loading committed schemas...")
    expected_top = None
    expected_sub: dict[str, dict] = {}

    top_schema_path = expected_dir / "settings.schema.json"
    if top_schema_path.exists():
        expected_top = json.loads(top_schema_path.read_text())
    else:
        print(f"   ⚠️  Top-level schema not found: {top_schema_path}")

    for domain in [
        "http",
        "cache",
        "retry",
        "logging",
        "telemetry",
        "security",
        "ratelimit",
        "extraction",
        "storage",
        "duckdb",
    ]:
        sub_path = expected_dir / f"settings.{domain}.subschema.json"
        if sub_path.exists():
            expected_sub[domain] = json.loads(sub_path.read_text())

    print(f"   Loaded {len(expected_sub)} submodel schemas")
    print()

    # Compare schemas
    print("🔄 Comparing schemas...")
    drifts: list[tuple[str, str, str]] = []  # (schema_name, expected, current)

    # Check top-level schema
    if expected_top is not None:
        expected_str = _normalize_schema(expected_top)
        current_str = _normalize_schema(current_top)

        if expected_str != current_str:
            drifts.append(("settings.schema.json", expected_str, current_str))

    # Check submodel schemas
    for domain, current_schema in current_sub.items():
        schema_name = f"settings.{domain}.subschema.json"

        if domain in expected_sub:
            expected_str = _normalize_schema(expected_sub[domain])
            current_str = _normalize_schema(current_schema)

            if expected_str != current_str:
                drifts.append((schema_name, expected_str, current_str))
        else:
            print(f"   ⚠️  Missing expected schema: {schema_name}")

    print()

    # Report results
    if not drifts:
        print("✅ No schema drift detected")
        print()
        return 0

    # Drift detected
    print(f"❌ Schema drift detected in {len(drifts)} file(s):")
    print()

    for schema_name, expected_str, current_str in drifts:
        print(f"📄 {schema_name}:")
        print()

        if verbose:
            # Show unified diff
            diff_lines = unified_diff(
                expected_str.splitlines(keepends=True),
                current_str.splitlines(keepends=True),
                fromfile=f"{schema_name} (expected)",
                tofile=f"{schema_name} (current)",
                lineterm="",
            )
            for line in diff_lines:
                print(f"   {line}", end="")
            print()
        else:
            # Just show summary
            expected_lines = len(expected_str.splitlines())
            current_lines = len(current_str.splitlines())
            print(f"   Expected: {expected_lines} lines")
            print(f"   Current:  {current_lines} lines")
            print("   Run with --verbose to see detailed diff")
            print()

    print("⚠️  Schema drift indicates changes to settings structure.")
    print("   This is expected when adding/removing settings fields.")
    print("   Review changes and commit updated schemas if intentional.")
    print()

    if fail_on_drift:
        print("❌ Build failed due to schema drift (--fail-on-drift=true)")
        return 1
    else:
        print("⚠️  Schema drift detected but build continuing (--fail-on-drift=false)")
        return 0


def main() -> int:
    """CLI entry point for schema drift detection.

    Usage:
        python -m DocsToKG.OntologyDownload.ci_schema_drift [OPTIONS]

    Options:
        --expected-dir PATH: Directory with committed schemas (default: docs/schemas/)
        --fail-on-drift: Exit with 1 if drift detected (default: True)
        --no-fail-on-drift: Continue even if drift detected
        --verbose: Show detailed diff output
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Check for schema drift in CI/CD pipeline",
        prog="ci-schema-drift",
    )
    parser.add_argument(
        "--expected-dir",
        type=Path,
        default=Path("docs/schemas"),
        help="Directory with committed schemas (default: docs/schemas/)",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        default=True,
        help="Exit with 1 if drift detected (default: True)",
    )
    parser.add_argument(
        "--no-fail-on-drift",
        action="store_false",
        dest="fail_on_drift",
        help="Continue even if drift detected",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed diff output",
    )

    args = parser.parse_args()

    return check_schema_drift(
        expected_dir=args.expected_dir,
        fail_on_drift=args.fail_on_drift,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
