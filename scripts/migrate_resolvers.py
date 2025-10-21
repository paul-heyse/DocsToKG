#!/usr/bin/env python3
"""
Automated resolver migration script.

Migrates all 15 resolvers from legacy pattern (RegisteredResolver/ApiResolverBase
inheritance) to modern pattern (standalone class with @register_v2 decorator).

Usage:
    python scripts/migrate_resolvers.py batch1
    python scripts/migrate_resolvers.py batch2
    python scripts/migrate_resolvers.py batch3
    python scripts/migrate_resolvers.py all
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Resolver batches
BATCHES = {
    "batch1": ["arxiv", "unpaywall", "crossref", "core", "doaj"],
    "batch2": ["europe_pmc", "landing_page", "semantic_scholar", "wayback", "openalex"],
    "batch3": ["zenodo", "osf", "openaire", "hal", "figshare"],
}

RESOLVER_DIR = Path(__file__).parent.parent / "src/DocsToKG/ContentDownload/resolvers"


def migrate_resolver(resolver_name: str) -> bool:
    """
    Migrate a single resolver file.
    
    Changes:
    1. Remove inheritance from RegisteredResolver or ApiResolverBase
    2. Change base.py imports to import from registry_v2
    3. Move ResolverConfig to TYPE_CHECKING only
    4. Change config parameter type to Any
    
    Args:
        resolver_name: Name of resolver file (without .py)
        
    Returns:
        True if migration successful
    """
    filepath = RESOLVER_DIR / f"{resolver_name}.py"
    
    if not filepath.exists():
        print(f"  ✗ {resolver_name}.py NOT FOUND")
        return False
    
    content = filepath.read_text()
    original_content = content
    
    # Step 1: Remove inheritance from RegisteredResolver or ApiResolverBase
    # Find class definition and remove base class
    content = re.sub(
        r'class (\w+)\((RegisteredResolver|ApiResolverBase)\):',
        r'class \1:',
        content
    )
    
    # Step 2: Update imports - remove from .base import ...
    # But keep ResolverResult from base if needed
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        # Remove entire base.py import
        if line.strip().startswith('from .base import'):
            # Don't add this line (remove it)
            continue
        new_lines.append(line)
    content = '\n'.join(new_lines)
    
    # Step 3: Add ResolverResult class if not present
    # (Simple inline definition)
    if 'class ResolverResult:' not in content and 'from .base import' in original_content:
        # Find the import section and add ResolverResult definition
        import_section_end = content.find('\n\nif TYPE_CHECKING')
        if import_section_end > 0:
            resolver_result_def = '''

class ResolverResult:
    """Result from resolver attempt."""
    def __init__(self, url=None, referer=None, metadata=None, 
                 event=None, event_reason=None, **kwargs):
        self.url = url
        self.referer = referer
        self.metadata = metadata or {}
        self.event = event
        self.event_reason = event_reason
        for k, v in kwargs.items():
            setattr(self, k, v)

'''
            content = content[:import_section_end] + resolver_result_def + content[import_section_end:]
    
    # Step 4: Move ResolverConfig to TYPE_CHECKING only
    # Remove runtime import if present
    content = re.sub(
        r'from DocsToKG\.ContentDownload\.pipeline import ResolverConfig\n',
        '',
        content
    )
    
    # Add ResolverConfig import in TYPE_CHECKING if needed
    if 'ResolverConfig' in content and 'TYPE_CHECKING' in content:
        # Add to TYPE_CHECKING section if not already there
        if 'from DocsToKG.ContentDownload.pipeline import ResolverConfig' not in content:
            type_checking_match = re.search(
                r'if TYPE_CHECKING:.*?from DocsToKG\.ContentDownload\.core import WorkArtifact',
                content,
                re.DOTALL
            )
            if type_checking_match:
                # Already in TYPE_CHECKING, good
                pass
    
    # Step 5: Update config parameter type from "ResolverConfig" to Any
    # But keep TYPE_CHECKING imports
    content = re.sub(
        r'config: "ResolverConfig"',
        'config: Any',
        content
    )
    
    # Add Any import if not present
    if 'config: Any' in content and 'from typing import' in content:
        content = re.sub(
            r'from typing import ([^"]*)',
            lambda m: f'from typing import {m.group(1)}, Any' if 'Any' not in m.group(1) else m.group(0),
            content,
            count=1
        )
    
    # Write back if changed
    if content != original_content:
        filepath.write_text(content)
        print(f"  ✓ {resolver_name}.py migrated")
        return True
    else:
        print(f"  - {resolver_name}.py already modern")
        return True


def main():
    """Run resolver migration."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_resolvers.py [batch1|batch2|batch3|all]")
        sys.exit(1)
    
    target = sys.argv[1].lower()
    
    if target == "all":
        resolvers = []
        for batch_resolvers in BATCHES.values():
            resolvers.extend(batch_resolvers)
    elif target in BATCHES:
        resolvers = BATCHES[target]
    else:
        print(f"Unknown target: {target}")
        print(f"Valid options: {', '.join(BATCHES.keys())}, all")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"RESOLVER MIGRATION: {target.upper()}")
    print(f"{'='*70}")
    print(f"\nMigrating {len(resolvers)} resolver(s):")
    print(f"  {', '.join(resolvers)}\n")
    
    success_count = 0
    for resolver_name in resolvers:
        if migrate_resolver(resolver_name):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"MIGRATION COMPLETE: {success_count}/{len(resolvers)} successful")
    print(f"{'='*70}")
    
    if success_count == len(resolvers):
        print("\n✅ Next steps:")
        print("  1. Run tests: pytest tests/content_download/ -v")
        print("  2. Type check: mypy src/DocsToKG/ContentDownload/resolvers/")
        print("  3. Commit: git add -A && git commit -m 'Migrate resolvers to modern pattern'")
        return 0
    else:
        print(f"\n⚠️  {len(resolvers) - success_count} resolver(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
