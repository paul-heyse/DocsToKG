#!/usr/bin/env python3
"""Generate a local intersphinx inventory for httpx.

This script spins up a tiny throwaway Sphinx project that indexes the
installed ``httpx`` package, then copies the resulting ``objects.inv`` into
``docs/_static/inventories/httpx`` so the main docs build can reference it
locally.

Usage:
    uv run python docs/scripts/fetch_httpx_inventory.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="httpx-inventory-"))
    try:
        docs_dir = tmp_dir / "docs"
        docs_dir.mkdir(parents=True)

        # Minimal conf.py
        (docs_dir / "conf.py").write_text(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "extensions = [\n"
            "    'sphinx.ext.autodoc',\n"
            "    'sphinx.ext.autosummary',\n"
            "    'sphinx.ext.napoleon',\n"
            "]\n"
            "autosummary_generate = True\n"
            "autodoc_typehints = 'description'\n"
            "html_theme = 'alabaster'\n",
            encoding="utf-8",
        )

        # Single page that pulls in the entire httpx namespace
        (docs_dir / "index.rst").write_text(
            "HTTPX API\n========\n\n"
            ".. autosummary::\n"
            "   :toctree: _auto\n"
            "   :recursive:\n\n"
            "   httpx\n",
            encoding="utf-8",
        )

        # Build the inventory
        build_dir = tmp_dir / "_build" / "html"
        subprocess.check_call(
            ["sphinx-build", "-b", "html", str(docs_dir), str(build_dir)]
        )

        # Copy into docs/_static/inventories/httpx
        target_dir = Path("docs/_static/inventories/httpx")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(build_dir / "objects.inv", target_dir / "objects.inv")
        print(f"Copied inventory to {target_dir / 'objects.inv'}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

