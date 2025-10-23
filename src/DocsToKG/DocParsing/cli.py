"""
Legacy CLI entry point.

This module re-exports the unified Typer application so both
`docparse inspect dataset --dataset chunks` and the other DocParsing
commands remain discoverable when tooling expects `DocParsing.cli`.
"""

from DocsToKG.DocParsing.cli_unified import app

__all__ = ["app"]
