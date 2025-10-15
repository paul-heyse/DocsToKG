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
