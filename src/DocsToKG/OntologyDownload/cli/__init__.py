"""CLI subpackage for observability and operations commands."""

from DocsToKG.OntologyDownload.cli import _normalize_plan_args as _normalize_plan_args_func
from DocsToKG.OntologyDownload.cli.obs_cmd import app as obs_app

__all__ = ["obs_app", "_normalize_plan_args_func"]
