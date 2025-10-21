"""CLI subpackage for observability and operations commands."""

# Re-export from parent cli.py module (use importlib to avoid circular import)
import importlib.util
from pathlib import Path

from DocsToKG.OntologyDownload.cli.obs_cmd import app as obs_app
from DocsToKG.OntologyDownload.cli_main import _normalize_plan_args

# Import the parent cli.py module directly
cli_py_path = Path(__file__).parent.parent / "cli.py"
spec = importlib.util.spec_from_file_location("DocsToKG.OntologyDownload.cli_impl", cli_py_path)
if spec and spec.loader:
    cli_impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_impl)

    # Re-export common symbols
    EXAMPLE_SOURCES_YAML = cli_impl.EXAMPLE_SOURCES_YAML
    net = cli_impl.net
    cli_main = cli_impl.cli_main
    normalize_config_path = cli_impl.normalize_config_path
    _resolve_specs_from_args = cli_impl._resolve_specs_from_args
    run_validators = cli_impl.run_validators
    CONFIG_DIR = cli_impl.CONFIG_DIR

__all__ = [
    "obs_app",
    "_normalize_plan_args",
    "EXAMPLE_SOURCES_YAML",
    "net",
    "cli_main",
    "normalize_config_path",
    "_resolve_specs_from_args",
    "run_validators",
    "CONFIG_DIR",
    "Path",
]
