# === NAVMAP v1 ===
# {
#   "module": "scripts.generate_config_schema",
#   "purpose": "Utility script for generate config schema workflows",
#   "sections": [
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Generate JSON Schema for ontology downloader configuration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
# --- Globals ---

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_PATH = SRC_ROOT / "DocsToKG" / "OntologyDownload" / "config.py"

if str(SRC_ROOT) not in sys.path:  # pragma: no cover - runtime path fix
    sys.path.insert(0, str(SRC_ROOT))

spec = importlib.util.spec_from_file_location("ontology_config", CONFIG_PATH)
if spec is None or spec.loader is None:  # pragma: no cover - defensive
    raise RuntimeError("Unable to load ontology configuration module")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
ResolvedConfig = config_module.ResolvedConfig
# --- Public Functions ---
# --- Module Entry Points ---


def main() -> None:
    """Generate schema file for documentation and tooling."""

    schema = ResolvedConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Ontology Downloader Configuration"
    schema["description"] = "Schema for DocsToKG ontology downloader sources.yaml"

    output_path = Path("docs/schemas/ontology-downloader-config.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2))

    print(f"Schema written to {output_path}")
    print(f"Schema has {len(schema.get('properties', {}))} root properties")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
