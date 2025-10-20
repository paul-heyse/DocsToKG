# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_api_about",
#   "purpose": "Validate DocsToKG.OntologyDownload.api.about metadata contract",
#   "sections": []
# }
# === /NAVMAP ===

"""Tests covering the ``DocsToKG.OntologyDownload.api.about`` helper."""

from __future__ import annotations

from DocsToKG.OntologyDownload import api


def test_about_includes_detailed_plugin_metadata() -> None:
    """``about`` should surface qualified paths and versions for plugins."""

    metadata = api.about()
    assert "plugins" in metadata

    plugins = metadata["plugins"]
    for kind in ("resolver", "validator"):
        assert kind in plugins
        inventory = plugins[kind]
        assert inventory, f"expected non-empty plugin inventory for {kind}"
        for name, details in inventory.items():
            assert isinstance(name, str) and name
            assert set(details) == {"qualified", "version"}
            assert isinstance(details["qualified"], str)
            assert isinstance(details["version"], str)
