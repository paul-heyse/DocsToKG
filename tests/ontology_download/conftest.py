import sys
from types import SimpleNamespace


class _PlaceholderClient:
    def __init__(self, *args, **kwargs):  # pragma: no cover - should be patched in tests
        pass

    def __getattr__(self, item):  # pragma: no cover - should be replaced in tests
        raise RuntimeError(f"Accessed placeholder client attribute '{item}' without patching")


if "bioregistry" not in sys.modules:
    sys.modules["bioregistry"] = SimpleNamespace(
        get_obo_download=lambda prefix: None,
        get_owl_download=lambda prefix: None,
        get_rdf_download=lambda prefix: None,
    )

if "ols_client" not in sys.modules:
    sys.modules["ols_client"] = SimpleNamespace(OlsClient=_PlaceholderClient)

if "ontoportal_client" not in sys.modules:
    sys.modules["ontoportal_client"] = SimpleNamespace(BioPortalClient=_PlaceholderClient)
