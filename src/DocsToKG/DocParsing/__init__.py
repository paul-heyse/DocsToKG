"""High-level facade exposing consolidated DocParsing modules and compatibility shims."""

from __future__ import annotations

import argparse
import os
import sys
import types
import warnings
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Callable, Iterable

from . import chunking as _chunking
from . import core as _core
from . import doctags as _doctags
from . import embedding as _embedding
from . import formats as _formats

__all__ = [
    "core",
    "formats",
    "doctags",
    "chunking",
    "embedding",
    "pipelines",
    "pdf_build_parser",
    "pdf_parse_args",
    "pdf_main",
    "html_build_parser",
    "html_parse_args",
    "html_main",
]

core = _core
formats = _formats
doctags = _doctags
chunking = _chunking
embedding = _embedding

# Legacy alias retained for backwards compatibility
pipelines = _doctags

pdf_build_parser = _doctags.pdf_build_parser
pdf_parse_args = _doctags.pdf_parse_args
pdf_main = _doctags.pdf_main
html_build_parser = _doctags.html_build_parser
html_parse_args = _doctags.html_parse_args
html_main = _doctags.html_main


# --- Compatibility Shim Infrastructure ---


def _populate_forwarding_module(module: types.ModuleType, target, *, doc_hint: str) -> None:
    """Populate ``module`` so that it forwards attributes to ``target``."""

    names: Iterable[str]
    if hasattr(target, "__all__"):
        names = getattr(target, "__all__")
    else:
        names = (name for name in dir(target) if not name.startswith("_"))

    for name in names:
        module.__dict__[name] = getattr(target, name)
    module.__dict__["__doc__"] = f"Compatibility shim: {doc_hint}"
    module.__dict__["__all__"] = list(
        getattr(target, "__all__", [name for name in dir(target) if not name.startswith("_")])
    )

    def __getattr__(attr: str):
        return getattr(target, attr)

    module.__getattr__ = __getattr__  # type: ignore[attr-defined]
    module.__dict__.setdefault("__file__", getattr(target, "__file__", __file__))


def _populate_cli_module(module: types.ModuleType) -> None:
    """Populate the legacy ``cli`` module shim."""

    _populate_forwarding_module(module, _core, doc_hint="cli → core")
    module.CommandHandler = _core.CommandHandler
    module.CLI_DESCRIPTION = _core.CLI_DESCRIPTION
    module.main = _core.main
    module.run_all = _core.run_all
    module.chunk = _core.chunk
    module.embed = _core.embed
    module.doctags = _core.doctags
    module._Command = _core._Command
    module.COMMANDS = _core.COMMANDS
    module.__all__ = [
        "CommandHandler",
        "CLI_DESCRIPTION",
        "main",
        "run_all",
        "chunk",
        "embed",
        "doctags",
    ]


def _populate_pdf_pipeline_module(module: types.ModuleType) -> None:
    """Populate the legacy ``pdf_pipeline`` module shim with a deprecation warning."""

    warnings.warn(
        "DocsToKG.DocParsing.pdf_pipeline is deprecated; use DocsToKG.DocParsing.doctags",
        DeprecationWarning,
        stacklevel=2,
    )

    backend = _doctags
    module.legacy_module = backend
    module.PREFERRED_PORT = backend.PREFERRED_PORT
    module.ProcessPoolExecutor = backend.ProcessPoolExecutor
    module.as_completed = backend.as_completed
    module.tqdm = backend.tqdm
    module.ensure_vllm = backend.ensure_vllm
    module.start_vllm = backend.start_vllm
    module.wait_for_vllm = backend.wait_for_vllm
    module.stop_vllm = backend.stop_vllm
    module.validate_served_models = backend.validate_served_models
    module.manifest_append = backend.manifest_append
    module.list_pdfs = backend.list_pdfs

    def parse_args(argv: object | None = None):
        """Legacy CLI argument parser for backwards-compatible imports."""
        return backend.pdf_parse_args(argv)

    parse_args.__module__ = module.__name__

    def main(args: object | None = None) -> int:
        """Legacy entry point delegating to :mod:`DocsToKG.DocParsing.doctags`."""
        if isinstance(args, argparse.Namespace):
            namespace = args
        else:
            namespace = parse_args() if args is None else parse_args(args)
        if not isinstance(namespace, argparse.Namespace):
            namespace = argparse.Namespace(**vars(namespace))
        if not hasattr(namespace, "vlm_prompt"):
            namespace.vlm_prompt = ""
        if not hasattr(namespace, "vlm_stop"):
            namespace.vlm_stop = []
        overrides = {
            "pdf_convert_one": module.__dict__.get("convert_one", backend.pdf_convert_one),
            "ProcessPoolExecutor": module.__dict__.get(
                "ProcessPoolExecutor", backend.ProcessPoolExecutor
            ),
            "as_completed": module.__dict__.get("as_completed", backend.as_completed),
            "tqdm": module.__dict__.get("tqdm", backend.tqdm),
        }
        originals = {name: getattr(backend, name) for name in overrides}
        try:
            for name, value in overrides.items():
                setattr(backend, name, value)
            return backend.pdf_main(namespace)
        finally:
            for name, original in originals.items():
                setattr(backend, name, original)

    main.__module__ = module.__name__

    module.parse_args = parse_args
    module.main = main
    module.convert_one = backend.pdf_convert_one
    module.PdfTask = backend.PdfTask
    module.PdfConversionResult = backend.PdfConversionResult
    module.__all__ = (
        "legacy_module",
        "PREFERRED_PORT",
        "ProcessPoolExecutor",
        "as_completed",
        "tqdm",
        "ensure_vllm",
        "start_vllm",
        "wait_for_vllm",
        "stop_vllm",
        "validate_served_models",
        "manifest_append",
        "list_pdfs",
        "parse_args",
        "main",
        "convert_one",
        "PdfTask",
        "PdfConversionResult",
    )


_SHIM_BUILDERS: dict[str, Callable[[types.ModuleType], None]] = {
    "_common": lambda module: _populate_forwarding_module(module, _core, doc_hint="_common → core"),
    "schemas": lambda module: _populate_forwarding_module(
        module, _formats, doc_hint="schemas → formats"
    ),
    "serializers": lambda module: _populate_forwarding_module(
        module, _formats, doc_hint="serializers → formats"
    ),
    "DoclingHybridChunkerPipelineWithMin": lambda module: _populate_forwarding_module(
        module, _chunking, doc_hint="DoclingHybridChunkerPipelineWithMin → chunking"
    ),
    "EmbeddingV2": lambda module: _populate_forwarding_module(
        module, _embedding, doc_hint="EmbeddingV2 → embedding"
    ),
    "pipelines": lambda module: _populate_forwarding_module(
        module, _doctags, doc_hint="pipelines → doctags"
    ),
    "cli": _populate_cli_module,
    "pdf_pipeline": _populate_pdf_pipeline_module,
}


class _DocParsingShimLoader(Loader):
    """Loader that populates deprecated DocParsing modules on demand."""

    def __init__(self, fullname: str, builder: Callable[[types.ModuleType], None]) -> None:
        """Store the module ``fullname`` and shim ``builder`` callback."""
        self._fullname = fullname
        self._builder = builder

    def create_module(self, spec: ModuleSpec) -> types.ModuleType:
        """Create a new module instance that will be populated by the shim."""
        module = types.ModuleType(spec.name)
        module.__loader__ = self
        module.__package__ = spec.parent
        return module

    def exec_module(
        self, module: types.ModuleType
    ) -> None:  # pragma: no cover - exercised via import tests
        """Execute the shim builder to populate ``module``."""
        self._builder(module)


class _DocParsingShimFinder(MetaPathFinder):
    """Meta-path finder that serves the compatibility shims defined above."""

    def find_spec(self, fullname: str, path, target=None):
        """Return a module spec when ``fullname`` matches a supported shim."""
        if not fullname.startswith(__name__ + "."):
            return None
        suffix = fullname.split(".", maxsplit=2)[-1]
        builder = _SHIM_BUILDERS.get(suffix)
        if builder is None:
            return None
        loader = _DocParsingShimLoader(fullname, builder)
        return ModuleSpec(fullname, loader, is_package=False)


if not os.getenv("DOCSTOKG_DOC_PARSING_DISABLE_SHIMS"):
    _finder = _DocParsingShimFinder()
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
