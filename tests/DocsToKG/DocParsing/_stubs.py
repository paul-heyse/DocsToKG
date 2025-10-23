"""Test stubs for optional third-party dependencies used by DocParsing CLI."""

from __future__ import annotations

import sys
import types


def install_docling_stubs() -> None:
    """Install lightweight stubs for optional docling dependencies."""

    docling_core = types.ModuleType("docling_core")
    transforms_pkg = types.ModuleType("docling_core.transforms")
    chunker_pkg = types.ModuleType("docling_core.transforms.chunker")
    base_mod = types.ModuleType("docling_core.transforms.chunker.base")
    hybrid_mod = types.ModuleType("docling_core.transforms.chunker.hybrid_chunker")
    hierarchical_mod = types.ModuleType("docling_core.transforms.chunker.hierarchical_chunker")
    tokenizer_pkg = types.ModuleType("docling_core.transforms.chunker.tokenizer")
    hf_mod = types.ModuleType("docling_core.transforms.chunker.tokenizer.huggingface")
    serializer_pkg = types.ModuleType("docling_core.transforms.serializer")
    serializer_base = types.ModuleType("docling_core.transforms.serializer.base")
    serializer_common = types.ModuleType("docling_core.transforms.serializer.common")
    serializer_markdown = types.ModuleType("docling_core.transforms.serializer.markdown")
    types_pkg = types.ModuleType("docling_core.types")
    doc_pkg = types.ModuleType("docling_core.types.doc")
    document_mod = types.ModuleType("docling_core.types.doc.document")

    class _BaseChunk:  # pragma: no cover - simple stub
        pass

    class _HybridChunker:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class _ChunkingDocSerializer:  # pragma: no cover - simple stub
        pass

    class _ChunkingSerializerProvider:  # pragma: no cover - simple stub
        pass

    class _HuggingFaceTokenizer:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class _DoclingDocument:  # pragma: no cover - simple stub
        pass

    class _DocTagsDocument:  # pragma: no cover - simple stub
        pass

    class _PictureItem:  # pragma: no cover - simple stub
        pass

    class _PictureClassificationData:  # pragma: no cover - simple stub
        pass

    class _PictureDescriptionData:  # pragma: no cover - simple stub
        pass

    class _PictureMoleculeData:  # pragma: no cover - simple stub
        pass

    class _BaseDocSerializer:  # pragma: no cover - simple stub
        pass

    class _SerializationResult:  # pragma: no cover - simple stub
        pass

    class _MarkdownParams:  # pragma: no cover - simple stub
        pass

    class _MarkdownPictureSerializer:  # pragma: no cover - simple stub
        pass

    class _MarkdownTableSerializer:  # pragma: no cover - simple stub
        pass

    base_mod.BaseChunk = _BaseChunk
    hybrid_mod.HybridChunker = _HybridChunker
    hierarchical_mod.ChunkingDocSerializer = _ChunkingDocSerializer
    hierarchical_mod.ChunkingSerializerProvider = _ChunkingSerializerProvider
    hf_mod.HuggingFaceTokenizer = _HuggingFaceTokenizer
    document_mod.DoclingDocument = _DoclingDocument
    document_mod.DocTagsDocument = _DocTagsDocument
    document_mod.PictureItem = _PictureItem
    document_mod.PictureClassificationData = _PictureClassificationData
    document_mod.PictureDescriptionData = _PictureDescriptionData
    document_mod.PictureMoleculeData = _PictureMoleculeData
    serializer_base.BaseDocSerializer = _BaseDocSerializer
    serializer_base.SerializationResult = _SerializationResult
    serializer_common.create_ser_result = lambda *args, **kwargs: _SerializationResult()
    serializer_markdown.MarkdownParams = _MarkdownParams
    serializer_markdown.MarkdownPictureSerializer = _MarkdownPictureSerializer
    serializer_markdown.MarkdownTableSerializer = _MarkdownTableSerializer

    docling_core.transforms = transforms_pkg  # type: ignore[attr-defined]
    transforms_pkg.chunker = chunker_pkg  # type: ignore[attr-defined]
    chunker_pkg.base = base_mod  # type: ignore[attr-defined]
    chunker_pkg.hybrid_chunker = hybrid_mod  # type: ignore[attr-defined]
    chunker_pkg.hierarchical_chunker = hierarchical_mod  # type: ignore[attr-defined]
    chunker_pkg.tokenizer = tokenizer_pkg  # type: ignore[attr-defined]
    tokenizer_pkg.huggingface = hf_mod  # type: ignore[attr-defined]
    transforms_pkg.serializer = serializer_pkg  # type: ignore[attr-defined]
    serializer_pkg.base = serializer_base  # type: ignore[attr-defined]
    serializer_pkg.common = serializer_common  # type: ignore[attr-defined]
    serializer_pkg.markdown = serializer_markdown  # type: ignore[attr-defined]
    docling_core.types = types_pkg  # type: ignore[attr-defined]
    types_pkg.doc = doc_pkg  # type: ignore[attr-defined]
    doc_pkg.document = document_mod  # type: ignore[attr-defined]

    sys.modules.setdefault("docling_core", docling_core)
    sys.modules.setdefault("docling_core.transforms", transforms_pkg)
    sys.modules.setdefault("docling_core.transforms.chunker", chunker_pkg)
    sys.modules.setdefault("docling_core.transforms.chunker.base", base_mod)
    sys.modules.setdefault("docling_core.transforms.chunker.hybrid_chunker", hybrid_mod)
    sys.modules.setdefault("docling_core.transforms.chunker.hierarchical_chunker", hierarchical_mod)
    sys.modules.setdefault("docling_core.transforms.chunker.tokenizer", tokenizer_pkg)
    sys.modules.setdefault(
        "docling_core.transforms.chunker.tokenizer.huggingface", hf_mod
    )
    sys.modules.setdefault("docling_core.transforms.serializer", serializer_pkg)
    sys.modules.setdefault("docling_core.transforms.serializer.base", serializer_base)
    sys.modules.setdefault("docling_core.transforms.serializer.common", serializer_common)
    sys.modules.setdefault("docling_core.transforms.serializer.markdown", serializer_markdown)
    sys.modules.setdefault("docling_core.types", types_pkg)
    sys.modules.setdefault("docling_core.types.doc", doc_pkg)
    sys.modules.setdefault("docling_core.types.doc.document", document_mod)


def install_third_party_stubs() -> None:
    """Install stubbed versions of optional third-party modules."""

    pyarrow_mod = types.ModuleType("pyarrow")
    parquet_mod = types.ModuleType("pyarrow.parquet")

    class _FakeTable:  # pragma: no cover - simple stub
        pass

    def _unavailable(*_args, **_kwargs):  # pragma: no cover - simple stub
        raise RuntimeError("Parquet operations are unavailable in tests")

    parquet_mod.ParquetWriter = _FakeTable  # type: ignore[attr-defined]
    parquet_mod.write_table = _unavailable  # type: ignore[attr-defined]
    pyarrow_mod.parquet = parquet_mod  # type: ignore[attr-defined]

    transformers_mod = types.ModuleType("transformers")

    class _AutoTokenizer:  # pragma: no cover - simple stub
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    transformers_mod.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]

    sys.modules.setdefault("pyarrow", pyarrow_mod)
    sys.modules.setdefault("pyarrow.parquet", parquet_mod)
    sys.modules.setdefault("transformers", transformers_mod)


def install_all_stubs() -> None:
    """Install all stub modules required for CLI tests."""

    install_docling_stubs()
    install_third_party_stubs()


__all__ = [
    "install_all_stubs",
    "install_docling_stubs",
    "install_third_party_stubs",
]
