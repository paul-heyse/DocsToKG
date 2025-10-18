# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.stubs",
#   "purpose": "Pytest coverage for docparsing stubs scenarios",
#   "sections": [
#     {
#       "id": "dependency-stubs",
#       "name": "dependency_stubs",
#       "anchor": "function-dependency-stubs",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""DocParsing dependency stubs used by integration tests."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Sequence

# --- Globals ---

__all__ = ["dependency_stubs"]


def dependency_stubs(dense_dim: int = 2560) -> None:
    """Install lightweight optional dependency stubs for integration tests."""

    def _install(name: str, module: ModuleType, *, force: bool = False) -> None:
        """Register a stub module when missing or when forcing replacement."""

        if not force and name in sys.modules:
            return
        sys.modules[name] = module

    # sentence_transformers stub -------------------------------------------------
    class _StubSparseValues:
        def __init__(self, count: int) -> None:
            self._count = count

        def numel(self) -> int:
            return self._count

    class _StubSparseRow:
        def __init__(self, tokens: List[str], weights: List[float]) -> None:
            self.tokens = tokens
            self.weights = weights

        def coalesce(self) -> "_StubSparseRow":
            return self

        def values(self) -> _StubSparseValues:
            return _StubSparseValues(len(self.tokens))

    class _StubSparseBatch:
        def __init__(self, rows: List[tuple[List[str], List[float]]]) -> None:
            self._rows = [_StubSparseRow(tokens, weights) for tokens, weights in rows]

        @property
        def shape(self) -> tuple[int, int]:
            if not self._rows:
                return (0, 0)
            width = max((len(row.tokens) for row in self._rows), default=0)
            return (len(self._rows), width)

        def __getitem__(self, index: int) -> _StubSparseRow:
            return self._rows[index]

    class _StubSparseEncoder:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature compatibility
            pass

        def encode(self, texts: Sequence[str]) -> _StubSparseBatch:
            rows: List[tuple[List[str], List[float]]] = []
            for text in texts:
                tokens = text.lower().split()
                if not tokens:
                    tokens = ["synthetic"]
                weights = [float(len(tok)) / max(len(tokens), 1) for tok in tokens]
                rows.append((tokens, weights))
            return _StubSparseBatch(rows)

        def decode(self, row: _StubSparseRow, top_k: int) -> List[tuple[str, float]]:
            pairs = list(zip(row.tokens, row.weights))
            return pairs[:top_k]

    sentence_module = ModuleType("sentence_transformers")
    sentence_module.SparseEncoder = _StubSparseEncoder
    _install("sentence_transformers", sentence_module, force=True)

    # vllm stub -----------------------------------------------------------------
    class _StubEmbedding:
        def __init__(self, vector: List[float]) -> None:
            self.outputs = SimpleNamespace(embedding=vector)

    class _StubLLM:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - match signature
            self._dim = dense_dim

        def embed(
            self, batch: Sequence[str], pooling_params: object | None = None
        ) -> List[_StubEmbedding]:
            outputs: List[_StubEmbedding] = []
            for text in batch:
                base = sum(ord(ch) for ch in text) % 997
                vector = [((base + i) % 997) / 997.0 for i in range(self._dim)]
                outputs.append(_StubEmbedding(vector))
            return outputs

    class _StubPoolingParams:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            self.normalize = kwargs.get("normalize", False)

    vllm_module = ModuleType("vllm")
    vllm_module.LLM = _StubLLM
    vllm_module.PoolingParams = _StubPoolingParams
    _install("vllm", vllm_module, force=True)

    # tqdm stub ----------------------------------------------------------------
    def _tqdm(iterable=None, **_kwargs):  # type: ignore[override]
        return iterable if iterable is not None else []

    tqdm_module = ModuleType("tqdm")
    tqdm_module.tqdm = _tqdm
    _install("tqdm", tqdm_module, force=True)

    # pydantic stub -----------------------------------------------------------
    _FIELD_UNSET = object()

    class _FieldInfo:
        def __init__(self, *, default=_FIELD_UNSET, default_factory=None) -> None:
            self.default = default
            self.default_factory = default_factory

    def _stub_field(*args, default=_FIELD_UNSET, default_factory=None, **_kwargs) -> _FieldInfo:
        if args:
            default = args[0]
        return _FieldInfo(default=default, default_factory=default_factory)

    class _PydanticMeta(type):
        def __new__(mcls, name, bases, namespace):
            field_defaults: Dict[str, object] = {}
            for key, value in list(namespace.items()):
                if isinstance(value, _FieldInfo):
                    if value.default_factory is not None:
                        default_value = value.default_factory()
                    elif value.default is not _FIELD_UNSET:
                        default_value = value.default
                    else:
                        default_value = None
                    namespace[key] = default_value
                    field_defaults[key] = default_value
            namespace.setdefault("__fields_defaults__", {})
            namespace["__fields_defaults__"].update(field_defaults)
            return super().__new__(mcls, name, bases, namespace)

    class _StubBaseModel(metaclass=_PydanticMeta):
        model_config: Dict[str, object] = {}

        def __init__(self, **kwargs) -> None:
            data = dict(getattr(self, "__fields_defaults__", {}))
            data.update(kwargs)
            for key, value in data.items():
                setattr(self, key, value)

        @classmethod
        def model_rebuild(cls) -> None:  # pragma: no cover - stub compatibility
            return None

        def model_dump(self, *args, **kwargs) -> Dict[str, object]:
            result: Dict[str, object] = {}
            for key, value in self.__dict__.items():
                if hasattr(value, "model_dump"):
                    result[key] = value.model_dump(*args, **kwargs)
                elif isinstance(value, list):
                    result[key] = [
                        item.model_dump(*args, **kwargs) if hasattr(item, "model_dump") else item
                        for item in value
                    ]
                else:
                    result[key] = value
            return result

    def _stub_field_validator(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def _stub_model_validator(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def _config_dict(**kwargs) -> Dict[str, object]:
        return kwargs

    pydantic_module = ModuleType("pydantic")
    pydantic_module.BaseModel = _StubBaseModel
    pydantic_module.Field = _stub_field
    pydantic_module.ConfigDict = _config_dict
    pydantic_module.field_validator = _stub_field_validator
    pydantic_module.model_validator = _stub_model_validator
    _install("pydantic", pydantic_module)

    # transformers + docling stubs ---------------------------------------------
    class _StubAutoTokenizer:
        def __init__(self, model_name: str, use_fast: bool = True) -> None:
            self.model_name = model_name
            self.use_fast = use_fast

        @classmethod
        def from_pretrained(cls, model_name: str, use_fast: bool = True):
            return cls(model_name, use_fast=use_fast)

        def __call__(self, text: str) -> List[str]:
            return text.split()

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = _StubAutoTokenizer
    _install("transformers", transformers_module, force=True)

    class _StubDocTagsDocument:
        def __init__(self, texts: Sequence[str]) -> None:
            self.texts = list(texts)

        @classmethod
        def from_doctags_and_image_pairs(
            cls, texts: Sequence[str], images=None, **_kwargs
        ) -> "_StubDocTagsDocument":
            return cls(texts)

    class _StubDoclingDocument:
        def __init__(self, paragraphs: Sequence[str], name: str) -> None:
            self.paragraphs = list(paragraphs)
            self.name = name

        @classmethod
        def load_from_doctags(cls, doc_tags: _StubDocTagsDocument, document_name: str):
            joined = "\n".join(doc_tags.texts)
            parts = [p.strip() for p in joined.split("\n\n") if p.strip()]
            if not parts:
                parts = [joined.strip() or "Synthetic paragraph"]
            return cls(parts, document_name)

        def __iter__(self):
            return iter(self.paragraphs)

    class _StubBaseChunk:
        def __init__(self, text: str) -> None:
            self.text = text
            self.meta = SimpleNamespace(doc_items=[], prov=[])

    class _StubHuggingFaceTokenizer:
        def __init__(self, tokenizer: _StubAutoTokenizer, max_tokens: int) -> None:
            self.tokenizer = tokenizer
            self.max_tokens = max_tokens

        def count_tokens(self, text: str) -> int:
            return max(1, len(self.tokenizer(text)))

    class _StubHybridChunker:
        def __init__(self, tokenizer: _StubHuggingFaceTokenizer, **_kwargs) -> None:
            self.tokenizer = tokenizer

        def chunk(self, dl_doc: _StubDoclingDocument):
            return [_StubBaseChunk(text) for text in dl_doc.paragraphs]

        def contextualize(self, chunk: _StubBaseChunk) -> str:
            return chunk.text

    docling_core = ModuleType("docling_core")
    transforms_module = ModuleType("docling_core.transforms")
    chunker_module = ModuleType("docling_core.transforms.chunker")
    base_module = ModuleType("docling_core.transforms.chunker.base")
    hybrid_module = ModuleType("docling_core.transforms.chunker.hybrid_chunker")
    hierarchical_module = ModuleType("docling_core.transforms.chunker.hierarchical_chunker")
    tokenizer_pkg = ModuleType("docling_core.transforms.chunker.tokenizer")
    hf_module = ModuleType("docling_core.transforms.chunker.tokenizer.huggingface")
    serializer_pkg = ModuleType("docling_core.transforms.serializer")
    serializer_base = ModuleType("docling_core.transforms.serializer.base")
    serializer_common = ModuleType("docling_core.transforms.serializer.common")
    serializer_markdown = ModuleType("docling_core.transforms.serializer.markdown")
    types_module = ModuleType("docling_core.types")
    doc_module = ModuleType("docling_core.types.doc")
    document_module = ModuleType("docling_core.types.doc.document")

    docling_core.transforms = transforms_module
    transforms_module.chunker = chunker_module
    chunker_module.base = base_module
    base_module.BaseChunk = _StubBaseChunk
    chunker_module.hybrid_chunker = hybrid_module
    hybrid_module.HybridChunker = _StubHybridChunker
    chunker_module.hierarchical_chunker = hierarchical_module
    chunker_module.tokenizer = tokenizer_pkg
    tokenizer_pkg.huggingface = hf_module
    hf_module.HuggingFaceTokenizer = _StubHuggingFaceTokenizer
    chunker_module.serializer = serializer_pkg
    serializer_pkg.base = serializer_base
    serializer_pkg.common = serializer_common
    serializer_pkg.markdown = serializer_markdown
    docling_core.types = types_module
    types_module.doc = doc_module
    doc_module.document = document_module
    document_module.DoclingDocument = _StubDoclingDocument
    document_module.DocTagsDocument = _StubDocTagsDocument
    document_module.PictureClassificationData = type(
        "PictureClassificationData", (), {"predicted_classes": []}
    )
    document_module.PictureDescriptionData = type("PictureDescriptionData", (), {"text": ""})
    document_module.PictureMoleculeData = type("PictureMoleculeData", (), {"smi": ""})

    class _StubPictureItem:
        def __init__(self) -> None:
            self.annotations = []

        def caption_text(self, _doc: object) -> str:
            return ""

    document_module.PictureItem = _StubPictureItem

    class _StubSerializationResult(SimpleNamespace):
        pass

    class _StubBaseDocSerializer:
        def post_process(self, text: str) -> str:
            return text

    class _StubChunkingDocSerializer(_StubBaseDocSerializer):
        def __init__(
            self, doc=None, table_serializer=None, picture_serializer=None, params=None
        ) -> None:
            self.doc = doc
            self.table_serializer = table_serializer
            self.picture_serializer = picture_serializer
            self.params = params

    class _StubChunkingSerializerProvider:
        def get_serializer(self, doc: object) -> _StubChunkingDocSerializer:
            return _StubChunkingDocSerializer(doc=doc)

    hierarchical_module.ChunkingDocSerializer = _StubChunkingDocSerializer
    hierarchical_module.ChunkingSerializerProvider = _StubChunkingSerializerProvider

    serializer_base.BaseDocSerializer = _StubBaseDocSerializer
    serializer_base.SerializationResult = _StubSerializationResult

    def _create_ser_result(
        *, text: str, span_source: object | None = None
    ) -> _StubSerializationResult:
        return _StubSerializationResult(text=text, span_source=span_source)

    serializer_common.create_ser_result = _create_ser_result

    class _StubMarkdownParams:
        def __init__(self, image_placeholder: str = "") -> None:
            self.image_placeholder = image_placeholder

    class _StubMarkdownPictureSerializer:
        pass

    class _StubMarkdownTableSerializer:
        pass

    serializer_markdown.MarkdownParams = _StubMarkdownParams
    serializer_markdown.MarkdownPictureSerializer = _StubMarkdownPictureSerializer
    serializer_markdown.MarkdownTableSerializer = _StubMarkdownTableSerializer

    _install("docling_core", docling_core, force=True)
    _install("docling_core.transforms", transforms_module, force=True)
    _install("docling_core.transforms.chunker", chunker_module, force=True)
    _install("docling_core.transforms.chunker.base", base_module, force=True)
    _install("docling_core.transforms.chunker.hybrid_chunker", hybrid_module, force=True)
    _install(
        "docling_core.transforms.chunker.hierarchical_chunker", hierarchical_module, force=True
    )
    _install("docling_core.transforms.chunker.tokenizer", tokenizer_pkg, force=True)
    _install("docling_core.transforms.chunker.tokenizer.huggingface", hf_module, force=True)
    _install("docling_core.transforms.serializer", serializer_pkg, force=True)
    _install("docling_core.transforms.serializer.base", serializer_base, force=True)
    _install("docling_core.transforms.serializer.common", serializer_common, force=True)
    _install("docling_core.transforms.serializer.markdown", serializer_markdown, force=True)
    _install("docling_core.types", types_module, force=True)
    _install("docling_core.types.doc", doc_module, force=True)
    _install("docling_core.types.doc.document", document_module, force=True)
