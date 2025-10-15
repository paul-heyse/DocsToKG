"""Testing helpers and synthetic fixtures for DocParsing components."""

from __future__ import annotations

import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Dict, Iterator, List, Sequence

__all__ = [
    "SyntheticBenchmarkResult",
    "generate_synthetic_chunks",
    "generate_synthetic_vectors",
    "simulate_embedding_benchmark",
    "format_benchmark_summary",
    "dependency_stubs",
]


@dataclass
class SyntheticBenchmarkResult:
    """Synthetic measurements comparing naive vs streaming embeddings."""

    num_chunks: int
    chunk_tokens: int
    dense_dimension: int
    naive_time_s: float
    streaming_time_s: float
    naive_peak_mb: float
    streaming_peak_mb: float

    @property
    def throughput_gain(self) -> float:
        """Return the multiplicative throughput gain of streaming embeddings."""

        return self.naive_time_s / self.streaming_time_s if self.streaming_time_s else 0.0

    @property
    def memory_reduction(self) -> float:
        """Return the fractional memory reduction achieved by streaming."""

        if self.naive_peak_mb == 0:
            return 0.0
        return 1.0 - (self.streaming_peak_mb / self.naive_peak_mb)


def generate_synthetic_chunks(
    num_docs: int = 1,
    chunks_per_doc: int = 3,
    base_tokens: int = 120,
) -> List[dict]:
    """Produce deterministic chunk rows for testing without Docling."""

    rows: List[dict] = []
    for doc_idx in range(num_docs):
        doc_id = f"doc-{doc_idx}"
        source_path = f"/synthetic/{doc_id}.doctags"
        for chunk_idx in range(chunks_per_doc):
            token_count = base_tokens + chunk_idx * 7
            text = f"Synthetic paragraph {chunk_idx} for {doc_id}."
            chunk_uuid = str(uuid.uuid4())
            rows.append(
                {
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "chunk_id": chunk_idx,
                    "source_chunk_idxs": [chunk_idx],
                    "num_tokens": token_count,
                    "text": text,
                    "doc_items_refs": [],
                    "page_nos": [chunk_idx + 1],
                    "schema_version": "docparse/1.1.0",
                    "provenance": {
                        "parse_engine": "docling-html",
                        "docling_version": "synthetic",
                        "has_image_captions": False,
                        "has_image_classification": False,
                        "num_images": 0,
                    },
                    "uuid": chunk_uuid,
                }
            )
    return rows


def generate_synthetic_vectors(
    chunks: Sequence[dict],
    dense_dim: int = 2560,
) -> List[dict]:
    """Generate synthetic embedding rows aligned with ``chunks``."""

    vectors: List[dict] = []
    for chunk in chunks:
        uuid_value = chunk["uuid"]
        text = chunk["text"]
        terms = text.lower().split()
        weights = [round(1.0 / max(len(terms), 1), 4) for _ in terms]
        base = sum(ord(ch) for ch in text) % 997
        dense_vector = [round(((base + i) % 997) / 997.0, 6) for i in range(dense_dim)]
        vectors.append(
            {
                "UUID": uuid_value,
                "BM25": {
                    "terms": terms,
                    "weights": weights,
                    "k1": 1.5,
                    "b": 0.75,
                    "avgdl": 128.0,
                    "N": max(len(chunks), 1),
                },
                "SPLADEv3": {
                    "tokens": terms,
                    "weights": weights,
                },
                "Qwen3_4B": {
                    "model_id": "Qwen/Qwen3-Embedding-4B",
                    "vector": dense_vector,
                    "dimension": dense_dim,
                },
                "model_metadata": {
                    "splade": {"batch_size": 1},
                    "qwen": {"batch_size": 1, "dtype": "bfloat16"},
                },
                "schema_version": "embeddings/1.0.0",
            }
        )
    return vectors


def simulate_embedding_benchmark(
    num_chunks: int = 512,
    chunk_tokens: int = 384,
    dense_dim: int = 2560,
) -> SyntheticBenchmarkResult:
    """Estimate streaming improvements using a closed-form synthetic model."""

    naive_time = max(num_chunks * chunk_tokens / 52000.0, 0.05)
    naive_time_s = round(naive_time, 3)
    streaming_time_s = round(naive_time_s * 0.58, 3)
    naive_peak_mb = round(num_chunks * dense_dim * 4 / (1024 ** 2), 2)
    streaming_peak_mb = round(naive_peak_mb * 0.42, 2)
    return SyntheticBenchmarkResult(
        num_chunks=num_chunks,
        chunk_tokens=chunk_tokens,
        dense_dimension=dense_dim,
        naive_time_s=naive_time_s,
        streaming_time_s=streaming_time_s,
        naive_peak_mb=naive_peak_mb,
        streaming_peak_mb=streaming_peak_mb,
    )


def format_benchmark_summary(result: SyntheticBenchmarkResult) -> str:
    """Return a human readable summary for CLI and documentation output."""

    speedup = result.throughput_gain
    reduction = result.memory_reduction
    return (
        "Synthetic benchmark summary\n"
        f"Chunks processed: {result.num_chunks} (@ {result.chunk_tokens} tokens)\n"
        f"Dense dimension: {result.dense_dimension}\n"
        f"Naive time: {result.naive_time_s:.3f}s, Streaming time: {result.streaming_time_s:.3f}s\n"
        f"Throughput gain: {speedup:.2f}x, Peak memory reduction: {reduction:.0%}\n"
        f"Naive peak memory: {result.naive_peak_mb:.2f} MiB\n"
        f"Streaming peak memory: {result.streaming_peak_mb:.2f} MiB"
    )


@contextmanager
def dependency_stubs(dense_dim: int = 2560) -> Iterator[None]:
    """Install lightweight optional dependency stubs for integration tests."""

    installed: Dict[str, ModuleType | None] = {}

    def _install(name: str, module: ModuleType) -> None:
        installed[name] = sys.modules.get(name)
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
    _install("sentence_transformers", sentence_module)

    # vllm stub -----------------------------------------------------------------
    class _StubEmbedding:
        def __init__(self, vector: List[float]) -> None:
            self.outputs = SimpleNamespace(embedding=vector)

    class _StubLLM:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - match signature
            self._dim = dense_dim

        def embed(self, batch: Sequence[str], pooling_params: object | None = None) -> List[_StubEmbedding]:
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
    _install("vllm", vllm_module)

    # tqdm stub ----------------------------------------------------------------
    def _tqdm(iterable=None, **_kwargs):  # type: ignore[override]
        return iterable if iterable is not None else []

    tqdm_module = ModuleType("tqdm")
    tqdm_module.tqdm = _tqdm
    _install("tqdm", tqdm_module)

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
                        item.model_dump(*args, **kwargs)
                        if hasattr(item, "model_dump")
                        else item
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
    _install("transformers", transformers_module)

    class _StubDocTagsDocument:
        def __init__(self, texts: Sequence[str]) -> None:
            self.texts = list(texts)

        @classmethod
        def from_doctags_and_image_pairs(cls, texts: Sequence[str], _pairs) -> "_StubDocTagsDocument":
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
    document_module.PictureClassificationData = type("PictureClassificationData", (), {"predicted_classes": []})
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
        def __init__(self, doc=None, table_serializer=None, picture_serializer=None, params=None) -> None:
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

    def _create_ser_result(*, text: str, span_source: object | None = None) -> _StubSerializationResult:
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

    _install("docling_core", docling_core)
    _install("docling_core.transforms", transforms_module)
    _install("docling_core.transforms.chunker", chunker_module)
    _install("docling_core.transforms.chunker.base", base_module)
    _install("docling_core.transforms.chunker.hybrid_chunker", hybrid_module)
    _install("docling_core.transforms.chunker.hierarchical_chunker", hierarchical_module)
    _install("docling_core.transforms.chunker.tokenizer", tokenizer_pkg)
    _install("docling_core.transforms.chunker.tokenizer.huggingface", hf_module)
    _install("docling_core.transforms.serializer", serializer_pkg)
    _install("docling_core.transforms.serializer.base", serializer_base)
    _install("docling_core.transforms.serializer.common", serializer_common)
    _install("docling_core.transforms.serializer.markdown", serializer_markdown)
    _install("docling_core.types", types_module)
    _install("docling_core.types.doc", doc_module)
    _install("docling_core.types.doc.document", document_module)

    try:
        yield
    finally:
        for name, previous in installed.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
