# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_atomic_writes",
#   "purpose": "Pytest coverage for content download atomic writes scenarios",
#   "sections": [
#     {
#       "id": "_stub_module",
#       "name": "_stub_module",
#       "anchor": "SM",
#       "kind": "function"
#     },
#     {
#       "id": "dummy_tokenizer",
#       "name": "DummyTokenizer",
#       "anchor": "DUMM",
#       "kind": "class"
#     },
#     {
#       "id": "dummy_hybrid_chunker",
#       "name": "DummyHybridChunker",
#       "anchor": "DUMM1",
#       "kind": "class"
#     },
#     {
#       "id": "configure_chunker_stubs",
#       "name": "configure_chunker_stubs",
#       "anchor": "CCS",
#       "kind": "function"
#     },
#     {
#       "id": "crashing_atomic_write",
#       "name": "crashing_atomic_write",
#       "anchor": "CAW",
#       "kind": "function"
#     },
#     {
#       "id": "prepare_data_root",
#       "name": "prepare_data_root",
#       "anchor": "PDR",
#       "kind": "function"
#     },
#     {
#       "id": "chunker_args",
#       "name": "chunker_args",
#       "anchor": "CA",
#       "kind": "function"
#     },
#     {
#       "id": "embeddings_args",
#       "name": "embeddings_args",
#       "anchor": "EA",
#       "kind": "function"
#     },
#     {
#       "id": "write_dummy_doctags",
#       "name": "write_dummy_doctags",
#       "anchor": "WDD",
#       "kind": "function"
#     },
#     {
#       "id": "write_dummy_chunks",
#       "name": "write_dummy_chunks",
#       "anchor": "WDC",
#       "kind": "function"
#     },
#     {
#       "id": "configure_embeddings_stubs",
#       "name": "configure_embeddings_stubs",
#       "anchor": "CES",
#       "kind": "function"
#     },
#     {
#       "id": "test_chunker_failure_leaves_no_partial_files",
#       "name": "test_chunker_failure_leaves_no_partial_files",
#       "anchor": "TCFLN",
#       "kind": "function"
#     },
#     {
#       "id": "test_embeddings_failure_cleans_temporary_files",
#       "name": "test_embeddings_failure_cleans_temporary_files",
#       "anchor": "TEFCT",
#       "kind": "function"
#     },
#     {
#       "id": "test_chunker_success_outputs_readable_file",
#       "name": "test_chunker_success_outputs_readable_file",
#       "anchor": "TCSOR",
#       "kind": "function"
#     },
#     {
#       "id": "test_chunker_promotes_image_metadata",
#       "name": "test_chunker_promotes_image_metadata",
#       "anchor": "TCPIM",
#       "kind": "function"
#     },
#     {
#       "id": "test_chunker_resume_after_failure",
#       "name": "test_chunker_resume_after_failure",
#       "anchor": "TCRAF",
#       "kind": "function"
#     },
#     {
#       "id": "test_chunker_concurrent_writes_isolated",
#       "name": "test_chunker_concurrent_writes_isolated",
#       "anchor": "TCCWI",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Atomic write behaviour tests for downloader and doc parsing pipelines."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List

import pytest

DOWNLOAD_DEPS_AVAILABLE = True
try:  # pragma: no cover - optional dependency wire-up
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised via skip
    requests = None  # type: ignore[assignment]
    DOWNLOAD_DEPS_AVAILABLE = False
else:  # pragma: no branch - simple fallback
    try:
        import pyalex  # type: ignore  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover - exercised via skip
        DOWNLOAD_DEPS_AVAILABLE = False

DOWNLOAD_TESTS_SKIP_REASON = "requests and pyalex required for content download atomic write tests"

if DOWNLOAD_DEPS_AVAILABLE:
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact, download_candidate

    class _DummyHeadResponse:
        status_code = 200
        headers = {"Content-Type": "application/pdf"}

        def close(self) -> None:  # pragma: no cover - no side effects
            return

    class _BaseDummyResponse:
        def __init__(self, status_code: int = 200, headers: Dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self.headers: Dict[str, str] = {"Content-Type": "application/pdf"}
            if headers:
                self.headers.update(headers)

        def __enter__(self) -> "_BaseDummyResponse":  # noqa: D401 - context manager protocol
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - context manager protocol
            return None

        def close(self) -> None:  # pragma: no cover - no resources to release
            return

    class _FailingResponse(_BaseDummyResponse):
        def iter_content(self, chunk_size: int):  # noqa: D401 - streaming interface
            yield b"%PDF-1.4\n"
            raise requests.exceptions.ChunkedEncodingError("simulated failure")  # type: ignore[arg-type]

    class _SuccessfulResponse(_BaseDummyResponse):
        def __init__(self, payload: bytes, headers: Dict[str, str] | None = None) -> None:
            super().__init__(status_code=200, headers=headers)
            self._payload = payload

        def iter_content(self, chunk_size: int):  # noqa: D401 - streaming interface
            yield self._payload

    class _DummySession:
        def __init__(self, response: _BaseDummyResponse) -> None:
            self._response = response

        def head(self, url: str, **kwargs: Any) -> _DummyHeadResponse:  # noqa: D401
            return _DummyHeadResponse()

        def get(self, url: str, **kwargs: Any) -> _BaseDummyResponse:  # noqa: D401
            return self._response

    def _make_artifact(tmp_path: Path) -> WorkArtifact:
        pdf_dir = tmp_path / "pdfs"
        html_dir = tmp_path / "html"
        pdf_dir.mkdir()
        html_dir.mkdir()
        return WorkArtifact(
            work_id="W-atomic",
            title="Atomic Test",
            publication_year=2024,
            doi="10.1234/atomic",
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="atomic",
            pdf_dir=pdf_dir,
            html_dir=html_dir,
        )

    def _download_with_session(
        session: _DummySession, tmp_path: Path
    ) -> tuple[WorkArtifact, Path, Dict[str, Dict[str, Any]]]:
        artifact = _make_artifact(tmp_path)
        context: Dict[str, Dict[str, Any]] = {"previous": {}}
        outcome = download_candidate(
            session,
            artifact,
            "https://example.org/test.pdf",
            referer=None,
            timeout=5.0,
            context=context,
        )
        return artifact, artifact.pdf_dir / "atomic.pdf", context, outcome

    @pytest.mark.skipif(not DOWNLOAD_DEPS_AVAILABLE, reason=DOWNLOAD_TESTS_SKIP_REASON)
    def test_partial_download_leaves_part_file(tmp_path: Path) -> None:
        session = _DummySession(_FailingResponse())
        artifact, final_path, _, outcome = _download_with_session(session, tmp_path)

        part_path = final_path.with_suffix(".pdf.part")
        assert outcome.classification == "request_error"
        assert not final_path.exists()
        assert part_path.exists()

    @pytest.mark.skipif(not DOWNLOAD_DEPS_AVAILABLE, reason=DOWNLOAD_TESTS_SKIP_REASON)
    def test_successful_download_records_digest(tmp_path: Path) -> None:
        body = b"A" * 1500
        payload = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n" + body + b"\n%%EOF\n"
        expected_sha = hashlib.sha256(payload).hexdigest()
        session = _DummySession(_SuccessfulResponse(payload))

        artifact, final_path, _, outcome = _download_with_session(session, tmp_path)

        assert outcome.classification == "pdf"
        assert outcome.sha256 == expected_sha
        assert outcome.content_length == len(payload)
        assert final_path.exists()
        assert not final_path.with_suffix(".pdf.part").exists()
        assert final_path.read_bytes() == payload


chunker_manifest_log: List[dict] = []
embeddings_manifest_log: List[dict] = []


def _stub_module(
    name: str, *, package: bool = False, attrs: Dict[str, object] | None = None
) -> ModuleType:
    """Create a lightweight module stub registered in ``sys.modules``."""

    module = ModuleType(name)
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    sys.modules.setdefault(name, module)
    return module


_stub_module("docling_core", package=True)
_stub_module("docling_core.transforms", package=True)
_stub_module("docling_core.transforms.chunker", package=True)
_stub_module("docling_core.transforms.chunker.base", attrs={"BaseChunk": type("BaseChunk", (), {})})
_stub_module(
    "docling_core.transforms.chunker.hybrid_chunker",
    attrs={"HybridChunker": type("HybridChunker", (), {})},
)
_stub_module(
    "docling_core.transforms.chunker.tokenizer",
    package=True,
)
_stub_module(
    "docling_core.transforms.chunker.tokenizer.huggingface",
    attrs={"HuggingFaceTokenizer": type("HuggingFaceTokenizer", (), {})},
)
_stub_module("docling_core.types", package=True)
_stub_module("docling_core.types.doc", package=True)
_stub_module(
    "docling_core.types.doc.document",
    attrs={
        "DoclingDocument": type("DoclingDocument", (), {}),
        "DocTagsDocument": type("DocTagsDocument", (), {}),
    },
)
_stub_module(
    "transformers",
    attrs={"AutoTokenizer": SimpleNamespace(from_pretrained=lambda *_, **__: object())},
)
_stub_module("tqdm", attrs={"tqdm": lambda iterable=None, **_: iterable})
_stub_module(
    "DocsToKG.DocParsing.serializers",
    attrs={"RichSerializerProvider": lambda: SimpleNamespace()},
)

import DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin as chunker  # noqa: E402
import DocsToKG.DocParsing.EmbeddingV2 as embeddings  # noqa: E402
from DocsToKG.DocParsing._common import jsonl_load  # noqa: E402


class DummyTokenizer:
    def __init__(
        self, tokenizer: object, max_tokens: int
    ) -> None:  # pragma: no cover - signature parity
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class DummyHybridChunker:
    def __init__(
        self, tokenizer: DummyTokenizer, merge_peers: bool, serializer_provider: object
    ) -> None:
        self._texts: Dict[str, List[str]] = {}

    def prime(self, mapping: Dict[str, List[str]]) -> None:
        self._texts = mapping

    def chunk(self, dl_doc: Dict[str, str]) -> List[tuple[str, int]]:
        texts = self._texts.get(dl_doc["name"], [dl_doc["name"]])
        return [(dl_doc["name"], idx) for idx, _ in enumerate(texts)]

    def contextualize(self, chunk: tuple[str, int]) -> str:
        doc_name, idx = chunk
        return self._texts[doc_name][idx]


def configure_chunker_stubs(
    monkeypatch, texts_map: Dict[str, List[str]], image_metadata_fn=None
) -> None:
    monkeypatch.setattr(
        chunker, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_, **__: object())
    )
    monkeypatch.setattr(chunker, "HuggingFaceTokenizer", DummyTokenizer)

    class DummyProvenance:
        def __init__(self, **kwargs):
            self.data = kwargs

    class DummyChunkRow:
        def __init__(self, **kwargs):
            payload = dict(kwargs)
            provenance = payload.get("provenance")
            if isinstance(provenance, DummyProvenance):
                payload["provenance"] = provenance.data
            self.data = payload

        def model_dump(
            self, mode: str = "json", exclude_none: bool = True
        ):  # pragma: no cover - passthrough
            return self.data

    monkeypatch.setattr(chunker, "ProvenanceMetadata", DummyProvenance)
    monkeypatch.setattr(chunker, "ChunkRow", DummyChunkRow)
    monkeypatch.setattr(chunker, "get_docling_version", lambda: "docling-stub")

    chunker_manifest_log.clear()
    original_manifest = chunker.manifest_append

    def record_manifest(stage, doc_id, status, **metadata):
        entry = {"stage": stage, "doc_id": doc_id, "status": status}
        entry.update(metadata)
        chunker_manifest_log.append(entry)
        original_manifest(stage, doc_id, status, **metadata)

    monkeypatch.setattr(chunker, "manifest_append", record_manifest)

    stub_chunker = DummyHybridChunker(tokenizer=None, merge_peers=True, serializer_provider=None)
    stub_chunker.prime(texts_map)

    def factory(*_, **__):
        return stub_chunker

    monkeypatch.setattr(chunker, "HybridChunker", factory)
    monkeypatch.setattr(chunker, "RichSerializerProvider", lambda: object())

    def _extract_refs_and_pages(chunk):
        return [], []

    monkeypatch.setattr(chunker, "extract_refs_and_pages", _extract_refs_and_pages)
    if image_metadata_fn is None:

        def _default_image_metadata(*_, **__):
            return False, False, 0

        image_metadata_fn = _default_image_metadata
    monkeypatch.setattr(chunker, "summarize_image_metadata", image_metadata_fn)
    monkeypatch.setattr(
        chunker,
        "coalesce_small_runs",
        lambda records, *_, **__: records,
    )
    monkeypatch.setattr(
        chunker,
        "build_doc",
        lambda doc_name, doctags_text: {"name": doc_name, "payload": doctags_text},
    )


@contextmanager
def crashing_atomic_write(module, crash_on_write: int):
    original = module.atomic_write

    @contextmanager
    def wrapper(path):
        with original(path) as handle:

            class Crashy:
                def __init__(self, inner):
                    self._inner = inner
                    self._writes = 0

                def write(self, data):
                    self._writes += 1
                    if self._writes > crash_on_write:
                        raise RuntimeError("Simulated crash")
                    return self._inner.write(data)

                def __getattr__(self, name):  # pragma: no cover - passthrough
                    return getattr(self._inner, name)

            yield Crashy(handle)

    try:
        module.atomic_write = wrapper
        yield
    finally:
        module.atomic_write = original


def prepare_data_root(tmp_path) -> SimpleNamespace:
    data_root = tmp_path / "Data"
    doctags_dir = data_root / "DocTagsFiles"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Vectors"
    manifests_dir = data_root / "Manifests"
    for directory in (doctags_dir, chunks_dir, vectors_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        data_root=data_root,
        doctags_dir=doctags_dir,
        chunks_dir=chunks_dir,
        vectors_dir=vectors_dir,
        manifests_dir=manifests_dir,
    )


def chunker_args(env: SimpleNamespace, **overrides):
    base = dict(
        data_root=env.data_root,
        in_dir=env.doctags_dir,
        out_dir=env.chunks_dir,
        min_tokens=1,
        max_tokens=128,
        tokenizer_model="stub/tokenizer",
        resume=False,
        force=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def embeddings_args(env: SimpleNamespace, **overrides):
    base = dict(
        data_root=env.data_root,
        chunks_dir=env.chunks_dir,
        out_dir=env.vectors_dir,
        resume=False,
        force=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def write_dummy_doctags(env: SimpleNamespace, name: str) -> None:
    (env.doctags_dir / f"{name}.doctags").write_text("dummy", encoding="utf-8")


def write_dummy_chunks(env: SimpleNamespace, name: str, rows: Iterable[dict]) -> None:
    path = env.chunks_dir / f"{name}.chunks.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            row = dict(row)
            row.setdefault("schema_version", "docparse/1.1.0")
            handle.write(json.dumps(row) + "\n")


def configure_embeddings_stubs(monkeypatch):
    monkeypatch.setattr(embeddings, "_ensure_splade_dependencies", lambda: None)
    monkeypatch.setattr(embeddings, "_ensure_qwen_dependencies", lambda: None)

    class DummyBM25:
        def __init__(self, **kwargs):
            self.data = kwargs

    class DummySPLADE:
        def __init__(self, **kwargs):
            self.data = kwargs

    class DummyDense:
        def __init__(self, **kwargs):
            self.data = kwargs

    class DummyVectorRow:
        def __init__(self, **kwargs):
            payload = dict(kwargs)
            bm25 = payload.get("BM25")
            if isinstance(bm25, DummyBM25):
                payload["BM25"] = bm25.data
            splade = payload.get("SPLADEv3")
            if isinstance(splade, DummySPLADE):
                payload["SPLADEv3"] = splade.data
            dense = payload.get("Qwen3_4B")
            if isinstance(dense, DummyDense):
                payload["Qwen3_4B"] = dense.data
            self.data = payload

        def model_dump(self, by_alias: bool = True):  # pragma: no cover - passthrough
            return self.data

    monkeypatch.setattr(embeddings, "BM25Vector", DummyBM25)
    monkeypatch.setattr(embeddings, "SPLADEVector", DummySPLADE)
    monkeypatch.setattr(embeddings, "DenseVector", DummyDense)
    monkeypatch.setattr(embeddings, "VectorRow", DummyVectorRow)

    embeddings_manifest_log.clear()
    original_embeddings_manifest = embeddings.manifest_append

    def record_embeddings_manifest(stage, doc_id, status, **metadata):
        entry = {"stage": stage, "doc_id": doc_id, "status": status}
        entry.update(metadata)
        embeddings_manifest_log.append(entry)
        original_embeddings_manifest(stage, doc_id, status, **metadata)

    monkeypatch.setattr(embeddings, "manifest_append", record_embeddings_manifest)

    def fake_splade(cfg, texts, batch_size=None):
        tokens = [[f"tok-{idx}"] for idx, _ in enumerate(texts)]
        weights = [[0.5] for _ in texts]
        return tokens, weights

    def fake_qwen(cfg, texts, batch_size=None):
        value = 1.0 / math.sqrt(2560)
        vector = [value] * 2560
        return [vector for _ in texts]

    monkeypatch.setattr(embeddings, "splade_encode", fake_splade)
    monkeypatch.setattr(embeddings, "qwen_embed", fake_qwen)


@pytest.mark.usefixtures("monkeypatch")
def test_chunker_failure_leaves_no_partial_files(tmp_path, monkeypatch):
    env = prepare_data_root(tmp_path)
    configure_chunker_stubs(monkeypatch, {"sample": ["alpha beta", "gamma delta"]})
    write_dummy_doctags(env, "sample")

    args = chunker_args(env)
    with crashing_atomic_write(chunker, crash_on_write=1):
        with pytest.raises(RuntimeError):
            chunker.main(args)

    out_file = env.chunks_dir / "sample.chunks.jsonl"
    assert not out_file.exists()
    assert not out_file.with_suffix(out_file.suffix + ".tmp").exists()

    assert chunker_manifest_log[-1]["status"] == "failure"


@pytest.mark.usefixtures("monkeypatch")
def test_embeddings_failure_cleans_temporary_files(tmp_path, monkeypatch):
    env = prepare_data_root(tmp_path)
    configure_embeddings_stubs(monkeypatch)
    rows = [
        {
            "uuid": "u1",
            "text": "hello world",
            "doc_id": "doc",
            "schema_version": "docparse/1.1.0",
        },
        {
            "uuid": "u2",
            "text": "another chunk",
            "doc_id": "doc",
            "schema_version": "docparse/1.1.0",
        },
        {"uuid": "u1", "text": "hello world", "doc_id": "doc"},
        {"uuid": "u2", "text": "another chunk", "doc_id": "doc"},
    ]
    write_dummy_chunks(env, "doc", rows)

    args = embeddings_args(env)
    with crashing_atomic_write(embeddings, crash_on_write=1):
        with pytest.raises(RuntimeError):
            embeddings.main(args)

    out_file = env.vectors_dir / "doc.vectors.jsonl"
    assert not out_file.exists()
    assert not out_file.with_suffix(out_file.suffix + ".tmp").exists()

    assert embeddings_manifest_log[-1]["status"] == "failure"


@pytest.mark.usefixtures("monkeypatch")
def test_chunker_success_outputs_readable_file(tmp_path, monkeypatch):
    env = prepare_data_root(tmp_path)
    configure_chunker_stubs(monkeypatch, {"sample": ["alpha beta", "gamma delta"]})
    write_dummy_doctags(env, "sample")

    args = chunker_args(env)
    assert chunker.main(args) == 0

    out_file = env.chunks_dir / "sample.chunks.jsonl"
    assert out_file.exists()
    rows = jsonl_load(out_file)
    assert len(rows) == 2
    assert {row["chunk_id"] for row in rows} == {0, 1}


@pytest.mark.usefixtures("monkeypatch")
def test_chunker_promotes_image_metadata(tmp_path, monkeypatch):
    env = prepare_data_root(tmp_path)

    def image_meta(chunk, text):
        _, idx = chunk
        if idx == 0:
            return True, False, 1
        return False, True, 3

    configure_chunker_stubs(
        monkeypatch, {"sample": ["alpha", "beta"]}, image_metadata_fn=image_meta
    )
    write_dummy_doctags(env, "sample")

    args = chunker_args(env)
    assert chunker.main(args) == 0

    rows = jsonl_load(env.chunks_dir / "sample.chunks.jsonl")
    assert rows[0]["has_image_captions"] is True
    assert rows[0]["has_image_classification"] is False
    assert rows[0]["num_images"] == 1
    assert rows[1]["has_image_captions"] is False
    assert rows[1]["has_image_classification"] is True
    assert rows[1]["num_images"] == 3


@pytest.mark.usefixtures("monkeypatch")
def test_chunker_resume_after_failure(tmp_path, monkeypatch):
    env = prepare_data_root(tmp_path)
    configure_chunker_stubs(monkeypatch, {"sample": ["alpha beta", "gamma delta"]})
    write_dummy_doctags(env, "sample")

    failing_args = chunker_args(env, resume=False)
    with crashing_atomic_write(chunker, crash_on_write=1):
        with pytest.raises(RuntimeError):
            chunker.main(failing_args)

    success_args = chunker_args(env, resume=True)
    assert chunker.main(success_args) == 0

    out_file = env.chunks_dir / "sample.chunks.jsonl"
    rows = jsonl_load(out_file)
    assert len(rows) == 2

    entries = [entry for entry in chunker_manifest_log if entry["doc_id"] == "sample.doctags"]
    statuses = [entry["status"] for entry in entries]
    assert statuses.count("failure") == 1
    assert statuses.count("success") == 1


@pytest.mark.usefixtures("monkeypatch")
def test_chunker_concurrent_writes_isolated(tmp_path, monkeypatch):
    env = prepare_data_root(tmp_path)
    texts_map = {
        "doc1": ["one two", "three four"],
        "doc2": ["five six", "seven eight"],
    }
    configure_chunker_stubs(monkeypatch, texts_map)

    dir1 = env.doctags_dir / "a"
    dir2 = env.doctags_dir / "b"
    dir1.mkdir()
    dir2.mkdir()
    (dir1 / "doc1.doctags").write_text("dummy", encoding="utf-8")
    (dir2 / "doc2.doctags").write_text("dummy", encoding="utf-8")

    def run(doc_dir):
        args = chunker_args(env, in_dir=doc_dir)
        return chunker.main(args)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [dir1, dir2]))

    assert results == [0, 0]

    for name in ("doc1", "doc2"):
        out_file = env.chunks_dir / f"{name}.chunks.jsonl"
        assert out_file.exists()
        rows = jsonl_load(out_file)
        texts = {row["text"] for row in rows}
        expected = set(texts_map[name])
        assert texts == expected
