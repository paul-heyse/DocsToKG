# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_atomic_writes",
#   "purpose": "Pytest coverage for content download atomic writes scenarios",
#   "sections": [
#     {
#       "id": "stub-module",
#       "name": "_stub_module",
#       "anchor": "function-stub-module",
#       "kind": "function"
#     },
#     {
#       "id": "dummytokenizer",
#       "name": "DummyTokenizer",
#       "anchor": "class-dummytokenizer",
#       "kind": "class"
#     },
#     {
#       "id": "dummyhybridchunker",
#       "name": "DummyHybridChunker",
#       "anchor": "class-dummyhybridchunker",
#       "kind": "class"
#     },
#     {
#       "id": "configure-chunker-stubs",
#       "name": "configure_chunker_stubs",
#       "anchor": "function-configure-chunker-stubs",
#       "kind": "function"
#     },
#     {
#       "id": "crashing-atomic-write",
#       "name": "crashing_atomic_write",
#       "anchor": "function-crashing-atomic-write",
#       "kind": "function"
#     },
#     {
#       "id": "prepare-data-root",
#       "name": "prepare_data_root",
#       "anchor": "function-prepare-data-root",
#       "kind": "function"
#     },
#     {
#       "id": "chunker-args",
#       "name": "chunker_args",
#       "anchor": "function-chunker-args",
#       "kind": "function"
#     },
#     {
#       "id": "embeddings-args",
#       "name": "embeddings_args",
#       "anchor": "function-embeddings-args",
#       "kind": "function"
#     },
#     {
#       "id": "write-dummy-doctags",
#       "name": "write_dummy_doctags",
#       "anchor": "function-write-dummy-doctags",
#       "kind": "function"
#     },
#     {
#       "id": "write-dummy-chunks",
#       "name": "write_dummy_chunks",
#       "anchor": "function-write-dummy-chunks",
#       "kind": "function"
#     },
#     {
#       "id": "configure-embeddings-stubs",
#       "name": "configure_embeddings_stubs",
#       "anchor": "function-configure-embeddings-stubs",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunker-failure-leaves-no-partial-files",
#       "name": "test_chunker_failure_leaves_no_partial_files",
#       "anchor": "function-test-chunker-failure-leaves-no-partial-files",
#       "kind": "function"
#     },
#     {
#       "id": "test-embeddings-failure-cleans-temporary-files",
#       "name": "test_embeddings_failure_cleans_temporary_files",
#       "anchor": "function-test-embeddings-failure-cleans-temporary-files",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunker-success-outputs-readable-file",
#       "name": "test_chunker_success_outputs_readable_file",
#       "anchor": "function-test-chunker-success-outputs-readable-file",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunker-promotes-image-metadata",
#       "name": "test_chunker_promotes_image_metadata",
#       "anchor": "function-test-chunker-promotes-image-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunker-resume-after-failure",
#       "name": "test_chunker_resume_after_failure",
#       "anchor": "function-test-chunker-resume-after-failure",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunker-concurrent-writes-isolated",
#       "name": "test_chunker_concurrent_writes_isolated",
#       "anchor": "function-test-chunker-concurrent-writes-isolated",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Integration tests for atomic download writes and chunker/embedding stubs.

By wiring stub chunkers, embedding workers, and fake HTTP sessions, these tests
assert that partial downloads are cleaned up on failure, digests are recorded on
success, concurrent writes remain isolated, and resume metadata is respected.
They model the higher-level DocParsing pipeline interactions that rely on the
ContentDownload atomic write guarantees. The suite also documents behaviour
expectations for downstream doc parsing workflows that consume the outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import pytest

from tests.content_download.stubs import dependency_stubs as install_content_download_stubs

install_content_download_stubs()

# --- Globals ---

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
    from DocsToKG.ContentDownload.cli import download_candidate
    from DocsToKG.ContentDownload.core import WorkArtifact

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

    class _DummyHeadResponse(_BaseDummyResponse):
        def __init__(self) -> None:
            super().__init__(status_code=200)

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

        def head(self, url: str, **kwargs: Any) -> _BaseDummyResponse:  # noqa: D401
            return _DummyHeadResponse()

        def get(self, url: str, **kwargs: Any) -> _BaseDummyResponse:  # noqa: D401
            return self._response

        def request(self, method: str, url: str, **kwargs: Any) -> _BaseDummyResponse:
            if method == "GET":
                return self.get(url, **kwargs)
            if method == "HEAD":
                return self.head(url, **kwargs)
            raise AssertionError(f"Unsupported method {method}")

    def _make_artifact(tmp_path: Path) -> WorkArtifact:
        pdf_dir = tmp_path / "pdfs"
        html_dir = tmp_path / "html"
        xml_dir = tmp_path / "xml"
        pdf_dir.mkdir()
        html_dir.mkdir()
        xml_dir.mkdir()
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
            xml_dir=xml_dir,
        )

    def _download_with_session(
        session: _DummySession, tmp_path: Path, enable_resume: bool = False
    ) -> tuple[WorkArtifact, Path, Dict[str, Dict[str, Any]], DownloadOutcome]:
        artifact = _make_artifact(tmp_path)
        context: Dict[str, Dict[str, Any]] = {"previous": {}}
        if enable_resume:
            context["enable_range_resume"] = True
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
    def test_partial_download_cleans_part_file(tmp_path: Path) -> None:
        session = _DummySession(_FailingResponse())
        artifact, final_path, _, outcome = _download_with_session(session, tmp_path)

        part_path = final_path.with_suffix(".pdf.part")
        assert outcome.classification is Classification.HTTP_ERROR
        assert not final_path.exists()
        assert not part_path.exists()  # Partial files should be cleaned up when resume is disabled

    @pytest.mark.skipif(not DOWNLOAD_DEPS_AVAILABLE, reason=DOWNLOAD_TESTS_SKIP_REASON)
    def test_successful_download_records_digest(tmp_path: Path) -> None:
        body = b"A" * 1500
        payload = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n" + body + b"\n%%EOF\n"
        expected_sha = hashlib.sha256(payload).hexdigest()
        session = _DummySession(_SuccessfulResponse(payload))

        artifact, final_path, _, outcome = _download_with_session(session, tmp_path)

        assert outcome.classification is Classification.PDF
        assert outcome.sha256 == expected_sha
        assert outcome.content_length == len(payload)
        assert final_path.exists()
        assert not final_path.with_suffix(".pdf.part").exists()
        assert final_path.read_bytes() == payload


chunker_manifest_log: List[dict] = []
embeddings_manifest_log: List[dict] = []
# --- Helper Functions ---


from docling_core.persistence import manifest_append, manifest_load  # noqa: E402
from docling_core.serializers import RichSerializerProvider  # noqa: E402

import DocsToKG.DocParsing.chunking as chunker  # noqa: E402
import DocsToKG.DocParsing.embedding as embeddings  # noqa: E402
from DocsToKG.ContentDownload.core import Classification  # noqa: E402
from DocsToKG.ContentDownload.pipeline import DownloadOutcome  # noqa: E402
from DocsToKG.DocParsing.core import iter_jsonl  # noqa: E402

if not hasattr(chunker, "manifest_append"):
    chunker.manifest_append = manifest_append  # type: ignore[assignment]
if not hasattr(chunker, "manifest_load"):
    chunker.manifest_load = manifest_load  # type: ignore[assignment]
if not hasattr(chunker, "RichSerializerProvider"):
    chunker.RichSerializerProvider = RichSerializerProvider  # type: ignore[assignment]


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
    patcher, texts_map: Dict[str, List[str]], image_metadata_fn=None
) -> None:
    chunking_runtime = chunker.runtime

    def _patch(name: str, value, *, package: bool = True, runtime: bool = True) -> None:
        if package:
            patcher.setattr(chunker, name, value, raising=False)
        if runtime:
            patcher.setattr(chunking_runtime, name, value, raising=False)

    _patch("AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_, **__: object()))
    _patch("HuggingFaceTokenizer", DummyTokenizer)
    _patch("HybridChunker", DummyHybridChunker)

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

    _patch("ProvenanceMetadata", DummyProvenance)
    _patch("ChunkRow", DummyChunkRow)
    _patch("get_docling_version", lambda: "docling-stub")

    chunker_manifest_log.clear()
    original_chunk_log_event = chunking_runtime.log_event

    def _log_event_wrapper(logger, level, message, **metadata):
        if metadata.get("stage") == "chunking" and "status" in metadata:
            chunker_manifest_log.append(
                {
                    "stage": metadata.get("stage"),
                    "doc_id": metadata.get("doc_id"),
                    "status": metadata.get("status"),
                }
            )
        return original_chunk_log_event(logger, level, message, **metadata)

    patcher.setattr(chunking_runtime, "log_event", _log_event_wrapper)

    def _record(status: str, original):
        def wrapper(*, stage, doc_id, **metadata):
            entry = {"stage": stage, "doc_id": doc_id, "status": status}
            entry.update(metadata)
            chunker_manifest_log.append(entry)
            return original(stage=stage, doc_id=doc_id, **metadata)

        return wrapper

    _patch(
        "manifest_log_failure",
        _record("failure", chunker.manifest_log_failure),
        runtime=False,
    )
    _patch(
        "manifest_log_success",
        _record("success", chunker.manifest_log_success),
        runtime=False,
    )
    _patch(
        "manifest_log_skip",
        _record("skip", chunker.manifest_log_skip),
        runtime=False,
    )

    stub_chunker = DummyHybridChunker(
        tokenizer=DummyTokenizer(tokenizer=object(), max_tokens=512),
        merge_peers=True,
        serializer_provider=RichSerializerProvider(),
    )
    stub_chunker.prime(texts_map)

    def factory(*_, **__):
        return stub_chunker

    _patch("HybridChunker", factory)
    _patch("RichSerializerProvider", RichSerializerProvider)

    chunk_result_cls = chunking_runtime.ChunkResult

    def _process_chunk_task_stub(task):
        doc_id = task.doc_id
        doc_name = task.doc_stem
        texts = texts_map.get(doc_name, texts_map.get(doc_id, [doc_id]))
        try:
            with chunker.atomic_write(task.output_path) as handle:
                for idx, text in enumerate(texts):
                    metadata = image_metadata_fn((doc_id, idx), text)
                    has_caption, has_classification, num_images, confidence = metadata
                    row = {
                        "doc_id": doc_id,
                        "chunk_id": idx,
                        "text": text,
                        "schema_version": getattr(
                            chunking_runtime, "CHUNK_SCHEMA_VERSION", "docparse/1.1.0"
                        ),
                        "has_image_captions": has_caption,
                        "has_image_classification": has_classification,
                        "num_images": num_images,
                        "image_confidence": confidence,
                    }
                    handle.write(json.dumps(row) + "\n")
        except Exception as exc:  # pragma: no cover - exercised in failure tests
            return chunk_result_cls(
                doc_id=doc_id,
                doc_stem=doc_name,
                status="error",
                duration_s=0.0,
                input_path=task.doc_path,
                output_path=task.output_path,
                input_hash=task.input_hash,
                chunk_count=0,
                total_tokens=0,
                parse_engine="docling-html",
                sanitizer_profile=None,
                anchors_injected=False,
                error=str(exc),
            )

        total_tokens = sum(len(text.split()) for text in texts)
        return chunk_result_cls(
            doc_id=doc_id,
            doc_stem=doc_name,
            status="success",
            duration_s=0.0,
            input_path=task.doc_path,
            output_path=task.output_path,
            input_hash=task.input_hash,
            chunk_count=len(texts),
            total_tokens=total_tokens,
            parse_engine="docling-html",
            sanitizer_profile=None,
            anchors_injected=False,
        )

    _patch("_process_chunk_task", _process_chunk_task_stub, runtime=True, package=False)
    _patch("_chunk_worker_initializer", lambda cfg: None, runtime=True, package=False)

    def _extract_refs_and_pages(chunk):
        return [], []

    _patch("extract_refs_and_pages", _extract_refs_and_pages)
    if image_metadata_fn is None:

        def _default_image_metadata(*_, **__):
            return False, False, 0, None

        image_metadata_fn = _default_image_metadata
    _patch("summarize_image_metadata", image_metadata_fn)
    _patch(
        "coalesce_small_runs",
        lambda records, *_, **__: records,
    )
    _patch(
        "build_doc",
        lambda doc_name, doctags_text: {"name": doc_name, "payload": doctags_text},
    )


@contextmanager
def crashing_atomic_write(module, crash_on_write: int):
    target_module = module
    if not hasattr(target_module, "atomic_write") and hasattr(target_module, "runtime"):
        target_module = target_module.runtime
    original = target_module.atomic_write

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

    runtime_module = getattr(module, "runtime", None)
    try:
        module._crash_after_write = crash_on_write
        if runtime_module is not None:
            runtime_module._crash_after_write = crash_on_write
        module.atomic_write = wrapper
        yield
    finally:
        module.atomic_write = original
        module._crash_after_write = None
        if runtime_module is not None:
            runtime_module._crash_after_write = None


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
        soft_barrier_margin=0,
        soft_barrier_every=0,
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


def configure_embeddings_stubs(patcher):
    embedding_runtime = embeddings.runtime

    def _patch_all(name: str, value) -> None:
        patcher.setattr(embeddings, name, value, raising=False)
        patcher.setattr(embedding_runtime, name, value, raising=False)

    _patch_all("_ensure_splade_dependencies", lambda: None)
    _patch_all("_ensure_qwen_dependencies", lambda: None)

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

    _patch_all("BM25Vector", DummyBM25)
    _patch_all("SPLADEVector", DummySPLADE)
    _patch_all("DenseVector", DummyDense)
    _patch_all("VectorRow", DummyVectorRow)

    embeddings_manifest_log.clear()
    original_embed_log_event = embedding_runtime.log_event

    def _embed_log_event(logger, level, message, **metadata):
        if metadata.get("stage") == "embedding" and "status" in metadata:
            embeddings_manifest_log.append(
                {
                    "stage": metadata.get("stage"),
                    "doc_id": metadata.get("doc_id"),
                    "status": metadata.get("status"),
                }
            )
        return original_embed_log_event(logger, level, message, **metadata)

    patcher.setattr(embedding_runtime, "log_event", _embed_log_event)

    def _embed_record(status: str, original):
        def wrapper(*, stage, doc_id, **metadata):
            entry = {"stage": stage, "doc_id": doc_id, "status": status}
            entry.update(metadata)
            embeddings_manifest_log.append(entry)
            return original(stage=stage, doc_id=doc_id, **metadata)

        return wrapper

    _patch_all(
        "manifest_log_failure",
        _embed_record("failure", embeddings.manifest_log_failure),
    )
    _patch_all(
        "manifest_log_success",
        _embed_record("success", embeddings.manifest_log_success),
    )
    _patch_all(
        "manifest_log_skip",
        _embed_record("skip", embeddings.manifest_log_skip),
    )

    def _process_chunk_file_vectors_stub(
        chunk_file,
        out_path,
        stats,
        args,
        validator,
        logger,
        content_hasher=None,
    ):
        crash_limit = getattr(embeddings, "_crash_after_write", None)
        if crash_limit is None:
            crash_limit = getattr(embedding_runtime, "_crash_after_write", None)
        rows: List[dict] = []
        with open(chunk_file, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))

        processed = 0
        with embeddings.atomic_write(out_path) as handle:
            for row in rows:
                handle.write(json.dumps(row))
                handle.write("\n")
                processed += 1
                if crash_limit is not None and processed >= crash_limit:
                    raise RuntimeError("Simulated crash")

        return processed, [1 for _ in range(processed)], [1.0 for _ in range(processed)]

    _patch_all("process_chunk_file_vectors", _process_chunk_file_vectors_stub)

    def _record(status: str, original):
        def wrapper(*, stage, doc_id, **metadata):
            entry = {"stage": stage, "doc_id": doc_id, "status": status}
            entry.update(metadata)
            embeddings_manifest_log.append(entry)
            return original(stage=stage, doc_id=doc_id, **metadata)

        return wrapper

    _patch_all(
        "manifest_log_failure",
        _record("failure", embedding_runtime.manifest_log_failure),
    )
    _patch_all(
        "manifest_log_success",
        _record("success", embedding_runtime.manifest_log_success),
    )
    _patch_all(
        "manifest_log_skip",
        _record("skip", embedding_runtime.manifest_log_skip),
    )

    def fake_splade(cfg, texts, batch_size=None):
        tokens = [[f"tok-{idx}"] for idx, _ in enumerate(texts)]
        weights = [[0.5] for _ in texts]
        return tokens, weights

    def fake_qwen(cfg, texts, batch_size=None):
        value = 1.0 / math.sqrt(2560)
        vector = [value] * 2560
        return [vector for _ in texts]

    _patch_all("splade_encode", fake_splade)
    _patch_all("qwen_embed", fake_qwen)


@pytest.mark.usefixtures("patcher")
# --- Test Cases ---


def test_chunker_failure_leaves_no_partial_files(tmp_path, patcher):
    env = prepare_data_root(tmp_path)
    configure_chunker_stubs(patcher, {"sample": ["alpha beta", "gamma delta"]})
    write_dummy_doctags(env, "sample")

    args = chunker_args(env)
    with crashing_atomic_write(chunker, crash_on_write=1):
        with pytest.raises(RuntimeError):
            chunker.main(args)

    out_file = env.chunks_dir / "sample.chunks.jsonl"
    assert not out_file.exists()
    assert not out_file.with_suffix(out_file.suffix + ".tmp").exists()

    assert chunker_manifest_log[-1]["status"] == "failure"


@pytest.mark.usefixtures("patcher")
def test_embeddings_failure_cleans_temporary_files(tmp_path, patcher):
    env = prepare_data_root(tmp_path)
    configure_embeddings_stubs(patcher)
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


@pytest.mark.usefixtures("patcher")
def test_chunker_success_outputs_readable_file(tmp_path, patcher):
    env = prepare_data_root(tmp_path)
    configure_chunker_stubs(patcher, {"sample": ["alpha beta", "gamma delta"]})
    write_dummy_doctags(env, "sample")

    args = chunker_args(env)
    assert chunker.main(args) == 0

    out_file = env.chunks_dir / "sample.chunks.jsonl"
    assert out_file.exists()
    rows = list(iter_jsonl(out_file))
    assert len(rows) == 2
    assert {row["chunk_id"] for row in rows} == {0, 1}


@pytest.mark.usefixtures("patcher")
def test_chunker_promotes_image_metadata(tmp_path, patcher):
    env = prepare_data_root(tmp_path)

    def image_meta(chunk, text):
        _, idx = chunk
        if idx == 0:
            return True, False, 1, 0.9
        return False, True, 3, 0.1

    configure_chunker_stubs(patcher, {"sample": ["alpha", "beta"]}, image_metadata_fn=image_meta)
    write_dummy_doctags(env, "sample")

    args = chunker_args(env)
    assert chunker.main(args) == 0

    rows = list(iter_jsonl(env.chunks_dir / "sample.chunks.jsonl"))
    assert rows[0]["has_image_captions"] is True
    assert rows[0]["has_image_classification"] is False
    assert rows[0]["num_images"] == 1
    assert rows[1]["has_image_captions"] is False
    assert rows[1]["has_image_classification"] is True
    assert rows[1]["num_images"] == 3


@pytest.mark.usefixtures("patcher")
def test_chunker_resume_after_failure(tmp_path, patcher):
    env = prepare_data_root(tmp_path)
    configure_chunker_stubs(patcher, {"sample": ["alpha beta", "gamma delta"]})
    write_dummy_doctags(env, "sample")

    failing_args = chunker_args(env, resume=False)
    with crashing_atomic_write(chunker, crash_on_write=1):
        with pytest.raises(RuntimeError):
            chunker.main(failing_args)

    success_args = chunker_args(env, resume=True)
    assert chunker.main(success_args) == 0

    out_file = env.chunks_dir / "sample.chunks.jsonl"
    rows = list(iter_jsonl(out_file))
    assert len(rows) == 2

    entries = [entry for entry in chunker_manifest_log if entry["doc_id"] == "sample.doctags"]
    statuses = [entry["status"] for entry in entries]
    assert statuses.count("failure") == 1
    assert statuses.count("success") == 1


@pytest.mark.usefixtures("patcher")
def test_chunker_concurrent_writes_isolated(tmp_path, patcher):
    env = prepare_data_root(tmp_path)
    texts_map = {
        "doc1": ["one two", "three four"],
        "doc2": ["five six", "seven eight"],
    }
    configure_chunker_stubs(patcher, texts_map)

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
        rows = list(iter_jsonl(out_file))
        texts = {row["text"] for row in rows}
        expected = set(texts_map[name])
        assert texts == expected
