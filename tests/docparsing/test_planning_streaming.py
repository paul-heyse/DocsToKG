"""Regression coverage ensuring planning iterators remain streaming."""

from __future__ import annotations

import gc
import sys
import types
from pathlib import Path
from typing import Dict

import pytest

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *args, **kwargs: {}
    yaml_stub.safe_dump = lambda *args, **kwargs: ""
    yaml_stub.load = yaml_stub.safe_load
    yaml_stub.dump = yaml_stub.safe_dump
    yaml_stub.FullLoader = object
    yaml_stub.Loader = object
    yaml_stub.Dumper = object
    sys.modules["yaml"] = yaml_stub

if "pydantic_core" not in sys.modules:
    try:  # pragma: no cover - exercised in environments without pydantic-core
        import pydantic_core as _pydantic_core  # type: ignore
    except ImportError:
        pydantic_core_stub = types.ModuleType("pydantic_core")

        class _StubValidationError(Exception):
            """Lightweight placeholder mirroring pydantic-core's ValidationError."""

            pass

        pydantic_core_stub.ValidationError = _StubValidationError
        pydantic_core_stub.__version__ = "2.41.4"
        sys.modules["pydantic_core"] = pydantic_core_stub
    else:  # pragma: no cover - when real pydantic-core is installed
        sys.modules["pydantic_core"] = _pydantic_core

if "pydantic_settings" not in sys.modules:
    pydantic_settings_stub = types.ModuleType("pydantic_settings")

    class _StubBaseSettings:
        """Minimal stand-in for ``pydantic_settings.BaseSettings``."""

        model_config: dict[str, object] | None = None

    def _settings_config_dict(**kwargs):
        return kwargs

    pydantic_settings_stub.BaseSettings = _StubBaseSettings
    pydantic_settings_stub.SettingsConfigDict = _settings_config_dict
    sys.modules["pydantic_settings"] = pydantic_settings_stub

if "pooch" not in sys.modules:
    pooch_stub = types.ModuleType("pooch")

    class _StubPooch:
        def fetch(self, *args, **kwargs):  # pragma: no cover - defensive stub
            raise RuntimeError("pooch.fetch not supported in tests")

    def _pooch_create(*args, **kwargs):
        return _StubPooch()

    class _StubHTTPDownloader:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs):  # pragma: no cover - stub behaviour
            raise RuntimeError("pooch.HTTPDownloader not supported in tests")

    pooch_stub.create = _pooch_create
    pooch_stub.HTTPDownloader = _StubHTTPDownloader
    sys.modules["pooch"] = pooch_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _StubResponse:
        def __init__(self, status_code: int = 200, text: str = "") -> None:
            self.status_code = status_code
            self.text = text

        def json(self) -> dict[str, object]:
            return {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("HTTP error")

    class _StubSession:
        def __enter__(self) -> "_StubSession":
            return self

        def __exit__(self, *args) -> None:
            return None

        def request(self, *args, **kwargs) -> _StubResponse:  # pragma: no cover - stub
            return _StubResponse()

        get = request
        post = request
        head = request

        def close(self) -> None:  # pragma: no cover - stub
            return None

    def _request(*args, **kwargs) -> _StubResponse:  # pragma: no cover - stub
        return _StubResponse()

    requests_stub.Session = _StubSession
    requests_stub.request = _request
    requests_stub.get = _request
    requests_stub.post = _request
    requests_stub.head = _request
    requests_stub.Response = _StubResponse
    requests_exceptions = types.ModuleType("requests.exceptions")

    class _StubRequestException(Exception):
        pass

    class _StubHTTPError(_StubRequestException):
        pass

    class _StubSSLError(_StubRequestException):
        pass

    requests_exceptions.RequestException = _StubRequestException
    requests_exceptions.HTTPError = _StubHTTPError
    requests_exceptions.SSLError = _StubSSLError
    sys.modules["requests.exceptions"] = requests_exceptions
    requests_stub.exceptions = requests_exceptions

    requests_adapters = types.ModuleType("requests.adapters")

    class _StubHTTPAdapter:
        def __init__(self, *args, **kwargs) -> None:
            return None

    requests_adapters.HTTPAdapter = _StubHTTPAdapter
    sys.modules["requests.adapters"] = requests_adapters
    requests_stub.adapters = requests_adapters
    sys.modules["requests"] = requests_stub

if "urllib3" not in sys.modules:
    urllib3_stub = types.ModuleType("urllib3")
    sys.modules["urllib3"] = urllib3_stub

if "urllib3.util" not in sys.modules:
    urllib3_util_stub = types.ModuleType("urllib3.util")
    sys.modules["urllib3.util"] = urllib3_util_stub
else:
    urllib3_util_stub = sys.modules["urllib3.util"]

if "urllib3.util.retry" not in sys.modules:
    urllib3_retry_stub = types.ModuleType("urllib3.util.retry")

    class _StubRetry:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def new(self, **kwargs):  # pragma: no cover - stub
            return self

    urllib3_retry_stub.Retry = _StubRetry
    sys.modules["urllib3.util.retry"] = urllib3_retry_stub
    urllib3_util_stub.retry = urllib3_retry_stub
    urllib3_stub.util = urllib3_util_stub

from tests.docparsing.stubs import dependency_stubs

dependency_stubs()

# Import after dependency stubs so core modules use fake optional deps.
from DocsToKG.DocParsing.core import ChunkDiscovery, planning  # noqa: E402


class _StreamingStub:
    """Lightweight sentinel that records concurrent live instances."""

    __slots__ = ("identifier", "_tracker")

    def __init__(self, identifier: int, tracker: Dict[str, int]) -> None:
        self.identifier = identifier
        self._tracker = tracker
        tracker["active"] += 1
        tracker["max_active"] = max(tracker["max_active"], tracker["active"])

    def __del__(self) -> None:  # pragma: no cover - depends on GC timing
        tracker = self._tracker
        tracker["active"] -= 1


def test_plan_chunk_streams_doctags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    total = 64
    tracker = {"active": 0, "max_active": 0}
    skips = {idx for idx in range(total) if idx % 5 == 0}

    data_root = tmp_path / "Data"
    doctags_dir = data_root / "DocTagsFiles"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    doctags_dir.mkdir(parents=True)
    chunks_dir.mkdir()

    def fake_iter_doctags(_directory: Path):
        for idx in range(total):
            yield _StreamingStub(idx, tracker)

    def fake_derive(doc: _StreamingStub, _in_dir: Path, out_dir: Path) -> tuple[str, Path]:
        doc_id = f"doc-{doc.identifier}"
        out_path = out_dir / f"{doc_id}.chunks.jsonl"
        if doc.identifier in skips:
            out_path.write_text("{}", encoding="utf-8")
        elif out_path.exists():
            out_path.unlink()
        return doc_id, out_path

    def fake_hash(doc: _StreamingStub) -> str:
        return f"hash-{doc.identifier}"

    import DocsToKG.DocParsing.chunking as chunking_module

    monkeypatch.setattr(
        chunking_module, "MANIFEST_STAGE", "docparse.chunks.manifest", raising=False
    )
    manifest = {
        f"doc-{idx}": {"input_hash": f"hash-{idx}", "status": "success"} for idx in range(total)
    }

    monkeypatch.setattr(planning, "detect_data_root", lambda *_args, **_kwargs: data_root)
    monkeypatch.setattr(planning, "iter_doctags", fake_iter_doctags)
    monkeypatch.setattr(planning, "derive_doc_id_and_chunks_path", fake_derive)
    monkeypatch.setattr(planning, "compute_content_hash", fake_hash)
    monkeypatch.setattr(planning, "load_manifest_index", lambda *_args, **_kwargs: manifest)

    result = planning.plan_chunk(["--data-root", str(data_root), "--resume"])

    gc.collect()
    assert tracker["active"] == 0
    assert tracker["max_active"] < total

    expected_skip = len(skips)
    assert result["process"]["count"] == total - expected_skip
    assert result["skip"]["count"] == expected_skip


def test_plan_embed_streams_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    total = 72
    tracker = {"active": 0, "max_active": 0}
    skips = {idx for idx in range(total) if idx % 4 == 0}

    data_root = tmp_path / "Data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    chunks_dir.mkdir(parents=True)
    vectors_dir.mkdir()

    index_lookup: Dict[Path, int] = {}

    def fake_iter_chunks(_directory: Path):
        for idx in range(total):
            tracker["active"] += 1
            tracker["max_active"] = max(tracker["max_active"], tracker["active"])
            logical = Path(f"doc-{idx}.chunks.jsonl")
            resolved = tmp_path / f"stream-{idx}.chunks.jsonl"
            index_lookup[resolved] = idx
            try:
                yield ChunkDiscovery(logical, resolved)
            finally:
                tracker["active"] -= 1

    def fake_derive(
        chunk_entry: ChunkDiscovery, _chunks_dir: Path, vectors_dir: Path
    ) -> tuple[str, Path]:
        idx = index_lookup[chunk_entry.resolved_path]
        doc_id = f"doc-{idx}"
        vector_path = vectors_dir / f"{doc_id}.vectors.jsonl"
        if idx in skips:
            vector_path.write_text("{}", encoding="utf-8")
        elif vector_path.exists():
            vector_path.unlink()
        return doc_id, vector_path

    def fake_hash(path: Path) -> str:
        idx = index_lookup[path]
        return f"hash-{idx}"

    import DocsToKG.DocParsing.embedding as embedding_module

    monkeypatch.setattr(
        embedding_module, "MANIFEST_STAGE", "docparse.embeddings.manifest", raising=False
    )
    manifest = {
        f"doc-{idx}": {"input_hash": f"hash-{idx}", "status": "success"} for idx in range(total)
    }

    monkeypatch.setattr(planning, "detect_data_root", lambda *_args, **_kwargs: data_root)
    monkeypatch.setattr(planning, "iter_chunks", fake_iter_chunks)
    monkeypatch.setattr(planning, "derive_doc_id_and_vectors_path", fake_derive)
    monkeypatch.setattr(planning, "compute_content_hash", fake_hash)
    monkeypatch.setattr(planning, "load_manifest_index", lambda *_args, **_kwargs: manifest)

    result = planning.plan_embed(["--data-root", str(data_root), "--resume"])

    gc.collect()
    assert tracker["active"] == 0
    assert tracker["max_active"] < total

    expected_skip = len(skips)
    assert result["process"]["count"] == total - expected_skip
    assert result["skip"]["count"] == expected_skip


def test_plan_chunk_missing_output_skips_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "Data"
    doctags_dir = data_root / "DocTagsFiles"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    doctags_dir.mkdir(parents=True)
    chunks_dir.mkdir()

    doc_path = doctags_dir / "doc-1.doctags"
    doc_path.write_text("{}", encoding="utf-8")

    doc_id = "doc-1"
    out_path = chunks_dir / f"{doc_id}.chunks.jsonl"

    import DocsToKG.DocParsing.chunking as chunking_module

    monkeypatch.setattr(
        chunking_module, "MANIFEST_STAGE", "docparse.chunks.manifest", raising=False
    )

    monkeypatch.setattr(planning, "detect_data_root", lambda *_args, **_kwargs: data_root)
    monkeypatch.setattr(planning, "iter_doctags", lambda _directory: [doc_path])
    monkeypatch.setattr(
        planning,
        "derive_doc_id_and_chunks_path",
        lambda _path, _in_dir, _out_dir: (doc_id, out_path),
    )

    manifest = {doc_id: {"input_hash": "previous"}}
    monkeypatch.setattr(planning, "load_manifest_index", lambda *_args, **_kwargs: manifest)

    calls = {"count": 0}

    def _fake_hash(_path: Path) -> str:
        calls["count"] += 1
        return "new-hash"

    monkeypatch.setattr(planning, "compute_content_hash", _fake_hash)

    result = planning.plan_chunk(["--data-root", str(data_root), "--resume"])

    assert result["process"]["count"] == 1
    assert result["skip"]["count"] == 0
    assert calls["count"] == 0


def test_plan_embed_missing_output_skips_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "Data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    chunks_dir.mkdir(parents=True)
    vectors_dir.mkdir()

    chunk_path = chunks_dir / "doc-1.chunks.jsonl"
    chunk_path.write_text("{}", encoding="utf-8")
    chunk_entry = ChunkDiscovery(chunk_path.relative_to(chunks_dir), chunk_path)

    doc_id = "doc-1"
    vector_path = vectors_dir / f"{doc_id}.vectors.jsonl"

    import DocsToKG.DocParsing.embedding as embedding_module

    monkeypatch.setattr(
        embedding_module, "MANIFEST_STAGE", "docparse.embeddings.manifest", raising=False
    )

    monkeypatch.setattr(planning, "detect_data_root", lambda *_args, **_kwargs: data_root)
    monkeypatch.setattr(planning, "iter_chunks", lambda _directory: [chunk_entry])
    monkeypatch.setattr(
        planning,
        "derive_doc_id_and_vectors_path",
        lambda _path, _chunks_dir, _vectors_dir: (doc_id, vector_path),
    )

    manifest = {doc_id: {"input_hash": "previous"}}
    monkeypatch.setattr(planning, "load_manifest_index", lambda *_args, **_kwargs: manifest)

    calls = {"count": 0}

    def _fake_hash(_path: Path) -> str:
        calls["count"] += 1
        return "new-hash"

    monkeypatch.setattr(planning, "compute_content_hash", _fake_hash)

    result = planning.plan_embed(["--data-root", str(data_root), "--resume"])

    assert result["process"]["count"] == 1
    assert result["skip"]["count"] == 0
    assert calls["count"] == 0


def test_plan_chunk_handles_symlinked_doctags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOCSTOKG_DATA_ROOT", raising=False)

    isolated_root = tmp_path / "IsolatedData"
    data_root = isolated_root / "Data"
    doctags_dir = data_root / "DocTagsFiles"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    doctags_dir.mkdir(parents=True)
    chunks_dir.mkdir(parents=True)

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_doctags = external_dir / "report.doctags"
    external_doctags.write_text("{}", encoding="utf-8")

    symlink_dir = doctags_dir / "linked"
    symlink_dir.mkdir()
    symlink_path = symlink_dir / "report.doctags"
    symlink_path.symlink_to(external_doctags)

    result = planning.plan_chunk(["--data-root", str(data_root)])

    assert result["process"]["count"] == 1
    assert result["skip"]["count"] == 0
    assert result["process"]["preview"] == ["linked/report.doctags"]


def test_plan_embed_handles_symlinked_chunks(tmp_path: Path, monkeypatch) -> None:
    # Temporarily disable the shared DOCSTOKG_DATA_ROOT to avoid interference from other tests
    monkeypatch.delenv("DOCSTOKG_DATA_ROOT", raising=False)

    # Use a completely isolated data root to avoid interference from other tests
    isolated_root = tmp_path / "IsolatedData"
    data_root = isolated_root / "Data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    chunks_dir.mkdir(parents=True)
    vectors_dir.mkdir(parents=True)

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_chunk = external_dir / "report.chunks.jsonl"
    external_chunk.write_text("{}\n", encoding="utf-8")

    symlink_dir = chunks_dir / "linked"
    symlink_dir.mkdir()
    symlink_path = symlink_dir / "report.chunks.jsonl"
    symlink_path.symlink_to(external_chunk)

    result = planning.plan_embed(["--data-root", str(data_root)])

    assert result["process"]["count"] == 1
    assert result["skip"]["count"] == 0
    assert result["process"]["preview"] == ["linked/report.doctags"]
