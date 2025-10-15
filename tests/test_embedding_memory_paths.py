import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import pytest

from tests._stubs import dependency_stubs


def _install_minimal_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install lightweight dependency stubs for embedding module tests."""

    tqdm_stub = ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **_: iterable if iterable is not None else []
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_stub)

    st_stub = ModuleType("sentence_transformers")

    class _SparseEncoder:  # pragma: no cover - trivial stub
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def encode(self, batch):
            class _Result:
                def __init__(self, size: int):
                    self._size = size

                def __getitem__(self, index):
                    return self

                def coalesce(self):
                    return self

                def values(self):
                    class _Values:
                        def __init__(self, size: int):
                            self._size = size

                        def numel(self):
                            return self._size

                    return _Values(len(batch[0]))

                def shape(self):  # pragma: no cover - defensive fallback
                    return (len(batch), 1)

            return _Result(len(batch))

        def decode(self, *_args, **_kwargs):
            return []

    st_stub.SparseEncoder = _SparseEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_stub)

    vllm_stub = ModuleType("vllm")

    class _LLM:  # pragma: no cover - trivial stub
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def embed(self, batch, pooling_params=None):
            class _Output:
                def __init__(self, dim: int):
                    self.outputs = type("E", (), {"embedding": [0.0] * dim})

            return [_Output(2560) for _ in batch]

    class _PoolingParams:  # pragma: no cover - trivial stub
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    vllm_stub.LLM = _LLM
    vllm_stub.PoolingParams = _PoolingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)


def test_process_pass_a_returns_stats_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`process_pass_a` should return BM25 statistics without chunk caches."""

    _install_minimal_stubs(monkeypatch)
    import DocsToKG.DocParsing.EmbeddingV2 as embed_module

    chunk_file = tmp_path / "sample.chunks.jsonl"
    chunk_file.write_text('{"text": "Example text"}\n', encoding="utf-8")

    stats = embed_module.process_pass_a([chunk_file], embed_module.get_logger(__name__))
    assert isinstance(stats, embed_module.BM25Stats)
    assert stats.N == 1
    rows = embed_module.jsonl_load(chunk_file)
    assert rows[0]["uuid"], "UUIDs should be assigned in-place"


def test_process_chunk_file_vectors_reads_texts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk texts should be sourced directly from file rows when encoding."""

    _install_minimal_stubs(monkeypatch)
    import DocsToKG.DocParsing.EmbeddingV2 as embed_module

    chunk_file = tmp_path / "doc.chunks.jsonl"
    chunk_file.write_text(
        json.dumps({"uuid": "u1", "text": "Hello world", "doc_id": "doc"}) + "\n",
        encoding="utf-8",
    )

    captured_texts: List[str] = []

    monkeypatch.setattr(
        embed_module,
        "splade_encode",
        lambda cfg, texts, batch_size=None: ([["tok"] for _ in texts], [[1.0] for _ in texts]),
    )
    monkeypatch.setattr(
        embed_module,
        "qwen_embed",
        lambda cfg, texts, batch_size=None: captured_texts.extend(texts)
        or [[1.0] + [0.0] * 2559],
    )
    captured_write_texts: List[str] = []

    def _write_vectors(
        path, uuids, texts, splade_results, qwen_results, stats, args, rows, validator, logger
    ) -> tuple[int, List[int], List[float]]:
        captured_write_texts.extend(texts)
        return len(uuids), [1] * len(uuids), [1.0] * len(uuids)

    monkeypatch.setattr(embed_module, "write_vectors", _write_vectors)

    args = embed_module.build_parser().parse_args(
        [
            "--chunks-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
            "--splade-model-dir",
            str(tmp_path),
            "--qwen-model-dir",
            str(tmp_path),
        ]
    )
    args.splade_cfg = embed_module.SpladeCfg(model_dir=tmp_path, cache_folder=tmp_path)
    args.qwen_cfg = embed_module.QwenCfg(model_dir=tmp_path)
    args.batch_size_splade = 1
    args.batch_size_qwen = 1
    args.bm25_k1 = 1.5
    args.bm25_b = 0.75
    args.out_dir = tmp_path

    stats = embed_module.BM25Stats(N=1, avgdl=1.0, df={})
    validator = embed_module.SPLADEValidator()

    embed_module.process_chunk_file_vectors(
        chunk_file, stats, args, validator, embed_module.get_logger(__name__)
    )

    assert captured_texts == ["Hello world"]
    assert captured_write_texts == ["Hello world"]


def test_cli_path_overrides_take_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI supplied model directories should override environment variables."""

    _install_minimal_stubs(monkeypatch)
    dependency_stubs()
    sys.modules.pop("DocsToKG.DocParsing.EmbeddingV2", None)
    import DocsToKG.DocParsing.EmbeddingV2 as embed_module

    env_splade = tmp_path / "env-splade"
    env_qwen = tmp_path / "env-qwen"
    cli_splade = tmp_path / "cli-splade"
    cli_qwen = tmp_path / "cli-qwen"
    cli_splade.mkdir()
    cli_qwen.mkdir()
    env_splade.mkdir()
    env_qwen.mkdir()

    monkeypatch.setenv("DOCSTOKG_SPLADE_DIR", str(env_splade))
    monkeypatch.setenv("DOCSTOKG_QWEN_DIR", str(env_qwen))

    captured: Dict[str, Path] = {}

    def _capture(
        chunk_file, stats, args, validator, logger
    ) -> tuple[int, List[int], List[float]]:
        captured["splade"] = args.splade_cfg.model_dir
        captured["qwen"] = args.qwen_cfg.model_dir
        return 0, [], []

    chunk_file = tmp_path / "example.chunks.jsonl"
    chunk_file.write_text(
        json.dumps({"uuid": "u1", "text": "text", "doc_id": "doc"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(embed_module, "iter_chunk_files", lambda _: [chunk_file])
    monkeypatch.setattr(embed_module, "process_pass_a", lambda files, logger: embed_module.BM25Stats(N=1, avgdl=1.0, df={}))
    monkeypatch.setattr(embed_module, "process_chunk_file_vectors", _capture)
    monkeypatch.setattr(embed_module, "load_manifest_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(embed_module, "compute_content_hash", lambda *_: "hash")

    exit_code = embed_module.main(
        [
            "--chunks-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
            "--splade-model-dir",
            str(cli_splade),
            "--qwen-model-dir",
            str(cli_qwen),
        ]
    )

    assert exit_code == 0
    assert captured["splade"] == cli_splade.resolve()
    assert captured["qwen"] == cli_qwen.resolve()


def test_offline_mode_requires_local_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline mode should raise when required models are absent."""

    _install_minimal_stubs(monkeypatch)
    import DocsToKG.DocParsing.EmbeddingV2 as embed_module

    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError) as exc:
        embed_module.main(
            [
                "--chunks-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
                "--splade-model-dir",
                str(missing / "splade"),
                "--qwen-model-dir",
                str(missing / "qwen"),
                "--offline",
            ]
        )

    message = str(exc.value)
    assert "SPLADE model directory missing" in message
    assert "Qwen model directory not found" in message
