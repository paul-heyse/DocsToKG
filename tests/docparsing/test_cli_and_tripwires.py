# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_cli_and_tripwires",
#   "purpose": "Pytest coverage for docparsing cli and tripwires scenarios",
#   "sections": [
#     {
#       "id": "tokencountingstub",
#       "name": "_TokenCountingStub",
#       "anchor": "class-tokencountingstub",
#       "kind": "class"
#     },
#     {
#       "id": "make-rec",
#       "name": "_make_rec",
#       "anchor": "function-make-rec",
#       "kind": "function"
#     },
#     {
#       "id": "reload-cli-modules",
#       "name": "_reload_cli_modules",
#       "anchor": "function-reload-cli-modules",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunk-and-embed-cli-with-dependency-stubs",
#       "name": "test_chunk_and_embed_cli_with_dependency_stubs",
#       "anchor": "function-test-chunk-and-embed-cli-with-dependency-stubs",
#       "kind": "function"
#     },
#     {
#       "id": "test-scripts-respect-data-root",
#       "name": "test_scripts_respect_data_root",
#       "anchor": "function-test-scripts-respect-data-root",
#       "kind": "function"
#     },
#     {
#       "id": "load-jsonl",
#       "name": "_load_jsonl",
#       "anchor": "function-load-jsonl",
#       "kind": "function"
#     },
#     {
#       "id": "test-golden-chunk-count-and-hash",
#       "name": "test_golden_chunk_count_and_hash",
#       "anchor": "function-test-golden-chunk-count-and-hash",
#       "kind": "function"
#     },
#     {
#       "id": "test-golden-vectors-hashes",
#       "name": "test_golden_vectors_hashes",
#       "anchor": "function-test-golden-vectors-hashes",
#       "kind": "function"
#     },
#     {
#       "id": "test-coalesce-small-runs-idempotent",
#       "name": "test_coalesce_small_runs_idempotent",
#       "anchor": "function-test-coalesce-small-runs-idempotent",
#       "kind": "function"
#     },
#     {
#       "id": "test-coalesce-small-runs-handles-unicode",
#       "name": "test_coalesce_small_runs_handles_unicode",
#       "anchor": "function-test-coalesce-small-runs-handles-unicode",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""DocParsing CLI, path resolution, and trip-wire regression tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings(
    "ignore",
    message=".*SwigPy.*__module__ attribute",
    category=DeprecationWarning,
)

pytest.importorskip("transformers")

pytestmark = pytest.mark.filterwarnings("ignore:.*SwigPy.*__module__ attribute:DeprecationWarning")

import DocsToKG.DocParsing.chunking.runtime as chunk_runtime  # noqa: E402
import DocsToKG.DocParsing.env as doc_env  # noqa: E402
from DocsToKG.DocParsing.chunking.runtime import (  # noqa: E402
    Rec,
    coalesce_small_runs,
)
from DocsToKG.DocParsing.io import (  # noqa: E402
    resolve_attempts_path,
    resolve_manifest_path,
)
from tests.docparsing.stubs import dependency_stubs  # noqa: E402

# Optional dependency used for property-based checks.
hypothesis = pytest.importorskip("hypothesis")  # noqa: E402
given = hypothesis.given
st = hypothesis.strategies


class _TokenCountingStub:
    """Lightweight tokenizer stub for coalescence tests."""

    def count_tokens(self, *, text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return len(stripped.split())


def _make_rec(
    text: str,
    *,
    n_tok: int | None = None,
    src_idxs: list[int] | None = None,
    refs: list[str] | None = None,
    pages: list[int] | None = None,
) -> Rec:
    """Create a Rec with sensible defaults for unit tests."""

    return Rec(
        text=text,
        n_tok=len(text.split()) if n_tok is None else n_tok,
        src_idxs=src_idxs or [],
        refs=refs or [],
        pages=pages or [1],
    )


CLI_COMMANDS: list[list[str]] = [
    ["chunk", "--help"],
    ["doctags", "--pdf", "--help"],
    ["doctags", "--html", "--help"],
    ["embed", "--help"],
]

GOLDEN_DIR = Path("tests/data/docparsing/golden")
GOLDEN_CHUNKS = GOLDEN_DIR / "sample.chunks.jsonl"
GOLDEN_VECTORS = GOLDEN_DIR / "sample.vectors.jsonl"


# --- CLI entry points ---


def _reload_core_cli():
    """Reload CLI modules so newly installed stubs are honoured."""

    module_names = [
        "DocsToKG.DocParsing.chunking.runtime",
        "DocsToKG.DocParsing.embedding.runtime",
        "DocsToKG.DocParsing.core",
    ]
    reloaded = None
    for name in module_names:
        module = importlib.import_module(name)
        reloaded = importlib.reload(module)
    assert reloaded is not None  # for type checkers
    return reloaded


def test_chunk_and_embed_cli_with_dependency_stubs(tmp_path: Path) -> None:
    """Run the chunking and embedding CLIs end-to-end with synthetic stubs."""

    data_root = tmp_path / "data"
    doc_dir = data_root / "DocTagsFiles"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doctags_path = doc_dir / "sample.doctags"
    doctags_path.write_text(
        json.dumps({"uuid": "chunk-1", "text": "Example text", "doc_id": "doc"}) + "\n",
        encoding="utf-8",
    )

    dependency_stubs()
    os.environ["DOCSTOKG_DATA_ROOT"] = str(data_root)
    os.environ["DOCSTOKG_SPLADE_DIR"] = str(tmp_path / "splade")
    os.environ["DOCSTOKG_QWEN_DIR"] = str(tmp_path / "qwen")
    os.environ["DOCSTOKG_MODEL_ROOT"] = str(tmp_path / "models")
    (tmp_path / "splade").mkdir()
    (tmp_path / "qwen").mkdir()
    (tmp_path / "models").mkdir()

    heading_markers_path = tmp_path / "heading_markers.json"
    heading_markers_path.write_text(
        json.dumps({"headings": ["Article ", "Section "]}), encoding="utf-8"
    )

    core_module = _reload_core_cli()
    with pytest.raises(SystemExit) as chunk_exit:
        core_module.main(
            [
                "chunk",
                "--in-dir",
                str(doc_dir),
                "--out-dir",
                str(data_root / "ChunkedDocTagFiles"),
                "--soft-barrier-margin",
                "32",
                "--heading-markers",
                str(heading_markers_path),
                "--help",
            ]
        )
    assert chunk_exit.value.code == 0

    with pytest.raises(SystemExit) as embed_exit:
        core_module.main(
            [
                "embed",
                "--chunks-dir",
                str(data_root / "ChunkedDocTagFiles"),
                "--out-dir",
                str(data_root / "Vectors"),
                "--help",
            ]
        )
    assert embed_exit.value.code == 0


def test_chunk_resume_skips_unchanged_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk resume mode should skip unchanged inputs and retain order."""

    dependency_stubs()

    data_root = tmp_path / "data"
    doc_dir = data_root / "DocTagsFiles"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    doc_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for name in ("splade", "qwen", "models"):
        (tmp_path / name).mkdir(exist_ok=True)

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))
    monkeypatch.setenv("DOCSTOKG_SPLADE_DIR", str(tmp_path / "splade"))
    monkeypatch.setenv("DOCSTOKG_QWEN_DIR", str(tmp_path / "qwen"))
    monkeypatch.setenv("DOCSTOKG_MODEL_ROOT", str(tmp_path / "models"))

    docs = {
        "alpha": {"uuid": "alpha-1", "text": "Alpha content", "doc_id": "alpha"},
        "beta": {"uuid": "beta-1", "text": "Beta content", "doc_id": "beta"},
    }
    for name, payload in docs.items():
        (doc_dir / f"{name}.doctags").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )

    chunk_args = [
        "--in-dir",
        str(doc_dir),
        "--out-dir",
        str(chunks_dir),
        "--resume",
        "--workers",
        "2",
    ]

    first_exit = chunk_runtime.main(chunk_args)
    assert first_exit == 0

    manifest_path = resolve_manifest_path("chunks", data_root)
    attempts_path = resolve_attempts_path("chunks", data_root)
    manifest_entries = list(_load_jsonl(manifest_path))
    attempts_entries = list(_load_jsonl(attempts_path))

    manifest_docs = [row["doc_id"] for row in manifest_entries if row["doc_id"] != "__config__"]
    assert manifest_docs == ["alpha.doctags", "beta.doctags"]

    beta_path = doc_dir / "beta.doctags"
    beta_path.write_text(
        json.dumps({"uuid": "beta-2", "text": "Beta content updated", "doc_id": "beta"}) + "\n",
        encoding="utf-8",
    )

    manifest_count = len(manifest_entries)
    attempts_count = len(attempts_entries)

    second_exit = chunk_runtime.main(chunk_args)
    assert second_exit == 0

    manifest_entries_after = list(_load_jsonl(manifest_path))
    attempts_entries_after = list(_load_jsonl(attempts_path))

    manifest_delta = [
        row for row in manifest_entries_after[manifest_count:] if row.get("doc_id") != "__config__"
    ]
    manifest_delta_doc_ids = [row["doc_id"] for row in manifest_delta]

    # Check that we have entries for both files
    assert set(manifest_delta_doc_ids) == {"alpha.doctags", "beta.doctags"}

    # Check that alpha is skipped and beta is successful
    alpha_entries = [row for row in manifest_delta if row["doc_id"] == "alpha.doctags"]
    beta_entries = [row for row in manifest_delta if row["doc_id"] == "beta.doctags"]

    assert len(alpha_entries) >= 1, "Should have at least one alpha entry"
    assert len(beta_entries) >= 1, "Should have at least one beta entry"

    # Check that alpha has skip status and beta has success status
    alpha_statuses = [row.get("status") for row in alpha_entries if "status" in row]
    beta_statuses = [row.get("status") for row in beta_entries if "status" in row]

    assert "skip" in alpha_statuses, "Alpha should be skipped"
    assert "success" in beta_statuses, "Beta should be successful"

    attempt_delta = attempts_entries_after[attempts_count:]
    attempt_statuses = [entry["status"] for entry in attempt_delta]
    attempt_file_ids = [entry["file_id"] for entry in attempt_delta]

    # Check that we have the expected statuses and file IDs
    assert set(attempt_statuses) == {"skip", "success"}
    assert set(attempt_file_ids) == {"alpha.doctags", "beta.doctags"}

    # Check that alpha has skip status and beta has success status
    alpha_attempts = [entry for entry in attempt_delta if entry["file_id"] == "alpha.doctags"]
    beta_attempts = [entry for entry in attempt_delta if entry["file_id"] == "beta.doctags"]

    alpha_attempt_statuses = [entry["status"] for entry in alpha_attempts]
    beta_attempt_statuses = [entry["status"] for entry in beta_attempts]

    assert "skip" in alpha_attempt_statuses, "Alpha should be skipped"
    assert "success" in beta_attempt_statuses, "Beta should be successful"


# --- CLI path smoke tests ---


def test_scripts_respect_data_root(tmp_path: Path) -> None:
    """CLIs should honor DOCSTOKG_DATA_ROOT when resolving defaults."""

    data_root = tmp_path / "DataRoot"
    data_root.mkdir()
    os.environ["DOCSTOKG_DATA_ROOT"] = str(data_root)

    for argv in CLI_COMMANDS:
        core_module = _reload_core_cli()
        with pytest.raises(SystemExit) as exc:
            core_module.main(argv)
        assert exc.value.code == 0
        assert doc_env.detect_data_root() == data_root.resolve()


# --- Trip-wire regression checks ---


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def test_golden_chunk_count_and_hash():
    """Golden chunk fixtures should maintain deterministic order and count."""

    rows = list(_load_jsonl(GOLDEN_CHUNKS))
    assert len(rows) == 2
    hashes = [hashlib.sha256(row["text"].encode("utf-8")).hexdigest() for row in rows]
    assert hashes == [
        "ff7b1a9b11207e16b8f98365c1467777a823164aeed1f558ce5012644c293ad6",
        "361d894e5ebd8afe5b803d36a93bd342094d0562c449b0851178f4e29d0a518e",
    ]


def test_golden_vectors_hashes():
    rows = list(_load_jsonl(GOLDEN_VECTORS))
    assert len(rows) == 2
    hashes = [
        hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest() for row in rows
    ]
    assert hashes == [
        "7011f30a24e73704820a0bd0379022ab9c670476f8a40f33ce51d03cd1cdf1c4",
        "7cc9bb1a4e532ba287a1354a8e0a518c448286ebf5b8d3cdc2824be9711acc23",
    ]


def test_coalesce_small_runs_idempotent():
    rows = [
        _make_rec("hello", src_idxs=[0]),
        _make_rec("world", src_idxs=[1]),
        _make_rec("another chunk", src_idxs=[2]),
    ]
    tokenizer = _TokenCountingStub()
    merged = coalesce_small_runs(rows, tokenizer=tokenizer, min_tokens=1)
    assert merged == rows


@given(st.text())
def test_coalesce_small_runs_handles_unicode(payload: str):
    rows = [_make_rec(payload, src_idxs=[0])]
    tokenizer = _TokenCountingStub()
    merged = coalesce_small_runs(rows, tokenizer=tokenizer, min_tokens=1)
    assert merged[0].text == payload
