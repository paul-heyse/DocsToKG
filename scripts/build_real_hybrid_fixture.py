#!/usr/bin/env python3
"""
Real Vector Fixture Builder

Samples production chunk/vector artifacts and emits a deterministic test fixture
with aligned JSONL payloads, manifest metadata, and canned search queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

DEFAULT_CHUNKS_DIR = Path("Data/ChunkedDocTagFiles")
DEFAULT_VECTORS_DIR = Path("Data/Vectors")
DEFAULT_OUTPUT_DIR = Path("Data/HybridScaleFixture")
DEFAULT_NAMESPACE = "hybrid-scale"
DEFAULT_NAMESPACES = ("research", "operations", "support")
DEFAULT_SAMPLE_SIZE = 3
DEFAULT_SEED = 1337
REDACTED_FIELDS = ("source_path",)
DEFAULT_MAX_CHUNKS_PER_DOC = 0


@dataclass(frozen=True)
class FixtureDocument:
    doc_id: str
    title: str
    chunk_file: Path
    vector_file: Path
    namespace: str
    chunk_sha256: str
    vector_sha256: str
    redacted_fields: Sequence[str]
    queries: Sequence[Mapping[str, object]]
    metadata: Mapping[str, object]
    chunk_count: int


def _parse_namespaces(raw: str | None) -> List[str]:
    if raw is None:
        return []
    parts = [part.strip() for part in raw.split(",")]
    return [part for part in parts if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic real-vector fixture")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--vectors-dir", type=Path, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument(
        "--namespaces",
        type=str,
        default=None,
        help="Comma-separated namespaces to distribute sampled documents across",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_DOC,
        help="Maximum chunks to retain per sampled document (0 disables the limit)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing fixture directory",
    )
    return parser.parse_args()


def list_candidate_docs(chunks_dir: Path, vectors_dir: Path) -> List[str]:
    def chunk_id(path: Path, suffix: str) -> str:
        name = path.name
        if name.endswith(suffix):
            return name[: -len(suffix)]
        return path.stem

    chunk_files = {chunk_id(path, ".chunks.jsonl") for path in chunks_dir.glob("*.chunks.jsonl")}
    vector_files = {
        chunk_id(path, ".vectors.jsonl") for path in vectors_dir.glob("*.vectors.jsonl")
    }
    candidates = []
    for doc_id in sorted(chunk_files & vector_files):
        chunk_path = chunks_dir / f"{doc_id}.chunks.jsonl"
        vector_path = vectors_dir / f"{doc_id}.vectors.jsonl"
        if not chunk_path.exists() or not vector_path.exists():
            continue
        if chunk_path.stat().st_size == 0 or vector_path.stat().st_size == 0:
            continue
        candidates.append(doc_id)
    return candidates


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: List[Dict[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def write_jsonl(path: Path, entries: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True))
            handle.write("\n")


def sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    repo_root = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def clean_chunk_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    cleaned: List[Dict[str, object]] = []
    for record in records:
        sanitized = dict(record)
        for field in REDACTED_FIELDS:
            if field in sanitized:
                sanitized[field] = f"<redacted:{field}>"
        cleaned.append(sanitized)
    return cleaned


def derive_title(doc_id: str, chunk_records: Sequence[Mapping[str, object]]) -> str:
    for record in chunk_records:
        text = str(record.get("text", "")).strip()
        if text:
            header = text.splitlines()[0].strip()
            if header and len(header.split()) >= 4:
                return header
    words = re.split(r"[_\-\s]+", doc_id)
    return " ".join(words[:8]).strip()


def derive_query(doc_id: str, title: str) -> str:
    title_tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9']+", title)]
    doc_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9']+", doc_id)
        if not re.fullmatch(r"\d+", token)
    ]
    combined = title_tokens + doc_tokens
    if not combined:
        return doc_id.lower()
    seen = []
    for token in combined:
        if token not in seen:
            seen.append(token)
        if len(seen) == 8:
            break
    return " ".join(seen)


def build_fixture_document(
    doc_id: str,
    *,
    namespace: str,
    acl_tag: str,
    source_chunk: Path,
    source_vector: Path,
    output_dir: Path,
    max_chunks: int,
) -> FixtureDocument:
    chunk_records = load_jsonl(source_chunk)
    cleaned_chunks = clean_chunk_records(chunk_records)
    if max_chunks and len(cleaned_chunks) > max_chunks:
        cleaned_chunks = cleaned_chunks[:max_chunks]
    vector_entries = {entry["UUID"]: entry for entry in load_jsonl(source_vector)}
    vector_records: List[Dict[str, object]] = []
    for record in cleaned_chunks:
        vector_id = str(record.get("uuid"))
        vector_payload = vector_entries.get(vector_id)
        if vector_payload is None:
            raise ValueError(f"Missing vector entry for chunk {vector_id}")
        vector = vector_payload.get("Qwen3-4B", {}).get("vector", [])
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Invalid dense vector for {vector_id}")
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Dense vector must be 1-D for {vector_id}")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"Dense vector contains NaN/Inf values for {vector_id}")
        norm = float(np.linalg.norm(arr))
        if norm <= 0.0:
            raise ValueError(f"Dense vector has zero norm for {vector_id}")
        vector_records.append(vector_payload)

    chunk_output = output_dir / "chunks" / source_chunk.name
    vector_output = output_dir / "vectors" / source_vector.name

    write_jsonl(chunk_output, cleaned_chunks)
    write_jsonl(vector_output, vector_records)

    title = derive_title(doc_id, cleaned_chunks)
    query = derive_query(doc_id, title)
    chunk_hash = sha256_digest(chunk_output)
    vector_hash = sha256_digest(vector_output)

    queries = [
        {
            "query": query,
            "expected_doc_id": doc_id,
            "namespace": namespace,
        }
    ]

    return FixtureDocument(
        doc_id=doc_id,
        title=title,
        chunk_file=chunk_output,
        vector_file=vector_output,
        namespace=namespace,
        chunk_sha256=chunk_hash,
        vector_sha256=vector_hash,
        redacted_fields=REDACTED_FIELDS,
        queries=queries,
        metadata={
            "title": title,
            "source": "real_fixture",
            "acl": [acl_tag],
        },
        chunk_count=len(cleaned_chunks),
    )


def write_manifest(
    output_dir: Path,
    *,
    seed: int,
    chunks_dir: Path,
    vectors_dir: Path,
    documents: Sequence[FixtureDocument],
) -> None:
    manifest = {
        "sample_seed": seed,
        "source": {
            "chunks_dir": str(chunks_dir),
            "vectors_dir": str(vectors_dir),
        },
        "documents": [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "namespace": doc.namespace,
                "chunk_file": repo_relative(doc.chunk_file),
                "vector_file": repo_relative(doc.vector_file),
                "chunk_sha256": doc.chunk_sha256,
                "vector_sha256": doc.vector_sha256,
                "redacted_fields": list(doc.redacted_fields),
                "queries": list(doc.queries),
                "metadata": doc.metadata,
                "chunk_count": doc.chunk_count,
            }
            for doc in documents
        ],
        "stats": {
            "total_documents": len(documents),
            "total_chunks": sum(doc.chunk_count for doc in documents),
            "namespaces": sorted({doc.namespace for doc in documents}),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_queries(output_dir: Path, documents: Sequence[FixtureDocument]) -> None:
    queries = []
    for doc in documents:
        for query in doc.queries:
            queries.append(
                {
                    "doc_id": doc.doc_id,
                    "namespace": doc.namespace,
                    "query": query["query"],
                    "expected_doc_id": query["expected_doc_id"],
                }
            )
    (output_dir / "queries.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")


def write_dataset_jsonl(output_dir: Path, documents: Sequence[FixtureDocument]) -> None:
    dataset_path = output_dir / "dataset.jsonl"
    dataset_entries = []
    for doc in documents:
        chunk_rel = repo_relative(doc.chunk_file)
        vector_rel = repo_relative(doc.vector_file)
        entry = {
            "document": {
                "doc_id": doc.doc_id,
                "namespace": doc.namespace,
                "chunk_file": chunk_rel,
                "vector_file": vector_rel,
                "metadata": dict(doc.metadata),
            },
            "queries": list(doc.queries),
        }
        dataset_entries.append(entry)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for entry in dataset_entries:
            handle.write(json.dumps(entry, ensure_ascii=True))
            handle.write("\n")


def write_readme(
    output_dir: Path,
    *,
    seed: int,
    namespaces: Sequence[str],
    sample_size: int,
    max_chunks: int,
) -> None:
    readme_path = output_dir / "README.md"
    namespace_text = ", ".join(namespaces) if namespaces else DEFAULT_NAMESPACE
    namespace_flag = (
        f" --namespaces {','.join(namespaces)}"
        if namespaces
        else f" --namespace {DEFAULT_NAMESPACE}"
    )
    content = f"""# Real Hybrid Search Fixture

This directory contains a deterministic sample of chunk/vector artifacts used for
real-vector regression tests. The fixture was generated with the following parameters:

- Namespaces: `{namespace_text}`
- Sample size: `{sample_size}`
- Seed: `{seed}`
- Max chunks per document: `{max_chunks}`

To regenerate the fixture, run:

```bash
python scripts/build_real_hybrid_fixture.py --seed {seed} --sample-size {sample_size}{namespace_flag} --max-chunks-per-doc {max_chunks}
```

Ensure the `Data/ChunkedDocTagFiles` and `Data/Vectors` directories are populated
before regenerating. The builder records source file hashes so changes can be
audited when refreshing the fixture.
"""
    readme_path.write_text(content, encoding="utf-8")


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory {path} already exists. Use --overwrite to replace it."
            )
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                for descendant in sorted(child.rglob("*"), reverse=True):
                    if descendant.is_file():
                        descendant.unlink()
                    else:
                        descendant.rmdir()
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    candidates = list_candidate_docs(args.chunks_dir, args.vectors_dir)
    if not candidates:
        raise SystemExit("No matching chunk/vector artifact pairs found.")
    if args.sample_size < 1:
        raise SystemExit("Sample size must be positive.")

    namespaces = _parse_namespaces(args.namespaces)
    if not namespaces:
        namespaces = [args.namespace] if args.namespace else list(DEFAULT_NAMESPACES)

    sample_size = min(args.sample_size, len(candidates))

    ensure_output_dir(args.output_dir, args.overwrite)

    rng = random.Random(args.seed)
    selection = rng.sample(candidates, sample_size)

    documents: List[FixtureDocument] = []
    total_chunks = 0
    for idx, doc_id in enumerate(selection):
        chunk_source = args.chunks_dir / f"{doc_id}.chunks.jsonl"
        vector_source = args.vectors_dir / f"{doc_id}.vectors.jsonl"
        namespace = namespaces[idx % len(namespaces)]
        acl_tag = f"acl::{namespace}"
        document = build_fixture_document(
            doc_id,
            namespace=namespace,
            acl_tag=acl_tag,
            source_chunk=chunk_source,
            source_vector=vector_source,
            output_dir=args.output_dir,
            max_chunks=args.max_chunks_per_doc,
        )
        documents.append(document)
        total_chunks += document.chunk_count

    # Keep the dataset stable by sorting on doc_id after sampling.
    documents.sort(key=lambda doc: doc.doc_id)

    namespaces_used = sorted({doc.namespace for doc in documents})

    if total_chunks < 500:
        print(
            f"[WARN] Fixture contains {total_chunks} chunks (< 500). "
            "Consider increasing --sample-size or --max-chunks-per-doc."
        )

    write_manifest(
        args.output_dir,
        seed=args.seed,
        chunks_dir=args.chunks_dir,
        vectors_dir=args.vectors_dir,
        documents=documents,
    )
    write_queries(args.output_dir, documents)
    write_dataset_jsonl(args.output_dir, documents)
    write_readme(
        args.output_dir,
        seed=args.seed,
        namespaces=namespaces_used,
        sample_size=sample_size,
        max_chunks=args.max_chunks_per_doc,
    )

    print(
        "Fixture generated with "
        f"{len(documents)} documents ({total_chunks} chunks) across namespaces {namespaces_used} "
        f"at {args.output_dir}"
    )


if __name__ == "__main__":
    main()
