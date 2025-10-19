"""Quickstart harness for DocsToKG HybridSearch."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List, Sequence

from DocsToKG.HybridSearch import (
    ChunkIngestionPipeline,
    DocumentInput,
    HybridSearchConfigManager,
    HybridSearchRequest,
    HybridSearchService,
    Observability,
)
from DocsToKG.HybridSearch.devtools.features import FeatureGenerator
from DocsToKG.HybridSearch.devtools.opensearch_simulator import OpenSearchSimulator
from DocsToKG.HybridSearch.service import infer_embedding_dim, load_dataset
from DocsToKG.HybridSearch.store import ChunkRegistry, FaissVectorStore

DEFAULT_CONFIG = {
    "dense": {"index_type": "flat", "oversample": 3, "enable_replication": False},
    "fusion": {
        "k0": 50.0,
        "mmr_lambda": 0.7,
        "max_chunks_per_doc": 2,
        "strict_highlights": False,
    },
    "retrieval": {"bm25_top_k": 20, "splade_top_k": 20, "dense_top_k": 20},
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the quickstart harness."""

    parser = argparse.ArgumentParser(
        description=(
            "Ingest the bundled hybrid dataset and execute a sample hybrid search. "
            "Requires a CUDA-enabled faiss build with GPU access."
        )
    )
    parser.add_argument(
        "--config",
        default="tmp/hybrid_quickstart.config.json",
        help="Path to a hybrid search config file (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset",
        default="tests/data/hybrid_dataset.jsonl",
        help="JSONL dataset describing documents and example queries.",
    )
    parser.add_argument(
        "--query",
        default="hybrid retrieval faiss",
        help="Query string to issue once ingestion completes.",
    )
    parser.add_argument(
        "--namespace",
        default="research",
        help="Namespace to target for the demo query (default: %(default)s).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=3,
        help="Number of results to return from the demo search (default: %(default)s).",
    )
    parser.add_argument(
        "--no-diversify",
        dest="diversify",
        action="store_false",
        help="Disable MMR diversification for the demo search.",
    )
    parser.set_defaults(diversify=True)
    return parser.parse_args(argv)


def _ensure_config(path: Path) -> None:
    """Write a default configuration when ``path`` is missing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    print(f"[hybrid-quickstart] wrote default config -> {path}")


def _build_documents(dataset: Sequence[dict]) -> List[DocumentInput]:
    """Materialise ``DocumentInput`` instances from dataset rows."""

    documents: List[DocumentInput] = []
    for entry in dataset:
        document = entry.get("document", {})
        documents.append(
            DocumentInput(
                doc_id=str(document.get("doc_id")),
                namespace=str(document.get("namespace", "default")),
                chunk_path=Path(str(document.get("chunk_file"))),
                vector_path=Path(str(document.get("vector_file"))),
                metadata=document.get("metadata", {}),
            )
        )
    return documents


def _format_score(value: float | None) -> str:
    """Return a printable representation for optional scores."""

    if value is None:
        return "n/a"
    return f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the quickstart harness."""

    args = _parse_args(argv)
    config_path = Path(args.config)
    dataset_path = Path(args.dataset)
    _ensure_config(config_path)

    dataset = load_dataset(dataset_path)
    if not dataset:
        raise SystemExit(f"No dataset rows found at {dataset_path}")

    manager = HybridSearchConfigManager(config_path)
    config = manager.get()
    embedding_dim = infer_embedding_dim(dataset)
    feature_generator = FeatureGenerator(embedding_dim=embedding_dim)
    dense_config = replace(
        config.dense,
        enable_replication=False,
        expected_ntotal=max(len(dataset) * 4, 1),
    )
    try:
        faiss_index = FaissVectorStore(dim=embedding_dim, config=dense_config)
    except RuntimeError as exc:  # pragma: no cover - depends on GPU availability
        raise SystemExit(
            "Failed to initialise FAISS GPU resources. Install faiss-gpu and ensure a CUDA device is visible."
        ) from exc

    registry = ChunkRegistry()
    opensearch = OpenSearchSimulator()
    observability = Observability()
    pipeline = ChunkIngestionPipeline(
        faiss_index=faiss_index,
        opensearch=opensearch,
        registry=registry,
        observability=observability,
    )
    documents = _build_documents(dataset)
    pipeline.upsert_documents(documents)
    namespaces = sorted({doc.namespace for doc in documents})
    print(
        f"[hybrid-quickstart] Ingested {pipeline.metrics.chunks_upserted} "
        f"chunks from {len(documents)} documents across namespaces: {namespaces}"
    )

    service: HybridSearchService | None = None
    response = None
    try:
        service = HybridSearchService(
            config_manager=manager,
            feature_generator=feature_generator,
            faiss_index=faiss_index,
            opensearch=opensearch,
            registry=registry,
            observability=observability,
        )
        request = HybridSearchRequest(
            query=args.query,
            namespace=args.namespace,
            filters={},
            page_size=args.page_size,
            diversification=args.diversify,
        )
        response = service.search(request)
    finally:
        if service is not None:
            service.close()

    if response is None:
        raise SystemExit("Search did not return a response.")

    if not response.results:
        print("[hybrid-quickstart] No results returned.")
        return 0

    top = response.results[0]
    print(
        f"[hybrid-quickstart] Top result doc_id={top.doc_id} "
        f"(fused score={top.score:.3f})"
    )
    for idx, result in enumerate(response.results, start=1):
        diagnostics = result.diagnostics
        print(
            f"  {idx:02d}. doc_id={result.doc_id} chunk={result.chunk_id} "
            f"dense={_format_score(diagnostics.dense_score)} "
            f"bm25={_format_score(diagnostics.bm25_score)} "
            f"splade={_format_score(diagnostics.splade_score)}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI utility
    raise SystemExit(main())
