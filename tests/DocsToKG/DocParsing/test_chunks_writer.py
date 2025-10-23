import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from DocsToKG.DocParsing.storage import parquet_schemas, paths
from DocsToKG.DocParsing.storage.chunks_writer import ParquetChunksWriter


def _make_row(idx: int) -> dict[str, object]:
    text = f"chunk text {idx}"
    return {
        "doc_id": f"doc-{idx // 10}",
        "chunk_id": idx,
        "text": text,
        "tokens": idx,
        "span": {"start": 0, "end": len(text)},
        "created_at": datetime.now(UTC),
        "schema_version": parquet_schemas.SCHEMA_VERSION_CHUNKS,
    }


def _row_iter(count: int) -> Iterable[dict[str, object]]:
    for i in range(count):
        yield _make_row(i)


def test_parquet_chunks_writer_streaming_batches(tmp_path: Path) -> None:
    writer = ParquetChunksWriter(batch_size=128)
    rel_id = "test/streaming"
    row_count = 1025

    result = writer.write(
        rows_iter=_row_iter(row_count),
        data_root=tmp_path,
        rel_id=rel_id,
        cfg_hash="abc123",
        created_by="pytest",
        dt_utc=datetime(2024, 1, 1, tzinfo=UTC),
    )

    assert result.rows_written == row_count
    output_path = result.paths[0]
    assert output_path.exists()

    parquet_file = pq.ParquetFile(output_path)
    assert result.row_group_count == parquet_file.metadata.num_row_groups
    assert parquet_file.metadata.num_row_groups >= math.ceil(row_count / writer.batch_size)

    footer = parquet_schemas.read_parquet_footer(str(output_path))
    assert footer["docparse.cfg_hash"] == "abc123"
    assert footer["docparse.created_by"] == "pytest"
    assert footer["docparse.created_at"] == "2024-01-01T00:00:00Z"

    manifest_path = paths.chunks_output_path(
        tmp_path, rel_id, fmt="parquet", ts=datetime(2024, 1, 1, tzinfo=UTC)
    )
    assert output_path == manifest_path
