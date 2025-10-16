# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_utilities",
#   "purpose": "Pytest coverage for ontology download utilities scenarios",
#   "sections": [
#     {
#       "id": "test_parse_rate_limit_to_rps_formats",
#       "name": "test_parse_rate_limit_to_rps_formats",
#       "anchor": "TPRLT",
#       "kind": "function"
#     },
#     {
#       "id": "test_parse_rate_limit_to_rps_invalid",
#       "name": "test_parse_rate_limit_to_rps_invalid",
#       "anchor": "PRLT1",
#       "kind": "function"
#     },
#     {
#       "id": "test_directory_size_counts_all_files",
#       "name": "test_directory_size_counts_all_files",
#       "anchor": "TDSCA",
#       "kind": "function"
#     },
#     {
#       "id": "test_directory_size_handles_empty_directory",
#       "name": "test_directory_size_handles_empty_directory",
#       "anchor": "TDSHE",
#       "kind": "function"
#     },
#     {
#       "id": "test_parse_iso_datetime_normalizes_timezone",
#       "name": "test_parse_iso_datetime_normalizes_timezone",
#       "anchor": "TPIDN",
#       "kind": "function"
#     },
#     {
#       "id": "test_parse_http_datetime_handles_gmt_header",
#       "name": "test_parse_http_datetime_handles_gmt_header",
#       "anchor": "TPHDH",
#       "kind": "function"
#     },
#     {
#       "id": "test_parse_version_timestamp_variants",
#       "name": "test_parse_version_timestamp_variants",
#       "anchor": "TPVTV",
#       "kind": "function"
#     },
#     {
#       "id": "test_infer_version_timestamp_from_composite_strings",
#       "name": "test_infer_version_timestamp_from_composite_strings",
#       "anchor": "TIVTF",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

import datetime as dt
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.ontology_download import (
    _directory_size,
    infer_version_timestamp,
    parse_http_datetime,
    parse_iso_datetime,
    parse_rate_limit_to_rps,
    parse_version_timestamp,
)


@pytest.mark.parametrize(
    "limit,expected",
    [
        ("5/second", 5.0),
        ("0.5/sec", 0.5),
        ("6/s", 6.0),
        ("120/minute", pytest.approx(2.0)),
        ("90/min", pytest.approx(1.5)),
        ("30/m", pytest.approx(0.5)),
        ("3600/hour", pytest.approx(1.0)),
        ("7200/h", pytest.approx(2.0)),
    ],
)
def test_parse_rate_limit_to_rps_formats(limit, expected):
    assert parse_rate_limit_to_rps(limit) == expected


@pytest.mark.parametrize("limit", [None, "", "ten/second", "5/century"])
def test_parse_rate_limit_to_rps_invalid(limit):
    assert parse_rate_limit_to_rps(limit) is None


def test_directory_size_counts_all_files(tmp_path: Path) -> None:
    base = tmp_path / "payload"
    base.mkdir()
    (base / "a.txt").write_bytes(b"a" * 10)
    nested = base / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"b" * 5)
    (nested / "c.bin").write_bytes(b"c" * 7)

    expected = 10 + 5 + 7
    assert _directory_size(base) == expected


def test_directory_size_handles_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert _directory_size(empty_dir) == 0


def test_parse_iso_datetime_normalizes_timezone() -> None:
    parsed = parse_iso_datetime("2024-01-01T00:00:00")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == dt.timedelta(0)


def test_parse_http_datetime_handles_gmt_header() -> None:
    parsed = parse_http_datetime("Wed, 21 Oct 2015 07:28:00 GMT")
    assert parsed is not None
    assert parsed.isoformat().endswith("+00:00")


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-01",
        "20240101",
        "2024_01_01",
        "2024/01/01",
        "2024-01-01T12:34:56",
        "20240101T123456",
    ],
)
def test_parse_version_timestamp_variants(value: str) -> None:
    parsed = parse_version_timestamp(value)
    assert parsed is not None
    assert parsed.year == 2024
    assert parsed.month == 1
    assert parsed.day == 1


@pytest.mark.parametrize(
    "value",
    [
        "release-2024-01-01",
        "build_20240101_extra",
        "snapshot-2024/01/01",
    ],
)
def test_infer_version_timestamp_from_composite_strings(value: str) -> None:
    parsed = infer_version_timestamp(value)
    assert parsed is not None
    assert parsed.year == 2024
    assert parsed.month == 1
    assert parsed.day == 1
