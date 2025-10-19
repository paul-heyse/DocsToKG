"""Tests for lazy, ordered file listing helpers."""

from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from DocsToKG.DocParsing.doctags import list_htmls, list_pdfs


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data", encoding="utf-8")


def test_list_pdfs_returns_sorted_iterator(tmp_path: Path) -> None:
    """PDF listings should yield lexicographically sorted paths lazily."""

    expected_relatives = [
        "b/report.pdf",
        "a/report.pdf",
        "a/notes/part.pdf",
        "a/notes/alpha.pdf",
        "alpha.pdf",
    ]
    for relative in expected_relatives:
        _write(tmp_path / relative)

    # Non-PDFs must be ignored even if nested alongside inputs.
    _write(tmp_path / "a/notes/ignore.txt")

    iterator = list_pdfs(tmp_path)
    assert isinstance(iterator, Iterator)

    expected = [
        (tmp_path / rel).relative_to(tmp_path).as_posix()
        for rel in sorted(expected_relatives)
    ]
    actual = [path.relative_to(tmp_path).as_posix() for path in iterator]

    assert actual == expected


def test_list_htmls_filters_and_sorts(tmp_path: Path) -> None:
    """HTML listings should apply filtering and preserve deterministic ordering."""

    html_files = [
        "docs/a/input.html",
        "docs/a/sub/INDEX.HTML",
        "docs/aa.htm",
        "docs/b/page.xhtml",
    ]
    excluded = [
        "docs/a/sub/page.normalized.html",
        "docs/b/ignore.txt",
    ]

    for relative in html_files + excluded:
        _write(tmp_path / relative)

    iterator = list_htmls(tmp_path)
    assert isinstance(iterator, Iterator)

    expected = [
        (tmp_path / rel).relative_to(tmp_path).as_posix()
        for rel in sorted(html_files)
    ]
    actual = [path.relative_to(tmp_path).as_posix() for path in iterator]

    assert actual == expected


def test_list_generators_scale_streaming(tmp_path: Path) -> None:
    """Large directory trees should be consumed incrementally without materialising lists."""

    total = 512
    for idx in range(total):
        _write(tmp_path / f"group{idx:04d}/doc{idx:04d}.pdf")
        _write(tmp_path / f"group{idx:04d}/doc{idx:04d}.html")

    pdf_iter = list_pdfs(tmp_path)
    assert isinstance(pdf_iter, Iterator)

    head = list(islice(pdf_iter, 5))
    assert len(head) == 5

    remaining = sum(1 for _ in pdf_iter)
    assert len(head) + remaining == total

    html_count = sum(1 for _ in list_htmls(tmp_path))
    assert html_count == total

    # Ensure normalized HTML derivatives are never yielded even when plentiful.
    _write(tmp_path / "group9999/doc9999.normalized.html")
    assert all(
        not path.name.endswith(".normalized.html") for path in list_htmls(tmp_path)
    )
