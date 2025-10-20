"""Regression coverage for DocTags CLI path resolution."""

from __future__ import annotations

import os
from pathlib import Path

from DocsToKG.DocParsing.core.cli import _resolve_doctags_paths, build_doctags_parser


def test_doctags_cli_respects_explicit_data_root(tmp_path: Path, patcher) -> None:
    """`--data-root` should not append an extra ``/Data`` segment."""

    data_root = tmp_path / "Data"
    data_root.mkdir()

    patcher.delitem(os.environ, "DOCSTOKG_DATA_ROOT", raising=False)

    parser = build_doctags_parser()
    args = parser.parse_args(["--data-root", str(data_root), "--mode", "html"])

    mode, input_dir, output_dir, resolved_root = _resolve_doctags_paths(args)

    assert mode == "html"
    assert resolved_root == str(data_root.resolve())
    assert input_dir == (data_root / "HTML").resolve()
    assert output_dir == (data_root / "DocTagsFiles").resolve()
