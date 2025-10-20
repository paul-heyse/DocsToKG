"""Validation tests for :mod:`DocsToKG.DocParsing.doctags` configuration objects."""

from __future__ import annotations

import pytest

from DocsToKG.DocParsing.doctags import DoctagsCfg


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_doctags_cfg_rejects_out_of_range_gpu_memory(tmp_path, value):
    """``DoctagsCfg.finalize`` should reject GPU utilization outside the [0, 1] range."""

    cfg = DoctagsCfg(
        data_root=tmp_path,
        input=tmp_path / "input",
        output=tmp_path / "output",
        gpu_memory_utilization=value,
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.finalize()

    message = str(excinfo.value)
    assert "gpu_memory_utilization must be between 0.0 and 1.0 (inclusive)" in message
    assert f"received {float(value)}" in message
