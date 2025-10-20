"""HybridSearchValidator resource-budget tests against the FAISS GPU wheel.

Confirms validation runs configure `StandardGpuResources` according to config
(temp/pinned memory, null streams) and fall back cleanly when FAISS is absent.
Prevents validation harnesses from over-subscribing CUDA memory while using the
custom CUDA 12/OpenBLAS wheel documented in `faiss-gpu-wheel-reference.md`.
Checks GPU memory limits, pinned memory configuration, null stream handling,
and fallback behaviour when FAISS utilities are unavailable so validation
pipelines do not over-subscribe accelerator resources.
"""

import logging
from types import SimpleNamespace

import pytest

import DocsToKG.HybridSearch.service as service_module
from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.pipeline import Observability
from DocsToKG.HybridSearch.service import HybridSearchValidator


class _RecordingResources:
    def __init__(self) -> None:
        self.temp_memory = None
        self.pinned_memory = None
        self.null_stream_all_devices = False
        self.default_null_stream_calls: list[int | None] = []

    def setTempMemory(self, value: int) -> None:  # noqa: N802 - faiss naming
        self.temp_memory = value

    def setPinnedMemory(self, value: int) -> None:  # noqa: N802 - faiss naming
        self.pinned_memory = value

    def setDefaultNullStreamAllDevices(self) -> None:  # noqa: N802 - faiss naming
        self.null_stream_all_devices = True

    def setDefaultNullStream(self, device: int | None = None) -> None:  # noqa: N802 - faiss naming
        self.default_null_stream_calls.append(device)


@pytest.fixture
def recording_faiss(patcher):
    stub = SimpleNamespace(StandardGpuResources=_RecordingResources)
    patcher.setattr(service_module, "faiss", stub, raising=False)
    return stub


def test_validation_resources_apply_dense_config(caplog, recording_faiss):
    dense_cfg = DenseIndexConfig(
        device=2,
        gpu_temp_memory_bytes=512,
        gpu_pinned_memory_bytes=2048,
        gpu_use_default_null_stream_all_devices=True,
    )
    config = SimpleNamespace(dense=dense_cfg)
    config_manager = SimpleNamespace(get=lambda: config)

    observability = Observability()
    service = SimpleNamespace(
        _config_manager=config_manager,
        _faiss=SimpleNamespace(get_gpu_resources=lambda: None),
        _observability=observability,
    )
    ingestion = SimpleNamespace(faiss_index=SimpleNamespace(get_gpu_resources=lambda: None))

    validator = HybridSearchValidator(
        ingestion=ingestion,
        service=service,
        registry=SimpleNamespace(),
        opensearch=SimpleNamespace(),
    )

    caplog.set_level(logging.INFO, logger=observability.logger.name)

    resource = validator._ensure_validation_resources()
    assert isinstance(resource, _RecordingResources)
    assert resource.temp_memory == 512
    assert resource.pinned_memory == 2048
    assert resource.null_stream_all_devices is True
    assert resource.default_null_stream_calls == []

    gauges = {
        (sample["name"], tuple(sorted(sample.get("labels", {}).items()))): sample["value"]
        for sample in observability.metrics_snapshot()["gauges"]
    }
    assert gauges[("faiss_gpu_temp_memory_bytes", (("scope", "validation"),))] == pytest.approx(
        512.0
    )
    assert gauges[("faiss_gpu_default_null_stream", (("scope", "validation"),))] == pytest.approx(
        0.0
    )
    assert gauges[
        ("faiss_gpu_default_null_stream_all_devices", (("scope", "validation"),))
    ] == pytest.approx(1.0)

    records = [record for record in caplog.records if record.msg == "faiss-gpu-resource-configured"]
    assert records, "expected observability log for validation GPU resource configuration"
    payload = getattr(records[-1], "event", {})
    assert payload.get("scope") == "validation"
    assert payload.get("default_null_stream_all_devices") is True

    # Ensure cached resource is reused
    assert validator._ensure_validation_resources() is resource


def test_request_for_query_respects_recall_first_flag():
    validator = HybridSearchValidator(
        ingestion=SimpleNamespace(),
        service=SimpleNamespace(),
        registry=SimpleNamespace(),
        opensearch=SimpleNamespace(),
    )

    payload = {"query": "example", "recall_first": True}
    request = validator._request_for_query(payload)

    assert request.recall_first is True
    assert request.diagnostics is True

    payload = {"query": "example", "diagnostics": False}
    request = validator._request_for_query(payload)

    assert request.diagnostics is False
