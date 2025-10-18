# Mypy Baseline — Content Download Suite

## Command Output

```
pre-commit run mypy --files tests/content_download/*.py src/DocsToKG/ContentDownload/telemetry.py
```

Output captured on 2025-10-18T14:05:55Z is stored verbatim in `/tmp/mypy_baseline.txt` for reference.

## Diagnostic Themes

| Location | Message (abridged) | Theme |
| --- | --- | --- |
| tests/content_download/test_network_unit.py:91 | return value `tuple[...]` incompatible with `Mock` | Return type mismatch |
| src/DocsToKG/ContentDownload/telemetry.py:139 | assign `str | None` to `str` variable | Optional field handling |
| src/DocsToKG/ContentDownload/telemetry.py:146 | assign `str | None` to `str` variable | Optional field handling |
| src/DocsToKG/ContentDownload/telemetry.py:156 | compare `int` with `None` via `>` | Optional arithmetic guard |
| src/DocsToKG/ContentDownload/telemetry.py:167 | compare `int` with `None` via `>` | Optional arithmetic guard |
| src/DocsToKG/ContentDownload/telemetry.py:501 | `reason_detail` expects `str | None` not `ReasonCode | str | None` | Telemetry API signature |
| tests/content_download/test_download_strategy_helpers.py:6 | missing stubs for `requests` | Missing third-party stubs |
| tests/content_download/test_download_strategy_helpers.py:7 | missing stubs for `requests.structures` | Missing third-party stubs |
| tests/content_download/test_download_strategy_helpers.py:22 | generator annotated with `bytes` return | Generator typing |
| tests/content_download/test_contract_surfaces.py:8 | missing stubs for `requests` | Missing third-party stubs |
| tests/content_download/test_contract_surfaces.py:175 | `_CaptureLogger` missing AttemptSink protocol members | AttemptSink protocol conformance |
| tests/content_download/test_runner_download_run.py:7 | missing stubs for `requests` | Missing third-party stubs |
| tests/content_download/test_runner_download_run.py:53 | `dict[str, object]` to `MutableMapping.update` invalid | MutableMapping update typing |
| tests/content_download/test_runner_download_run.py:225 | assign `StubProvider` to `WorkProvider | None` | WorkProvider protocol conformance |
| tests/content_download/test_runner_download_run.py:226 | return `StubProvider` where `WorkProvider` required | WorkProvider protocol conformance |
| tests/content_download/test_regression_compatibility.py:76 | `classification` passed as `str` | Classification enum alignment |
| tests/content_download/test_regression_compatibility.py:215 | `SimpleNamespace` passed to `load_resolver_config` | Argument type mismatch |
| tests/content_download/test_networking.py:597 | missing stubs for `requests` | Missing third-party stubs |
| tests/content_download/test_networking.py:1098 | `classification` passed as `str` | Classification enum alignment |
| tests/content_download/test_networking.py:1428 | `_make_artifact` redefined | Helper duplication |
| tests/content_download/test_networking.py:2192 | `pyalex` missing attribute `Topics` | PyAlex fake coverage |
| tests/content_download/test_networking.py:2193 | `pyalex` missing attribute `Works` | PyAlex fake coverage |
| tests/content_download/test_networking.py:2195 | `pyalex` missing attribute `mailto` | PyAlex fake coverage |
| tests/content_download/test_networking.py:2196 | `pyalex` missing attribute `config` | PyAlex fake coverage |
| tests/content_download/test_networking.py:3297 | list element of type `None` where `str` expected | Optional string normalization |
| tests/content_download/test_networking.py:3331 | `_make_artifact` redefined | Helper duplication |
| tests/content_download/test_networking.py:3381 | unexpected kw arg `pdf_urls` | Helper signature mismatch |
| tests/content_download/test_networking.py:3416 | `RunTelemetry` abstract (missing context manager) | RunTelemetry context management |
| tests/content_download/test_networking.py:3438 | `classification` passed as `str` | Classification enum alignment |
| tests/content_download/test_networking.py:3505 | `RunTelemetry` abstract (missing context manager) | RunTelemetry context management |
| tests/content_download/test_networking.py:3537 | Resolver callback typed `Callable[..., None]` | ResolverPipeline callback typing |
| tests/content_download/test_networking.py:3571 | `RunTelemetry` abstract (missing context manager) | RunTelemetry context management |
| tests/content_download/test_networking.py:3641 | `classification` passed as `str` | Classification enum alignment |
| tests/content_download/test_networking.py:3669 | `ListLogger` lacks AttemptSink context methods | AttemptSink protocol conformance |
| tests/content_download/test_networking.py:3706 | optional URL passed to `normalize_url` | Optional string normalization |
| tests/content_download/test_networking.py:3747 | optional URL passed to `ManifestUrlIndex.get` | Optional string normalization |
| tests/content_download/test_networking.py:3810 | optional URL passed to `normalize_url` | Optional string normalization |
| tests/content_download/test_atomic_writes.py:198 | `_DummySession.head` returns `_DummyHeadResponse` | Response class hierarchy |
| tests/content_download/test_atomic_writes.py:239 | `_download_with_session` returns 4-tuple, annotated 3 | Helper return arity |
| tests/content_download/test_atomic_writes.py:244 | unpack expects 4 values | Helper return arity |
| tests/content_download/test_atomic_writes.py:258 | unpack expects 4 values | Helper return arity |
| tests/content_download/test_atomic_writes.py:353 | fake `docling_core.persistence.manifest_append` missing | Fake dependency coverage |
| tests/content_download/test_atomic_writes.py:355 | fake `docling_core.persistence.manifest_load` missing | Fake dependency coverage |
| tests/content_download/test_atomic_writes.py:357 | fake `docling_core.serializers.RichSerializerProvider` missing | Fake dependency coverage |
| tests/content_download/test_atomic_writes.py:467 | `DummyHybridChunker` tokenizer arg is `None` | Chunker initialization typing |

## Stub Dependencies Added

- `types-requests` — provides typing coverage for `requests` and `requests.structures`, eliminating the missing stub diagnostics
  observed in the baseline run across networking-focused tests.
