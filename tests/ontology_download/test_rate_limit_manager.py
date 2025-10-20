from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.io import rate_limit as rate_mod
from DocsToKG.OntologyDownload.settings import DownloadConfiguration


@pytest.fixture(autouse=True)
def reset_rate_manager():
    rate_mod.reset()
    yield
    rate_mod.reset()


def test_get_bucket_normalises_keys():
    config = DownloadConfiguration()
    bucket1 = rate_mod.get_bucket(http_config=config, service="OLS", host="Example.Org")
    bucket2 = rate_mod.get_bucket(http_config=config, service="ols", host="example.org")
    assert bucket1 is bucket2


def test_service_override_rebuilds_limiter():
    config = DownloadConfiguration()
    config.rate_limits["ols"] = "2/second"
    first = rate_mod.get_bucket(http_config=config, service="ols", host="ols.example")
    config.rate_limits["ols"] = "1/second"
    second = rate_mod.get_bucket(http_config=config, service="ols", host="ols.example")
    assert first is not second


def test_shared_rate_limit_uses_sqlite_backend(tmp_path: Path):
    config = DownloadConfiguration(shared_rate_limit_dir=tmp_path)
    config.rate_limits["ols"] = "1/second"
    bucket = rate_mod.get_bucket(http_config=config, service="ols", host="shared.example")
    sqlite_path = tmp_path / "ratelimit.sqlite"
    assert sqlite_path.exists()
    assert bucket is rate_mod.get_bucket(
        http_config=config, service="ols", host="shared.example"
    )


def test_reset_clears_cached_limiters():
    config = DownloadConfiguration()
    bucket1 = rate_mod.get_bucket(http_config=config, service="ols", host="reset.example")
    rate_mod.reset()
    bucket2 = rate_mod.get_bucket(http_config=config, service="ols", host="reset.example")
    assert bucket1 is not bucket2

def test_custom_bucket_provider_bypasses_manager():
    class StubLimiter:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def consume(self, tokens: float = 1.0) -> None:
            self.calls.append(tokens)

    config = DownloadConfiguration()
    stub = StubLimiter()
    config.set_bucket_provider(lambda service, http_config, host: stub)

    handle = rate_mod.get_bucket(http_config=config, service="custom", host="example")
    assert handle is stub

    handle.consume(3.0)
    assert stub.calls == [3.0]

def test_legacy_mode_returns_legacy_bucket():
    config = DownloadConfiguration(rate_limiter="legacy")
    bucket = rate_mod.get_bucket(http_config=config, service="legacy", host="example")
    assert isinstance(bucket, getattr(rate_mod, "_LegacyTokenBucket"))
