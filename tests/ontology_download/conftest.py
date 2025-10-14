import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace


class _StubRequestsModule(SimpleNamespace):
    """Lightweight stand-in for the requests module used in tests."""


class _RequestException(Exception):
    pass


class _HTTPError(_RequestException):
    def __init__(self, message: str = "", response=None) -> None:
        super().__init__(message)
        self.response = response


class _Timeout(_RequestException):
    pass


class _ConnectionError(_RequestException):
    pass


class _SSLError(_RequestException):
    pass


class _StubSession:
    def get(self, *args, **kwargs):  # pragma: no cover - patched in tests
        raise RuntimeError("requests.Session stub used without monkeypatching")


class _StubResponse:
    def __init__(self) -> None:
        self.status_code = None
        self.headers = {}
        self.url = ""


class _PlaceholderClient:
    def __init__(self, *args, **kwargs):  # pragma: no cover - should be patched in tests
        pass

    def __getattr__(self, item):  # pragma: no cover - should be replaced in tests
        raise RuntimeError(f"Accessed placeholder client attribute '{item}' without patching")


if "bioregistry" not in sys.modules:
    sys.modules["bioregistry"] = SimpleNamespace(
        get_obo_download=lambda prefix: None,
        get_owl_download=lambda prefix: None,
        get_rdf_download=lambda prefix: None,
    )

if "ols_client" not in sys.modules:
    sys.modules["ols_client"] = SimpleNamespace(OlsClient=_PlaceholderClient)

if "ontoportal_client" not in sys.modules:
    sys.modules["ontoportal_client"] = SimpleNamespace(BioPortalClient=_PlaceholderClient)

if "requests" not in sys.modules:
    sys.modules["requests"] = _StubRequestsModule(
        Session=_StubSession,
        HTTPError=_HTTPError,
        Timeout=_Timeout,
        ConnectionError=_ConnectionError,
        RequestException=_RequestException,
        Response=_StubResponse,
        exceptions=SimpleNamespace(SSLError=_SSLError),
    )

if "pystow" not in sys.modules:
    class _StubPystow(SimpleNamespace):
        def join(self, *segments):
            root = Path(os.environ.get("PYSTOW_HOME", Path.home() / ".data"))
            return root.joinpath(*segments)

    sys.modules["pystow"] = _StubPystow()

if "psutil" not in sys.modules:
    class _StubProcess:
        def memory_info(self):
            return SimpleNamespace(rss=0)

    sys.modules["psutil"] = SimpleNamespace(Process=lambda: _StubProcess())

if "pooch" not in sys.modules:
    class _HTTPDownloader:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, url, output_file, pooch_logger):  # pragma: no cover - overridden in tests
            raise NotImplementedError("pooch downloader stub should be overridden")

    def _retrieve(url, *, path, fname, downloader, **kwargs):
        cache_dir = Path(path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        output = cache_dir / fname
        downloader(url, str(output), logging.getLogger("pooch"))
        return str(output)

    sys.modules["pooch"] = SimpleNamespace(HTTPDownloader=_HTTPDownloader, retrieve=_retrieve)
