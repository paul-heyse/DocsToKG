from __future__ import annotations

from dataclasses import dataclass

from DocsToKG.ContentDownload.api import DownloadPlan
from DocsToKG.ContentDownload.download_execution import (
    _cleanup_staging_artifacts,
    stream_candidate_payload,
)


@dataclass
class _FakeHeadResponse:
    status_code: int = 200
    headers: dict[str, str] | None = None


class _LeakTrackingSession:
    """Minimal session that fails if responses are not closed."""

    def __init__(
        self,
        *,
        body: bytes,
        status_code: int = 200,
        from_cache: bool = False,
        revalidated: bool = False,
    ) -> None:
        self._body = body
        self._status_code = status_code
        self._from_cache = from_cache
        self._revalidated = revalidated
        self.active_streams = 0
        self.limit = 1
        self.stream_calls = 0

    def head(self, url: str, **_: object) -> _FakeHeadResponse:  # noqa: D401
        return _FakeHeadResponse()

    def stream(self, method: str, url: str, **_: object):  # noqa: D401
        if method.upper() != "GET":
            raise AssertionError("stream_candidate_payload should use GET")

        session = self

        class _ResponseContext:
            def __init__(self) -> None:
                self._response: _StreamResponse | None = None

            def __enter__(self) -> _StreamResponse:
                session.stream_calls += 1
                session.active_streams += 1
                if session.active_streams > session.limit:
                    raise RuntimeError("connection pool exhausted")
                self._response = _StreamResponse(
                    session,
                    body=session._body,
                    status_code=session._status_code,
                    from_cache=session._from_cache,
                    revalidated=session._revalidated,
                )
                return self._response

            def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401
                if self._response is not None:
                    self._response.close()
                return False

        return _ResponseContext()

    def get(self, url: str, **_: object):  # noqa: D401
        raise AssertionError("stream_candidate_payload must call session.stream")

    def release(self) -> None:
        self.active_streams -= 1


class _StreamResponse:
    def __init__(
        self,
        session: _LeakTrackingSession,
        *,
        body: bytes,
        status_code: int,
        from_cache: bool,
        revalidated: bool,
    ) -> None:
        self._session = session
        self._body = body
        self.status_code = status_code
        self.headers = {
            "Content-Type": "application/pdf",
            "Content-Length": str(len(body)),
        }
        self.extensions = {
            "from_cache": from_cache,
            "revalidated": revalidated,
        }

    def iter_bytes(self, *, chunk_size: int):  # noqa: D401
        yield self._body

    def close(self) -> None:
        self._session.release()


def _make_plan() -> DownloadPlan:
    return DownloadPlan(
        url="https://example.org/paper.pdf",
        resolver_name="unit-test",
        expected_mime="application/pdf",
    )


def test_stream_candidate_payload_releases_connection_between_calls() -> None:
    session = _LeakTrackingSession(body=b"payload")
    plan = _make_plan()

    for _ in range(3):
        result = stream_candidate_payload(plan, session=session, telemetry=None, run_id="run")
        assert result.bytes_written == len(b"payload")
        assert session.active_streams == 0
        assert session.stream_calls >= 1
        _cleanup_staging_artifacts(result.path_tmp, result.staging_path)

    assert session.stream_calls == 3
    assert session.active_streams == 0


def test_stream_candidate_payload_closes_response_on_not_modified() -> None:
    session = _LeakTrackingSession(body=b"", status_code=304, revalidated=True)
    plan = _make_plan()

    result = stream_candidate_payload(plan, session=session, telemetry=None, run_id="run")

    assert result.http_status == 304
    assert result.path_tmp == ""
    assert session.stream_calls == 1
    assert session.active_streams == 0
