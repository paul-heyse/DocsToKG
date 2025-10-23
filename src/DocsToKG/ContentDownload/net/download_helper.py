# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.net.download_helper",
#   "purpose": "Download helpers for ContentDownload pipeline.",
#   "sections": [
#     {
#       "id": "downloaderror",
#       "name": "DownloadError",
#       "anchor": "class-downloaderror",
#       "kind": "class"
#     },
#     {
#       "id": "stream-download-to-file",
#       "name": "stream_download_to_file",
#       "anchor": "function-stream-download-to-file",
#       "kind": "function"
#     },
#     {
#       "id": "head-request",
#       "name": "head_request",
#       "anchor": "function-head-request",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Download helpers for ContentDownload pipeline.

Provides high-level download functions integrated with:
- HTTPX client singleton
- Audited redirects
- Streaming to temp files
- Atomic promotion
- Telemetry emission
- Error handling
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from .client import _normalize_host_for_telemetry, get_http_client, request_with_redirect_audit
from .instrumentation import (
    CacheStatus,
    NetRequestEventBuilder,
    RequestStatus,
    get_net_request_emitter,
)

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Download operation failed."""

    pass


def stream_download_to_file(
    config: Any,
    url: str,
    dest: Path,
    *,
    service: str | None = None,
    role: str | None = None,
    chunk_size: int = 1024 * 1024,
    expected_length: int | None = None,
) -> Path:
    """
    Download URL to file with streaming, atomic promotion, and telemetry.

    Args:
        config: ContentDownloadConfig
        url: URL to download
        dest: Destination file path
        service: Service name (for telemetry)
        role: Role name (for telemetry)
        chunk_size: Stream chunk size
        expected_length: Expected Content-Length (for verification)

    Returns:
        Path to downloaded file

    Raises:
        DownloadError: If download fails
    """
    client = get_http_client(config)
    emitter = get_net_request_emitter()
    request_id = os.urandom(8).hex()

    # Get host for telemetry
    try:
        host = _normalize_host_for_telemetry(url)
    except Exception:
        host = "unknown"

    builder = NetRequestEventBuilder(request_id)
    builder.with_request("GET", url, host)
    builder.with_context(service=service, role=role)

    # Ensure destination directory exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file in same directory (atomic rename)
    temp_path: Path | None = None
    resp: httpx.Response | None = None
    cache_status = CacheStatus.MISS
    bytes_written = 0

    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".tmp-", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)

            # Send request with audited redirects; STREAM
            try:
                resp = request_with_redirect_audit(
                    client,
                    "GET",
                    url,
                    stream=True,
                )
                resp.raise_for_status()
            except httpx.HTTPError as e:
                status_code = 0
                http_version = "HTTP/1.1"

                # Extract response info if available in exception
                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                    http_version = e.response.http_version
                    error_type = "http_error"
                    error_status = RequestStatus.ERROR
                    e.response.close()
                else:
                    error_type = "network_error"
                    error_status = RequestStatus.NETWORK_ERROR
                    if resp is not None:
                        resp.close()

                builder.with_response(status_code, http_version)
                builder.with_error(error_type, str(e), error_status)
                emitter.emit(builder.build())
                raise DownloadError(f"HTTP error: {e}") from e

            if resp.status_code == 304:
                cache_status = CacheStatus.REVALIDATED
            elif resp.extensions.get("from_cache"):
                cache_status = CacheStatus.HIT

            # Update response metadata
            builder.with_response(resp.status_code, resp.http_version, http2=False)
            builder.with_cache(cache_status, from_cache=cache_status != CacheStatus.MISS)

            # Get content length
            content_length = int(resp.headers.get("Content-Length", 0) or 0)
            builder.with_data(bytes_read=content_length)

            # Stream to temp file
            try:
                for chunk in resp.iter_bytes(chunk_size=chunk_size):
                    if chunk:
                        tmp_file.write(chunk)
                        bytes_written += len(chunk)

                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            except Exception as e:
                builder.with_error("write_error", str(e), RequestStatus.ERROR)
                emitter.emit(builder.build())
                raise DownloadError(f"Write error: {e}") from e
            finally:
                if resp is not None:
                    resp.close()

        # Verify Content-Length if provided
        if expected_length and bytes_written != expected_length:
            error_msg = f"Size mismatch: expected {expected_length}, got {bytes_written}"
            builder.with_error("size_mismatch", error_msg, RequestStatus.ERROR)
            emitter.emit(builder.build())
            raise DownloadError(error_msg)

        builder.with_data(bytes_read=bytes_written, bytes_written=bytes_written)

        # Atomic rename
        try:
            if temp_path is None:
                raise DownloadError("Temporary path was not created")
            os.replace(temp_path, dest)
            dir_fd = os.open(str(dest.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)  # Sync directory
            finally:
                os.close(dir_fd)
        except Exception as e:
            builder.with_error("rename_error", str(e), RequestStatus.ERROR)
            emitter.emit(builder.build())
            raise DownloadError(f"Rename error: {e}") from e

        # Success
        builder.with_response(resp.status_code, resp.http_version)
        builder.with_cache(cache_status)
        emitter.emit(builder.build())

        logger.info(f"Downloaded {url} → {dest} ({bytes_written} bytes)")
        return dest

    except Exception:
        # Cleanup temp file on error
        try:
            if temp_path and temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
        raise


def head_request(
    config: Any,
    url: str,
    *,
    service: str | None = None,
    role: str | None = None,
) -> httpx.Response:
    """
    Send HEAD request with audited redirects and telemetry.

    Args:
        config: ContentDownloadConfig
        url: URL to request
        service: Service name (for telemetry)
        role: Role name (for telemetry)

    Returns:
        HTTP response

    Raises:
        httpx.HTTPError: If request fails
    """
    client = get_http_client(config)
    emitter = get_net_request_emitter()
    request_id = os.urandom(8).hex()

    try:
        host = _normalize_host_for_telemetry(url)
    except Exception:
        host = "unknown"

    builder = NetRequestEventBuilder(request_id)
    builder.with_request("HEAD", url, host)
    builder.with_context(service=service, role=role)

    try:
        resp = request_with_redirect_audit(client, "HEAD", url)
        resp.raise_for_status()

        cache_status = CacheStatus.MISS
        if resp.status_code == 304:
            cache_status = CacheStatus.REVALIDATED
        elif resp.extensions.get("from_cache"):
            cache_status = CacheStatus.HIT

        builder.with_response(resp.status_code, resp.http_version, http2=False)
        builder.with_cache(cache_status, from_cache=cache_status != CacheStatus.MISS)
        builder.with_data(bytes_read=int(resp.headers.get("Content-Length", 0) or 0))

        emitter.emit(builder.build())
        return resp

    except httpx.HTTPError as e:
        status_code = 0
        http_version = "HTTP/1.1"

        # Extract response info if available in exception
        if isinstance(e, httpx.HTTPStatusError):
            status_code = e.response.status_code
            http_version = e.response.http_version
            error_type = "http_error"
            error_status = RequestStatus.ERROR
        else:
            error_type = "network_error"
            error_status = RequestStatus.NETWORK_ERROR

        builder.with_response(status_code, http_version)
        builder.with_error(error_type, str(e), error_status)
        emitter.emit(builder.build())
        raise
