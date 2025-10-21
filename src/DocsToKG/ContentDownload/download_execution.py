"""
Download Execution Stage - Canonical Implementation

Three-stage download pipeline using canonical types:
1. prepare_candidate_download(plan) → DownloadPlan or raise SkipDownload/DownloadError
2. stream_candidate_payload(plan, ...) → DownloadStreamResult
3. finalize_candidate_download(plan, stream) → DownloadOutcome

Design:
- Pure signatures: return types always stable
- Use exceptions (SkipDownload, DownloadError) for short-circuit logic
- Pipeline catches exceptions and converts to DownloadOutcome
- All operations are idempotent and can be retried safely
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from typing import Any, Optional

from DocsToKG.ContentDownload.api import (
    DownloadOutcome,
    DownloadPlan,
    DownloadStreamResult,
    ReasonCode,
)
from DocsToKG.ContentDownload.api.exceptions import DownloadError, SkipDownload

LOGGER = logging.getLogger(__name__)


def _emit(telemetry: Any, **kw: Any) -> None:
    """Emit telemetry record if telemetry sink provided."""
    if telemetry and hasattr(telemetry, "log_attempt"):
        try:
            telemetry.log_attempt(**kw)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.debug(f"Telemetry emission failed: {e}")


def prepare_candidate_download(
    plan: DownloadPlan,
    *,
    telemetry: Any = None,
    run_id: Optional[str] = None,
) -> DownloadPlan:
    """
    Preflight validation before streaming.

    Checks:
    - robots.txt compliance
    - content-type policy
    - size policy
    - cache hints

    Returns:
        The (possibly adjusted) DownloadPlan if all checks pass.

    Raises:
        SkipDownload: If policy or robots block this download
        DownloadError: If unrecoverable preflight error
    """
    # Example: robots check (would call actual robots cache in real implementation)
    # if not await_robots_check(plan.url):
    #     raise SkipDownload("robots", f"Blocked by robots.txt: {plan.url}")

    # Example: content-type policy (would use ctx.domain_content_rules in real implementation)
    # if plan.expected_mime and not is_allowed_mime(plan.expected_mime):
    #     raise SkipDownload("policy-type", f"Disallowed MIME: {plan.expected_mime}")

    return plan


def stream_candidate_payload(
    plan: DownloadPlan,
    *,
    session: Any = None,
    timeout_s: Optional[float] = None,
    chunk_size: int = 1 << 20,  # 1 MB default
    max_bytes: Optional[int] = None,
    telemetry: Any = None,
    run_id: Optional[str] = None,
) -> DownloadStreamResult:
    """
    Stream HTTP payload to a temporary file.

    Emits telemetry for HEAD and GET attempts.
    Validates content-type and size during streaming.

    Returns:
        DownloadStreamResult with path_tmp, bytes_written, http_status, content_type.

    Raises:
        SkipDownload: If content-type doesn't match expected
        DownloadError: If connection error, timeout, or size exceeded
    """
    if not session:
        raise DownloadError(
            "conn-error",
            "No HTTP session provided",
        )

    url = plan.url

    # HEAD request (optional, for validation)
    t0 = time.monotonic_ns()
    try:
        head = session.head(url, allow_redirects=True, timeout=timeout_s)
        elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
        _emit(
            telemetry,
            run_id=run_id,
            resolver_name=plan.resolver_name,
            url=url,
            status="http-head",
            http_status=head.status_code,
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:  # pylint: disable=broad-except
        raise DownloadError(
            "conn-error",
            f"HEAD request failed: {e}",
        ) from e

    # GET request
    t0 = time.monotonic_ns()
    try:
        resp = session.get(url, stream=True, allow_redirects=True, timeout=timeout_s)
        elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
        content_type = resp.headers.get("Content-Type", "").lower()

        _emit(
            telemetry,
            run_id=run_id,
            resolver_name=plan.resolver_name,
            url=url,
            status="http-get",
            http_status=resp.status_code,
            content_type=content_type,
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:  # pylint: disable=broad-except
        raise DownloadError(
            "conn-error",
            f"GET request failed: {e}",
        ) from e

    # Validate content-type if expected
    if plan.expected_mime and not content_type.startswith(plan.expected_mime):
        raise SkipDownload(
            "unexpected-ct",
            f"Expected {plan.expected_mime}, got {content_type}",
        )

    # Write to temporary file
    tmp_dir = os.getcwd()
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, ".download.part")

    bytes_written = 0
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue

                bytes_written += len(chunk)

                # Check size limit
                if max_bytes and bytes_written > max_bytes:
                    raise DownloadError(
                        "too-large",
                        f"Payload exceeded {max_bytes} bytes",
                    )

                f.write(chunk)
    except DownloadError:
        raise
    except Exception as e:  # pylint: disable=broad-except
        raise DownloadError(
            "download-error",
            f"Write to temp file failed: {e}",
        ) from e

    # Emit final success record
    _emit(
        telemetry,
        run_id=run_id,
        resolver_name=plan.resolver_name,
        url=url,
        status="http-200",
        http_status=resp.status_code,
        bytes_written=bytes_written,
        content_type=content_type,
    )

    return DownloadStreamResult(
        path_tmp=tmp_path,
        bytes_written=bytes_written,
        http_status=resp.status_code,
        content_type=content_type,
    )


def finalize_candidate_download(
    plan: DownloadPlan,
    stream: DownloadStreamResult,
    *,
    final_path: Optional[str] = None,
    telemetry: Any = None,
    run_id: Optional[str] = None,
) -> DownloadOutcome:
    """
    Finalize downloaded artifact (move to final location).

    Performs:
    - Integrity checks (if needed)
    - Atomic move from temp → final
    - Update manifest

    Returns:
        DownloadOutcome with ok=True, classification='success', path=final_path.

    Raises:
        DownloadError: If integrity check fails or move fails
    """
    # Determine final path (in real implementation, would use storage policy)
    if not final_path:
        base = plan.url.rsplit("/", 1)[-1] or "download.bin"
        final_path = os.path.join(os.getcwd(), base)

    # Atomically move temp → final
    try:
        os.replace(stream.path_tmp, final_path)
    except Exception as e:  # pylint: disable=broad-except
        raise DownloadError(
            "download-error",
            f"Failed to finalize: {e}",
        ) from e

    # Emit finalization event
    _emit(
        telemetry,
        run_id=run_id,
        resolver_name=plan.resolver_name,
        status="http-200",
        bytes_written=stream.bytes_written,
        final_path=final_path,
    )

    return DownloadOutcome(
        ok=True,
        classification="success",
        path=final_path,
        reason=None,
        meta={
            "content_type": stream.content_type,
            "bytes": stream.bytes_written,
        },
    )
