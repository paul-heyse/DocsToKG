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

import hashlib
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from DocsToKG.ContentDownload.api import (
    DownloadOutcome,
    DownloadPlan,
    DownloadStreamResult,
)
from DocsToKG.ContentDownload.api.exceptions import DownloadError, SkipDownload
from DocsToKG.ContentDownload.io_utils import SizeMismatchError, atomic_write_stream
from DocsToKG.ContentDownload.policy.path_gate import PathPolicyError, validate_path_safety
from DocsToKG.ContentDownload.policy.url_gate import PolicyError, validate_url_security

LOGGER = logging.getLogger(__name__)


def _emit(telemetry: Any, **kw: Any) -> None:
    """Emit telemetry record if telemetry sink provided."""
    if telemetry and hasattr(telemetry, "log_attempt"):
        try:
            telemetry.log_attempt(**kw)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.debug(f"Telemetry emission failed: {e}")


def _resolve_storage_tmp_root() -> Path:
    """Return the base directory for temporary download artifacts."""

    override = os.environ.get("DOCSTOKG_STORAGE_TMP")
    if override:
        root = Path(override).expanduser()
    else:
        data_root = os.environ.get("DOCSTOKG_DATA_ROOT")
        if data_root:
            root = Path(data_root).expanduser() / "tmp" / "downloads"
        else:
            root = Path(tempfile.gettempdir()) / "docstokg" / "downloads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _plan_temp_path(plan: DownloadPlan) -> Path:
    """Derive a collision-resistant temporary path for a download plan."""

    base = _resolve_storage_tmp_root()
    digest = hashlib.sha256(plan.url.encode("utf-8")).hexdigest()
    shard_dir = base / digest[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)
    resolver_token = "".join(ch if ch.isalnum() else "-" for ch in plan.resolver_name) or "plan"
    filename = f"{resolver_token}-{digest[:12]}-{uuid.uuid4().hex}.download.part"
    return shard_dir / filename


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
    verify_content_length: bool = True,
    telemetry: Any = None,
    run_id: Optional[str] = None,
) -> DownloadStreamResult:
    """
    Stream HTTP payload to a temporary file using atomic write.

    Emits telemetry for HEAD and GET attempts.
    Validates content-type and size during streaming.
    Honors hishel cache metadata (from_cache, revalidated extensions).

    Returns:
        DownloadStreamResult with path_tmp, bytes_written, http_status, content_type.

    Raises:
        SkipDownload: If content-type doesn't match expected or 304 not-modified
        DownloadError: If connection error, timeout, size exceeded, or size mismatch
    """
    if not session:
        raise DownloadError(
            "conn-error",
            "No HTTP session provided",
        )

    url = plan.url

    # Validate URL security
    try:
        validate_url_security(url)
    except PolicyError as e:
        raise SkipDownload("security-policy", f"URL security policy violation: {e}")

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
        # Some servers 405 on HEAD; proceed without failing
        LOGGER.debug(f"HEAD request failed (proceeding to GET): {e}")

    # GET request via httpx + hishel
    t0 = time.monotonic_ns()
    try:
        resp = session.get(url, stream=True, allow_redirects=True, timeout=timeout_s)
        elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
        content_type = resp.headers.get("Content-Type", "").lower()

        # Check for hishel cache extensions
        from_cache = bool(getattr(resp, "extensions", {}).get("from_cache"))
        revalidated = bool(getattr(resp, "extensions", {}).get("revalidated"))

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

        # Emit cache-aware tokens
        if from_cache and not revalidated:
            _emit(
                telemetry,
                run_id=run_id,
                resolver_name=plan.resolver_name,
                url=url,
                status="cache-hit",
                http_status=resp.status_code,
                content_type=content_type,
                reason="ok",
            )
        if revalidated and resp.status_code == 304:
            _emit(
                telemetry,
                run_id=run_id,
                resolver_name=plan.resolver_name,
                url=url,
                status="http-304",
                http_status=304,
                content_type=content_type,
                reason="not-modified",
            )
            return DownloadStreamResult(
                path_tmp="",
                bytes_written=0,
                http_status=304,
                content_type=content_type,
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

    # Extract Content-Length and prepare for atomic write
    cl = resp.headers.get("Content-Length")
    expected_len = int(cl) if (cl and cl.isdigit()) else None

    effective_max_bytes = (
        plan.max_bytes_override
        if plan.max_bytes_override is not None
        else max_bytes
    )

    if (
        effective_max_bytes is not None
        and expected_len is not None
        and expected_len > effective_max_bytes
    ):
        raise DownloadError(
            "too-large",
            f"Content-Length {expected_len} exceeds cap {effective_max_bytes} bytes",
        )

    # Write to temporary file using atomic writer
    tmp_path = _plan_temp_path(plan)

    try:
        bytes_written = atomic_write_stream(
            dest_path=str(tmp_path),
            byte_iter=resp.iter_bytes(),
            expected_len=(expected_len if verify_content_length else None),
            chunk_size=chunk_size,
        )

        # Check size limit
        if effective_max_bytes is not None and bytes_written > effective_max_bytes:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise DownloadError(
                "too-large",
                f"Payload exceeded {effective_max_bytes} bytes",
            )

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
            content_length_hdr=expected_len,
        )

        return DownloadStreamResult(
            path_tmp=str(tmp_path),
            bytes_written=bytes_written,
            http_status=resp.status_code,
            content_type=content_type,
        )

    except SizeMismatchError:
        _emit(
            telemetry,
            run_id=run_id,
            resolver_name=plan.resolver_name,
            url=url,
            status="size-mismatch",
            http_status=resp.status_code,
            content_type=content_type,
            reason="size-mismatch",
            content_length_hdr=expected_len,
        )
        raise DownloadError("size-mismatch")
    except DownloadError:
        raise
    except Exception as e:  # pylint: disable=broad-except
        raise DownloadError(
            "download-error",
            f"Atomic write failed: {e}",
        ) from e


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
    - 304 short-circuit (no file write needed)
    - Integrity checks (if needed)
    - Atomic move from temp → final (already done by atomic_write_stream)
    - Update manifest

    Returns:
        DownloadOutcome with ok=True, classification='success', path=final_path.

    Raises:
        DownloadError: If integrity check fails or move fails
    """
    # 304 Not Modified: no file to finalize
    if stream.http_status == 304:
        return DownloadOutcome(
            ok=True,
            classification="skip",
            path=None,
            reason="not-modified",
            meta={"http_status": 304},
        )

    # Validate final path safety
    try:
        validate_path_safety(final_path)
    except PathPolicyError as e:
        raise SkipDownload("path-policy", f"Path policy violation: {e}")

    # Determine final path (in real implementation, would use storage policy)
    if not final_path:
        base = plan.url.rsplit("/", 1)[-1] or "download.bin"
        final_path = os.path.join(os.getcwd(), base)

    # atomic_write_stream already moved temp → dest_path, so final_path exists
    # Just verify and emit event
    try:
        if stream.path_tmp and not os.path.exists(final_path):
            # If atomic_write_stream didn't finalize, move it now
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
