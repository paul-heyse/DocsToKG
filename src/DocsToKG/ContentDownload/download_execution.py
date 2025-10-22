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
import tempfile
import time
from dataclasses import replace
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from DocsToKG.ContentDownload.api import (
    AttemptRecord,
    DownloadOutcome,
    DownloadPlan,
    DownloadStreamResult,
)
from DocsToKG.ContentDownload.api.exceptions import DownloadError, SkipDownload
from DocsToKG.ContentDownload.io_utils import SizeMismatchError, atomic_write_stream
from DocsToKG.ContentDownload.policy.path_gate import PathPolicyError, validate_path_safety
from DocsToKG.ContentDownload.policy.url_gate import PolicyError, validate_url_security
from DocsToKG.ContentDownload.robots import RobotsCache
from DocsToKG.ContentDownload.telemetry import (
    ATTEMPT_REASON_NOT_MODIFIED,
    ATTEMPT_REASON_OK,
    ATTEMPT_REASON_SIZE_MISMATCH,
    ATTEMPT_STATUS_CACHE_HIT,
    ATTEMPT_STATUS_HTTP_200,
    ATTEMPT_STATUS_HTTP_304,
    ATTEMPT_STATUS_HTTP_GET,
    ATTEMPT_STATUS_HTTP_HEAD,
    ATTEMPT_STATUS_SIZE_MISMATCH,
)

LOGGER = logging.getLogger(__name__)


def _emit(
    telemetry: Any,
    *,
    run_id: Optional[str],
    resolver_name: str,
    url: str,
    status: str,
    http_status: Optional[int] = None,
    elapsed_ms: Optional[int] = None,
    meta: Optional[dict[str, Any]] = None,
    **extra_meta: Any,
) -> None:
    """Emit telemetry record if telemetry sink provided."""
    if not telemetry or not hasattr(telemetry, "log_attempt"):
        return

    payload: dict[str, Any] = {}
    if meta:
        payload.update(meta)
    if extra_meta:
        payload.update(extra_meta)

    try:
        telemetry.log_attempt(
            AttemptRecord(
                run_id=str(run_id or ""),
                resolver_name=resolver_name,
                url=url,
                status=status,
                http_status=http_status,
                elapsed_ms=elapsed_ms,
                meta=payload,
            )
        )
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.debug(f"Telemetry emission failed: {e}")


def _ctx_get(ctx: Any, name: str, default: Any = None) -> Any:
    """Return attribute ``name`` from ``ctx`` supporting mapping and attr access."""

    if ctx is None:
        return default

    if hasattr(ctx, name):
        return getattr(ctx, name)

    if isinstance(ctx, Mapping):
        if name in ctx:
            return ctx[name]

    getter = getattr(ctx, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except TypeError:
            pass

    return default


def _infer_user_agent(ctx: Any) -> str:
    """Infer user-agent string from context or fall back to default."""

    default = "DocsToKG/ContentDownload"

    http_config = _ctx_get(ctx, "http_config", None)
    if http_config and hasattr(http_config, "user_agent"):
        return getattr(http_config, "user_agent")

    candidate = _ctx_get(ctx, "user_agent", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate

    extra = _ctx_get(ctx, "extra", None)
    if isinstance(extra, Mapping):
        ua = extra.get("user_agent")
        if isinstance(ua, str) and ua.strip():
            return ua

    return default


def _resolve_domain_policy(ctx: Any, url: str) -> Optional[Mapping[str, Any]]:
    """Return domain-specific policy mapping for ``url`` if configured."""

    rules = _ctx_get(ctx, "domain_content_rules", None)
    if not isinstance(rules, Mapping):
        return None

    host = urlsplit(url).hostname
    if not host:
        return None

    host_key = host.lower()
    entry = rules.get(host_key)
    if entry is None and host_key.startswith("www."):
        entry = rules.get(host_key[4:])
    if entry is None:
        # Also try without leading www when original host lacked it
        alt = f"www.{host_key}"
        entry = rules.get(alt)
    if isinstance(entry, Mapping):
        return entry
    return None


def _resolve_plan_hints(ctx: Any, plan: DownloadPlan) -> Mapping[str, Any]:
    """Return resolver-provided hints for ``plan`` if available."""

    hints = _ctx_get(ctx, "resolver_hints", None)
    if not isinstance(hints, Mapping):
        return {}

    url_hint = hints.get(plan.url)
    if isinstance(url_hint, Mapping):
        return url_hint

    resolver_hint = hints.get(plan.resolver_name)
    if isinstance(resolver_hint, Mapping):
        return resolver_hint

    return {}


def _coerce_allowed_types(policy: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract a normalized tuple of allowed MIME prefixes from policy mapping."""

    if not isinstance(policy, Mapping):
        return ()

    for key in ("allowed_types", "allowed_mime", "allow"):
        value = policy.get(key)
        if value:
            if isinstance(value, str):
                return (value.lower(),)
            if isinstance(value, (list, tuple, set)):
                return tuple(str(item).lower() for item in value if item)
    return ()


def _effective_max_bytes(plan: DownloadPlan, ctx: Any) -> Optional[int]:
    """Return the effective max-bytes limit considering plan overrides and context."""

    if plan.max_bytes_override is not None:
        return plan.max_bytes_override

    ctx_limit = _ctx_get(ctx, "max_bytes", None)
    if isinstance(ctx_limit, int):
        return ctx_limit

    download_policy = _ctx_get(ctx, "download_policy", None)
    if download_policy and hasattr(download_policy, "max_bytes"):
        limit = getattr(download_policy, "max_bytes")
        if isinstance(limit, int):
            return limit

    return None


def prepare_candidate_download(
    plan: DownloadPlan,
    *,
    session: Any = None,
    ctx: Any = None,
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
    url = plan.url

    # Validate URL security and normalize if necessary
    try:
        normalized_url = validate_url_security(url, _ctx_get(ctx, "http_config", None))
    except PolicyError as e:
        raise SkipDownload("policy-type", f"URL security policy violation: {e}") from e

    if normalized_url != url:
        plan = replace(plan, url=normalized_url)
        url = normalized_url

    # Apply robots guard if a session is available
    robots_checker = _ctx_get(ctx, "robots_checker", None)
    if robots_checker is None and session is not None:
        robots_checker = RobotsCache()

    if robots_checker is not None and session is not None:
        user_agent = _infer_user_agent(ctx)
        if not robots_checker.is_allowed(session, url, user_agent):
            raise SkipDownload("robots", f"Blocked by robots.txt: {url}")

    # Content policy enforcement via domain rules
    domain_policy = _resolve_domain_policy(ctx, url)
    allowed_types = _coerce_allowed_types(domain_policy or {})
    expected_mime = (plan.expected_mime or "").lower()
    if allowed_types and expected_mime:
        if not any(expected_mime.startswith(prefix) for prefix in allowed_types):
            raise SkipDownload("policy-type", f"Disallowed MIME: {plan.expected_mime}")

    # Size policy based on context or plan overrides combined with resolver hints
    effective_max = _effective_max_bytes(plan, ctx)
    hints = _resolve_plan_hints(ctx, plan)
    hinted_size = hints.get("content_length") or hints.get("size")
    if (
        isinstance(hinted_size, int)
        and hinted_size >= 0
        and isinstance(effective_max, int)
        and hinted_size > effective_max
    ):
        raise SkipDownload(
            "policy-size",
            f"Expected size {hinted_size} exceeds limit {effective_max}",
        )

    if effective_max is not None and plan.max_bytes_override != effective_max:
        plan = replace(plan, max_bytes_override=effective_max)

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
    base_headers: dict[str, str] = {}
    if plan.referer:
        base_headers["Referer"] = plan.referer
    if plan.etag:
        base_headers["If-None-Match"] = plan.etag
    if plan.last_modified:
        base_headers["If-Modified-Since"] = plan.last_modified

    effective_max_bytes = (
        plan.max_bytes_override
        if plan.max_bytes_override is not None
        else max_bytes
    )

    t0 = time.monotonic_ns()
    try:
        head_kwargs = {"allow_redirects": True, "timeout": timeout_s}
        if base_headers:
            head_kwargs["headers"] = dict(base_headers)
        head = session.head(url, **head_kwargs)
        elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
        _emit(
            telemetry,
            run_id=run_id,
            resolver_name=plan.resolver_name,
            url=url,
            status=ATTEMPT_STATUS_HTTP_HEAD,
            http_status=head.status_code,
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:  # pylint: disable=broad-except
        # Some servers 405 on HEAD; proceed without failing
        LOGGER.debug(f"HEAD request failed (proceeding to GET): {e}")

    # GET request via httpx + hishel
    t0 = time.monotonic_ns()
    try:
        get_kwargs = {
            "stream": True,
            "allow_redirects": True,
            "timeout": timeout_s,
        }
        if base_headers:
            get_kwargs["headers"] = dict(base_headers)
        resp = session.get(url, **get_kwargs)
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
            status=ATTEMPT_STATUS_HTTP_GET,
            http_status=resp.status_code,
            elapsed_ms=elapsed_ms,
            meta={
                "content_type": content_type,
                "from_cache": from_cache,
                "revalidated": revalidated,
            },
        )

        # Emit cache-aware tokens
        if from_cache and not revalidated:
            _emit(
                telemetry,
                run_id=run_id,
                resolver_name=plan.resolver_name,
                url=url,
                status=ATTEMPT_STATUS_CACHE_HIT,
                http_status=resp.status_code,
                meta={
                    "content_type": content_type,
                    "reason": "ok",
                    "from_cache": True,
                },
            )
        if revalidated and resp.status_code == 304:
            _emit(
                telemetry,
                run_id=run_id,
                resolver_name=plan.resolver_name,
                url=url,
                status=ATTEMPT_STATUS_HTTP_304,
                http_status=304,
                meta={
                    "content_type": content_type,
                    "reason": "not-modified",
                    "revalidated": True,
                    "from_cache": from_cache,
                },
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

    staging_dir: Optional[str] = None
    tmp_path: Optional[str] = None
    # Determine effective byte limit (plan override wins)
    effective_max_bytes = (
        plan.max_bytes_override
        if plan.max_bytes_override is not None
        else max_bytes
    )

    # Write to temporary file using atomic writer
    tmp_dir = os.getcwd()
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, ".download.part")

    bytes_streamed = 0

    def _iter_with_limit(source_iter):
        nonlocal bytes_streamed
        for chunk in source_iter:
            if not chunk:
                yield chunk
                continue
            new_total = bytes_streamed + len(chunk)
            if (
                effective_max_bytes is not None
                and new_total > effective_max_bytes
            ):
                raise DownloadError(
                    "too-large",
                    f"Payload exceeded {effective_max_bytes} bytes",
                )
            bytes_streamed = new_total
            yield chunk

    try:
        staging_dir, tmp_path = _prepare_staging_destination(run_id, plan.resolver_name)
        bytes_written = atomic_write_stream(
            dest_path=tmp_path,
            byte_iter=_iter_with_limit(
                resp.iter_bytes(chunk_size=chunk_size)
            ),
            expected_len=(expected_len if verify_content_length else None),
            chunk_size=chunk_size,
        )

        # Check size limit
        if max_bytes and bytes_written > max_bytes:
            _cleanup_staging_artifacts(tmp_path, staging_dir)
        if (
            effective_max_bytes is not None
            and bytes_written > effective_max_bytes
        ):
            try:
                os.unlink(tmp_path)
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
            status=ATTEMPT_STATUS_HTTP_200,
            http_status=resp.status_code,
            meta={
                "bytes": bytes_written,
                "content_type": content_type,
                "content_length_hdr": expected_len,
                "from_cache": from_cache,
                "revalidated": revalidated,
            },
        )

        return DownloadStreamResult(
            path_tmp=tmp_path,
            bytes_written=bytes_written,
            http_status=resp.status_code,
            content_type=content_type,
            staging_path=staging_dir,
        )

    except SizeMismatchError:
        _cleanup_staging_artifacts(tmp_path, staging_dir)
        _emit(
            telemetry,
            run_id=run_id,
            resolver_name=plan.resolver_name,
            url=url,
            status=ATTEMPT_STATUS_SIZE_MISMATCH,
            http_status=resp.status_code,
            meta={
                "content_type": content_type,
                "reason": "size-mismatch",
                "content_length_hdr": expected_len,
            },
        )
        raise DownloadError("size-mismatch")
    except DownloadError:
        _cleanup_staging_artifacts(tmp_path, staging_dir)
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    except Exception as e:  # pylint: disable=broad-except
        _cleanup_staging_artifacts(tmp_path, staging_dir)
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

    # Determine final path (in real implementation, would use storage policy)
    if final_path:
        dest_path = final_path
    else:
        base = plan.url.rsplit("/", 1)[-1] or "download.bin"
        dest_path = os.path.join(os.getcwd(), base)

    # Validate final path safety after deriving destination
    try:
        dest_path = validate_path_safety(dest_path)
    except PathPolicyError as e:
        raise SkipDownload("path-policy", f"Path policy violation: {e}")

    # atomic_write_stream already moved temp → dest_path, so final_path exists
    # Just verify and emit event
    cleanup_dirs = False
    same_destination = False
    try:
        if stream.path_tmp:
            same_destination = os.path.abspath(stream.path_tmp) == os.path.abspath(final_path)

            if not same_destination and not os.path.exists(final_path):
                # If atomic_write_stream didn't finalize, move it now
                os.replace(stream.path_tmp, final_path)
            elif not same_destination:
                _cleanup_staging_artifacts(stream.path_tmp, None)
        cleanup_dirs = not same_destination
        if stream.path_tmp and not os.path.exists(dest_path):
            # If atomic_write_stream didn't finalize, move it now
            os.replace(stream.path_tmp, dest_path)
    except Exception as e:  # pylint: disable=broad-except
        raise DownloadError(
            "download-error",
            f"Failed to finalize: {e}",
        ) from e
    finally:
        if cleanup_dirs and stream.staging_path:
            _cleanup_staging_artifacts(None, stream.staging_path)

    # Emit finalization event
    _emit(
        telemetry,
        run_id=run_id,
        resolver_name=plan.resolver_name,
        url=plan.url,
        status="http-200",
        http_status=stream.http_status,
        meta={
            "bytes": stream.bytes_written,
            "final_path": final_path,
            "content_type": stream.content_type,
        },
    )

    return DownloadOutcome(
        ok=True,
        classification="success",
        path=dest_path,
        reason=None,
        meta={
            "content_type": stream.content_type,
            "bytes": stream.bytes_written,
        },
    )
