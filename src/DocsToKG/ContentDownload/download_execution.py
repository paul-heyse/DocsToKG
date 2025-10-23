# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.download_execution",
#   "purpose": "Download Execution Stage - Canonical Implementation.",
#   "sections": [
#     {
#       "id": "log-io-attempt",
#       "name": "_log_io_attempt",
#       "anchor": "function-log-io-attempt",
#       "kind": "function"
#     },
#     {
#       "id": "ctx-get",
#       "name": "_ctx_get",
#       "anchor": "function-ctx-get",
#       "kind": "function"
#     },
#     {
#       "id": "infer-user-agent",
#       "name": "_infer_user_agent",
#       "anchor": "function-infer-user-agent",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-domain-policy",
#       "name": "_resolve_domain_policy",
#       "anchor": "function-resolve-domain-policy",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-plan-hints",
#       "name": "_resolve_plan_hints",
#       "anchor": "function-resolve-plan-hints",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-allowed-types",
#       "name": "_coerce_allowed_types",
#       "anchor": "function-coerce-allowed-types",
#       "kind": "function"
#     },
#     {
#       "id": "effective-max-bytes",
#       "name": "_effective_max_bytes",
#       "anchor": "function-effective-max-bytes",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-storage-tmp-root",
#       "name": "_resolve_storage_tmp_root",
#       "anchor": "function-resolve-storage-tmp-root",
#       "kind": "function"
#     },
#     {
#       "id": "prepare-staging-destination",
#       "name": "_prepare_staging_destination",
#       "anchor": "function-prepare-staging-destination",
#       "kind": "function"
#     },
#     {
#       "id": "cleanup-staging-artifacts",
#       "name": "_cleanup_staging_artifacts",
#       "anchor": "function-cleanup-staging-artifacts",
#       "kind": "function"
#     },
#     {
#       "id": "prepare-candidate-download",
#       "name": "prepare_candidate_download",
#       "anchor": "function-prepare-candidate-download",
#       "kind": "function"
#     },
#     {
#       "id": "stream-candidate-payload",
#       "name": "stream_candidate_payload",
#       "anchor": "function-stream-candidate-payload",
#       "kind": "function"
#     },
#     {
#       "id": "finalize-candidate-download",
#       "name": "finalize_candidate_download",
#       "anchor": "function-finalize-candidate-download",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

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
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from DocsToKG.ContentDownload.api import (
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
    SimplifiedAttemptRecord,
)

LOGGER = logging.getLogger(__name__)


def _log_io_attempt(
    telemetry: Any,
    *,
    run_id: str | None,
    resolver_name: str,
    url: str,
    verb: str,
    status: str,
    http_status: int | None = None,
    reason: str | None = None,
    elapsed_ms: int | None = None,
    content_type: str | None = None,
    bytes_written: int | None = None,
    content_length_hdr: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Emit a :class:`SimplifiedAttemptRecord` via ``telemetry.log_io_attempt``."""

    if not telemetry or not hasattr(telemetry, "log_io_attempt"):
        return

    extra_payload: Mapping[str, Any]
    if extra:
        extra_payload = dict(extra)
    else:
        extra_payload = {}

    try:
        telemetry.log_io_attempt(
            SimplifiedAttemptRecord(
                ts=datetime.now(UTC),
                run_id=run_id,
                resolver=resolver_name,
                url=url,
                verb=verb,
                status=status,
                http_status=http_status,
                content_type=content_type,
                reason=reason,
                elapsed_ms=elapsed_ms,
                bytes_written=bytes_written,
                content_length_hdr=content_length_hdr,
                extra=extra_payload,
            )
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.debug("Telemetry emission failed: %s", exc)


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
        return http_config.user_agent

    candidate = _ctx_get(ctx, "user_agent", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate

    extra = _ctx_get(ctx, "extra", None)
    if isinstance(extra, Mapping):
        ua = extra.get("user_agent")
        if isinstance(ua, str) and ua.strip():
            return ua

    return default


def _resolve_domain_policy(ctx: Any, url: str) -> Mapping[str, Any] | None:
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


def _effective_max_bytes(plan: DownloadPlan, ctx: Any) -> int | None:
    """Return the effective max-bytes limit considering plan overrides and context."""

    if plan.max_bytes_override is not None:
        return plan.max_bytes_override

    ctx_limit = _ctx_get(ctx, "max_bytes", None)
    if isinstance(ctx_limit, int):
        return ctx_limit

    download_policy = _ctx_get(ctx, "download_policy", None)
    if download_policy and hasattr(download_policy, "max_bytes"):
        limit = download_policy.max_bytes
        if isinstance(limit, int):
            return limit

    return None


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


def _prepare_staging_destination(
    run_id: str | None,
    resolver_name: str,
    *,
    filename: str | None = None,
) -> tuple[Path, Path]:
    """Allocate a dedicated staging directory for a download attempt."""

    root = _resolve_storage_tmp_root() / "staging"
    run_token = (run_id or "adhoc").strip() or "adhoc"
    run_segment = "".join(ch if ch.isalnum() else "-" for ch in run_token)
    resolver_segment = "".join(ch if ch.isalnum() else "-" for ch in resolver_name) or "resolver"
    staging_dir = root / run_segment / f"{resolver_segment}-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    file_name = filename or f"payload-{uuid.uuid4().hex}.part"
    return staging_dir, staging_dir / file_name


def _cleanup_staging_artifacts(
    path_tmp: Path | str | None,
    staging_dir: Path | str | None,
) -> None:
    """Best-effort cleanup for staging files and directories."""

    if path_tmp:
        tmp_path = Path(path_tmp)
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    if staging_dir:
        dir_path = Path(staging_dir)
        if dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)


def prepare_candidate_download(
    plan: DownloadPlan,
    *,
    session: Any = None,
    ctx: Any = None,
    telemetry: Any = None,
    run_id: str | None = None,
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
        if not robots_checker.is_allowed(
            session,
            url,
            user_agent,
            telemetry=telemetry,
            run_id=run_id,
            resolver=getattr(plan, "resolver_name", None),
        ):
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
    timeout_s: float | None = None,
    chunk_size: int = 1 << 20,  # 1 MB default
    max_bytes: int | None = None,
    verify_content_length: bool = True,
    telemetry: Any = None,
    run_id: str | None = None,
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
        plan.max_bytes_override if plan.max_bytes_override is not None else max_bytes
    )

    t0 = time.monotonic_ns()
    try:
        head_kwargs = {"allow_redirects": True, "timeout": timeout_s}
        if base_headers:
            head_kwargs["headers"] = dict(base_headers)
        head = session.head(url, **head_kwargs)
        elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
        head_headers = getattr(head, "headers", None)
        head_content_type: str | None = None
        if head_headers:
            ct_raw = head_headers.get("Content-Type")
            if isinstance(ct_raw, str):
                head_content_type = ct_raw.lower()
        _log_io_attempt(
            telemetry,
            run_id=run_id,
            resolver_name=plan.resolver_name,
            url=url,
            verb="HEAD",
            status=ATTEMPT_STATUS_HTTP_HEAD,
            http_status=head.status_code,
            elapsed_ms=elapsed_ms,
            content_type=head_content_type,
        )
    except Exception as e:  # pylint: disable=broad-except
        # Some servers 405 on HEAD; proceed without failing
        LOGGER.debug(f"HEAD request failed (proceeding to GET): {e}")

    # GET request via httpx + hishel
    t0 = time.monotonic_ns()
    get_kwargs = {
        "allow_redirects": True,
        "timeout": timeout_s,
    }
    if base_headers:
        get_kwargs["headers"] = dict(base_headers)

    staging_dir: Path | None = None
    tmp_path: Path | None = None
    bytes_streamed = 0
    bytes_written = 0
    content_type = ""
    from_cache = False
    revalidated = False
    expected_len: int | None = None
    http_status: int | None = None
    not_modified_result: DownloadStreamResult | None = None
    stream_result: DownloadStreamResult | None = None
    resp_started = False

    try:
        with session.stream("GET", url, **get_kwargs) as resp:
            resp_started = True
            elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
            content_type = resp.headers.get("Content-Type", "").lower()
            http_status = resp.status_code

            # Check for hishel cache extensions
            extensions = getattr(resp, "extensions", {})
            from_cache = bool(extensions.get("from_cache"))
            revalidated = bool(extensions.get("revalidated"))

            _log_io_attempt(
                telemetry,
                run_id=run_id,
                resolver_name=plan.resolver_name,
                url=url,
                verb="GET",
                status=ATTEMPT_STATUS_HTTP_GET,
                http_status=http_status,
                elapsed_ms=elapsed_ms,
                content_type=content_type,
                extra={
                    "from_cache": from_cache,
                    "revalidated": revalidated,
                },
            )

            # Emit cache-aware tokens
            if from_cache and not revalidated:
                _log_io_attempt(
                    telemetry,
                    run_id=run_id,
                    resolver_name=plan.resolver_name,
                    url=url,
                    verb="GET",
                    status=ATTEMPT_STATUS_CACHE_HIT,
                    http_status=http_status,
                    content_type=content_type,
                    reason=ATTEMPT_REASON_OK,
                    extra={
                        "from_cache": True,
                    },
                )

            if revalidated and http_status == 304:
                _log_io_attempt(
                    telemetry,
                    run_id=run_id,
                    resolver_name=plan.resolver_name,
                    url=url,
                    verb="GET",
                    status=ATTEMPT_STATUS_HTTP_304,
                    http_status=304,
                    content_type=content_type,
                    reason=ATTEMPT_REASON_NOT_MODIFIED,
                    extra={
                        "revalidated": True,
                        "from_cache": from_cache,
                    },
                )
                not_modified_result = DownloadStreamResult(
                    path_tmp="",
                    bytes_written=0,
                    http_status=304,
                    content_type=content_type,
                )
            else:
                if plan.expected_mime and not content_type.startswith(plan.expected_mime):
                    raise SkipDownload(
                        "unexpected-ct",
                        f"Expected {plan.expected_mime}, got {content_type}",
                    )

                cl = resp.headers.get("Content-Length")
                expected_len = int(cl) if (cl and cl.isdigit()) else None

                effective_max_bytes = (
                    plan.max_bytes_override if plan.max_bytes_override is not None else max_bytes
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

                def _iter_with_limit(source_iter):
                    nonlocal bytes_streamed
                    for chunk in source_iter:
                        if not chunk:
                            yield chunk
                            continue
                        new_total = bytes_streamed + len(chunk)
                        if effective_max_bytes is not None and new_total > effective_max_bytes:
                            raise DownloadError(
                                "too-large",
                                f"Payload exceeded {effective_max_bytes} bytes",
                            )
                        bytes_streamed = new_total
                        yield chunk

                url_path = Path(urlsplit(plan.url).path)
                staging_name = f"{url_path.name or 'download.bin'}.part"

                staging_dir, tmp_path = _prepare_staging_destination(
                    run_id,
                    plan.resolver_name,
                    filename=staging_name,
                )
                bytes_written = atomic_write_stream(
                    dest_path=str(tmp_path),
                    byte_iter=_iter_with_limit(resp.iter_bytes(chunk_size=chunk_size)),
                    expected_len=(expected_len if verify_content_length else None),
                    chunk_size=chunk_size,
                )

                if effective_max_bytes is not None and bytes_written > effective_max_bytes:
                    _cleanup_staging_artifacts(tmp_path, staging_dir)
                    raise DownloadError(
                        "too-large",
                        f"Payload exceeded {effective_max_bytes} bytes",
                    )

                _log_io_attempt(
                    telemetry,
                    run_id=run_id,
                    resolver_name=plan.resolver_name,
                    url=url,
                    verb="STREAM",
                    status=ATTEMPT_STATUS_HTTP_200,
                    http_status=http_status,
                    reason=ATTEMPT_REASON_OK,
                    content_type=content_type,
                    bytes_written=bytes_written,
                    content_length_hdr=expected_len,
                    extra={
                        "from_cache": from_cache,
                        "revalidated": revalidated,
                    },
                )

                stream_result = DownloadStreamResult(
                    path_tmp=str(tmp_path),
                    bytes_written=bytes_written,
                    http_status=http_status,
                    content_type=content_type,
                    staging_path=str(staging_dir) if staging_dir else None,
                )
    except SizeMismatchError:
        _cleanup_staging_artifacts(tmp_path, staging_dir)
        _log_io_attempt(
            telemetry,
            run_id=run_id,
            resolver_name=plan.resolver_name,
            url=url,
            verb="STREAM",
            status=ATTEMPT_STATUS_SIZE_MISMATCH,
            http_status=http_status,
            reason=ATTEMPT_REASON_SIZE_MISMATCH,
            content_type=content_type,
            bytes_written=bytes_streamed,
            content_length_hdr=expected_len,
        )
        raise DownloadError("size-mismatch")
    except DownloadError:
        _cleanup_staging_artifacts(tmp_path, staging_dir)
        raise
    except Exception as e:  # pylint: disable=broad-except
        if not resp_started:
            raise DownloadError(
                "conn-error",
                f"GET request failed: {e}",
            ) from e
        _cleanup_staging_artifacts(tmp_path, staging_dir)
        raise DownloadError(
            "download-error",
            f"Atomic write failed: {e}",
        ) from e

    if not_modified_result is not None:
        return not_modified_result

    if stream_result is not None:
        return stream_result

    raise DownloadError("download-error", "GET request produced no payload")


def finalize_candidate_download(
    plan: DownloadPlan,
    stream: DownloadStreamResult,
    *,
    final_path: str | None = None,
    storage_settings: Any = None,
    storage_root: str | None = None,
    telemetry: Any = None,
    run_id: str | None = None,
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

    root_candidate = storage_root
    if root_candidate is None and storage_settings is not None:
        root_candidate = getattr(storage_settings, "root_dir", None)
    if root_candidate is None:
        root_candidate = os.getcwd()

    artifact_root = Path(root_candidate)
    if final_path:
        destination = Path(final_path)
        if not destination.is_absolute():
            destination = artifact_root / destination
    else:
        base_name = Path(urlsplit(plan.url).path).name or "download.bin"
        destination = artifact_root / base_name

    try:
        safe_final_path = validate_path_safety(str(destination), artifact_root=str(artifact_root))
    except PathPolicyError as e:
        raise SkipDownload("path-policy", f"Path policy violation: {e}")

    final_path_obj = Path(safe_final_path)
    tmp_path = Path(stream.path_tmp) if stream.path_tmp else None
    staging_dir = Path(stream.staging_path) if stream.staging_path else None

    if tmp_path is None:
        raise DownloadError("download-error", "Missing temporary payload path")

    cleanup_staging = False
    try:
        if not tmp_path.exists():
            raise DownloadError("download-error", "Temporary payload missing on disk")

        same_destination = tmp_path.resolve() == final_path_obj.resolve()
        if not same_destination:
            final_path_obj.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(tmp_path), str(final_path_obj))
            cleanup_staging = True
        else:
            cleanup_staging = False
    except Exception as e:  # pylint: disable=broad-except
        _cleanup_staging_artifacts(tmp_path, staging_dir)
        raise DownloadError(
            "download-error",
            f"Failed to finalize: {e}",
        ) from e

    if cleanup_staging and staging_dir:
        _cleanup_staging_artifacts(None, staging_dir)

    # Emit finalization event
    _log_io_attempt(
        telemetry,
        run_id=run_id,
        resolver_name=plan.resolver_name,
        url=plan.url,
        verb="FINALIZE",
        status=ATTEMPT_STATUS_HTTP_200,
        http_status=stream.http_status,
        reason=ATTEMPT_REASON_OK,
        content_type=stream.content_type,
        bytes_written=stream.bytes_written,
        extra={
            "final_path": str(final_path) if final_path else None,
        },
    )

    return DownloadOutcome(
        ok=True,
        classification="success",
        path=safe_final_path,
        reason=None,
        meta={
            "content_type": stream.content_type,
            "bytes": stream.bytes_written,
        },
    )
