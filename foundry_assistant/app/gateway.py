"""APIM request metadata and safe gateway error handling.

This module deliberately contains no model client state.  Every model call in
an agent run receives its own immutable header mapping so concurrent users
cannot overwrite one another's correlation or semantic-cache partition.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import uuid
from collections.abc import Iterator
from email.utils import parsedate_to_datetime
from typing import Any


CORRELATION_HEADER = "x-correlation-id"
CACHE_PARTITION_HEADER = "x-cache-partition"
CACHE_ELIGIBILITY_HEADER = "x-cache-eligible"

_SAFE_CORRELATION = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SAFE_RETRY_AFTER = re.compile(r"^[A-Za-z0-9,: -]{1,128}$")


def correlation_id(raw_value: str | None = None) -> str:
    """Accept a small opaque caller ID or generate a random one.

    Free-form values are not reflected into responses or downstream headers;
    this prevents accidental PII propagation and header/log injection.
    """

    if raw_value and _SAFE_CORRELATION.fullmatch(raw_value):
        return raw_value
    return uuid.uuid4().hex


def cache_partition(
    *,
    secret: str,
    user_id: str,
    session_id: str,
    request_id: str,
    cache_eligible: bool,
) -> str:
    """Return an opaque HMAC partition without exposing identity or session IDs.

    Opted-in, read-only requests share a stable user/session partition so APIM
    semantic caching can reuse similar answers.  All other requests include
    the request correlation ID, which defensively prevents cache reuse even if
    the APIM policy does not honor ``x-cache-eligible: false``.
    """

    material = f"user={user_id}\x00session={session_id}"
    if not cache_eligible:
        material += f"\x00request={request_id}"
    digest = hmac.new(
        secret.encode("utf-8"), material.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"v1-{digest}"


def model_request_headers(
    *,
    cache_secret: str,
    user_id: str,
    session_id: str,
    request_id: str,
    cache_eligible: bool,
) -> dict[str, str]:
    """Build immutable per-run headers forwarded to every APIM model call."""

    return {
        CORRELATION_HEADER: request_id,
        CACHE_PARTITION_HEADER: cache_partition(
            secret=cache_secret,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            cache_eligible=cache_eligible,
        ),
        CACHE_ELIGIBILITY_HEADER: "true" if cache_eligible else "false",
    }


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Walk framework wrappers without trusting or rendering exception text."""

    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending and len(seen) < 20:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
        # AgentFrameworkException stores its inner exception in args[1].
        pending.extend(arg for arg in current.args if isinstance(arg, BaseException))


def _safe_retry_after(response: Any) -> str | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    raw = headers.get("retry-after")
    if raw is not None:
        value = str(raw).strip()
        if value.isdigit() and int(value) <= 86_400:
            return value
        if _SAFE_RETRY_AFTER.fullmatch(value):
            try:
                parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                pass
            else:
                return value

    raw_ms = headers.get("retry-after-ms") or headers.get("x-ms-retry-after-ms")
    if raw_ms is not None:
        try:
            milliseconds = int(str(raw_ms).strip())
        except ValueError:
            return None
        if 0 <= milliseconds <= 86_400_000:
            return str(math.ceil(milliseconds / 1000))
    return None


def rate_limit_retry_after(exc: BaseException) -> tuple[bool, str | None]:
    """Identify a nested OpenAI/APIM 429 and return a sanitized Retry-After."""

    for current in _exception_chain(exc):
        response = getattr(current, "response", None)
        status_code = getattr(current, "status_code", None) or getattr(
            response, "status_code", None
        )
        if status_code == 429:
            return True, _safe_retry_after(response)
    return False, None
