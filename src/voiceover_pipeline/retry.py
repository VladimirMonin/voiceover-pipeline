from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import requests

T = TypeVar("T")

_RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}
_NON_RETRYABLE_HTTP = {400, 401, 403, 404, 422}


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0
    enabled: bool = True

    @property
    def total_attempts(self) -> int:
        if not self.enabled:
            return 1
        return max(1, self.attempts)


def is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, (
        requests.Timeout,
        requests.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    )):
        return True
    if isinstance(error, (FileNotFoundError, PermissionError, ValueError)):
        return False

    text = str(error)
    http_match = re.search(r"HTTP\s+(\d{3})", text)
    if http_match:
        status = int(http_match.group(1))
        if status in _NON_RETRYABLE_HTTP:
            return False
        return status in _RETRYABLE_HTTP

    lowered = text.lower()
    non_retryable_markers = [
        "invalid model",
        "invalid voice",
        "reference audio not found",
        "missing 'audio' field",
        "missing audio url",
        "unknown qwen mode",
        "out of memory",
        "cuda out of memory",
    ]
    if any(marker in lowered for marker in non_retryable_markers):
        return False

    retryable_markers = [
        "timeout",
        "timed out",
        "temporar",
        "connection reset",
        "connection aborted",
        "connection error",
        "still pending",
        "empty body",
        "empty audio",
    ]
    return any(marker in lowered for marker in retryable_markers)


def run_with_retry(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    on_retry: Callable[[int, BaseException, float], Any] | None = None,
) -> T:
    last_error: BaseException | None = None
    for attempt in range(1, policy.total_attempts + 1):
        try:
            return operation()
        except BaseException as error:
            last_error = error
            if attempt >= policy.total_attempts or not is_retryable_error(error):
                raise
            delay = min(policy.delay_seconds * (2 ** (attempt - 1)), policy.max_delay_seconds)
            delay += random.uniform(0, min(0.5, delay / 4))
            if on_retry:
                on_retry(attempt + 1, error, delay)
            time.sleep(delay)
    assert last_error is not None
    raise last_error
