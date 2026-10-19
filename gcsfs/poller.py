import asyncio
import logging
import math
import random
import time
from typing import Any, Awaitable, Callable, Optional, Tuple, TypeVar

from google.api_core import exceptions as api_exceptions
from google.api_core.operation_async import AsyncOperation

from gcsfs.retry import is_transient_exception

logger = logging.getLogger("gcsfs")
T = TypeVar("T")
#: Strong references, so fire-and-forget cancel tasks are not garbage collected.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()

#: Deadline for a single operation.done() status check, in seconds.
PER_POLL_RPC_TIMEOUT: float = 15.0
#: Polls that finish after this many seconds log at INFO instead of DEBUG.
SLOW_LRO_LOG_THRESHOLD: float = 2.0
#: After n consecutive transient errors, the next delay is at least
#: min(base * 2 ** (n - 1), cap / 1.25), jittered by 0.75-1.25.
_TRANSIENT_ERROR_BACKOFF_BASE: float = 0.5
_TRANSIENT_ERROR_BACKOFF_CAP: float = 30.0


def _calculate_poll_delay(
    elapsed: float,
    *,
    slope: float = 0.20,
    floor: float = 0.2,
    cap: float = 30.0,
    jitter: Tuple[float, float] = (0.75, 1.25),
    random_fn: Callable[[float, float], float] = random.uniform,
) -> float:
    """Returns the jittered delay in seconds before the next status check."""
    # Capping at cap / jitter[1] keeps the jittered delay at or below cap.
    return min(max(elapsed * slope, floor), cap / jitter[1]) * random_fn(*jitter)


def _is_transient_poll_exception(exc: Exception) -> bool:
    """Returns whether a failed status check should be retried on the next poll."""
    return isinstance(exc, asyncio.TimeoutError) or is_transient_exception(exc)


def _validate_lro_timeout(timeout: Optional[float]) -> None:
    """Raises ValueError unless timeout is None or a positive finite number."""
    if timeout is None:
        return
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            "timeout must be a positive finite number of seconds, or None for "
            f"no limit; got {timeout!r}"
        )


async def poll_until(
    check_fn: Callable[[], Awaitable[Tuple[bool, T]]],
    *,
    delay_fn: Callable[[float], float] = _calculate_poll_delay,
    timeout: Optional[float] = None,
    operation_id: Optional[str] = None,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Sleeps, then calls ``check_fn``, until it reports done or ``timeout`` passes.

    Args:
        check_fn: Async callable returning ``(is_done, result)``. Transient
            errors are retried on the next poll, with an exponential backoff
            while they repeat; other errors are raised.
        delay_fn: Maps the elapsed seconds to the delay before the next check.
        timeout: Overall limit in seconds, or ``None`` for no limit.
        operation_id: Name used in log and error messages.
        time_fn: Monotonic clock, in seconds.
        sleep_fn: Async sleep, in seconds.

    Raises:
        ValueError: If ``timeout`` is not ``None`` or a positive finite number.
        asyncio.TimeoutError: If ``timeout`` passes first. If the last check
            failed with a transient error, it is chained as ``__cause__``.
    """
    _validate_lro_timeout(timeout)
    start_time = time_fn()
    op_desc = f" for '{operation_id}'" if operation_id else ""
    logger.debug("Starting polling%s...", op_desc)

    attempts = 1
    transient_errors = 0
    last_exc: Optional[Exception] = None
    while True:
        elapsed = time_fn() - start_time
        remaining = None if timeout is None else timeout - elapsed
        if remaining is not None and remaining <= 0:
            checks = attempts - 1
            checks_desc = f"{checks} status check{'' if checks == 1 else 's'}"
            logger.warning(
                "Polling timed out%s after %.2fs and %s.",
                op_desc,
                elapsed,
                checks_desc,
            )
            raise asyncio.TimeoutError(
                f"Polling timed out{op_desc} after {elapsed:.2f}s and {checks_desc}."
            ) from last_exc

        delay = delay_fn(elapsed)
        if transient_errors:
            # The exponent is bounded so a long outage can't overflow a float.
            backoff = min(
                _TRANSIENT_ERROR_BACKOFF_BASE * 2 ** min(transient_errors - 1, 16),
                _TRANSIENT_ERROR_BACKOFF_CAP / 1.25,
            )
            delay = max(delay, backoff * random.uniform(0.75, 1.25))
        if remaining is not None:
            delay = min(delay, remaining)
        logger.debug(
            "Poll attempt #%d%s: elapsed=%.3fs, sleeping %.3fs before check",
            attempts,
            op_desc,
            elapsed,
            delay,
        )
        await sleep_fn(delay)

        try:
            is_done, result = await check_fn()
            last_exc = None
            transient_errors = 0
            if is_done:
                final_elapsed = time_fn() - start_time
                log_level = (
                    logging.INFO
                    if final_elapsed >= SLOW_LRO_LOG_THRESHOLD
                    else logging.DEBUG
                )
                logger.log(
                    log_level,
                    "Polling completed%s in %.3fs across %d attempts.",
                    op_desc,
                    final_elapsed,
                    attempts,
                )
                return result
        except Exception as e:
            if not _is_transient_poll_exception(e):
                raise
            last_exc = e
            transient_errors += 1
            logger.debug(
                "Transient transport error during status check #%d%s: %s",
                attempts,
                op_desc,
                e,
            )

        attempts += 1


async def unwrap_operation_result(operation: AsyncOperation) -> Any:
    """Awaits ``operation.result()``, raising a typed error if the operation failed.

    Args:
        operation: The operation to wait for.

    Raises:
        google.api_core.exceptions.GoogleAPICallError: The subclass for the
            operation's error code (for example ``NotFound``), or the original
            error if it has no int code.
    """
    try:
        return await operation.result()
    except api_exceptions.GoogleAPICallError as e:
        if type(e) is api_exceptions.GoogleAPICallError and e.errors:
            code = getattr(e.errors[0], "code", None)
            if isinstance(code, int):
                raise api_exceptions.from_grpc_status(
                    code,
                    e.message,
                    errors=e.errors,
                    response=e.response,
                ) from e
        raise


def _retrieve_task_outcome(task: "asyncio.Future[Any]") -> None:
    """Reads an abandoned task's outcome so asyncio does not log it as unretrieved."""
    if not task.cancelled():
        task.exception()


async def _wait_keeping_cancel(aw: Awaitable[T], timeout: float) -> T:
    """Awaits ``aw`` for at most ``timeout`` seconds without losing a cancel."""
    # On Python 3.10 and 3.11, asyncio.wait_for returns the result instead of
    # raising CancelledError when the calling task is cancelled in the same
    # loop step in which aw finishes, so the cancel is lost. asyncio.wait does
    # not catch the cancel, so it always propagates.
    task = asyncio.ensure_future(aw)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        task.add_done_callback(_retrieve_task_outcome)
        raise
    if not done:
        task.cancel()
        task.add_done_callback(_retrieve_task_outcome)
        raise asyncio.TimeoutError(f"Status check exceeded {timeout:.2f}s.")
    return task.result()


async def poll_lro(
    operation: AsyncOperation,
    *,
    timeout: Optional[float] = None,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Waits for a GAPIC ``AsyncOperation`` to finish and returns its result.

    Args:
        operation: The operation to wait for.
        timeout: Overall limit in seconds, or ``None`` for no limit.
        time_fn: Monotonic clock, in seconds.
        sleep_fn: Async sleep, in seconds.

    Raises:
        ValueError: If ``timeout`` is not ``None`` or a positive finite number.
        asyncio.TimeoutError: If ``timeout`` passes before the operation ends.
        asyncio.CancelledError: If cancelled; ``operation.cancel()`` is then
            sent in the background.
        google.api_core.exceptions.GoogleAPICallError: If the operation failed.
    """
    _validate_lro_timeout(timeout)

    op_id = operation.operation.name or "LRO"

    if operation.operation.done:
        logger.debug(
            "LRO %s completed synchronously in memory at t=0; skipping polling.",
            op_id,
        )
        return await unwrap_operation_result(operation)

    start_time = time_fn()

    async def lro_complete() -> Tuple[bool, None]:
        remaining_budget = (
            max(0.0, timeout - (time_fn() - start_time))
            if timeout is not None
            else PER_POLL_RPC_TIMEOUT
        )
        rpc_timeout = min(PER_POLL_RPC_TIMEOUT, max(1.0, remaining_budget))

        # retry=None: the GAPIC default retry would add a second, slower retry
        # loop inside a single poll; poll_until already retries transient errors.
        # A per-poll timeout raises asyncio.TimeoutError, which poll_until
        # treats as transient.
        is_done = await _wait_keeping_cancel(
            operation.done(retry=None), timeout=rpc_timeout
        )
        return bool(is_done), None

    try:
        await poll_until(
            lro_complete,
            # Looked up at call time so tests can patch the cadence.
            delay_fn=_calculate_poll_delay,
            timeout=timeout,
            operation_id=op_id,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
    except asyncio.CancelledError:
        elapsed = time_fn() - start_time
        logger.warning(
            "LRO %s polling cancelled after %.3fs; "
            "dispatching server-side cancellation signal.",
            op_id,
            elapsed,
        )

        async def _send_cancel() -> None:
            # AsyncOperation.cancel() first awaits done() with api-core's
            # default retry, so this can back off and retry; wait_for bounds
            # the whole call, including that extra poll.
            try:
                await asyncio.wait_for(operation.cancel(), timeout=PER_POLL_RPC_TIMEOUT)
            except Exception as exc:
                logger.debug(
                    "Failed to send LRO cancellation signal for %s: %s",
                    op_id,
                    exc,
                )

        cancel_task = asyncio.create_task(_send_cancel())
        _BACKGROUND_TASKS.add(cancel_task)
        cancel_task.add_done_callback(_BACKGROUND_TASKS.discard)
        raise

    return await unwrap_operation_result(operation)
