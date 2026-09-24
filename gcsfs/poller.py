import asyncio
import inspect
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Tuple, TypeVar

from google.api_core import exceptions as api_exceptions

logger = logging.getLogger("gcsfs")
T = TypeVar("T")
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()

#: Linear growth rate of poll delay relative to elapsed time (5% of elapsed seconds).
DEFAULT_LRO_POLL_SLOPE: float = 0.05
#: Default minimum delay between polling attempts in seconds (200 ms).
DEFAULT_LRO_POLL_FLOOR: float = 0.200
#: Hard minimum safety floor in seconds (50 ms) to prevent busy-looping or DoS-ing the server.
MIN_SAFE_LRO_POLL_FLOOR: float = 0.050
#: Maximum delay cap between polling attempts in seconds (30 s).
DEFAULT_LRO_POLL_CAP: float = 30.0
#: Lower multiplier bound for uniform random jitter (-25%).
DEFAULT_LRO_JITTER_MIN: float = 0.75
#: Upper multiplier bound for uniform random jitter (+25%).
DEFAULT_LRO_JITTER_MAX: float = 1.25
#: Per-RPC deadline in seconds for individual operation.done() status checks.
PER_POLL_RPC_TIMEOUT: float = 15.0
#: Elapsed duration threshold in seconds above which completed LROs log at INFO instead of DEBUG.
SLOW_LRO_LOG_THRESHOLD: float = 2.0
#: Default outer timeout budget in seconds for HNS LRO operations (5 minutes).
DEFAULT_HNS_LRO_TIMEOUT: float = 300.0


@dataclass(frozen=True)
class PollStatus:
    """Tracks the progress of an active polling loop (elapsed time and attempt count).

    Attributes:
        total_elapsed: Cumulative monotonic wall-clock seconds elapsed since the
            polling loop began (excluding any initial in-memory check #0). Must be a
            finite, non-negative number.
        attempt: 1-based index of the current polling attempt. Starts at 1 for the
            first scheduled network check and increments after each status query.
    """

    total_elapsed: float
    attempt: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_elapsed) or self.total_elapsed < 0.0:
            raise ValueError(
                f"total_elapsed must be a non-negative finite number, got {self.total_elapsed}"
            )
        # In Python, bool is a subclass of int (isinstance(True, int) is True),
        # so bool must be explicitly rejected before checking isinstance(..., int).
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError(
                f"attempt must be a positive integer >= 1, got {self.attempt}"
            )


class PollSchedule:
    """Calculates how long to wait between polling attempts and when to stop polling.

    Wraps a scheduling function ``Callable[[PollStatus], Optional[float]]`` that
    maps the current polling state to either a delay or termination signal.

    Contract:
        - Returning ``float``: The calculated delay in seconds to sleep before the
          next poll check. Must be non-negative and finite.
        - Returning ``None``: Signals that the schedule recommends terminating the
          polling loop (e.g., maximum duration budget exhausted). When ``None``
          is received, the polling runner aborts and raises
          ``asyncio.TimeoutError``.
    """

    def __init__(self, schedule_fn: Callable[[PollStatus], Optional[float]]):
        """Initializes schedule with underlying delay calculation function."""
        self._fn = schedule_fn

    def __call__(self, status: PollStatus) -> Optional[float]:
        """Computes the delay in seconds for the next poll attempt, or None to abort."""
        return self._fn(status)

    @classmethod
    def linear_elapsed(cls, slope: float = DEFAULT_LRO_POLL_SLOPE) -> "PollSchedule":
        """Linear schedule where delay grows relative to elapsed wall-clock time:

        delay(t) = slope * t.

        Example:
            >>> sched = PollSchedule.linear_elapsed(slope=1.0)
            >>> sched(PollStatus(total_elapsed=10.0))
            10.0
        """
        if not math.isfinite(slope) or slope <= 0.0:
            raise ValueError(f"slope must be a positive finite number, got {slope}")
        return cls(lambda s: slope * max(0.0, s.total_elapsed))

    def floor(self, min_delay: float) -> "PollSchedule":
        """Clamps any calculated delay up to at least min_delay (requiring min_delay >= MIN_SAFE_LRO_POLL_FLOOR).

        Example:
            >>> sched = PollSchedule.linear_elapsed(slope=1.0).floor(2.0)
            >>> sched(PollStatus(total_elapsed=1.0))  # 1.0s clamped up to 2.0s
            2.0
            >>> sched(PollStatus(total_elapsed=5.0))  # 5.0s > 2.0s, unchanged
            5.0
        """
        # Enforce a 50ms hard minimum floor so a caller cannot configure a near-zero
        # delay that would busy-loop and overwhelm (DoS) the Storage Control server.
        if not math.isfinite(min_delay) or min_delay < MIN_SAFE_LRO_POLL_FLOOR:
            raise ValueError(
                f"min_delay must be a finite number >= {MIN_SAFE_LRO_POLL_FLOOR} "
                f"(50ms minimum to prevent server overload), got {min_delay}"
            )
        return type(self)(
            lambda s: max(min_delay, d) if (d := self(s)) is not None else None
        )

    def cap(self, max_delay: float) -> "PollSchedule":
        """Clamps any calculated delay above max_delay down to max_delay.

        Example:
            >>> sched = PollSchedule.linear_elapsed(slope=1.0).cap(30.0)
            >>> sched(PollStatus(total_elapsed=10.0))  # 10.0s < 30.0s, unchanged
            10.0
            >>> sched(PollStatus(total_elapsed=50.0))  # 50.0s clamped down to 30.0s
            30.0
        """
        # Because .cap() is typically chained after .floor() and computes
        # min(max_delay, d), max_delay must also be >= MIN_SAFE_LRO_POLL_FLOOR (50ms)
        # so a downstream .cap() cannot override an upstream .floor() below 50ms.
        if not math.isfinite(max_delay) or max_delay < MIN_SAFE_LRO_POLL_FLOOR:
            raise ValueError(
                f"max_delay must be a finite number >= {MIN_SAFE_LRO_POLL_FLOOR}, got {max_delay}"
            )
        return type(self)(
            lambda s: min(max_delay, d) if (d := self(s)) is not None else None
        )

    def with_jitter(
        self,
        min_factor: float = DEFAULT_LRO_JITTER_MIN,
        max_factor: float = DEFAULT_LRO_JITTER_MAX,
        random_fn: Callable[[float, float], float] = random.uniform,
    ) -> "PollSchedule":
        """Applies uniform multiplicative random jitter to the calculated delay.

        Example:
            >>> sched = PollSchedule.linear_elapsed(slope=1.0).with_jitter(0.75, 1.25)
            >>> delay = sched(PollStatus(total_elapsed=10.0))
            >>> 7.5 <= delay <= 12.5
            True
        """
        if not (
            math.isfinite(min_factor)
            and math.isfinite(max_factor)
            and 0.0 <= min_factor <= max_factor
        ):
            raise ValueError(
                f"Invalid jitter bounds: [{min_factor}, {max_factor}] "
                "(must be finite numbers satisfying 0 <= min <= max)"
            )
        return type(self)(
            lambda s: (
                (d * random_fn(min_factor, max_factor))
                if (d := self(s)) is not None
                else None
            )
        )

    def max_duration(self, max_seconds: float) -> "PollSchedule":
        """Aborts (returns None) once total elapsed time exceeds max_seconds.

        Example:
            >>> sched = PollSchedule.linear_elapsed(slope=1.0).max_duration(10.0)
            >>> sched(PollStatus(total_elapsed=9.5))  # 9.5s clamped to 0.5s remaining budget
            0.5
            >>> sched(PollStatus(total_elapsed=10.0)) is None  # budget exhausted -> stop polling
            True
        """
        if not math.isfinite(max_seconds) or max_seconds <= 0.0:
            raise ValueError(
                f"max_seconds must be a positive finite number, got {max_seconds}"
            )

        def _schedule(s: PollStatus) -> Optional[float]:
            remaining = max_seconds - s.total_elapsed
            if remaining <= 0:
                return None
            delay = self(s)
            return min(delay, remaining) if delay is not None else None

        return type(self)(_schedule)


def get_default_hns_lro_cadence() -> PollSchedule:
    """Constructs the default HNS LRO cadence with safety clamps and jitter.

    Multiplicative jitter is applied after ``.floor(DEFAULT_LRO_POLL_FLOOR)``
    and ``.cap(DEFAULT_LRO_POLL_CAP / DEFAULT_LRO_JITTER_MAX)``. This ensures
    that the nominal baseline floor is randomized across distributed clients
    (150-250 ms for the 200 ms floor) and that the upper delay bound retains
    its full jitter spread (18.0-30.0 s at the 24.0 s pre-jitter cap) without
    exceeding ``DEFAULT_LRO_POLL_CAP`` (30.0 s), preventing synchronized
    thundering herds against the control plane.
    """
    return (
        PollSchedule.linear_elapsed(slope=DEFAULT_LRO_POLL_SLOPE)
        .floor(DEFAULT_LRO_POLL_FLOOR)
        .cap(DEFAULT_LRO_POLL_CAP / DEFAULT_LRO_JITTER_MAX)
        .with_jitter(DEFAULT_LRO_JITTER_MIN, DEFAULT_LRO_JITTER_MAX)
    )


async def poll_until(
    check_fn: Callable[[PollStatus], Awaitable[Tuple[bool, T]]],
    schedule: PollSchedule,
    operation_id: Optional[str] = None,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Polls an async check function until it reports completion or the schedule expires."""
    start_time = time_fn()
    op_desc = f" for '{operation_id}'" if operation_id else ""
    logger.debug("Starting polling%s...", op_desc)

    attempts = 1

    while True:
        now = time_fn()
        elapsed = now - start_time
        status = PollStatus(
            total_elapsed=elapsed,
            attempt=attempts,
        )

        delay = schedule(status)
        if delay is None:
            logger.warning(
                "Polling timed out%s after %.2fs across %d attempts.",
                op_desc,
                elapsed,
                attempts,
            )
            raise asyncio.TimeoutError(
                f"Polling operation timed out{op_desc} after {elapsed:.2f}s "
                f"across {attempts} attempts."
            )

        logger.debug(
            "Poll attempt #%d%s: elapsed=%.3fs, sleeping %.3fs before check",
            attempts,
            op_desc,
            elapsed,
            delay,
        )
        await sleep_fn(delay)

        try:
            is_done, result = await check_fn(status)
        except (
            api_exceptions.ServiceUnavailable,
            api_exceptions.TooManyRequests,
            api_exceptions.DeadlineExceeded,
            api_exceptions.InternalServerError,
            asyncio.TimeoutError,
        ) as e:
            logger.debug(
                "Transient transport error during status check #%d%s: %s",
                attempts,
                op_desc,
                e,
            )
            is_done, result = False, None

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

        attempts += 1


async def _unwrap_operation_result(
    operation: Any, timeout: Optional[float] = None
) -> Any:
    """Unpacks operation result or maps server-side error code to typed GoogleAPICallError."""
    try:
        return await operation.result(timeout=timeout)
    except api_exceptions.GoogleAPICallError as e:
        if type(e) is not api_exceptions.GoogleAPICallError:
            raise
        if (
            e.errors
            and hasattr(e.errors[0], "code")
            and hasattr(api_exceptions, "from_grpc_status")
        ):
            raise api_exceptions.from_grpc_status(
                e.errors[0].code,
                e.message,
                errors=e.errors,
                response=e.response,
            ) from e
        raise


def _is_operation_already_done_in_memory(operation: Any) -> bool:
    """Zero-RPC check inspecting underlying proto message for completion at t=0."""
    raw_op = getattr(operation, "_operation", None) or getattr(
        operation, "operation", None
    )
    if raw_op is not None and getattr(raw_op, "done", False) is True:
        return True
    return False


def _get_operation_name(operation: Any) -> Optional[str]:
    """Extracts the server-side operation name from a GAPIC AsyncOperation."""
    raw_op = getattr(operation, "_operation", None) or getattr(
        operation, "operation", None
    )
    name = getattr(raw_op, "name", None) or getattr(operation, "name", None)
    return name if isinstance(name, str) and name else None


async def poll_lro(
    operation: Any,
    schedule: Optional[PollSchedule] = None,
    timeout: Optional[float] = None,
    path1: Optional[str] = None,
    path2: Optional[str] = None,
    request_id: Optional[str] = None,
    rpc_retry: Optional[Any] = None,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Awaits completion of a GAPIC AsyncOperation using a low-lag PollSchedule."""
    op_name = _get_operation_name(operation)
    op_id = op_name if op_name else (request_id or "LRO")
    path_ctx = f"'{path1}' -> '{path2}'" if path1 and path2 else op_id
    log_op_id = f"{path_ctx} (op: {op_id})" if path1 and path2 else op_id

    if _is_operation_already_done_in_memory(operation):
        logger.debug(
            "LRO %s completed synchronously in memory at t=0; skipping polling.",
            path_ctx,
        )
        return await _unwrap_operation_result(operation)

    base_schedule = schedule if schedule is not None else get_default_hns_lro_cadence()
    active_schedule = (
        base_schedule.max_duration(timeout) if timeout is not None else base_schedule
    )

    if (
        rpc_retry is not None
        and hasattr(rpc_retry, "with_delay")
        and hasattr(rpc_retry, "with_timeout")
    ):
        rpc_retry = rpc_retry.with_delay(initial=0.1, maximum=1.0).with_timeout(2.0)

    start_time = time_fn()

    async def lro_complete(status: PollStatus) -> Tuple[bool, None]:
        remaining_budget = (
            max(0.0, timeout - (time_fn() - start_time))
            if timeout is not None
            else PER_POLL_RPC_TIMEOUT
        )
        rpc_timeout = min(PER_POLL_RPC_TIMEOUT, max(1.0, remaining_budget))

        done_kwargs = {"retry": rpc_retry} if rpc_retry is not None else {}
        is_done = await asyncio.wait_for(
            operation.done(**done_kwargs), timeout=rpc_timeout
        )
        return bool(is_done), None

    try:
        await poll_until(
            lro_complete,
            schedule=active_schedule,
            operation_id=log_op_id,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
    except asyncio.CancelledError:
        elapsed = time_fn() - start_time
        logger.warning(
            "HNS rename %s polling cancelled after %.3fs (op: %s); "
            "dispatching server-side cancellation signal.",
            path_ctx,
            elapsed,
            op_id,
        )
        if hasattr(operation, "cancel"):

            async def _send_cancel() -> None:
                try:
                    maybe_coro = operation.cancel()
                    if inspect.isawaitable(maybe_coro):
                        await asyncio.wait_for(maybe_coro, timeout=PER_POLL_RPC_TIMEOUT)
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

    return await _unwrap_operation_result(operation)
