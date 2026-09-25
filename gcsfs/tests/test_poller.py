import asyncio
import math
from types import SimpleNamespace
from unittest import mock

import grpc
import pytest
from google.api_core import exceptions as api_exceptions

from gcsfs.poller import (
    _BACKGROUND_TASKS,
    DEFAULT_LRO_POLL_CAP,
    DEFAULT_LRO_POLL_FLOOR,
    MIN_SAFE_LRO_POLL_FLOOR,
    PollSchedule,
    PollStatus,
    _unwrap_operation_result,
    get_default_hns_lro_cadence,
    poll_lro,
    poll_until,
)


class FakeVirtualClock:
    """Deterministic virtual clock and sleep recorder for hermetic polling tests."""

    def __init__(self, start: float = 0.0):
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


class TestPollStatus:
    """Unit tests for PollStatus validation."""

    @pytest.mark.parametrize("bad_elapsed", [-0.1, math.nan, math.inf, -math.inf])
    def test_poll_status_rejects_invalid_elapsed(self, bad_elapsed):
        with pytest.raises(
            ValueError, match="total_elapsed must be a non-negative finite number"
        ):
            PollStatus(total_elapsed=bad_elapsed, attempt=1)

    @pytest.mark.parametrize("bad_attempt", [0, -1, -10, True, False, 1.5])
    def test_poll_status_rejects_invalid_attempt(self, bad_attempt):
        with pytest.raises(ValueError, match="attempt"):
            PollStatus(total_elapsed=0.0, attempt=bad_attempt)


class TestPollSchedule:
    """Unit tests for PollSchedule combinators and default HNS cadence."""

    @pytest.mark.parametrize("bad_slope", [0.0, -0.05, math.nan, math.inf])
    def test_linear_elapsed_rejects_invalid_slope(self, bad_slope):
        with pytest.raises(ValueError, match="slope must be a positive finite number"):
            PollSchedule.linear_elapsed(slope=bad_slope)

    @pytest.mark.parametrize("bad_floor", [0.01, 0.0, -0.01, math.nan, math.inf])
    def test_floor_rejects_invalid_min_delay(self, bad_floor):
        sched = PollSchedule.linear_elapsed()
        with pytest.raises(
            ValueError,
            match=f"min_delay must be a finite number >= {MIN_SAFE_LRO_POLL_FLOOR}",
        ):
            sched.floor(bad_floor)

    def test_floor_clamps_low_delay_and_applies_jitter(self):
        # At total_elapsed=0.0, linear_elapsed produces 0.0s, which .floor(0.05)
        # clamps up to MIN_SAFE_LRO_POLL_FLOOR (50ms).
        sched = PollSchedule.linear_elapsed(slope=1.0).floor(MIN_SAFE_LRO_POLL_FLOOR)
        assert sched(PollStatus(total_elapsed=0.0)) == pytest.approx(
            MIN_SAFE_LRO_POLL_FLOOR
        )

        # Chaining .with_jitter(0.75, 1.25) after .floor(0.05) randomizes the
        # 50ms floor across [37.5ms, 62.5ms] to prevent synchronized polling spikes.
        jittered_sched = sched.with_jitter(0.75, 1.25)
        for _ in range(20):
            delay = jittered_sched(PollStatus(total_elapsed=0.0))
            assert delay is not None
            assert 0.0375 <= delay <= 0.0625

    @pytest.mark.parametrize("bad_cap", [0.01, 0.0, -1.0, math.nan, math.inf])
    def test_cap_rejects_invalid_max_delay(self, bad_cap):
        sched = PollSchedule.linear_elapsed()
        with pytest.raises(
            ValueError,
            match=f"max_delay must be a finite number >= {MIN_SAFE_LRO_POLL_FLOOR}",
        ):
            sched.cap(bad_cap)

    @pytest.mark.parametrize(
        "min_f, max_f", [(-0.1, 1.0), (1.2, 0.8), (math.nan, 1.0), (0.8, math.inf)]
    )
    def test_with_jitter_rejects_invalid_bounds(self, min_f, max_f):
        sched = PollSchedule.linear_elapsed()
        with pytest.raises(ValueError, match="Invalid jitter bounds"):
            sched.with_jitter(min_f, max_f)

    @pytest.mark.parametrize("bad_max", [0.0, -5.0, math.nan, math.inf])
    def test_max_duration_rejects_invalid_seconds(self, bad_max):
        sched = PollSchedule.linear_elapsed()
        with pytest.raises(
            ValueError, match="max_seconds must be a positive finite number"
        ):
            sched.max_duration(bad_max)

    def test_default_cadence_initial_delay_and_linear_growth(self):
        sched = get_default_hns_lro_cadence()

        # At t=0, base floor is 200ms, jittered by [0.75, 1.25] -> [150ms, 250ms]
        for _ in range(50):
            d0 = sched(PollStatus(total_elapsed=0.0))
            assert d0 is not None
            assert 0.150 <= d0 <= 0.250

        # At t=10s, 5% linear slope is 0.500s, jittered -> [0.375s, 0.625s]
        for _ in range(50):
            d10 = sched(PollStatus(total_elapsed=10.0))
            assert d10 is not None
            assert 0.375 <= d10 <= 0.625

        # At t=1000s, pre-jitter cap is 24s (30s / 1.25), jittered -> [18.0s, 30.0s]
        for _ in range(50):
            d_large = sched(PollStatus(total_elapsed=1000.0))
            assert d_large is not None
            assert 18.0 <= d_large <= 30.0
            assert d_large <= DEFAULT_LRO_POLL_CAP


class TestPollLroAndRunners:
    """Virtual-time unit tests for poll_until, _unwrap_operation_result, and poll_lro."""

    @pytest.mark.asyncio
    async def test_zero_rpc_in_memory_check_completes_at_t0(self):
        clock = FakeVirtualClock()
        operation = mock.AsyncMock()
        operation._operation = SimpleNamespace(
            done=True, name="projects/_/buckets/b/operations/op-fast"
        )
        operation.result.return_value = {"status": "ok"}

        res = await poll_lro(
            operation,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert res == {"status": "ok"}
        assert clock.sleeps == []
        assert clock.now == 0.0
        operation.done.assert_not_called()
        operation.result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_poll_lro_polls_with_virtual_clock_until_done(self):
        clock = FakeVirtualClock()
        operation = mock.AsyncMock()
        operation._operation = SimpleNamespace(
            done=False, name="projects/_/buckets/b/operations/op-1"
        )
        # Complete on the 3rd check
        operation.done.side_effect = [False, False, True]
        operation.result.return_value = "completed_folder"

        sched = PollSchedule.linear_elapsed(0.05).floor(DEFAULT_LRO_POLL_FLOOR)
        res = await poll_lro(
            operation,
            schedule=sched,
            timeout=300.0,
            path1="b/src",
            path2="b/dst",
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert res == "completed_folder"
        assert len(clock.sleeps) == 3
        assert clock.sleeps[0] == pytest.approx(0.200)
        assert operation.done.await_count == 3
        operation.result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_poll_until_absorbs_transient_transport_glitches(self):
        clock = FakeVirtualClock()
        calls = 0

        async def flaky_check(status: PollStatus):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise api_exceptions.ServiceUnavailable("503 backend glitch")
            if calls == 2:
                raise api_exceptions.TooManyRequests("429 quota spike")
            if calls == 3:
                raise api_exceptions.DeadlineExceeded("504 gateway deadline")
            if calls == 4:
                raise api_exceptions.InternalServerError("500 internal error")
            if calls == 5:
                raise asyncio.TimeoutError("per-RPC deadline")
            return True, "recovered"

        sched = PollSchedule.linear_elapsed(0.05).floor(0.200)
        res = await poll_until(
            flaky_check,
            schedule=sched,
            operation_id="op-flaky",
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert res == "recovered"
        assert calls == 6
        assert len(clock.sleeps) == 6

    @pytest.mark.asyncio
    async def test_poll_lro_clamps_rpc_retry_delay_and_timeout(self):
        clock = FakeVirtualClock()
        operation = mock.AsyncMock()
        operation._operation = SimpleNamespace(
            done=False, name="projects/_/buckets/b/operations/op-retry-clamp"
        )
        operation.done.return_value = True
        operation.result.return_value = "done_with_clamped_retry"

        mock_retry = mock.MagicMock()
        mock_with_delay = mock.MagicMock()
        mock_clamped_retry = mock.MagicMock()
        mock_retry.with_delay.return_value = mock_with_delay
        mock_with_delay.with_timeout.return_value = mock_clamped_retry

        sched = PollSchedule.linear_elapsed(0.05).floor(0.200)
        res = await poll_lro(
            operation,
            schedule=sched,
            rpc_retry=mock_retry,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert res == "done_with_clamped_retry"
        mock_retry.with_delay.assert_called_once_with(initial=0.1, maximum=1.0)
        mock_with_delay.with_timeout.assert_called_once_with(2.0)
        operation.done.assert_awaited_once_with(retry=mock_clamped_retry)

    @pytest.mark.asyncio
    async def test_poll_lro_timeout_raises_timeout_error_with_operation_id(self):
        clock = FakeVirtualClock()
        operation = mock.AsyncMock()
        operation._operation = SimpleNamespace(
            done=False, name="projects/_/buckets/b/operations/op-stalled"
        )
        operation.done.return_value = False

        sched = PollSchedule.linear_elapsed(0.05).floor(0.500)
        with pytest.raises(asyncio.TimeoutError, match="op-stalled"):
            await poll_lro(
                operation,
                schedule=sched,
                timeout=1.2,
                path1="b/src",
                path2="b/dst",
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        assert clock.now == pytest.approx(1.2)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw_code, expected_cls",
        [
            (grpc.StatusCode.ALREADY_EXISTS, api_exceptions.AlreadyExists),
            (6, api_exceptions.AlreadyExists),
            (10, api_exceptions.Aborted),
            (5, api_exceptions.NotFound),
        ],
    )
    async def test_unwrap_operation_result_maps_grpc_status_to_typed_exception(
        self, raw_code, expected_cls
    ):
        operation = mock.AsyncMock()
        raw_err = api_exceptions.GoogleAPICallError(
            "LRO failed",
            errors=[SimpleNamespace(code=raw_code)],
        )
        operation.result.side_effect = raw_err

        with pytest.raises(expected_cls):
            await _unwrap_operation_result(operation)

    @pytest.mark.asyncio
    async def test_poll_lro_cancellation_dispatches_server_cancel(self):
        clock = FakeVirtualClock()
        operation = mock.AsyncMock()
        operation._operation = SimpleNamespace(
            done=False, name="projects/_/buckets/b/operations/op-cancel"
        )
        operation.done.side_effect = asyncio.CancelledError()

        sched = PollSchedule.linear_elapsed(0.05).floor(0.200)
        with pytest.raises(asyncio.CancelledError):
            await poll_lro(
                operation,
                schedule=sched,
                path1="b/src",
                path2="b/dst",
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        await asyncio.gather(*list(_BACKGROUND_TASKS), return_exceptions=True)
        operation.cancel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_poll_lro_cancellation_bounds_hung_server_cancel(self, monkeypatch):
        monkeypatch.setattr("gcsfs.poller.PER_POLL_RPC_TIMEOUT", 0.01)
        clock = FakeVirtualClock()
        operation = mock.MagicMock()
        operation._operation = SimpleNamespace(
            done=False, name="projects/_/buckets/b/operations/op-cancel-hung"
        )
        operation.done = mock.AsyncMock(side_effect=asyncio.CancelledError())

        async def hung_cancel():
            await asyncio.sleep(10.0)

        operation.cancel = mock.MagicMock(side_effect=hung_cancel)

        sched = PollSchedule.linear_elapsed(0.05).floor(0.200)
        with pytest.raises(asyncio.CancelledError):
            await poll_lro(
                operation,
                schedule=sched,
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        await asyncio.gather(*list(_BACKGROUND_TASKS), return_exceptions=True)
        operation.cancel.assert_called_once()
