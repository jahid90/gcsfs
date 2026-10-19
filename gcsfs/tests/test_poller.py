import asyncio
import math
from types import SimpleNamespace
from unittest import mock

import pytest
from google.api_core import exceptions as api_exceptions
from google.cloud import storage_control_v2
from google.rpc import code_pb2, status_pb2

from gcsfs.poller import (
    _BACKGROUND_TASKS,
    _calculate_poll_delay,
    poll_lro,
    poll_until,
    unwrap_operation_result,
)
from gcsfs.tests.lro_fakes import FakeLroFactory, operation_pb


@pytest.fixture
def fake_lro():
    """Same as the conftest fixture, so this module also runs with --noconftest."""
    return FakeLroFactory()


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


def linear_delay(floor):
    """Unjittered cadence: 5% of the elapsed time, at least ``floor`` seconds."""
    return lambda elapsed: max(0.05 * elapsed, floor)


class TestCalculatePollDelay:
    """Unit tests for the default polling cadence."""

    def test_floor(self):
        assert _calculate_poll_delay(0.0, random_fn=lambda a, b: 1.0) == 0.2
        assert _calculate_poll_delay(0.5, random_fn=lambda a, b: 1.0) == 0.2

    def test_grows_linearly_with_elapsed_time(self):
        assert _calculate_poll_delay(10.0, random_fn=lambda a, b: 1.0) == 2.0
        assert _calculate_poll_delay(
            10.0, slope=0.05, random_fn=lambda a, b: 1.0
        ) == pytest.approx(0.5)

    def test_cap_keeps_full_jitter_spread_below_cap(self):
        jitter_args = []

        def high(a, b):
            jitter_args.append((a, b))
            return b

        assert _calculate_poll_delay(1000.0, random_fn=high) == 30.0
        assert _calculate_poll_delay(1000.0, random_fn=lambda a, b: a) == 18.0
        assert jitter_args == [(0.75, 1.25)]

    def test_default_cadence_bounds(self):
        for _ in range(50):
            assert 0.15 <= _calculate_poll_delay(0.0) <= 0.25
            assert 1.5 <= _calculate_poll_delay(10.0) <= 2.5
            assert 18.0 <= _calculate_poll_delay(1000.0) <= 30.0


class TestPollRunners:
    """Virtual-time unit tests for poll_until, unwrap_operation_result, and poll_lro."""

    @pytest.mark.asyncio
    async def test_zero_rpc_in_memory_check_completes_at_t0(self, fake_lro):
        clock = FakeVirtualClock()
        folder = storage_control_v2.Folder(name="projects/_/buckets/b/folders/dst/")
        lro = fake_lro.succeeded(folder, name="projects/_/buckets/b/operations/op-fast")

        res = await poll_lro(
            lro.op,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert res == folder
        assert clock.sleeps == []
        assert clock.now == 0.0
        lro.get_operation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_poll_lro_polls_with_virtual_clock_until_done(
        self, fake_lro, monkeypatch
    ):
        monkeypatch.setattr("gcsfs.poller._calculate_poll_delay", linear_delay(0.2))
        clock = FakeVirtualClock()
        folder = storage_control_v2.Folder(name="projects/_/buckets/b/folders/dst/")
        # Complete on the 3rd check
        lro = fake_lro.succeeded(
            folder, after=3, name="projects/_/buckets/b/operations/op-1"
        )

        res = await poll_lro(
            lro.op,
            timeout=300.0,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert res == folder
        assert len(clock.sleeps) == 3
        assert clock.sleeps[0] == pytest.approx(0.200)
        assert lro.get_operation.await_count == 3
        # Every poll bypasses the GAPIC default retry; poll_until is the only
        # retry loop.
        for call in lro.get_operation.await_args_list:
            assert call.kwargs["retry"] is None

    @pytest.mark.asyncio
    async def test_poll_until_absorbs_transient_transport_glitches(self):
        clock = FakeVirtualClock()
        calls = 0

        async def flaky_check():
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

        res = await poll_until(
            flaky_check,
            delay_fn=linear_delay(0.2),
            operation_id="op-flaky",
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert res == "recovered"
        assert calls == 6
        assert len(clock.sleeps) == 6

    @pytest.mark.asyncio
    async def test_poll_until_backs_off_on_consecutive_transient_errors(self):
        clock = FakeVirtualClock()
        unavailable = api_exceptions.ServiceUnavailable("503")
        check = mock.AsyncMock(
            side_effect=[unavailable] * 3 + [(False, None), unavailable, (True, "ok")]
        )

        res = await poll_until(
            check,
            delay_fn=lambda elapsed: 0.1,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert res == "ok"
        first, *backoffs, after_success, after_one_error = clock.sleeps[:6]
        assert first == 0.1
        # Floors of 0.5, 1 and 2 s after 1, 2 and 3 consecutive errors, +-25%.
        for sleep, floor in zip(backoffs, [0.5, 1.0, 2.0]):
            assert 0.75 * floor <= sleep <= 1.25 * floor
        # A successful check resets the count.
        assert after_success == 0.1
        assert 0.375 <= after_one_error <= 0.625

    @pytest.mark.asyncio
    async def test_poll_until_transient_backoff_is_capped(self):
        clock = FakeVirtualClock()
        check = mock.AsyncMock(
            side_effect=[api_exceptions.TooManyRequests("429")] * 12 + [(True, "ok")]
        )

        await poll_until(
            check,
            delay_fn=lambda elapsed: 0.1,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert max(clock.sleeps) <= 30.0
        assert min(clock.sleeps[-3:]) >= 18.0

    @pytest.mark.asyncio
    async def test_poll_until_transient_backoff_is_clamped_to_the_deadline(self):
        clock = FakeVirtualClock()
        check = mock.AsyncMock(side_effect=api_exceptions.ServiceUnavailable("503"))

        with pytest.raises(asyncio.TimeoutError):
            await poll_until(
                check,
                delay_fn=lambda elapsed: 0.1,
                timeout=3.0,
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        assert clock.now == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_poll_until_non_transient_error_raises_without_backoff(self):
        clock = FakeVirtualClock()
        check = mock.AsyncMock(
            side_effect=[
                api_exceptions.ServiceUnavailable("503"),
                api_exceptions.PermissionDenied("403"),
            ]
        )

        with pytest.raises(api_exceptions.PermissionDenied):
            await poll_until(
                check,
                delay_fn=lambda elapsed: 0.1,
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        assert check.await_count == 2
        assert len(clock.sleeps) == 2

    @pytest.mark.asyncio
    async def test_poll_until_keeps_polling_after_unknown(self):
        clock = FakeVirtualClock()
        check = mock.AsyncMock(
            side_effect=[api_exceptions.Unknown("transport reset"), (True, "ok")]
        )

        res = await poll_until(
            check,
            delay_fn=linear_delay(0.2),
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        assert res == "ok"
        assert check.await_count == 2

    @pytest.mark.asyncio
    async def test_poll_until_reraises_non_transient_error(self):
        clock = FakeVirtualClock()
        check = mock.AsyncMock(side_effect=api_exceptions.NotFound("no such op"))

        with pytest.raises(api_exceptions.NotFound, match="no such op"):
            await poll_until(
                check,
                delay_fn=linear_delay(0.2),
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        check.assert_awaited_once()
        assert len(clock.sleeps) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "max_seconds, expected_checks, expected_desc",
        [
            (0.5, 1, "1 status check"),
            (1.0, 2, "2 status checks"),
        ],
    )
    async def test_poll_until_timeout_message_counts_status_checks(
        self, max_seconds, expected_checks, expected_desc
    ):
        clock = FakeVirtualClock()
        check = mock.AsyncMock(return_value=(False, None))

        with pytest.raises(asyncio.TimeoutError) as exc_info:
            await poll_until(
                check,
                delay_fn=lambda elapsed: 0.5,
                timeout=max_seconds,
                operation_id="op-1",
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        assert check.await_count == expected_checks
        assert str(exc_info.value) == (
            f"Polling timed out for 'op-1' after {max_seconds:.2f}s "
            f"and {expected_desc}."
        )

    @pytest.mark.asyncio
    async def test_poll_until_timeout_chains_last_transient_error(self):
        clock = FakeVirtualClock()
        last = api_exceptions.ServiceUnavailable("still down")
        check = mock.AsyncMock(
            side_effect=[api_exceptions.TooManyRequests("slow down"), last, last]
        )

        with pytest.raises(asyncio.TimeoutError) as exc_info:
            await poll_until(
                check,
                delay_fn=lambda elapsed: 0.5,
                timeout=1.2,
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        assert exc_info.value.__cause__ is last

    @pytest.mark.asyncio
    async def test_poll_until_timeout_has_no_cause_after_successful_check(self):
        clock = FakeVirtualClock()
        check = mock.AsyncMock(
            side_effect=[
                api_exceptions.ServiceUnavailable("blip"),
                (False, None),
                (False, None),
            ]
        )

        with pytest.raises(asyncio.TimeoutError) as exc_info:
            await poll_until(
                check,
                delay_fn=lambda elapsed: 0.5,
                timeout=1.2,
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        assert exc_info.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_poll_lro_timeout_raises_timeout_error_with_operation_id(
        self, fake_lro, monkeypatch
    ):
        monkeypatch.setattr("gcsfs.poller._calculate_poll_delay", linear_delay(0.5))
        clock = FakeVirtualClock()
        lro = fake_lro.pending(name="projects/_/buckets/b/operations/op-stalled")

        with pytest.raises(asyncio.TimeoutError) as exc_info:
            await poll_lro(
                lro.op,
                timeout=1.2,
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        assert clock.now == pytest.approx(1.2)
        # Polls at 0.5 s, 1.0 s and 1.2 s; the message names the operation.
        assert lro.get_operation.await_count == 3
        assert str(exc_info.value) == (
            "Polling timed out for 'projects/_/buckets/b/operations/op-stalled' "
            "after 1.20s and 3 status checks."
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_timeout", [0, 0.0, -1.0, math.inf, math.nan, True, "300"]
    )
    @pytest.mark.parametrize("done_at_t0", [True, False])
    async def test_poll_lro_rejects_invalid_timeout_before_any_rpc(
        self, fake_lro, bad_timeout, done_at_t0
    ):
        clock = FakeVirtualClock()
        lro = fake_lro.succeeded() if done_at_t0 else fake_lro.pending()

        with pytest.raises(ValueError, match="timeout must be a positive finite"):
            await poll_lro(
                lro.op,
                timeout=bad_timeout,
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        lro.get_operation.assert_not_awaited()
        assert clock.sleeps == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code, expected_cls",
        [
            (code_pb2.ALREADY_EXISTS, api_exceptions.AlreadyExists),
            (code_pb2.ABORTED, api_exceptions.Aborted),
            (code_pb2.NOT_FOUND, api_exceptions.NotFound),
        ],
    )
    async def test_unwrap_operation_result_maps_grpc_status_to_typed_exception(
        self, code, expected_cls
    ):
        operation = mock.AsyncMock()
        # A failed LRO carries its error as a google.rpc.Status, whose code is
        # always an int.
        raw_err = api_exceptions.GoogleAPICallError(
            "LRO failed",
            errors=[status_pb2.Status(code=code, message="LRO failed")],
        )
        operation.result.side_effect = raw_err

        with pytest.raises(expected_cls):
            await unwrap_operation_result(operation)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [object(), SimpleNamespace(code="INTERNAL"), SimpleNamespace(code=[13])],
        ids=["no_code", "str_code", "unhashable_code"],
    )
    async def test_unwrap_operation_result_reraises_error_without_int_code(self, error):
        operation = mock.AsyncMock()
        raw_err = api_exceptions.GoogleAPICallError("LRO failed", errors=[error])
        operation.result.side_effect = raw_err

        with pytest.raises(api_exceptions.GoogleAPICallError) as exc_info:
            await unwrap_operation_result(operation)

        assert exc_info.value is raw_err

    @pytest.mark.asyncio
    async def test_poll_lro_cancellation_dispatches_server_cancel(
        self, fake_lro, caplog, monkeypatch
    ):
        monkeypatch.setattr("gcsfs.poller._calculate_poll_delay", linear_delay(0.2))
        clock = FakeVirtualClock()
        # The 1st poll is cancelled; the real AsyncOperation.cancel() then polls
        # once more (still pending) before sending CancelOperation.
        name = "projects/_/buckets/b/operations/op-cancel"
        lro = fake_lro.sequence(
            [asyncio.CancelledError(), operation_pb(name=name)], name=name
        )

        with caplog.at_level("WARNING", logger="gcsfs"):
            with pytest.raises(asyncio.CancelledError):
                await poll_lro(
                    lro.op,
                    time_fn=clock.time,
                    sleep_fn=clock.sleep,
                )

        await asyncio.gather(*list(_BACKGROUND_TASKS), return_exceptions=True)
        lro.cancel_operation.assert_awaited_once()
        [warning] = [r.getMessage() for r in caplog.records if "cancelled" in r.msg]
        assert warning.count(name) == 1

    @pytest.mark.asyncio
    async def test_poll_lro_cancellation_bounds_hung_server_cancel(
        self, fake_lro, monkeypatch
    ):
        monkeypatch.setattr("gcsfs.poller.PER_POLL_RPC_TIMEOUT", 0.01)
        monkeypatch.setattr("gcsfs.poller._calculate_poll_delay", linear_delay(0.2))
        clock = FakeVirtualClock()
        name = "projects/_/buckets/b/operations/op-cancel-hung"
        lro = fake_lro.sequence(
            [asyncio.CancelledError(), operation_pb(name=name)], name=name
        )

        async def hung_cancel(*_args, **_kwargs):
            await asyncio.sleep(10.0)

        lro.cancel_operation.side_effect = hung_cancel

        with pytest.raises(asyncio.CancelledError):
            await poll_lro(
                lro.op,
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        await asyncio.wait_for(
            asyncio.gather(*list(_BACKGROUND_TASKS), return_exceptions=True),
            timeout=5.0,
        )
        lro.cancel_operation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_poll_lro_keeps_cancel_that_races_a_finished_status_check(
        self, fake_lro, monkeypatch
    ):
        """A cancel that lands in the same loop step as a status check finishing
        must still raise CancelledError and send CancelOperation.

        asyncio.wait_for on Python 3.10 and 3.11 returns the check's result in
        that case and drops the cancel, so polling would run on until timeout.
        """
        monkeypatch.setattr("gcsfs.poller._calculate_poll_delay", linear_delay(0.2))
        clock = FakeVirtualClock()
        lro = fake_lro.pending(name="projects/_/buckets/b/operations/op-race")
        replay = lro.get_operation.side_effect
        tasks = {}

        def cancel_poller_then_report_pending(*args, **kwargs):
            if lro.get_operation.await_count == 1:
                tasks["poller"].cancel()
            return replay(*args, **kwargs)

        async def sleep_and_yield(delay):
            # Yield like asyncio.sleep, so a cancel still pending on the
            # poller is delivered here, as it would be in production.
            await clock.sleep(delay)
            await asyncio.sleep(0)

        lro.get_operation.side_effect = cancel_poller_then_report_pending
        tasks["poller"] = asyncio.create_task(
            poll_lro(
                lro.op,
                timeout=1.0,
                time_fn=clock.time,
                sleep_fn=sleep_and_yield,
            )
        )

        with pytest.raises(asyncio.CancelledError):
            await tasks["poller"]

        await asyncio.wait_for(
            asyncio.gather(*list(_BACKGROUND_TASKS), return_exceptions=True),
            timeout=5.0,
        )
        # One status check by the poller, then the one AsyncOperation.cancel()
        # makes before it sends CancelOperation.
        assert lro.get_operation.await_count == 2
        lro.cancel_operation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_poll_lro_retries_after_per_poll_timeout(self, fake_lro, monkeypatch):
        """A status check that exceeds the per-poll deadline is cancelled and
        counts as a transient error, so the next poll runs."""
        monkeypatch.setattr("gcsfs.poller.PER_POLL_RPC_TIMEOUT", 0.05)
        monkeypatch.setattr("gcsfs.poller._calculate_poll_delay", linear_delay(0.2))
        clock = FakeVirtualClock()
        lro = fake_lro.succeeded(after=1)
        replay = lro.get_operation.side_effect
        hung_check_cancelled = asyncio.Event()

        async def first_check_hangs(*args, **kwargs):
            if lro.get_operation.await_count == 1:
                try:
                    await asyncio.sleep(10.0)
                except asyncio.CancelledError:
                    hung_check_cancelled.set()
                    raise
            return replay(*args, **kwargs)

        lro.get_operation.side_effect = first_check_hangs

        await poll_lro(
            lro.op,
            timeout=10.0,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        await asyncio.wait_for(hung_check_cancelled.wait(), timeout=5.0)
        assert lro.get_operation.await_count == 2
        # The timed-out check counts as a transient error, so the next sleep
        # is the first backoff step (0.5 s +-25%).
        assert clock.sleeps[0] == pytest.approx(0.2)
        assert 0.375 <= clock.sleeps[1] <= 0.625
        lro.cancel_operation.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_timeout", [0, -1.0, math.inf, math.nan, True])
    async def test_poll_until_rejects_invalid_timeout(self, bad_timeout):
        clock = FakeVirtualClock()
        check = mock.AsyncMock(return_value=(True, "ok"))

        with pytest.raises(ValueError, match="timeout must be a positive finite"):
            await poll_until(
                check, timeout=bad_timeout, time_fn=clock.time, sleep_fn=clock.sleep
            )

        check.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_poll_until_clamps_last_sleep_to_the_deadline(self):
        clock = FakeVirtualClock()
        check = mock.AsyncMock(return_value=(False, None))

        with pytest.raises(asyncio.TimeoutError):
            await poll_until(
                check,
                delay_fn=lambda elapsed: 10.0,
                timeout=1.5,
                time_fn=clock.time,
                sleep_fn=clock.sleep,
            )

        assert clock.sleeps == [1.5]
        check.assert_awaited_once()
