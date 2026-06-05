"""
Smoke-tests for calendar_node booking flows.

Covers:
  - Two sequential bookings (issue 1 regression)
  - book_appointment raises -> user gets error message, not crash
  - send_confirmation_email raises -> booking still succeeds
  - schedule_reminder raises -> booking still succeeds
  - tz-aware ISO slots: reminder trigger time matches local time, not UTC

Run: uv run python scripts/test_calendar.py
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage

from agent.state import AgentState

# Reflect what the FIXED get_free_slots() emits: tz-aware ISO strings (+05:00 for PKT)
FAKE_SLOTS = [
    {"slot": "Monday 10:00 AM", "iso": "2026-06-08T10:00:00+05:00"},
    {"slot": "Tuesday 2:00 PM", "iso": "2026-06-09T14:00:00+05:00"},
]

FAKE_BOOKING = {
    "slot": "Monday 10:00 AM",
    "meet_link": "https://meet.google.com/fake-link",
}

BOOKING_ERROR_MSG = "Sorry, I had trouble booking that. Please try again."


def _state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [HumanMessage(content="book a test drive")],
        "intent": "book_appointment",
        "user_email": None,
        "customer_name": None,
        "selected_slot": None,
        "available_slots": [],
        "meeting_link": None,
        "rag_context": None,
        "final_response": None,
        "last_results": [],
    }
    base.update(overrides)
    return base


async def _drive_booking(initial_state: AgentState) -> AgentState:
    """Drive calendar_node through up to 3 turns and return final state."""
    from agent.nodes.calendar_node import calendar_node

    state = dict(initial_state)

    result = await calendar_node(state)
    state.update(result)

    state["messages"] = [HumanMessage(content="first one")]
    result = await calendar_node(state)
    state.update(result)

    if state.get("selected_slot") and not state.get("meeting_link"):
        state["messages"] = [HumanMessage(content="my name is Test User")]
        result = await calendar_node(state)
        state.update(result)

    return state


def _print_result(label: str, state: AgentState, *, expect_success: bool) -> bool:
    response = state.get("final_response", "")
    link = state.get("meeting_link")
    succeeded = bool(link)

    sep = "=" * 60
    status = "PASS" if succeeded == expect_success else "FAIL"
    print(f"\n{sep}")
    print(f"  {label}  [{status}]")
    print(f"{sep}")
    print(f"  response   : {response[:90]}")
    print(f"  meeting_link: {link}")
    return succeeded == expect_success


async def main() -> None:
    results: list[bool] = []

    # ── test 1 & 2: sequential bookings ───────────────────────────────────────
    with (
        patch("agent.nodes.calendar_node._calendar.get_free_slots", new_callable=AsyncMock) as mock_slots,
        patch("agent.nodes.calendar_node._calendar.book_appointment", new_callable=AsyncMock) as mock_book,
        patch("agent.nodes.calendar_node.send_confirmation_email", new_callable=AsyncMock),
        patch("agent.nodes.calendar_node.start_scheduler"),
        patch("agent.nodes.calendar_node.schedule_reminder"),
    ):
        mock_slots.return_value = FAKE_SLOTS
        mock_book.return_value = FAKE_BOOKING

        state1 = await _drive_booking(_state())
        results.append(_print_result("TEST 1 — first booking", state1, expect_success=True))

        state2 = await _drive_booking(_state(
            customer_name=state1.get("customer_name"),
            user_email=state1.get("user_email"),
            available_slots=[],
            selected_slot=None,
            meeting_link=None,
        ))
        results.append(_print_result("TEST 2 — second booking (issue 1 regression)", state2, expect_success=True))

        print(f"\n  book_appointment call count: {mock_book.call_count} (expected 2)")

    # ── test 3: book_appointment raises -> user-facing error, no crash ─────────
    with (
        patch("agent.nodes.calendar_node._calendar.get_free_slots", new_callable=AsyncMock) as mock_slots,
        patch("agent.nodes.calendar_node._calendar.book_appointment", new_callable=AsyncMock) as mock_book,
        patch("agent.nodes.calendar_node.send_confirmation_email", new_callable=AsyncMock),
        patch("agent.nodes.calendar_node.start_scheduler"),
        patch("agent.nodes.calendar_node.schedule_reminder"),
    ):
        mock_slots.return_value = FAKE_SLOTS
        mock_book.side_effect = RuntimeError("Google Calendar API timeout")

        state3 = await _drive_booking(_state())
        got_error_msg = BOOKING_ERROR_MSG in (state3.get("final_response") or "")
        slots_cleared = not state3.get("available_slots") and not state3.get("selected_slot")
        ok = got_error_msg and slots_cleared
        results.append(ok)

        sep = "=" * 60
        status = "PASS" if ok else "FAIL"
        print(f"\n{sep}")
        print(f"  TEST 3 — book_appointment raises -> error message  [{status}]")
        print(f"{sep}")
        print(f"  response      : {state3.get('final_response', '')[:90]}")
        print(f"  got_error_msg : {got_error_msg}")
        print(f"  slots_cleared : {slots_cleared}")

    # ── test 4: email raises -> booking still succeeds ─────────────────────────
    with (
        patch("agent.nodes.calendar_node._calendar.get_free_slots", new_callable=AsyncMock) as mock_slots,
        patch("agent.nodes.calendar_node._calendar.book_appointment", new_callable=AsyncMock) as mock_book,
        patch("agent.nodes.calendar_node.send_confirmation_email", new_callable=AsyncMock) as mock_email,
        patch("agent.nodes.calendar_node.start_scheduler"),
        patch("agent.nodes.calendar_node.schedule_reminder"),
    ):
        mock_slots.return_value = FAKE_SLOTS
        mock_book.return_value = FAKE_BOOKING
        mock_email.side_effect = RuntimeError("Gmail quota exceeded")

        state4 = await _drive_booking(_state())
        results.append(_print_result("TEST 4 — email raises -> booking succeeds anyway", state4, expect_success=True))

    # ── test 5: schedule_reminder raises -> booking still succeeds ─────────────
    with (
        patch("agent.nodes.calendar_node._calendar.get_free_slots", new_callable=AsyncMock) as mock_slots,
        patch("agent.nodes.calendar_node._calendar.book_appointment", new_callable=AsyncMock) as mock_book,
        patch("agent.nodes.calendar_node.send_confirmation_email", new_callable=AsyncMock),
        patch("agent.nodes.calendar_node.start_scheduler"),
        patch("agent.nodes.calendar_node.schedule_reminder", side_effect=RuntimeError("scheduler not running")),
    ):
        mock_slots.return_value = FAKE_SLOTS
        mock_book.return_value = FAKE_BOOKING

        state5 = await _drive_booking(_state())
        results.append(_print_result("TEST 5 — schedule_reminder raises -> booking succeeds anyway", state5, expect_success=True))

    # ── test 6: tz-aware ISO -> reminder fires at correct local time ──────────
    from scheduler.reminder_jobs import schedule_reminder

    sep = "=" * 60
    print(f"\n{sep}")
    print("  TEST 6 — tz-aware ISO -> reminder fires at correct PKT local time")
    print(f"{sep}")

    pkt = ZoneInfo("Asia/Karachi")

    # 10:00 AM PKT = 05:00 UTC; reminder should fire at 09:30 AM PKT = 04:30 UTC
    slot_pkt = "2026-06-08T10:00:00+05:00"
    expected_trigger_utc = datetime(2026, 6, 8, 4, 30, tzinfo=timezone.utc)

    captured: list[datetime] = []

    class _FakeJob:
        pass

    def _fake_add_job(_fn, trigger, run_date, **kwargs):
        captured.append(run_date)
        return _FakeJob()

    with patch("scheduler.reminder_jobs._scheduler") as mock_sched:
        mock_sched.add_job.side_effect = _fake_add_job
        schedule_reminder(
            meeting_iso=slot_pkt,
            customer_email="test@example.com",
            customer_name="Test",
            meet_link="",
            slot_display="Monday 10:00 AM",
        )

    ok6 = False
    if captured:
        trigger = captured[0]
        # Normalise to UTC for comparison
        trigger_utc = trigger.astimezone(timezone.utc).replace(microsecond=0)
        ok6 = trigger_utc == expected_trigger_utc
        print(f"  trigger (UTC) : {trigger_utc.isoformat()}")
        print(f"  expected (UTC): {expected_trigger_utc.isoformat()}")
        print(f"  match         : {ok6}")
    else:
        print("  ERROR: add_job was never called (slot in the past?)")

    # Also verify naive ISO now uses TIMEZONE env var, not UTC
    # Naive "10:00" in Asia/Karachi should also produce 04:30 UTC trigger
    slot_naive = "2026-06-08T10:00:00"
    captured2: list[datetime] = []

    def _fake_add_job2(_fn, trigger, run_date, **kwargs):
        captured2.append(run_date)
        return _FakeJob()

    with (
        patch("scheduler.reminder_jobs._scheduler") as mock_sched2,
        patch("scheduler.reminder_jobs._ReminderSettings") as mock_settings,
    ):
        mock_sched2.add_job.side_effect = _fake_add_job2
        mock_settings.return_value.timezone = "Asia/Karachi"
        schedule_reminder(
            meeting_iso=slot_naive,
            customer_email="test@example.com",
            customer_name="Test",
            meet_link="",
            slot_display="Monday 10:00 AM",
        )

    ok6b = False
    if captured2:
        trigger2 = captured2[0].astimezone(timezone.utc).replace(microsecond=0)
        ok6b = trigger2 == expected_trigger_utc
        print(f"\n  naive ISO fallback (UTC): {trigger2.isoformat()}")
        print(f"  expected (UTC)          : {expected_trigger_utc.isoformat()}")
        print(f"  match                   : {ok6b}")
    else:
        print("  ERROR: add_job not called for naive ISO test")

    results.append(ok6 and ok6b)
    status = "PASS" if (ok6 and ok6b) else "FAIL"
    print(f"\n  [{status}]")

    # ── summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{sep}")
    print(f"  {passed}/{total} tests passed {'OK' if passed == total else 'FAIL'}")
    print(f"{sep}\n")


if __name__ == "__main__":
    asyncio.run(main())
