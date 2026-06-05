"""
Multi-turn calendar booking flow tests.

Mocks all external I/O (Google Calendar, Gmail, APScheduler).
Exercises the 3-turn state machine in calendar_node via the full graph.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

import agent.nodes.calendar_node as cal_module
from agent.graph import build_graph

FAKE_SLOTS = [
    {"slot": "Monday June 9 at 10:00 AM", "iso": "2026-06-09T10:00:00"},
    {"slot": "Monday June 9 at 10:30 AM", "iso": "2026-06-09T10:30:00"},
    {"slot": "Monday June 9 at 11:00 AM", "iso": "2026-06-09T11:00:00"},
]

FAKE_BOOKING = {
    "confirmed": True,
    "slot": "Monday June 9 at 10:00 AM",
    "meet_link": "https://meet.google.com/fake-link-abc",
    "event_id": "fake_event_123",
}


@pytest.fixture
def g():
    from langgraph.checkpoint.memory import MemorySaver
    return build_graph(checkpointer=MemorySaver())


@pytest.fixture
def mock_router_llm():
    response = MagicMock()
    response.content = "book_appointment"
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


async def test_full_booking_flow(g, mock_router_llm):
    thread = {"configurable": {"thread_id": "test-booking-001"}}

    with patch("agent.router._get_llm", return_value=mock_router_llm), \
         patch.object(cal_module._calendar, "get_free_slots", new_callable=AsyncMock) as mock_slots, \
         patch.object(cal_module._calendar, "book_appointment", new_callable=AsyncMock) as mock_book, \
         patch("agent.nodes.calendar_node.send_confirmation_email", new_callable=AsyncMock) as mock_email, \
         patch("agent.nodes.calendar_node.schedule_reminder", return_value="job_123") as mock_sched, \
         patch("agent.nodes.calendar_node.start_scheduler"):

        mock_slots.return_value = FAKE_SLOTS
        mock_book.return_value = FAKE_BOOKING
        mock_email.return_value = True

        # ── turn 1: initiate booking ──────────────────────────────────────────
        r1 = await g.ainvoke(
            {"messages": [HumanMessage(content="I want to book a test drive")]},
            config=thread,
        )
        assert r1.get("available_slots") == FAKE_SLOTS
        assert "Monday June 9" in (r1.get("final_response") or "")
        mock_slots.assert_called_once()

        # ── turn 2: select first slot ─────────────────────────────────────────
        r2 = await g.ainvoke(
            {"messages": [HumanMessage(content="The first one please")]},
            config=thread,
        )
        assert r2.get("selected_slot") == FAKE_SLOTS[0]["iso"]
        assert "email" in (r2.get("final_response") or "").lower()

        # ── turn 3: provide email → book + confirm + remind ───────────────────
        r3 = await g.ainvoke(
            {"messages": [HumanMessage(content="Send it to test@example.com")]},
            config=thread,
        )
        assert r3.get("user_email") == "test@example.com"
        assert r3.get("meeting_link") == FAKE_BOOKING["meet_link"]
        assert "booked" in (r3.get("final_response") or "").lower()

        mock_book.assert_called_once_with(
            slot_iso=FAKE_SLOTS[0]["iso"],
            customer_name="Valued Customer",
            customer_email="test@example.com",
        )

        mock_email.assert_called_once()
        email_kw = mock_email.call_args.kwargs
        assert email_kw["to_email"] == "test@example.com"
        assert email_kw["meet_link"] == FAKE_BOOKING["meet_link"]
        assert email_kw["slot"] == FAKE_BOOKING["slot"]

        mock_sched.assert_called_once()
        sched_kw = mock_sched.call_args.kwargs
        assert sched_kw["customer_email"] == "test@example.com"
        assert sched_kw["meet_link"] == FAKE_BOOKING["meet_link"]
        assert sched_kw["meeting_iso"] == FAKE_SLOTS[0]["iso"]


async def test_no_slots_available(g, mock_router_llm):
    """When calendar is fully booked, agent says so gracefully."""
    thread = {"configurable": {"thread_id": "test-no-slots"}}

    with patch("agent.router._get_llm", return_value=mock_router_llm), \
         patch.object(cal_module._calendar, "get_free_slots", new_callable=AsyncMock) as mock_slots:

        mock_slots.return_value = []

        r = await g.ainvoke(
            {"messages": [HumanMessage(content="Book a test drive")]},
            config=thread,
        )
        response = (r.get("final_response") or "").lower()
        assert "slot" in response or "open" in response or "week" in response


async def test_slot_not_matched_reprompts(g, mock_router_llm):
    """Gibberish slot reply → agent reprompts with available slots."""
    thread = {"configurable": {"thread_id": "test-no-match"}}

    with patch("agent.router._get_llm", return_value=mock_router_llm), \
         patch.object(cal_module._calendar, "get_free_slots", new_callable=AsyncMock) as mock_slots:

        mock_slots.return_value = FAKE_SLOTS

        await g.ainvoke(
            {"messages": [HumanMessage(content="Book a test drive")]},
            config=thread,
        )
        r2 = await g.ainvoke(
            {"messages": [HumanMessage(content="purple elephant")]},
            config=thread,
        )
        assert r2.get("selected_slot") is None
        assert "Monday June 9" in (r2.get("final_response") or "")


async def test_invalid_email_reprompts(g, mock_router_llm):
    """Non-email text at email step → agent reprompts."""
    thread = {"configurable": {"thread_id": "test-bad-email"}}

    with patch("agent.router._get_llm", return_value=mock_router_llm), \
         patch.object(cal_module._calendar, "get_free_slots", new_callable=AsyncMock) as mock_slots:

        mock_slots.return_value = FAKE_SLOTS

        await g.ainvoke(
            {"messages": [HumanMessage(content="Book a test drive")]},
            config=thread,
        )
        await g.ainvoke(
            {"messages": [HumanMessage(content="first one")]},
            config=thread,
        )
        r3 = await g.ainvoke(
            {"messages": [HumanMessage(content="I dunno")]},
            config=thread,
        )
        assert r3.get("user_email") is None
        assert "email" in (r3.get("final_response") or "").lower()
