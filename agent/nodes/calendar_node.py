"""
Calendar node — multi-turn test drive booking.

Step 1  available_slots empty            → fetch slots → offer choices
Step 2  slots offered, no selected_slot  → fuzzy-match slot → ask for name
                                           (skips name step if customer_name already known)
Step 3  slot set, no user_email          → collect name → book with DEALERSHIP_EMAIL
"""
from __future__ import annotations

import re
from datetime import datetime

import structlog
from langchain_core.messages import AIMessage
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.state import AgentState
from scheduler.reminder_jobs import schedule_reminder, start_scheduler
from tools.calendar_tools import GoogleCalendarTools
from tools.email_tools import send_confirmation_email

log = structlog.get_logger()

_calendar = GoogleCalendarTools()


class CalendarSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    dealership_email: str
    timezone: str = "UTC"


# ── constants ─────────────────────────────────────────────────────────────────

_ORDINALS: dict[str, int] = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
}

_DAYS = frozenset({
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
})

# Captures: "2pm" "2:30pm" "2:30 PM" "14:00" "2" "9:30" "9 AM"
# \b at start prevents matching digits mid-word; \b at end prevents "3rd" matching "3"
_TIME_RE = re.compile(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b', re.I)

_NAME_PREFIXES = (
    "my name is", "i'm", "i am", "it's", "its",
    "call me", "this is", "name is",
)


# ── slot matching ─────────────────────────────────────────────────────────────

def _normalize_hour(hour: int, meridiem: str | None) -> int:
    """Convert parsed hour + meridiem to 24hr int."""
    if meridiem == "pm" and hour < 12:
        return hour + 12
    if meridiem == "am" and hour == 12:
        return 0
    return hour


def match_slot(user_text: str, available_slots: list[dict]) -> dict | None:
    """
    Match user speech to a slot.

    Time comparison uses slot["iso"] (exact 24hr) to avoid 12/24hr display mismatches.
    For bare digits with no meridiem (e.g. "at 2"), tries both AM and PM interpretations
    so "friday at 2" correctly matches a 2:00 PM slot.
    Falls back to ordinal words (first/1st …) only when neither day nor time found.
    """
    text = user_text.lower()

    user_day = next((d for d in _DAYS if d in text), None)

    # ── extract time ──────────────────────────────────────────────────────────
    user_hour_24: int | None = None
    user_hour_alt: int | None = None   # PM interpretation for ambiguous bare digit
    user_min = 0

    m = _TIME_RE.search(text)
    if m:
        h = int(m.group(1))
        user_min = int(m.group(2) or 0)
        meridiem = (m.group(3) or "").lower() or None
        user_hour_24 = _normalize_hour(h, meridiem)
        if meridiem is None and h < 12:
            user_hour_alt = h + 12   # e.g. "at 2" → also try 14:00

    # ── ordinal fallback: only when no day AND no time found ──────────────────
    if user_day is None and user_hour_24 is None:
        for word in re.split(r'\W+', text):
            if word in _ORDINALS:
                idx = _ORDINALS[word]
                if idx < len(available_slots):
                    return available_slots[idx]
        return None

    # ── day + time match (slot ISO used for exact 24hr comparison) ────────────
    for slot in available_slots:
        slot_lower = slot["slot"].lower()
        slot_day = next((d for d in _DAYS if d in slot_lower), None)
        day_ok = (user_day is None) or (slot_day == user_day)

        if user_hour_24 is None:
            if day_ok:
                return slot
            continue

        slot_dt = datetime.fromisoformat(slot["iso"])
        slot_hour, slot_min = slot_dt.hour, slot_dt.minute

        hour_ok = slot_hour == user_hour_24 or (
            user_hour_alt is not None and slot_hour == user_hour_alt
        )
        min_ok = slot_min == user_min

        if day_ok and hour_ok and min_ok:
            return slot
        if day_ok and hour_ok:   # minutes didn't match — hour+day is close enough
            return slot

    return None


# ── name extraction ───────────────────────────────────────────────────────────

def _extract_name(text: str) -> str | None:
    text = text.strip()
    lower = text.lower()
    for prefix in _NAME_PREFIXES:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    text = text.strip(".,!?\"'")
    words = text.split()
    if 1 <= len(words) <= 5:
        return text.title()
    return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _slots_to_voice(slots: list[dict]) -> str:
    names = [s["slot"] for s in slots]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return ", ".join(names[:-1]) + f", or {names[-1]}"


_BOOKING_ERROR = "Sorry, I had trouble booking that. Please try again."


async def _book(selected_slot: str, customer_name: str) -> tuple[str, dict]:
    """Run the full booking + email + reminder pipeline.

    book_appointment exceptions propagate — caller catches and surfaces a user-facing error.
    Email and reminder failures are logged but never block the booking response.
    """
    settings = CalendarSettings()
    email = settings.dealership_email

    log.info("booking.calling", slot_iso=selected_slot, name=customer_name, email=email)
    booking = await _calendar.book_appointment(
        slot_iso=selected_slot,
        customer_name=customer_name,
        customer_email=email,
    )
    log.info("booking.result", result=booking)
    meet_link = (booking or {}).get("meet_link", "")
    slot_display = (booking or {}).get("slot", selected_slot)

    try:
        await send_confirmation_email(
            to_email=email,
            customer_name=customer_name,
            slot=slot_display,
            meet_link=meet_link,
        )
    except Exception as exc:
        log.error("booking.email_failed", name=customer_name, slot=slot_display, error=str(exc))

    try:
        start_scheduler()
        schedule_reminder(
            meeting_iso=selected_slot,
            customer_email=email,
            customer_name=customer_name,
            meet_link=meet_link,
            slot_display=slot_display,
        )
    except Exception as exc:
        log.error("booking.reminder_failed", name=customer_name, slot=slot_display, error=str(exc))

    answer = (
        f"Perfect! Your test drive is booked for {slot_display}, {customer_name}. "
        "Our team will be ready for you and you'll get a reminder 30 minutes before!"
    )
    log.info("calendar_node.booked", name=customer_name, slot=slot_display)
    return answer, {
        "customer_name": customer_name,
        "user_email": None,       # clear so a second booking isn't silently dropped at step-3 guard
        "meeting_link": meet_link,
        "available_slots": [],
        "selected_slot": None,
    }


# ── node ──────────────────────────────────────────────────────────────────────

async def calendar_node(state: AgentState) -> dict:
    available_slots = state.get("available_slots") or []
    selected_slot = state.get("selected_slot")
    user_email = state.get("user_email")
    customer_name = state.get("customer_name")
    user_text = state["messages"][-1].content

    # ── step 1: fetch and offer slots ─────────────────────────────────────────
    if not available_slots:
        slots = await _calendar.get_free_slots(days_ahead=7)
        if not slots:
            answer = (
                "I don't see any open slots in the next week. "
                "Would you like me to check a bit further out?"
            )
            log.info("calendar_node.no_slots")
            return {"final_response": answer, "messages": [AIMessage(content=answer)]}

        answer = f"I have {_slots_to_voice(slots)} available. Which works best for you?"
        log.info("calendar_node.step1_offered", count=len(slots))
        return {
            "available_slots": slots,
            "customer_name": None,
            "selected_slot": None,
            "user_email": None,
            "meeting_link": None,
            "final_response": answer,
            "messages": [AIMessage(content=answer)],
        }

    # ── step 2: match slot ────────────────────────────────────────────────────
    if not selected_slot:
        chosen = match_slot(user_text, available_slots)
        if not chosen:
            answer = (
                f"I didn't quite catch that. I have {_slots_to_voice(available_slots)} — "
                "which one works for you?"
            )
            log.info("calendar_node.step2_no_match", text=user_text[:60])
            return {"final_response": answer, "messages": [AIMessage(content=answer)]}

        log.info("calendar_node.step2_slot_chosen", slot=chosen["slot"])

        if customer_name:
            # Name already known — book immediately, skip name step
            try:
                answer, updates = await _book(chosen["iso"], customer_name)
            except Exception as exc:
                log.error("calendar_node.book_failed", slot=chosen["iso"], name=customer_name, error=str(exc))
                return {
                    "final_response": _BOOKING_ERROR,
                    "messages": [AIMessage(content=_BOOKING_ERROR)],
                    "available_slots": [],
                    "selected_slot": None,
                }
            updates.update({"final_response": answer, "messages": [AIMessage(content=answer)]})
            return updates

        # Ask for name
        answer = "What name should I put for the booking?"
        return {
            "selected_slot": chosen["iso"],
            "final_response": answer,
            "messages": [AIMessage(content=answer)],
        }

    # ── step 3: collect name → book with DEALERSHIP_EMAIL ─────────────────────
    if not user_email:
        if not customer_name:
            name = _extract_name(user_text)
            if not name:
                answer = "I didn't catch that — what name should I put for the appointment?"
                log.info("calendar_node.step3_no_name")
                return {"final_response": answer, "messages": [AIMessage(content=answer)]}
            customer_name = name

        try:
            answer, updates = await _book(selected_slot, customer_name)
        except Exception as exc:
            log.error("calendar_node.book_failed", slot=selected_slot, name=customer_name, error=str(exc))
            return {
                "final_response": _BOOKING_ERROR,
                "messages": [AIMessage(content=_BOOKING_ERROR)],
                "available_slots": [],
                "selected_slot": None,
            }
        updates.update({"final_response": answer, "messages": [AIMessage(content=answer)]})
        return updates

    # Already booked — passthrough
    log.info("calendar_node.passthrough")
    return {}
