"""
Google Calendar tools — free slot lookup and appointment booking.

Auth: reads token.json (created by scripts/auth_google.py).
      token.json contains refresh_token so self-refreshes automatically.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger()

TOKEN_PATH = Path("token.json")

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
]

_WORK_START = 9    # 9 AM
_WORK_END = 18     # 6 PM
_SLOT_MINUTES = 30
_MAX_SLOTS = 5


class CalendarSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    google_calendar_id: str = "primary"
    timezone: str = "UTC"


def _fmt_slot(dt: datetime) -> str:
    """Format datetime as human-readable slot string, cross-platform (no %-d)."""
    time_part = dt.strftime("%I:%M %p")
    if time_part[0] == "0":
        time_part = time_part[1:]
    return f"{dt.strftime('%A %B')} {dt.day} at {time_part}"


class GoogleCalendarTools:

    def __init__(self) -> None:
        self._settings = CalendarSettings()
        self._tz = ZoneInfo(self._settings.timezone)
        self._creds: Credentials | None = None
        # Eagerly load if token.json is present so callers can inspect self._creds
        if TOKEN_PATH.exists():
            self._creds = self._load_creds()

    def _load_creds(self) -> Credentials:
        """Load token.json, refresh if expired, raise RuntimeError with a fix command on any failure."""
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        except FileNotFoundError:
            raise RuntimeError(
                "token.json not found. "
                "Run: uv run python scripts/auth_google.py"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load Google credentials: {exc}") from exc

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    TOKEN_PATH.write_text(creds.to_json())
                    log.info("google_creds.refreshed")
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to refresh Google token: {exc}. "
                        "Re-run: uv run python scripts/auth_google.py"
                    ) from exc
            else:
                raise RuntimeError(
                    "Google credentials expired and no refresh token. "
                    "Re-run: uv run python scripts/auth_google.py"
                )
        return creds

    def _get_service(self):
        if not Path("credentials.json").exists():
            raise RuntimeError(
                "credentials.json not found. "
                "Download from Google Cloud Console and place it in the project root."
            )
        if not TOKEN_PATH.exists():
            raise RuntimeError(
                "token.json not found. "
                "Run: uv run python scripts/auth_google.py"
            )
        # Re-load each call so mid-session expiry is caught and refreshed
        self._creds = self._load_creds()
        return build("calendar", "v3", credentials=self._creds)

    # ── public async API ──────────────────────────────────────────────────────

    async def get_free_slots(self, days_ahead: int = 7) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_free_slots_sync, days_ahead)

    async def book_appointment(
        self,
        slot_iso: str,
        customer_name: str,
        customer_email: str,
        notes: str = "",
    ) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._book_sync,
            slot_iso, customer_name, customer_email, notes,
        )

    # ── sync implementations (run in executor) ────────────────────────────────

    def _get_free_slots_sync(self, days_ahead: int) -> list[dict]:
        service = self._get_service()
        now = datetime.now(tz=self._tz)

        # Round up to next 30-min boundary
        if now.minute < 30:
            cursor = now.replace(minute=30, second=0, microsecond=0)
        else:
            cursor = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        time_max = (now + timedelta(days=days_ahead)).replace(
            hour=_WORK_END, minute=0, second=0, microsecond=0
        )

        # Query busy periods for the whole window
        body = {
            "timeMin": now.isoformat(),
            "timeMax": time_max.isoformat(),
            "items": [{"id": self._settings.google_calendar_id}],
        }
        result = service.freebusy().query(body=body).execute()
        raw_busy = (
            result.get("calendars", {})
            .get(self._settings.google_calendar_id, {})
            .get("busy", [])
        )
        busy: list[tuple[datetime, datetime]] = []
        for b in raw_busy:
            b_start = datetime.fromisoformat(b["start"].replace("Z", "+00:00")).astimezone(self._tz)
            b_end = datetime.fromisoformat(b["end"].replace("Z", "+00:00")).astimezone(self._tz)
            busy.append((b_start, b_end))

        slots: list[dict] = []

        while len(slots) < _MAX_SLOTS and cursor < time_max:
            # Skip weekends
            if cursor.weekday() >= 5:
                cursor = (cursor + timedelta(days=1)).replace(
                    hour=_WORK_START, minute=0, second=0, microsecond=0
                )
                continue

            # Snap to work-start if before opening
            if cursor.hour < _WORK_START:
                cursor = cursor.replace(hour=_WORK_START, minute=0, second=0, microsecond=0)
                continue

            # Roll to next day if past closing
            if cursor.hour >= _WORK_END:
                cursor = (cursor + timedelta(days=1)).replace(
                    hour=_WORK_START, minute=0, second=0, microsecond=0
                )
                continue

            slot_end = cursor + timedelta(minutes=_SLOT_MINUTES)
            is_busy = any(b_start < slot_end and b_end > cursor for b_start, b_end in busy)

            if not is_busy:
                slots.append({
                    "slot": _fmt_slot(cursor),
                    "iso": cursor.isoformat(),   # preserves tz offset so reminder fires at correct local time
                })

            cursor += timedelta(minutes=_SLOT_MINUTES)

        log.info("calendar.free_slots", count=len(slots), days_ahead=days_ahead)
        return slots

    def _book_sync(
        self,
        slot_iso: str,
        customer_name: str,
        customer_email: str,
        notes: str,
    ) -> dict:
        service = self._get_service()

        start_dt = datetime.fromisoformat(slot_iso)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=self._tz)
        end_dt = start_dt + timedelta(minutes=_SLOT_MINUTES)
        slot_label = _fmt_slot(start_dt)

        event = {
            "summary": f"Test Drive — {customer_name}",
            "description": notes or f"Test drive booked via voice agent for {customer_name}.",
            "start": {
                "dateTime": start_dt.isoformat(),
                "timeZone": self._settings.timezone,
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": self._settings.timezone,
            },
            "attendees": [{"email": customer_email, "displayName": customer_name}],
            "conferenceData": {
                "createRequest": {
                    "requestId": str(uuid.uuid4()),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 30},
                    {"method": "popup", "minutes": 10},
                ],
            },
        }

        created = service.events().insert(
            calendarId=self._settings.google_calendar_id,
            body=event,
            conferenceDataVersion=1,
            sendUpdates="all",
        ).execute()

        meet_link = ""
        for ep in created.get("conferenceData", {}).get("entryPoints", []):
            if ep.get("entryPointType") == "video":
                meet_link = ep.get("uri", "")
                break

        log.info(
            "calendar.booked",
            event_id=created["id"],
            slot=slot_label,
            customer=customer_email,
        )
        return {
            "confirmed": True,
            "slot": slot_label,
            "meet_link": meet_link,
            "event_id": created["id"],
        }
