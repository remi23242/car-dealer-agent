"""
APScheduler reminder jobs — in-memory store, AsyncIOScheduler.

Call start_scheduler() once at app startup (main.py).
schedule_reminder() is safe to call before start; jobs queue up.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic_settings import BaseSettings, SettingsConfigDict


class _ReminderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    timezone: str = "UTC"

log = structlog.get_logger()

_scheduler = AsyncIOScheduler()


def start_scheduler() -> None:
    if not _scheduler.running:
        _scheduler.start()
        log.info("scheduler.started")


async def _fire_reminder(
    to_email: str,
    customer_name: str,
    meet_link: str,
    slot_display: str,
) -> None:
    from tools.email_tools import send_reminder_email
    try:
        await send_reminder_email(
            to_email=to_email,
            customer_name=customer_name,
            slot=slot_display,
            meet_link=meet_link,
        )
        log.info("reminder.fired", to=to_email, slot=slot_display)
    except Exception as exc:
        log.error("reminder.fire_failed", to=to_email, error=str(exc), exc_info=True)


def schedule_reminder(
    meeting_iso: str,
    customer_email: str,
    customer_name: str,
    meet_link: str,
    slot_display: str,
) -> str:
    meeting_dt = datetime.fromisoformat(meeting_iso)
    if meeting_dt.tzinfo is None:
        meeting_dt = meeting_dt.replace(tzinfo=ZoneInfo(_ReminderSettings().timezone))

    trigger_time = meeting_dt - timedelta(minutes=30)
    now = datetime.now(tz=timezone.utc)

    job_id = f"reminder_{customer_email}_{meeting_iso}"

    if trigger_time <= now:
        log.warning(
            "reminder.skipped_past",
            job_id=job_id,
            trigger=trigger_time.isoformat(),
        )
        return job_id

    _scheduler.add_job(
        _fire_reminder,
        trigger="date",
        run_date=trigger_time,
        kwargs={
            "to_email": customer_email,
            "customer_name": customer_name,
            "meet_link": meet_link,
            "slot_display": slot_display,
        },
        id=job_id,
        replace_existing=True,
    )
    log.info("reminder.scheduled", job_id=job_id, fires_at=trigger_time.isoformat())
    return job_id
