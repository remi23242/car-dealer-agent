"""
Gmail email tools — confirmation and reminder emails via Gmail API.

Uses same token.json as calendar_tools (gmail.send scope included).
"""
from __future__ import annotations

import asyncio
import base64
from email.mime.text import MIMEText
from pathlib import Path

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


class EmailSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    gmail_sender_email: str


def _get_gmail_service():
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
        log.info("email.token_refreshed")
    return build("gmail", "v1", credentials=creds)


def _build_message(sender: str, to: str, subject: str, body: str) -> dict:
    msg = MIMEText(body, "plain")
    msg["to"] = to
    msg["from"] = sender
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return {"raw": raw}


def _send_sync(to_email: str, subject: str, body: str) -> bool:
    settings = EmailSettings()
    service = _get_gmail_service()
    message = _build_message(settings.gmail_sender_email, to_email, subject, body)
    service.users().messages().send(userId="me", body=message).execute()
    log.info("email.sent", to=to_email, subject=subject)
    return True


async def send_confirmation_email(
    to_email: str,
    customer_name: str,
    slot: str,
    meet_link: str,
    dealership_name: str = "Premier Auto Dealership",
) -> bool:
    subject = "Your Test Drive is Confirmed!"
    body = (
        f"Hi {customer_name},\n\n"
        f"Your test drive at {dealership_name} is confirmed for {slot}.\n\n"
        f"Join via Google Meet: {meet_link}\n\n"
        "See you soon!\n"
        f"— {dealership_name} Team"
    )
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _send_sync, to_email, subject, body)


async def send_reminder_email(
    to_email: str,
    customer_name: str,
    slot: str,
    meet_link: str,
    dealership_name: str = "Premier Auto Dealership",
) -> bool:
    subject = "Reminder: Your Test Drive in 30 Minutes"
    body = (
        f"Hi {customer_name},\n\n"
        f"Your test drive at {dealership_name} is coming up at {slot} — just 30 minutes away!\n\n"
        f"Join via Google Meet: {meet_link}\n\n"
        "See you soon!\n"
        f"— {dealership_name} Team"
    )
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _send_sync, to_email, subject, body)
