"""
Google OAuth2 setup — run once before using Calendar or Gmail features.

Usage:
    uv run python scripts/auth_google.py

Requires credentials.json in the project root (download from Google Cloud Console:
  APIs & Services → Credentials → OAuth 2.0 Client ID → Download JSON).

Saves token.json on success. token.json is gitignored — never commit it.
"""
from pathlib import Path

CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
]


def main() -> None:
    if not CREDENTIALS_FILE.exists():
        print(
            "credentials.json not found — download it from "
            "Google Cloud Console and place in project root.\n"
            "  Console → APIs & Services → Credentials → "
            "OAuth 2.0 Client IDs → ⬇ Download JSON"
        )
        return

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)

    TOKEN_FILE.write_text(creds.to_json())
    print("Auth successful! token.json saved.")


if __name__ == "__main__":
    main()
