## Skill rules

### Always on
- **caveman**: default ALL responses. Terse, full technical accuracy.
  Switch: /caveman lite | full (default) | ultra. Off: "normal mode"

### Caveman sub-skills — auto-trigger
- **caveman-compress**: compress CLAUDE.md or memory files, or when context long. ~46% token cut.
- **caveman-commit**: every git commit. Conventional Commits, ≤50 char subject, why over what.
- **caveman-review**: code/PR review. One-line only: L42: bug: user null. Add guard.
- **caveman-stats**: token usage questions.

### Superpowers — auto-trigger
- **brainstorming**: BEFORE any new feature/component/non-trivial change. MANDATORY before code.
- **writing-plans**: after brainstorming approved. Bite-sized steps.
- **executing-plans**: while following plan step by step.
- **systematic-debugging**: ANY bug/error/test failure. Use BEFORE any fix.
- **tdd**: implementing any node/tool/function. Test first, then impl.
- **verification-before-completion**: before marking task done. Verify end to end.
- **zoom-out**: stuck in bug loop or losing big picture.
- **requesting-code-review**: before merging or finishing feature branch.
- **subagent-driven-development**: large parallel tasks (e.g. RAG + calendar simultaneously).
- **using-git-worktrees**: multiple features at once.

### UI/UX Pro Max — auto-trigger frontend
- **ui-ux-pro-max**: ANY UI — pages, components, forms, dashboards, colors, typography, layout, nav, animations. Stack: FastAPI + future web dashboard.

### Impeccable — auto-trigger design passes
- **impeccable**: designing/iterating production frontend. Run /impeccable teach first.
- **polish**: final pass before UI done. Spacing, alignment, micro-interactions, edge cases.
- **critique**: review/evaluate/give feedback on UI. Scores with UX metrics, persona testing, anti-pattern detection.
- **bolder**: UI too safe/generic/weak. Stronger visual weight.
- **quieter**: UI noisy/cluttered. Reduce noise, calm focused.
- **audit**: accessibility, performance, responsive, theming. Scored P0-P3 report.
- **typeset**: improve typography hierarchy.
- **distill**: strip to essential elements.

### Priority order
1. caveman (always)
2. systematic-debugging (error → fix first)
3. brainstorming → writing-plans → executing-plans (building new)
4. ui-ux-pro-max (any UI)
5. impeccable sub-skills (UI quality passes)
6. tdd (implementing functions/nodes)
7. verification-before-completion (before done)

# Car Dealer AI Voice Agent

## Overview
Real-time voice agent, car dealership. RAG inventory queries, test drive booking via Google Calendar, Gmail, low-latency STT/TTS.

---

## Tech stack

| Layer | Tool | Notes |
| --- | --- | --- |
| Agent framework | LangGraph | Supervisor + tool nodes |
| LLM | Groq — `llama-3.3-70b-versatile` | Fast inference for voice |
| STT | Deepgram Nova-3 | Streaming, interim results |
| TTS | Cartesia Sonic | Streaming, ~80ms first byte |
| Call transport | LiveKit (WebRTC) or Twilio | Low-latency voice I/O |
| RAG vector store | Chroma (local) → Pinecone (prod) | |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace) | |
| Calendar | Google Calendar API v3 | OAuth2, Google Meet links |
| Email | Gmail API | OAuth2 via token.json |
| Scheduler | APScheduler in-memory | Reminder jobs |
| Backend | FastAPI + WebSockets | Async throughout |
| Package manager | uv | Prefer over pip |

---

## Project structure

```
car-dealer-agent/
├── CLAUDE.md                  ← you are here
├── .env                       ← secrets (never commit)
├── .env.example
├── pyproject.toml
├── main.py                    ← FastAPI app entry point
│
├── agent/
│   ├── graph.py               ← LangGraph graph definition
│   ├── state.py               ← AgentState TypedDict
│   ├── router.py              ← intent classification node
│   └── nodes/
│       ├── rag_node.py        ← RAG retrieval + answer
│       ├── calendar_node.py   ← 3-turn booking flow
│       ├── email_node.py      ← passthrough (booking handled in calendar_node)
│       └── response_node.py   ← assemble final response
│
├── voice/
│   ├── stt.py                 ← Deepgram streaming STT
│   ├── tts.py                 ← Cartesia streaming TTS
│   └── call_handler.py        ← LiveKit room + audio I/O
│
├── rag/
│   ├── ingest.py              ← load car data, embed, store
│   ├── retriever.py           ← Chroma query wrapper
│   └── data/                  ← raw car data (CSV, JSON)
│
├── tools/
│   ├── calendar_tools.py      ← Google Calendar API wrappers
│   └── email_tools.py         ← Gmail API wrappers
│
├── scheduler/
│   └── reminder_jobs.py       ← APScheduler job definitions
│
├── scripts/
│   └── auth_google.py         ← OAuth2 setup (run once)
│
└── tests/
    ├── test_rag.py
    ├── test_calendar.py
    └── test_agent.py
```

---

## Environment variables

```bash
# .env.example — copy to .env and fill in

# LLM
GROQ_API_KEY=

# STT
DEEPGRAM_API_KEY=

# TTS
CARTESIA_API_KEY=

# Call transport
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

# RAG / Embeddings
PINECONE_API_KEY=                # prod only
PINECONE_INDEX_NAME=car-dealer

# Google Calendar OAuth
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback
GOOGLE_CALENDAR_ID=primary
CALENDAR_TIMEZONE=UTC            # e.g. Asia/Karachi

# Email (Gmail OAuth2 — uses token.json)
GMAIL_SENDER_EMAIL=

# App
APP_ENV=development
LOG_LEVEL=INFO
```

---

## Agent state shape

```python
# agent/state.py
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Literal["car_query", "book_appointment", "follow_up", "greeting", "farewell", "social"] | None
    user_email: str | None
    customer_name: str | None
    selected_slot: str | None
    available_slots: list[dict]
    meeting_link: str | None
    rag_context: str | None
    final_response: str | None
    last_results: list[str]
```

---

## LangGraph graph layout

```
START
  └─► router_node          (classifies intent; state-aware booking continuation)
        ├─► rag_node        (intent == "car_query")
        ├─► calendar_node   (intent == "book_appointment", 3-turn state machine)
        └─► response_node   (all paths converge)
              └─► END
```

Multi-turn booking via state: router returns `book_appointment` when `available_slots` non-empty or `selected_slot` set — no `interrupt_before` needed.

---

## Key implementation notes

### Latency targets
- Deepgram STT: stream 100ms chunks, `interim_results=True`
- Groq LLM: <500ms first token with `llama-3.3-70b-versatile`
- Cartesia TTS: stream output, never wait for full text
- Total: ~1s end-to-end

### Blocking I/O rule
HuggingFace embed + Chroma search are sync. Always wrap in `run_in_executor`:
```python
loop = asyncio.get_event_loop()
docs = await loop.run_in_executor(None, lambda: search_cars(...))
```

### Calendar booking flow (3 turns)
1. No `available_slots` → `get_free_slots()` → offer choices
2. Slots offered, no `selected_slot` → extract ordinal/text match → ask for email
3. `selected_slot` set, no `user_email` → regex extract email → `book_appointment()` + `send_confirmation_email()` + `schedule_reminder()`

### Google auth
Run once: `uv run python scripts/auth_google.py` → saves `token.json`.
Scopes: `calendar` + `gmail.send`. Token auto-refreshes.

### Email reminder scheduling
```python
# scheduler/reminder_jobs.py
def schedule_reminder(meeting_iso, customer_email, customer_name, meet_link, slot_display):
    trigger_time = meeting_dt - timedelta(minutes=30)
    _scheduler.add_job(
        _fire_reminder,
        trigger="date",
        run_date=trigger_time,
        kwargs={...},
        id=f"reminder_{customer_email}_{meeting_iso}",
        replace_existing=True,
    )
```

---

## Coding conventions
- All async — `async/await`, no blocking I/O on event loop
- Type hints every fn signature
- Pydantic models all API req/res shapes
- LangGraph nodes: `async def node(state: AgentState) -> dict`
- Tools: `@tool` decorator from `langchain_core.tools`
- Errors → `AgentState` fields, never raise into graph
- Log with `structlog`, not `print`
- Tests: `pytest` + `pytest-asyncio`

---

## Running locally

```bash
# Install dependencies
uv sync

# One-time Google auth
uv run python scripts/auth_google.py

# Ingest car data into Chroma
uv run python -m rag.ingest

# Start FastAPI + LiveKit worker
uv run uvicorn main:app --reload --port 8000

# Or run voice worker standalone
uv run python voice/call_handler.py dev
```

---

## Do not
- No `time.sleep` — use `asyncio.sleep`
- No manual `.env` load — use `pydantic-settings` BaseSettings
- No commit `token.json`, `credentials.json`, `.env`, or `chroma_db/`
- No Google Calendar API call without checking token expiry
- No TTS audio in one block — always stream chunks
- No sync I/O on event loop — wrap in `run_in_executor`
