import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta

import structlog
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from livekit.api import AccessToken, VideoGrants
from livekit.agents import WorkerOptions
from livekit.agents.worker import AgentServer
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.graph import graph

log = structlog.get_logger()

# Forward livekit.agents standard-library logs into uvicorn's output
logging.getLogger("livekit.agents").setLevel(logging.INFO)
logging.getLogger("livekit").setLevel(logging.INFO)


# ── settings ──────────────────────────────────────────────────────────────────

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""


# ── lifespan: start LiveKit worker alongside FastAPI ─────────────────────────

_worker_task: asyncio.Task | None = None

_PLAYGROUND_ROOM = "car-dealer"


def _make_room_token(settings: AppSettings, room: str, identity: str) -> str:
    return (
        AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(timedelta(hours=4))
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_task

    from scheduler.reminder_jobs import start_scheduler
    start_scheduler()

    # Pre-load HuggingFace embedding model + Chroma DB so the first real
    # caller doesn't pay the cold-start penalty (~3-8 seconds on CPU).
    try:
        import asyncio as _asyncio
        from rag.retriever import search_cars as _search_cars
        await _asyncio.get_running_loop().run_in_executor(
            None, lambda: _search_cars("warmup", top_k=1)
        )
        log.info("rag.embedding_warmup_done")
    except Exception as _e:
        log.warning("rag.embedding_warmup_failed", reason=str(_e))

    try:
        from voice.call_handler import VoiceSettings, entrypoint

        vs = VoiceSettings()
        opts = WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=vs.livekit_url,
            api_key=vs.livekit_api_key,
            api_secret=vs.livekit_api_secret,
        )
        server = AgentServer.from_server_options(opts)

        @server.on("worker_registered")
        def _on_registered(worker_id: str, server_info) -> None:
            log.info(
                "livekit.worker_registered",
                worker_id=worker_id,
                url=vs.livekit_url,
                region=getattr(server_info, "region", ""),
            )

        _worker_task = asyncio.create_task(server.run(), name="livekit-worker")
        log.info("livekit.worker_started", url=vs.livekit_url)
    except Exception as e:
        log.warning("livekit.worker_skipped", reason=str(e))

    # print playground URL so developer can click straight into a call
    try:
        _s = AppSettings()
        if _s.livekit_api_key and _s.livekit_api_secret:
            _token = _make_room_token(_s, _PLAYGROUND_ROOM, f"dev-{uuid.uuid4().hex[:6]}")
            _playground = f"https://agents-playground.livekit.io/#token={_token}"
            print(f"\n  LiveKit Playground: {_playground}\n", flush=True)
    except Exception as _e:
        log.warning("playground_url_skipped", reason=str(_e))

    # print all registered routes
    routes = [
        f"  {m:7} {r.path}"
        for r in app.routes
        if hasattr(r, "methods") and r.methods
        for m in sorted(r.methods)
        if m not in ("HEAD", "OPTIONS")
    ]
    # include websocket routes
    ws_routes = [f"  WS      {r.path}" for r in app.routes if not hasattr(r, "methods")]
    for line in sorted(routes) + ws_routes:
        log.info("route.registered", route=line.strip())

    yield  # app runs here

    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await _worker_task
        except (asyncio.CancelledError, Exception):
            pass
        log.info("livekit.worker_stopped")


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Car Dealer AI Voice Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: str = ""


class ChatResponse(BaseModel):
    thread_id: str
    intent: str | None
    response: str | None


class TokenResponse(BaseModel):
    token: str
    room_name: str
    ws_url: str


class SimpleTokenResponse(BaseModel):
    token: str
    url: str
    room: str


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/token", response_model=SimpleTokenResponse)
async def token() -> SimpleTokenResponse:
    """Generate a LiveKit join token for the car-dealer room (playground-ready)."""
    settings = AppSettings()
    identity = f"user-{uuid.uuid4().hex[:8]}"
    tok = _make_room_token(settings, _PLAYGROUND_ROOM, identity)
    log.info("token.issued", identity=identity, room=_PLAYGROUND_ROOM)
    return SimpleTokenResponse(token=tok, url=settings.livekit_url, room=_PLAYGROUND_ROOM)


@app.get("/livekit-token", response_model=TokenResponse)
async def livekit_token(
    identity: str = Query(..., description="Participant identity (user ID)"),
    room: str = Query("", description="Room name — omit to auto-create"),
) -> TokenResponse:
    """
    Generate a LiveKit room-join token for a participant.
    The LiveKit agent worker joins automatically when the first human enters.
    """
    settings = AppSettings()
    room_name = room.strip() or f"room-{uuid.uuid4().hex[:8]}"
    tok = _make_room_token(settings, room_name, identity)
    log.info("token.issued", identity=identity, room=room_name)
    return TokenResponse(token=tok, room_name=room_name, ws_url=settings.livekit_url)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    log.info("chat.request", thread_id=thread_id, message=req.message[:80])

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )

    intent = result.get("intent")
    response = result.get("final_response")

    if intent == "book_appointment" and response is None:
        response = (
            "I can book a test drive for you. "
            "Please provide your preferred date and time, and your email address."
        )

    log.info("chat.response", thread_id=thread_id, intent=intent, chars=len(response or ""))
    return ChatResponse(thread_id=thread_id, intent=intent, response=response)


@app.post("/chat/resume", response_model=ChatResponse)
async def chat_resume(req: ChatRequest) -> ChatResponse:
    """Resume an interrupted graph (book_appointment slot confirmation)."""
    if not req.thread_id:
        return ChatResponse(thread_id="", intent=None, response="thread_id required to resume.")

    config = {"configurable": {"thread_id": req.thread_id}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )

    response = result.get("final_response")
    if response is None:
        response = "Still waiting for a slot confirmation — please provide date, time, and email."

    log.info("chat.resume", thread_id=req.thread_id, intent=result.get("intent"))
    return ChatResponse(
        thread_id=req.thread_id,
        intent=result.get("intent"),
        response=response,
    )


@app.websocket("/call")
async def call_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info("call.connected")
    # Voice handled by the LiveKit worker started in lifespan().
    await websocket.close()
