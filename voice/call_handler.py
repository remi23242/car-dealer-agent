"""
LiveKit voice agent pipeline:
  Deepgram Nova-3 STT → LangGraph agent → Deepgram Aura TTS

Run locally:
  uv run python voice/call_handler.py dev
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid

import structlog
from langchain_core.messages import HumanMessage
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm as agents_llm,
)
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    LLMStream,
    Tool,
)
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins import deepgram
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.graph import graph

log = structlog.get_logger()

# Primary: split after . ! ? followed by whitespace.
# Keeps abbreviations like "U.S." intact (no space follows).
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
# Fallback: split after comma — only used when buffer exceeds 6 words.
_COMMA_RE = re.compile(r"(?<=,)\s+")

_TTS_TAG = "tts-stream"  # tag on synthesis LLM calls we want to stream to TTS
_SYNTHESIS_NODES = frozenset({"rag_node", "response_node"})


def _split_sentences(buf: str) -> tuple[list[str], str]:
    """Return (complete sentences, remaining incomplete text).

    Falls back to comma splits when buf exceeds 6 words and no sentence
    boundary is found — keeps TTS latency low on long comma-separated clauses.
    """
    parts = _SENT_RE.split(buf)
    if len(parts) > 1:
        complete = [s.strip() for s in parts[:-1] if s.strip()]
        return complete, parts[-1]
    if len(buf.split()) > 6:
        comma_parts = _COMMA_RE.split(buf)
        if len(comma_parts) > 1:
            complete = [s.strip() for s in comma_parts[:-1] if s.strip()]
            return complete, comma_parts[-1]
    return [], buf


def normalize_for_tts(text: str) -> str:
    replacements = {
        " HP": " horsepower",
        " hp": " horsepower",
        "(HP)": "(horsepower)",
        " MPG": " miles per gallon",
        " mpg": " miles per gallon",
        " SUV": " S U V",
        " AWD": " all wheel drive",
        " FWD": " front wheel drive",
        " RWD": " rear wheel drive",
        " 4WD": " four wheel drive",
        "4dr": "4 door",
        "2dr": "2 door",
        " V8": " V 8",
        " V6": " V 6",
        " V4": " V 4",
        "0.0 HP": "horsepower",
        " cc": " cubic centimeters",
        " L ": " liter ",
        "1.5L": "1.5 liter",
        "2.0L": "2.0 liter",
        "3.5L": "3.5 liter",
        " rpm": " R P M",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # $1,705,769 → 1,705,769 dollars  (proper comma groups only, no trailing comma)
    text = re.sub(r'\$([0-9]+(?:,[0-9]+)*)', r'\1 dollars', text)
    # 1001.0 → 1001  (whole numbers with redundant .0 suffix)
    text = re.sub(r'\b(\d+)\.0\b', r'\1', text)
    return text


AGENT_INSTRUCTIONS = (
    "You are a car dealership voice assistant. "
    "Respond in max 2 short sentences. Conversational, no lists, no markdown."
)


# ── settings ─────────────────────────────────────────────────────────────────

class VoiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    deepgram_api_key: str
    deepgram_tts_voice: str = "aura-2-thalia-en"
    dealership_name: str = "Premier Auto Dealership"
    agent_name: str = "Alex"


# ── graph warmup + opening greeting ──────────────────────────────────────────

async def _greet(session: AgentSession, dealership_name: str, agent_name: str) -> None:
    await asyncio.sleep(1.0)
    greeting = (
        f"Thank you for calling {dealership_name}! "
        f"This is {agent_name}, your virtual assistant. "
        "How can I help you find your perfect vehicle today?"
    )
    try:
        await session.say(greeting)
        log.info("car_dealer_agent.greeted")
    except Exception as exc:
        log.warning("car_dealer_agent.greet_failed", error=str(exc))


async def _warmup_graph(tid: str) -> None:
    try:
        t0 = time.monotonic()
        # Use a dedicated warmup thread so state doesn't bleed into the real conversation.
        # Real message required — router_node crashes on empty messages list.
        await graph.ainvoke(
            {"messages": [HumanMessage(content="Do you have any SUVs?")]},
            config={"configurable": {"thread_id": f"warmup-{tid}"}},
        )
        log.info("graph.warmup_done", ms=round((time.monotonic() - t0) * 1000), thread_id=tid)
    except Exception as exc:
        log.warning("graph.warmup_failed", error=str(exc), exc_info=True)


# ── custom LLM wrapping LangGraph ────────────────────────────────────────────

class LangGraphStream(LLMStream):
    """One LangGraph turn. Cancels immediately if a newer user utterance arrives."""

    def __init__(
        self,
        owner: "LangGraphLLM",
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        generation: int,
        cancel_event: asyncio.Event,
    ) -> None:
        super().__init__(owner, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._generation = generation
        self._cancel_event = cancel_event

    async def _run(self) -> None:
        log.info("_run_entered", generation=self._generation)
        gen = self._generation
        t0 = time.monotonic()
        log.info("run.start", generation=gen)
        chunks_sent = 0
        try:
            llm: LangGraphLLM = self._llm  # type: ignore[assignment]

            # Extract last user utterance
            user_text = ""
            for msg in reversed(self._chat_ctx.messages()):
                if msg.role == "user":
                    parts = msg.content if isinstance(msg.content, list) else [msg.content]
                    for part in parts:
                        if isinstance(part, str) and part.strip():
                            user_text = part.strip()
                            break
                if user_text:
                    break

            if not user_text:
                log.warning("langgraph_stream.no_user_text")
                return

            t_start = time.monotonic()
            log.info("langgraph_stream.run_start", generation=gen, text=user_text[:60])

            intent: str | None = None
            t_graph_start: float | None = None
            t_first_token: float | None = None
            t_first_chunk: float | None = None

            async def _stream() -> None:
                nonlocal chunks_sent, intent, t_graph_start, t_first_token, t_first_chunk
                log.info("run.stream_entered", ms=round((time.monotonic() - t0) * 1000), generation=gen)
                token_buf = ""
                filler_sent = False

                # Snapshot state BEFORE graph runs so we know if mid-booking.
                # Filler for book_appointment only fires on step 1 (no slots yet).
                booking_in_progress = False
                try:
                    _snap = await graph.aget_state({"configurable": {"thread_id": llm._thread_id}})
                    if _snap and _snap.values:
                        booking_in_progress = bool(
                            _snap.values.get("available_slots") or _snap.values.get("selected_slot")
                        )
                except Exception:
                    pass

                # ── latency 1: endpoint → graph start ────────────────────────
                t_graph_start = time.monotonic()
                log.info("run.graph_start", ms=round((t_graph_start - t0) * 1000), generation=gen)
                log.info(
                    "latency.endpoint_to_graph_ms",
                    ms=round((t_graph_start - t_start) * 1000),
                    generation=gen,
                )

                t_first_event_logged = False

                async for evt in graph.astream_events(
                    {"messages": [HumanMessage(content=user_text)]},
                    config={"configurable": {"thread_id": llm._thread_id}},
                    version="v2",
                ):
                    if self._cancel_event.is_set() or llm._generation != gen:
                        return

                    if not t_first_event_logged:
                        log.info("run.first_event", ms=round((time.monotonic() - t0) * 1000), generation=gen)
                        log.info(
                            "timing.graph_first_event",
                            ms=round((time.monotonic() - t_start) * 1000),
                            evt_type=evt["event"],
                            generation=gen,
                        )
                        t_first_event_logged = True

                    etype = evt["event"]
                    node = evt.get("metadata", {}).get("langgraph_node", "")

                    if (
                        etype == "on_chat_model_stream"
                        and node in _SYNTHESIS_NODES
                        and _TTS_TAG in evt.get("tags", [])
                    ):
                        token = evt["data"]["chunk"].content
                        if token:
                            if t_first_token is None:
                                # ── latency 2: graph start → first token ──────
                                t_first_token = time.monotonic()
                                log.info("run.first_token", ms=round((t_first_token - t0) * 1000), generation=gen)
                                log.info(
                                    "timing.first_synth_token",
                                    ms=round((t_first_token - t_start) * 1000),
                                    generation=gen,
                                )
                                log.info(
                                    "latency.graph_to_first_token_ms",
                                    ms=round((t_first_token - t_graph_start) * 1000),
                                    generation=gen,
                                )
                            token_buf += token
                            sentences, token_buf = _split_sentences(token_buf)
                            for sentence in sentences:
                                if self._cancel_event.is_set() or llm._generation != gen:
                                    return
                                if t_first_chunk is None:
                                    # ── latency 3: first token → first chunk ──
                                    t_first_chunk = time.monotonic()
                                    log.info("run.first_chunk", ms=round((t_first_chunk - t0) * 1000), generation=gen)
                                    log.info(
                                        "latency.first_token_to_first_chunk_ms",
                                        ms=round((t_first_chunk - t_first_token) * 1000),
                                        generation=gen,
                                    )
                                self._event_ch.send_nowait(ChatChunk(
                                    id=f"lg-{gen}-{chunks_sent}",
                                    delta=ChoiceDelta(role="assistant", content=normalize_for_tts(sentence)),
                                ))
                                chunks_sent += 1

                    elif etype == "on_chain_end" and node == "router_node":
                        out = evt["data"].get("output", {})
                        if isinstance(out, dict):
                            intent = out.get("intent") or intent
                            # Emit filler so user hears something immediately
                            # while RAG / calendar API runs (typically 5-10s).
                            if not filler_sent and chunks_sent == 0:
                                if intent == "car_query":
                                    filler = "Let me check our inventory for you!"
                                elif intent == "book_appointment" and not booking_in_progress:
                                    filler = "Let me pull up our available times!"
                                else:
                                    filler = None
                                if filler and not self._cancel_event.is_set():
                                    self._event_ch.send_nowait(ChatChunk(
                                        id=f"lg-{gen}-filler",
                                        delta=ChoiceDelta(role="assistant", content=filler),
                                    ))
                                    filler_sent = True
                                    log.info("filler.sent", intent=intent, generation=gen)

                # Flush trailing text when groq stream ends — inside _stream() so
                # it fires before asyncio.wait returns.
                remainder = token_buf.strip()
                if remainder and not self._cancel_event.is_set() and llm._generation == gen:
                    if t_first_chunk is None and t_first_token is not None:
                        t_first_chunk = time.monotonic()
                        log.info(
                            "latency.first_token_to_first_chunk_ms",
                            ms=round((t_first_chunk - t_first_token) * 1000),
                            generation=gen,
                        )
                    self._event_ch.send_nowait(ChatChunk(
                        id=f"lg-{gen}-{chunks_sent}",
                        delta=ChoiceDelta(role="assistant", content=normalize_for_tts(remainder)),
                    ))
                    chunks_sent += 1

            stream_task = asyncio.create_task(_stream(), name=f"stream-gen{gen}")
            cancel_task = asyncio.create_task(self._cancel_event.wait(), name=f"cancel-gen{gen}")

            done, pending = await asyncio.wait(
                {stream_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            if cancel_task in done:
                log.info("langgraph_stream.interrupted", generation=gen)
                return

            if llm._generation != gen:
                log.info("langgraph_stream.stale", generation=gen)
                return

            if stream_task in done and (exc := stream_task.exception()):
                raise exc

            t_done = time.monotonic()
            log.info(
                "latency.turn_total_ms",
                ms=round((t_done - t_start) * 1000),
                graph_to_first_token_ms=(
                    round((t_first_token - t_graph_start) * 1000)
                    if t_first_token and t_graph_start else None
                ),
                chunks=chunks_sent,
                generation=gen,
            )

            if chunks_sent == 0:
                fallback = None
                try:
                    snap = await graph.aget_state(
                        {"configurable": {"thread_id": llm._thread_id}}
                    )
                    fallback = snap.values.get("final_response")
                except Exception:
                    pass
                if not fallback:
                    if intent == "book_appointment":
                        fallback = (
                            "I can schedule a test drive for you. "
                            "What date and time works best, and what's your email address?"
                        )
                    else:
                        fallback = "Could you say that again?"
                self._event_ch.send_nowait(ChatChunk(
                    id=f"lg-{gen}-fallback",
                    delta=ChoiceDelta(role="assistant", content=normalize_for_tts(fallback)),
                ))

        except asyncio.CancelledError:
            raise
        except Exception as _exc:
            log.error("langgraph_stream._run_crashed", error=str(_exc), exc_info=True)
            self._event_ch.send_nowait(ChatChunk(
                id="crash-fallback",
                delta=ChoiceDelta(
                    role="assistant",
                    content="I had a small issue there. Could you repeat that?",
                ),
            ))
        finally:
            log.info("langgraph_stream.run_end", chunks=chunks_sent, generation=gen)


class LangGraphLLM(agents_llm.LLM):
    """LangGraph-backed LLM adapter with interruption support.

    Each call to chat() cancels the previous in-flight graph.ainvoke so the
    latest user utterance always wins — no response queuing.
    """

    def __init__(self, thread_id: str) -> None:
        super().__init__()
        self._thread_id = thread_id
        self._generation: int = 0
        self._cancel: asyncio.Event | None = None

    @property
    def model(self) -> str:
        return "llama-3.3-70b-versatile"

    @property
    def provider(self) -> str:
        return "groq-langgraph"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **_kwargs,
    ) -> LangGraphStream:
        if self._cancel is not None:
            self._cancel.set()

        self._generation += 1
        self._cancel = asyncio.Event()

        log.info("langgraph_llm.new_turn", generation=self._generation)

        return LangGraphStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            generation=self._generation,
            cancel_event=self._cancel,
        )


# ── CarDealerAgent: per-room session lifecycle ────────────────────────────────

class CarDealerAgent:
    """Owns one AgentSession for a single LiveKit room.

    Ties together Deepgram STT → LangGraphLLM → Cartesia TTS.
    thread_id scoped to room_name so conversation memory persists per call.
    """

    def __init__(self, room_name: str) -> None:
        self._room_name = room_name
        self._thread_id = room_name or str(uuid.uuid4())
        self._session: AgentSession | None = None

    async def start(self, room: rtc.Room) -> None:
        settings = VoiceSettings()

        tts_instance = deepgram.TTS(
            model=settings.deepgram_tts_voice,
            api_key=settings.deepgram_api_key,
        )

        self._session = AgentSession(
            stt=deepgram.STT(
                model="nova-3",
                language="en-US",
                interim_results=True,
                punctuate=True,
                smart_format=True,
                endpointing_ms=300,
                api_key=settings.deepgram_api_key,
            ),
            llm=LangGraphLLM(thread_id=self._thread_id),
            tts=tts_instance,
            allow_interruptions=True,
            min_interruption_duration=0.3,
            min_interruption_words=1,
        )

        log.info("car_dealer_agent.starting", room=self._room_name)
        await self._session.start(
            agent=Agent(instructions=AGENT_INSTRUCTIONS),
            room=room,
        )
        log.info("car_dealer_agent.started", room=self._room_name)

        # Greet caller after 1s and warm up LangGraph concurrently.
        asyncio.create_task(_greet(self._session, settings.dealership_name, settings.agent_name))
        asyncio.create_task(_warmup_graph(self._thread_id))

    async def stop(self) -> None:
        if self._session is not None:
            try:
                await self._session.aclose()
            except Exception as exc:
                log.warning("car_dealer_agent.stop_error", error=str(exc))
            self._session = None
        log.info("car_dealer_agent.stopped", room=self._room_name)


# ── LiveKit entrypoint ────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    log.info("call.participant_joined", identity=participant.identity, room=ctx.room.name)

    agent = CarDealerAgent(room_name=ctx.room.name or participant.identity)
    ctx.add_shutdown_callback(agent.stop)
    await agent.start(room=ctx.room)
    log.info("call.session_started", room=ctx.room.name)


# ── worker entry ──────────────────────────────────────────────────────────────

def run_agent() -> None:
    """Start the LiveKit worker. Call this from main.py or run directly."""
    settings = VoiceSettings()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        )
    )


if __name__ == "__main__":
    run_agent()
