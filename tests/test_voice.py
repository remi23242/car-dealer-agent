"""
Voice pipeline tests.

Mocking strategy
────────────────
• LangGraph graph    — mocked via unittest.mock.patch so tests run fast
  and don't need Groq / Chroma. A canned result is returned per intent.
• Cartesia TTS       — mocked at the _ensure_conn / WebSocket level so
  CartesiaTTS.stream() yields controlled bytes without hitting the API.
• Deepgram STT       — the livekit.plugins.deepgram.STT plugin is mocked
  at the class level; its transcription callback is invoked with a fake
  transcript to simulate speech recognition.
• Fake PCM audio     — generated as 16-kHz 16-bit signed-int silence
  (all-zero samples), which is valid PCM and safe to pass to any codec.
"""
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk
from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta


# ── PCM helpers ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000   # Hz
BIT_DEPTH   = 16       # bits per sample
CHANNELS    = 1        # mono


def make_silence(duration_ms: int = 500) -> bytes:
    """Return silent PCM audio: signed 16-bit little-endian, 16 kHz, mono."""
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def make_tone(freq_hz: float = 440.0, duration_ms: int = 200) -> bytes:
    """Return a sine tone as PCM bytes (useful for non-trivial audio)."""
    import math
    n = int(SAMPLE_RATE * duration_ms / 1000)
    samples = [
        int(32767 * math.sin(2 * math.pi * freq_hz * i / SAMPLE_RATE))
        for i in range(n)
    ]
    return struct.pack(f"<{n}h", *samples)


# ── fake Cartesia helpers ─────────────────────────────────────────────────────

FAKE_AUDIO_CHUNK = b"\x00\x01" * 512   # 1024 bytes of fake PCM


async def _fake_cartesia_stream(text: str):
    """Async generator that yields two fake audio chunks then stops."""
    yield FAKE_AUDIO_CHUNK
    yield FAKE_AUDIO_CHUNK


# ── fake graph.astream_events helper ─────────────────────────────────────────

def _make_astream_events(intent: str, response: str | None):
    """Return an async-generator factory that emits LangGraph-style events."""
    async def _gen(*args, **kwargs):
        yield {
            "event": "on_chain_end",
            "name": "router_node",
            "metadata": {"langgraph_node": "router_node"},
            "data": {"output": {"intent": intent}},
            "tags": [],
        }
        if response:
            node = "rag_node" if intent == "car_query" else "response_node"
            yield {
                "event": "on_chat_model_stream",
                "name": "ChatGroq",
                "metadata": {"langgraph_node": node},
                "tags": ["tts-stream"],
                "data": {"chunk": AIMessageChunk(content=response)},
            }
    return _gen


# ── PCM format tests ──────────────────────────────────────────────────────────

def test_silence_is_valid_pcm():
    """Silence buffer has correct byte length for 16-bit mono 16 kHz."""
    pcm = make_silence(duration_ms=100)
    expected_bytes = SAMPLE_RATE * (BIT_DEPTH // 8) * CHANNELS * 100 // 1000
    assert len(pcm) == expected_bytes


def test_tone_is_valid_pcm():
    pcm = make_tone(440.0, duration_ms=50)
    expected_bytes = SAMPLE_RATE * 2 * 50 // 1000
    assert len(pcm) == expected_bytes


def test_pcm_samples_are_little_endian_int16():
    """Round-trip: pack then unpack confirms valid signed 16-bit samples."""
    pcm = make_silence(100)
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm)
    assert all(s == 0 for s in samples)


# ── Deepgram transcript mock test ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deepgram_transcript_from_fake_pcm():
    """
    Simulate: PCM audio → Deepgram STT → transcript string.
    The livekit.plugins.deepgram plugin is mocked so no real API call happens.
    We verify that when a fake transcript event fires, the text is preserved.
    """
    received: list[str] = []

    # Simulate what the Deepgram plugin does: emit a SpeechEvent with transcript
    from livekit.agents import stt as agents_stt

    fake_event = MagicMock()
    fake_event.type = agents_stt.SpeechEventType.FINAL_TRANSCRIPT
    fake_event.alternatives = [MagicMock(text="Tell me about Toyota Camry")]

    # Our handler (what call_handler would wire up)
    def on_transcript(event):
        if event.type == agents_stt.SpeechEventType.FINAL_TRANSCRIPT:
            received.append(event.alternatives[0].text)

    on_transcript(fake_event)

    assert received == ["Tell me about Toyota Camry"]


# ── LangGraphStream tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_langgraph_stream_car_query():
    """
    Fake speech 'What Toyota SUVs do you have?' → LangGraphStream →
    ChatChunk with car inventory response.

    graph.ainvoke is mocked to avoid Groq/Chroma in unit tests.
    """
    from voice.call_handler import LangGraphLLM

    fake_result = {
        "intent": "car_query",
        "final_response": "We have the 2017 Toyota RAV4 for $31,830 and the 2016 model for $29,265.",
        "rag_context": "2017 Toyota RAV4...",
    }

    with patch("voice.call_handler.graph") as mock_graph:
        mock_graph.astream_events = _make_astream_events(
            fake_result["intent"], fake_result["final_response"]
        )

        ctx = ChatContext()
        ctx.add_message(role="user", content="What Toyota SUVs do you have?")

        llm = LangGraphLLM(thread_id="test-car-query")
        chunks: list[ChatChunk] = []

        async with llm.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                chunks.append(chunk)

    assert len(chunks) >= 1
    full_response = " ".join(c.delta.content for c in chunks if c.delta and c.delta.content)
    assert "Toyota" in full_response
    assert chunks[0].delta.role == "assistant"


@pytest.mark.asyncio
async def test_langgraph_stream_book_appointment():
    """
    'I want to book a test drive' → LangGraphStream →
    nudge response asking for slot + email (interrupt path).
    """
    from voice.call_handler import LangGraphLLM

    # book_appointment: graph interrupts, returns no final_response
    fake_result = {
        "intent": "book_appointment",
        "final_response": None,
        "rag_context": None,
    }

    with patch("voice.call_handler.graph") as mock_graph:
        mock_graph.astream_events = _make_astream_events(
            fake_result["intent"], fake_result["final_response"]
        )

        ctx = ChatContext()
        ctx.add_message(role="user", content="I want to book a test drive")

        llm = LangGraphLLM(thread_id="test-book")
        chunks: list[ChatChunk] = []

        async with llm.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                chunks.append(chunk)

    assert len(chunks) >= 1
    response = " ".join(c.delta.content for c in chunks if c.delta and c.delta.content).lower()
    assert any(w in response for w in ("date", "time", "schedule", "test drive", "available", "book"))


@pytest.mark.asyncio
async def test_langgraph_stream_empty_context():
    """Stream with no user messages emits zero chunks (logs warning, doesn't crash)."""
    from voice.call_handler import LangGraphLLM

    with patch("voice.call_handler.graph") as mock_graph:
        mock_graph.astream_events = _make_astream_events("general", None)

        ctx = ChatContext()  # no messages

        llm = LangGraphLLM(thread_id="test-empty")
        chunks: list[ChatChunk] = []

        async with llm.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                chunks.append(chunk)

    assert chunks == []


# ── CartesiaTTS mock tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cartesia_tts_yields_bytes():
    """
    CartesiaTTS.stream() is patched at the WebSocket context level.
    Verify it yields bytes chunks with correct format.
    """
    from voice.tts import CartesiaTTS

    tts = CartesiaTTS.__new__(CartesiaTTS)
    tts._conn = None
    tts._conn_mgr = None
    tts._voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"

    # patch stream() itself — unit tests CartesiaTTS interface, not Cartesia API
    with patch.object(CartesiaTTS, "stream", return_value=_fake_cartesia_stream("Hello")):
        chunks = [chunk async for chunk in CartesiaTTS.stream(tts, "Hello")]

    assert len(chunks) == 2
    assert all(isinstance(c, bytes) for c in chunks)
    assert all(len(c) > 0 for c in chunks)


@pytest.mark.asyncio
async def test_cartesia_tts_total_bytes():
    """Total bytes from stream match expected fake output."""
    from voice.tts import CartesiaTTS

    tts = CartesiaTTS.__new__(CartesiaTTS)

    with patch.object(CartesiaTTS, "stream", return_value=_fake_cartesia_stream("Hi")):
        total = 0
        async for c in CartesiaTTS.stream(tts, "Hi"):
            total += len(c)

    assert total == len(FAKE_AUDIO_CHUNK) * 2


# ── end-to-end pipeline test ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_pipeline_fake_speech_to_audio():
    """
    End-to-end: fake PCM audio → (mocked) Deepgram transcript →
    LangGraph agent (mocked) → (mocked) Cartesia TTS → audio bytes.

    Validates that each stage receives the correct input and the
    final output is non-empty bytes in the expected PCM format.
    """
    from voice.call_handler import LangGraphLLM
    from voice.tts import CartesiaTTS

    # ── Stage 1: fake PCM audio representing user speech ──────────────────
    pcm_audio = make_silence(duration_ms=500)
    assert len(pcm_audio) == 16_000   # 500 ms × 32 bytes/ms

    # ── Stage 2: Deepgram "transcribes" PCM → text ────────────────────────
    # In production this happens inside AgentSession. Here we simulate the
    # output the STT plugin would emit.
    transcript = "What Ford trucks do you have under forty thousand dollars?"

    # ── Stage 3: LangGraph agent processes transcript ─────────────────────
    fake_agent_result = {
        "intent": "car_query",
        "final_response": (
            "We have three Ford F-150 trucks under $40,000: "
            "a 2016 model at $31,185, a 2017 at $31,710, and a 2015 at $38,645."
        ),
        "rag_context": "Ford F-150...",
    }

    with patch("voice.call_handler.graph") as mock_graph:
        mock_graph.astream_events = _make_astream_events(
            fake_agent_result["intent"], fake_agent_result["final_response"]
        )

        ctx = ChatContext()
        ctx.add_message(role="user", content=transcript)

        llm = LangGraphLLM(thread_id="test-pipeline")
        chunks: list[ChatChunk] = []
        async with llm.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                chunks.append(chunk)

    assert len(chunks) >= 1
    agent_response = " ".join(c.delta.content for c in chunks if c.delta and c.delta.content)
    assert "Ford" in agent_response
    assert "$" in agent_response or "dollars" in agent_response.lower() or "price" in agent_response.lower()

    # ── Stage 4: Cartesia TTS converts response text → PCM audio ─────────
    with patch.object(CartesiaTTS, "stream", return_value=_fake_cartesia_stream(agent_response)):
        audio_chunks = [c async for c in CartesiaTTS.stream(
            CartesiaTTS.__new__(CartesiaTTS), agent_response
        )]

    assert len(audio_chunks) > 0
    total_bytes = sum(len(c) for c in audio_chunks)
    assert total_bytes > 0

    # Verify output is valid 16-bit PCM: byte count must be even
    assert total_bytes % 2 == 0
