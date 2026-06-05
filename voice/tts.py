from collections.abc import AsyncIterator

import structlog
from cartesia import AsyncCartesia
from cartesia.resources.tts import (
    AsyncTTSResourceConnection,
    AsyncTTSResourceConnectionManager,
)
from cartesia.types import RawOutputFormatParam, VoiceSpecifierParam
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger(__name__)

# PCM 16-bit signed little-endian @ 16 kHz — matches Deepgram STT and LiveKit native format
OUTPUT_FORMAT: RawOutputFormatParam = {
    "container": "raw",
    "encoding": "pcm_s16le",
    "sample_rate": 16000,
}


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    cartesia_api_key: str
    cartesia_model: str = "sonic-3"
    # Cartesia voice ID — default is "Helpful Woman" (neutral, clear)
    cartesia_voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"


class CartesiaTTS:
    """Streaming TTS via Cartesia Sonic WebSocket.

    Maintains a persistent WS connection for low-latency reuse across turns.
    Each stream() call creates a fresh context on the same connection.

    Usage:
        async with CartesiaTTS() as tts:
            async for chunk in tts.stream("Hello, how can I help?"):
                await websocket.send_bytes(chunk)
    """

    def __init__(self) -> None:
        settings = TTSSettings()
        self._client = AsyncCartesia(api_key=settings.cartesia_api_key)
        self._model = settings.cartesia_model
        self._voice_id = settings.cartesia_voice_id
        self._conn: AsyncTTSResourceConnection | None = None
        self._conn_mgr: AsyncTTSResourceConnectionManager | None = None

    async def _ensure_conn(self) -> AsyncTTSResourceConnection:
        """Lazy-init a persistent WebSocket connection; reconnect if closed."""
        if self._conn is None:
            self._conn_mgr = self._client.tts.websocket_connect()
            self._conn = await self._conn_mgr.__aenter__()
            log.info("tts.connected", model=self._model)
        return self._conn

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM audio chunks (16-bit signed LE, 16 kHz, mono) as they arrive.

        Do NOT buffer — pipe each chunk directly to the WebSocket for ~80ms TTFB.
        Creates a fresh context per call; safe to call concurrently on same instance.
        """
        voice: VoiceSpecifierParam = {"mode": "id", "id": self._voice_id}

        try:
            conn = await self._ensure_conn()
            ctx = conn.context(
                model_id=self._model,
                voice=voice,
                output_format=OUTPUT_FORMAT,
            )

            # continue_=False signals this is the complete utterance — no no_more_inputs() needed
            await ctx.send(
                transcript=text,
                voice=voice,
                continue_=False,
            )

            chunk_count = 0
            async for event in ctx.receive():
                # ctx.receive() auto-terminates on "done"/"error" events and cleans up
                # the context queue — do NOT break early on chunk.done
                if event.type == "chunk":
                    audio = event.audio  # decoded bytes from base64 Chunk.data
                    if audio:
                        chunk_count += 1
                        yield audio
                elif event.type == "error":
                    msg = getattr(event, "message", str(event))
                    log.error("tts.error", message=msg)
                    raise RuntimeError(f"Cartesia TTS error: {msg}")
                # "done" and "flush_done" handled by ctx.receive() — no action needed

            log.info("tts.stream_done", chunks=chunk_count, chars=len(text))

        except Exception:
            await self._reset_conn()
            raise

    async def _reset_conn(self) -> None:
        try:
            if self._conn_mgr is not None:
                await self._conn_mgr.__aexit__(None, None, None)
        except Exception:
            pass
        self._conn = None
        self._conn_mgr = None
        log.info("tts.reconnecting")

    async def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        await self._reset_conn()
        log.info("tts.closed")

    async def __aenter__(self) -> "CartesiaTTS":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
