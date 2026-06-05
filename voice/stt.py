import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator

import structlog
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    deepgram_api_key: str
    deepgram_model: str = "nova-2"
    deepgram_language: str = "en-US"
    deepgram_sample_rate: int = 16_000


@dataclass
class TranscriptResult:
    transcript: str
    is_final: bool
    speech_final: bool = False


class DeepgramSTT:
    """Persistent Deepgram live-transcription WebSocket.

    Usage:
        stt = DeepgramSTT()
        await stt.connect()
        await stt.stream(audio_bytes)          # non-blocking send
        async for result in stt.receive():     # yields TranscriptResult
            if result.is_final:
                ...
        await stt.disconnect()
    """

    def __init__(self) -> None:
        settings = STTSettings()
        self._settings = settings
        self._client = DeepgramClient(
            settings.deepgram_api_key,
            config=DeepgramClientOptions(options={"keepalive": "true"}),
        )
        self._connection = None
        self._queue: asyncio.Queue[TranscriptResult] = asyncio.Queue()
        self._connected = False

    async def connect(self) -> None:
        """Open persistent Deepgram WebSocket. Must be called before stream()."""
        connection = self._client.listen.asynclive.v("1")
        self._connection = connection
        queue = self._queue

        async def on_open(_conn, _event, **kwargs) -> None:
            logger.info("deepgram_connected", model=self._settings.deepgram_model)

        async def on_transcript(_conn, result, **kwargs) -> None:
            try:
                alt = result.channel.alternatives[0]
                transcript = alt.transcript
                if not transcript:
                    return
                await queue.put(
                    TranscriptResult(
                        transcript=transcript,
                        is_final=result.is_final,
                        speech_final=getattr(result, "speech_final", False),
                    )
                )
            except Exception as exc:
                logger.warning("transcript_parse_error", error=str(exc))

        async def on_utterance_end(_conn, _event, **kwargs) -> None:
            logger.debug("utterance_end")

        async def on_error(_conn, error, **kwargs) -> None:
            logger.error("deepgram_error", error=str(error))

        async def on_close(_conn, _event, **kwargs) -> None:
            self._connected = False
            logger.info("deepgram_socket_closed")

        connection.on(LiveTranscriptionEvents.Open, on_open)
        connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        connection.on(LiveTranscriptionEvents.Error, on_error)
        connection.on(LiveTranscriptionEvents.Close, on_close)

        options = LiveOptions(
            model=self._settings.deepgram_model,
            language=self._settings.deepgram_language,
            sample_rate=self._settings.deepgram_sample_rate,
            encoding="linear16",
            channels=1,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            endpointing=150,
            punctuate=True,
            smart_format=True,
        )

        started = await connection.start(options)
        if not started:
            raise RuntimeError("Deepgram connection failed to start")
        self._connected = True

    async def disconnect(self) -> None:
        """Close the WebSocket. Safe to call even if already disconnected."""
        if self._connection and self._connected:
            await self._connection.finish()
            self._connected = False

    async def stream(self, audio_chunk: bytes) -> TranscriptResult | None:
        """Send 100ms PCM chunk. Returns any immediately queued transcript or None.

        Transcripts arrive async via Deepgram callbacks. Use receive() to consume
        them as an async generator; this method returns None most of the time.
        """
        if not self._connected or self._connection is None:
            raise RuntimeError("Not connected — call connect() first")
        await self._connection.send(audio_chunk)
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def receive(self) -> AsyncGenerator[TranscriptResult, None]:
        """Async generator — yields every TranscriptResult as it arrives.

        Runs until disconnect() is called. Use alongside stream() in a parallel
        task to consume transcripts without blocking audio send.
        """
        while self._connected:
            try:
                result = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                yield result
            except asyncio.TimeoutError:
                continue

    async def __aenter__(self) -> "DeepgramSTT":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()
