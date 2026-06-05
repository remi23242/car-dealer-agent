# Car Dealer AI Voice Agent

AI-powered voice agent for car dealerships — answers inventory questions and books test drives via phone.

---

## Built With

![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Framework-FF6B35?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-00D4AA?style=flat-square)
![Deepgram](https://img.shields.io/badge/Deepgram-Nova--3_STT-1A73E8?style=flat-square)
![Cartesia](https://img.shields.io/badge/Cartesia-Sonic_TTS-9B59B6?style=flat-square)
![LiveKit](https://img.shields.io/badge/LiveKit-WebRTC-E74C3C?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-F39C12?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=flat-square&logoColor=black)
![Google Calendar](https://img.shields.io/badge/Google_Calendar-Appointments-4285F4?style=flat-square)
![Gmail](https://img.shields.io/badge/Gmail-Confirmations-EA4335?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)

---

## Features

- **Real-time voice calls** via LiveKit WebRTC — low-latency full-duplex audio
- **RAG-powered inventory search** — 11,000+ vehicles, semantic retrieval with HuggingFace + ChromaDB
- **Natural conversation** — intent routing across queries, bookings, greetings, and follow-ups
- **Google Calendar integration** — books test drive appointments with real-time slot availability
- **Google Meet links** — auto-generated and included in every booking confirmation
- **Gmail confirmations + reminders** — email sent on booking, reminder 30 minutes before appointment
- **Interruption handling** — voice pipeline handles mid-sentence interruptions gracefully
- **Streaming pipeline** — Deepgram interim results → Groq <500ms first token → Cartesia ~80ms TTS → ~1s end-to-end

---

## Project Structure

```
car-dealer-agent/
├── main.py                    # FastAPI app entry point
├── pyproject.toml             # dependencies (managed by uv)
├── .env.example               # environment variable template
│
├── agent/
│   ├── graph.py               # LangGraph graph definition
│   ├── state.py               # AgentState TypedDict
│   ├── router.py              # intent classification node (Groq LLM)
│   └── nodes/
│       ├── rag_node.py        # RAG retrieval + answer generation
│       ├── calendar_node.py   # 3-turn test drive booking state machine
│       ├── email_node.py      # email passthrough (booking handled in calendar_node)
│       └── response_node.py   # assembles final spoken response
│
├── voice/
│   ├── stt.py                 # Deepgram Nova-3 streaming STT
│   ├── tts.py                 # Cartesia Sonic streaming TTS
│   └── call_handler.py        # LiveKit room management + audio I/O
│
├── rag/
│   ├── ingest.py              # load car data, embed, store in Chroma
│   ├── retriever.py           # Chroma semantic search wrapper
│   └── data/                  # raw car inventory CSV (gitignored)
│
├── tools/
│   ├── calendar_tools.py      # Google Calendar API wrappers
│   └── email_tools.py         # Gmail API wrappers
│
├── scheduler/
│   └── reminder_jobs.py       # APScheduler reminder job definitions
│
├── scripts/
│   └── auth_google.py         # one-time Google OAuth2 setup
│
└── tests/
    ├── test_rag.py
    ├── test_calendar.py
    └── test_agent.py
```

---

## Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- Node.js 18+ (for LiveKit CLI, optional)
- A [Kaggle account](https://www.kaggle.com) to download the car dataset

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/car-dealer-agent.git
cd car-dealer-agent
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in all API keys (see [API Keys](#api-keys) below).

### 4. Get API keys

| Service | Where to get it |
|---|---|
| Groq | [console.groq.com](https://console.groq.com) → API Keys |
| Deepgram | [console.deepgram.com](https://console.deepgram.com) → API Keys |
| Cartesia | [play.cartesia.ai](https://play.cartesia.ai) → API Keys |
| LiveKit | [cloud.livekit.io](https://cloud.livekit.io) → Project Settings |
| Google Calendar + Gmail | See step 5 below |

### 5. Google OAuth setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → Enable **Google Calendar API** and **Gmail API**
3. Create **OAuth 2.0 credentials** (Desktop app) → download `credentials.json`
4. Place `credentials.json` in the project root
5. Run the auth script once:

```bash
uv run python scripts/auth_google.py
```

This opens a browser window for consent and saves `token.json`. Token auto-refreshes thereafter.

### 6. Download car inventory dataset

Download the [Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) from Kaggle and place the CSV in `rag/data/`.

### 7. Ingest into ChromaDB

```bash
uv run python -m rag.ingest
```

This embeds all vehicles with `BAAI/bge-small-en-v1.5` and stores them in `rag/chroma_db/`.

---

## API Keys

| Service | Purpose | Free tier? |
|---|---|---|
| [Groq](https://console.groq.com) | LLM inference (Llama 3.3 70B) | Yes — generous rate limits |
| [Deepgram](https://console.deepgram.com) | Speech-to-text (Nova-3) | Yes — $200 credit on signup |
| [Cartesia](https://play.cartesia.ai) | Text-to-speech (Sonic) | Yes — trial credits |
| [LiveKit](https://cloud.livekit.io) | WebRTC voice transport | Yes — free cloud tier |
| [Google Calendar](https://console.cloud.google.com) | Appointment booking | Yes — standard OAuth quota |
| [Gmail](https://console.cloud.google.com) | Confirmation + reminder emails | Yes — standard OAuth quota |
| [HuggingFace](https://huggingface.co) | Embeddings (`bge-small-en-v1.5`) | Yes — runs locally, no API key |

---

## Running Locally

```bash
# Start the FastAPI server
uv run uvicorn main:app --reload --port 8000

# Start the LiveKit voice worker (separate terminal)
uv run python voice/call_handler.py dev
```

The server exposes:
- `GET /` — health check
- `POST /chat` — text-based agent (testing without voice)
- `WebSocket /ws/voice` — voice session endpoint

---

## How It Works

```
Caller
  │
  ▼
LiveKit WebRTC ──► Deepgram Nova-3 (STT, streaming)
                         │
                         ▼
                   LangGraph Router
                   (intent classification via Groq)
                         │
               ┌─────────┴──────────┐
               ▼                    ▼
          RAG Node             Calendar Node
     (ChromaDB search)      (3-turn booking flow)
      HuggingFace embed      Google Calendar API
                             Gmail confirmation
               └─────────┬──────────┘
                          ▼
                   Response Node
                   (assemble answer)
                          │
                          ▼
                  Cartesia Sonic (TTS, streaming)
                          │
                          ▼
                       Caller
```

**Latency targets:** STT interim results at 100ms chunks → Groq first token <500ms → Cartesia first audio byte ~80ms → total ~1s end-to-end.

**Booking flow (3 turns):**
1. "Book a test drive" → fetch live calendar slots → offer choices
2. User picks slot → confirm selection → ask for name
3. User gives name → create Calendar event + Meet link → send Gmail confirmation → schedule 30-min reminder

---

## License

MIT
