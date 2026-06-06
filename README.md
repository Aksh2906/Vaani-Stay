# VaaniStay — AI Voice Booking Agent for Himachal Pradesh Homestays

VaaniStay is a real-time AI voice agent that handles phone-based booking enquiries for homestays in Himachal Pradesh. It receives live phone calls via Exotel, transcribes speech using Deepgram, generates contextual responses using Groq (Llama 3.3 70B), and speaks back to the caller in Hindi using Edge TTS — all in real time over a single WebSocket connection.

## Architecture

```
Caller (Phone)
    |
Exotel (WebSocket)
    |
main.py ── Deepgram STT (Hindi, Nova-2)
    |
stream_tts.py ── Hindsight Cloud (RAG + Memory) --> Groq LLM --> Edge TTS
    |
Audio frames streamed back to caller via Exotel WebSocket
    |
On call end: Gemini extracts booking --> Firebase Firestore
```

## Core Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI WebSocket server. Receives Exotel audio, pipes to Deepgram for STT, dispatches transcripts to the LLM pipeline, and handles post-call booking extraction via Gemini. |
| `stream_tts.py` | LLM + TTS pipeline. Recalls context from Hindsight Cloud, streams Groq responses, synthesises Hindi audio via Edge TTS, and sends Linear16 PCM frames back to the caller. Supports barge-in (user interruption). |
| `templates.py` | System prompts for the booking agent, query rewriter, and memory extraction. |
| `firebase_functions.py` | Writes extracted booking data and notifications to Firestore after each call. |
| `api.py` | Standalone REST API (optional). Exposes `/receive_transcript` and `/get_agent_response` endpoints for non-telephony integrations. |
| `ingest_hindsight_cloud.py` | One-time script to upload property data from `data/` into the Hindsight Cloud knowledge base. |

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- ffmpeg/ffplay installed and on PATH
- [ngrok](https://ngrok.com/) for exposing local server to Exotel

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_...
HINDSIGHT_API_KEY=hsk_...
Deepgram_API_KEY=...
Google_Gimini_API_KEY=...       # Used only for post-call booking extraction
```

## Firebase Setup

1. Download your Firebase service account JSON from Firebase Console > Project Settings > Service Accounts.
2. Place it in the project root as `vaani-stay-firebase-adminsdk-fbsvc-*.json`.
3. The path is configured in `firebase_functions.py` line 15.

## Installation

```bash
uv pip install fastapi uvicorn websockets httpx edge-tts groq pydub python-dotenv google-generativeai firebase-admin hindsight
```

## Running

### 1. Ingest knowledge base (one-time)

```bash
uv run python ingest_hindsight_cloud.py
```

This uploads all files from `data/` into the `vaani_knowledge_base` bank on Hindsight Cloud.

### 2. Start the voice server

```bash
uvicorn main:app --port 8080
```

### 3. Expose to the internet

```bash
ngrok http 8080
```

Copy the ngrok URL and configure your Exotel call flow to connect to:

```
wss://<ngrok-subdomain>.ngrok.app/stream
```

### 4. Test

Call the Exotel number linked to the above WebSocket URL. The agent will answer in Hindi/Hinglish, retrieve relevant homestay information from Hindsight Cloud, and guide the caller through a booking.

## Key Design Decisions

**Transcript debouncing**: Deepgram often splits a single utterance into multiple `is_final` events. A 1.2-second debounce window merges these into one transcript before sending to the LLM, preventing half-sentence confusion.

**Barge-in support**: If the caller speaks while the agent is responding, the current audio playback is immediately cancelled and the new transcript is processed.

**Linear16 PCM encoding**: Exotel WebSocket uses Linear16 at 8kHz. The TTS pipeline converts Edge TTS MP3 output to raw PCM via pydub, avoiding the mulaw encoding mismatch that causes audio artefacts.

**Persistent Hindsight event loop**: The Hindsight SDK caches its HTTP session on the first event loop. A single persistent background thread with its own event loop handles all Hindsight operations to avoid "Event loop is closed" errors.

**Short responses**: The system prompt and `max_tokens=150` enforce concise, phone-friendly responses (under 2 sentences). Long AI-generated paragraphs are unusable over a phone call.

## Data Directory

The `data/` folder contains property descriptions, itineraries, FAQs, and pricing information in text format. These are ingested into Hindsight Cloud and retrieved at query time.
