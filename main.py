# ═══════════════════════════════════════════════════════════════════════════════
# VaaniStay — Full AI Voice Booking Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
# Exotel → Deepgram STT → RAG (api.py) → edge_tts → Exotel (bidirectional)
# On call end → Gemini extraction → Firestore → Dashboard updates
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import websockets
import json
import base64
import io
import os
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import httpx
import edge_tts
import google.generativeai as genai

# audioop removed in Python 3.13+, use audioop-lts fallback
try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from pydub import AudioSegment

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

DEEPGRAM_API_KEY = os.getenv("Deepgram_API_KEY", "")
GEMINI_API_KEY = os.getenv("Google_Gimini_API_KEY", "")
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY", "")

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8080")
TTS_VOICE = "hi-IN-MadhurNeural"
SILENT_FRAME = b"\x00" * 1600  # 100ms of silence in linear16 8kHz

app = FastAPI(title="VaaniStay Voice Pipeline")

# ── Gemini setup ──
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-3.1-flash-lite")
    print("🤖 Gemini configured for booking extraction")
else:
    gemini_model = None
    print("⚠️ No Gemini API key — booking extraction disabled")

# ── Firebase setup ──
try:
    from firebase_functions import process_completed_call, db as firebase_db
    FIREBASE_ENABLED = firebase_db is not None
    if FIREBASE_ENABLED:
        print("🔥 Firebase integration ready")
    else:
        print("⚠️ Firebase not initialized — dashboard won't update")
except ImportError:
    FIREBASE_ENABLED = False
    print("⚠️ firebase_functions.py not found")

# ── Hindsight setup (for end-of-call memory retention) ──
hindsight_client = None
try:
    if HINDSIGHT_API_KEY:
        from hindsight import HindsightClient
        hindsight_client = HindsightClient(
            api_key=HINDSIGHT_API_KEY,
            base_url="https://api.hindsight.vectorize.io"
        )
        print("🧠 Hindsight client ready for memory retention")
except Exception as e:
    print(f"⚠️ Hindsight not initialized: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Audio Conversion Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def convert_mp3_to_mulaw_8k(mp3_bytes: bytes) -> bytes:
    """Convert MP3 audio bytes → G.711 μ-law at 8kHz mono (Exotel format)."""
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm = audio.raw_data
    return audioop.lin2ulaw(pcm, 2)


def make_exotel_media_frame(mulaw_bytes: bytes, chunk_size: int = 640) -> list[str]:
    """
    Split μ-law audio into Exotel-compatible media frames.
    Each frame is ~80ms of audio at 8kHz (640 bytes).
    Returns list of JSON strings ready to send.
    """
    frames = []
    for i in range(0, len(mulaw_bytes), chunk_size):
        chunk = mulaw_bytes[i:i + chunk_size]
        payload = base64.b64encode(chunk).decode("utf-8")
        frame = json.dumps({
            "event": "media",
            "media": {"payload": payload}
        })
        frames.append(frame)
    return frames


# ═══════════════════════════════════════════════════════════════════════════════
# TTS: Text → Audio bytes (edge_tts)
# ═══════════════════════════════════════════════════════════════════════════════

async def text_to_audio_bytes(text: str) -> bytes:
    """Convert text to MP3 bytes using edge_tts."""
    mp3_buffer = io.BytesIO()
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
    return mp3_buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# RAG: Call api.py for agent response
# ═══════════════════════════════════════════════════════════════════════════════

async def call_rag_api(transcript: str, caller_phone: str) -> str | None:
    """
    POST transcript to api.py → poll for response.
    Returns agent response text, or None on failure.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Step 1: Send transcript
        try:
            resp = await client.post(
                f"{RAG_API_URL}/receive_transcript",
                json={"user_id": caller_phone, "transcript": transcript}
            )
            if resp.status_code != 200:
                print(f"⚠️ RAG POST failed: {resp.status_code}")
                return None
        except Exception as e:
            print(f"⚠️ RAG POST error: {e}")
            return None

        # Step 2: Poll for response (api.py processes synchronously,
        # so response should be ready after POST returns)
        for attempt in range(5):
            try:
                r = await client.get(
                    f"{RAG_API_URL}/get_agent_response/{caller_phone}"
                )
                if r.status_code == 200:
                    agent_text = r.json().get("response", "")
                    if agent_text:
                        return agent_text
            except Exception:
                pass
            await asyncio.sleep(0.3)

    print(f"⚠️ No RAG response after 5 attempts for {caller_phone}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini: Extract booking from full transcript
# ═══════════════════════════════════════════════════════════════════════════════

BOOKING_EXTRACTION_PROMPT = """You are a booking information extractor for a homestay called "Hilltop Haven Homestay".

Analyze this phone call transcript between a guest and the AI receptionist. Extract booking details if any were discussed.

TRANSCRIPT:
{transcript}

Return a JSON object with these fields. Use null for any field that wasn't mentioned:
{{
  "guestName": "string or null",
  "guestPhone": "string or null",
  "checkIn": "DD/MM/YYYY or null",
  "checkOut": "DD/MM/YYYY or null",
  "roomType": "Deluxe/Suite/Cottage/Standard or null",
  "numGuests": number or null,
  "pricePerNight": number or null,
  "totalPrice": number or null,
  "specialRequests": "string or null",
  "status": "confirmed" or "enquiry",
  "summary": "one-line summary of the call"
}}

Rules:
- If the guest confirmed a booking, status = "confirmed"
- If they just asked questions without booking, status = "enquiry"
- If dates are relative ("kal", "next week"), calculate from today's date ({today})
- Return ONLY the JSON, no markdown, no extra text
"""


def extract_booking_from_transcript(transcript: str, caller_phone: str = "") -> dict | None:
    """Use Gemini to extract structured booking data from a call transcript."""
    if not gemini_model or not transcript.strip():
        return None

    try:
        prompt = BOOKING_EXTRACTION_PROMPT.format(
            transcript=transcript,
            today=datetime.now().strftime("%d/%m/%Y")
        )
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()

        # Clean markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        booking = json.loads(text)

        # Fill in caller phone if not extracted
        if not booking.get("guestPhone") and caller_phone:
            booking["guestPhone"] = caller_phone

        # Defaults for missing fields
        booking.setdefault("guestName", "Unknown Guest")
        booking.setdefault("roomType", "Standard")
        booking.setdefault("numGuests", 1)
        booking.setdefault("pricePerNight", 3000)
        booking.setdefault("totalPrice", booking.get("pricePerNight", 3000))
        booking.setdefault("specialRequests", "")
        booking.setdefault("status", "enquiry")

        if not booking.get("checkIn"):
            booking["checkIn"] = datetime.now().strftime("%d/%m/%Y")
        if not booking.get("checkOut"):
            booking["checkOut"] = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")

        print(f"🤖 Gemini extracted: {booking.get('guestName')} | {booking.get('status')} | {booking.get('roomType')}")
        return booking

    except Exception as e:
        print(f"❌ Gemini extraction failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def health(request: Request):
    return PlainTextResponse("VaaniStay Voice Pipeline running ✅")


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket /stream — The Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    print("📞 WebSocket connected")

    # ── Per-call state ──
    full_transcript_parts: list[str] = []
    caller_phone = ""
    call_start_time = asyncio.get_event_loop().time()
    tts_lock = asyncio.Lock()  # Prevents overlapping TTS responses
    interrupt_event = asyncio.Event()  # Set when user speaks to cancel current TTS
    current_tts_task = None  # Track the active TTS task

    # ── Deepgram connection ──
    dg_url = (
        "wss://api.deepgram.com/v1/listen"
        "?encoding=linear16"
        "&sample_rate=8000"
        "&model=nova-2"
        "&language=hi"
        "&punctuate=true"
        "&smart_format=true"
        "&interim_results=true"
        "&endpointing=300"
        "&utterance_end_ms=1000"
        "&filler_words=false"
    )

    try:
        dg_ws = await websockets.connect(
            dg_url,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            open_timeout=10,
            ping_interval=20,
            ping_timeout=60,
        )
    except Exception as e:
        print(f"❌ Deepgram connect failed: {e}")
        await websocket.close()
        return

    print("🎙️ Deepgram connected")
    await dg_ws.send(SILENT_FRAME)

    # ── RAG + TTS handler (spawned as task per final transcript) ──
    async def handle_rag_and_tts(transcript: str, phone: str):
        """Called per final sentence: RAG → TTS → stream audio back."""
        try:
            from stream_tts import stream_llm_to_exotel
            await stream_llm_to_exotel(
                transcript=transcript,
                phone=phone,
                hindsight_client=hindsight_client,
                websocket=websocket,
                tts_lock=tts_lock,
                text_to_audio_bytes=text_to_audio_bytes,
                convert_mp3_to_mulaw_8k=convert_mp3_to_mulaw_8k,
                make_exotel_media_frame=make_exotel_media_frame,
                interrupt_event=interrupt_event
            )
        except Exception as e:
            print(f"❌ RAG+TTS stream error: {e}")

    # ── Keep-alive for Deepgram ──
    async def keep_alive():
        try:
            while True:
                await asyncio.sleep(5)
                await dg_ws.send(SILENT_FRAME)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # ── Receive audio from Exotel → forward to Deepgram ──
    async def receive_from_exotel():
        nonlocal caller_phone
        try:
            async for raw in websocket.iter_text():
                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event = frame.get("event", "")

                if event == "connected":
                    print("✅ Exotel stream connected")

                elif event == "start":
                    info = frame.get("start", {})
                    caller_phone = info.get("from", "")
                    print(f"🔊 Stream started — call from: {caller_phone}")

                elif event == "media":
                    audio_b64 = frame["media"]["payload"]
                    audio_bytes = base64.b64decode(audio_b64)
                    await dg_ws.send(audio_bytes)

                elif event == "stop":
                    reason = frame.get("stop", {}).get("reason", "unknown")
                    print(f"🔴 Stream stopped: {reason}")
                    break

        except Exception as e:
            print(f"Exotel error: {e}")
        finally:
            await dg_ws.close()

    # ── Receive transcripts from Deepgram (with debounce) ──
    pending_transcript = []  # Accumulates rapid-fire final transcripts
    debounce_task = None  # Timer task for debouncing

    async def _process_debounced():
        """Called after debounce window expires — sends merged transcript to LLM."""
        nonlocal current_tts_task, pending_transcript
        await asyncio.sleep(1.2)  # Wait 1.2s for more finals to arrive

        if not pending_transcript:
            return

        merged = " ".join(pending_transcript)
        pending_transcript.clear()

        # Barge-in: cancel current TTS if playing
        if current_tts_task and not current_tts_task.done():
            print("🛑 User interrupted — cancelling current response")
            interrupt_event.set()
            await asyncio.sleep(0.15)

        interrupt_event.clear()

        print(f"[MERGED TRANSCRIPT]: {merged}")
        current_tts_task = asyncio.create_task(
            handle_rag_and_tts(merged, caller_phone)
        )

    async def receive_from_deepgram():
        nonlocal current_tts_task, debounce_task
        try:
            async for msg in dg_ws:
                data = json.loads(msg)

                if data.get("type") not in ("Results", None):
                    continue

                channel = data.get("channel", {})
                alternatives = channel.get("alternatives", [])

                if not alternatives:
                    continue

                transcript = alternatives[0].get("transcript", "")

                if transcript:
                    is_final = data.get("is_final", False)
                    tag = "TRANSCRIPT ✅" if is_final else "interim   ..."
                    print(f"[{tag}] {transcript}")

                    if is_final:
                        full_transcript_parts.append(transcript)
                        pending_transcript.append(transcript)

                        # Cancel previous debounce timer (reset the window)
                        if debounce_task and not debounce_task.done():
                            debounce_task.cancel()

                        # Start new debounce timer
                        debounce_task = asyncio.create_task(_process_debounced())

        except Exception as e:
            print(f"Deepgram error: {e}")

    # ── Run all concurrent tasks ──
    ka_task = asyncio.create_task(keep_alive())

    await asyncio.gather(
        receive_from_exotel(),
        receive_from_deepgram()
    )

    ka_task.cancel()

    # ═════════════════════════════════════════════════════════════════════════
    # CALL ENDED — Post-processing pipeline
    # ═════════════════════════════════════════════════════════════════════════
    call_duration = int(asyncio.get_event_loop().time() - call_start_time)
    full_transcript = " ".join(full_transcript_parts)
    print(f"📴 Call ended | Duration: {call_duration}s | Transcript: {len(full_transcript)} chars")

    if not full_transcript.strip():
        print("ℹ️ No transcript captured — skipping post-processing")
        return

    # ── Step 1: Extract booking with Gemini ──
    print("🔄 Extracting booking with Gemini...")
    booking_data = extract_booking_from_transcript(full_transcript, caller_phone)

    if booking_data:
        booking_data["transcript"] = full_transcript
        booking_data["callDuration"] = call_duration
        booking_data["guestPhone"] = caller_phone or booking_data.get("guestPhone", "")
        booking_data["callerPhone"] = caller_phone

        # ── Step 2: Write to Firestore ──
        if FIREBASE_ENABLED:
            print("🔥 Writing to Firestore...")
            process_completed_call(booking_data)
            print("📱 Dashboard will update in ~2 seconds!")
        else:
            print("⚠️ Firebase not enabled — booking not saved")
            print(f"   Extracted: {json.dumps(booking_data, indent=2, default=str)}")
    else:
        print("ℹ️ No booking data extracted from this call")

    # ── Step 3: Retain full transcript to Hindsight user memory ──
    if hindsight_client and caller_phone:
        try:
            retention = (
                f"Full call transcript ({datetime.now().strftime('%d/%m/%Y %H:%M')}):\n"
                f"{full_transcript}"
            )
            hindsight_client.retain(bank_id=caller_phone, content=retention)
            print(f"🧠 Transcript retained to Hindsight for {caller_phone}")
        except Exception as e:
            print(f"⚠️ Hindsight retention failed: {e}")