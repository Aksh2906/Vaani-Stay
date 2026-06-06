import os
import io
import json
import base64
import asyncio
import threading
import edge_tts
from groq import AsyncGroq
from dotenv import load_dotenv
from pydub import AudioSegment

from templates import BOOKING_AGENT_SYSTEM_PROMPT

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: Please set GROQ_API_KEY in .env")
    exit(1)

client = AsyncGroq(api_key=GROQ_API_KEY)

# In-memory history for fast conversation flow
global_conversation_history = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Audio Conversion — Linear16 PCM at 8kHz (matches Exotel's WebSocket format)
# ═══════════════════════════════════════════════════════════════════════════════

def convert_mp3_to_pcm_8k(mp3_bytes: bytes) -> bytes:
    """Convert MP3 audio bytes → raw Linear16 PCM at 8kHz mono."""
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    return audio.raw_data


def make_linear16_media_frames(pcm_bytes: bytes, chunk_size: int = 1280) -> list[str]:
    """Split linear16 PCM into Exotel media frames (1280 bytes = 80ms at 8kHz)."""
    frames = []
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i + chunk_size]
        payload = base64.b64encode(chunk).decode("utf-8")
        frame = json.dumps({
            "event": "media",
            "media": {"payload": payload}
        })
        frames.append(frame)
    return frames


# ═══════════════════════════════════════════════════════════════════════════════
# Hindsight — Persistent background event loop (fixes "Event loop is closed")
# ═══════════════════════════════════════════════════════════════════════════════
# The Hindsight SDK caches its aiohttp session on the first event loop.
# If we create/close loops per-call, the cached session breaks.
# Solution: ONE persistent loop in a background thread, reused for all calls.

_hindsight_loop = None
_hindsight_thread = None


def _start_hindsight_loop():
    """Start a persistent background event loop for Hindsight operations."""
    global _hindsight_loop, _hindsight_thread
    if _hindsight_loop is not None:
        return

    _hindsight_loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(_hindsight_loop)
        _hindsight_loop.run_forever()

    _hindsight_thread = threading.Thread(target=_run, daemon=True)
    _hindsight_thread.start()
    print("  🧠 Hindsight background loop started")


def _hindsight_recall(hindsight_client, bank_id, query):
    """Submit a recall to the persistent Hindsight event loop."""
    _start_hindsight_loop()
    future = asyncio.run_coroutine_threadsafe(
        hindsight_client.arecall(bank_id=bank_id, query=query),
        _hindsight_loop
    )
    try:
        return future.result(timeout=10)
    except Exception as e:
        print(f"  Hindsight recall error ({bank_id}): {e}")
        return None


def _hindsight_retain(hindsight_client, bank_id, content):
    """Submit a retain to the persistent Hindsight event loop."""
    _start_hindsight_loop()
    future = asyncio.run_coroutine_threadsafe(
        hindsight_client.aretain(bank_id=bank_id, content=content),
        _hindsight_loop
    )
    try:
        future.result(timeout=10)
        print(f"  🧠 Memory retained for {bank_id}")
    except Exception as e:
        print(f"  ⚠️ Retain failed ({bank_id}): {e}")


def format_hindsight_memories(results, title="Past Memories") -> str:
    if not results or not isinstance(results, list):
        return "No memories found."
    lines = [f"--- {title} ---"]
    for r in results:
        if hasattr(r, 'content'):
            lines.append(f"• {r.content}")
        elif isinstance(r, str):
            lines.append(f"• {r}")
        elif isinstance(r, dict) and 'content' in r:
            lines.append(f"• {r['content']}")
    lines.append("--------------------------")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline — with barge-in support
# ═══════════════════════════════════════════════════════════════════════════════

async def stream_llm_to_exotel(
    transcript: str,
    phone: str,
    hindsight_client,
    websocket,
    tts_lock,
    text_to_audio_bytes,
    convert_mp3_to_mulaw_8k,  # from main.py (NOT used — we use linear16)
    make_exotel_media_frame,  # from main.py (NOT used — we use linear16)
    interrupt_event=None
):
    """
    Called by main.py after STT produces a final transcript.
    Flow: Hindsight recall → Stream LLM → Single TTS → Linear16 PCM → Exotel.
    """
    print(f"\n[STT Received]: {transcript}")

    # ── 1. Hindsight Recall (run on persistent background loop) ──
    kb_text = "No specific itinerary or property info found."
    user_memories = "No guest profile information yet."

    if hindsight_client:
        loop = asyncio.get_event_loop()
        kb_future = loop.run_in_executor(
            None, _hindsight_recall, hindsight_client, "vaani_knowledge_base", transcript
        )
        user_future = loop.run_in_executor(
            None, _hindsight_recall, hindsight_client, phone, transcript
        )
        kb_results, user_results = await asyncio.gather(kb_future, user_future)

        if kb_results:
            kb_text = format_hindsight_memories(kb_results, "Retrieved Knowledge")
        if user_results:
            user_memories = format_hindsight_memories(user_results, "Guest Profile & Memories")

    if interrupt_event and interrupt_event.is_set():
        print("🛑 Interrupted during Hindsight recall, aborting")
        return

    system_prompt = BOOKING_AGENT_SYSTEM_PROMPT.format(
        retrieved_context=kb_text,
        user_memories=user_memories,
        booking_state="IDLE"
    )

    # ── 2. Build conversation messages ──
    if phone not in global_conversation_history:
        global_conversation_history[phone] = []

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(global_conversation_history[phone][-10:])
    messages.append({"role": "user", "content": transcript})

    # ── 3. Stream LLM (for fast terminal display) ──
    stream = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=150,  # Cap output length for phone conversations
        stream=True
    )

    full_agent_response = ""
    print("[Agent]: ", end="", flush=True)
    async for chunk in stream:
        if interrupt_event and interrupt_event.is_set():
            print("\n🛑 Interrupted during LLM generation, aborting")
            return
        content = chunk.choices[0].delta.content
        if content:
            full_agent_response += content
            print(content, end="", flush=True)

    print()

    if not full_agent_response.strip():
        print("⚠️ LLM returned empty response, skipping TTS")
        return

    if interrupt_event and interrupt_event.is_set():
        print("🛑 Interrupted before TTS, aborting")
        global_conversation_history[phone].append({"role": "user", "content": transcript})
        global_conversation_history[phone].append({"role": "assistant", "content": full_agent_response})
        return

    # ── 4. Single TTS call → Linear16 PCM → Exotel ──
    async with tts_lock:
        print(f"[TTS] Synthesizing ({len(full_agent_response)} chars)...")

        mp3_bytes = await text_to_audio_bytes(full_agent_response)
        if not mp3_bytes:
            print("⚠️ TTS returned empty audio")
            return

        pcm_bytes = convert_mp3_to_pcm_8k(mp3_bytes)
        frames = make_linear16_media_frames(pcm_bytes)

        duration_secs = len(pcm_bytes) / (8000 * 2)
        print(f"[TTS] Streaming {len(frames)} frames ({duration_secs:.1f}s) to caller...")

        for frame in frames:
            if interrupt_event and interrupt_event.is_set():
                print("🛑 User interrupted! Stopping audio playback.")
                break

            try:
                await websocket.send_text(frame)
            except Exception:
                break
            await asyncio.sleep(0.08)

    print("[Finished Playback]")

    # ── 5. Update conversation history ──
    global_conversation_history[phone].append({"role": "user", "content": transcript})
    global_conversation_history[phone].append({"role": "assistant", "content": full_agent_response})

    # ── 6. Retain memory (non-blocking, on persistent loop) ──
    if hindsight_client:
        retention_str = f"Guest said: '{transcript}'\nAgent replied: '{full_agent_response}'"
        asyncio.get_event_loop().run_in_executor(
            None, _hindsight_retain, hindsight_client, phone, retention_str
        )
