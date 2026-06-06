# Himachal Pradesh Homestay — RAG Pipeline

Adapted from the Feynman Digital Twin RAG pipeline.
This module powers the AI booking agent's retrieval layer.

---

## Architecture

```
data/
  itineraries/   ← 45 JSON/PDF/TXT itineraries
  properties/    ← Homestay room & property descriptions
  faqs/          ← Common guest FAQs
  policies/      ← Cancellation, house rules, booking policies

       ↓ loader.py
   Page records (text + metadata)

       ↓ chunker.py
   Sentence chunks with sliding window context (window=3)

       ↓ embedder.py
   BGE-small-en-v1.5 embeddings → ChromaDB

                    At query time:
                         ↓
       User message → retriever.py (query rewriter → Gemini Flash)
                         ↓
              Hybrid Retrieval (Dense + BM25)
                         ↓
              Score Fusion (60% dense + 40% BM25)
              + Source Authority + Domain Boost
                         ↓
              BGE Cross-Encoder Reranking
                         ↓
              Top-5 context windows → System Prompt
                         ↓
              Gemini Flash → Booking Agent Response
                         ↓
              manager.py → Memory extraction + update
              booking_state.py → Booking state tracking
```

---

## Changes from Feynman Pipeline

| Component | Feynman | Himachal Homestay |
|---|---|---|
| `loader.py` | PDF + TXT | PDF + TXT + **JSON itineraries** (primary) |
| `chunker.py` | Generic sentences | Same + **preserves structured itinerary lines** (Day N:, Accommodation:) |
| `helpers.py` | 5 physics domains | **7 travel domains** (destination, accommodation, activities, pricing, logistics, food, policies) |
| `templates.py` | Feynman persona prompt | **Deva booking agent prompt** + booking state injection + memory extraction prompt |
| `manager.py` | Feynman identity memory | **Guest travel preference memory** (budget, interests, group type, contact) |
| `retriever.py` | Same architecture | + **booking intent detection** → domain boost for pricing/accommodation chunks |
| `embedder.py` | Feynman source authority | **Travel doc type authority** (itinerary=1.0, property=0.95, faq=0.90, policy=0.85) |
| `booking_state.py` | ❌ (did not exist) | ✅ **NEW** — full booking lifecycle (idle → collecting → confirming → confirmed) |

---

## Itinerary JSON Format

Place your 45 itineraries in `data/itineraries/` as JSON files:

```json
{
  "title": "Spiti Valley Explorer — 8D/7N",
  "duration_days": 8,
  "region": "Spiti",
  "price_per_person": 18000,
  "highlights": ["Tabo Monastery", "Key Monastery", "Chandratal Lake"],
  "inclusions": ["Accommodation", "Meals (MAP)", "Inner Line Permit", "Cab"],
  "exclusions": ["Flights", "Personal expenses"],
  "notes": "Inner Line Permit required. Road opens May–October.",
  "days": [
    {
      "day": 1,
      "title": "Delhi → Shimla",
      "description": "Overnight Volvo bus from Delhi to Shimla.",
      "activities": ["Travel"],
      "accommodation": "Shimla Homestay, The Mall",
      "meals": "Dinner included"
    }
  ]
}
```

---

## Directory Structure

```
src/
  pipeline/
    loader.py          ← Document loading (PDF, TXT, JSON)
    chunker.py         ← Sentence chunker with sliding window
    embedder.py        ← BGE embedding + ChromaDB ingestion
    retriever.py       ← Hybrid retriever (Dense + BM25 + Reranker)
    manager.py         ← Guest memory management
    booking_state.py   ← Booking session state tracker (NEW)
    clients.py         ← BGE model loaders (cached)
  utils/
    helpers.py         ← Domain classification (7 travel domains)
  prompts/
    templates.py       ← System prompts + query rewriter + memory extractor
memory/
  guest_memory.json    ← Persistent guest profile
bookings/
  confirmed_bookings.json  ← Confirmed booking log
```

---

## Booking State Lifecycle

```
IDLE → COLLECTING → CONFIRMING → CONFIRMED
```

- **IDLE**: No booking in progress
- **COLLECTING**: Agent is gathering dates, guests, room preference, contact
- **CONFIRMING**: All fields collected, summary shown to guest
- **CONFIRMED**: Guest confirmed → Reference ID generated (HP-YYYYMMDD-XXXX)

---

## Integration Notes for the Team

- The RAG pipeline is completely decoupled from the Streamlit UI — import and call `HybridRetriever.retrieve()` from anywhere.
- `BookingStateManager` is session-scoped — instantiate once per conversation, store in `st.session_state`.
- `manager.py` handles both JSON long-term memory and ChromaDB vector memory — call `extract_memory_from_turn()` after every agent response.
- The query rewriter costs one Gemini API call per user turn. If rate limits are a concern, disable it for single-turn queries (when `chat_history` is empty, the rewriter already skips).
