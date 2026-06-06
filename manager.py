"""
manager.py — Guest memory and context management for the Himachal Homestay RAG pipeline.

Manages two memory layers:
  1. Long-term JSON memory  — guest profile, preferences, past trips
  2. ChromaDB vector memory — semantically indexed facts about the guest

Also handles conversation summarization to keep the context window manageable.

Adapted from the Feynman pipeline manager.py.
Changes:
  - Guest profile schema (replaces Feynman identity fields)
  - Booking-relevant preference fields (interests, budget, travel style)
  - Memory extraction now uses MEMORY_EXTRACTION_PROMPT from templates
  - summarize_conversation updated for travel booking context
"""

import os
import json
import hashlib
import time



from templates import MEMORY_EXTRACTION_PROMPT

MEMORY_FILE = "./memory/guest_memory.json"
GUEST_MEMORY_COLLECTION = "guest_memory"

# Memory extraction now uses local ChromaDB
# ---------------------------------------------------------------------------

# JSON LONG-TERM MEMORY
# ---------------------------------------------------------------------------

def load_memory() -> dict:
    """
    Loads long-term guest memory from disk.
    Migrates old formats automatically.
    """
    default_memory = {
        "profile": {
            "name": None,
            "contact": None,           # WhatsApp / email
        },
        "travel_preferences": {
            "budget_range": None,      # e.g. "budget", "mid-range", "luxury"
            "travel_style": None,      # e.g. "adventure", "leisure", "culture"
            "interests": [],           # e.g. ["trekking", "photography"]
            "preferred_regions": [],   # e.g. ["Spiti", "Manali"]
            "group_type": None,        # solo / couple / family / friends
        },
        "important_facts": [],         # Arbitrary notable facts
        "past_bookings": [],           # List of past booking ref IDs
        "conversation_summary": ""
    }

    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "profile" in data:
                # Already in new format — fill any missing keys
                for k, v in default_memory.items():
                    if k not in data:
                        data[k] = v
                return data

            # Migrate from old flat format
            migrated = default_memory.copy()
            migrated["profile"]["name"] = data.get("guest_name") or data.get("user_name")
            migrated["important_facts"] = data.get("important_facts", [])
            save_memory(migrated)
            return migrated

        except Exception:
            pass

    return default_memory


def save_memory(memory: dict):
    """Persists long-term guest memory to disk."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def update_memory_from_extraction(memory: dict, extracted: dict) -> dict:
    """
    Merges LLM-extracted facts into the memory structure.
    Called after each conversation turn.
    """
    if not extracted:
        return memory

    # Profile fields
    if extracted.get("guest_name"):
        memory["profile"]["name"] = extracted["guest_name"]
    if extracted.get("contact"):
        memory["profile"]["contact"] = extracted["contact"]

    # Travel preferences
    prefs = memory.setdefault("travel_preferences", {})
    if extracted.get("budget_range"):
        prefs["budget_range"] = extracted["budget_range"]
    if extracted.get("travel_style"):
        prefs["travel_style"] = extracted["travel_style"]
    if extracted.get("interests"):
        existing = set(prefs.get("interests", []))
        existing.update(extracted["interests"])
        prefs["interests"] = list(existing)
    if extracted.get("preferred_regions"):
        existing = set(prefs.get("preferred_regions", []))
        existing.update(extracted["preferred_regions"])
        prefs["preferred_regions"] = list(existing)
    if extracted.get("group_type"):
        prefs["group_type"] = extracted["group_type"]

    # Free-form important facts
    for fact_key in ["special_requirements", "dietary", "notes"]:
        if extracted.get(fact_key):
            fact_text = f"{fact_key}: {extracted[fact_key]}"
            if fact_text not in memory.get("important_facts", []):
                memory.setdefault("important_facts", []).append(fact_text)

    return memory


def format_memory_for_prompt(memory: dict, user_id: str = None, chroma_client = None, embedder = None) -> str:
    """
    Formats the guest memory dict into a readable string for injection
    into the system prompt. Recalls memories from ChromaDB if available.
    """
    lines = []
    
    # Recall memories from local ChromaDB
    if chroma_client and embedder:
        try:
            results = retrieve_guest_vector_memories(chroma_client, embedder, "What are the user's travel preferences, budget, group type, and previous inquiries?")
            if results:
                print(f"[{user_id}] [LOCAL MEMORY RECALL] Found {len(results)} memories")
                lines.append("--- Past Memories ---")
                for r in results:
                    lines.append(f"• {r}")
                    print(f"  - {r}")
                lines.append("--------------------------")
            else:
                print(f"[{user_id}] [LOCAL MEMORY RECALL] No memories found.")
        except Exception as e:
            print(f"[WARNING] Local memory recall failed: {e}")

    profile = memory.get("profile", {})

    if profile.get("name"):
        lines.append(f"Guest name: {profile['name']}")
    if profile.get("contact"):
        lines.append(f"Contact: {profile['contact']}")

    prefs = memory.get("travel_preferences", {})
    if prefs.get("budget_range"):
        lines.append(f"Budget range: {prefs['budget_range']}")
    if prefs.get("travel_style"):
        lines.append(f"Travel style: {prefs['travel_style']}")
    if prefs.get("interests"):
        lines.append(f"Interests: {', '.join(prefs['interests'])}")
    if prefs.get("preferred_regions"):
        lines.append(f"Interested in: {', '.join(prefs['preferred_regions'])}")
    if prefs.get("group_type"):
        lines.append(f"Group type: {prefs['group_type']}")

    for fact in memory.get("important_facts", []):
        lines.append(f"• {fact}")

    if memory.get("conversation_summary"):
        lines.append(f"\nConversation so far: {memory['conversation_summary']}")

    return "\n".join(lines) if lines else "No guest profile information yet."


# ---------------------------------------------------------------------------
# CHROMADB VECTOR MEMORY (short-term semantic facts)
# ---------------------------------------------------------------------------

def get_guest_memory_collection(chroma_client):
    """Gets or creates the guest_memory ChromaDB collection."""
    return chroma_client.get_or_create_collection(
        name=GUEST_MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def add_guest_vector_memory(chroma_client, embedder, memory_text: str):
    """
    Embeds and stores a guest memory fact in ChromaDB.
    Uses deterministic MD5 ID to avoid duplicates.
    """
    if not memory_text.strip():
        return

    collection = get_guest_memory_collection(chroma_client)
    cid = hashlib.md5(memory_text.encode('utf-8')).hexdigest()
    emb = embedder.encode([memory_text], normalize_embeddings=True).tolist()[0]

    try:
        existing = collection.get(ids=[cid], include=[])
        if not existing.get("ids"):
            collection.add(
                ids=[cid],
                embeddings=[emb],
                documents=[memory_text],
                metadatas=[{"created_at": float(time.time())}]
            )
    except Exception:
        pass


def retrieve_guest_vector_memories(chroma_client, embedder, user_query: str, top_k: int = 3) -> list[str]:
    """
    Retrieves relevant guest memory facts from ChromaDB based on query similarity.
    Cosine distance threshold: 0.65 (drops low-relevance memories).
    """
    collection = get_guest_memory_collection(chroma_client)
    if collection.count() == 0:
        return []

    query_text = f"Represent this sentence for searching relevant passages: {user_query}"
    query_embedding = embedder.encode([query_text], normalize_embeddings=True).tolist()[0]

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count())
        )

        memories = []
        if results["documents"] and results["distances"]:
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                if dist < 0.65:
                    memories.append(doc)
        return memories

    except Exception:
        return []


# ---------------------------------------------------------------------------
# MEMORY EXTRACTION (LLM-powered)
# ---------------------------------------------------------------------------

def extract_memory_from_turn(user_message: str, assistant_message: str,
                              existing_memory: dict, api_key: str, user_id: str = None, chroma_client = None, embedder = None) -> dict:
    """
    Calls Gemini Flash to extract new guest facts from the latest exchange.
    Returns a dict of extracted fields (may be empty if nothing new).
    Also retains facts in local ChromaDB if available.
    """
    existing_str = format_memory_for_prompt(existing_memory, user_id, chroma_client, embedder)
    prompt = MEMORY_EXTRACTION_PROMPT.format(
        existing_memory=existing_str,
        user_message=user_message,
        assistant_message=assistant_message
    )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()

        extracted = json.loads(raw)
        
        # Retain fact in local ChromaDB
        if chroma_client and embedder and extracted:
            try:
                fact_str = f"User shared the following facts: {json.dumps(extracted)}"
                print(f"[{user_id}] [LOCAL MEMORY RETAIN] Retaining: {fact_str}")
                add_guest_vector_memory(chroma_client, embedder, fact_str)
            except Exception as e:
                print(f"[WARNING] Local memory retain failed: {e}")

        return extracted if isinstance(extracted, dict) else {}

    except Exception as e:
        print(f"[WARNING] Memory extraction failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# CONVERSATION SUMMARIZATION
# ---------------------------------------------------------------------------

def summarize_conversation(messages: list[dict], api_key: str, current_summary: str = "") -> str:
    """
    Calls Gemini Flash to produce an updated conversation summary.
    Triggered when the context window grows beyond a threshold.
    Retains all booking-relevant context.
    """
    if not messages:
        return current_summary

    dialogue_text = ""
    for msg in messages:
        role = "Guest" if msg["role"] == "user" else "Deva (Assistant)"
        dialogue_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a conversation summarizer for a travel booking assistant.
Here is the existing summary:
"{current_summary}"

Here is the recent conversation:
---
{dialogue_text}
---

Write an updated, concise summary of the full conversation. Focus on:
- Guest's name and contact (if shared)
- Destination interest and travel dates
- Number of guests, group type, budget
- Itinerary preferences and interests
- Any booking details or confirmations
- Open questions or unresolved items

Return only the summary, no preamble.
"""

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARNING] Summarization failed: {e}")
        return current_summary
