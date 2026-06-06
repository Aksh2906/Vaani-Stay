"""
loader.py — Document loader for Himachal Pradesh Homestay RAG pipeline.

Supports:
  - PDF itineraries / brochures / travel guides
  - TXT itineraries
  - JSON itineraries (primary format for structured 45-itinerary corpus)
  - Room/property description files

Each loaded page/record is returned as:
{
    "text": str,
    "metadata": {
        "source": str,
        "page": int,
        "title": str,
        "document_type": str,   # "itinerary" | "property" | "faq" | "policy"
        "region": str,          # e.g. "Manali", "Spiti", "Shimla" (auto-detected)
        "duration_days": int    # extracted from itinerary if available
    }
}
"""

import os
import json
import re
import fitz  # PyMuPDF

# Known Himachal regions for auto-tagging
HIMACHAL_REGIONS = [
    "manali", "shimla", "spiti", "lahaul", "kasol", "kufri", "dharamshala",
    "mcleod ganj", "dalhousie", "chamba", "kullu", "mandi", "kinnaur",
    "sarahan", "chitkul", "kalpa", "nako", "tabo", "kaza", "pin valley",
    "bir billing", "palampur", "barot", "jibhi", "tirthan", "great himalayan",
    "rohtang", "solang", "triund", "khajjiar", "chail", "naldehra", "rampur"
]


def detect_region(text: str) -> str:
    """Auto-detects the primary Himachal region mentioned in a text block."""
    text_lower = text.lower()
    for region in HIMACHAL_REGIONS:
        if region in text_lower:
            # Capitalize properly
            return region.title()
    return "General Himachal"


def extract_duration(text: str) -> int:
    """Tries to extract trip duration in days from itinerary text (e.g. '7 Days / 6 Nights')."""
    match = re.search(r'(\d+)\s*(?:day|days|night|nights)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def load_pdf(file_path: str, doc_type: str) -> list[dict]:
    """Loads a PDF file page by page."""
    pages_data = []
    source_name = os.path.basename(file_path)

    try:
        doc = fitz.open(file_path)
        title = doc.metadata.get("title") or source_name

        for idx, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue

            pages_data.append({
                "text": text,
                "metadata": {
                    "source": source_name,
                    "page": idx + 1,
                    "title": title,
                    "document_type": doc_type,
                    "region": detect_region(text),
                    "duration_days": extract_duration(text)
                }
            })
    except Exception as e:
        print(f"[ERROR] Failed to load PDF {file_path}: {e}")

    return pages_data


def load_txt(file_path: str, doc_type: str) -> list[dict]:
    """Loads a plain text itinerary or property description."""
    source_name = os.path.basename(file_path)
    pages_data = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        if not text.strip():
            return pages_data

        # Split long TXT files into ~1000-word chunks to approximate pages
        words = text.split()
        chunk_size = 1000
        for chunk_idx in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[chunk_idx:chunk_idx + chunk_size])
            pages_data.append({
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "page": (chunk_idx // chunk_size) + 1,
                    "title": source_name.replace("_", " ").replace(".txt", "").title(),
                    "document_type": doc_type,
                    "region": detect_region(chunk_text),
                    "duration_days": extract_duration(chunk_text)
                }
            })
    except Exception as e:
        print(f"[ERROR] Failed to load TXT {file_path}: {e}")

    return pages_data


def load_json_itinerary(file_path: str) -> list[dict]:
    """
    Loads a structured JSON itinerary.

    Expected schema (flexible — adapt to your actual JSON structure):
    {
        "title": "Manali Adventure 7D/6N",
        "duration_days": 7,
        "region": "Manali",
        "days": [
            {
                "day": 1,
                "title": "Arrival in Manali",
                "description": "...",
                "activities": [...],
                "accommodation": "Himalayan Homestay, Old Manali"
            },
            ...
        ],
        "highlights": [...],
        "inclusions": [...],
        "exclusions": [...],
        "price_per_person": 12000,
        "notes": "..."
    }

    Each day becomes one page-level record so chunking is fine-grained.
    """
    pages_data = []
    source_name = os.path.basename(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        title = data.get("title", source_name)
        duration = data.get("duration_days", extract_duration(title))
        region = data.get("region", detect_region(title))
        price = data.get("price_per_person", None)

        # Build a header summary block (overview page)
        highlights = data.get("highlights", [])
        inclusions = data.get("inclusions", [])
        exclusions = data.get("exclusions", [])
        notes = data.get("notes", "")

        overview_parts = [
            f"Itinerary: {title}",
            f"Duration: {duration} days",
            f"Region: {region}",
        ]
        if price:
            overview_parts.append(f"Price per person: ₹{price}")
        if highlights:
            overview_parts.append("Highlights: " + ", ".join(str(h) for h in highlights))
        if inclusions:
            overview_parts.append("Inclusions: " + ", ".join(str(i) for i in inclusions))
        if exclusions:
            overview_parts.append("Exclusions: " + ", ".join(str(e) for e in exclusions))
        if notes:
            overview_parts.append(f"Notes: {notes}")

        overview_text = "\n".join(overview_parts)
        pages_data.append({
            "text": overview_text,
            "metadata": {
                "source": source_name,
                "page": 0,  # 0 = overview
                "title": title,
                "document_type": "itinerary",
                "region": region,
                "duration_days": duration
            }
        })

        # Each day as a separate chunk
        for day_obj in data.get("days", []):
            day_num = day_obj.get("day", 0)
            day_title = day_obj.get("title", f"Day {day_num}")
            description = day_obj.get("description", "")
            activities = day_obj.get("activities", [])
            accommodation = day_obj.get("accommodation", "")
            meals = day_obj.get("meals", "")

            parts = [f"Day {day_num}: {day_title}", description]
            if activities:
                parts.append("Activities: " + ", ".join(str(a) for a in activities))
            if accommodation:
                parts.append(f"Accommodation: {accommodation}")
            if meals:
                parts.append(f"Meals: {meals}")

            day_text = "\n".join(p for p in parts if p)

            pages_data.append({
                "text": day_text,
                "metadata": {
                    "source": source_name,
                    "page": day_num,
                    "title": f"{title} — Day {day_num}",
                    "document_type": "itinerary",
                    "region": region,
                    "duration_days": duration
                }
            })

    except Exception as e:
        print(f"[ERROR] Failed to load JSON itinerary {file_path}: {e}")

    return pages_data


def load_document_file(file_path: str, doc_type: str) -> list[dict]:
    """
    Dispatcher: loads a single file based on its extension.
    doc_type should be one of: "itinerary", "property", "faq", "policy"
    """
    ext = file_path.lower().split(".")[-1]

    if ext == "pdf":
        return load_pdf(file_path, doc_type)
    elif ext == "txt":
        return load_txt(file_path, doc_type)
    elif ext == "json":
        return load_json_itinerary(file_path)
    else:
        print(f"[WARNING] Unsupported file type: {file_path}")
        return []


def load_data_directory(data_dir: str) -> list[dict]:
    """
    Scans the data directory for subdirectories and loads all documents.

    Expected directory layout:
        data/
          itineraries/     ← 45 JSON/PDF/TXT itineraries
          properties/      ← Homestay room & property descriptions
          faqs/            ← Common guest FAQs
          policies/        ← Booking, cancellation, and house policies

    Returns a flat list of page records.
    """
    all_pages = []

    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return all_pages

    subdir_type_map = {
        "itineraries": "itinerary",
        "properties": "property",
        "faqs": "faq",
        "policies": "policy"
    }

    for subdir, doc_type in subdir_type_map.items():
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            continue

        for fname in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, fname)
            if os.path.isfile(file_path):
                pages = load_document_file(file_path, doc_type)
                all_pages.extend(pages)
                print(f"[LOADER] {doc_type.upper()} | {fname} → {len(pages)} page(s)")

    print(f"[LOADER] Total pages loaded: {len(all_pages)}")
    return all_pages
