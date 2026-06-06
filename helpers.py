"""
helpers.py — Domain classification helpers for the Himachal Homestay RAG pipeline.

Replaces the Feynman physics domain taxonomy with a 7-domain taxonomy 
relevant to travel booking: destinations, accommodation, activities, 
pricing/booking, logistics, food, and policies.
"""

import re
import numpy as np

# ---------------------------------------------------------------------------
# DOMAIN KEYWORDS
# ---------------------------------------------------------------------------
DOMAIN_KEYWORDS = {
    "destination_info": [
        "manali", "shimla", "spiti", "kullu", "kasol", "dharamshala", "mcleod",
        "dalhousie", "chamba", "kinnaur", "sarahan", "chitkul", "kaza", "tabo",
        "bir billing", "jibhi", "tirthan", "barot", "palampur", "khajjiar",
        "chail", "kufri", "naldehra", "rohtang", "solang", "triund", "rampur",
        "sangla", "kalpa", "nako", "pin valley", "lahaul", "great himalayan",
        "altitude", "pass", "valley", "river", "lake", "village", "trek route",
        "himalaya", "himachal", "snow", "peak", "mountain"
    ],
    "accommodation": [
        "room", "homestay", "cottage", "cabin", "dorm", "dormitory", "bed",
        "double", "twin", "single", "suite", "deluxe", "standard", "luxury",
        "check-in", "check-out", "property", "stay", "night", "accommodation",
        "wifi", "hot water", "heater", "blanket", "balcony", "view", "amenities",
        "bathroom", "attached", "common", "kitchen", "meals included", "host"
    ],
    "activities": [
        "trek", "trekking", "hike", "hiking", "camping", "paragliding", "rafting",
        "skiing", "snowboarding", "snowfall", "adventure", "jeep safari", "bike",
        "cycling", "fishing", "birdwatching", "monastery", "temple", "waterfall",
        "hot spring", "apple orchard", "village walk", "photography", "yoga",
        "meditation", "bonfire", "stargazing", "river crossing", "rappelling",
        "local market", "cultural tour", "heritage", "festival", "fair"
    ],
    "pricing_and_booking": [
        "price", "cost", "rate", "tariff", "per person", "per night", "₹", "inr",
        "rupee", "budget", "package", "offer", "discount", "advance", "deposit",
        "payment", "upi", "bank transfer", "booking", "reservation", "confirm",
        "availability", "slots", "dates", "season", "peak season", "off season",
        "group booking", "couple", "family", "solo", "charges", "extra", "fee"
    ],
    "logistics_and_travel": [
        "how to reach", "route", "distance", "km", "hours", "bus", "taxi",
        "cab", "train", "flight", "chandigarh", "delhi", "airport", "railway",
        "volvo", "hrtc", "private cab", "self drive", "road", "highway",
        "nh3", "nh21", "nh505", "manali highway", "permit", "inner line",
        "restricted area", "best time", "season", "monsoon", "winter", "summer",
        "april", "may", "june", "october", "november", "december", "january",
        "february", "march", "snowfall", "road block", "closed", "open"
    ],
    "food_and_dining": [
        "food", "meal", "breakfast", "lunch", "dinner", "thali", "dosa",
        "tibetan", "momos", "noodles", "rajma chawal", "siddu", "madra",
        "local cuisine", "himachali", "vegetarian", "vegan", "restaurant",
        "cafe", "dhaba", "included", "not included", "dietary", "allergies"
    ],
    "policies_and_faqs": [
        "cancellation", "refund", "policy", "rules", "guidelines", "check in time",
        "check out time", "smoking", "alcohol", "pets", "children", "extra bed",
        "id proof", "aadhaar", "passport", "government id", "couple policy",
        "unmarried", "quiet hours", "noise", "waste", "plastic", "eco", "terms",
        "conditions", "emergency", "contact", "support", "helpline"
    ]
}

# ---------------------------------------------------------------------------
# SEMANTIC PROTOTYPES
# One sentence per domain that best captures its semantic center.
# ---------------------------------------------------------------------------
PROTOTYPES = {
    "destination_info": (
        "Travel destinations in Himachal Pradesh including Manali, Spiti Valley, "
        "Shimla, Dharamshala, Kasol and other Himalayan hill stations with altitude, "
        "geography, and sightseeing information."
    ),
    "accommodation": (
        "Homestay rooms, cottages, dormitories, amenities, check-in and check-out "
        "procedures, room types, bedding, bathrooms, and property facilities in Himachal Pradesh."
    ),
    "activities": (
        "Adventure activities, trekking, camping, paragliding, skiing, river rafting, "
        "monastery visits, village walks, cultural tours, and outdoor experiences in the Himalayas."
    ),
    "pricing_and_booking": (
        "Package prices per person per night, booking confirmation, payment methods, "
        "advance deposit, seasonal rates, group discounts, and reservation process."
    ),
    "logistics_and_travel": (
        "How to reach Himachal Pradesh by bus, taxi, train, or flight from Delhi or Chandigarh, "
        "road conditions, best travel season, permits for restricted areas, and route distances."
    ),
    "food_and_dining": (
        "Himachali cuisine, meals included in package, local food, breakfast lunch dinner, "
        "vegetarian options, Tibetan food, cafes and dhabas near the homestay."
    ),
    "policies_and_faqs": (
        "Cancellation and refund policy, house rules, ID proof requirements, couple policy, "
        "check-in check-out timings, smoking alcohol pet guidelines, and emergency contacts."
    )
}

# In-memory cache for embedded domain prototypes
_prototype_embeddings = None


def get_prototype_embeddings(embedder):
    """Generates and caches BGE embeddings for the domain prototypes."""
    global _prototype_embeddings
    if _prototype_embeddings is None and embedder is not None:
        domains = list(PROTOTYPES.keys())
        texts = [PROTOTYPES[d] for d in domains]
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        _prototype_embeddings = dict(zip(domains, embeddings))
    return _prototype_embeddings


def determine_domain(text: str, embedder=None) -> str:
    """
    Classifies a text chunk into one of 7 domains using keyword heuristics.
    Falls back to semantic prototype cosine similarity if ambiguous.
    """
    clean_text = text.lower()

    # 1. Keyword-based matching
    scores = {}
    for domain, kw_list in DOMAIN_KEYWORDS.items():
        count = 0
        for kw in kw_list:
            if len(kw) < 4:
                matches = re.findall(rf"\b{re.escape(kw)}\b", clean_text)
            else:
                matches = re.findall(re.escape(kw), clean_text)
            count += len(matches)
        if count > 0:
            scores[domain] = count

    if scores:
        max_score = max(scores.values())
        winners = [d for d, s in scores.items() if s == max_score]
        if len(winners) == 1:
            return winners[0]

    # 2. Semantic prototype fallback
    if embedder is not None:
        try:
            proto_embeds = get_prototype_embeddings(embedder)
            if proto_embeds:
                chunk_emb = embedder.encode(text, normalize_embeddings=True)
                best_domain = "destination_info"
                best_sim = -1.0

                for domain, proto_emb in proto_embeds.items():
                    sim = np.dot(chunk_emb, proto_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_domain = domain

                if best_sim > 0.30:
                    return best_domain
        except Exception:
            pass

    return "destination_info"  # Safe default for travel content
