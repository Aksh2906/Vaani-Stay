"""
retriever.py — Hybrid RAG retriever for the Himachal Pradesh Homestay pipeline.

Architecture is identical to the Feynman pipeline:
  Dense (ChromaDB) + Sparse (BM25) → Score Fusion → Cross-Encoder Reranking

Only changes:
  - Query rewriter prompt references travel/booking context
  - Source authority weights updated for travel document types
  - Domain-aware score boosting for booking-intent queries
"""

import re
import numpy as np
import google.generativeai as genai
from rank_bm25 import BM25Okapi

from templates import QUERY_REWRITER_PROMPT


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())


# Domains that indicate booking intent — boost pricing/policy chunks when detected
BOOKING_INTENT_DOMAINS = {"pricing_and_booking", "accommodation", "policies_and_faqs"}

# Query signals that indicate booking intent
BOOKING_INTENT_SIGNALS = [
    "book", "reserve", "availability", "price", "cost", "how much",
    "confirm", "room", "dates", "check in", "check out", "payment"
]


def detect_booking_intent(query: str) -> bool:
    """Returns True if the query appears to be booking-related."""
    q_lower = query.lower()
    return any(signal in q_lower for signal in BOOKING_INTENT_SIGNALS)


class HybridRetriever:
    """
    Production-grade hybrid retriever for the Himachal Homestay RAG pipeline.

    Pipeline:
      1. Dense vector retrieval (ChromaDB, Top 40)
      2. Sparse BM25 keyword retrieval (Top 40)
      3. Hybrid score fusion with source authority + domain boost
      4. Cross-Encoder reranking (Top 5)
    """

    def __init__(self, chroma_collection, embedder, reranker):
        self.collection = chroma_collection
        self.embedder = embedder
        self.reranker = reranker
        self.bm25 = None
        self.corpus_ids = []
        self.corpus_metadatas = []
        self.corpus_documents = []
        self.refresh_bm25()

    def refresh_bm25(self):
        """Re-fetches all chunks from Chroma and rebuilds the BM25 index."""
        count = self.collection.count()
        if count == 0:
            self.bm25 = None
            self.corpus_ids = []
            self.corpus_metadatas = []
            self.corpus_documents = []
            return

        results = self.collection.get(include=["metadatas", "documents"])
        self.corpus_ids = results.get("ids", [])
        self.corpus_metadatas = results.get("metadatas", [])
        self.corpus_documents = results.get("documents", [])

        tokenized_corpus = [tokenize(doc) for doc in self.corpus_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[RETRIEVER] BM25 index built on {len(self.corpus_ids)} chunks.")

    def rewrite_query(self, user_query: str, chat_history: list[dict], api_key: str) -> str:
        """
        Uses Gemini Flash to rewrite a context-dependent query into a
        standalone search query using the last 6 conversation turns.
        """
        if not chat_history:
            return user_query

        recent_turns = chat_history[-6:]
        history_str = ""
        for msg in recent_turns:
            role = "Guest" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        prompt = QUERY_REWRITER_PROMPT.format(
            history=history_str.strip(),
            latest_message=user_query
        )

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            rewritten = response.text.strip()
            if rewritten and len(rewritten) > 2:
                return rewritten
        except Exception as e:
            print(f"[WARNING] Query rewriting failed: {e}")

        return user_query

    def retrieve(self, query: str, top_k: int = 5):
        """
        Full hybrid retrieval pipeline.

        Returns:
            final_windows: list[str]  — top-k context windows for the prompt
            debug_records: list[dict] — scoring breakdown for each result
        """
        if self.collection.count() == 0 or self.bm25 is None:
            return [], []

        is_booking_query = detect_booking_intent(query)

        # ---- A. DENSE RETRIEVAL ----
        bge_query = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.embedder.encode([bge_query], normalize_embeddings=True).tolist()[0]

        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(40, self.collection.count())
        )

        dense_candidates = {}
        if dense_results["ids"] and dense_results["distances"]:
            for idx, cid in enumerate(dense_results["ids"][0]):
                dist = dense_results["distances"][0][idx]
                sim = max(0.0, min(1.0, 1.0 - dist))
                meta = dense_results["metadatas"][0][idx]
                doc = dense_results["documents"][0][idx]

                # Source authority — itineraries and properties are highest quality
                auth = float(meta.get("source_authority", 0.75))

                # Domain boost for booking-intent queries
                domain = meta.get("domain", "destination_info")
                domain_boost = 1.15 if (is_booking_query and domain in BOOKING_INTENT_DOMAINS) else 1.0

                dense_score = sim * auth * domain_boost
                dense_candidates[cid] = {
                    "doc": doc,
                    "metadata": meta,
                    "dense_score": dense_score,
                    "distance": dist
                }

        # ---- B. SPARSE RETRIEVAL (BM25) ----
        q_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)

        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 0.0
        min_bm25 = min(bm25_scores) if len(bm25_scores) > 0 else 0.0
        bm25_range = max_bm25 - min_bm25

        bm25_candidates = {}
        top_indices = np.argsort(bm25_scores)[-40:][::-1]

        for idx in top_indices:
            score = bm25_scores[idx]
            if score <= 0.0:
                continue
            cid = self.corpus_ids[idx]
            doc = self.corpus_documents[idx]
            meta = self.corpus_metadatas[idx]
            norm_score = (score - min_bm25) / bm25_range if bm25_range > 0 else (score / max_bm25 if max_bm25 > 0 else 0.0)
            bm25_candidates[cid] = {
                "doc": doc,
                "metadata": meta,
                "bm25_score": norm_score
            }

        # ---- C. HYBRID FUSION (60% dense + 40% BM25) ----
        union_ids = set(dense_candidates.keys()).union(set(bm25_candidates.keys()))
        fused_results = []

        for cid in union_ids:
            dense_info = dense_candidates.get(cid)
            bm25_info = bm25_candidates.get(cid)

            doc = dense_info["doc"] if dense_info else bm25_info["doc"]
            meta = dense_info["metadata"] if dense_info else bm25_info["metadata"]
            d_score = dense_info["dense_score"] if dense_info else 0.0
            b_score = bm25_info["bm25_score"] if bm25_info else 0.0

            hybrid_score = 0.6 * d_score + 0.4 * b_score
            fused_results.append({
                "id": cid,
                "doc": doc,
                "metadata": meta,
                "dense_score": d_score,
                "bm25_score": b_score,
                "hybrid_score": hybrid_score
            })

        fused_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_20 = fused_results[:20]

        if not top_20:
            return [], []

        # ---- D. CROSS-ENCODER RERANKING ----
        pairs = [(query, c["metadata"].get("window", c["doc"])) for c in top_20]
        reranker_scores = self.reranker.predict(pairs).tolist()

        reranked_results = []
        for idx, cand in enumerate(top_20):
            raw_score = reranker_scores[idx]
            prob_score = 1.0 / (1.0 + np.exp(-raw_score))
            cand["reranker_score"] = float(prob_score)
            reranked_results.append(cand)

        reranked_results.sort(key=lambda x: x["reranker_score"], reverse=True)
        top_results = reranked_results[:top_k]

        final_windows = [c["metadata"].get("window", c["doc"]) for c in top_results]
        debug_records = []

        for rank, c in enumerate(top_results, 1):
            meta = c["metadata"]
            debug_records.append({
                "rank": rank,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 1),
                "title": meta.get("title", "Unknown"),
                "region": meta.get("region", "General"),
                "domain": meta.get("domain", "destination_info"),
                "document_type": meta.get("document_type", "itinerary"),
                "source_authority": meta.get("source_authority", 0.75),
                "dense_score": round(c["dense_score"], 4),
                "bm25_score": round(c["bm25_score"], 4),
                "hybrid_score": round(c["hybrid_score"], 4),
                "reranker_score": round(c["reranker_score"], 4),
                "original_sentence": c["doc"],
                "window": meta.get("window", c["doc"])
            })

        return final_windows, debug_records
