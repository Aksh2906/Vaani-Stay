"""
embedder.py — Batch BGE embedding and ChromaDB ingestion for the Himachal Homestay pipeline.

Identical architecture to the Feynman pipeline embedder.
Changes:
  - Source authority weights reflect travel document types
    (itinerary = 1.0, property = 0.95, faq = 0.90, policy = 0.85)
  - Metadata includes region and duration_days from the loader
"""

import hashlib
import time
import streamlit as st

from helpers import determine_domain


def generate_deterministic_id(source: str, page: int, sentence_index: int) -> str:
    """Generates a unique, deterministic MD5 chunk ID."""
    input_str = f"{source}_page{page}_sent{sentence_index}"
    return hashlib.md5(input_str.encode('utf-8')).hexdigest()


# Source authority weights for Himachal travel document types
SOURCE_AUTHORITY = {
    "itinerary": 1.00,   # Core content — highest trust
    "property": 0.95,    # Official property descriptions
    "faq": 0.90,         # Curated Q&A
    "policy": 0.85       # Policies — accurate but less conversational
}


def ingest_chunks_to_chroma(chunks: list[dict], embedder, collection, batch_size: int = 64) -> int:
    """
    Embeds chunks in batches and inserts them into ChromaDB.
    Skips chunks that already exist (deterministic ID deduplication).

    Returns:
        inserted_count: number of new vectors added
    """
    total_chunks = len(chunks)
    if total_chunks == 0:
        return 0

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.info("Pre-processing chunks and checking for duplicates...")

    inserted_count = 0

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]

        batch_ids = []
        batch_metadatas = []
        batch_docs = []
        batch_texts_to_embed = []

        for c in batch:
            meta = c["metadata"]
            cid = generate_deterministic_id(meta["source"], meta["page"], c["sentence_index"])

            # Domain classification (keyword-first, semantic fallback)
            domain = determine_domain(c["original_sentence"], embedder)

            # Source authority by document type
            doc_type = meta.get("document_type", "itinerary")
            source_auth = SOURCE_AUTHORITY.get(doc_type, 0.80)

            batch_ids.append(cid)
            batch_texts_to_embed.append(c["original_sentence"])
            batch_docs.append(c["original_sentence"])

            batch_metadatas.append({
                "chunk_id": cid,
                "source": meta["source"],
                "page": int(meta["page"]),
                "title": meta.get("title", meta["source"]),
                "document_type": doc_type,
                "region": meta.get("region", "General Himachal"),
                "duration_days": int(meta.get("duration_days", 0)),
                "domain": domain,
                "source_authority": float(source_auth),
                "sentence_index": int(c["sentence_index"]),
                "window": c["window"],
                "created_at": float(time.time())
            })

        # Check for existing IDs to prevent re-insertion
        try:
            existing = collection.get(ids=batch_ids, include=[])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            existing_ids = set()

        # Embed only new documents
        # BGE: no instruction prefix needed for document embeddings
        embeddings = embedder.encode(
            batch_texts_to_embed,
            batch_size=batch_size,
            normalize_embeddings=True
        ).tolist()

        write_ids, write_embeddings, write_docs, write_metadatas = [], [], [], []

        for idx, cid in enumerate(batch_ids):
            if cid not in existing_ids:
                write_ids.append(cid)
                write_embeddings.append(embeddings[idx])
                write_docs.append(batch_docs[idx])
                write_metadatas.append(batch_metadatas[idx])

        if write_ids:
            collection.add(
                ids=write_ids,
                embeddings=write_embeddings,
                documents=write_docs,
                metadatas=write_metadatas
            )
            inserted_count += len(write_ids)

        progress = min((i + batch_size) / total_chunks, 1.0)
        progress_bar.progress(progress)
        status_text.info(
            f"Ingested {min(i + batch_size, total_chunks)} / {total_chunks} chunks "
            f"({inserted_count} new vectors added)..."
        )

    progress_bar.empty()
    status_text.empty()

    print(f"[EMBEDDER] Ingestion complete. {inserted_count} new vectors added to ChromaDB.")
    return inserted_count
