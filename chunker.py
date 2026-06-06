"""
chunker.py — Sentence-level sliding window chunker for the Himachal Homestay RAG pipeline.

Architecture is identical to the Feynman pipeline (HybridChunker), 
with one addition: structured fields (Day titles, Accommodation lines, 
Price lines) are preserved as atomic sentences to prevent them from 
being split mid-sentence by the regex splitter.
"""

import re


# Lines that should never be broken mid-sentence during splitting
PRESERVE_PATTERNS = [
    r"^Day \d+",
    r"^Accommodation:",
    r"^Activities:",
    r"^Meals:",
    r"^Price",
    r"^Duration:",
    r"^Inclusions:",
    r"^Exclusions:",
    r"^Highlights:",
    r"^Check-in",
    r"^Check-out",
    r"^Note:",
]
_PRESERVE_RE = re.compile("|".join(PRESERVE_PATTERNS), re.IGNORECASE)


class HybridChunker:
    """
    Splits documents into sentences with sliding window context.

    window_size=3 means each sentence's context includes up to 3 sentences
    before and after it — giving the retriever a richer passage to score
    while the embedding is computed only on the center sentence.
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Splits a text block into sentences.
        Structural lines (Day N:, Accommodation:, etc.) are treated as 
        atomic units and never split further.
        """
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text).strip()

        # Split into lines first to handle structured itinerary lines
        raw_lines = text.split('\n')
        sentences = []

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue

            # Preserve structured itinerary lines as-is (don't sentence-split them)
            if _PRESERVE_RE.match(line):
                sentences.append(line)
                continue

            # Regex sentence boundary splitter (avoids splitting on Mr., e.g., etc.)
            sentence_end = re.compile(
                r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
            )
            parts = sentence_end.split(line)
            sentences.extend(s.strip() for s in parts if s.strip())

        return sentences

    def chunk_document(self, document: dict) -> list[dict]:
        """
        Chunks a single document page/record.
        Returns a list of chunk dicts.
        """
        text = document["text"]
        metadata = document["metadata"]

        # Split on double newlines to get paragraphs, then sentence-split each
        paragraphs = re.split(r'\n\n+', text)
        all_sentences = []
        for para in paragraphs:
            if para.strip():
                all_sentences.extend(self.split_into_sentences(para))

        chunks = []
        total = len(all_sentences)

        for i in range(total):
            center = all_sentences[i]
            start = max(0, i - self.window_size)
            end = min(total, i + self.window_size + 1)
            window_context = " ".join(all_sentences[start:end])

            chunks.append({
                "original_sentence": center,
                "window": window_context,
                "sentence_index": i,
                "metadata": metadata.copy()
            })

        return chunks

    def chunk_all(self, documents: list[dict]) -> list[dict]:
        """Chunks all documents and returns a single flat list."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        print(f"[CHUNKER] Total chunks produced: {len(all_chunks)}")
        return all_chunks
