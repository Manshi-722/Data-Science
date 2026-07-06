"""
Text chunking utilities.

Splits raw document text into overlapping chunks so that:
- chunks are small enough for accurate embedding + retrieval
- overlap preserves context that would otherwise be cut at a boundary
"""
import re
from typing import List, Dict


def _split_into_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter (no heavy NLP dependency needed).
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Chunk text by characters, breaking on sentence boundaries where possible.
    chunk_size / chunk_overlap are measured in characters.
    """
    sentences = _split_into_sentences(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying over the overlap tail of the previous chunk
            overlap_text = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap_text} {sentence}".strip()

    if current:
        chunks.append(current)

    # Fallback: if a document has no sentence punctuation at all, hard-split it.
    if not chunks and text.strip():
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    return chunks


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Turn a list of {source, text} documents into a flat list of chunk records:
    {id, source, chunk_index, text}
    """
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            records.append({
                "id": f"{doc['source']}::{i}",
                "source": doc["source"],
                "chunk_index": i,
                "text": piece,
            })
    return records
