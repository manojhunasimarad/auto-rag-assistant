"""Document loading and chunking.

This module has zero third-party dependencies on purpose: chunking is pure
text processing, and keeping it dependency-free means it's trivial to unit
test (see tests/test_chunking.py) without needing sentence-transformers,
langchain, or any network access.

Why character-based chunking instead of token-based
-----------------------------------------------------
A "correct" production system would chunk by token count (using the
tokenizer of whatever embedding/LLM model you're using) so chunks line up
exactly with the model's context window. We chunk by character count with
a generous safety margin instead, because it needs no tokenizer dependency
and the corpus here is short technical prose where char-count and
token-count track each other closely enough (~4 chars/token for English).
This is a real trade-off worth stating out loud in an interview: character
chunking is simpler and dependency-free, token chunking is more precise
and is what you'd actually want once prompts get close to a context-window
limit.

Why paragraph-aware splitting instead of a hard character cutoff
------------------------------------------------------------------
Cutting a chunk mid-sentence hurts retrieval quality (the embedding for a
half-sentence is a worse representation of its meaning) and hurts
generation quality (the model has to work with a truncated thought). We
split on paragraph boundaries first, and only fall back to splitting a
too-long paragraph on sentence boundaries, so the vast majority of chunks
end on a natural boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # filename the chunk came from
    chunk_index: int  # position of this chunk within its source document


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def _split_sentences(paragraph: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    return [s for s in sentences if s]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split `text` into overlapping chunks of roughly `chunk_size` chars.

    Strategy: greedily pack whole paragraphs (falling back to sentences for
    an over-long paragraph) into a running buffer until adding the next
    unit would exceed `chunk_size`; then flush the buffer as a chunk and
    seed the next buffer with the trailing `overlap` characters of the one
    just flushed, so retrieval doesn't lose context that happened to sit
    right on a chunk boundary.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]

    # Expand any paragraph that's already bigger than chunk_size into
    # sentence-sized units so the packer below always has small-enough
    # pieces to work with.
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_split_sentences(para))

    chunks: list[str] = []
    buffer = ""
    for unit in units:
        candidate = f"{buffer}\n\n{unit}" if buffer else unit
        if len(candidate) <= chunk_size or not buffer:
            buffer = candidate
        else:
            chunks.append(buffer)
            tail = buffer[-overlap:] if overlap else ""
            buffer = f"{tail}\n\n{unit}" if tail else unit
    if buffer:
        chunks.append(buffer)

    return chunks


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Return [(filename, full_text), ...] for every .md file in docs_dir."""
    docs_dir = Path(docs_dir)
    pairs = []
    for path in sorted(docs_dir.glob("*.md")):
        pairs.append((path.name, path.read_text(encoding="utf-8")))
    return pairs


def build_chunks(
    docs_dir: Path, chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    """Load every doc in docs_dir and chunk it, returning a flat list."""
    all_chunks: list[Chunk] = []
    for filename, text in load_documents(docs_dir):
        for idx, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            all_chunks.append(Chunk(text=chunk, source=filename, chunk_index=idx))
    return all_chunks
