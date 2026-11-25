#!/usr/bin/env python3
"""Build the vector index from data/docs/*.md.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --embedder tfidf   # dependency-free
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import build_chunks
from src.config import SETTINGS, write_index_config
from src.embeddings import get_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedder",
        choices=["sentence-transformers", "tfidf"],
        default=SETTINGS.embedder_backend,
        help="Embedding backend to use (default: from EMBEDDER_BACKEND / .env)",
    )
    parser.add_argument("--chunk-size", type=int, default=SETTINGS.chunk_size)
    parser.add_argument("--overlap", type=int, default=SETTINGS.chunk_overlap)
    args = parser.parse_args()

    print(f"Loading + chunking documents from {SETTINGS.data_dir} ...")
    chunks = build_chunks(SETTINGS.data_dir, args.chunk_size, args.overlap)
    if not chunks:
        raise SystemExit(f"No documents found in {SETTINGS.data_dir} -- nothing to index.")
    print(f"  -> {len(chunks)} chunks from "
          f"{len({c.source for c in chunks})} source documents")

    print(f"Embedding with backend={args.embedder!r} ...")
    embedder = get_embedder(args.embedder, SETTINGS.sentence_transformer_model)
    texts = [c.text for c in chunks]
    embedder.fit(texts)  # no-op for sentence-transformers, fits vocab for tfidf
    vectors = embedder.embed(texts)

    from src.vectorstore import VectorStore

    store = VectorStore()
    store.add(
        ids=[f"{c.source}::{c.chunk_index}" for c in chunks],
        vectors=vectors,
        documents=texts,
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )

    SETTINGS.index_dir.mkdir(parents=True, exist_ok=True)
    store.save(SETTINGS.index_dir)
    embedder.save(SETTINGS.index_dir)
    write_index_config(
        SETTINGS.index_dir,
        embedder_backend=args.embedder,
        sentence_transformer_model=SETTINGS.sentence_transformer_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        num_chunks=len(chunks),
    )
    print(f"Saved index to {SETTINGS.index_dir}/ ({len(chunks)} vectors)")


if __name__ == "__main__":
    main()
