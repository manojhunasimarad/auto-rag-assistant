#!/usr/bin/env python3
"""Retrieval evaluation: hit-rate@k and MRR against a small hand-labeled set.

This is deliberately simple -- it does not evaluate the generated answer
text (that needs either human judgment or an LLM-as-judge setup, both out
of scope for a week-long MVP), only whether the retriever pulls the right
*document* into the top-k for each question. That's still the single most
useful automated signal for a RAG system, because if retrieval is wrong,
generation quality is capped no matter how good the prompt is -- this is
usually the first thing worth being able to show numbers for in an
interview.

Metrics:
- hit_rate@k: fraction of questions where the expected source document
  appears anywhere in the top-k retrieved chunks.
- MRR (mean reciprocal rank): average of 1/rank of the first chunk from
  the expected source document (0 if it never appears in top-k) -- this
  rewards getting the right document ranked *first*, not just present.

Usage:
    uv run python eval/eval_retrieval.py
    uv run python eval/eval_retrieval.py --k 2
    uv run python eval/eval_retrieval.py --embedder tfidf   # compare against the semantic backend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SETTINGS, read_index_config
from src.embeddings import load_embedder
from src.vectorstore import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=SETTINGS.top_k)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).resolve().parent / "eval_questions.json",
    )
    args = parser.parse_args()

    if not (SETTINGS.index_dir / "meta.json").exists():
        raise SystemExit(
            f"No index found at {SETTINGS.index_dir}. Run "
            "`uv run python scripts/build_index.py` first."
        )

    index_config = read_index_config(SETTINGS.index_dir)
    vectorstore = VectorStore.load(SETTINGS.index_dir)
    embedder = load_embedder(
        index_config["embedder_backend"], index_config["sentence_transformer_model"], SETTINGS.index_dir
    )

    questions = json.loads(args.questions.read_text(encoding="utf-8"))

    hits = 0
    reciprocal_ranks = []
    rows = []

    for item in questions:
        query_vector = embedder.embed([item["question"]])[0]
        results = vectorstore.query(query_vector, k=args.k)
        sources_in_order = [r.metadata.get("source") for r in results]

        rank = None
        for i, source in enumerate(sources_in_order, start=1):
            if source == item["expected_source"]:
                rank = i
                break

        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        rows.append((item["question"], item["expected_source"], sources_in_order, rank))

    n = len(questions)
    hit_rate = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0

    print(f"Index: embedder_backend={index_config['embedder_backend']!r}  k={args.k}\n")
    for question, expected, retrieved, rank in rows:
        status = f"HIT (rank {rank})" if rank else "MISS"
        print(f"[{status}] {question}")
        print(f"    expected: {expected}")
        print(f"    retrieved: {retrieved}")
    print(f"\nhit_rate@{args.k} = {hit_rate:.2f}  ({hits}/{n})")
    print(f"MRR@{args.k}      = {mrr:.2f}")


if __name__ == "__main__":
    main()
