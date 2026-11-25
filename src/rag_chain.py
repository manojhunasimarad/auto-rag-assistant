"""Retrieval + grounded generation.

This is the "RAG" in RAG: given a question, (1) embed it, (2) retrieve the
top-k most similar chunks from the vector store, (3) build a prompt that
forces the model to answer only from those chunks and cite them, (4) call
the LLM backend, (5) return the answer plus the sources it's grounded in.

The system prompt is the main lever against hallucination here: it tells
the model explicitly to say "I don't know" rather than invent an answer
when the retrieved context doesn't cover the question, and to cite chunk
numbers. This doesn't make hallucination impossible (nothing except a
verifier/critic step really does), but it's the standard first line of
defense and it's the one interviewers most often ask about.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.embeddings import Embedder
from src.llm import LLMBackend
from src.vectorstore import QueryResult, VectorStore

SYSTEM_PROMPT = """You are a technical assistant answering questions about \
in-vehicle systems engineering (AUTOSAR Adaptive, IEEE1722/AVTP, hypervisor \
bring-up, ROS2-cloud bridging) using ONLY the numbered context chunks the \
user provides.

Rules:
1. Answer only using facts present in the provided context chunks.
2. Every claim you make must be followed by a citation like [1] or [2] \
referring to the chunk number it came from.
3. If the context does not contain enough information to answer, say so \
plainly instead of guessing.
4. Be concise. Prefer a direct answer over restating the question."""


def build_prompt(query: str, results: list[QueryResult]) -> tuple[str, str]:
    if not results:
        user = (
            f"Question: {query}\n\n"
            "No context chunks were retrieved (the index may be empty). "
            "Say you don't have enough information to answer."
        )
        return SYSTEM_PROMPT, user

    context_blocks = []
    for i, r in enumerate(results, start=1):
        context_blocks.append(f"[{i}] (source: {r.metadata.get('source', 'unknown')})\n{r.document}")
    context = "\n\n".join(context_blocks)

    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer, citing chunk numbers:"
    return SYSTEM_PROMPT, user


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]  # unique source filenames, in citation order
    retrieved: list[QueryResult]


class RagPipeline:
    def __init__(
        self,
        vectorstore: VectorStore,
        embedder: Embedder,
        llm_backend: LLMBackend,
        top_k: int = 4,
    ) -> None:
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.llm_backend = llm_backend
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> list[QueryResult]:
        query_vector = self.embedder.embed([query])[0]
        return self.vectorstore.query(query_vector, k=k or self.top_k)

    def answer(self, query: str, k: int | None = None) -> RagAnswer:
        results = self.retrieve(query, k=k)
        system, user = build_prompt(query, results)
        raw_answer = self.llm_backend.chat(system, user)

        seen: set[str] = set()
        sources: list[str] = []
        for r in results:
            source = r.metadata.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                sources.append(source)

        return RagAnswer(answer=raw_answer, sources=sources, retrieved=results)
