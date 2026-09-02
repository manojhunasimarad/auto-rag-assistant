"""FastAPI serving layer.

Two endpoints:
  GET  /health          -- liveness check
  POST /query            -- ask a question, either straight RAG or via the agent

Run it:
    uv run uvicorn src.api:app --reload

Then:
    curl -X POST localhost:8000/query -H "content-type: application/json" \\
         -d '{"question": "How does AVTP carry CAN signals across domains?"}'
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import default_tools, run_agent
from src.config import SETTINGS, read_index_config
from src.embeddings import load_embedder
from src.llm import get_llm_backend
from src.rag_chain import RagPipeline
from src.vectorstore import VectorStore

app = FastAPI(title="auto-rag-assistant", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field("rag", pattern="^(rag|agent)$")
    k: int | None = Field(None, ge=1, le=20)


class QueryResponse(BaseModel):
    mode: str
    answer: str
    sources: list[str]


@lru_cache(maxsize=1)
def _get_pipeline() -> RagPipeline:
    """Load the index + backends once per process, not once per request.

    An embedding model (and, for the real backend, an LLM client) is
    expensive enough to construct that doing it per-request would make
    every API call noticeably slower for no benefit -- the index and
    model don't change between requests.
    """
    if not (SETTINGS.index_dir / "meta.json").exists():
        raise RuntimeError(
            f"No index found at {SETTINGS.index_dir}. Run "
            "`uv run python scripts/build_index.py` first."
        )

    index_config = read_index_config(SETTINGS.index_dir)
    vectorstore = VectorStore.load(SETTINGS.index_dir)
    embedder = load_embedder(
        index_config["embedder_backend"],
        index_config["sentence_transformer_model"],
        SETTINGS.index_dir,
    )
    import os

    llm_backend = get_llm_backend(
        SETTINGS.llm_backend,
        SETTINGS.openai_model,
        SETTINGS.anthropic_model,
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("ANTHROPIC_API_KEY", ""),
    )
    return RagPipeline(vectorstore, embedder, llm_backend, top_k=SETTINGS.top_k)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        rag = _get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if request.mode == "rag":
        result = rag.answer(request.question, k=request.k)
        return QueryResponse(mode="rag", answer=result.answer, sources=result.sources)

    # mode == "agent"
    tools = default_tools(rag)
    result = run_agent(request.question, rag.llm_backend, tools)
    sources = []
    for step in result.steps:
        if step.tool_used == "search_knowledge_base" and step.observation:
            for line in step.observation.splitlines():
                if line.startswith("["):
                    # Lines look like: "[1] (source.md, score=0.42): ..."
                    start = line.find("(")
                    end = line.find(",", start)
                    if start != -1 and end != -1:
                        source = line[start + 1 : end]
                        if source not in sources:
                            sources.append(source)
    return QueryResponse(mode="agent", answer=result.final_answer, sources=sources)
