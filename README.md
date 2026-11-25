# auto-rag-assistant

A RAG + agentic knowledge assistant over in-vehicle systems engineering
documentation — AUTOSAR Adaptive Platform, IEEE1722/AVTP cross-domain
signal transport, Type-1 hypervisor (Xen/QNX) multi-OS bring-up, and a
ROS2-to-cloud gRPC bridge. The knowledge base is a small corpus of
technical write-ups drawn from real automotive ECU/board bring-up work
(KPIT, LTTS), not a generic demo dataset — the point is to have a project
whose content the author actually understands cold, not just the RAG
plumbing around it.

Built to demonstrate the core of a Data Scientist / AIM role at Citi:
semantic search, a RAG framework, an agentic layer (LangChain), FastAPI
serving, and basic retrieval evaluation.

## Why this exists

This repo is intentionally not "LangChain + Chroma + OpenAI, wired
together and hoped for the best." Every component is behind a small
interface with at least two implementations — a real one and a
dependency-free/offline one — specifically so the pipeline's *logic* (not
just its plumbing) is unit-tested and provably correct, independent of
whether you have an API key, a GPU, or even internet access at the
moment. See `docs/ARCHITECTURE.md` for the reasoning behind every one of
these choices, and `docs/INTERVIEW_NOTES.md` for how each JD requirement
maps to a specific piece of this code.

| Layer | Real implementation | Offline/test implementation |
|---|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) — genuine semantic embeddings | `TfidfEmbedder` (scikit-learn) — keyword-overlap, no download, no network |
| Vector store | Hand-rolled NumPy cosine-similarity store (exact search) | *(same — see docs/ARCHITECTURE.md for why this isn't Chroma/FAISS)* |
| LLM | OpenAI or Anthropic chat API | `MockBackend` — deterministic, no API key, no cost |
| Agent | `src/agent_langchain.py` — LangChain `AgentExecutor` | `src/agent.py` — hand-rolled ReAct loop, fully unit-tested |

## Project layout

```
data/docs/            the knowledge base (6 markdown documents)
src/
  chunking.py          paragraph/sentence-aware text chunking
  embeddings.py         pluggable embedder: sentence-transformers | tfidf
  vectorstore.py        NumPy cosine-similarity vector store
  llm.py                 pluggable chat backend: openai | anthropic | mock
  rag_chain.py           retrieval + prompt construction + grounded answer
  agent.py                hand-rolled ReAct agent (search_knowledge_base, calculator tools)
  agent_langchain.py      the same agent built with LangChain's AgentExecutor
  api.py                   FastAPI app: /health, /query
scripts/
  build_index.py          CLI: chunk + embed + index data/docs/
  ask.py                    CLI: ask a question (--mode rag|agent|agent-langchain)
eval/
  eval_questions.json      12 hand-labeled retrieval questions
  eval_retrieval.py         hit-rate@k / MRR evaluation
tests/                    pytest suite for chunking, vectorstore, rag_chain, agent
docs/
  ARCHITECTURE.md          design decisions and trade-offs, written out
  INTERVIEW_NOTES.md        JD-to-code mapping + likely interview questions
```

## Setup

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # installs everything, including dev deps (pytest)
cp .env.example .env         # defaults to LLM_BACKEND=mock, EMBEDDER_BACKEND=sentence-transformers
```

If you don't have (or don't want to spend) an OpenAI/Anthropic key yet,
leave `LLM_BACKEND=mock` — the whole pipeline runs end to end, the answer
just comes back clearly labeled as a mock response instead of a real
model completion. Switch to `openai` or `anthropic` in `.env` (and set the
matching API key) whenever you want real generated answers.

## Build the index

```bash
uv run python scripts/build_index.py
```

This reads every `.md` file in `data/docs/`, chunks it, embeds every
chunk, and writes the index to `index/` (gitignored — rebuild it locally,
don't commit vectors).

Use `--embedder tfidf` to build the dependency-free version instead (no
model download, no network) — useful for a quick sanity check or if
you're offline.

## Ask a question

```bash
# straight RAG
uv run python scripts/ask.py "How does AVTP carry accelerator pedal angle across domains?"

# the hand-rolled agent (routes to search_knowledge_base or calculator)
uv run python scripts/ask.py --mode agent "What's 1722 * 2 + 8?"

# the LangChain agent (needs LLM_BACKEND=openai or anthropic)
uv run python scripts/ask.py --mode agent-langchain "Why use Xen instead of a single guest OS?"
```

## Run the API

```bash
uv run uvicorn src.api:app --reload
```

```bash
curl -X POST localhost:8000/query \
  -H "content-type: application/json" \
  -d '{"question": "Why is gRPC used instead of REST for the ROS2-cloud bridge?", "mode": "rag"}'
```

## Run the tests

```bash
uv run pytest -v
```

The test suite covers chunking, the vector store, the RAG prompt/answer
pipeline (using `TfidfEmbedder` + `MockBackend`, so it needs no network,
no API key, and no model download), and the hand-rolled agent's
tool-routing logic (using a scripted fake LLM so the loop is
deterministic). `src/agent_langchain.py` is intentionally not covered —
see `docs/ARCHITECTURE.md` for why.

## Evaluate retrieval

```bash
uv run python eval/eval_retrieval.py
```

Reports `hit_rate@k` and `MRR@k` against `eval/eval_questions.json` — 12
hand-labeled questions, each with the document that should be retrieved.
Even the dependency-free TF-IDF backend gets `hit_rate@3 = 1.00` on this
corpus (small, topically distinct documents), which is itself a useful
talking point: TF-IDF is a strong baseline on a corpus this size and
distinctness, and the real payoff of dense semantic embeddings shows up
on paraphrased or synonym-heavy queries and on corpora too large or
overlapping for keyword matching to separate cleanly — see
`docs/ARCHITECTURE.md`.

## A note on how this was built

This repo was scaffolded in a sandboxed environment with no PyPI access
(no way to install langchain, chromadb, sentence-transformers, fastapi,
etc.), which is exactly why the project is structured the way it is:
every piece that could be tested without those dependencies (chunking,
the vector store, TF-IDF-backed retrieval, the RAG prompt pipeline, the
hand-rolled agent's tool-routing loop) *was* tested, end-to-end, against
this repo's actual corpus, including a full run of
`scripts/build_index.py` + `eval/eval_retrieval.py`. The pieces that need
those packages (`SentenceTransformerEmbedder`, `OpenAIBackend`,
`AnthropicBackend`, `agent_langchain.py`, `api.py`) are written correctly
against their current, documented APIs, but should be run and sanity
checked on a normal machine with internet access before a live demo —
`uv sync` there, then walk through the Setup section above.
