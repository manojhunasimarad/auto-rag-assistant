# Architecture and design decisions

This document is the "why", not the "what" — the README covers what each
file does; this covers why it's built that way, including the trade-offs
deliberately made and what a production version would add. Every section
here is meant to be something you can talk through out loud in an
interview.

## 1. The pipeline, end to end

```
data/docs/*.md
      |  chunking.py: paragraph-aware split, ~800 chars, 120 char overlap
      v
   Chunk[]  (text, source filename, chunk_index)
      |  embeddings.py: Embedder.fit() + .embed()
      v
   vectors (n_chunks x dim)
      |  vectorstore.py: VectorStore.add()  (L2-normalized, stored as one NumPy matrix)
      v
   index/  (vectors.npy + meta.json + index_config.json)

query
      |  embeddings.py: Embedder.embed([query])
      v
   query_vector
      |  vectorstore.py: VectorStore.query()  (cosine similarity, top-k)
      v
   QueryResult[]  (id, score, document, metadata)
      |  rag_chain.py: build_prompt()  (numbered, cited context blocks)
      v
   (system_prompt, user_prompt)
      |  llm.py: LLMBackend.chat()
      v
   answer text  ->  rag_chain.RagAnswer(answer, sources, retrieved)
```

The agent (`agent.py`) wraps this same pipeline as a `search_knowledge_base`
tool, plus a `calculator` tool, inside a Thought/Action/Observation loop —
see section 5.

## 2. Chunking: character-based, paragraph-aware, with overlap

**Decision**: split on paragraph boundaries (falling back to sentence
boundaries for an over-long paragraph), pack greedily up to ~800
characters, keep a 120-character overlap between consecutive chunks.

**Why character count, not token count**: a token-exact chunker needs a
tokenizer tied to whichever embedding/LLM model you're using — that's
more precise but adds a dependency and couples the chunker to one model's
tokenizer. Character count is model-agnostic and, for English technical
prose, tracks token count closely enough (~4 chars/token) that it doesn't
matter at this corpus size. Trade-off worth stating: this stops being
"close enough" once you're packing chunks right up against a small
context window, where you'd want the real tokenizer.

**Why paragraph/sentence-aware instead of a hard character cutoff**:
a chunk that's cut mid-sentence produces a worse embedding (it's a
partial thought, not a coherent unit of meaning) and a worse prompt for
the LLM to reason over. Respecting natural boundaries costs a little
chunk-size variance in exchange for chunks that are always coherent
units.

**Why overlap**: a fact that happens to sit right at a chunk boundary
would otherwise be split across two chunks and might not be fully
retrievable by either. 120 characters (about 15% of chunk size) is enough
to carry a sentence or two of context across the boundary without
meaningfully increasing index size.

**What a bigger system would add**: chunk size and overlap here are fixed
constants chosen by inspection of this corpus. A production system tunes
these against the actual retrieval eval (see section 6) rather than by
guesswork, and often uses a *hierarchical* strategy — small chunks for
precise retrieval, but expanding back out to the parent section/document
when constructing the prompt, so the LLM gets both precision and context.

## 3. Embeddings: pluggable, with a real semantic backend and an offline fallback

**Decision**: one `Embedder` interface, two implementations —
`SentenceTransformerEmbedder` (`all-MiniLM-L6-v2`, 384-dim dense vectors)
as the real backend, `TfidfEmbedder` (scikit-learn) as a dependency-free
fallback used by the automated tests and available via `--embedder tfidf`
on the CLI.

**Why MiniLM specifically**: it's small enough to run on CPU with no GPU
requirement, fast enough for interactive queries, and its retrieval
quality on short technical passages is good relative to its size — a
reasonable default when you don't have a reason yet to pay for a larger
embedding model or an API-based one (OpenAI/Cohere embeddings). The
trade-off: a bigger model (or a domain-finetuned one) would very likely
retrieve better on harder, more paraphrased queries; MiniLM is the "good
default," not the ceiling.

**Why TF-IDF is in the codebase at all**: TF-IDF is a *keyword-overlap*
representation, not a semantic one — "throttle position" and "accelerator
angle" would not score as similar even though they mean nearly the same
thing here. It's included specifically so the full pipeline can be
unit-tested (and run in CI) without a model download or network access.
This is a genuinely common pattern in real ML repos: keep the expensive/
networked dependency behind an interface so tests don't need it. It also
produces an interesting empirical result on this project's own eval (see
section 6): on a small corpus of topically distinct documents, TF-IDF
does about as well as dense embeddings, because there's little vocabulary
overlap between the documents doing the confusing. That gap would open up
fast on paraphrased queries or a larger, more overlapping corpus — which
is exactly why production RAG systems don't rely on TF-IDF alone.

**What a bigger system would add**: hybrid retrieval — combine the dense
score with a sparse (BM25/TF-IDF) score, since they fail on different
kinds of queries (dense embeddings can miss exact identifiers/error codes
that a keyword match would catch instantly; keyword search misses
paraphrases and synonyms that dense embeddings catch easily) — and
usually a reranking step (a cross-encoder that re-scores the top ~20
candidates more expensively than the first-pass retriever) before the
final top-k goes into the prompt.

## 4. Vector store: hand-rolled NumPy, not Chroma/FAISS/Pinecone

**Decision**: a small class doing brute-force cosine similarity over one
NumPy matrix (`vectorstore.py`), persisted as a `.npy` file plus a JSON
metadata sidecar.

**Why not a real vector database for a corpus this size**: for a few
hundred to a few thousand chunks, brute-force search over a NumPy matrix
*is* exact nearest-neighbor search, adds zero extra dependencies, needs
no running service, and answers a query in low single-digit milliseconds.
It's also trivially unit-testable (`tests/test_vectorstore.py`) with
nothing but numpy.

**Where this breaks down, and what you'd reach for instead**: brute-force
search is O(n) per query in the number of stored vectors. Once a corpus
gets into the hundreds of thousands to millions of chunks, that stops
being fast enough, and you want an approximate nearest-neighbor (ANN)
index — HNSW (what Chroma and many others use) or IVF-based indexes,
trading a small amount of recall for sub-linear query time. Chroma, FAISS,
or a managed vector DB (Pinecone, pgvector, etc.) also add things a
hand-rolled store doesn't: persistence at scale, filtering by metadata at
query time, and horizontal scaling. The `VectorStore` interface here
(`add`, `query`, `save`, `load`) is deliberately narrow so a Chroma-backed
implementation could be swapped in without touching `rag_chain.py` or
`agent.py` at all.

## 5. Two agent implementations, on purpose

**Decision**: `agent.py` is a hand-rolled ReAct-style loop (system prompt
describing tools + response format, parse the model's Thought/Action/
Action Input, execute the named tool, feed the Observation back, repeat
until Final Answer). `agent_langchain.py` is the same two tools
(`search_knowledge_base`, `calculator`) wired up with LangChain's
`create_tool_calling_agent` + `AgentExecutor`.

**Why both exist**: the JD explicitly names LangChain, so
`agent_langchain.py` is the direct answer to that requirement. But this
sandboxed environment had no PyPI access to actually install and run
LangChain while building this repo, so `agent.py` is what's provably
correct — it's covered by `tests/test_agent.py` using a scripted fake LLM
backend that returns a fixed sequence of responses, which lets the
tool-routing logic (does the agent call `calculator` for an arithmetic
question and `search_knowledge_base` for a domain question? does it
terminate on `Final Answer`? does it stop instead of looping forever if
the model never produces one?) be tested deterministically with zero API
cost and zero flakiness.

There's also a real pedagogical point here, not just a sandboxing
workaround: writing the loop by hand makes explicit everything
`AgentExecutor` normally hides — an "agent" is a prompt describing tools
and a response grammar, a parser, and a loop. Being able to explain that
mechanism directly, rather than "LangChain handles it," is usually the
difference between a shallow and a strong answer to an agent-architecture
question in an interview.

**Safety note on the calculator tool**: its input ultimately comes from
LLM output, so it cannot be `eval()`'d directly. `calculator_tool` parses
the expression with `ast.parse` and walks the resulting AST through a
small allow-list of node types (`Constant`, `BinOp`/`UnaryOp` for
`+ - * / **`), rejecting anything else (names, calls, attributes,
imports) — the classic "restricted AST evaluator" pattern for taking
untrusted numeric expressions safely.

## 6. Evaluation: retrieval hit-rate/MRR, not answer-quality scoring

**Decision**: `eval/eval_retrieval.py` checks, for 12 hand-labeled
questions, whether the *document* that should answer each question shows
up in the top-k retrieved chunks (`hit_rate@k`), and how highly it's
ranked when it does (`MRR@k`, mean reciprocal rank).

**Why retrieval eval and not answer eval**: if retrieval pulls the wrong
document, the generated answer is capped in quality no matter how good
the prompt or the model is — retrieval is the bottleneck to fix first,
and it's the one part of a RAG system you can evaluate cheaply and
objectively (a document either is or isn't the right source) without
needing human judgment or an LLM-as-judge setup. Answer-quality
evaluation (is the generated text actually correct, well-cited, and not
hallucinated) is real and important but needs either a human rater or a
second LLM call to judge the first one's output — both reasonable next
steps, both out of scope for a one-week MVP.

**What the numbers on this repo's own corpus mean**: both the TF-IDF and
sentence-transformers backends score close to `hit_rate@3 = 1.0` here, for
the reason given in section 3 — a small corpus of topically distinct
documents doesn't stress-test the *semantic* part of semantic search.
Being able to say exactly why a benchmark result doesn't mean what it
might look like it means is worth more in an interview than the number
itself.

## 7. Prompting and hallucination mitigation

**Decision**: `rag_chain.SYSTEM_PROMPT` instructs the model to answer only
from the numbered context chunks, cite a chunk number for every claim,
and explicitly say when the context doesn't have enough information
rather than guessing.

This is the standard first line of defense against hallucination in a
RAG system — it does not make hallucination impossible (nothing short of
a verification/critic step over the generated answer really does), but
grounding + forced citation + an explicit "say I don't know" instruction
measurably reduces it and, just as importantly, makes it *checkable*: a
citation that doesn't actually support its claim is a much easier bug to
catch (automatically or by a human) than an uncited hallucination.

## 8. What's deliberately out of scope for this MVP

Worth naming these directly rather than pretending the project is
"done" — a production version of this would add: authentication/rate
limiting on the API, structured logging and tracing per request (which
chunks were retrieved, what the model actually said, latency), a
reranking stage, hybrid dense+sparse retrieval, an answer-quality eval
loop (human or LLM-as-judge), conversation memory for multi-turn
follow-ups, streaming responses from the API instead of blocking on the
full generation, and swapping the hand-rolled vector store for a real
ANN-backed one once the corpus stops being "a handful of documents."
