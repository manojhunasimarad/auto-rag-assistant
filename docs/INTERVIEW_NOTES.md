# Interview notes — Citi AIM Data Scientist JD mapping + Q&A prep

Personal prep notes, kept in the repo deliberately: an interviewer who
looks at this repo should see not just working code but evidence you
understand *why* it's built this way and where its limits are.

## 1. JD bullet -> where it lives in this repo

| JD requirement | Where it's demonstrated | Honest gap / what to say if pushed |
|---|---|---|
| "Development of solutions using RAG framework for building Gen AI solutions" | `src/rag_chain.py`, `src/embeddings.py`, `src/vectorstore.py` — full ingest -> chunk -> embed -> retrieve -> cite -> generate pipeline | none — this is the core of the project |
| "Development of search techniques / esp semantic search" | `src/embeddings.py` (`SentenceTransformerEmbedder`) + `src/vectorstore.py` (cosine similarity) | Only dense retrieval is implemented; no hybrid/BM25 fusion or reranking yet — named explicitly as a next step in `docs/ARCHITECTURE.md` §3, §8 |
| "Development of Agentic solution using frameworks like Langchain / Google Agent space" | `src/agent_langchain.py` (LangChain `AgentExecutor`) and `src/agent.py` (hand-rolled ReAct loop, same tools) | Google ADK/Agent Space specifically isn't used — say plainly you built the equivalent pattern (tool-calling agent loop) in LangChain and can pick up ADK's API quickly since the underlying concepts (tools, planner, executor loop) transfer directly |
| "AI – Familiarity with ADK, Fast API... Prompt engineering, Deep learning models, LLMs" | FastAPI: `src/api.py`. Prompt engineering: `rag_chain.SYSTEM_PROMPT`, `agent._AGENT_SYSTEM_TEMPLATE`, both with explicit design rationale in ARCHITECTURE.md | ADK gap as above; deep learning models beyond a pretrained sentence embedding model aren't part of this project — draw on MTech coursework if asked to go deeper |
| "Rigorously evaluate model performance... cross-validation, A/B testing, ROC/AUC, F1-score" | `eval/eval_retrieval.py` — hit-rate@k and MRR@k against a held-out labeled question set, same *principle* (quantitative eval against ground truth, not vibes) applied to retrieval | This project's task (retrieval) isn't classification, so ROC/AUC/F1 don't directly apply here — be ready to define them cleanly from classic ML knowledge (see §3 below) rather than pointing back at this repo for those specific metrics |
| "Programming: Foundational knowledge in Python (Pandas, NumPy, Scikit-learn)" | `numpy` in `vectorstore.py`, `scikit-learn`'s `TfidfVectorizer` in `embeddings.py` | pandas isn't used here since there's no tabular data in this project — mention it's core to your MTech coursework work |
| "Version control... Git" | This repo itself, with incremental, buildable-at-each-step commit history | — |
| PySpark / big data technologies, classic ML (classification/regression/clustering), SQL, data visualization | Not part of this project | Say directly: this project was scoped to demonstrate the GenAI/RAG/agent bullets specifically, since that's the newest and most differentiating part of the JD; classic ML, SQL, and PySpark are things you'd bring from coursework/prior projects, not something you needed to prove here in a week |

**Delivering that gap-honesty well matters as much as the code**: naming
what's out of scope, and why, before being asked reads as engineering
judgment. Getting caught overclaiming what a demo project covers reads
the opposite way.

## 2. Likely questions about THIS project, with grounded answers

**"Walk me through your RAG pipeline."**
Documents get chunked paragraph-aware with overlap (`chunking.py`), each
chunk gets embedded (`embeddings.py`, MiniLM by default), stored in a
cosine-similarity vector index (`vectorstore.py`). At query time the
question gets embedded the same way, top-k chunks come back, and
`rag_chain.build_prompt` builds a system+user prompt that numbers each
chunk, forces the model to cite chunk numbers per claim, and tells it to
say "I don't know" rather than guess when context is insufficient. The
LLM call and the answer (with deduplicated source filenames) come back as
one `RagAnswer`.

**"Why that chunk size / chunking strategy?"**
~800 characters, paragraph-aware with sentence-level fallback for
over-long paragraphs, 120-char overlap. Character-based instead of
token-based because it's model-agnostic and close enough for this corpus
(~4 chars/token for English prose); paragraph-aware because cutting
mid-sentence hurts both the embedding quality and the LLM's ability to
reason over the chunk. Full reasoning in `docs/ARCHITECTURE.md` §2.

**"How do you evaluate retrieval quality?"**
`eval/eval_retrieval.py` — 12 hand-labeled (question, expected source
document) pairs, computing hit-rate@k (did the right doc show up in the
top k) and MRR@k (how highly ranked was it). It's cheap, objective, and
catches the highest-leverage failure mode first: if retrieval is wrong,
no amount of prompt engineering fixes the generated answer.

**"How do you prevent hallucination?"**
Grounding + forced citation + an explicit "say you don't know" instruction
in the system prompt (`rag_chain.SYSTEM_PROMPT`). It's not a guarantee —
nothing short of a verification/critic pass over the answer really is —
but it's the standard first line of defense, and forced citation makes
hallucination *checkable*, because an uncited or wrongly-cited claim is
much easier to catch than a bare one.

**"Why your vector store instead of Chroma/FAISS/Pinecone?"**
For a corpus this size (tens of documents, tens of chunks per doc),
brute-force cosine similarity over one NumPy matrix is exact nearest-
neighbor search, has zero extra dependencies, and answers in milliseconds
— and it's fully unit-testable with nothing but numpy. It stops being the
right choice once you're at hundreds of thousands to millions of vectors,
where you need an approximate nearest-neighbor index (HNSW/IVF) for
sub-linear query time — that's what Chroma/FAISS/a managed vector DB
give you, plus persistence at scale and metadata filtering. The
`VectorStore` interface here is narrow on purpose so a Chroma-backed
implementation could replace it without touching any calling code.

**"What's the difference between semantic and keyword search, concretely,
in your project?"**
The repo has both: `SentenceTransformerEmbedder` for real dense semantic
embeddings, `TfidfEmbedder` (sparse, keyword-overlap) as a dependency-free
fallback. On this project's own corpus they score about the same on
retrieval eval, because the six documents are topically distinct enough
that keyword overlap alone separates them well — that gap would open up
on paraphrased queries ("throttle angle" vs "accelerator pedal angle") or
a larger, more overlapping corpus, which is the actual case for semantic
embeddings in production.

**"How does your agent decide which tool to call?"**
The agent (`agent.py`) gets a system prompt listing its tools
(`search_knowledge_base`, `calculator`) and a strict Thought/Action/Action
Input/Observation format; the LLM itself decides, per step, which tool
(if any) is relevant, and the loop parses that decision, executes the
tool, and feeds the result back as an Observation until the model emits a
Final Answer. `agent_langchain.py` does the same thing through LangChain's
tool-calling agent, which does essentially the same loop with more
abstraction. Good demo question to actually run: "what's 1722 * 2 + 8?"
routes to the calculator; a domain question routes to
search_knowledge_base.

**"What would you change to make this production-ready?"**
Auth/rate-limiting on the API, structured per-request logging (what was
retrieved, what the model said, latency), hybrid dense+sparse retrieval
with reranking, an answer-quality eval loop (human or LLM-as-judge, not
just retrieval hit-rate), multi-turn conversation memory, streaming
responses instead of blocking on full generation, and a real ANN-backed
vector store once the corpus grows past what brute-force search handles
comfortably. All named explicitly in `docs/ARCHITECTURE.md` §8 — worth
skimming right before the interview.

**"Tell me about a time you significantly improved something's
performance/cost."** *(behavioral, but answerable from real experience,
and — deliberately — also documented as one of this project's own corpus
files, `data/docs/build_optimization_incredibuild.md`)*
At LTTS, a client's embedded build was taking 24+ hours on a single AWS
EC2 build host. Brought in IncrediBuild to distribute the compilation
graph across a pool of EC2 helper instances — the real wins were cleaning
up undeclared build-graph dependencies first (distributing a build with
hidden dependencies just makes flaky failures show up faster), sizing the
helper pool against the build graph's actual parallelism rather than just
adding machines, and getting helper/source locality right on EC2. Result:
24+ hours down to about 3 hours, demonstrated to the client as a direct
before/after comparison.

## 3. Classic ML/stats definitions worth having crisp (JD explicitly lists these)

- **Precision / Recall / F1**: precision = of what you predicted
  positive, how much was actually positive; recall = of what was
  actually positive, how much did you catch; F1 = harmonic mean of the
  two, useful when you care about both and classes are imbalanced.
- **ROC/AUC**: ROC plots true-positive rate vs. false-positive rate across
  classification thresholds; AUC is the area under that curve — a
  threshold-independent measure of how well the model ranks positives
  above negatives (0.5 = random, 1.0 = perfect separation).
- **Cross-validation**: split data into k folds, train on k-1 and
  validate on the held-out fold, rotate, average the metric — reduces
  variance in your performance estimate versus a single train/test split,
  especially on smaller datasets.
- **A/B testing**: randomly split users/traffic between a control and a
  treatment, measure the metric that matters, use a significance test to
  decide whether the observed difference is real or noise — the standard
  way to validate a model change actually helps in production, not just
  offline.

## 4. Day-of logistics

- If live-demoing, run `uv run python scripts/ask.py --mode agent "..."`
  with `LLM_BACKEND` set to a real key beforehand and confirm it works on
  the actual interview machine/network *before* the call — see the "note
  on how this was built" in the README for why that verification step
  matters here specifically.
- If the network or API key fails live, `LLM_BACKEND=mock` still runs the
  full pipeline end-to-end and visibly proves the plumbing works — say so
  explicitly rather than letting a failed API call look like broken code.
- Have `eval/eval_retrieval.py`'s output ready to show or paste — a real
  number (hit_rate@3 = 1.00 on this corpus, with the honest caveat about
  what that number does and doesn't prove) is stronger than a claim.
