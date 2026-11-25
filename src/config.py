"""Central configuration.

Kept as one small module, read once, rather than scattering `os.getenv`
calls through the codebase — every other module imports `SETTINGS` from
here instead of touching the environment directly. That's the whole
reason this file exists: one place to see every knob the project has.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv isn't installed yet (e.g. before `uv sync`). Fall back
    # to whatever is already in the environment rather than crashing.
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "docs"
INDEX_DIR = ROOT_DIR / "index"


@dataclass(frozen=True)
class Settings:
    # --- chunking ---
    chunk_size: int = 800  # characters, not tokens -- see docs/ARCHITECTURE.md
    chunk_overlap: int = 120

    # --- retrieval ---
    top_k: int = 4

    # --- backends ---
    embedder_backend: str = os.getenv("EMBEDDER_BACKEND", "sentence-transformers")
    llm_backend: str = os.getenv("LLM_BACKEND", "mock")

    # --- models ---
    sentence_transformer_model: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    # --- paths ---
    data_dir: Path = DATA_DIR
    index_dir: Path = INDEX_DIR


SETTINGS = Settings()


def write_index_config(index_dir: Path, **kwargs) -> None:
    """Record how the index at `index_dir` was built (embedder backend,
    model name, chunk size/overlap) so anything loading the index later
    -- the API, the eval script, scripts/ask.py -- uses the SAME embedder
    the index was built with, instead of silently trusting whatever
    EMBEDDER_BACKEND happens to be set in the environment at load time.
    A mismatch there (e.g. index built with tfidf, loaded with
    sentence-transformers) would produce nonsense: the query vector and
    the stored vectors wouldn't even have the same dimensionality.
    """
    import json

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index_config.json").write_text(json.dumps(kwargs, indent=2), encoding="utf-8")


def read_index_config(index_dir: Path) -> dict:
    import json

    path = Path(index_dir) / "index_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No index_config.json in {index_dir} -- run scripts/build_index.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
