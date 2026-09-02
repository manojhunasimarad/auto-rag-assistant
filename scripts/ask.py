#!/usr/bin/env python3
"""Ask the assistant a question from the command line.

Usage:
    uv run python scripts/ask.py "How does AVTP carry CAN signals?"
    uv run python scripts/ask.py --mode agent "What's 1722 * 2 + 8?"
    uv run python scripts/ask.py --mode agent-langchain "..."   # needs langchain installed
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import default_tools, run_agent
from src.config import SETTINGS, read_index_config
from src.embeddings import load_embedder
from src.llm import get_llm_backend
from src.rag_chain import RagPipeline
from src.vectorstore import VectorStore


def load_pipeline() -> RagPipeline:
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
    llm_backend = get_llm_backend(
        SETTINGS.llm_backend,
        SETTINGS.openai_model,
        SETTINGS.anthropic_model,
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("ANTHROPIC_API_KEY", ""),
    )
    return RagPipeline(vectorstore, embedder, llm_backend, top_k=SETTINGS.top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["rag", "agent", "agent-langchain"], default="rag")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    rag = load_pipeline()

    if args.mode == "rag":
        result = rag.answer(args.question, k=args.k)
        print("\n--- Answer ---")
        print(result.answer)
        print("\n--- Sources ---")
        for s in result.sources:
            print(f"  - {s}")

    elif args.mode == "agent":
        tools = default_tools(rag)
        result = run_agent(args.question, rag.llm_backend, tools)
        print("\n--- Agent trace ---")
        for i, step in enumerate(result.steps, start=1):
            print(f"[step {i}] tool={step.tool_used}")
            if step.observation:
                print(f"          observation={step.observation[:200]}")
        print("\n--- Final Answer ---")
        print(result.final_answer)

    else:  # agent-langchain
        from src.agent_langchain import build_langchain_agent

        executor = build_langchain_agent(
            rag,
            SETTINGS.llm_backend,
            SETTINGS.openai_model if SETTINGS.llm_backend == "openai" else SETTINGS.anthropic_model,
            os.getenv("OPENAI_API_KEY", "") if SETTINGS.llm_backend == "openai" else os.getenv("ANTHROPIC_API_KEY", ""),
        )
        result = executor.invoke({"input": args.question})
        print("\n--- Final Answer ---")
        print(result["output"])


if __name__ == "__main__":
    main()
