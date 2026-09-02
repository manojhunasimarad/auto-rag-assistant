"""Pluggable LLM chat backends.

Same idea as embeddings.py: one narrow interface (`chat(system, user) ->
str`), three implementations, selected by the `LLM_BACKEND` env var. This
is what lets rag_chain.py and agent.py be written once and swapped between
"call a real model" and "run with zero network access, zero API cost, zero
API key" just by changing one setting -- which matters a lot for a repo
someone else (an interviewer) is going to clone and try to run without
necessarily wanting to hand it an API key first.
"""
from __future__ import annotations

import textwrap
from typing import Protocol


class LLMBackend(Protocol):
    def chat(self, system: str, user: str) -> str: ...


class MockBackend:
    """Deterministic, offline, extractive 'answer'.

    Doesn't call any model. It just formats the retrieved context back to
    the user with a clear label that it's a mock response, so:
      - the full pipeline (retrieve -> prompt -> "generate" -> return with
        citations) can be run and unit-tested with no API key and no
        network, including in CI;
      - anyone skimming the output can immediately tell this came from the
        mock backend rather than mistaking a canned answer for a real one.
    """

    def chat(self, system: str, user: str) -> str:
        # `user` is the fully-built RAG/agent prompt, which already
        # contains the numbered, cited context chunks -- see
        # rag_chain.build_prompt(). We don't re-parse it; the mock backend
        # is intentionally dumb, it just proves the plumbing works.
        preview = textwrap.shorten(user, width=600, placeholder=" [...]")
        return (
            "[MOCK BACKEND -- no LLM was called; set LLM_BACKEND=openai or "
            "anthropic in .env for a real answer]\n\n"
            f"Prompt the real model would have received:\n{preview}"
        )


class OpenAIBackend:
    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set (required for LLM_BACKEND=openai)")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the `openai` package is not installed. Run `uv sync` on a "
                "machine with internet access, or set LLM_BACKEND=mock."
            ) from exc

        self._model = model
        self._client = OpenAI(api_key=api_key)

    def chat(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""


class AnthropicBackend:
    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set (required for LLM_BACKEND=anthropic)"
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the `anthropic` package is not installed. Run `uv sync` on a "
                "machine with internet access, or set LLM_BACKEND=mock."
            ) from exc

        self._model = model
        self._client = Anthropic(api_key=api_key)

    def chat(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0.1,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def get_llm_backend(
    backend: str, openai_model: str, anthropic_model: str, openai_key: str, anthropic_key: str
) -> LLMBackend:
    if backend == "mock":
        return MockBackend()
    if backend == "openai":
        return OpenAIBackend(openai_model, openai_key)
    if backend == "anthropic":
        return AnthropicBackend(anthropic_model, anthropic_key)
    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (expected 'mock', 'openai', or 'anthropic')")
