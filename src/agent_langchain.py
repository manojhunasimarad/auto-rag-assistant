"""LangChain-based agent.

The JD explicitly calls out "Agentic solution using frameworks like
Langchain / Google Agent space" -- this file is the direct answer to that
line. It's functionally equivalent to the hand-rolled ReAct loop in
`src/agent.py` (same two tools: search_knowledge_base, calculator), but
built on LangChain's `create_tool_calling_agent` + `AgentExecutor`
instead of a manual prompt/parse/loop.

Why this file is separate from src/agent.py, and not the only agent
implementation
-------------------------------------------------------------------------
This repo was scaffolded in a sandbox with no PyPI access, so LangChain
could never actually be installed or run here -- everything in this file
is written correctly against the current LangChain API from knowledge, not
verified by execution in this repo's own CI. `src/agent.py` is the
dependency-free version that IS covered by tests/test_agent.py, so the
project has at least one agent implementation that's provably correct
rather than "looks right." Being able to explain that distinction --
what's tested vs. what's written-but-unverified, and why -- is itself a
reasonable thing to bring up if an interviewer asks "did you test this."

Run it (on a machine with internet + an API key):
    uv run python scripts/ask.py --mode agent-langchain "your question"
"""
from __future__ import annotations

from src.agent import calculator_tool
from src.rag_chain import RagPipeline


def build_langchain_agent(rag: RagPipeline, llm_backend: str, model: str, api_key: str):
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the in-vehicle systems engineering knowledge base (AUTOSAR
        Adaptive, IEEE1722/AVTP, hypervisor bring-up, ROS2-cloud bridging,
        build optimization). Input: a natural-language query."""
        results = rag.retrieve(query)
        if not results:
            return "No relevant documents found."
        return "\n".join(
            f"[{i}] ({r.metadata.get('source', 'unknown')}): {r.document[:300]}"
            for i, r in enumerate(results, start=1)
        )

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression, e.g. '3 * (4 + 2)'."""
        return calculator_tool(expression)

    tools = [search_knowledge_base, calculator]

    if llm_backend == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.1)
    elif llm_backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, api_key=api_key, temperature=0.1)
    else:
        raise ValueError(
            "The LangChain agent needs a real tool-calling model: set "
            "LLM_BACKEND=openai or LLM_BACKEND=anthropic. (The mock backend "
            "can't drive LangChain's tool-calling loop -- there's no model "
            "actually deciding which tool to call.)"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a technical assistant for in-vehicle systems engineering "
                "questions. Use the available tools when they help answer the "
                "question, and cite sources by filename when you use "
                "search_knowledge_base.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
