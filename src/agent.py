"""A minimal, dependency-free ReAct-style agent.

Why hand-roll this instead of only using LangChain's AgentExecutor
----------------------------------------------------------------------
`src/agent_langchain.py` shows the same agent built with LangChain
(`create_tool_calling_agent` + `AgentExecutor`), which is what the JD
actually asks for and is the version worth demoing if the interviewer
wants to see LangChain specifically. This file exists alongside it for
three reasons that are worth being able to say out loud in an interview:

1. It has zero third-party dependencies, so it's the version that's
   actually tested in this repo (tests/test_agent.py) without needing
   LangChain installed -- which matters a lot in a sandboxed/offline dev
   environment, and is a realistic CI concern.
2. It makes the mechanics LangChain normally hides completely explicit:
   an "agent" here is nothing more than (a) a system prompt describing
   some tools and a response format, (b) a loop that asks the LLM what to
   do next, (c) a parser that turns the LLM's text into a tool call, and
   (d) code that executes that tool call and feeds the result back in.
   Being able to explain *that*, rather than "AgentExecutor handles it",
   is usually what separates a strong answer from a weak one in an
   interview about agentic systems.
3. It's a legitimate, if minimal, implementation of the ReAct pattern
   (Reason+Act, Yao et al. 2022) that the LangChain agent is itself built
   on top of.

Tools available to the agent
-----------------------------
- `search_knowledge_base`: runs retrieval + RAG generation over the
  in-vehicle systems corpus (delegates to RagPipeline).
- `calculator`: evaluates a basic arithmetic expression. Included
  specifically so there's a clear example of the agent choosing NOT to
  use the knowledge base -- a good demo question is something like
  "what's 1722 * 2 + 8?", which the agent should route to the calculator
  instead of a document search.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Callable

from src.llm import LLMBackend
from src.rag_chain import RagPipeline

MAX_STEPS_DEFAULT = 4


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


# --- calculator tool -------------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a numeric-only AST node. No names, no calls, no attributes --
    this is the whole point: it makes arbitrary code execution impossible
    even though the expression string ultimately comes from an LLM's output."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator_tool(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:  # noqa: BLE001 - tool errors get surfaced as text observations
        return f"Error evaluating '{expression}': {exc}"


def make_knowledge_base_tool(rag: RagPipeline) -> Tool:
    def _search(query: str) -> str:
        results = rag.retrieve(query)
        if not results:
            return "No relevant documents found in the knowledge base."
        lines = []
        for i, r in enumerate(results, start=1):
            snippet = r.document[:300].replace("\n", " ")
            lines.append(f"[{i}] ({r.metadata.get('source', 'unknown')}, score={r.score:.2f}): {snippet}")
        return "\n".join(lines)

    return Tool(
        name="search_knowledge_base",
        description=(
            "Search the in-vehicle systems engineering knowledge base "
            "(AUTOSAR Adaptive, IEEE1722/AVTP, hypervisor bring-up, ROS2-cloud "
            "bridging, build optimization). Input: a natural-language query."
        ),
        func=_search,
    )


def default_tools(rag: RagPipeline) -> dict[str, Tool]:
    calc = Tool(
        name="calculator",
        description="Evaluate a basic arithmetic expression, e.g. '3 * (4 + 2)'. Input: the expression.",
        func=calculator_tool,
    )
    kb = make_knowledge_base_tool(rag)
    return {t.name: t for t in (kb, calc)}


# --- the agent loop ---------------------------------------------------------

_AGENT_SYSTEM_TEMPLATE = """You are an agent that answers questions by \
reasoning step by step and, when useful, calling tools. Respond using \
EXACTLY this format, one step per turn:

Thought: <your reasoning about what to do next>
Action: <one of: {tool_names}>
Action Input: <the input to give that tool>

...or, once you have enough information to answer:

Thought: <your reasoning>
Final Answer: <your answer to the user's question>

Available tools:
{tool_descriptions}

Only ever output ONE Thought/Action/Action Input block, or ONE Thought/Final \
Answer block, per response. Do not fabricate an Observation -- that will be \
provided to you after each Action."""

_ACTION_RE = re.compile(r"Action:\s*(?P<tool>[^\n]+)\nAction Input:\s*(?P<input>.+)", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(?P<answer>.+)", re.DOTALL)


@dataclass
class AgentStep:
    thought_and_action: str  # raw LLM output for this step
    tool_used: str | None
    tool_input: str | None
    observation: str | None


@dataclass
class AgentResult:
    final_answer: str
    steps: list[AgentStep]


def _build_system_prompt(tools: dict[str, Tool]) -> str:
    tool_names = ", ".join(tools.keys())
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools.values())
    return _AGENT_SYSTEM_TEMPLATE.format(tool_names=tool_names, tool_descriptions=tool_descriptions)


def run_agent(
    query: str,
    llm_backend: LLMBackend,
    tools: dict[str, Tool],
    max_steps: int = MAX_STEPS_DEFAULT,
) -> AgentResult:
    system_prompt = _build_system_prompt(tools)
    transcript = f"Question: {query}"
    steps: list[AgentStep] = []

    for _ in range(max_steps):
        response = llm_backend.chat(system_prompt, transcript)

        final_match = _FINAL_RE.search(response)
        if final_match:
            steps.append(
                AgentStep(thought_and_action=response, tool_used=None, tool_input=None, observation=None)
            )
            return AgentResult(final_answer=final_match.group("answer").strip(), steps=steps)

        action_match = _ACTION_RE.search(response)
        if not action_match:
            # The model didn't follow the format. Treat its raw response as
            # the final answer rather than looping forever on malformed output.
            steps.append(
                AgentStep(thought_and_action=response, tool_used=None, tool_input=None, observation=None)
            )
            return AgentResult(final_answer=response.strip(), steps=steps)

        tool_name = action_match.group("tool").strip()
        tool_input = action_match.group("input").strip().splitlines()[0].strip()

        tool = tools.get(tool_name)
        if tool is None:
            observation = f"Error: unknown tool '{tool_name}'. Available: {', '.join(tools)}"
        else:
            observation = tool.func(tool_input)

        steps.append(
            AgentStep(
                thought_and_action=response,
                tool_used=tool_name,
                tool_input=tool_input,
                observation=observation,
            )
        )
        transcript += f"\n{response}\nObservation: {observation}"

    return AgentResult(
        final_answer="Agent stopped: reached max_steps without a Final Answer.",
        steps=steps,
    )
