from src.agent import Tool, calculator_tool, run_agent


class ScriptedLLM:
    """A fake LLMBackend that returns pre-written responses in sequence,
    so the agent LOOP (parse action -> run tool -> feed observation back)
    can be tested deterministically without any real model."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []  # (system, user) for each call, for assertions

    def chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0)


def _calc_tools() -> dict[str, Tool]:
    return {
        "calculator": Tool(
            name="calculator",
            description="Evaluate an arithmetic expression.",
            func=calculator_tool,
        )
    }


def test_calculator_tool_basic_arithmetic():
    assert calculator_tool("3 * (4 + 2)") == "18.0" or calculator_tool("3 * (4 + 2)") == "18"


def test_calculator_tool_rejects_unsafe_input():
    result = calculator_tool("__import__('os').system('echo hi')")
    assert result.startswith("Error")


def test_agent_answers_directly_when_llm_gives_final_answer_immediately():
    llm = ScriptedLLM(["Thought: I know this already.\nFinal Answer: 42"])
    result = run_agent("what is the answer?", llm, tools={})

    assert result.final_answer == "42"
    assert len(result.steps) == 1
    assert result.steps[0].tool_used is None


def test_agent_calls_calculator_tool_then_finishes():
    llm = ScriptedLLM(
        [
            "Thought: This needs arithmetic.\nAction: calculator\nAction Input: 6 * 7",
            "Thought: Now I have the result.\nFinal Answer: The answer is 42.",
        ]
    )
    result = run_agent("what's 6 * 7?", llm, tools=_calc_tools())

    assert result.final_answer == "The answer is 42."
    assert len(result.steps) == 2
    assert result.steps[0].tool_used == "calculator"
    assert result.steps[0].tool_input == "6 * 7"
    assert result.steps[0].observation in ("42.0", "42")

    # the second LLM call should have seen the observation fed back in
    second_call_user_message = llm.calls[1][1]
    assert "Observation:" in second_call_user_message
    assert "42" in second_call_user_message


def test_agent_reports_unknown_tool_as_observation_not_crash():
    llm = ScriptedLLM(
        [
            "Thought: try a tool that doesn't exist.\nAction: time_machine\nAction Input: 1985",
            "Thought: that failed, I'll just answer.\nFinal Answer: I cannot travel in time.",
        ]
    )
    result = run_agent("send me to 1985", llm, tools=_calc_tools())

    assert "unknown tool" in result.steps[0].observation.lower()
    assert result.final_answer == "I cannot travel in time."


def test_agent_stops_at_max_steps_without_final_answer():
    # Every response is a valid Action with no Final Answer -- the agent
    # must not loop forever.
    llm = ScriptedLLM(
        [
            "Thought: still working.\nAction: calculator\nAction Input: 1 + 1",
            "Thought: still working.\nAction: calculator\nAction Input: 2 + 2",
        ]
    )
    result = run_agent("loop forever?", llm, tools=_calc_tools(), max_steps=2)

    assert "max_steps" in result.final_answer
    assert len(result.steps) == 2
