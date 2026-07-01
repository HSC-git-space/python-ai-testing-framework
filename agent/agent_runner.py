"""
agent_runner.py

Sets up the agent -- the "test subject" the evaluator judges.

LANGCHAIN CONCEPTS:
1. @tool wraps a plain function so the LLM knows it exists. The docstring
   is read by the LLM to decide when to call it -- write it for the model,
   not just for humans.
2. An agent = LLM + tools + a decision loop. You do not hand-code branching
   logic ("if weather question, call get_weather") -- the LLM infers it
   from tool descriptions.
3. AgentExecutor runs the loop: send prompt -> LLM decides tool or answer
   -> if tool, call it, feed result back -> repeat until final answer.
4. MOCK MODE: MockAgent fakes tool selection via keyword matching so the
   evaluator harness is fully testable offline, no API key required.
   Swapping in the real LangChain agent later is a one-line change in
   run_agent() -- the evaluator does not care which one produced the trace.
"""

from agent.tools import get_weather, get_temperature, get_population


class MockAgent:
    """
    Fakes tool-calling behavior without an LLM. Produces the same trace
    shape a real agent would: a list of tool_name/tool_input/tool_output
    dicts, plus a final answer string.
    """

    def run(self, user_input: str) -> dict:
        trace = []
        input_lower = user_input.lower()

        known_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune"]
        mentioned_cities = [c for c in known_cities if c.lower() in input_lower]

        if not mentioned_cities:
            return {
                "trace": trace,
                "final_answer": "I could not identify a city in your question.",
            }

        for city in mentioned_cities:
            if "weather" in input_lower:
                result = get_weather(city)
                trace.append({
                    "tool_name": "weather_tool",
                    "tool_input": {"city": city},
                    "tool_output": result,
                })
            if "temperature" in input_lower or "hot" in input_lower or "cold" in input_lower:
                result = get_temperature(city)
                trace.append({
                    "tool_name": "temperature_tool",
                    "tool_input": {"city": city},
                    "tool_output": result,
                })
            if "population" in input_lower or "people" in input_lower:
                result = get_population(city)
                trace.append({
                    "tool_name": "population_tool",
                    "tool_input": {"city": city},
                    "tool_output": result,
                })

        final_answer = self._build_final_answer(trace)
        return {"trace": trace, "final_answer": final_answer}

    def _build_final_answer(self, trace: list) -> str:
        if not trace:
            return "I am not sure which information you are asking for."
        parts = []
        for step in trace:
            name = step["tool_name"]
            output = step["tool_output"]
            parts.append(name + " for the requested city returned " + str(output))
        return "; ".join(parts)


def run_agent(user_input: str, use_mock: bool = True) -> dict:
    """
    Single entry point used by tests and the evaluator.

    Returns a dict shaped like:
        trace: list of tool_name/tool_input/tool_output dicts
        final_answer: str

    use_mock=True by default so the whole suite runs offline. The real
    LangChain path gets wired in once API credits are available -- same
    mock-first pattern as the judge evaluator elsewhere in this repo.
    """
    if use_mock:
        return MockAgent().run(user_input)

    raise NotImplementedError(
        "Real LangChain agent path not wired in yet. Use use_mock=True."
    )