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

BUG FOUND AND FIXED (documented as part of the learning process):
The original version of this file extracted mentioned cities by iterating
over a fixed known_cities list and checking membership, e.g.:
    mentioned_cities = [c for c in known_cities if c.lower() in input_lower]
This meant the ORDER of mentioned_cities always matched known_cities's
internal order (Mumbai, Delhi, Bangalore...), never the order the user
actually typed them in. For a query like "compare Delhi and Mumbai",
Mumbai would always be processed first regardless of what the user said,
purely because it happens to be first in a hardcoded list.

This matters because check_sequence in the evaluator is meant to verify
the agent's tool-call order reflects correct reasoning. A sequence check
built on top of accidental, input-independent ordering would pass or fail
for the wrong reasons -- it would be testing "does this match
known_cities's list order" rather than "did the agent respond in a way
that reflects the user's actual request."

FIX: cities are now extracted by finding each one's position in the
input string (str.find), then sorted by that position. Order now
genuinely reflects what the user typed.
"""

from agent.tools import get_weather, get_temperature, get_population

KNOWN_CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune"]


class MockAgent:
    """
    Fakes tool-calling behavior without an LLM. Produces the same trace
    shape a real agent would: a list of tool_name/tool_input/tool_output
    dicts, plus a final answer string.
    """

    def run(self, user_input: str) -> dict:
        trace = []
        input_lower = user_input.lower()

        mentioned_cities = self._extract_cities_in_order(input_lower)

        if not mentioned_cities:
            return {
                "trace": trace,
                "final_answer": "I could not identify a city in your question.",
            }

        is_comparison = any(
            keyword in input_lower for keyword in ["compare", "difference", " vs ", "versus"]
        )

        if is_comparison and len(mentioned_cities) >= 2:
            trace = self._handle_temperature_comparison(mentioned_cities[:2])
            final_answer = self._build_comparison_answer(trace, mentioned_cities[:2])
            return {"trace": trace, "final_answer": final_answer}

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

    def _extract_cities_in_order(self, input_lower: str) -> list:
        """
        Finds which known cities appear in the input, ordered by their
        actual position in the string -- not by known_cities's internal
        list order. This is the fix described in the module docstring.
        """
        found = []
        for city in KNOWN_CITIES:
            position = input_lower.find(city.lower())
            if position != -1:
                found.append((position, city))
        found.sort(key=lambda pair: pair[0])
        return [city for _, city in found]

    def _handle_temperature_comparison(self, cities: list) -> list:
        """
        Calls temperature_tool for each of the two cities, in the order
        the user mentioned them. This is the multi-step case that makes
        check_sequence meaningful -- there are now two calls to the same
        tool, and their order should reflect the user's input order.
        """
        trace = []
        for city in cities:
            result = get_temperature(city)
            trace.append({
                "tool_name": "temperature_tool",
                "tool_input": {"city": city},
                "tool_output": result,
            })
        return trace

    def _build_comparison_answer(self, trace: list, cities: list) -> str:
        """
        Builds a final answer that states both temperatures AND the
        computed difference. This gives check_final_answer_uses_tool_output
        something concrete to verify beyond a single restated number --
        it should find both individual tool outputs represented, since
        the difference alone would not "ground" the answer in the raw
        tool outputs.
        """
        if len(trace) < 2:
            return "Could not retrieve temperatures for both cities."

        temp1 = trace[0]["tool_output"]
        temp2 = trace[1]["tool_output"]
        city1 = cities[0]
        city2 = cities[1]
        difference = abs(temp1 - temp2)

        return (
            f"{city1} is {temp1} degrees Celsius and {city2} is {temp2} "
            f"degrees Celsius. The difference is {difference} degrees."
        )

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