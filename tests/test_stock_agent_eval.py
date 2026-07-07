"""
test_stock_agent_eval.py

Tests the LangGraph stock/index agent (agent.langgraph_agent_runner.run_agent)
against agent_evaluator.py's existing checks. Unlike test_agent_eval.py,
this file is NOT offline and NOT fully deterministic - decide_node makes
a real Claude call and execute_tools_node hits real live yfinance data on
every run. A real ANTHROPIC_API_KEY must be set in .env for these to pass.

Because decide_node's tool choice is a real LLM decision, some checks
below use a tolerance-scoring helper instead of a strict equality assert,
to distinguish "the code is broken" from "the LLM took a different but
still valid path." agent_evaluator.py itself is untouched and stays
strict/deterministic - this tolerance logic is local to this test file
only, not baked into the evaluator.

Numeric grounding (whether final_answer actually contains the real
number from trace) stays strict/binary throughout, since there is no
meaningful "partial credit" for citing a wrong number.
"""

from dotenv import load_dotenv

load_dotenv()

from agent.langgraph_agent_runner import run_agent
from agent.agent_evaluator import AgentEvaluator

TOLERANCE_THRESHOLD = 0.7


def score_tool_choice(actual_tool_name: str, acceptable_tools: dict) -> float:
    """
    acceptable_tools maps tool_name -> score, e.g.
    {"get_index_level": 1.0, "get_current_price": 0.5}
    Any tool_name not present in the dict scores 0.0.
    """
    return acceptable_tools.get(actual_tool_name, 0.0)


def score_arg_match(actual_value: str, expected_value: str) -> float:
    """
    Case-insensitive exact match scores 1.0, otherwise 0.0. Kept binary
    for now since ticker/index-name near-misses (e.g. wrong ticker
    entirely) should not get partial credit - there is no meaningful
    "half right" ticker.
    """
    if str(actual_value).strip().lower() == str(expected_value).strip().lower():
        return 1.0
    return 0.0


def test_sensex_now_picks_acceptable_tool():
    result = run_agent("What is the Sensex right now?")

    assert len(result["trace"]) >= 1, "Expected at least one tool call"
    entry = result["trace"][0]

    tool_score = score_tool_choice(
        entry["tool_name"],
        acceptable_tools={"get_index_level": 1.0, "get_current_price": 0.5},
    )
    assert tool_score >= TOLERANCE_THRESHOLD, (
        f"Tool choice '{entry['tool_name']}' scored {tool_score}, below threshold"
    )

    assert entry["tool_output"] != -1.0, "Tool returned failure sentinel -1.0"
    assert not str(entry["tool_output"]).startswith("ERROR"), "Tool returned an error string"


def test_sensex_now_final_answer_grounded():
    result = run_agent("What is the Sensex right now?")
    evaluator = AgentEvaluator()

    score = evaluator.check_final_answer_uses_tool_output(
        trace=result["trace"],
        final_answer=result["final_answer"],
    )

    assert score.passed, f"Grounding check failed: {score.reason}"


def test_nifty_week_ago_uses_historical_tool_and_correct_period():
    result = run_agent("What was the Nifty a week ago?")

    assert len(result["trace"]) >= 1, "Expected at least one tool call"
    entry = result["trace"][0]

    assert entry["tool_name"] == "get_historical_price", (
        f"Expected get_historical_price for a historical query, got '{entry['tool_name']}'"
    )

    ticker_score = score_arg_match(entry["tool_input"].get("ticker", ""), "^NSEI")
    period_score = score_arg_match(entry["tool_input"].get("period", ""), "1wk")

    assert ticker_score >= TOLERANCE_THRESHOLD, f"Ticker mismatch: {entry['tool_input']}"
    assert period_score >= TOLERANCE_THRESHOLD, f"Period mismatch: {entry['tool_input']}"

    assert entry["tool_output"] != -1.0, "Tool returned failure sentinel -1.0"


def test_reliance_current_price_uses_correct_ticker():
    result = run_agent("What is the current price of Reliance stock?")

    assert len(result["trace"]) >= 1, "Expected at least one tool call"
    entry = result["trace"][0]

    assert entry["tool_name"] == "get_current_price", (
        f"Expected get_current_price, got '{entry['tool_name']}'"
    )

    ticker_score = score_arg_match(entry["tool_input"].get("ticker", ""), "RELIANCE.NS")
    assert ticker_score >= TOLERANCE_THRESHOLD, f"Ticker mismatch: {entry['tool_input']}"

    assert entry["tool_output"] != -1.0, "Tool returned failure sentinel -1.0"


def test_sensex_comparison_calls_two_tools():
    result = run_agent("Compare the Sensex today versus a month ago.")

    assert len(result["trace"]) == 2, (
        f"Expected exactly 2 tool calls for a comparison query, got {len(result['trace'])}"
    )

    tool_names = [entry["tool_name"] for entry in result["trace"]]

    current_score = max(
        score_tool_choice(name, {"get_index_level": 1.0, "get_current_price": 1.0})
        for name in tool_names
    )
    historical_score = max(
        score_tool_choice(name, {"get_historical_price": 1.0})
        for name in tool_names
    )

    assert current_score >= TOLERANCE_THRESHOLD, f"No current-price-equivalent tool found in {tool_names}"
    assert historical_score >= TOLERANCE_THRESHOLD, f"No historical tool found in {tool_names}"

    for entry in result["trace"]:
        assert entry["tool_output"] != -1.0, f"Tool {entry['tool_name']} returned failure sentinel -1.0"


def test_sensex_comparison_final_answer_grounded_in_both_values():
    result = run_agent("Compare the Sensex today versus a month ago.")
    evaluator = AgentEvaluator()

    score = evaluator.check_final_answer_uses_tool_output(
        trace=result["trace"],
        final_answer=result["final_answer"],
    )

    assert score.passed, f"Grounding check failed: {score.reason}"


def test_unknown_tool_name_is_recorded_not_crashed():
    """
    Guards execute_tools_node's error-handling path directly, not
    dependent on decide_node ever actually producing a bad tool_name.
    Confirms the ERROR-string behavior agreed on earlier is real and
    testable without needing an LLM to misbehave on demand.
    """
    from agent.langgraph_agent_runner import execute_tools_node

    fake_state = {
        "user_input": "irrelevant for this test",
        "tool_calls": [{"tool_name": "get_nonexistent_tool", "tool_input": {}}],
        "trace": [],
        "final_answer": "",
    }

    result_state = execute_tools_node(fake_state)

    assert len(result_state["trace"]) == 1
    assert result_state["trace"][0]["tool_output"].startswith("ERROR")