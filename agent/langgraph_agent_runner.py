"""
langgraph_agent_runner.py

LangGraph rebuild of the agent runner, replacing MockAgent's keyword
matching with an explicit state machine driven by a real LLM decision
step. Produces the same trace shape (list of dicts with tool_name,
tool_input, tool_output) and final_answer string that agent_evaluator.py
already expects - agent_evaluator.py itself needs zero changes.

GRAPH SHAPE (linear for this first version, no branching/loops yet):

    START -> decide_node -> execute_tools_node -> final_answer_node -> END

decide_node: the only node that calls the LLM to DECIDE which tool(s)
to call and with what arguments, based on user_input. Uses Claude's
tool-calling (function-calling) API - this is the real, non-deterministic
replacement for MockAgent's keyword matching.

execute_tools_node: takes the LLM's tool decision(s) from decide_node
and actually calls the real functions in stock_tools.py, building up
trace in the exact shape agent_evaluator.py expects.

final_answer_node: calls the LLM again, this time to compose a final
answer grounded in the real tool outputs collected in trace.
"""

import os
from typing import Any, Dict, List, TypedDict

import anthropic
from langgraph.graph import StateGraph, START, END

from agent.stock_tools import get_current_price, get_historical_price, get_index_level

MODEL_NAME = "claude-sonnet-4-6"

TOOL_DEFINITIONS = [
    {
        "name": "get_current_price",
        "description": "Get the latest current closing price for a stock or index ticker, e.g. '^BSESN' for Sensex, '^NSEI' for Nifty, or 'RELIANCE.NS' for a stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The yfinance ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_historical_price",
        "description": "Get the closing price of a ticker from a period ago. period must be one of: 1d, 5d, 1wk, 1mo, 3mo, 6mo, 1y.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The yfinance ticker symbol"},
                "period": {"type": "string", "description": "How far back, e.g. '1wk' for a week ago"}
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "get_index_level",
        "description": "Get the current level of a named index by plain name, e.g. 'sensex' or 'nifty'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "index_name": {"type": "string", "description": "Plain index name, e.g. 'sensex' or 'nifty'"}
            },
            "required": ["index_name"]
        }
    },
]

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


class AgentState(TypedDict):
    user_input: str
    tool_calls: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    final_answer: str


def decide_node(state: AgentState) -> AgentState:
    """
    Calls Claude with tool definitions and lets it decide which tool(s)
    to call and with what arguments, based on user_input. Stores the
    LLM's raw tool-call decisions in state["tool_calls"] - execute_tools_node
    will actually run them next.
    """
    client = _get_client()

    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=500,
        tools=TOOL_DEFINITIONS,
        messages=[{"role": "user", "content": state["user_input"]}]
    )

    tool_calls = []
    for block in message.content:
        if block.type == "tool_use":
            tool_calls.append({
                "tool_name": block.name,
                "tool_input": block.input
            })

    state["tool_calls"] = tool_calls
    return state


def execute_tools_node(state: AgentState) -> AgentState:
    """
    Takes the LLM's tool decisions from decide_node and actually calls
    the real functions in stock_tools.py, building up state["trace"] in
    the exact {tool_name, tool_input, tool_output} shape agent_evaluator.py
    expects.

    If a tool_name does not match any real function (should not happen
    given the schema passed to decide_node, but not assumed impossible),
    the trace entry is still recorded with an error-string tool_output
    rather than skipped or raised, so agent_evaluator.py can see and
    grade the bad decision instead of it being silently lost.
    """
    trace = []

    for call in state["tool_calls"]:
        tool_name = call["tool_name"]
        tool_input = call["tool_input"]

        if tool_name == "get_current_price":
            tool_output = get_current_price(tool_input["ticker"])
        elif tool_name == "get_historical_price":
            tool_output = get_historical_price(tool_input["ticker"], tool_input["period"])
        elif tool_name == "get_index_level":
            tool_output = get_index_level(tool_input["index_name"])
        else:
            tool_output = f"ERROR: unknown tool '{tool_name}'"

        trace.append({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output
        })

    state["trace"] = trace
    return state


def final_answer_node(state: AgentState) -> AgentState:
    """
    Calls Claude again to compose a final answer grounded in the real
    tool outputs collected in trace, alongside the original user_input
    so Claude knows what to actually do with those numbers (e.g. compute
    a comparison, not just recite them).

    If trace is empty (decide_node returned zero tool_calls, e.g. an
    out-of-scope question), Claude is still called and allowed to answer
    ungrounded - this is deliberate, not an oversight. A fixed fallback
    string here would hide that failure mode from
    check_final_answer_uses_tool_output in agent_evaluator.py instead of
    exposing it for grading.
    """
    client = _get_client()

    trace_summary_lines = []
    for entry in state["trace"]:
        trace_summary_lines.append(
            f"Tool: {entry['tool_name']}, Input: {entry['tool_input']}, Output: {entry['tool_output']}"
        )
    trace_summary = "\n".join(trace_summary_lines) if trace_summary_lines else "No tools were called."

    prompt = (
        f"Original question: {state['user_input']}\n\n"
        f"Tool results:\n{trace_summary}\n\n"
        "Using the tool results above, answer the original question directly "
        "and concisely. If numbers are involved, state them exactly as given "
        "in the tool results, with no comma thousand-separators, no currency "
        "symbols, and no rounding or reformatting. For example, write 78401.86 "
        "exactly like that, not 78,401.86 or Rs 78,401.86. You may still "
        "explain or describe the number in plain words around it."
    )

    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    final_answer_text = ""
    for block in message.content:
        if block.type == "text":
            final_answer_text += block.text

    state["final_answer"] = final_answer_text
    return state

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decide", decide_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("final_answer", final_answer_node)

    graph.add_edge(START, "decide")
    graph.add_edge("decide", "execute_tools")
    graph.add_edge("execute_tools", "final_answer")
    graph.add_edge("final_answer", END)

    return graph.compile()


def run_agent(user_input: str) -> dict:
    app = build_graph()
    initial_state: AgentState = {
        "user_input": user_input,
        "tool_calls": [],
        "trace": [],
        "final_answer": "",
    }
    result = app.invoke(initial_state)
    return {"trace": result["trace"], "final_answer": result["final_answer"]}