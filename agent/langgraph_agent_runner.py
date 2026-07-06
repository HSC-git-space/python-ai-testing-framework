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
to call and with what arguments, based on user_input. This is the real,
non-deterministic replacement for MockAgent's keyword matching.

execute_tools_node: takes the LLM's tool decision(s) and actually calls
the real functions in stock_tools.py, building up trace in the exact
shape agent_evaluator.py expects.

final_answer_node: calls the LLM again, this time to compose a final
answer grounded in the real tool outputs collected in trace.
"""

from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, START, END

from agent.stock_tools import get_current_price, get_historical_price, get_index_level


class AgentState(TypedDict):
    user_input: str
    trace: List[Dict[str, Any]]
    final_answer: str


def decide_node(state: AgentState) -> AgentState:
    """
    STUB - real LLM tool-calling logic goes here next.
    For now, returns state unchanged so the graph wiring can be tested
    end to end before the real decision logic is written.
    """
    return state


def execute_tools_node(state: AgentState) -> AgentState:
    """
    STUB - real tool execution logic goes here next.
    For now, returns state unchanged.
    """
    return state


def final_answer_node(state: AgentState) -> AgentState:
    """
    STUB - real answer-composition logic goes here next.
    For now, returns state unchanged.
    """
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
        "trace": [],
        "final_answer": "",
    }
    result = app.invoke(initial_state)
    return {"trace": result["trace"], "final_answer": result["final_answer"]}   