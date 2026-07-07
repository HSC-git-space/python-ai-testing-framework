"""
test_deepeval_stock_agent.py

Wraps DeepEval's AnswerRelevancyMetric and FaithfulnessMetric around the
real LangGraph stock agent (agent.langgraph_agent_runner.run_agent).

This is deliberately NOT offline and NOT deterministic - run_agent makes
a real Claude tool-decision call, hits real live yfinance data, and
composes a real Claude final answer, on every run. DeepEval's own judge
model (AnthropicModel, defaulting to ANTHROPIC_API_KEY from the
environment) then scores that real output. A real ANTHROPIC_API_KEY
must be set in .env locally or as a GitHub Actions secret in CI.

retrieval_context for each test case is built from trace's tool_output
values, turned into strings - the real equivalent of "retrieved
documents" for this agent, since there is no traditional RAG document
store here, only real tool call results.

Threshold of 0.7 is used for both metrics, matching DeepEval's own
common convention for AnswerRelevancyMetric and FaithfulnessMetric,
not simply copied from this repo's own EVAL_PASS_THRESHOLD env var by
coincidence - the two happen to agree.

Honest positioning, not oversold: this wrapper does not prove anything
new about mock-vs-live gaps - that was already established by the
LangGraph rebuild's own test suite (test_stock_agent_eval.py). Its
value here is ATS/resume-keyword recognition for "DeepEval integration",
using cases already picked, not new skill depth.
"""

from dotenv import load_dotenv

load_dotenv()

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase

from agent.langgraph_agent_runner import run_agent

THRESHOLD = 0.7
JUDGE_MODEL = AnthropicModel(model="claude-sonnet-4-6")


def build_retrieval_context(trace: list) -> list:
    return [
        f"{entry['tool_name']}({entry['tool_input']}) returned {entry['tool_output']}"
        for entry in trace
    ]


def test_sensex_now_deepeval_relevancy_and_faithfulness():
    query = "What is the Sensex right now?"
    result = run_agent(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result["final_answer"],
        retrieval_context=build_retrieval_context(result["trace"]),
    )

    relevancy = AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL)
    faithfulness = FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL)

    assert_test(test_case, [relevancy, faithfulness])


def test_nifty_week_ago_deepeval_relevancy_and_faithfulness():
    query = "What was the Nifty a week ago?"
    result = run_agent(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result["final_answer"],
        retrieval_context=build_retrieval_context(result["trace"]),
    )

    relevancy = AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL)
    faithfulness = FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL)

    assert_test(test_case, [relevancy, faithfulness])


def test_sensex_comparison_deepeval_relevancy_and_faithfulness():
    query = "Compare the Sensex today versus a month ago."
    result = run_agent(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result["final_answer"],
        retrieval_context=build_retrieval_context(result["trace"]),
    )

    relevancy = AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL)
    faithfulness = FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL)

    assert_test(test_case, [relevancy, faithfulness])