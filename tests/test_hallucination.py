import pytest
from models.eval_request import EvalRequest
from evaluators.hallucination_evaluator import HallucinationEvaluator


@pytest.fixture
def hallucination_evaluator():
    return HallucinationEvaluator()


def test_no_hallucination_detected(hallucination_evaluator):
    request = EvalRequest(
        prompt="What is the capital of France?",
        evaluators=["hallucination"],
        ground_truth=[
            "The capital of France is Paris",
            "France is a country in Western Europe"
        ]
    )
    response = "The capital of France is Paris. It is located in Western Europe."

    score = hallucination_evaluator.evaluate(response, request)

    assert score.passed is True
    assert score.score >= 0.7


def test_hallucination_detected(hallucination_evaluator):
    request = EvalRequest(
        prompt="What is the capital of France?",
        evaluators=["hallucination"],
        ground_truth=[
            "The capital of France is Paris",
            "France is a country in Western Europe"
        ]
    )
    response = "The capital of France is Lyon. It is a major city in Europe."

    score = hallucination_evaluator.evaluate(response, request)

    assert score.passed is False


def test_no_ground_truth_defined(hallucination_evaluator):
    request = EvalRequest(
        prompt="Tell me something.",
        evaluators=["hallucination"],
        ground_truth=[]
    )
    response = "Here is something interesting."

    score = hallucination_evaluator.evaluate(response, request)

    assert score.passed is True
    assert score.score == 1.0


def test_partial_hallucination(hallucination_evaluator):
    request = EvalRequest(
        prompt="What is the capital of France?",
        evaluators=["hallucination"],
        ground_truth=[
            "The capital of France is Paris",
            "Water boils at 100 degrees Celsius"
        ]
    )
    # Response contradicts Paris fact — says Lyon instead
    # Water boiling fact is not addressed at all
    # Evaluator should detect at least one unsupported fact
    response = "The capital of France is Lyon."

    score = hallucination_evaluator.evaluate(response, request)

    assert score.score < 1.0