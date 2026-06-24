import pytest
from models.eval_request import EvalRequest
from evaluators.length_evaluator import LengthEvaluator


@pytest.fixture
def length_evaluator():
    return LengthEvaluator()


def test_response_within_length_bounds(length_evaluator):
    request = EvalRequest(
        prompt="What is Python?",
        evaluators=["length"],
        min_length=10,
        max_length=500
    )
    response = "Python is a high level programming language known for simplicity."

    score = length_evaluator.evaluate(response, request)

    assert score.passed is True
    assert score.score == 1.0


def test_response_too_short(length_evaluator):
    request = EvalRequest(
        prompt="What is Python?",
        evaluators=["length"],
        min_length=100,
        max_length=500
    )
    response = "Python is a language."

    score = length_evaluator.evaluate(response, request)

    assert score.passed is False


def test_response_too_long(length_evaluator):
    request = EvalRequest(
        prompt="What is Python?",
        evaluators=["length"],
        min_length=10,
        max_length=20
    )
    response = "Python is a high level programming language known for its simplicity and readability."

    score = length_evaluator.evaluate(response, request)

    assert score.passed is False


def test_no_length_bounds_defined(length_evaluator):
    request = EvalRequest(
        prompt="What is Python?",
        evaluators=["length"]
    )
    response = "Python is a language."

    score = length_evaluator.evaluate(response, request)

    assert score.passed is True
    assert score.score == 1.0
