import pytest
from models.eval_request import EvalRequest
from evaluators.tone_evaluator import ToneEvaluator


@pytest.fixture
def tone_evaluator():
    return ToneEvaluator()


def test_formal_tone_detected(tone_evaluator):
    request = EvalRequest(
        prompt="Explain machine learning.",
        evaluators=["tone"],
        expected_tone="formal"
    )
    response = "Machine learning is a subset of artificial intelligence. Furthermore, it enables systems to learn from data. Therefore, it is widely used in industry."

    score = tone_evaluator.evaluate(response, request)

    assert score.passed is True


def test_informal_tone_detected(tone_evaluator):
    request = EvalRequest(
        prompt="What is ML?",
        evaluators=["tone"],
        expected_tone="informal"
    )
    response = "Hey! ML is basically just teaching computers stuff. It's pretty cool and gonna change everything."

    score = tone_evaluator.evaluate(response, request)

    assert score.passed is True


def test_formal_expected_but_informal_detected(tone_evaluator):
    request = EvalRequest(
        prompt="Explain machine learning.",
        evaluators=["tone"],
        expected_tone="formal"
    )
    response = "Hey guys! ML is basically just teaching computers stuff. It's pretty cool yeah."

    score = tone_evaluator.evaluate(response, request)

    assert score.passed is False


def test_no_tone_defined(tone_evaluator):
    request = EvalRequest(
        prompt="Tell me something.",
        evaluators=["tone"]
    )
    response = "Here is something interesting about the world."

    score = tone_evaluator.evaluate(response, request)

    assert score.passed is True
    assert score.score == 1.0
    