import pytest
from models.eval_request import EvalRequest
from evaluators.keyword_evaluator import KeywordEvaluator


@pytest.fixture
def keyword_evaluator():
    return KeywordEvaluator()


def test_all_required_keywords_present(keyword_evaluator):
    request = EvalRequest(
        prompt="What is an API?",
        evaluators=["keyword"],
        required_keywords=["API", "request", "response"]
    )
    response = "An API is an interface that handles request and response cycles."

    score = keyword_evaluator.evaluate(response, request)

    assert score.passed is True
    assert score.score == 1.0


def test_missing_required_keyword(keyword_evaluator):
    request = EvalRequest(
        prompt="What is an API?",
        evaluators=["keyword"],
        required_keywords=["API", "request", "response"]
    )
    response = "An API is an interface."

    score = keyword_evaluator.evaluate(response, request)

    assert score.passed is False
    assert score.score < 1.0


def test_forbidden_keyword_present(keyword_evaluator):
    request = EvalRequest(
        prompt="Explain REST APIs.",
        evaluators=["keyword"],
        forbidden_keywords=["SOAP", "deprecated"]
    )
    response = "REST APIs are modern. SOAP is deprecated."

    score = keyword_evaluator.evaluate(response, request)

    assert score.passed is False


def test_no_keywords_defined(keyword_evaluator):
    request = EvalRequest(
        prompt="Tell me something.",
        evaluators=["keyword"]
    )
    response = "Here is something interesting."

    score = keyword_evaluator.evaluate(response, request)

    assert score.passed is True
    assert score.score == 1.0