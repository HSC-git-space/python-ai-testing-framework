import pytest
from models.eval_request import EvalRequest
from engine.eval_engine import EvalEngine


@pytest.fixture
def eval_engine():
    return EvalEngine()


@pytest.mark.integration
def test_consistent_response_length(eval_engine):
    """
    Same prompt run 3 times.
    All responses should meet minimum length — tests output stability.
    """
    request = EvalRequest(
        prompt="What is the capital of France? Answer in one sentence.",
        provider="anthropic",
        evaluators=["length"],
        min_length=5,
        max_length=500,
        test_id="ND001"
    )

    results = []
    for _ in range(3):
        result = eval_engine.run(request)
        results.append(result)

    passed_count = sum(1 for r in results if r.overall_passed)
    assert passed_count >= 2, f"Only {passed_count}/3 runs passed length check"


@pytest.mark.integration
def test_consistent_keyword_presence(eval_engine):
    """
    Same prompt run 3 times.
    Paris should appear in at least 2 of 3 responses.
    """
    request = EvalRequest(
        prompt="What is the capital of France?",
        provider="anthropic",
        evaluators=["keyword"],
        required_keywords=["Paris"],
        test_id="ND002"
    )

    results = []
    for _ in range(3):
        result = eval_engine.run(request)
        results.append(result)

    passed_count = sum(1 for r in results if r.overall_passed)
    assert passed_count >= 2, f"Only {passed_count}/3 runs contained required keywords"