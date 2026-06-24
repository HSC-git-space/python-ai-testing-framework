import pytest
from models.eval_request import EvalRequest
from engine.eval_engine import EvalEngine


@pytest.fixture
def eval_engine():
    return EvalEngine()


@pytest.mark.integration
def test_consistency_same_prompt(eval_engine):
    """
    Runs consistency check on a factual prompt.
    Expects consistent output class across N runs.
    """
    request = EvalRequest(
        prompt="What is the capital of France? Answer in one sentence.",
        provider="anthropic",
        evaluators=["consistency"],
        required_keywords=["Paris"],
        test_id="CON001"
    )

    score = eval_engine.run_consistency_check(
        request=request,
        n=3,
        classifier_fn=lambda r: "paris" in r.lower()
    )

    assert score.passed is True
    assert score.score >= 0.6


@pytest.mark.integration
def test_consistency_factual_prompt(eval_engine):
    """
    Tests that a factual science prompt returns
    consistent output class across runs.
    """
    request = EvalRequest(
        prompt="What temperature does water boil at in Celsius?",
        provider="anthropic",
        evaluators=["consistency"],
        required_keywords=["100"],
        test_id="CON002"
    )

    score = eval_engine.run_consistency_check(
        request=request,
        n=3,
        classifier_fn=lambda r: "100" in r
    )

    assert score.passed is True
    assert score.score >= 0.6