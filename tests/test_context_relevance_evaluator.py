import pytest
from evaluators.context_relevance_evaluator import ContextRelevanceEvaluator
from models.eval_request import EvalRequest


@pytest.fixture
def evaluator():
    return ContextRelevanceEvaluator(use_mock=True)


def test_relevant_context_passes(evaluator):
    """All retrieved passages are on-topic for the prompt -> high score."""
    request = EvalRequest(
        prompt="What is the return policy?",
        provider="mock",
        evaluators=["context_relevance"],
        retrieved_context=[
            "Returns are accepted within 30 days of purchase.",
            "The return policy excludes final-sale items."
        ]
    )
    response = "Returns are accepted within 30 days."

    result = evaluator.evaluate(response, request)

    assert result.passed is True
    assert result.score == 1.0
    assert result.evaluator_name == "context_relevance"


def test_irrelevant_context_fails(evaluator):
    """
    Pinned interview example: retriever pulls completely unrelated
    passages (store founding history) for a return-policy question.
    This is a retrieval bug, not a generation bug - the LLM being
    honest about not knowing doesn't change that context relevance
    should FAIL here.
    """
    request = EvalRequest(
        prompt="What is the store's return policy?",
        provider="mock",
        evaluators=["context_relevance"],
        retrieved_context=[
            "The store was founded in 1998.",
            "The founder's name is John Smith."
        ]
    )
    response = "I don't have information on the return policy."

    result = evaluator.evaluate(response, request)

    assert result.passed is False
    assert result.score < 0.5


def test_mixed_context_scores_partially(evaluator):
    """One relevant passage, one irrelevant passage -> partial score."""
    request = EvalRequest(
        prompt="What is the return policy?",
        provider="mock",
        evaluators=["context_relevance"],
        retrieved_context=[
            "Returns are accepted within 30 days of purchase.",
            "The store was founded in 1998."
        ]
    )
    response = "Returns are accepted within 30 days."

    result = evaluator.evaluate(response, request)

    assert 0.0 < result.score < 1.0


def test_empty_context_returns_not_applicable(evaluator):
    """No retrieved_context provided -> evaluator short-circuits to a pass."""
    request = EvalRequest(
        prompt="What is the return policy?",
        provider="mock",
        evaluators=["context_relevance"],
        retrieved_context=[]
    )
    response = "Returns are accepted within 30 days."

    result = evaluator.evaluate(response, request)

    assert result.passed is True
    assert result.score == 1.0
    assert "not applicable" in result.reason.lower()


def test_empty_prompt_returns_not_applicable(evaluator):
    """No prompt provided -> evaluator short-circuits to a pass."""
    request = EvalRequest(
        prompt="",
        provider="mock",
        evaluators=["context_relevance"],
        retrieved_context=["Some context passage."]
    )
    response = "Some response."

    result = evaluator.evaluate(response, request)

    assert result.passed is True
    assert result.score == 1.0
    assert "not applicable" in result.reason.lower()


def test_context_relevance_independent_of_faithfulness(evaluator):
    """
    Documents the core distinction: context relevance measures retrieval
    quality, not whether the LLM's response was faithful to what it was
    given. A response can be fully honest/faithful while the underlying
    context is garbage - this test only checks the context_relevance
    score itself, independent of what the response says.
    """
    request = EvalRequest(
        prompt="What are the store's opening and closing hours?",
        provider="mock",
        evaluators=["context_relevance"],
        retrieved_context=[
            "The store offers free shipping on orders over $50.",
            "Customer service can be reached by phone or email."
        ]
    )
    response = "I don't have that information available."

    result = evaluator.evaluate(response, request)

    assert result.passed is False


def test_session_stats_tracked(evaluator):
    """Mock mode should have zero cost, but latency should still be tracked."""
    request = EvalRequest(
        prompt="What is the return policy?",
        provider="mock",
        evaluators=["context_relevance"],
        retrieved_context=["Returns are accepted within 30 days of purchase."]
    )
    evaluator.evaluate("Returns are accepted within 30 days.", request)

    stats = evaluator.get_session_stats()

    assert stats["total_cost_usd"] == 0.0
    assert stats["total_latency_seconds"] >= 0.0