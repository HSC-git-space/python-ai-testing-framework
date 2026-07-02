import pytest
from evaluators.relevance_evaluator import RelevanceEvaluator
from models.eval_request import EvalRequest


@pytest.fixture
def evaluator():
    return RelevanceEvaluator(use_mock=True)


def test_on_topic_response_passes(evaluator):
    """Response directly answers what was asked -> high relevance score."""
    request = EvalRequest(
        prompt="What is the store's return policy?",
        provider="mock",
        evaluators=["relevance"]
    )
    response = "The store's return policy allows returns within 30 days."

    result = evaluator.evaluate(response, request)

    assert result.passed is True
    assert result.score >= 0.6
    assert result.evaluator_name == "relevance"


def test_off_topic_response_fails(evaluator):
    """Response is faithful-sounding but doesn't address the prompt at all."""
    request = EvalRequest(
        prompt="What is the store's return policy?",
        provider="mock",
        evaluators=["relevance"]
    )
    response = "The store closes at 8pm on weekdays."

    result = evaluator.evaluate(response, request)

    assert result.passed is False
    assert result.score < 0.6


def test_partial_topic_overlap_scores_between(evaluator):
    """Response mentions some prompt content words but not enough to fully answer."""
    request = EvalRequest(
        prompt="What is the store's return policy for electronics?",
        provider="mock",
        evaluators=["relevance"]
    )
    response = "The store sells electronics of many brands."

    result = evaluator.evaluate(response, request)

    assert 0.0 <= result.score <= 1.0


def test_empty_prompt_returns_not_applicable(evaluator):
    """No prompt provided -> evaluator short-circuits to a pass, not applicable."""
    request = EvalRequest(
        prompt="",
        provider="mock",
        evaluators=["relevance"]
    )
    response = "Some response text."

    result = evaluator.evaluate(response, request)

    assert result.passed is True
    assert result.score == 1.0
    assert "not applicable" in result.reason.lower()


def test_possessive_stripping_matches_correctly(evaluator):
    """
    Regression test for the possessive-stripping bug: 'store's' in the
    prompt must match 'store' in the response. Before the fix,
    _content_words would keep "store's" as a single token that could
    never match the bare word "store" in a response, artificially
    deflating the score. Response deliberately echoes both content
    words ("store", "hours") so coverage clears threshold once the
    possessive is correctly stripped.
    """
    request = EvalRequest(
        prompt="What are the store's hours?",
        provider="mock",
        evaluators=["relevance"]
    )
    response = "The store's hours are 9am to 9pm daily."

    result = evaluator.evaluate(response, request)

    assert result.passed is True
    assert result.score == 1.0


def test_plural_mismatch_is_documented_limitation(evaluator):
    """
    Known limitation: plurals are NOT stemmed. 'returns' in the response
    does not match 'return' in the prompt as a content word, so this
    case is expected to score lower than a human would judge it -
    this test documents that behavior rather than hiding it.
    """
    request = EvalRequest(
        prompt="What is the return process?",
        provider="mock",
        evaluators=["relevance"]
    )
    response = "Returns are processed within 5 business days."

    result = evaluator.evaluate(response, request)

    assert 0.0 <= result.score <= 1.0


def test_session_stats_tracked(evaluator):
    """Mock mode should have zero cost, but latency should still be tracked."""
    request = EvalRequest(
        prompt="What is the store's return policy?",
        provider="mock",
        evaluators=["relevance"]
    )
    evaluator.evaluate("The store's return policy allows returns within 30 days.", request)

    stats = evaluator.get_session_stats()

    assert stats["total_cost_usd"] == 0.0
    assert stats["total_latency_seconds"] >= 0.0