import pytest
from evaluators.semantic_relevance_evaluator import SemanticRelevanceEvaluator
from models.eval_request import EvalRequest


@pytest.fixture(scope="module")
def evaluator():
    # scope="module": model load is slow, share one instance across all tests in this file
    return SemanticRelevanceEvaluator(similarity_threshold=0.6)


def test_easy_paraphrase_no_shared_words_passes(evaluator):
    """
    Calibration case 1: paraphrase with almost no shared vocabulary.
    Real measured score (from sandbox_embedding_test.py): 0.67 -> PASS.
    Lexical RelevanceEvaluator would fail this case since "store"/"shop"
    and "closes"/"shuts" share no literal words; this evaluator should
    catch it via embeddings. These are the exact original sentences
    used during real calibration, not approximated paraphrases of them -
    cosine similarity is sensitive to specific wording, so reusing a
    "similar" pair does not reproduce the same score.
    """
    request = EvalRequest(
        prompt="What time does the store close?",
        provider="mock",
        evaluators=["semantic_relevance"]
    )
    response = "The shop shuts at 6pm."

    result = evaluator.evaluate(response, request)

    assert result.passed is True
    assert result.score == pytest.approx(0.67, abs=0.05)
    assert result.evaluator_name == "semantic_relevance"


def test_hard_paraphrase_different_structure_passes(evaluator):
    """
    Calibration case 2: paraphrase with different sentence structure and
    length (refund timing example). Real measured score: 0.75 -> PASS,
    higher than the easy case, showing this isn't just catching simple
    synonym swaps. Exact original sentences from calibration.
    """
    request = EvalRequest(
        prompt="The refund will be processed within 5 to 7 business days after we receive the returned item.",
        provider="mock",
        evaluators=["semantic_relevance"]
    )
    response = "Once the item is returned, expect your money back in about a week."

    result = evaluator.evaluate(response, request)

    assert result.passed is True
    assert result.score == pytest.approx(0.75, abs=0.05)


def test_adversarial_shared_vocabulary_opposite_meaning_fails(evaluator):
    """
    Calibration case 3 (adversarial): shared vocabulary, opposite
    meaning. Real measured score: 0.53 -> FAIL. This is the documented
    limitation case - embeddings capture topical relatedness more than
    logical negation, so this scores meaningfully lower than the two
    genuine paraphrases but not dramatically low. The 0.6 threshold
    exists specifically to separate this case from the two above.
    Exact original sentences from calibration.
    """
    request = EvalRequest(
        prompt="The refund will be processed within 5 to 7 business days.",
        provider="mock",
        evaluators=["semantic_relevance"]
    )
    response = "Refunds are not available for this item."

    result = evaluator.evaluate(response, request)

    assert result.passed is False
    assert result.score == pytest.approx(0.53, abs=0.05)


def test_empty_prompt_returns_not_applicable(evaluator):
    """No prompt provided -> evaluator short-circuits to a pass, not applicable."""
    request = EvalRequest(
        prompt="",
        provider="mock",
        evaluators=["semantic_relevance"]
    )
    response = "Some response text."

    result = evaluator.evaluate(response, request)

    assert result.passed is True
    assert result.score == 1.0
    assert "not applicable" in result.reason.lower()


def test_session_stats_tracked_with_zero_cost(evaluator):
    """
    No use_mock split here - this evaluator always runs real local
    inference, but local inference has zero API cost either way.
    Latency should still be tracked since real computation happens.
    """
    request = EvalRequest(
        prompt="What is the store's return policy?",
        provider="mock",
        evaluators=["semantic_relevance"]
    )
    evaluator.evaluate("The store's return policy allows returns within 30 days.", request)

    stats = evaluator.get_session_stats()

    assert stats["total_cost_usd"] == 0.0
    assert stats["total_latency_seconds"] >= 0.0


def test_score_always_between_0_and_1(evaluator):
    """Cosine similarity should be bounded; sanity check across a normal case."""
    request = EvalRequest(
        prompt="How do I reset my password?",
        provider="mock",
        evaluators=["semantic_relevance"]
    )
    response = "To reset your password, click 'Forgot password' on the login page."

    result = evaluator.evaluate(response, request)

    assert 0.0 <= result.score <= 1.0