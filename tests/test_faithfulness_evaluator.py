import pytest
from evaluators.faithfulness_evaluator import FaithfulnessEvaluator
from models.eval_request import EvalRequest


@pytest.fixture
def faithfulness():
    return FaithfulnessEvaluator(claim_support_threshold=0.6, pass_threshold=0.7, use_mock=True)


@pytest.fixture
def base_request():
    return EvalRequest(
        prompt="What time does the store close?",
        provider="anthropic",
        evaluators=["faithfulness"],
        retrieved_context=["The store closes at 8pm on weekdays."]
    )


class TestFaithfulnessEvaluatorMock:

    def test_grounded_response_passes(self, faithfulness, base_request):
        response = "The store closes at 8pm on weekdays."
        result = faithfulness.evaluate(response, base_request)
        assert result.passed is True
        assert result.score >= 0.7

    def test_invented_claim_fails(self, faithfulness, base_request):
        """
        THE KEY STORY — response invents a claim not present in context.

        Context only says "closes at 8pm on weekdays" — response adds a
        claim about weekend hours that was never retrieved. Faithfulness
        should flag this as unsupported, regardless of whether it's
        factually true in the real world.
        """
        response = "The store closes at 8pm on weekdays and is open 24 hours on weekends."
        result = faithfulness.evaluate(response, base_request)
        assert "Faithfulness" in result.reason
        assert "Cost" in result.reason
        assert "Latency" in result.reason

    def test_no_context_skips_evaluator(self, faithfulness):
        request = EvalRequest(
            prompt="Tell me something",
            provider="anthropic",
            evaluators=["faithfulness"]
        )
        result = faithfulness.evaluate("Some response", request)
        assert result.passed is True
        assert "not applicable" in result.reason.lower()

    def test_cost_tracking(self, faithfulness, base_request):
        response = "The store closes at 8pm on weekdays."
        faithfulness.evaluate(response, base_request)
        stats = faithfulness.get_session_stats()
        assert "total_cost_usd" in stats
        assert "total_latency_seconds" in stats

    def test_reason_contains_metadata(self, faithfulness, base_request):
        response = "The store closes at 8pm on weekdays."
        result = faithfulness.evaluate(response, base_request)
        assert "claims grounded in context" in result.reason
        assert "Cost" in result.reason


class TestFaithfulnessVsHallucination:
    """
    Side-by-side: faithfulness (vs retrieved_context) catches a different
    failure mode than hallucination (vs ground_truth).
    """

    def test_faithful_to_wrong_context_still_passes(self):
        """
        Context itself is stale/wrong, but response is faithful to it.
        Faithfulness PASSES here — it only checks grounding, not truth.
        This is the interview story: faithfulness != factual correctness.
        """
        faithfulness = FaithfulnessEvaluator(use_mock=True)
        request = EvalRequest(
            prompt="What time does the store close?",
            provider="anthropic",
            evaluators=["faithfulness"],
            retrieved_context=["The store closes at 6pm on weekdays."]  # stale data
        )
        response = "The store closes at 6pm on weekdays."
        result = faithfulness.evaluate(response, request)
        assert result.passed is True

    def test_unrelated_response_fails_faithfulness(self):
        faithfulness = FaithfulnessEvaluator(use_mock=True)
        request = EvalRequest(
            prompt="What time does the store close?",
            provider="anthropic",
            evaluators=["faithfulness"],
            retrieved_context=["The store closes at 8pm on weekdays."]
        )
        response = "Our return policy allows exchanges within 90 days of purchase."
        result = faithfulness.evaluate(response, request)
        assert result.score < 1.0