import time
from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class RelevanceEvaluator(BaseEvaluator):
    """
    RAGAS-lite: Answer relevance evaluator.

    Checks whether the response actually addresses the prompt — independent
    of whether it's factually correct or grounded in context.

    Why this is a separate axis from faithfulness:
    A response can be 100% faithful to retrieved_context (every claim
    grounded, nothing invented) while still being non-relevant — e.g. it
    dumps unrelated facts from the context instead of answering what was
    asked. Faithfulness alone would score this well; relevance catches it.

    Concrete example:
    Prompt: "What is the store's return policy?"
    Context: ["The store closes at 8pm.", "Returns are accepted within 30 days."]
    Response: "The store closes at 8pm."
    -> Faithfulness: PASS (claim is grounded in context)
    -> Relevance: FAIL (doesn't answer the actual question asked)

    Mock approach: checks whether the response's content words overlap
    with the prompt's content words (proxy for "on-topic"), and penalizes
    responses that are just verbatim context dumps unrelated to the prompt.
    Real RAGAS generates synthetic questions from the response and checks
    similarity to the original prompt — noted as the upgrade path.
    """

    COST_PER_1K_INPUT_TOKENS = 0.000003
    COST_PER_1K_OUTPUT_TOKENS = 0.000015

    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
        "at", "to", "for", "and", "or", "but", "it", "its", "be", "been",
        "has", "have", "had", "this", "that", "with", "from", "by", "what",
        "how", "why", "when", "does", "do", "did"
    }

    def __init__(
            self,
            relevance_threshold: float = 0.4,
            pass_threshold: float = 0.6,
            use_mock: bool = True
    ):
        self.relevance_threshold = relevance_threshold
        self.pass_threshold = pass_threshold
        self.use_mock = use_mock
        self.total_cost = 0.0
        self.total_latency = 0.0

    def evaluate(self, response: str, request: EvalRequest) -> EvaluatorScore:
        prompt = request.prompt

        if not prompt:
            return EvaluatorScore(
                evaluator_name="relevance",
                score=1.0,
                passed=True,
                reason="No prompt provided — relevance evaluator not applicable"
            )

        start = time.time()

        if self.use_mock:
            result = self._mock_relevance(response, prompt)
            cost = 0.0
        else:
            result = self._real_relevance(response, prompt)
            cost = self._estimate_cost(response, prompt)

        latency = time.time() - start
        self.total_cost += cost
        self.total_latency += latency

        score = result["score"]
        passed = score >= self.pass_threshold

        reason = (
            f"Relevance: {'PASS' if passed else 'FAIL'} "
            f"(topic overlap: {score:.2f}) | "
            f"Reason: {result['reason']} | "
            f"Cost: ${cost:.6f} | "
            f"Latency: {latency:.2f}s"
        )

        return EvaluatorScore(
            evaluator_name="relevance",
            score=round(score, 3),
            passed=passed,
            reason=reason
        )

    def _mock_relevance(self, response: str, prompt: str) -> dict:
        prompt_words = self._content_words(prompt)
        response_lower = response.lower()

        if not prompt_words:
            return {"score": 1.0, "reason": "No content words extracted from prompt"}

        words_present = sum(1 for w in prompt_words if w in response_lower)
        coverage = words_present / len(prompt_words)

        if coverage >= self.relevance_threshold:
            reason = "Response addresses the topic of the prompt"
        else:
            reason = "Response does not appear to address what was asked"

        return {"score": coverage, "reason": reason}

    def _content_words(self, text: str) -> list:
        words = [w.strip("?.,!").lower() for w in text.split()]
        return [w for w in words if len(w) > 3 and w not in self.STOP_WORDS]

    @staticmethod
    def _real_relevance(response: str, prompt: str) -> dict:
        raise NotImplementedError(
            "Real relevance evaluation requires ANTHROPIC_API_KEY. "
            "Set use_mock=False only after configuring your API key. "
            "See README for setup instructions."
        )

    def _estimate_cost(self, response: str, prompt: str) -> float:
        prompt_text = response + prompt
        input_tokens = len(prompt_text) / 4
        output_tokens = 60
        return (
                (input_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS +
                (output_tokens / 1000) * self.COST_PER_1K_OUTPUT_TOKENS
        )

    def get_session_stats(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_latency_seconds": round(self.total_latency, 3)
        }