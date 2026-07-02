import time
from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class FaithfulnessEvaluator(BaseEvaluator):
    """
    RAGAS-lite: Faithfulness evaluator.

    Checks whether the response is grounded in the retrieved_context —
    i.e. does the response only make claims that the context actually supports?

    This is DIFFERENT from HallucinationEvaluator:
    - HallucinationEvaluator checks the response against ground_truth
      (fixed facts about reality — "is this factually correct?")
    - FaithfulnessEvaluator checks the response against retrieved_context
      (what was actually retrieved and handed to the LLM — "did the model
      stick to what it was given, even if what it was given is wrong?")

    Concrete distinction:
    If retrieved_context says "the store closes at 8pm" (even if that's
    stale/wrong data), and the response says "the store closes at 8pm",
    faithfulness PASSES — the model was faithful to its source.
    Hallucination-against-ground-truth might still FAIL if the real
    current closing time is 9pm. Two different failure modes, two evaluators.

    Why this matters for RAG systems specifically: a RAG pipeline's biggest
    risk isn't the LLM being wrong in general — it's the LLM ignoring or
    embellishing beyond what was actually retrieved. Faithfulness isolates
    that specific failure mode from general factual correctness.

    Mock-first: same pattern as JudgeEvaluator. Real judge call wired in
    later once API key / RAGAS integration is available.
    """

    COST_PER_1K_INPUT_TOKENS = 0.000003
    COST_PER_1K_OUTPUT_TOKENS = 0.000015

    def __init__(
            self,
            claim_support_threshold: float = 0.6,
            pass_threshold: float = 0.7,
            use_mock: bool = True
    ):
        self.claim_support_threshold = claim_support_threshold
        self.pass_threshold = pass_threshold
        self.use_mock = use_mock
        self.total_cost = 0.0
        self.total_latency = 0.0

    def evaluate(self, response: str, request: EvalRequest) -> EvaluatorScore:
        context = request.retrieved_context or []

        if not context:
            return EvaluatorScore(
                evaluator_name="faithfulness",
                score=1.0,
                passed=True,
                reason="No retrieved_context provided — faithfulness evaluator not applicable"
            )

        start = time.time()

        if self.use_mock:
            result = self._mock_faithfulness(response, context)
            cost = 0.0
        else:
            result = self._real_faithfulness(response, context)
            cost = self._estimate_cost(response, context)

        latency = time.time() - start
        self.total_cost += cost
        self.total_latency += latency

        score = result["score"]
        passed = score >= self.pass_threshold

        reason = (
            f"Faithfulness: {'PASS' if passed else 'FAIL'} "
            f"({result['supported_claims']}/{result['total_claims']} claims grounded in context) | "
            f"Reason: {result['reason']} | "
            f"Cost: ${cost:.6f} | "
            f"Latency: {latency:.2f}s"
        )

        return EvaluatorScore(
            evaluator_name="faithfulness",
            score=round(score, 3),
            passed=passed,
            reason=reason
        )

    def _mock_faithfulness(self, response: str, context: list) -> dict:
        """
        Mock approach: split response into naive claim sentences, check each
        claim's content words for overlap against the combined context.
        Real RAGAS does this with an LLM extracting atomic claims — this is
        the string-overlap stand-in, same spirit as JudgeEvaluator's _mock_judge.
        """
        claims = self._split_into_claims(response)
        if not claims:
            return {
                "score": 1.0,
                "supported_claims": 0,
                "total_claims": 0,
                "reason": "No claims extracted from response"
            }

        combined_context = " ".join(context).lower()

        supported = 0
        unsupported_example = None

        for claim in claims:
            claim_words = [w for w in claim.lower().split() if len(w) > 4]
            if not claim_words:
                supported += 1
                continue

            words_present = sum(1 for w in claim_words if w in combined_context)
            coverage = words_present / len(claim_words)

            if coverage >= self.claim_support_threshold:
                supported += 1
            elif unsupported_example is None:
                unsupported_example = claim.strip()

        score = supported / len(claims)

        if score >= self.pass_threshold:
            reason = "Response claims are grounded in retrieved context"
        else:
            reason = f"Unsupported claim: '{unsupported_example[:80]}'" if unsupported_example else "Claims not grounded in context"

        return {
            "score": score,
            "supported_claims": supported,
            "total_claims": len(claims),
            "reason": reason
        }

    @staticmethod
    def _split_into_claims(response: str) -> list:
        sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".")]
        return [s for s in sentences if s]

    @staticmethod
    def _real_faithfulness(response: str, context: list) -> dict:
        raise NotImplementedError(
            "Real faithfulness evaluation requires ANTHROPIC_API_KEY. "
            "Set use_mock=False only after configuring your API key. "
            "See README for setup instructions."
        )

    def _estimate_cost(self, response: str, context: list) -> float:
        prompt_text = response + " ".join(context)
        input_tokens = len(prompt_text) / 4
        output_tokens = 100
        return (
                (input_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS +
                (output_tokens / 1000) * self.COST_PER_1K_OUTPUT_TOKENS
        )

    def get_session_stats(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_latency_seconds": round(self.total_latency, 3)
        }