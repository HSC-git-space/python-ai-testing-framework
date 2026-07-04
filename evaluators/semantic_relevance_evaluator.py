# evaluators/semantic_relevance_evaluator.py

import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class SemanticRelevanceEvaluator(BaseEvaluator):
    """
    RAGAS-lite: Semantic relevance evaluator (embedding-based).

    Upgrade path over RelevanceEvaluator's lexical word-overlap approach.
    Checks whether the response addresses the prompt using sentence
    embeddings and cosine similarity, catching paraphrases and synonymy
    that pure word-overlap misses.

    Why this exists alongside RelevanceEvaluator, not replacing it:
    RelevanceEvaluator's lexical check fails on valid paraphrases with
    no shared vocabulary (e.g. "store closes at 6pm" vs "shop shuts at
    6pm" — same meaning, almost no shared words). This evaluator catches
    that case by comparing meaning, not literal words.

    Calibration (from real runs against three test cases):
    - Easy paraphrase, no shared words:            0.67 -> PASS
    - Hard paraphrase, different structure/length: 0.75 -> PASS
    - Adversarial: shared vocabulary, opposite
      meaning (e.g. "refund in 5 days" vs
      "refund not available"):                     0.53 -> FAIL

    Known limitation: this model captures topical relatedness more
    strongly than logical negation. Two sentences that share vocabulary
    but directly contradict each other can still score moderately
    (~0.5), since embedding similarity responds to topic overlap as
    well as meaning. The 0.6 threshold is chosen specifically to
    separate the adversarial case above from the two genuine paraphrase
    cases, based on observed scores — not an arbitrary default.

    Note on use_mock: unlike other evaluators in this framework, this
    evaluator has no meaningful mock/real split. It always runs a real,
    local computation (no API call, no cost either way) rather than
    choosing between a cheap simulated check and a paid API call. The
    use_mock parameter is intentionally omitted here rather than added
    just for surface consistency with other evaluators.
    """

    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        self.total_cost = 0.0
        self.total_latency = 0.0
        self._model = None  # lazy-loaded on first use, not on construction

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def evaluate(self, response: str, request: EvalRequest) -> EvaluatorScore:
        prompt = request.prompt

        if not prompt:
            return EvaluatorScore(
                evaluator_name="semantic_relevance",
                score=1.0,
                passed=True,
                reason="No prompt provided — semantic relevance evaluator not applicable"
            )

        start = time.time()

        similarity = self._compute_similarity(response, prompt)
        cost = 0.0  # local model inference — no API call, no cost

        latency = time.time() - start
        self.total_cost += cost
        self.total_latency += latency

        passed = similarity >= self.similarity_threshold

        reason = (
            f"Semantic relevance: {'PASS' if passed else 'FAIL'} "
            f"(cosine similarity: {similarity:.2f}, threshold: {self.similarity_threshold}) | "
            f"Cost: ${cost:.6f} | "
            f"Latency: {latency:.2f}s"
        )

        return EvaluatorScore(
            evaluator_name="semantic_relevance",
            score=round(similarity, 3),
            passed=passed,
            reason=reason
        )

    def _compute_similarity(self, response: str, prompt: str) -> float:
        model = self._get_model()
        prompt_vec = model.encode([prompt])
        response_vec = model.encode([response])
        return float(cosine_similarity(prompt_vec, response_vec)[0][0])

    def get_session_stats(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_latency_seconds": round(self.total_latency, 3)
        }