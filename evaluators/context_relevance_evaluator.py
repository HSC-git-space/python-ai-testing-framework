import time
from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class ContextRelevanceEvaluator(BaseEvaluator):
    """
    RAGAS-lite: Context relevance evaluator.

    Checks whether the retrieved_context is actually relevant to the prompt -
    independent of what the LLM did with it afterward.

    This is a RETRIEVAL-quality check, not a generation-quality check.
    The other two evaluators assume retrieval already happened correctly:
    - FaithfulnessEvaluator: did the response stick to the context?
    - RelevanceEvaluator: did the response address the prompt?

    ContextRelevanceEvaluator asks a question neither of those can catch:
    was the retriever's OUTPUT even worth handing to the LLM in the first place?

    Concrete example of a bug this catches that the other two miss:
    Prompt: "What is the store's return policy?"
    Context: ["The store was founded in 1998.", "The founder's name is John Smith."]
    Response: "I don't have information on the return policy."
    -> Faithfulness: PASS (response invents nothing, stays within context)
    -> Relevance: arguably PASS-ish (response is honest about not knowing)
    -> Context relevance: FAIL - the retriever pulled completely unrelated
       passages. The LLM did nothing wrong; the retrieval step did.

    This is why context relevance exists as its own metric in RAGAS: it
    isolates the retrieval component of a RAG pipeline from the generation
    component, so you know WHICH half of the system to fix.

    Mock approach: checks each retrieved passage's content-word overlap
    against the prompt. Real RAGAS uses an LLM to extract sentences from
    context relevant to the question and scores the ratio.
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
            passage_relevance_threshold: float = 0.5,
            pass_threshold: float = 0.5,
            use_mock: bool = True
    ):
        self.passage_relevance_threshold = passage_relevance_threshold
        self.pass_threshold = pass_threshold
        self.use_mock = use_mock
        self.total_cost = 0.0
        self.total_latency = 0.0

    def evaluate(self, response: str, request: EvalRequest) -> EvaluatorScore:
        context = request.retrieved_context or []
        prompt = request.prompt

        if not context:
            return EvaluatorScore(
                evaluator_name="context_relevance",
                score=1.0,
                passed=True,
                reason="No retrieved_context provided - context relevance evaluator not applicable"
            )

        if not prompt:
            return EvaluatorScore(
                evaluator_name="context_relevance",
                score=1.0,
                passed=True,
                reason="No prompt provided - context relevance evaluator not applicable"
            )

        start = time.time()

        if self.use_mock:
            result = self._mock_context_relevance(context, prompt)
            cost = 0.0
        else:
            result = self._real_context_relevance(context, prompt)
            cost = self._estimate_cost(context, prompt)

        latency = time.time() - start
        self.total_cost += cost
        self.total_latency += latency

        score = result["score"]
        passed = score >= self.pass_threshold

        reason = (
            f"Context relevance: {'PASS' if passed else 'FAIL'} "
            f"({result['relevant_passages']}/{result['total_passages']} passages relevant to prompt) | "
            f"Reason: {result['reason']} | "
            f"Cost: ${cost:.6f} | "
            f"Latency: {latency:.2f}s"
        )

        return EvaluatorScore(
            evaluator_name="context_relevance",
            score=round(score, 3),
            passed=passed,
            reason=reason
        )

    def _mock_context_relevance(self, context: list, prompt: str) -> dict:
        prompt_words = self._content_words(prompt)

        if not prompt_words:
            return {
                "score": 1.0,
                "relevant_passages": len(context),
                "total_passages": len(context),
                "reason": "No content words extracted from prompt"
            }

        relevant_count = 0
        irrelevant_example = None

        for passage in context:
            passage_lower = passage.lower()
            words_present = sum(1 for w in prompt_words if w in passage_lower)
            coverage = words_present / len(prompt_words)

            if coverage >= self.passage_relevance_threshold:
                relevant_count += 1
            elif irrelevant_example is None:
                irrelevant_example = passage.strip()

        score = relevant_count / len(context)

        if score >= self.pass_threshold:
            reason = "Retrieved context is topically relevant to the prompt"
        else:
            reason = f"Irrelevant passage retrieved: '{irrelevant_example[:80]}'" if irrelevant_example else "Retrieved context does not match prompt topic"

        return {
            "score": score,
            "relevant_passages": relevant_count,
            "total_passages": len(context),
            "reason": reason
        }

    def _content_words(self, text: str) -> list:
        """
        KNOWN LIMITATION: this is a plain string-overlap proxy, not real
        NLP. Possessives are stripped ("store's" -> "store", "stores'" ->
        "store") since they're trivial and common in questions like
        "What is the store's policy?" - but plurals are NOT stemmed
        ("return" vs "returns" are treated as different words). Real
        stemming/lemmatization is out of scope for a mock evaluator; this
        mirrors the same documented-limitation posture as
        HallucinationEvaluator's negation-blindness and
        RelevanceEvaluator's identical plural-blindness limitation.
        Upgrade path: swap in a real NLP similarity model (e.g. embedding
        cosine similarity) when moving off mock.
        """
        words = [w.strip("?.,!").lower() for w in text.split()]
        words = [w[:-2] if w.endswith("'s") else w for w in words]
        words = [w[:-1] if w.endswith("s'") else w for w in words]
        return [w for w in words if len(w) > 3 and w not in self.STOP_WORDS]

    @staticmethod
    def _real_context_relevance(context: list, prompt: str) -> dict:
        raise NotImplementedError(
            "Real context relevance evaluation requires ANTHROPIC_API_KEY. "
            "Set use_mock=False only after configuring your API key. "
            "See README for setup instructions."
        )

    def _estimate_cost(self, context: list, prompt: str) -> float:
        prompt_text = " ".join(context) + prompt
        input_tokens = len(prompt_text) / 4
        output_tokens = 80
        return (
                (input_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS +
                (output_tokens / 1000) * self.COST_PER_1K_OUTPUT_TOKENS
        )

    def get_session_stats(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_latency_seconds": round(self.total_latency, 3)
        }