import re

from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class HallucinationEvaluator(BaseEvaluator):
    """
    Checks a response against ground_truth facts to catch unsupported claims.

    Rule-based — not ML. Two signals:
    1. Key content words from the fact must appear in the response
    2. Negation words alongside fact words may indicate contradiction

    KNOWN LIMITATION: misses paraphrased contradictions. If ground truth says
    "the meeting is on Tuesday" and response says "the meeting is on a different day"
    — no shared negation word, so this evaluator misses it. Catching that class
    of error requires NLI models or LLM-as-judge, not string matching.
    This gap is why hallucination detection is a research problem, not a solved one.
    """

    # Stop words — these are ignored when checking key content words
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
        "at", "to", "for", "and", "or", "but", "it", "its", "be", "been",
        "has", "have", "had", "this", "that", "with", "from", "by"
    }

    NEGATION_WORDS = [
        "not", "isn't", "wasn't", "never", "didn't",
        "doesn't", "no longer", "cannot", "can't"
    ]

    def evaluate(self, response: str, request: EvalRequest) -> EvaluatorScore:
        facts = request.ground_truth or []

        if not facts:
            return EvaluatorScore(
                evaluator_name="hallucination_evaluator",
                score=1.0,
                passed=True,
                reason="No ground_truth facts configured; evaluator not applicable"
            )

        response_lower = response.lower()
        response_has_negation = any(neg in response_lower for neg in self.NEGATION_WORDS)

        supported_count = 0
        contradiction_found = False
        unsupported_facts = []

        for fact in facts:
            # Extract only content words — strip stop words
            all_words = set(re.findall(r"\w+", fact.lower()))
            content_words = all_words - self.STOP_WORDS

            if not content_words:
                supported_count += 1
                continue

            # ALL content words must appear in the response
            matched = sum(1 for w in content_words if w in response_lower)
            overlap_ratio = matched / len(content_words)

            if overlap_ratio >= 0.8:
                supported_count += 1
                if response_has_negation:
                    contradiction_found = True
            else:
                unsupported_facts.append(fact)

        if contradiction_found:
            return EvaluatorScore(
                evaluator_name="hallucination_evaluator",
                score=0.0,
                passed=False,
                reason="Possible contradiction detected: ground truth fact words "
                       "present alongside negation in response"
            )

        score = supported_count / len(facts)
        passed = score == 1.0

        reason = f"{supported_count}/{len(facts)} ground truth facts supported by response"
        if unsupported_facts:
            reason += f"; unsupported: {unsupported_facts}"

        return EvaluatorScore(
            evaluator_name="hallucination_evaluator",
            score=score,
            passed=passed,
            reason=reason
        )