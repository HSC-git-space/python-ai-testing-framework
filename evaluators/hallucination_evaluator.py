# evaluators/hallucination_evaluator.py

import re

from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class HallucinationEvaluator(BaseEvaluator):
    """
    Checks a response against a list of ground_truth facts (request.ground_truth:
    List[str]) to catch unsupported or contradicted claims.

    This is RULE-BASED, not ML-based. It is NOT a real hallucination detector
    in the research sense (no NLI model, no semantic entailment, no RAGAS-style
    faithfulness scoring). It can only catch two narrow signals:

      1. Omission — a ground truth fact's key words don't appear in the
         response at all (the response simply didn't address it).
      2. Naive negation — a fact's words mostly appear, but a negation word
         (not, isn't, never, didn't) also appears somewhere in the response,
         which MIGHT indicate the fact was contradicted.

    KNOWN LIMITATION (important for interviews): this evaluator misses
    paraphrased contradictions. If ground truth says "the meeting is on
    Tuesday" and the response says "the meeting is definitely on a different
    day," there's no shared negation word and no overlapping fact words to
    flag — this evaluator would call that "unsupported" at best, not
    "contradicted," and could miss it entirely if enough other words overlap.
    Catching that class of error requires semantic entailment (NLI models)
    or LLM-as-judge approaches, not substring/word-overlap heuristics.
    This gap is the entire reason hallucination detection is a research
    problem and not a solved string-matching problem.

    Only fires when request.ground_truth is non-empty.
    Score = supported_facts / total_facts. Any detected contradiction
    forces score to 0.0 — a contradiction is worse than an omission.
    """

    WORD_OVERLAP_THRESHOLD = 0.7
    NEGATION_WORDS = ["not", "isn't", "wasn't", "never", "didn't", "doesn't", "no longer", "cannot", "can't"]

    def evaluate(self, request: EvalRequest, response: str) -> EvaluatorScore:
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
            fact_words = set(re.findall(r"\w+", fact.lower()))
            if not fact_words:
                continue

            matched_words = sum(1 for w in fact_words if w in response_lower)
            overlap_ratio = matched_words / len(fact_words)

            if overlap_ratio >= self.WORD_OVERLAP_THRESHOLD:
                supported_count += 1
                # Fact's words are present, but response also contains negation —
                # narrow, explainable signal that this MIGHT be a contradiction
                if response_has_negation:
                    contradiction_found = True
            else:
                unsupported_facts.append(fact)

        if contradiction_found:
            return EvaluatorScore(
                evaluator_name="hallucination_evaluator",
                score=0.0,
                passed=False,
                reason="Possible contradiction detected: a ground truth fact's words "
                       "are present alongside a negation word in the response"
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