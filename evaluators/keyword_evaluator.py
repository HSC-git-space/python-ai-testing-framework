# evaluators/keyword_evaluator.py

from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class KeywordEvaluator(BaseEvaluator):
    """
    Checks a response for required keywords and forbidden keywords.

    Score = fraction of required keywords found in the response.
    A single forbidden keyword match forces score to 0.0 and passed to False,
    regardless of how many required keywords were found.

    If no required_keywords are configured for this request, the evaluator
    is treated as not applicable and passes by default with score 1.0.

    Java equivalent: a custom assertion helper that loops over expected
    substrings, but returns a graded score object instead of a boolean assert.
    """

    def evaluate(self, response: str, request: EvalRequest) -> EvaluatorScore:
        response_lower = response.lower()

        # Forbidden keywords are a hard fail — checked first, short-circuits everything else
        forbidden_hits = [
            kw for kw in (request.forbidden_keywords or [])
            if kw.lower() in response_lower
        ]
        if forbidden_hits:
            return EvaluatorScore(
                evaluator_name="keyword_evaluator",
                score=0.0,
                passed=False,
                reason=f"Forbidden keyword(s) found: {', '.join(forbidden_hits)}"
            )

        required = request.required_keywords or []

        # No required keywords configured — not applicable, pass by default
        if not required:
            return EvaluatorScore(
                evaluator_name="keyword_evaluator",
                score=1.0,
                passed=True,
                reason="No required keywords configured; evaluator not applicable"
            )

        found = [kw for kw in required if kw.lower() in response_lower]
        score = len(found) / len(required)
        passed = score == 1.0

        return EvaluatorScore(
            evaluator_name="keyword_evaluator",
            score=score,
            passed=passed,
            reason=f"{len(found)}/{len(required)} required keywords found; no forbidden keywords present"
        )