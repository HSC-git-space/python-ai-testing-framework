# evaluators/consistency_evaluator.py

from typing import List, Callable, Optional

from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class ConsistencyEvaluator(BaseEvaluator):
    """
    Checks whether N responses to the SAME prompt are consistent with each
    other — not identical text, but consistent in some measurable class
    of outcome (e.g. all positive sentiment, all containing the same key
    fact, all passing/failing the same classifier function).

    This evaluator does NOT fit the standard BaseEvaluator.evaluate(request,
    response: str) contract, because it inherently needs multiple responses
    to compare against each other rather than one response checked against
    static criteria. Forcing it into that shape would mean either changing
    the shared interface for every other evaluator, or smuggling response
    data through the request object — both worse than acknowledging this
    evaluator is structurally different.

    Real logic lives in evaluate_consistency(). The inherited evaluate()
    is implemented only to satisfy the abstract contract and intentionally
    raises NotImplementedError, with a message pointing callers to the
    correct method — eval_engine.py must call this evaluator differently
    from the other four.

    Core concept: non-determinism. Same prompt, different output every run.
    Cannot assert response == expected. Must assert output falls within
    an acceptable class across repeated runs.
    """

    def evaluate(self, request: EvalRequest, response: str) -> EvaluatorScore:
        raise NotImplementedError(
            "ConsistencyEvaluator does not support single-response evaluation. "
            "Call evaluate_consistency(request, responses, classifier_fn) instead."
        )

    def evaluate_consistency(
            self,
            request: EvalRequest,
            responses: List[str],
            classifier_fn: Callable[[str], str],
    ) -> EvaluatorScore:
        """
        Args:
            request: the EvalRequest used to generate all N responses
                      (same prompt, run multiple times).
            responses: list of raw response strings from N separate
                        LLM calls against the same prompt.
            classifier_fn: a function that maps a single response string
                            to a "class" label (e.g. "positive"/"negative",
                            or a category name). This is supplied by the
                            caller — ConsistencyEvaluator doesn't know what
                            consistency means for a given test case, it just
                            measures whether the classes agree.

        Returns:
            EvaluatorScore where score = fraction of responses that share
            the majority class. passed = True if ALL responses share the
            same class (full consistency required by default).
        """
        if len(responses) < 2:
            return EvaluatorScore(
                evaluator_name="consistency_evaluator",
                score=1.0,
                passed=True,
                reason=f"Only {len(responses)} response(s) provided; consistency check requires at least 2 — not applicable"
            )

        classes = [classifier_fn(r) for r in responses]

        class_counts = {}
        for c in classes:
            class_counts[c] = class_counts.get(c, 0) + 1

        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]

        score = majority_count / len(responses)
        passed = majority_count == len(responses)  # full agreement required

        return EvaluatorScore(
            evaluator_name="consistency_evaluator",
            score=score,
            passed=passed,
            reason=f"{majority_count}/{len(responses)} responses classified as '{majority_class}'; "
                   f"class distribution: {class_counts}"
        )