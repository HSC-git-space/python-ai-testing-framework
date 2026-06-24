from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class LengthEvaluator(BaseEvaluator):
    """
    Checks whether the response length (in characters) falls within
    min_length and/or max_length bounds set on the EvalRequest.

    Only fires when at least one bound is configured. If neither is set,
    the evaluator is not applicable and passes by default.

    Score gives partial credit based on how close the response is to the
    configured bounds, for visibility in reports. passed is strictly
    boolean — true only if all configured bounds are satisfied.

    Java equivalent: a custom assertion like
    assertTrue(actual.length() >= min && actual.length() <= max),
    but returning a graded score object instead of throwing on failure.
    """

    def evaluate(self, response: str, request: EvalRequest) -> EvaluatorScore:
        min_length = request.min_length
        max_length = request.max_length

        # Fail fast on misconfigured test data — this is a test bug, not a product bug
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError(
                f"Invalid length constraints: min_length ({min_length}) "
                f"is greater than max_length ({max_length})"
            )

        # No constraints configured — not applicable, pass by default
        if min_length is None and max_length is None:
            return EvaluatorScore(
                evaluator_name="length_evaluator",
                score=1.0,
                passed=True,
                reason="No length constraints configured; evaluator not applicable"
            )

        actual_length = len(response)

        too_short = min_length is not None and actual_length < min_length
        too_long = max_length is not None and actual_length > max_length

        if too_short:
            score = min(actual_length / min_length, 1.0) if min_length > 0 else 1.0
            return EvaluatorScore(
                evaluator_name="length_evaluator",
                score=score,
                passed=False,
                reason=f"Response length {actual_length} is below min_length {min_length}"
            )

        if too_long:
            score = min(max_length / actual_length, 1.0) if actual_length > 0 else 0.0
            return EvaluatorScore(
                evaluator_name="length_evaluator",
                score=score,
                passed=False,
                reason=f"Response length {actual_length} exceeds max_length {max_length}"
            )

        return EvaluatorScore(
            evaluator_name="length_evaluator",
            score=1.0,
            passed=True,
            reason=f"Response length {actual_length} within bounds (min={min_length}, max={max_length})"
        )