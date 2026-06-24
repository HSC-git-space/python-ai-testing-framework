# evaluators/tone_evaluator.py

from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class ToneEvaluator(BaseEvaluator):
    """
    Checks whether a response matches an expected tone: "formal" or "informal".

    Rule-based heuristic using word lists — NOT semantic/ML-based.
    Limitation: can be fooled by phrasing that doesn't use listed markers
    but still reads as the wrong tone (e.g. formal vocabulary used sarcastically).
    Uses simple substring matching, which can produce false positives
    (e.g. "hi" could match inside a longer word) — accepted tradeoff for simplicity.

    Only fires when request.expected_tone is set. If no tone markers of
    either kind are found in the response, the result is treated as
    inconclusive and fails — a quality gate should not pass on ambiguous
    signal, even though it found no evidence of failure either.

    Java equivalent: a rule-based keyword classifier, similar in spirit to
    a regex-based content filter — useful but explicitly not as robust as
    a trained NLP classifier.
    """

    INFORMAL_MARKERS = [
        "don't", "can't", "won't", "isn't", "didn't", "gonna", "wanna",
        "hey", "yeah", "lol", "awesome", "cool", "stuff", "kinda", "sorta",
        "guys", "totally", "super ", "haha"
    ]

    FORMAL_MARKERS = [
        "furthermore", "therefore", "however", "pursuant to", "please find",
        "i would like to", "we are pleased to", "kindly", "in accordance with",
        "consequently", "nevertheless", "respectfully", "sincerely"
    ]

    PASS_THRESHOLD = 0.6

    def evaluate(self, response: str, request: EvalRequest) -> EvaluatorScore:
        expected_tone = request.expected_tone

        if not expected_tone:
            return EvaluatorScore(
                evaluator_name="tone_evaluator",
                score=1.0,
                passed=True,
                reason="No expected_tone configured; evaluator not applicable"
            )

        response_lower = response.lower()

        informal_count = sum(1 for marker in self.INFORMAL_MARKERS if marker in response_lower)
        formal_count = sum(1 for marker in self.FORMAL_MARKERS if marker in response_lower)

        if informal_count == 0 and formal_count == 0:
            return EvaluatorScore(
                evaluator_name="tone_evaluator",
                score=0.5,
                passed=False,
                reason=f"No tone markers detected for either formal or informal; "
                       f"cannot confirm expected tone '{expected_tone}' — treated as inconclusive fail"
            )

        if expected_tone == "informal":
            matches_expected = informal_count
            matches_opposite = formal_count
        elif expected_tone == "formal":
            matches_expected = formal_count
            matches_opposite = informal_count
        else:
            raise ValueError(f"Unsupported expected_tone value: '{expected_tone}' (expected 'formal' or 'informal')")

        score = matches_expected / (matches_expected + matches_opposite)
        passed = score >= self.PASS_THRESHOLD

        return EvaluatorScore(
            evaluator_name="tone_evaluator",
            score=score,
            passed=passed,
            reason=f"Expected tone '{expected_tone}': {matches_expected} matching marker(s), "
                   f"{matches_opposite} opposite-tone marker(s) found"
        )