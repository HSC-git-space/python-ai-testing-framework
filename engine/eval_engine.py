# engine/eval_engine.py

from typing import Dict, List

from models.eval_request import EvalRequest
from models.eval_result import EvalResult, EvaluatorScore

from providers.base_provider import BaseProvider
from providers.anthropic_provider import AnthropicProvider
from providers.openai_provider import OpenAIProvider

from evaluators import (
    BaseEvaluator,
    KeywordEvaluator,
    LengthEvaluator,
    ToneEvaluator,
    HallucinationEvaluator,
)
from evaluators.consistency_evaluator import ConsistencyEvaluator


class EvalEngine:
    """
    Orchestrates a single eval run: picks a provider, calls complete()
    (which returns both response text and latency), runs all configured
    evaluators against the response, and assembles one EvalResult.

    Provider and evaluator registries are built once in __init__ — not
    re-created on every run() call. Java equivalent: a service class with
    its dependencies wired in the constructor, not re-instantiated per request.

    Note: BaseProvider.complete() already returns (response_text, latency_seconds)
    as a tuple, so the engine does not time the call itself — that
    responsibility lives inside each provider implementation.

    consistency_evaluator is deliberately excluded from the standard
    evaluator registry used by run(), because it requires multiple responses
    to a repeated prompt, not a single response — see run_consistency_check()
    for that separate flow.
    """

    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {
            "anthropic": AnthropicProvider(),
            "openai": OpenAIProvider(),
        }

        self.evaluators: Dict[str, BaseEvaluator] = {
            "keyword_evaluator": KeywordEvaluator(),
            "length_evaluator": LengthEvaluator(),
            "tone_evaluator": ToneEvaluator(),
            "hallucination_evaluator": HallucinationEvaluator(),
        }

        # Held separately — not part of the standard evaluate() loop
        self.consistency_evaluator = ConsistencyEvaluator()

    def run(self, request: EvalRequest) -> EvalResult:
        provider = self.providers.get(request.provider)
        if provider is None:
            return EvalResult(
                prompt=request.prompt,
                response="",
                provider=request.provider,
                latency_seconds=0.0,
                evaluator_scores=[],
                overall_passed=False,
                overall_score=0.0,
            )

        response_text, latency_seconds = provider.complete(request)

        evaluator_scores: List[EvaluatorScore] = []

        for evaluator_name in request.evaluators:
            evaluator = self.evaluators.get(evaluator_name)

            if evaluator is None:
                # Unknown/misconfigured evaluator name — hard fail, don't silently skip
                evaluator_scores.append(
                    EvaluatorScore(
                        evaluator_name=evaluator_name,
                        score=0.0,
                        passed=False,
                        reason=f"Unknown evaluator '{evaluator_name}' — check for typo in request.evaluators"
                    )
                )
                continue

            evaluator_scores.append(evaluator.evaluate(request, response_text))

        if evaluator_scores:
            overall_score = sum(s.score for s in evaluator_scores) / len(evaluator_scores)
            overall_passed = all(s.passed for s in evaluator_scores)
        else:
            # No evaluators configured at all — response was fetched but nothing was checked
            overall_score = 1.0
            overall_passed = True

        return EvalResult(
            prompt=request.prompt,
            response=response_text,
            provider=request.provider,
            latency_seconds=latency_seconds,
            evaluator_scores=evaluator_scores,
            overall_passed=overall_passed,
            overall_score=overall_score,
        )

    def run_consistency_check(self, request: EvalRequest, n: int, classifier_fn) -> EvaluatorScore:
        """
        Calls the configured provider's complete() N times against the same
        request.prompt and checks consistency of the resulting responses
        using ConsistencyEvaluator.

        Separate from run() because this evaluator needs multiple responses
        from N repeated provider calls, not a single response.
        """
        provider = self.providers.get(request.provider)
        if provider is None:
            return EvaluatorScore(
                evaluator_name="consistency_evaluator",
                score=0.0,
                passed=False,
                reason=f"Unknown provider: '{request.provider}'"
            )

        responses = []
        for _ in range(n):
            response_text, _ = provider.complete(request)
            responses.append(response_text)

        return self.consistency_evaluator.evaluate_consistency(request, responses, classifier_fn)