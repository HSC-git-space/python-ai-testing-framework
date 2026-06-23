# evaluators/base_evaluator.py

from abc import ABC, abstractmethod

from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.

    Every evaluator (keyword, length, tone, consistency, hallucination)
    must implement evaluate() and return an EvaluatorScore.

    Java equivalent: an interface with a single method, implemented
    by each concrete evaluator class.
    """

    @abstractmethod
    def evaluate(self, request: EvalRequest, response: str) -> EvaluatorScore:
        """
        Evaluate a single LLM response against the criteria in the request.

        Args:
            request: The EvalRequest containing the original prompt and
                      evaluation criteria (keywords, length bounds, tone, etc.)
            response: The raw text response returned by the LLM provider.

        Returns:
            EvaluatorScore: name of evaluator, numeric score, pass/fail,
                             and a human-readable reason.
        """
        raise NotImplementedError