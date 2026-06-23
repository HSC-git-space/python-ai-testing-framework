from abc import ABC, abstractmethod
from models.eval_request import EvalRequest


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Every provider must implement the complete() method.
    The evaluation engine only ever talks to this interface —
    it never knows whether Anthropic or OpenAI is underneath.
    """

    @abstractmethod
    def complete(self, request: EvalRequest) -> tuple[str, float]:
        """
        Send a prompt to the LLM and return the response.

        Args:
            request: EvalRequest object containing the prompt and config

        Returns:
            tuple of (response_text, latency_seconds)
            Every provider must return both — response and how long it took
        """
        pass