import time
from openai import OpenAI
from config.config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS
from models.eval_request import EvalRequest
from providers.base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    Concrete implementation of BaseProvider for OpenAI.
    Same contract as AnthropicProvider — complete() takes an EvalRequest
    and returns (response_text, latency_seconds).
    The eval engine cannot tell the difference from the outside.
    """

    def __init__(self):
        # Initialise the official OpenAI SDK client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL

    def complete(self, request: EvalRequest) -> tuple[str, float]:
        """
        Send prompt to OpenAI and return (response_text, latency_seconds)
        """
        # Start the timer before the API call
        start_time = time.perf_counter()

        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": request.prompt}
            ]
        )

        # Stop the timer immediately after response is received
        latency = time.perf_counter() - start_time

        # Extract text from OpenAI's response structure
        # Different path from Anthropic — this is exactly why abstraction exists
        # Anthropic: message.content[0].text
        # OpenAI: completion.choices[0].message.content
        response_text = completion.choices[0].message.content