import time
import anthropic
from config.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MAX_TOKENS
from models.eval_request import EvalRequest
from providers.base_provider import BaseProvider


class AnthropicProvider(BaseProvider):
    """
    Concrete implementation of BaseProvider for Anthropic Claude.
    All Anthropic-specific details are contained here.
    Nothing outside this file needs to know how Claude works.
    """

    def __init__(self):
        # Initialise the official Anthropic SDK client
        # This is the equivalent of creating a requests.Session() in Repo 1
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = ANTHROPIC_MODEL

    def complete(self, request: EvalRequest) -> tuple[str, float]:
        """
        Send prompt to Claude and return (response_text, latency_seconds)
        """
        # Start the timer before the API call
        start_time = time.perf_counter()

        message = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": request.prompt}
            ]
        )

        # Stop the timer immediately after response is received
        latency = time.perf_counter() - start_time

        # Extract the text from Anthropic's response structure
        # response.content is a list — [0] is the first content block
        # .text is the actual string inside that block
        response_text = message.content[0].text

        return response_text, latency