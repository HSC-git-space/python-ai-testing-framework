import os
from dotenv import load_dotenv

from providers.anthropic_provider import AnthropicProvider
from models.eval_request import EvalRequest
from evaluators.hallucination_evaluator import HallucinationEvaluator
from evaluators.judge_evaluator import JudgeEvaluator

load_dotenv()

request = EvalRequest(
    prompt="At what temperature does water boil at standard atmospheric pressure?",
    provider="anthropic",
    ground_truth=["Water boils at 100 degrees Celsius at standard pressure"],
    evaluators=["hallucination", "judge"]
)

provider = AnthropicProvider()
response_text, latency = provider.complete(request)

print("=" * 60)
print("REAL LLM RESPONSE:")
print(response_text)
print(f"Latency: {latency:.2f}s")
print("=" * 60)

hallucination_evaluator = HallucinationEvaluator()
hallucination_result = hallucination_evaluator.evaluate(response_text, request)
print("\nHallucinationEvaluator result:")
print(hallucination_result)

judge_evaluator = JudgeEvaluator(use_mock=False)
judge_result = judge_evaluator.evaluate(response_text, request)
print("\nJudgeEvaluator result:")
print(judge_result)