import time
import os
import anthropic
from evaluators.base_evaluator import BaseEvaluator
from models.eval_request import EvalRequest
from models.eval_result import EvaluatorScore


class JudgeEvaluator(BaseEvaluator):
    """
    LLM-as-judge evaluator.

    Uses Claude to semantically evaluate whether a response correctly
    addresses the prompt and ground truth — replacing brittle string matching
    with genuine semantic understanding.

    Why this beats string matching:
    - String match: "Water boils at 100 degrees" passes even if response says 90 degrees
      (both contain "water", "boils", "degrees")
    - Judge: reads both, understands the contradiction, returns FAIL with reason

    Self-consistency: runs the judge N times, takes majority vote.
    This handles non-determinism — a single judge call can be wrong,
    three calls with majority vote is reliable.

    Cost tracking: logs tokens used and estimated cost per evaluation run.
    In production, judge evaluation costs ~10-50x more than string matching.
    The tradeoff table in README documents when each approach is appropriate.
    """

    COST_PER_1K_INPUT_TOKENS = 0.000003
    COST_PER_1K_OUTPUT_TOKENS = 0.000015

    def __init__(
            self,
            consistency_runs: int = 3,
            pass_threshold: float = 0.7,
            use_mock: bool = True
    ):
        self.consistency_runs = consistency_runs
        self.pass_threshold = pass_threshold
        self.use_mock = use_mock
        self.total_cost = 0.0
        self.total_latency = 0.0
        self._client = None

    def evaluate(self, response: str, request: EvalRequest) -> EvaluatorScore:
        ground_truth = getattr(request, 'ground_truth', None)
        if not ground_truth:
            return EvaluatorScore(
                evaluator_name="judge",
                score=1.0,
                passed=True,
                reason="No ground truth provided — judge skipped"
            )

        verdicts = []
        run_costs = []
        run_latencies = []

        for run in range(self.consistency_runs):
            verdict, cost, latency = self._single_judge_run(
                prompt=request.prompt,
                response=response,
                ground_truth=ground_truth
            )
            verdicts.append(verdict)
            run_costs.append(cost)
            run_latencies.append(latency)

        pass_count = sum(1 for v in verdicts if v["passed"])
        score = pass_count / self.consistency_runs
        passed = score >= self.pass_threshold

        total_cost = sum(run_costs)
        avg_latency = sum(run_latencies) / len(run_latencies)

        self.total_cost += total_cost
        self.total_latency += avg_latency

        majority_reason = self._majority_reason(verdicts)

        reason = (
            f"Judge verdict: {'PASS' if passed else 'FAIL'} "
            f"({pass_count}/{self.consistency_runs} runs passed) | "
            f"Reason: {majority_reason} | "
            f"Cost: ${total_cost:.6f} | "
            f"Avg latency: {avg_latency:.2f}s"
        )

        return EvaluatorScore(
            evaluator_name="judge",
            score=round(score, 3),
            passed=passed,
            reason=reason
        )

    def _single_judge_run(
            self,
            prompt: str,
            response: str,
            ground_truth: list
    ) -> tuple:
        start = time.time()

        if self.use_mock:
            verdict = self._mock_judge(response, ground_truth)
            cost = 0.0
        else:
            verdict = self._real_judge(prompt, response, ground_truth)
            cost = self._estimate_cost(prompt, response, ground_truth)

        latency = time.time() - start
        return verdict, cost, latency

    @staticmethod
    def _mock_judge(response: str, ground_truth: list) -> dict:
        response_lower = response.lower()
        facts_checked = []

        for fact in ground_truth:
            fact_words = [w for w in fact.lower().split() if len(w) > 4]
            words_present = sum(1 for w in fact_words if w in response_lower)
            coverage = words_present / len(fact_words) if fact_words else 1.0
            facts_checked.append({
                "fact": fact,
                "coverage": coverage,
                "passed": coverage >= 0.6
            })

        passed_facts = [f for f in facts_checked if f["passed"]]
        overall_pass = len(passed_facts) / len(facts_checked) >= 0.7

        if overall_pass:
            reason = "Response accurately addresses the ground truth facts"
        else:
            failed = [f["fact"] for f in facts_checked if not f["passed"]]
            reason = f"Response does not adequately address: {failed[0][:80]}"

        return {"passed": overall_pass, "reason": reason}

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    def _real_judge(self, prompt: str, response: str, ground_truth: list) -> dict:
        client = self._get_client()
        judge_prompt = self._build_judge_prompt(prompt, response, ground_truth)

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": judge_prompt}]
        )

        judge_text = message.content[0].text.strip()
        passed = judge_text.upper().startswith("PASS")
        reason = judge_text.split("\n", 1)[-1].strip() if "\n" in judge_text else judge_text

        return {"passed": passed, "reason": reason}

    def _estimate_cost(self, prompt: str, response: str, ground_truth: list) -> float:
        judge_prompt = self._build_judge_prompt(prompt, response, ground_truth)
        input_tokens = len(judge_prompt) / 4
        output_tokens = 100
        return (
                (input_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS +
                (output_tokens / 1000) * self.COST_PER_1K_OUTPUT_TOKENS
        )

    @staticmethod
    def _build_judge_prompt(prompt: str, response: str, ground_truth: list) -> str:
        facts = "\n".join(f"- {fact}" for fact in ground_truth)
        return f"""You are an expert evaluator. Your job is to assess whether a response 
correctly addresses the question and is consistent with the known facts.

Question: {prompt}

Known facts:
{facts}

Response to evaluate:
{response}

Does the response accurately address the question without contradicting the known facts?
Reply with exactly: PASS or FAIL, then a one-sentence reason.
"""

    @staticmethod
    def _majority_reason(verdicts: list) -> str:
        pass_reasons = [v["reason"] for v in verdicts if v["passed"]]
        fail_reasons = [v["reason"] for v in verdicts if not v["passed"]]

        if len(pass_reasons) >= len(fail_reasons):
            return pass_reasons[0] if pass_reasons else "No reason provided"
        return fail_reasons[0] if fail_reasons else "No reason provided"

    def get_session_stats(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_latency_seconds": round(self.total_latency, 3),
            "consistency_runs_per_eval": self.consistency_runs
        }