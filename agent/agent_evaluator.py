"""
agent_evaluator.py

Evaluates whether an agent's tool-calling behavior was CORRECT â€” a
different problem from judging a single text response. This does NOT
inherit BaseEvaluator (its evaluate() signature is locked to a plain
text response + EvalRequest, which doesn't fit an agent trace). Instead
this reuses EvaluatorScore as its return type so it still plugs into
existing aggregation/reporting.
"""

from typing import Any, Dict, List

from models.eval_result import EvaluatorScore


class AgentEvaluator:

    def check_tool_selection(
            self,
            trace: List[Dict[str, Any]],
            expected_tools: List[str],
    ) -> EvaluatorScore:
        called_tools = {step["tool_name"] for step in trace}
        expected_set = set(expected_tools)

        if not expected_set:
            passed = len(called_tools) == 0
            score = 1.0 if passed else 0.0
            reason = (
                "No tools were expected and none were called."
                if passed
                else f"No tools were expected, but agent called: {called_tools}"
            )
            return EvaluatorScore(
                evaluator_name="AgentToolSelection",
                score=score, passed=passed, reason=reason,
            )

        correctly_called = called_tools & expected_set
        score = len(correctly_called) / len(expected_set)
        passed = called_tools == expected_set

        missing = expected_set - called_tools
        unexpected = called_tools - expected_set
        reason_parts = []
        if missing:
            reason_parts.append(f"missing expected tool(s): {missing}")
        if unexpected:
            reason_parts.append(f"called unexpected tool(s): {unexpected}")
        reason = "All expected tools were called correctly." if passed else "; ".join(reason_parts)

        return EvaluatorScore(
            evaluator_name="AgentToolSelection",
            score=score, passed=passed, reason=reason,
        )

    def check_argument_correctness(
            self,
            trace: List[Dict[str, Any]],
            expected_args_by_tool: Dict[str, Dict[str, Any]],
    ) -> EvaluatorScore:
        relevant_calls = [step for step in trace if step["tool_name"] in expected_args_by_tool]

        if not relevant_calls:
            return EvaluatorScore(
                evaluator_name="AgentArgumentCorrectness",
                score=0.0, passed=False,
                reason="No tool calls matched any tool with an expected argument set.",
            )

        correct_count = 0
        mismatches = []
        for step in relevant_calls:
            tool_name = step["tool_name"]
            actual_args = step["tool_input"]
            expected_args = expected_args_by_tool[tool_name]
            if actual_args == expected_args:
                correct_count += 1
            else:
                mismatches.append(f"{tool_name}: expected {expected_args}, got {actual_args}")

        score = correct_count / len(relevant_calls)
        passed = score == 1.0
        reason = "All tool arguments matched expectations exactly." if passed else "; ".join(mismatches)

        return EvaluatorScore(
            evaluator_name="AgentArgumentCorrectness",
            score=score, passed=passed, reason=reason,
        )

    def check_sequence(
            self,
            trace: List[Dict[str, Any]],
            expected_order: List[str],
    ) -> EvaluatorScore:
        actual_order = [step["tool_name"] for step in trace]
        passed = actual_order == expected_order
        score = 1.0 if passed else 0.0
        reason = (
            "Tool call sequence matched expected order exactly."
            if passed
            else f"Expected order {expected_order}, got {actual_order}"
        )
        return EvaluatorScore(
            evaluator_name="AgentSequenceCorrectness",
            score=score, passed=passed, reason=reason,
        )

    def check_final_answer_uses_tool_output(
            self,
            trace: List[Dict[str, Any]],
            final_answer: str,
    ) -> EvaluatorScore:
        if not trace:
            return EvaluatorScore(
                evaluator_name="AgentFinalAnswerGrounding",
                score=0.0, passed=False,
                reason="No tool calls in trace â€” nothing to ground the answer against.",
            )

        grounded_count = 0
        missing_outputs = []
        for step in trace:
            output_str = str(step["tool_output"])
            if output_str in final_answer:
                grounded_count += 1
            else:
                missing_outputs.append(f"{step['tool_name']} output '{output_str}' not found in final answer")

        score = grounded_count / len(trace)
        passed = score == 1.0
        reason = "Final answer reflects all tool outputs." if passed else "; ".join(missing_outputs)

        return EvaluatorScore(
            evaluator_name="AgentFinalAnswerGrounding",
            score=score, passed=passed, reason=reason,
        )

    def evaluate(
            self,
            trace: List[Dict[str, Any]],
            final_answer: str,
            expected_tools: List[str],
            expected_args_by_tool: Dict[str, Dict[str, Any]],
            expected_order: List[str],
    ) -> List[EvaluatorScore]:
        return [
            self.check_tool_selection(trace, expected_tools),
            self.check_argument_correctness(trace, expected_args_by_tool),
            self.check_sequence(trace, expected_order),
            self.check_final_answer_uses_tool_output(trace, final_answer),
        ]
