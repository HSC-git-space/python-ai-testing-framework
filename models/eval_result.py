from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class EvaluatorScore(BaseModel):
    # Name of the evaluator that produced this score
    evaluator_name: str

    # Score between 0.0 and 1.0
    score: float

    # Whether this evaluator passed based on threshold
    passed: bool

    # Human readable reason for the score
    reason: str


class EvalResult(BaseModel):
    # The original prompt that was sent
    prompt: str

    # The raw response text from the LLM
    response: str

    # Which provider was used — anthropic or openai
    provider: str

    # How long the LLM took to respond in seconds
    latency_seconds: float

    # Individual score from each evaluator
    evaluator_scores: List[EvaluatorScore] = Field(default_factory=list)

    # Overall pass/fail — True only if all evaluators passed
    overall_passed: bool = False

    # Overall aggregate score across all evaluators
    overall_score: float = 0.0