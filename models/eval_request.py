from pydantic import BaseModel, Field
from typing import List, Optional


class EvalRequest(BaseModel):
    # The prompt to send to the LLM
    prompt: str

    # Which provider to use — defaults to anthropic
    provider: str = "anthropic"

    # Which evaluators to run against the response
    # e.g. ["keyword", "length", "tone"]
    evaluators: List[str] = Field(default_factory=list)

    # Keywords that must be present in the response
    required_keywords: List[str] = Field(default_factory=list)

    # Keywords that must NOT be present in the response
    forbidden_keywords: List[str] = Field(default_factory=list)

    # Expected tone — "formal" or "informal"
    expected_tone: Optional[str] = None

    # Expected minimum and maximum response length in characters
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    # Ground truth facts for hallucination detection
    ground_truth: Optional[List[str]] = Field(default_factory=list)

    # Retrieved context/documents the response should be grounded in,
    # used by faithfulness and relevance evaluators (RAGAS-lite).
    # Distinct from ground_truth: ground_truth checks factual correctness
    # against reality; context checks whether the response is consistent
    # with what it was actually given to work from, regardless of
    # whether that context is itself correct.
    retrieved_context: Optional[List[str]] = Field(default_factory=list)

    # Test identifier — useful in batch eval runs and reports
    test_id: Optional[str] = None