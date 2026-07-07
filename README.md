# python-ai-testing-framework

![CI](https://img.shields.io/badge/CI-passing-brightgreen)

An AI evaluation framework for testing LLM responses built with Python, pytest, and the Anthropic API.

Every design decision has a reason. Every file can be explained and defended.

## Background

Standard API testing is deterministic - send input X, expect output Y, assert equality. LLMs break that model. The same prompt can return different responses on every run. You cannot assert exact equality. You need a different approach - score-based evaluation, threshold-based pass/fail, and probabilistic assertions across multiple runs.

This framework is built around that mental model.

## What Is Built

- A provider abstraction layer so evaluation logic works with any LLM - currently Anthropic Claude and OpenAI
- Ten evaluation modules - keyword presence, response length, tone classification, output consistency, rule-based hallucination detection, LLM-as-judge semantic evaluation, a RAGAS-lite trio (faithfulness, answer relevance, context relevance) for RAG pipeline evaluation, and an embedding-based semantic relevance evaluator that upgrades the word-overlap relevance check to catch paraphrases
- An evaluation engine that wires providers and evaluators together into one structured result per run
- Dataset-driven evaluation using CSV and Excel
- Structured logging on every LLM call - prompt sent, response received, latency, eval scores
- Two real agents evaluated by one framework-agnostic AgentEvaluator - a deterministic MockAgent and a real LangGraph agent making genuine LLM tool-calling decisions against live stock market data
- A DeepEval integration wrapping AnswerRelevancy and Faithfulness metrics around the real LangGraph agent's output
- Docker for running the framework in a clean isolated environment
- GitHub Actions CI - tests run automatically on every push

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pytest | Test runner |
| Anthropic SDK | Primary LLM provider |
| OpenAI SDK | Secondary LLM provider |
| LangGraph | Explicit state-machine orchestration for the real stock agent |
| DeepEval | Third-party LLM evaluation framework, wrapped around the real agent's output |
| yfinance | Live stock/index market data - free, real, no API key needed |
| Pydantic | Input and output model validation |
| Pandas | Dataset loading - CSV and Excel |
| sentence-transformers | Embedding model for semantic similarity |
| scikit-learn | Cosine similarity computation |
| python-dotenv | Environment variable configuration |
| pytest-html | HTML test report |
| Docker | Isolated test execution environment |
| GitHub Actions | CI pipeline |

## Project Structure

```
python-ai-testing-framework/
|
|-- config/
|   `-- config.py               # All config in one place - API keys, model name, thresholds
|
|-- models/
|   |-- eval_request.py         # Pydantic - defines what goes into an eval run
|   `-- eval_result.py          # Pydantic - defines what comes out of an eval run
|
|-- providers/
|   |-- base_provider.py        # Abstract base class - the contract every provider must follow
|   |-- anthropic_provider.py   # Claude implementation
|   `-- openai_provider.py      # OpenAI implementation
|
|-- evaluators/
|   |-- base_evaluator.py       # Abstract base class - the contract every evaluator must follow
|   |-- keyword_evaluator.py    # Required and forbidden keyword checks
|   |-- length_evaluator.py     # Response length validation
|   |-- tone_evaluator.py       # Formal vs informal tone classification
|   |-- consistency_evaluator.py # Output consistency across N repeated runs
|   |-- hallucination_evaluator.py # Rule-based fact checking against known ground truth
|   |-- judge_evaluator.py      # LLM-as-judge semantic evaluation with self-consistency voting
|   |-- faithfulness_evaluator.py # RAGAS-lite - checks response claims against retrieved context
|   |-- relevance_evaluator.py  # RAGAS-lite - checks response addresses the prompt
|   |-- semantic_relevance_evaluator.py # Embedding-based upgrade to relevance - catches paraphrases word-overlap misses
|   `-- context_relevance_evaluator.py # RAGAS-lite - checks retrieved context is relevant to the prompt
|
|-- engine/
|   `-- eval_engine.py          # Wires providers and evaluators together per eval run
|
|-- agent/
|   |-- agent_evaluator.py      # Framework-agnostic tool-calling correctness evaluator - used by both agents below
|   |-- agent_runner.py         # Original MockAgent - keyword matching over weather/temperature/population tools
|   |-- tools.py                # Original MockAgent's tools - unchanged, not part of the LangGraph rebuild
|   |-- stock_tools.py          # Real yfinance-backed tools - current price, historical price, index level
|   `-- langgraph_agent_runner.py # LangGraph rebuild - real Claude tool-calling decision, real tool execution, real grounded final answer
|
|-- utils/
|   |-- logger.py               # Structured logging setup
|   |-- timer.py                # Latency tracking utility
|   `-- data_loader.py          # CSV and Excel loader
|
|-- datasets/
|   |-- prompts.csv             # Eval prompts with expected behaviour and constraints
|   `-- ground_truth.xlsx       # Known facts used by the hallucination evaluator
|
|-- tests/
|   |-- test_keyword_eval.py    # Unit tests for keyword evaluator
|   |-- test_length_eval.py     # Unit tests for length evaluator
|   |-- test_tone_eval.py       # Unit tests for tone evaluator
|   |-- test_hallucination.py   # Unit tests for hallucination evaluator
|   |-- test_judge_eval.py      # Unit tests for judge evaluator - includes before/after comparison
|   |-- test_faithfulness_evaluator.py # Unit tests for faithfulness evaluator, including faithfulness-vs-hallucination distinction
|   |-- test_relevance_evaluator.py # Unit tests for relevance evaluator, including possessive-stripping regression test
|   |-- test_semantic_relevance_evaluator.py # Unit tests for semantic relevance evaluator, pinned against three real calibration cases
|   |-- test_context_relevance_evaluator.py # Unit tests for context relevance evaluator, including the pinned retrieval-bug example
|   |-- test_agent_eval.py      # Unit tests for AgentEvaluator against the original MockAgent - fully offline, no API key needed
|   |-- test_stock_agent_eval.py # Real end-to-end tests for the LangGraph stock agent - hits live Claude and live yfinance on every run
|   |-- test_deepeval_stock_agent.py # DeepEval AnswerRelevancy/Faithfulness wrapper around the same LangGraph stock agent
|   |-- test_consistency_eval.py # Integration tests - needs live API key
|   `-- test_non_determinism.py  # Integration tests - needs live API key
|
|-- conftest.py                 # Session-scoped fixtures and logging
|-- pytest.ini                  # Test paths, markers, report config
|-- Dockerfile                  # Container setup for running the framework
|-- .env.example                # Shows required environment variables - no real keys committed
`-- requirements.txt            # All dependencies
```

## How the Evaluation Flow Works

```
EvalRequest (Pydantic)
        |
        v
EvalEngine.run()
        |
        |-- providers[request.provider].complete(request)
        |       `-- Returns (response_text, latency_seconds)
        |
        `-- for each evaluator in request.evaluators:
                evaluator.evaluate(response_text, request)
                        `-- Returns EvaluatorScore(evaluator_name, score, passed, reason)
                |
                v
        EvalResult (Pydantic)
                `-- prompt, response, provider, latency, evaluator_scores, overall_passed, overall_score
```

The engine does not know which provider it is talking to. It does not know which evaluator it is running. It only talks to the `BaseProvider` and `BaseEvaluator` abstract interfaces. Swap Claude for GPT-4 by changing one config value - no eval logic changes.

## The Evaluators

### KeywordEvaluator

Checks required keywords are present in the response and forbidden keywords are absent. Case insensitive. Score is the proportion of keyword conditions satisfied.

### LengthEvaluator

Validates response character count against `min_length` and `max_length` from the request. Returns a perfect score if no bounds are configured - the evaluator only fires when constraints exist.

### ToneEvaluator

Rule-based classification. Counts formal indicator words - *therefore, furthermore, consequently* - against informal indicator words - *hey, gonna, yeah*. Whichever list has more matches determines the detected tone. Known limitation: mixed-tone responses can produce incorrect classifications.

### ConsistencyEvaluator

Runs the same prompt N times. Applies a classifier function to each response. Score is the proportion of runs that matched the expected output class. This tests whether the LLM produces stable output across repeated runs - not whether any single response is correct.

### HallucinationEvaluator

Checks the response against known facts in `EvalRequest.ground_truth`. Strips stop words, extracts content words, checks whether enough content words from each fact appear in the response.

**Known limitation - important:** This evaluator cannot detect numeric contradictions or paraphrased errors. If ground truth says "Water boils at 100 degrees Celsius" and the response says "Water boils at 90 degrees" - both contain the words *water, boils, degrees* - so the overlap check passes even though the value is wrong. This is the exact failure mode the JudgeEvaluator was built to fix.

### JudgeEvaluator

Uses Claude to semantically evaluate whether a response correctly addresses the prompt and is consistent with known ground truth facts - replacing brittle string matching with genuine semantic understanding.

The concrete before/after story:

```
Ground truth: "Water boils at 100 degrees Celsius at standard pressure"
Response:     "Water boils at 90 degrees Celsius."

HallucinationEvaluator result: PASS
Reason: response contains "water", "boils", "degrees" - string overlap sufficient

JudgeEvaluator result: FAIL
Reason: Response states 90 degrees, which contradicts the known fact of 100 degrees
```

This is the exact failure mode documented in the hallucination evaluator's known limitation section. The judge catches what string matching cannot.

**Self-consistency voting:**

A single LLM judge call can itself be non-deterministic. The JudgeEvaluator runs the judge N times (default 3) and takes the majority vote. This means the evaluation result is stable even when individual judge calls vary.

```python
# 3 judge runs, majority vote
verdicts = [PASS, PASS, FAIL]  # 2/3 pass
score = 0.667
passed = True  # above 0.7 threshold? No - FAIL in this case
```

**Cost and latency tracking:**

Every evaluation run logs cost (estimated from token counts) and latency. This matters in production - a judge call costs approximately 10-50x more than a string match call. The tradeoff table below documents when each approach is appropriate.

### FaithfulnessEvaluator - RAGAS-lite

Checks whether the response's claims are grounded in `EvalRequest.retrieved_context` - distinct from HallucinationEvaluator, which checks against `ground_truth`. Splits the response into claim-sentences and checks word-overlap coverage per claim against the combined context.

Key distinction: faithfulness can PASS even if the context itself is stale or wrong - it only checks "did the model stick to what it was given," not "is what it was given true."

```
Context:  "The store closes at 6pm." (stale/wrong)
Response: "The store closes at 6pm."

Faithfulness: PASS - response is grounded in its source, even though the
source itself may be factually outdated. Faithfulness measures grounding,
not truth.
```

### RelevanceEvaluator - RAGAS-lite

Checks whether the response addresses `EvalRequest.prompt` - independent of whether it's correct or grounded in context. Uses content-word overlap between prompt and response as a topic-match proxy.

Key distinction: a response can be 100% faithful to context while being totally irrelevant - e.g. it answers a different question using only material that was present in the context.

```
Context:  ["The store closes at 8pm.", "Returns are accepted within 30 days."]
Prompt:   "What is the store's return policy?"
Response: "The store closes at 8pm."

Faithfulness: PASS (claim is grounded)
Relevance:    FAIL (doesn't answer the actual question asked)
```

Known limitation: possessives are stripped ("store's" -> "store") but plurals are not stemmed ("return" != "returns") - the same documented-limitation posture as HallucinationEvaluator's negation-blindness. Real stemming is out of scope for a string-overlap mock; SemanticRelevanceEvaluator below is the embedding-based upgrade path for the paraphrase side of this problem.

### SemanticRelevanceEvaluator

Embedding-based upgrade to RelevanceEvaluator's word-overlap approach. Uses sentence embeddings (all-MiniLM-L6-v2) and cosine similarity to check whether the response addresses the prompt - catching paraphrases and synonymy that pure word-overlap misses entirely.

Key distinction: RelevanceEvaluator fails on valid paraphrases that share no vocabulary. This evaluator catches that case by comparing meaning, not literal words.

```
Prompt:   "What time does the store close?"
Response: "The shop shuts at 6pm."

RelevanceEvaluator (word-overlap):    FAIL - "store"/"shop", "close"/"shuts"
                                       share no literal words
SemanticRelevanceEvaluator (cosine):  PASS - similarity 0.67, correctly
                                       recognizes this as the same meaning
```

Calibration (three real runs against all-MiniLM-L6-v2, threshold set at 0.6 based on these observed scores):

```
Easy paraphrase, no shared words:              0.67 -> PASS
Hard paraphrase, different structure/length:   0.75 -> PASS
Adversarial: shared vocabulary, opposite
meaning ("refund in 5-7 days" vs
"refunds not available"):                      0.53 -> FAIL
```

Known limitation: this model captures topical relatedness more strongly than logical negation. Two sentences that share vocabulary but directly contradict each other can still score moderately (~0.5), since embedding similarity responds to topic overlap as well as meaning - the 0.6 threshold is chosen specifically to separate this adversarial case from genuine paraphrases, not an arbitrary default.

Note on mocking: unlike every other evaluator in this framework, this one has no meaningful mock/real split. It always runs real local inference - no API call, no cost either way - so a `use_mock` flag would be fake consistency rather than honest design, and is deliberately omitted.

### ContextRelevanceEvaluator - RAGAS-lite

Checks whether `retrieved_context` itself is relevant to the prompt - a retrieval-quality check, not a generation-quality check. Scores the proportion of retrieved passages that individually clear a content-word overlap threshold against the prompt.

Key distinction: this isolates which half of a RAG pipeline is broken. A response can be honest and faithful while the retriever pulled garbage - that failure mode is invisible to the other two evaluators.

```
Prompt:   "What is the store's return policy?"
Context:  ["The store was founded in 1998.", "The founder's name is John Smith."]
Response: "I don't have information on the return policy."

Faithfulness:      PASS (invents nothing)
Relevance:         arguably okay (honest about not knowing)
Context relevance: FAIL - the retriever pulled completely unrelated
                    passages. The LLM did nothing wrong; retrieval did.
```

Why three RAGAS-lite evaluators instead of one combined score: a single blended "RAG quality" score can't tell you which component broke. Splitting faithfulness, relevance, and context relevance into independent axes mirrors how a real RAG pipeline gets debugged: is retrieval good (context relevance) -> did the model stay faithful to it (faithfulness) -> does the final answer address the ask (relevance).

Known limitation: same string-overlap content-word proxy as RelevanceEvaluator, including the possessive-stripping fix and unstemmed-plurals tradeoff. `passage_relevance_threshold` (0.5) is a coarse per-passage cutoff - on short prompts with very few content words, a single incidental word match can push a passage close to the threshold. Tuned empirically, not derived from a principled formula.

### AgentEvaluator

Evaluates tool-calling correctness for an agent - a fundamentally different testing problem from text-in/text-out evaluation. Checks tool selection correctness, argument correctness, call-sequence correctness for multi-tool queries, and whether the agent's final answer is actually grounded in the tool results it received (catches hallucinated numbers even when the right tool was called correctly).

**AgentEvaluator itself is framework-agnostic and required zero changes** across two very different agents built against it - it only operates on a plain trace (a list of `{tool_name, tool_input, tool_output}` dicts) plus a final answer string. Whatever produces that trace shape is irrelevant to this file.

**Agent 1 - original MockAgent** (`agent/agent_runner.py`): keyword-matching over a small weather/temperature/population toolset. Fully deterministic, no API key needed, tested in `test_agent_eval.py`.

**Agent 2 - LangGraph rebuild** (`agent/langgraph_agent_runner.py`): a real, non-deterministic agent built as an explicit LangGraph state machine over a stock/index domain (Sensex, Nifty, individual stocks via yfinance):

```
START -> decide_node -> execute_tools_node -> final_answer_node -> END
```

- `decide_node` calls Claude with real tool definitions via Claude's native tool-calling API - this is the actual "agent," replacing MockAgent's keyword matching with a genuine LLM decision.
- `execute_tools_node` calls real functions in `stock_tools.py` against live yfinance data - no mocking, since yfinance is free and real at zero marginal cost, the same philosophy as SemanticRelevanceEvaluator. If a tool_name doesn't match a real function, the trace records an error string rather than skipping silently or crashing, so AgentEvaluator can see and grade the bad decision instead of it being lost.
- `final_answer_node` calls Claude again to compose an answer grounded in the real tool outputs collected in the trace.

Because `decide_node` is a real LLM call, its tool choice is not perfectly deterministic - for an ambiguous query like "what is the Sensex right now," Claude may reasonably call either `get_index_level` or `get_current_price` on different runs, both valid. `test_stock_agent_eval.py` handles this with tolerance-scored assertions on tool choice, while keeping numeric grounding (does the final answer contain the real tool output) strict and binary, since there is no meaningful "partial credit" for citing a wrong number. This tolerance logic lives only in the test file - `agent_evaluator.py` itself stays strict and untouched, since it is the reusable, framework-agnostic grading core used by both agents and any future one.

Real example, run against live data:

```
Query: "Compare the Sensex today versus a month ago."

trace:
  get_index_level(sensex) -> 78285.07
  get_historical_price(^BSESN, 1mo) -> 73524.26

final_answer: "The Sensex has risen by 4760.81 points (~6.48%) over the
past month..."
```

Both the tool outputs and the derived percentage change in the final answer are real, checked against each other, not hallucinated.

**Known issue found and fixed during this build:** the first version of `final_answer_node`'s prompt let Claude format numbers naturally for a human reader (e.g. `78,401.86` with a comma). `check_final_answer_uses_tool_output` does a literal string search for `str(tool_output)` (e.g. `78401.86`, no comma) inside the final answer - so a correct, human-readable answer was failing a strict grounding check purely on formatting. Fixed by constraining `final_answer_node`'s prompt to state numbers with no thousand-separators, rather than loosening the evaluator - `agent_evaluator.py` is reused across both agents and future ones, so the agent's output should conform to the grading contract, not the other way around.

### DeepEval Integration

Wraps DeepEval's `AnswerRelevancyMetric` and `FaithfulnessMetric` around the LangGraph stock agent's real output (`test_deepeval_stock_agent.py`), using DeepEval's built-in `AnthropicModel` (model `claude-sonnet-4-6`, same model used everywhere else in this repo) as the judge, rather than DeepEval's OpenAI default - keeping the repo to one LLM provider dependency instead of introducing a second one just for this wrapper.

`retrieval_context` for each test case is built from the agent's own `trace` - each `tool_output` turned into a string - since there is no traditional RAG document store here, only real tool call results. Both metrics use DeepEval's own common convention threshold of 0.7.

```
Query: "What is the Sensex right now?"

retrieval_context: ["get_index_level({'index_name': 'sensex'}) returned 78285.07"]
actual_output:      "The Sensex is currently at 78285.07."

AnswerRelevancyMetric:  PASS (answer addresses the question asked)
FaithfulnessMetric:     PASS (answer sticks to the real tool output, no invention)
```

**Honest positioning:** this integration does not prove anything new about mock-vs-live coverage - that gap was already closed by the LangGraph rebuild's own test suite (`test_stock_agent_eval.py`), which runs the real agent end to end and checks it against `agent_evaluator.py`. DeepEval is a recognized, named third-party evaluation library; wrapping it around cases already picked closes an ATS/resume-keyword gap, not a competence gap. Both are legitimate reasons to have it in a repo, but they are different reasons, and this README doesn't pretend it's the latter.

**Cost note:** each test in this file triggers real API calls at every layer - the agent's own `decide_node` and `final_answer_node` calls, plus DeepEval's own judge model call per metric per test case. This is deliberately run in CI on every push in the current setup, which is a real, accepted cost/latency tradeoff for a solo, actively-developed project - not the recommended pattern at team/production scale, where slow, real-API-hitting evals are normally tiered onto a schedule or manual trigger rather than gating every commit.

## Design Decisions and Tradeoffs

### When to use string matching vs LLM-as-judge

| Factor | String Match (HallucinationEvaluator) | LLM-as-Judge (JudgeEvaluator) |
|---|---|---|
| Cost per evaluation | Near zero | $0.0001-$0.001 per run |
| Latency | < 1ms | 1-5 seconds |
| Catches numeric errors | No | Yes |
| Catches paraphrased errors | No | Yes |
| Catches keyword presence | Yes | Yes |
| Non-determinism | None | Requires self-consistency |
| Scales to 10,000 evals | Yes | Expensive |
| Best for | Keyword gates, format checks, fast CI | Factual accuracy, semantic correctness |

Decision rule: Use string matching as a fast first gate. Use the judge only when semantic accuracy matters and the cost is justified. In CI, string matching runs on every push. Judge evaluation runs on scheduled or pre-release checks.

### When to use word-overlap vs embedding-based relevance

| Factor | Word-Overlap (RelevanceEvaluator) | Embeddings (SemanticRelevanceEvaluator) |
|---|---|---|
| Cost per evaluation | Zero | Zero (local inference) |
| Latency | < 1ms | ~10-50ms (CPU, local model) |
| Catches literal keyword matches | Yes | Yes |
| Catches paraphrases with no shared vocabulary | No | Yes |
| Catches negation / contradiction | Partially (shared words still overlap) | Weakly (shares the same blind spot) |
| Needs a model download | No | Yes - one-time (~90MB) |
| Best for | Fast CI gate, exact-phrasing checks | Catching real paraphrase-shaped relevance failures |

Decision rule: neither evaluator is a strict upgrade over the other - they fail differently. Run both. RelevanceEvaluator is the near-zero-cost first pass; SemanticRelevanceEvaluator catches the paraphrase cases the word-overlap check is structurally blind to. Neither reliably catches negation, which is why that limitation is documented on both rather than treated as solved by adding embeddings.

### Why self-consistency voting

A single judge call can return different verdicts on the same input across runs. Running the judge 3 times and taking the majority vote reduces variance without a significant cost increase. Three runs at claude-haiku pricing costs approximately $0.0003-0.001 per evaluation - acceptable for pre-release quality gates.

### Why cost tracking

In production evaluation pipelines, cost is a real constraint. An eval suite running 1,000 prompts through a judge evaluator at $0.001 per run costs $1. The same suite at $0.01 per run costs $10. Tracking cost per evaluation from the start - even in mock mode - establishes the instrumentation needed to make informed decisions about evaluation frequency and model selection.

### Why the mock-first approach

The judge evaluator ships with `use_mock=True` by default. The mock simulates realistic judge behavior without requiring API credits. This means the self-consistency logic, cost tracking, latency logging, and the before/after string-match comparison are all fully testable without any external dependency. The real API call is a one-line swap once credentials are configured. The three RAGAS-lite evaluators and the agent evaluator follow the identical pattern - mock-first, cost/latency tracked from the start, `_real_*` methods raising `NotImplementedError` until API credentials are wired in.

SemanticRelevanceEvaluator is a deliberate exception to this pattern, not an inconsistency in it: every other evaluator has a genuine mock (free, fake, simplified) vs. real (paid API call) split to toggle. This evaluator has no such split - it always runs real local computation, at zero cost either way. Adding a `use_mock` flag that doesn't actually toggle anything meaningful would be fake consistency rather than honest design, so it's omitted and explained in the class docstring instead.

The LangGraph stock agent is a further, deliberate departure from mock-first, for a different reason: the entire point of that rebuild was proving real end-to-end LLM tool-calling behavior against live data, not simulating it. A mocked version would answer a different, less interesting question than the one the rebuild set out to answer.

### Why RAG evaluation needed three separate evaluators, not one

The first instinct when adding RAG evaluation was to scope it narrowly - build one relevance check and stop. The better call was building all three RAGAS-lite metrics (faithfulness, answer relevance, context relevance) together, while the underlying pattern was still fresh, because: (a) context relevance is one of RAGAS's actual four core metrics, not an optional extension - skipping it means only half-understanding what RAGAS measures; (b) it's a genuinely different bug class from faithfulness or relevance - bad retrieval vs bad generation - invisible unless explicitly tested for; (c) it produces a coherent three-layer diagnostic narrative instead of three disconnected checks: is retrieval good -> did the model stay faithful to it -> does the final answer address the ask.

### Why the agent was rebuilt on LangGraph instead of extended in place

The original MockAgent used an implicit decision loop - keyword matching standing in for a real LLM decision, with no inspectable intermediate state. Rebuilding on LangGraph makes control flow explicit: defined nodes and edges forming an actual state machine, so a wrong intermediate step can be caught even when the final answer accidentally comes out right - not just whether the final output looked correct. The stock/index domain was chosen over extending the old weather/temperature/population domain specifically because yfinance ties naturally to real multi-tool sequencing (e.g. "compare Sensex today vs a month ago") and needs no API key, while still mapping directly onto the same four AgentEvaluator criteria with zero changes required there.

### Why the stock tools take yfinance-native period strings, not exact dates

`get_historical_price` takes period strings like `1wk`, `1mo` rather than exact dates, deliberately - the agent has to map natural language ("a week ago") to the correct period string itself, which is a real, catchable failure mode. Testing only against pre-sanitized exact dates would hide exactly the kind of mapping error a real agent needs to get right.

## The Provider Layer

Both providers extend `BaseProvider` and implement one method:

```python
def complete(self, request: EvalRequest) -> tuple[str, float]:
    # Returns (response_text, latency_seconds)
```

Anthropic response path: `message.content[0].text`
OpenAI response path: `completion.choices[0].message.content`

These paths are different. That difference is exactly why the abstraction exists - the engine never sees either path. It just calls `complete()`.

## Non-Determinism Handling

LLMs do not produce identical output on every run. The framework handles this with probabilistic assertions - not exact match assertions.

```python
# Fragile - will fail randomly even when the system is working correctly
assert response == "The capital of France is Paris."

# Stable - allows for natural variation while still catching real failures
passed_count = sum(1 for r in results if r.overall_passed)
assert passed_count >= 2  # At least 2 of 3 runs must pass
```

This is the core mental model shift from deterministic API testing to LLM evaluation. The JudgeEvaluator applies the same principle to the judge itself via self-consistency voting. The LangGraph stock agent's tests apply the same principle again, one level up: tool *choice* is tolerance-scored across acceptable alternatives, while numeric grounding stays a strict assertion, since not every kind of correctness tolerates the same amount of variance.

## Setup

### Prerequisites

- Python 3.11
- Anthropic API key with credits (for integration tests, live judge evaluation, and the LangGraph stock agent)

### Local setup

```powershell
git clone https://github.com/HSC-git-space/python-ai-testing-framework
cd python-ai-testing-framework
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```
ANTHROPIC_API_KEY=your-key-here
```

Run unit tests - no API key needed:

```powershell
python -m pytest tests/ -m "not integration" -v
```

Run all tests including integration and the live LangGraph stock agent - API key required:

```powershell
python -m pytest tests/ -v
```

Run with HTML report:

```powershell
python -m pytest tests/ -m "not integration" --html=reports/report.html --self-contained-html -v
```

### Docker

```powershell
docker build -t ai-eval-framework .
docker run --env-file .env ai-eval-framework
```

## CI/CD

GitHub Actions triggers on every push to main.

Steps:

1. Checkout code
2. Set up Python 3.11
3. Install dependencies
4. Run the full test suite, including the live LangGraph stock agent tests and the DeepEval wrapper - `ANTHROPIC_API_KEY` is configured as a GitHub Actions secret
5. Upload HTML report as artifact

Unlike the original unit-tests-vs-integration-tests split described in the Setup section above, the stock agent and DeepEval test files are **not** currently gated behind the `integration` marker - they run unfiltered on every push, hitting live Claude and live yfinance each time. This is a deliberate, accepted cost/latency tradeoff for a solo, actively-developed project (see the Cost note under DeepEval Integration above), not the pattern recommended at team scale, where this category of test is normally moved to a scheduled or manually-triggered job instead of gating every commit.

## Test Markers

| Marker | What it covers |
|---|---|
| keyword | Keyword evaluator tests |
| length | Length evaluator tests |
| tone | Tone evaluator tests |
| hallucination | Hallucination evaluator tests |
| judge | Judge evaluator tests |
| faithfulness | Faithfulness evaluator tests |
| relevance | Relevance evaluator tests |
| context_relevance | Context relevance evaluator tests |
| consistency | Consistency evaluator tests |
| non_determinism | Non-determinism tests |
| integration | Requires live API key - skipped in CI by default |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| ANTHROPIC_API_KEY | required | Anthropic API key |
| OPENAI_API_KEY | optional | OpenAI API key - present in CI secrets but not currently used by any active code path; DeepEval's judge uses AnthropicModel instead |
| DEFAULT_PROVIDER | anthropic | Primary provider |
| ANTHROPIC_MODEL | claude-sonnet-4-6 | Model to use |
| OPENAI_MODEL | gpt-4o | OpenAI model |
| MAX_TOKENS | 1024 | Max tokens per response |
| TEMPERATURE | 0.7 | Sampling temperature |
| REQUEST_TIMEOUT | 30 | API timeout seconds |
| EVAL_PASS_THRESHOLD | 0.7 | Minimum passing score |
| NON_DETERMINISM_RUNS | 5 | Runs for consistency checks |

## Known Limitations

- Hallucination evaluator uses string matching - cannot detect numeric contradictions or paraphrased errors (see JudgeEvaluator for the fix)
- Judge evaluator mock simulates but does not replicate real Claude judgment - set `use_mock=False` with a valid API key for production use
- Tone evaluator is heuristic - can misclassify mixed-tone responses
- RelevanceEvaluator and ContextRelevanceEvaluator use string-overlap content-word matching - possessives are stripped but plurals are not stemmed ("return" != "returns"); real stemming is out of scope for a mock evaluator
- SemanticRelevanceEvaluator closes the paraphrase blind spot above via embeddings, but shares the negation-blindness problem: contradictory sentences with shared vocabulary can still score moderately similar (~0.5), since cosine similarity responds to topical overlap as well as meaning - neither the word-overlap nor the embedding approach reliably detects negation
- ContextRelevanceEvaluator's `passage_relevance_threshold` (0.5) is a coarse per-passage cutoff tuned empirically rather than derived formulaically - very short prompts are more sensitive to single-word incidental matches
- The LangGraph stock agent's `decide_node` is a real, non-deterministic LLM call - tool choice can legitimately vary between runs for ambiguous queries (e.g. `get_index_level` vs `get_current_price` for "Sensex right now"). `test_stock_agent_eval.py` handles this with tolerance scoring on tool choice specifically, not by loosening numeric grounding checks
- The LangGraph rebuild is intentionally linear for this first version - no branching, conditional edges, retries, or loops. That is scoped as an honest first step, not a hidden shortcut
- The stock agent and DeepEval tests are not gated behind the `integration` marker and currently run on every CI push, incurring real API cost and latency on every commit - see the CI/CD section above
- Integration tests require paid API credits
- Non-determinism tests may occasionally fail at boundary thresholds due to natural LLM variation - this is expected behaviour, not a framework bug

## What Comes Next

- RAG evaluation - done. FaithfulnessEvaluator, RelevanceEvaluator, and ContextRelevanceEvaluator together form a RAGAS-lite core covering the retrieval and generation halves of a RAG pipeline independently.
- Agentic component - done, and rebuilt once. A LangChain-style MockAgent with tool-calling correctness evaluation shipped first; it was then rebuilt on LangGraph as an explicit state machine (`decide -> execute_tools -> final_answer`) making genuine, non-deterministic Claude tool-calling decisions against live stock/index market data, with the original AgentEvaluator reused unchanged across both.
- Semantic relevance - done. SemanticRelevanceEvaluator adds embedding-based cosine similarity as an upgrade path over RelevanceEvaluator's word-overlap approach, closing the no-shared-vocabulary paraphrase gap documented as a known limitation.
- DeepEval integration - done. AnswerRelevancy and Faithfulness metrics wrapped around the real LangGraph stock agent's output, using AnthropicModel as judge to keep a single LLM provider dependency.
- Branching/conditional LangGraph edges - not started. The current graph is linear by design; a natural next step would be a conditional edge that re-runs `decide_node` if `execute_tools_node` records an error-string tool output, rather than proceeding straight to a final answer with a known-bad trace entry.
- Batch evaluation runner - load full CSV dataset, run all evaluators, generate summary pass rate report
- Provider comparison - run same eval dataset against Claude and GPT-4, compare scores side by side