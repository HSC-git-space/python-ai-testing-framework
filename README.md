# python-ai-testing-framework

![CI](https://github.com/HSC-git-space/python-ai-testing-framework/actions/workflows/ci.yml/badge.svg)

An AI evaluation framework for testing LLM responses built with Python, pytest, and the Anthropic API.

Every design decision has a reason. Every file can be explained and defended.

---

## Background

Standard API testing is deterministic — send input X, expect output Y, assert equality. LLMs break that model. The same prompt can return different responses on every run. You cannot assert exact equality. You need a different approach — score-based evaluation, threshold-based pass/fail, and probabilistic assertions across multiple runs.

This framework is built around that mental model.

---

## What Is Built

- A provider abstraction layer so evaluation logic works with any LLM — currently Anthropic Claude and OpenAI
- Six evaluation modules — keyword presence, response length, tone classification, output consistency, rule-based hallucination detection, and LLM-as-judge semantic evaluation
- An evaluation engine that wires providers and evaluators together into one structured result per run
- Dataset-driven evaluation using CSV and Excel
- Structured logging on every LLM call — prompt sent, response received, latency, eval scores
- Docker for running the framework in a clean isolated environment
- GitHub Actions CI — unit tests run automatically on every push, integration tests separated out

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pytest | Test runner |
| Anthropic SDK | Primary LLM provider |
| OpenAI SDK | Secondary LLM provider |
| Pydantic | Input and output model validation |
| Pandas | Dataset loading — CSV and Excel |
| python-dotenv | Environment variable configuration |
| pytest-html | HTML test report |
| Docker | Isolated test execution environment |
| GitHub Actions | CI pipeline |

---

## Project Structure

```
python-ai-testing-framework/
│
├── config/
│   └── config.py               # All config in one place — API keys, model name, thresholds
│
├── models/
│   ├── eval_request.py         # Pydantic — defines what goes into an eval run
│   └── eval_result.py          # Pydantic — defines what comes out of an eval run
│
├── providers/
│   ├── base_provider.py        # Abstract base class — the contract every provider must follow
│   ├── anthropic_provider.py   # Claude implementation
│   └── openai_provider.py      # OpenAI implementation
│
├── evaluators/
│   ├── base_evaluator.py       # Abstract base class — the contract every evaluator must follow
│   ├── keyword_evaluator.py    # Required and forbidden keyword checks
│   ├── length_evaluator.py     # Response length validation
│   ├── tone_evaluator.py       # Formal vs informal tone classification
│   ├── consistency_evaluator.py # Output consistency across N repeated runs
│   ├── hallucination_evaluator.py # Rule-based fact checking against known ground truth
│   └── judge_evaluator.py      # LLM-as-judge semantic evaluation with self-consistency voting
│
├── engine/
│   └── eval_engine.py          # Wires providers and evaluators together per eval run
│
├── utils/
│   ├── logger.py               # Structured logging setup
│   ├── timer.py                # Latency tracking utility
│   └── data_loader.py          # CSV and Excel loader
│
├── datasets/
│   ├── prompts.csv             # Eval prompts with expected behaviour and constraints
│   └── ground_truth.xlsx       # Known facts used by the hallucination evaluator
│
├── tests/
│   ├── test_keyword_eval.py    # Unit tests for keyword evaluator
│   ├── test_length_eval.py     # Unit tests for length evaluator
│   ├── test_tone_eval.py       # Unit tests for tone evaluator
│   ├── test_hallucination.py   # Unit tests for hallucination evaluator
│   ├── test_judge_eval.py      # Unit tests for judge evaluator — includes before/after comparison
│   ├── test_consistency_eval.py # Integration tests — needs live API key
│   └── test_non_determinism.py  # Integration tests — needs live API key
│
├── conftest.py                 # Session-scoped fixtures and logging
├── pytest.ini                  # Test paths, markers, report config
├── Dockerfile                  # Container setup for running the framework
├── .env.example                # Shows required environment variables — no real keys committed
└── requirements.txt            # All dependencies
```

---

## How the Evaluation Flow Works

```
EvalRequest (Pydantic)
        │
        ▼
EvalEngine.run()
        │
        ├── providers[request.provider].complete(request)
        │       └── Returns (response_text, latency_seconds)
        │
        └── for each evaluator in request.evaluators:
                evaluator.evaluate(response_text, request)
                        └── Returns EvaluatorScore(evaluator_name, score, passed, reason)
                │
                ▼
        EvalResult (Pydantic)
                └── prompt, response, provider, latency, evaluator_scores, overall_passed, overall_score
```

The engine does not know which provider it is talking to. It does not know which evaluator it is running. It only talks to the `BaseProvider` and `BaseEvaluator` abstract interfaces. Swap Claude for GPT-4 by changing one config value — no eval logic changes.

---

## The Evaluators

### KeywordEvaluator
Checks required keywords are present in the response and forbidden keywords are absent. Case insensitive. Score is the proportion of keyword conditions satisfied.

### LengthEvaluator
Validates response character count against `min_length` and `max_length` from the request. Returns a perfect score if no bounds are configured — the evaluator only fires when constraints exist.

### ToneEvaluator
Rule-based classification. Counts formal indicator words — *therefore, furthermore, consequently* — against informal indicator words — *hey, gonna, yeah*. Whichever list has more matches determines the detected tone. Known limitation: mixed-tone responses can produce incorrect classifications.

### ConsistencyEvaluator
Runs the same prompt N times. Applies a classifier function to each response. Score is the proportion of runs that matched the expected output class. This tests whether the LLM produces stable output across repeated runs — not whether any single response is correct.

### HallucinationEvaluator
Checks the response against known facts in `EvalRequest.ground_truth`. Strips stop words, extracts content words, checks whether enough content words from each fact appear in the response.

**Known limitation — important:** This evaluator cannot detect numeric contradictions or paraphrased errors. If ground truth says "Water boils at 100 degrees Celsius" and the response says "Water boils at 90 degrees" — both contain the words *water, boils, degrees* — so the overlap check passes even though the value is wrong. This is the exact failure mode the JudgeEvaluator was built to fix.

### JudgeEvaluator
Uses Claude to semantically evaluate whether a response correctly addresses the prompt and is consistent with known ground truth facts — replacing brittle string matching with genuine semantic understanding.

**The concrete before/after story:**

```
Ground truth: "Water boils at 100 degrees Celsius at standard pressure"
Response:     "Water boils at 90 degrees Celsius."

HallucinationEvaluator result: PASS
Reason: response contains "water", "boils", "degrees" — string overlap sufficient

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
passed = True  # above 0.7 threshold? No — FAIL in this case
```

**Cost and latency tracking:**

Every evaluation run logs cost (estimated from token counts) and latency. This matters in production — a judge call costs approximately 10-50x more than a string match call. The tradeoff table below documents when each approach is appropriate.

---

## Design Decisions and Tradeoffs

### When to use string matching vs LLM-as-judge

| Factor | String Match (HallucinationEvaluator) | LLM-as-Judge (JudgeEvaluator) |
|---|---|---|
| Cost per evaluation | Near zero | $0.0001–$0.001 per run |
| Latency | < 1ms | 1–5 seconds |
| Catches numeric errors | No | Yes |
| Catches paraphrased errors | No | Yes |
| Catches keyword presence | Yes | Yes |
| Non-determinism | None | Requires self-consistency |
| Scales to 10,000 evals | Yes | Expensive |
| Best for | Keyword gates, format checks, fast CI | Factual accuracy, semantic correctness |

**Decision rule:** Use string matching as a fast first gate. Use the judge only when semantic accuracy matters and the cost is justified. In CI, string matching runs on every push. Judge evaluation runs on scheduled or pre-release checks.

### Why self-consistency voting

A single judge call can return different verdicts on the same input across runs. Running the judge 3 times and taking the majority vote reduces variance without a significant cost increase. Three runs at claude-haiku pricing costs approximately $0.0003-0.001 per evaluation — acceptable for pre-release quality gates.

### Why cost tracking

In production evaluation pipelines, cost is a real constraint. An eval suite running 1,000 prompts through a judge evaluator at $0.001 per run costs $1. The same suite at $0.01 per run costs $10. Tracking cost per evaluation from the start — even in mock mode — establishes the instrumentation needed to make informed decisions about evaluation frequency and model selection.

### Why the mock-first approach

The judge evaluator ships with `use_mock=True` by default. The mock simulates realistic judge behavior without requiring API credits. This means the self-consistency logic, cost tracking, latency logging, and the before/after string-match comparison are all fully testable without any external dependency. The real API call is a one-line swap once credentials are configured.

---

## The Provider Layer

Both providers extend `BaseProvider` and implement one method:

```python
def complete(self, request: EvalRequest) -> tuple[str, float]:
    # Returns (response_text, latency_seconds)
```

Anthropic response path: `message.content[0].text`
OpenAI response path: `completion.choices[0].message.content`

These paths are different. That difference is exactly why the abstraction exists — the engine never sees either path. It just calls `complete()`.

---

## Non-Determinism Handling

LLMs do not produce identical output on every run. The framework handles this with probabilistic assertions — not exact match assertions.

```python
# Fragile — will fail randomly even when the system is working correctly
assert response == "The capital of France is Paris."

# Stable — allows for natural variation while still catching real failures
passed_count = sum(1 for r in results if r.overall_passed)
assert passed_count >= 2  # At least 2 of 3 runs must pass
```

This is the core mental model shift from deterministic API testing to LLM evaluation. The JudgeEvaluator applies the same principle to the judge itself via self-consistency voting.

---

## Setup

### Prerequisites
- Python 3.11
- Anthropic API key with credits (for integration tests and live judge evaluation)

### Local setup
```bash
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

### Run unit tests — no API key needed
```bash
python -m pytest tests/ -m "not integration" -v
```

### Run all tests including integration — API key required
```bash
python -m pytest tests/ -v
```

### Run with HTML report
```bash
python -m pytest tests/ -m "not integration" --html=reports/report.html --self-contained-html -v
```

---

## Docker

```bash
docker build -t ai-eval-framework .
docker run --env-file .env ai-eval-framework
```

---

## CI/CD

GitHub Actions triggers on every push to main.

Steps:
1. Checkout code
2. Set up Python 3.11
3. Install dependencies
4. Run unit tests — integration tests excluded via `-m "not integration"`
5. Upload HTML report as artifact

Unit tests run without any API key. Integration tests need `ANTHROPIC_API_KEY` configured as a GitHub secret.

---

## Test Markers

| Marker | What it covers |
|---|---|
| `keyword` | Keyword evaluator tests |
| `length` | Length evaluator tests |
| `tone` | Tone evaluator tests |
| `hallucination` | Hallucination evaluator tests |
| `judge` | Judge evaluator tests |
| `consistency` | Consistency evaluator tests |
| `non_determinism` | Non-determinism tests |
| `integration` | Requires live API key — skipped in CI by default |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `OPENAI_API_KEY` | optional | OpenAI API key |
| `DEFAULT_PROVIDER` | anthropic | Primary provider |
| `ANTHROPIC_MODEL` | claude-sonnet-4-6 | Model to use |
| `OPENAI_MODEL` | gpt-4o | OpenAI model |
| `MAX_TOKENS` | 1024 | Max tokens per response |
| `TEMPERATURE` | 0.7 | Sampling temperature |
| `REQUEST_TIMEOUT` | 30 | API timeout seconds |
| `EVAL_PASS_THRESHOLD` | 0.7 | Minimum passing score |
| `NON_DETERMINISM_RUNS` | 5 | Runs for consistency checks |

---

## Known Limitations

- Hallucination evaluator uses string matching — cannot detect numeric contradictions or paraphrased errors (see JudgeEvaluator for the fix)
- Judge evaluator mock simulates but does not replicate real Claude judgment — set `use_mock=False` with a valid API key for production use
- Tone evaluator is heuristic — can misclassify mixed-tone responses
- Integration tests require paid API credits
- Non-determinism tests may occasionally fail at boundary thresholds due to natural LLM variation — this is expected behaviour, not a framework bug

---

## What Comes Next

- **Agentic component** — build a small LangChain agent as a test subject within this repo, evaluate tool-calling correctness (right tool, right order, correct arguments)
- **RAG evaluation hooks** — add faithfulness and relevance evaluator modules following the existing pattern
- **Batch evaluation runner** — load full CSV dataset, run all evaluators, generate summary pass rate report
- **Provider comparison** — run same eval dataset against Claude and GPT-4, compare scores side by side