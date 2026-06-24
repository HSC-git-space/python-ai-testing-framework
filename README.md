![CI](https://github.com/HSC-git-space/python-ai-testing-framework/actions/workflows/ci.yml/badge.svg)

# python-ai-testing-framework

An AI evaluation framework for testing LLM responses built with Python, pytest, and the Anthropic API.

Built as part of a deliberate transition from Java SDET work into AI Quality Engineering. Every design decision has a reason. Every file can be explained and defended.

---

## Background

Standard API testing is deterministic — send input X, expect output Y, assert equality. LLMs break that model. Same prompt can return different responses on every run. You cannot assert exact equality. You need a different approach — score-based evaluation, threshold-based pass/fail, and probabilistic assertions across multiple runs.

This framework is built around that mental model.

---

## What Is Built

- A provider abstraction layer so evaluation logic works with any LLM — currently Anthropic Claude and OpenAI
- Five evaluation modules — keyword presence, response length, tone classification, output consistency, and rule-based hallucination detection
- An evaluation engine that wires providers and evaluators together into one structured result per run
- Dataset-driven evaluation using CSV and Excel — same pattern as my data-driven testing repo
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

│   └── hallucination_evaluator.py # Fact checking against known ground truth

│

├── engine/

│   └── eval_engine.py          # Wires providers and evaluators together per eval run

│

├── utils/

│   ├── logger.py               # Structured logging setup

│   ├── timer.py                # Latency tracking utility

│   └── data_loader.py          # CSV and Excel loader — same pattern as Repo 2

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

│   ├── test_consistency_eval.py # Integration tests — needs live API key

│   └── test_non_determinism.py # Integration tests — needs live API key

│

├── conftest.py                 # Session-scoped fixtures and logging

├── pytest.ini                  # Test paths, markers, report config

├── Dockerfile                  # Container setup for running the framework

├── .env.example                # Shows required environment variables — no real keys committed

└── requirements.txt            # All dependencies

---

## How the Evaluation Flow Works
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

└── Returns EvaluatorScore(score, passed, reason)

│

▼

EvalResult (Pydantic)

└── prompt, response, provider, latency, evaluator_scores, overall_passed, overall_score

The engine does not know which provider it is talking to. It does not know which evaluator it is running. It only talks to the `BaseProvider` and `BaseEvaluator` abstract interfaces. Swap Claude for GPT-4 by changing one config value — no eval logic changes.

In Java terms — this is programming to an interface, not an implementation. `BaseProvider` is the interface. `AnthropicProvider` and `OpenAIProvider` are the concrete implementations.

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

## The Evaluators

### KeywordEvaluator
Checks required keywords are present in the response and forbidden keywords are absent. Case insensitive. Score is the proportion of keyword conditions satisfied.

### LengthEvaluator
Validates response character count against `min_length` and `max_length` from the request. Returns a perfect score if no bounds are configured — the evaluator only fires when constraints exist.

### ToneEvaluator
Rule-based classification. Counts formal indicator words — therefore, furthermore, consequently — against informal indicator words — hey, gonna, yeah. Whichever list has more matches determines the detected tone. Known limitation: mixed-tone responses can produce incorrect classifications.

### ConsistencyEvaluator
Runs the same prompt N times. Applies a classifier function to each response. Score is the proportion of runs that matched the expected output class. This tests whether the LLM produces stable output across repeated runs — not whether any single response is correct.

### HallucinationEvaluator
Checks the response against known facts in `EvalRequest.ground_truth`. Strips stop words, extracts content words, checks whether enough content words from each fact appear in the response.

**Known limitation — important:** This evaluator cannot detect numeric contradictions or paraphrased errors. If ground truth says "Water boils at 100 degrees Celsius" and the response says "Water boils at 90 degrees" — both contain the words water, boils, degrees — so the overlap check passes even though the value is wrong. Catching this class of error requires semantic understanding — either an NLI model like `bart-large-mnli` or an LLM-as-judge approach where Claude itself evaluates whether the response contradicts the fact. That is the next evolution beyond rule-based detection. This limitation is documented intentionally because understanding where your own system breaks down is part of building reliable evaluation infrastructure.

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

This is the core mental model shift from deterministic API testing to LLM evaluation.

---

## Setup

### Prerequisites
- Python 3.11
- Anthropic API key with credits (for integration tests)

### Local setup

```bash
git clone https://github.com/HSC-git-space/python-ai-testing-framework
cd python-ai-testing-framework
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:
ANTHROPIC_API_KEY=your-key-here

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

Build and run the framework in a container:

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
| `consistency` | Consistency evaluator tests |
| `non_determinism` | Non-determinism tests |
| `integration` | Requires live API key — skipped in CI by default |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| ANTHROPIC_API_KEY | required | Anthropic API key |
| OPENAI_API_KEY | optional | OpenAI API key |
| DEFAULT_PROVIDER | anthropic | Primary provider |
| ANTHROPIC_MODEL | claude-sonnet-4-6 | Model to use |
| OPENAI_MODEL | gpt-4o | OpenAI model |
| MAX_TOKENS | 1024 | Max tokens per response |
| TEMPERATURE | 0.7 | Sampling temperature |
| REQUEST_TIMEOUT | 30 | API timeout seconds |
| EVAL_PASS_THRESHOLD | 0.7 | Minimum passing score |
| NON_DETERMINISM_RUNS | 5 | Runs for consistency checks |

---

## Known Limitations

- Hallucination evaluator uses string matching — cannot detect numeric contradictions or paraphrased errors
- Tone evaluator is heuristic — can misclassify mixed-tone responses
- Integration tests require paid API credits
- Non-determinism tests may occasionally fail at boundary thresholds due to natural LLM variation — this is expected behaviour, not a framework bug

---

## What Comes Next

- LLM as judge evaluator — use Claude to semantically evaluate responses against ground truth facts rather than string matching
- Batch evaluation runner — load full CSV dataset, run all evaluators, generate summary pass rate report
- RAG evaluation support — evaluate retrieval augmented generation pipelines
- Provider comparison — run same eval dataset against Claude and GPT-4, compare scores side by side