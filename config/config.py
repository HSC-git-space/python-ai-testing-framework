import os
from dotenv import load_dotenv

# Load .env file into shell environment at startup
# Without this, os.getenv() returns None locally
load_dotenv()

# --- LLM Provider Config ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Which provider to use by default — can be overridden per test
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "anthropic")

# --- Model Config ---
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# --- Request Config ---
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# --- Eval Config ---
# Minimum score for an eval to be considered passing
EVAL_PASS_THRESHOLD = float(os.getenv("EVAL_PASS_THRESHOLD", "0.7"))

# How many times to run a prompt for non-determinism tests
NON_DETERMINISM_RUNS = int(os.getenv("NON_DETERMINISM_RUNS", "5"))