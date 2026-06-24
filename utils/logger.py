import logging
import json
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger that outputs structured JSON logs.
    Every LLM call, response, and eval result gets logged here.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_eval_summary(logger: logging.Logger, test_id: str, passed: bool, score: float, latency: float):
    """
    Logs a structured summary of one eval run.
    Machine readable format — could feed into a monitoring system.
    """
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "test_id": test_id,
        "passed": passed,
        "overall_score": round(score, 3),
        "latency_seconds": round(latency, 3)
    }
    logger.info(json.dumps(summary))