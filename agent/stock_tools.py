"""
stock_tools.py

Deterministic tool functions for the stock/index domain, using yfinance
for real market data. No mocking here - yfinance itself is free and
requires no API key, so these hit real data every time, similar in
spirit to how semantic_relevance_evaluator always runs real local
inference. Framework-agnostic: returns plain Python types only, no
LangChain or LangGraph objects, so agent_evaluator.py needs zero changes.
"""

import yfinance as yf

INDEX_TICKERS = {
    "sensex": "^BSESN",
    "nifty": "^NSEI",
    "nifty50": "^NSEI",
    "bse": "^BSESN",
    "nse": "^NSEI",
}

VALID_PERIODS = ["1d", "5d", "1wk", "1mo", "3mo", "6mo", "1y"]


def get_current_price(ticker: str) -> float:
    """
    Returns the latest available closing price for the given ticker.
    ticker should be a yfinance-recognized symbol, e.g. "RELIANCE.NS",
    "^BSESN", "^NSEI", or a plain US ticker like "AAPL".
    Returns -1.0 if no data is available for the ticker.
    """
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if data.empty:
            return -1.0
        return round(float(data["Close"].iloc[-1]), 2)
    except Exception:
        return -1.0


def get_historical_price(ticker: str, period: str) -> float:
    """
    Returns the closing price from `period` ago for the given ticker.
    period must be one of VALID_PERIODS (yfinance-native period strings),
    e.g. "1wk" for a week ago, "1mo" for a month ago.
    Returns -1.0 if the period is invalid or no data is available.
    """
    if period not in VALID_PERIODS:
        return -1.0
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return -1.0
        return round(float(data["Close"].iloc[0]), 2)
    except Exception:
        return -1.0


def get_index_level(index_name: str) -> float:
    """
    Returns the current level of a named index (Sensex, Nifty).
    index_name is matched case-insensitively against INDEX_TICKERS.
    Returns -1.0 if the index name is not recognized or data is
    unavailable.
    """
    key = index_name.strip().lower()
    ticker = INDEX_TICKERS.get(key)
    if ticker is None:
        return -1.0
    return get_current_price(ticker)