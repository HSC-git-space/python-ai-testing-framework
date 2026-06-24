import time
from contextlib import contextmanager


def get_latency(start_time: float) -> float:
    """
    Returns elapsed time in seconds since start_time.
    start_time should be captured using time.perf_counter()
    """
    return round(time.perf_counter() - start_time, 3)


@contextmanager
def timer():
    """
    Context manager for timing a block of code.

    Usage:
        with timer() as t:
            # code to time
        latency = t()
    """
    start = time.perf_counter()
    yield lambda: round(time.perf_counter() - start, 3)