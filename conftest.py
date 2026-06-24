import pytest
from engine.eval_engine import EvalEngine
from utils.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def eval_engine():
    """
    Session-scoped EvalEngine fixture.
    Created once for the entire test suite — not recreated per test.
    Equivalent to @BeforeSuite in TestNG.
    """
    logger.info("Initialising EvalEngine for test session")
    engine = EvalEngine()
    return engine


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    """
    Autouse fixture — fires automatically for every test session.
    Logs session start and end without being declared in any test.
    """
    logger.info("Test session started")
    yield
    logger.info("Test session completed")