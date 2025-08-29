import os
from loguru import logger
import sys
import time
from contextlib import contextmanager
import uuid

def setup_logging(app_name):
    logger.remove()  # remove default stderr sink
    logger.add(sys.stdout, serialize=True, level=os.getenv("LOG_LEVEL", "INFO"), backtrace=True, diagnose=False)
    return logger.bind(run_id=uuid.uuid4().hex[:12], app=app_name)

# small helper for timing
@contextmanager
def span(name: str, **start_fields):
    _t0 = time.perf_counter()
    logger.bind(**start_fields).info(f"{name}.start")
    try:
        yield
        logger.info(f"{name}.end", latency_ms=int((time.perf_counter() - _t0) * 1000))
    except Exception as e:
        logger.exception(f"{name}.error")
        raise