from loguru import logger
import sys, json, time, uuid, os
from contextlib import contextmanager
from datetime import timezone

def compact_json_sink(msg):
    r = msg.record
    out = {
        "time": r["time"].astimezone(timezone.utc).isoformat(),
        "level": r["level"].name,
        "module": r["module"],
        "message": r["message"],
    }

    for k, v in r["extra"].items():
            if k not in out:
                out[k] = v

    if r["exception"]:
        out["exception"] = {
            "type": r["exception"].type.__name__,
            "message": str(r["exception"].value),
        }
    print(json.dumps(out), file=sys.stdout)

def setup_logging(app_name):
    logger.remove()  # remove default stderr sink
    logger.add(compact_json_sink, serialize=True, level=os.getenv("LOG_LEVEL", "INFO"), backtrace=False, diagnose=False)
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