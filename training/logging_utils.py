import logging
import os
import sys
from contextlib import contextmanager
from typing import Optional


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def setup_logging(log_path: str, level: int = logging.INFO):
    """Configure logging to both file and stdout."""
    ensure_dir(os.path.dirname(log_path))
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    logging.getLogger().info("Logging to %s", log_path)


@contextmanager
def tee_stdout_stderr(log_path: str):
    """Context manager that tees stdout/stderr to a log file."""
    ensure_dir(os.path.dirname(log_path))
    log_file = open(log_path, "a")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    class Tee(object):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    try:
        sys.stdout = Tee(orig_stdout, log_file)
        sys.stderr = Tee(orig_stderr, log_file)
        yield
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()

