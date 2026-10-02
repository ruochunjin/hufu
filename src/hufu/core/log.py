"""Task-scoped logging helpers that coexist with tqdm progress displays."""

import logging
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # Minimal installations do not require tqdm.
    tqdm = None

current_task: ContextVar[str] = ContextVar("task", default="hotpotqa")
current_query_id: ContextVar[str] = ContextVar("query_id", default="")
class TqdmLoggingHandler(logging.StreamHandler):
    """Stream handler that writes without corrupting an active tqdm display."""

    def __init__(self, level=logging.NOTSET):
        """Initialize a stderr handler at the requested logging level."""

        super().__init__(sys.stderr)
        self.setLevel(level)
    def emit(self, record):
        """Format and emit one record through ``tqdm.write``."""

        try:
            msg = self.format(record)
            if tqdm is None:
                self.stream.write(msg + self.terminator)
            else:
                tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)
def getLogger(name: str, level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Create a tqdm-safe logger scoped to the current task and query ID."""

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    qid = current_query_id.get()
    task = current_task.get()
    formatter = logging.Formatter(
        fmt='%(message)s',
    )
    console = TqdmLoggingHandler()
    console.setLevel(logging.ERROR)
    console.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if logger.handlers:
        # Reconfiguration replaces prior handlers instead of duplicating output.
        logger.handlers.clear()
    logger.addHandler(console)
    # A source checkout may be mounted read-only.  File logging is useful
    # during experiments but must never make importing a module fail.
    log_path = Path(log_dir) / task
    if qid:
        log_path = log_path / qid
    try:
        log_path.mkdir(exist_ok=True, parents=True)
        file_handler = RotatingFileHandler(
            filename=log_path / f"{name}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
    except OSError:
        file_handler = None
    if file_handler is not None:
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
