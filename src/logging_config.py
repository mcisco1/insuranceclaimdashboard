"""Centralized logging configuration for the insurance claims pipeline.

Replaces ad-hoc print statements with structured logging. Provides
console output plus a rotating file handler that writes to ``logs/pipeline.log``.

Usage
-----
    from src.logging_config import setup_logging
    setup_logging()  # call once at startup
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import BASE_DIR

_LOG_DIR = BASE_DIR / "logs"
_LOG_FILE = _LOG_DIR / "pipeline.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> None:
    """Configure the root logger with console and optional file handlers.

    Parameters
    ----------
    level : int
        Logging level (default ``logging.INFO``).
    log_to_file : bool
        If *True*, also write to ``logs/pipeline.log`` via a rotating
        file handler (5 MB max, 3 backups).
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers on repeated calls
    if root.handlers:
        return

    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    if log_to_file:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Silence overly chatty third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
