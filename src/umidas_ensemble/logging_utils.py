from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "umidas_ensemble", level: int = logging.INFO) -> logging.Logger:
    """Get a package logger with a simple console handler.

    The function is idempotent: handlers are only added once.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    logger.propagate = False
    return logger
