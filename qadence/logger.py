from __future__ import annotations

import logging
import os
import sys

logging_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

LOG_STREAM_HANDLER = sys.stdout

DEFAULT_LOGGING_LEVEL = logging.INFO

# FIXME: introduce a better handling of the configuration
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "warning").upper()


def get_logger(name: str) -> logging.Logger:
    logger: logging.Logger = logging.getLogger(name)

    level = logging_levels.get(LOGGING_LEVEL, DEFAULT_LOGGING_LEVEL)
    logger.setLevel(level)

    formatter = logging.Formatter("%(levelname) -5s %(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    # formatter = logging.Formatter(LOG_FORMAT)
    sh = logging.StreamHandler(LOG_STREAM_HANDLER)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
