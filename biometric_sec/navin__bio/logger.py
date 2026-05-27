"""
logger.py
─────────
Centralised logging setup.  Import `logger` from this module in every other file.
"""

import logging
import sys
from config import LOG_FILE, LOG_LEVEL, LOG_TO_FILE, LOG_TO_CONSOLE

_FMT  = "%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger wired to the project handlers."""
    log = logging.getLogger(name)

    if log.handlers:          # avoid duplicate handlers on re-import
        return log

    log.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    formatter = logging.Formatter(_FMT, datefmt=_DATE)

    if LOG_TO_CONSOLE:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        log.addHandler(ch)

    if LOG_TO_FILE:
        try:
            fh = logging.FileHandler(LOG_FILE)
            fh.setFormatter(formatter)
            log.addHandler(fh)
        except OSError:
            pass   # log directory might not exist in test environments

    return log
