import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from EmoBIRDv2.utils.constants import EMOBIRD_ROOT

LOG_DIR = os.path.join(EMOBIRD_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "emobirdv2.log")


def get_logger(
    name: str = "EmoBIRDv2",
    level: int = logging.INFO,
    per_run: bool = False,
    file_prefix: str | None = None,
) -> logging.Logger:
    """
    Returns a configured logger with:
    - Console handler (stderr)
    - Rotating file handler at logs/emobirdv2.log (1 MB, 3 backups)

    Safe to call multiple times; handlers are added only once per logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Ensure logs directory exists
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        # If directory can't be created, continue with console-only logging
        pass

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (if path available)
    try:
        if per_run:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = (file_prefix or name).lower().replace(" ", "_")
            path = os.path.join(LOG_DIR, f"{base}_{ts}.log")
            fh = logging.FileHandler(path)
        else:
            fh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # If file handler fails, continue with console-only
        pass

    return logger
