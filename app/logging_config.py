import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lstm_trainer")
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler("logs/training.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
