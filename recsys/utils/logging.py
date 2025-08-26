import logging, sys

def setup_logger(name: str = "recsys", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
    return logger
