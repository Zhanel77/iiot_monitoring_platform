import logging
from app.config import Settings


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("mqtt-consumer")
    logger.setLevel(Settings.LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger