# /home/tkh/repos/hugging_face/youtube_comment_analyzer/src/logger_config.py
import sys
from loguru import logger

def setup_logger():
    """
    Configures the Loguru logger for the application.

    This setup removes the default handler and adds a new one with a
    more informative format. It directs logs to stderr, which is a
    standard practice for application logging.
    """
    logger.remove()  # Remove the default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    return logger

# Instantiate the logger so it can be imported and used across the application
app_logger = setup_logger()