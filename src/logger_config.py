# /home/tkh/repos/hugging_face/youtube_comment_analyzer/src/logger_config.py
import sys
from loguru import logger
from src.config import PROJECT_ROOT

def setup_logger():
    """
    Configures the Loguru logger for the application.

    This setup removes the default handler and adds two new ones:
    1. A handler for colorful, formatted output to the console (stderr).
    2. A handler for writing all log messages (INFO and above) to a
       file named `app_log.txt` in the project root.
    """
    logger.remove()  # Remove the default handler
    
    # Console logger
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
    
    # File logger
    log_file_path = PROJECT_ROOT / "app_log.txt"
    logger.add(
        log_file_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",  # Rotates the log file when it reaches 10 MB
        retention="7 days", # Keeps log files for 7 days
        enqueue=True,      # Makes logging thread-safe
        backtrace=True,    # Shows full stack trace on exceptions
        diagnose=True,     # Adds exception variable values
    )

    return logger

# Instantiate the logger so it can be imported and used across the application
app_logger = setup_logger()