import logging
import os
from datetime import datetime

def create_logger(logfile):
    """
    Create a logger instance that cleans the logfile before writing.
    """
    

    logger = logging.getLogger(logfile)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
