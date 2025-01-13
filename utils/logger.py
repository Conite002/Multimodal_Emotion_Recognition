import logging
import os
from datetime import datetime

def create_logger(logfile):
    logger = logging.getLogger(logfile)  # Create a logger with the logfile name
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(logfile)  # Write logs to the specified file
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger