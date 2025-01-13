import logging
import os
from datetime import datetime

def setup_logger(log_folder, dim, name):
    """Set up a logger with a unique file name."""
    os.makedirs(log_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = os.path.join(log_folder, f"{name}_embedding_dim_{dim}_{timestamp}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filemode="w"
    )
    logger = logging.getLogger()
    logger.info(f"Logger initialized for dimension {dim}. Log file: {log_file}")
    return logger, log_file
