import logging
import os
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """
    Setup a logger that writes to a file and console.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        
    return logger

def get_session_logger(log_dir="outputs/logs/"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a unique log file for the session
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"neuroai_session_{timestamp}.log")
    
    return setup_logger("NeuroAI", log_file)
