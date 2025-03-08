import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    """Set up and configure logger"""
    logger = logging.getLogger('timesheet_analyzer')
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Add file handler
    file_handler = RotatingFileHandler(
        'logs/app.log', 
        maxBytes=1024*1024*5,  # 5 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger