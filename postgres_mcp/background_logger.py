"""Background process logging utility with unbuffered output."""
import os
import sys
import logging
from pathlib import Path

def setup_background_logging(job_id: str, log_dir: str = "/tmp/4dllm_logs") -> logging.Logger:
    """
    Set up logging for background processes with unbuffered output.
    
    Args:
        job_id: Unique identifier for this job
        log_dir: Directory to store log files
        
    Returns:
        Logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    log_file = log_path / f"analysis_{job_id}.log"
    
    # Set up logger
    logger = logging.getLogger(f"4dllm_background_{job_id}")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler with no buffering
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler (also unbuffered)
    console_handler = logging.StreamHandler(sys.stdout)
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
    
    # Force unbuffered output
    sys.stdout.flush()
    sys.stderr.flush()
    
    return logger, log_file