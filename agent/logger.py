"""
Structured run logger - writes to logs/runs/.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def setup_logger(name: str = "agent", log_dir: str = "logs/runs") -> logging.Logger:
    """
    Set up a logger that writes structured run logs.
    
    Args:
        name: Logger name
        log_dir: Directory to store run logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # File handler
    log_file = log_path / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "agent") -> logging.Logger:
    """Get an existing logger instance."""
    return logging.getLogger(name)


def log_run(input_data: Dict[str, Any], output: Dict[str, Any], 
            latency: float, success: bool, log_dir: str = "logs/runs"):
    """
    Log a run's input, output, and metadata.
    
    Args:
        input_data: The input data
        output: The agent's output
        latency: Run latency in seconds
        success: Whether the run was successful
        log_dir: Directory to store run logs
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    run_record = {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "output": output,
        "latency_seconds": latency,
        "success": success
    }
    
    log_file = log_path / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(run_record, f, indent=2)
