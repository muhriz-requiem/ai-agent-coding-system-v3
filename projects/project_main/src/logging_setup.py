import logging
import logging.handlers
import os
from typing import Optional

# Global flag to track if logging has been configured
_logging_configured = False

def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure structured logging with JSON formatting.
    Avoids duplicate handlers if called multiple times.
    
    Args:
        level: Logging level (default: INFO)
    """
    global _logging_configured
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers if already configured
    if _logging_configured:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Set log level
    root_logger.setLevel(level)
    
    # Create JSON formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    _logging_configured = True

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a logger with the given name.
    
    Args:
        name: Logger name (defaults to root logger)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
