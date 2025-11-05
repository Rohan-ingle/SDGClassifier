"""
Centralized logging configuration for SDG Classifier

This module provides a unified logging setup that creates:
- Console output with colored formatting
- Detailed file logs with rotation
- Separate error logs
- JSON-formatted logs for machine parsing
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        if sys.stdout.isatty():  # Only colorize if output is to terminal
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    module_name: Optional[str] = None,
    enable_file_logging: bool = True,
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        module_name: Name of the module (used for logger name and log filename)
        enable_file_logging: Whether to enable file logging

    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get logger
    logger_name = module_name if module_name else "sdg_classifier"
    logger = logging.getLogger(logger_name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    if enable_file_logging:
        # Detailed file handler with rotation
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f"{logger_name}_{timestamp}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / log_filename,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Error file handler (only errors and critical)
        error_log_filename = f"{logger_name}_errors_{timestamp}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / error_log_filename,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        logger.addHandler(error_handler)

        # JSON file handler for structured logging
        json_log_filename = f"{logger_name}_{timestamp}.jsonl"
        json_handler = logging.handlers.RotatingFileHandler(
            log_path / json_log_filename,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

    return logger


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for easy parsing and analysis"""

    def format(self, record):
        import json
        
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add any extra fields from the log record
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def log_function_call(logger):
    """
    Decorator to automatically log function calls with parameters and results.
    
    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            return result
    """
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log function entry
            logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__} successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def get_logger(module_name: str, log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Convenience function to get a configured logger.
    
    Args:
        module_name: Name of the module requesting the logger
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    return setup_logging(log_dir=log_dir, log_level=log_level, module_name=module_name)
