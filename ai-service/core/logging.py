"""
Logging configuration for TTBall_4 AI Service

Provides structured logging with JSON format, performance monitoring,
and proper log management for the AI service components.
"""

import logging
import logging.config
import json
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
from contextlib import contextmanager

from .config import get_settings


class TTBallFormatter(logging.Formatter):
    """
    Custom JSON formatter for TTBall_4 AI Service logs
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add performance data if present
        if hasattr(record, 'duration'):
            log_data["performance"] = {
                "duration_ms": record.duration,
                "operation": getattr(record, 'operation', 'unknown')
            }
        
        # Add request context if present
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        
        return json.dumps(log_data)


class PerformanceLogger:
    """
    Logger for performance monitoring
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @contextmanager
    def measure(self, operation: str, **kwargs):
        """
        Context manager for measuring operation performance
        
        Args:
            operation: Name of the operation being measured
            **kwargs: Additional context data
        """
        start_time = time.time()
        
        try:
            yield
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.logger.error(
                f"Operation '{operation}' failed after {duration:.2f}ms",
                extra={
                    'duration': duration,
                    'operation': operation,
                    'status': 'error',
                    'extra_fields': kwargs
                },
                exc_info=True
            )
            raise
            
        else:
            duration = (time.time() - start_time) * 1000
            self.logger.info(
                f"Operation '{operation}' completed in {duration:.2f}ms",
                extra={
                    'duration': duration,
                    'operation': operation,
                    'status': 'success',
                    'extra_fields': kwargs
                }
            )


def performance_monitor(operation: str = None, logger: logging.Logger = None):
    """
    Decorator for monitoring function performance
    
    Args:
        operation: Name of the operation (defaults to function name)
        logger: Logger instance (defaults to function's module logger)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            operation_name = operation or f"{func.__module__}.{func.__name__}"
            
            perf_logger = PerformanceLogger(func_logger)
            
            with perf_logger.measure(operation_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def setup_logging(
    log_level: str = None,
    log_file: str = None,
    json_format: bool = True
) -> Dict[str, Any]:
    """
    Setup logging configuration for the AI service
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        json_format: Whether to use JSON formatting
        
    Returns:
        Dict: Logging configuration
    """
    settings = get_settings()
    
    level = log_level or settings.log_level
    
    # Base logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": TTBallFormatter,
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json" if json_format else "simple",
                "level": level,
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": level,
                "handlers": ["console"],
                "propagate": False
            },
            "ttball": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    
    # Add file handler if log file is specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "formatter": "json" if json_format else "simple",
            "level": level
        }
        
        # Add file handler to loggers
        for logger_config in config["loggers"].values():
            logger_config["handlers"].append("file")
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    return config


def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger instance with TTBall_4 context
    
    Args:
        name: Logger name (defaults to caller's module)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'ttball')
    
    return logging.getLogger(name)


def log_model_performance(
    model_name: str,
    operation: str,
    duration: float,
    input_size: Optional[tuple] = None,
    output_size: Optional[tuple] = None,
    memory_usage: Optional[float] = None,
    logger: logging.Logger = None
):
    """
    Log model performance metrics
    
    Args:
        model_name: Name of the model
        operation: Operation performed (inference, training, etc.)
        duration: Operation duration in seconds
        input_size: Input data size/shape
        output_size: Output data size/shape
        memory_usage: Memory usage in MB
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger("ttball.models")
    
    performance_data = {
        "model_name": model_name,
        "operation": operation,
        "duration_ms": duration * 1000,
        "input_size": input_size,
        "output_size": output_size,
        "memory_usage_mb": memory_usage
    }
    
    logger.info(
        f"Model performance: {model_name} {operation} completed in {duration*1000:.2f}ms",
        extra={
            "extra_fields": performance_data,
            "category": "model_performance"
        }
    )


def log_request_context(request_id: str, user_id: str = None):
    """
    Create a logger with request context
    
    Args:
        request_id: Unique request identifier
        user_id: Optional user identifier
        
    Returns:
        logging.Logger: Logger with request context
    """
    logger = get_logger()
    
    # Create a custom adapter to add context to all log messages
    class ContextAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            extra = kwargs.get('extra', {})
            extra.update({
                'request_id': request_id,
                'user_id': user_id
            })
            kwargs['extra'] = extra
            return msg, kwargs
    
    return ContextAdapter(logger, {})


# Predefined loggers for different components
def get_model_logger() -> logging.Logger:
    """Get logger for model operations"""
    return get_logger("ttball.models")


def get_api_logger() -> logging.Logger:
    """Get logger for API operations"""
    return get_logger("ttball.api")


def get_video_logger() -> logging.Logger:
    """Get logger for video processing"""
    return get_logger("ttball.video")


def get_analysis_logger() -> logging.Logger:
    """Get logger for analysis operations"""
    return get_logger("ttball.analysis")


# Performance monitoring decorators for specific components
def monitor_model_performance(model_name: str):
    """Decorator for monitoring model performance"""
    return performance_monitor(
        operation=f"model.{model_name}",
        logger=get_model_logger()
    )


def monitor_api_performance(endpoint: str):
    """Decorator for monitoring API performance"""
    return performance_monitor(
        operation=f"api.{endpoint}",
        logger=get_api_logger()
    )


def monitor_video_performance(operation: str):
    """Decorator for monitoring video processing performance"""
    return performance_monitor(
        operation=f"video.{operation}",
        logger=get_video_logger()
    )


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    
    Provides a consistent logging interface for all AI service components.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
    
    @property
    def logger(self) -> logging.Logger:
        """Get or create a logger for this class"""
        if self._logger is None:
            class_name = self.__class__.__name__
            module_name = self.__class__.__module__
            logger_name = f"{module_name}.{class_name}"
            self._logger = get_logger(logger_name)
        return self._logger
    
    def log_info(self, message: str, **kwargs):
        """Log info message with optional extra fields"""
        self.logger.info(message, extra={'extra_fields': kwargs} if kwargs else None)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message with optional extra fields"""
        self.logger.warning(message, extra={'extra_fields': kwargs} if kwargs else None)
    
    def log_error(self, message: str, exception: Exception = None, **kwargs):
        """Log error message with optional exception and extra fields"""
        extra = {'extra_fields': kwargs} if kwargs else None
        if exception:
            self.logger.error(message, exc_info=exception, extra=extra)
        else:
            self.logger.error(message, extra=extra)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message with optional extra fields"""
        self.logger.debug(message, extra={'extra_fields': kwargs} if kwargs else None)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(
            f"Performance: {operation} completed in {duration:.2f}ms",
            extra={
                'duration': duration,
                'operation': operation,
                'extra_fields': kwargs
            }
        )


# Export all functions and classes for easy importing
__all__ = [
    'get_logger', 'setup_logging', 'performance_monitor', 
    'log_model_performance', 'log_request_context',
    'get_model_logger', 'get_api_logger', 'get_video_logger', 'get_analysis_logger',
    'LoggerMixin', 'PerformanceLogger', 'TTBallFormatter',
    'monitor_model_performance', 'monitor_api_performance', 'monitor_video_performance'
] 