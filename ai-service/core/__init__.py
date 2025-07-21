"""
TTBall_4 AI Service Core Package

This package contains the core functionality for the AI service including:
- Configuration management
- Dependency injection
- Logging utilities
- Base classes and utilities
"""

from .config import Settings, get_settings
# Commented out to fix circular import issue
# from .dependencies import get_model_manager, get_video_processor, get_analysis_service
from .logging import setup_logging, get_logger

__all__ = [
    "Settings",
    "get_settings", 
    # "get_model_manager",
    # "get_video_processor", 
    # "get_analysis_service",
    "setup_logging",
    "get_logger"
]

__version__ = "1.0.0" 