"""
Configuration management for TTBall_4 AI Service

Handles environment variables, model configurations, and service settings
with proper validation and defaults.
"""

import os
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """
    
    # Service Configuration
    service_name: str = Field(default="ttball5-ai-service", env="SERVICE_NAME")
    service_version: str = Field(default="3.0.0", env="SERVICE_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8005, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Model Configuration - Updated to use local model path
    model_path: str = Field(default="/app/models/yolo11n.pt", env="MODEL_PATH")
    gemma_model_name: str = Field(default="../models/gemma-3n-E4B", env="GEMMA_MODEL_NAME")
    gemma_use_local: bool = Field(default=True, env="GEMMA_USE_LOCAL")
    detection_confidence: float = Field(default=0.6, env="DETECTION_CONFIDENCE")
    detection_iou: float = Field(default=0.4, env="DETECTION_IOU")
    
    # Processing Configuration
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_extensions: List[str] = Field(
        default=["mp4", "avi", "mov", "mkv", "webm"], 
        env="ALLOWED_EXTENSIONS"
    )
    processing_timeout: int = Field(default=300, env="PROCESSING_TIMEOUT")  # 5 minutes
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    
    # Storage Configuration
    upload_dir: str = Field(default="/tmp/uploads", env="UPLOAD_DIR")
    results_dir: str = Field(default="/tmp/results", env="RESULTS_DIR")
    
    # External Services
    backend_url: str = Field(default="http://backend:8000", env="BACKEND_URL")
    database_url: str = Field(default="", env="DATABASE_URL")
    redis_url: str = Field(default="redis://redis:6379", env="REDIS_URL")
    
    # AI/ML Configuration
    device: str = Field(default="cpu", env="DEVICE")  # cpu, cuda, mps
    num_workers: int = Field(default=2, env="NUM_WORKERS")
    enable_trajectory: bool = Field(default=True, env="ENABLE_TRAJECTORY")
    enable_3d_analysis: bool = Field(default=True, env="ENABLE_3D_ANALYSIS")
    
    # Performance Configuration
    cache_size: int = Field(default=1000, env="CACHE_SIZE")
    max_concurrent_jobs: int = Field(default=5, env="MAX_CONCURRENT_JOBS")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("detection_confidence", "detection_iou")
    @classmethod
    def validate_confidence_iou(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence and IoU values must be between 0.0 and 1.0")
        return v
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        valid_devices = ["cpu", "cuda", "mps"]
        if v.lower() not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v.lower()
    
    @field_validator("allowed_extensions")
    @classmethod
    def validate_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return [ext.lower() for ext in v]
    
    def get_gemma_model_path(self):
        """Get the full path to the Gemma model"""
        if self.gemma_use_local:
            # Use local model path relative to the project root
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", self.gemma_model_name))
        else:
            # Use Hugging Face model name for download
            return "google/gemma-3n-E4B"
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create model directory if specified
        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Create local Gemma model directory if using local files
        if self.gemma_use_local:
            gemma_path = self.get_gemma_model_path()
            gemma_dir = os.path.dirname(gemma_path)
            if gemma_dir:
                os.makedirs(gemma_dir, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Returns:
        Settings: Application settings
    """
    settings = Settings()
    settings.create_directories()
    return settings


# Development and testing configurations
class DevelopmentSettings(Settings):
    """Development-specific settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    reload: bool = True
    

class TestingSettings(Settings):
    """Testing-specific settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    upload_dir: str = "/tmp/test_uploads"
    results_dir: str = "/tmp/test_results"
    

class ProductionSettings(Settings):
    """Production-specific settings"""
    debug: bool = False
    log_level: str = "INFO"
    reload: bool = False


def get_settings_for_environment(env: str = None) -> Settings:
    """
    Get settings for specific environment
    
    Args:
        env: Environment name (development, testing, production)
        
    Returns:
        Settings: Environment-specific settings
    """
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    if env.lower() == "development":
        return DevelopmentSettings()
    elif env.lower() == "testing":
        return TestingSettings()
    elif env.lower() == "production":
        return ProductionSettings()
    else:
        return Settings() 