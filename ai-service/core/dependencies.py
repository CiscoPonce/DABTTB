"""
Dependency injection for TTBall_4 AI Service

Provides FastAPI dependencies for model manager, video processor,
and analysis service with proper lifecycle management.
"""

from functools import lru_cache
from typing import Generator

from fastapi import Depends

from .config import Settings, get_settings
from .logging import get_logger
from services.model_manager import ModelManager
from services.video_processor import VideoProcessor
from services.analysis_service import AIAnalysisService


# Logger for dependency management
logger = get_logger(__name__)


@lru_cache()
def get_model_manager(settings: Settings = Depends(get_settings)) -> ModelManager:
    """
    Get or create ModelManager instance
    
    Args:
        settings: Application settings
        
    Returns:
        ModelManager: Configured model manager instance
    """
    logger.info("Initializing ModelManager")
    
    try:
        model_manager = ModelManager(
            model_path=settings.model_path,
            gemma_model_name=settings.gemma_model_name,
            device=settings.device,
            confidence=settings.detection_confidence,
            iou=settings.detection_iou
        )
        
        logger.info("ModelManager initialized successfully")
        return model_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize ModelManager: {e}", exc_info=True)
        raise


@lru_cache()
def get_video_processor(
    settings: Settings = Depends(get_settings),
    model_manager: ModelManager = Depends(get_model_manager)
) -> VideoProcessor:
    """
    Get or create VideoProcessor instance
    
    Args:
        settings: Application settings
        model_manager: Model manager instance
        
    Returns:
        VideoProcessor: Configured video processor instance
    """
    logger.info("Initializing VideoProcessor")
    
    try:
        video_processor = VideoProcessor(
            model_manager=model_manager,
            batch_size=settings.batch_size,
            timeout=settings.processing_timeout
        )
        
        logger.info("VideoProcessor initialized successfully")
        return video_processor
        
    except Exception as e:
        logger.error(f"Failed to initialize VideoProcessor: {e}", exc_info=True)
        raise


@lru_cache()
def get_analysis_service(
    settings: Settings = Depends(get_settings),
    model_manager: ModelManager = Depends(get_model_manager),
    video_processor: VideoProcessor = Depends(get_video_processor)
) -> AIAnalysisService:
    """
    Get or create AnalysisService instance
    
    Args:
        settings: Application settings
        model_manager: Model manager instance
        video_processor: Video processor instance
        
    Returns:
        AnalysisService: Configured analysis service instance
    """
    logger.info("Initializing AnalysisService")
    
    try:
        analysis_service = AIAnalysisService(
            model_manager=model_manager,
            video_processor=video_processor,
            enable_trajectory=settings.enable_trajectory,
            enable_3d_analysis=settings.enable_3d_analysis
        )
        
        logger.info("AnalysisService initialized successfully")
        return analysis_service
        
    except Exception as e:
        logger.error(f"Failed to initialize AnalysisService: {e}", exc_info=True)
        raise


# Dependency for graceful shutdown
def get_shutdown_handler(
    model_manager: ModelManager = Depends(get_model_manager),
    video_processor: VideoProcessor = Depends(get_video_processor),
    analysis_service: AIAnalysisService = Depends(get_analysis_service)
) -> Generator[None, None, None]:
    """
    Dependency for handling graceful shutdown
    
    Yields control and then performs cleanup on shutdown
    """
    try:
        yield
    finally:
        logger.info("Performing graceful shutdown")
        
        # Cleanup services in reverse order
        try:
            if hasattr(analysis_service, 'cleanup'):
                analysis_service.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up AnalysisService: {e}")
        
        try:
            if hasattr(video_processor, 'cleanup'):
                video_processor.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up VideoProcessor: {e}")
        
        try:
            if hasattr(model_manager, 'cleanup'):
                model_manager.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up ModelManager: {e}")
        
        logger.info("Graceful shutdown completed")


# Health check dependencies
def get_health_status(
    model_manager: ModelManager = Depends(get_model_manager),
    video_processor: VideoProcessor = Depends(get_video_processor),
    analysis_service: AIAnalysisService = Depends(get_analysis_service)
) -> dict:
    """
    Get health status of all services
    
    Returns:
        dict: Health status information
    """
    status = {
        "status": "healthy",
        "services": {}
    }
    
    # Check ModelManager health
    try:
        model_health = model_manager.health_check() if hasattr(model_manager, 'health_check') else {"status": "unknown"}
        status["services"]["model_manager"] = model_health
    except Exception as e:
        status["services"]["model_manager"] = {"status": "unhealthy", "error": str(e)}
        status["status"] = "degraded"
    
    # Check VideoProcessor health
    try:
        video_health = video_processor.health_check() if hasattr(video_processor, 'health_check') else {"status": "unknown"}
        status["services"]["video_processor"] = video_health
    except Exception as e:
        status["services"]["video_processor"] = {"status": "unhealthy", "error": str(e)}
        status["status"] = "degraded"
    
    # Check AnalysisService health
    try:
        analysis_health = analysis_service.health_check() if hasattr(analysis_service, 'health_check') else {"status": "unknown"}
        status["services"]["analysis_service"] = analysis_health
    except Exception as e:
        status["services"]["analysis_service"] = {"status": "unhealthy", "error": str(e)}
        status["status"] = "degraded"
    
    # Determine overall status
    unhealthy_services = [
        name for name, service_status in status["services"].items()
        if service_status.get("status") == "unhealthy"
    ]
    
    if unhealthy_services:
        if len(unhealthy_services) == len(status["services"]):
            status["status"] = "unhealthy"
        else:
            status["status"] = "degraded"
    
    return status


# Performance monitoring dependency
def get_performance_metrics(
    model_manager: ModelManager = Depends(get_model_manager),
    video_processor: VideoProcessor = Depends(get_video_processor)
) -> dict:
    """
    Get performance metrics from services
    
    Returns:
        dict: Performance metrics
    """
    metrics = {
        "model_manager": {},
        "video_processor": {},
        "system": {}
    }
    
    # Get model manager metrics
    try:
        if hasattr(model_manager, 'get_metrics'):
            metrics["model_manager"] = model_manager.get_metrics()
    except Exception as e:
        logger.error(f"Error getting model manager metrics: {e}")
        metrics["model_manager"] = {"error": str(e)}
    
    # Get video processor metrics
    try:
        if hasattr(video_processor, 'get_metrics'):
            metrics["video_processor"] = video_processor.get_metrics()
    except Exception as e:
        logger.error(f"Error getting video processor metrics: {e}")
        metrics["video_processor"] = {"error": str(e)}
    
    # Get system metrics
    try:
        import psutil
        metrics["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        
        # Add GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                metrics["system"]["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)
                metrics["system"]["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)
        except ImportError:
            pass
            
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        metrics["system"] = {"error": str(e)}
    
    return metrics


# Validation dependencies
def validate_file_upload(
    settings: Settings = Depends(get_settings)
):
    """
    Dependency for validating file uploads
    
    Args:
        settings: Application settings
        
    Returns:
        Callable: File validation function
    """
    def validate_file(filename: str, content_length: int = 0) -> dict:
        """
        Validate uploaded file
        
        Args:
            filename: Name of the uploaded file
            content_length: Size of the file in bytes
            
        Returns:
            dict: Validation result
        """
        result = {"valid": True, "errors": []}
        
        # Check file extension
        if '.' not in filename:
            result["valid"] = False
            result["errors"].append("File must have an extension")
        else:
            extension = filename.rsplit('.', 1)[1].lower()
            if extension not in settings.allowed_extensions:
                result["valid"] = False
                result["errors"].append(f"Extension '{extension}' not allowed. Allowed: {settings.allowed_extensions}")
        
        # Check file size
        if content_length > settings.max_file_size:
            result["valid"] = False
            result["errors"].append(f"File too large. Max size: {settings.max_file_size / (1024*1024):.1f}MB")
        
        return result
    
    return validate_file


# Rate limiting dependency (placeholder for future implementation)
def get_rate_limiter():
    """
    Get rate limiter instance
    
    Placeholder for future rate limiting implementation
    """
    class NoOpRateLimiter:
        def check_rate_limit(self, key: str) -> bool:
            return True
    
    return NoOpRateLimiter() 