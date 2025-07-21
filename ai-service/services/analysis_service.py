"""
TTBall_4 AI Analysis Service
Main orchestration service implementing VisionCraft 2024-2025 best practices
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from core.config import get_settings
from core.logging import get_logger, LoggerMixin
from models.ai_models import AnalysisRequest, AnalysisResponse, BallDetection, TrajectoryPrediction, AnomalyAnalysis
from models.video_models import ProcessingJob, VideoFrame, DetectionResult
from .model_manager import ModelManager, ModelType
from .video_processor import VideoProcessor
from .anomaly_service import AnomalyDetectionService

logger = get_logger(__name__)

class AnalysisStatus(str, Enum):
    """Analysis job status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class AnalysisJob:
    """Represents a complete analysis job"""
    job_id: str
    request: AnalysisRequest
    status: AnalysisStatus
    progress: float
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    results: Optional[AnalysisResponse] = None

class AIAnalysisService(LoggerMixin):
    """
    Main AI analysis service implementing VisionCraft patterns:
    - Orchestrates ModelManager and VideoProcessor
    - Manages analysis jobs with progress tracking
    - Implements async processing with background tasks
    - Provides real-time status updates
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        
        # Initialize core services
        self.model_manager = ModelManager()
        self.video_processor = VideoProcessor()
        self.anomaly_service = AnomalyDetectionService()
        
        # Job management
        self.analysis_jobs: Dict[str, AnalysisJob] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.processing_lock = asyncio.Lock()
        
        # VisionCraft: Configure concurrent analysis
        self.max_concurrent_jobs = self.settings.num_workers or 2
        self.active_jobs = 0
        
        self.log_info("AIAnalysisService initialized", extra={
            "max_concurrent_jobs": self.max_concurrent_jobs
        })

    async def startup(self):
        """Initialize the service and start background workers"""
        try:
            self.log_info("Starting AI Analysis Service")
            
            # Start background job processor
            asyncio.create_task(self._job_processor())
            
            self.log_info("AI Analysis Service started successfully")
            
        except Exception as e:
            self.log_error("Failed to start AI Analysis Service", extra={
                "error": str(e)
            })
            raise

    async def submit_analysis(self, request: AnalysisRequest) -> str:
        """
        Submit a new analysis request
        
        Args:
            request: Analysis request with video and configuration
            
        Returns:
            Job ID for tracking progress
        """
        try:
            job_id = f"analysis_{int(time.time() * 1000)}"
            
            job = AnalysisJob(
                job_id=job_id,
                request=request,
                status=AnalysisStatus.QUEUED,
                progress=0.0,
                created_at=time.time()
            )
            
            self.analysis_jobs[job_id] = job
            
            # Add to processing queue
            await self.job_queue.put(job_id)
            
            self.log_info("Analysis job submitted", extra={
                "job_id": job_id,
                "video_id": request.video_id,
                "video_path": request.video_path,
                "enable_trajectory": request.enable_trajectory,
                "enable_multimodal": request.enable_multimodal
            })
            
            return job_id
            
        except Exception as e:
            self.log_error("Failed to submit analysis", extra={
                "error": str(e),
                "video_id": getattr(request, 'video_id', 'unknown')
            })
            raise

    async def get_analysis_status(self, job_id: str) -> Optional[AnalysisJob]:
        """Get current status of an analysis job"""
        return self.analysis_jobs.get(job_id)

    async def get_analysis_results(self, job_id: str) -> Optional[AnalysisResponse]:
        """Get results of a completed analysis job"""
        job = self.analysis_jobs.get(job_id)
        if job and job.status == AnalysisStatus.COMPLETED:
            return job.results
        return None

    async def cancel_analysis(self, job_id: str) -> bool:
        """Cancel an analysis job"""
        try:
            job = self.analysis_jobs.get(job_id)
            if not job:
                return False
            
            if job.status in [AnalysisStatus.QUEUED, AnalysisStatus.PROCESSING]:
                job.status = AnalysisStatus.CANCELLED
                job.completed_at = time.time()
                
                # Cancel video processing if active
                await self.video_processor.cancel_job(job.request.video_id)
                
                self.log_info("Analysis job cancelled", extra={"job_id": job_id})
                return True
            
            return False
            
        except Exception as e:
            self.log_error("Failed to cancel analysis", extra={
                "error": str(e),
                "job_id": job_id
            })
            return False

    async def _job_processor(self):
        """Background job processor implementing VisionCraft async patterns"""
        self.log_info("Starting background job processor")
        
        while True:
            try:
                # Wait for job in queue
                job_id = await self.job_queue.get()
                
                # Check if we can process (respect concurrency limits)
                if self.active_jobs >= self.max_concurrent_jobs:
                    # Put job back in queue and wait
                    await self.job_queue.put(job_id)
                    await asyncio.sleep(1.0)
                    continue
                
                # Process job in background
                asyncio.create_task(self._process_analysis_job(job_id))
                
            except Exception as e:
                self.log_error("Job processor error", extra={"error": str(e)})
                await asyncio.sleep(1.0)

    async def _process_analysis_job(self, job_id: str):
        """Process a single analysis job with VisionCraft best practices"""
        async with self.processing_lock:
            self.active_jobs += 1
        
        try:
            job = self.analysis_jobs.get(job_id)
            if not job or job.status != AnalysisStatus.QUEUED:
                return
            
            self.log_info("Starting analysis job processing", extra={
                "job_id": job_id,
                "video_id": job.request.video_id
            })
            
            # Update job status
            job.status = AnalysisStatus.PROCESSING
            job.started_at = time.time()
            job.progress = 0.1
            
            # Stage 1: Load required models
            await self._ensure_models_loaded(job)
            job.progress = 0.2
            
            # Stage 2: Start video processing
            video_job = await self.video_processor.process_video_async(
                video_id=job.request.video_id,
                video_path=job.request.video_path,
                enable_detection=True,
                enable_tracking=job.request.enable_trajectory,
                enable_multimodal=job.request.enable_multimodal
            )
            job.progress = 0.3
            
            # Stage 3: Monitor video processing and run analysis
            results = await self._run_analysis_pipeline(job, video_job)
            job.progress = 0.9
            
            # Stage 4: Finalize results
            job.results = results
            job.status = AnalysisStatus.COMPLETED
            job.completed_at = time.time()
            job.progress = 1.0
            
            processing_time = job.completed_at - job.started_at
            
            self.log_info("Analysis job completed", extra={
                "job_id": job_id,
                "video_id": job.request.video_id,
                "processing_time": processing_time,
                "detections_count": len(results.detections) if results.detections else 0
            })
            
        except Exception as e:
            job.status = AnalysisStatus.ERROR
            job.error_message = str(e)
            job.completed_at = time.time()
            
            self.log_error("Analysis job failed", extra={
                "error": str(e),
                "job_id": job_id,
                "video_id": getattr(job.request, 'video_id', 'unknown')
            })
            
        finally:
            async with self.processing_lock:
                self.active_jobs -= 1

    async def _ensure_models_loaded(self, job: AnalysisJob):
        """Ensure required models are loaded based on job requirements"""
        try:
            models_to_load = []
            
            # Always need YOLO11 for ball detection
            models_to_load.append(ModelType.YOLO11)
            
            # Load Gemma 3n if multimodal analysis is enabled
            if job.request.enable_multimodal:
                models_to_load.append(ModelType.GEMMA_3N)
            
            for model_type in models_to_load:
                status = await self.model_manager.get_model_status(model_type)
                
                if status.name != "READY":
                    self.log_info("Loading model", extra={
                        "model_type": model_type,
                        "job_id": job.job_id
                    })
                    
                    if model_type == ModelType.YOLO11:
                        await self.model_manager.load_yolo11(
                            model_size=self.settings.yolo_model_size
                        )
                    elif model_type == ModelType.GEMMA_3N:
                        await self.model_manager.load_gemma_3n(
                            variant=self.settings.gemma_model_variant
                        )
            
            self.log_info("All required models loaded", extra={
                "job_id": job.job_id,
                "models": [m.value for m in models_to_load]
            })
            
        except Exception as e:
            self.log_error("Failed to load models", extra={
                "error": str(e),
                "job_id": job.job_id
            })
            raise

    async def _run_analysis_pipeline(
        self, 
        job: AnalysisJob, 
        video_job: ProcessingJob
    ) -> AnalysisResponse:
        """Run the complete analysis pipeline with VisionCraft async patterns"""
        try:
            self.log_info("Running analysis pipeline", extra={
                "job_id": job.job_id,
                "video_id": job.request.video_id
            })
            
            # Monitor video processing progress
            while video_job.status == "processing":
                await asyncio.sleep(0.5)
                video_status = await self.video_processor.get_processing_status(job.request.video_id)
                if video_status:
                    # Update job progress based on video processing
                    job.progress = 0.3 + (0.5 * video_status.progress)
            
            if video_job.status != "completed":
                raise RuntimeError(f"Video processing failed: {video_job.error_message}")
            
            # Run ball detection analysis
            detections = await self._detect_ball_objects(job)
            
            # Run trajectory analysis if enabled
            trajectory = None
            if job.request.enable_trajectory and detections:
                trajectory = await self._analyze_trajectory(job, detections)
            
            # Run multimodal analysis if enabled
            multimodal_results = None
            if job.request.enable_multimodal:
                multimodal_results = await self._analyze_multimodal(job)

            # Run anomaly analysis if detections are available
            anomaly_results = None
            if detections:
                # Convert detections to the format expected by anomaly_service
                frame_analyses = []
                for det in detections:
                    frame_analyses.append({
                        "timestamp": det.timestamp,
                        "ball_detected": True,
                        "confidence": det.confidence,
                        "detection_info": {
                            "center": [det.bbox[0] + det.bbox[2] / 2, det.bbox[1] + det.bbox[3] / 2],
                            "bbox": det.bbox
                        }
                    })
                
                # Placeholder for video metadata (replace with actual metadata from video_job if available)
                # For now, using dummy fps and resolution.
                # In a real scenario, video_job should provide this.
                video_metadata = {
                    "fps": 30.0,  # Assuming 30 FPS for now
                    "duration": video_job.duration, # Assuming video_job has duration
                    "resolution": [video_job.width, video_job.height] # Assuming video_job has width and height
                }
                
                anomaly_results = self.anomaly_service.analyze_anomalies_in_trajectory(frame_analyses, video_metadata)
            
            # Create analysis response
            response = AnalysisResponse(
                job_id=job.job_id,
                video_id=job.request.video_id,
                status="completed",
                processing_time=time.time() - job.started_at,
                detections=detections,
                trajectory=trajectory,
                multimodal_analysis=multimodal_results,
                anomaly_analysis=anomaly_results,
                metadata={
                    "video_path": job.request.video_path,
                    "analysis_type": job.request.analysis_type,
                    "enable_trajectory": job.request.enable_trajectory,
                    "enable_multimodal": job.request.enable_multimodal,
                    "model_versions": {
                        "yolo11": self.settings.yolo_model_size,
                        "gemma_3n": self.settings.gemma_model_variant if job.request.enable_multimodal else None
                    }
                }
            )
            
            self.log_info("Analysis pipeline completed", extra={
                "job_id": job.job_id,
                "detections_count": len(detections) if detections else 0,
                "has_trajectory": trajectory is not None,
                "has_multimodal": multimodal_results is not None
            })
            
            return response
            
        except Exception as e:
            self.log_error("Analysis pipeline failed", extra={
                "error": str(e),
                "job_id": job.job_id
            })
            raise

    async def _detect_ball_objects(self, job: AnalysisJob) -> List[BallDetection]:
        """Detect ball objects using YOLO11"""
        try:
            self.log_info("Running ball detection", extra={"job_id": job.job_id})
            
            # This is where we'll integrate with the actual YOLO11 model
            # For now, simulate detection results
            detections = []
            
            # Simulate processing video frames
            await asyncio.sleep(1.0)
            
            # Create sample detections for demonstration
            for i in range(10):
                detection = BallDetection(
                    confidence=0.85 + (i * 0.01),
                    bbox=[100.0 + i * 10, 150.0 + i * 5, 50.0, 50.0],
                    frame_number=i * 5,
                    timestamp=i * 0.167  # ~6 FPS
                )
                detections.append(detection)
            
            self.log_info("Ball detection completed", extra={
                "job_id": job.job_id,
                "detections_count": len(detections)
            })
            
            return detections
            
        except Exception as e:
            self.log_error("Ball detection failed", extra={
                "error": str(e),
                "job_id": job.job_id
            })
            raise

    async def _analyze_trajectory(
        self, 
        job: AnalysisJob, 
        detections: List[BallDetection]
    ) -> Optional[TrajectoryPrediction]:
        """Analyze ball trajectory using physics-based models"""
        try:
            if not detections:
                return None
            
            self.log_info("Running trajectory analysis", extra={"job_id": job.job_id})
            
            # Simulate trajectory analysis
            await asyncio.sleep(0.5)
            
            # Calculate trajectory from detections
            trajectory_points = []
            for detection in detections:
                # Convert bbox to center point
                center_x = detection.bbox[0] + detection.bbox[2] / 2
                center_y = detection.bbox[1] + detection.bbox[3] / 2
                trajectory_points.append([center_x, center_y, detection.timestamp])
            
            trajectory = TrajectoryPrediction(
                trajectory_points=trajectory_points,
                predicted_path=trajectory_points,  # Simplified prediction
                physics_analysis={
                    "velocity": {"x": 150.0, "y": -50.0, "magnitude": 158.1},
                    "acceleration": {"x": 0.0, "y": 9.81},
                    "spin": {"rate": 2500.0, "axis": "topspin"}
                },
                confidence=0.87
            )
            
            self.log_info("Trajectory analysis completed", extra={
                "job_id": job.job_id,
                "trajectory_points": len(trajectory_points),
                "confidence": trajectory.confidence
            })
            
            return trajectory
            
        except Exception as e:
            self.log_error("Trajectory analysis failed", extra={
                "error": str(e),
                "job_id": job.job_id
            })
            return None

    async def _analyze_multimodal(self, job: AnalysisJob) -> Optional[Dict[str, Any]]:
        """Analyze video using Gemma 3n multimodal capabilities"""
        try:
            self.log_info("Running multimodal analysis", extra={"job_id": job.job_id})
            
            # This is where we'll integrate with Gemma 3n
            # For now, simulate multimodal analysis
            await asyncio.sleep(2.0)
            
            multimodal_results = {
                "scene_description": "Table tennis match with two players, indoor setting with good lighting",
                "action_recognition": [
                    {"action": "forehand_stroke", "timestamp": 1.2, "confidence": 0.92},
                    {"action": "backhand_stroke", "timestamp": 3.5, "confidence": 0.88},
                    {"action": "serve", "timestamp": 0.1, "confidence": 0.95}
                ],
                "game_analysis": {
                    "rally_length": 8,
                    "stroke_count": 12,
                    "estimated_skill_level": "intermediate",
                    "game_pace": "moderate"
                },
                "technical_analysis": {
                    "ball_spin_detected": True,
                    "playing_style": "aggressive baseline",
                    "court_coverage": "good"
                }
            }
            
            self.log_info("Multimodal analysis completed", extra={
                "job_id": job.job_id,
                "actions_detected": len(multimodal_results["action_recognition"])
            })
            
            return multimodal_results
            
        except Exception as e:
            self.log_error("Multimodal analysis failed", extra={
                "error": str(e),
                "job_id": job.job_id
            })
            return None

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get model manager status
            model_resources = await self.model_manager.get_system_resources()
            
            # Get active jobs
            active_jobs = [job for job in self.analysis_jobs.values() 
                          if job.status == AnalysisStatus.PROCESSING]
            
            # Get video processor status
            video_jobs = await self.video_processor.get_active_jobs()
            
            status = {
                "service": "ai_analysis",
                "active_analysis_jobs": len(active_jobs),
                "active_video_jobs": len(video_jobs),
                "total_jobs_processed": len(self.analysis_jobs),
                "system_resources": model_resources,
                "service_health": "healthy",
                "uptime": time.time() - getattr(self, '_startup_time', time.time())
            }
            
            return status
            
        except Exception as e:
            self.log_error("Failed to get system status", extra={"error": str(e)})
            return {"service": "ai_analysis", "status": "error", "error": str(e)}

    async def cleanup(self):
        """Cleanup service resources"""
        try:
            self.log_info("Cleaning up AI Analysis Service")
            
            # Cancel all active jobs
            for job_id, job in self.analysis_jobs.items():
                if job.status == AnalysisStatus.PROCESSING:
                    await self.cancel_analysis(job_id)
            
            # Cleanup model manager
            await self.model_manager.cleanup()
            
            self.log_info("AI Analysis Service cleanup completed")
            
        except Exception as e:
            self.log_error("Failed to cleanup AI Analysis Service", extra={
                "error": str(e)
            }) 