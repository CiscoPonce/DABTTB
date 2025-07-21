"""
TTBall_4 Video Processor Service
Implements VisionCraft 2024-2025 best practices for async video processing
"""

import asyncio
import cv2
import numpy as np
import tempfile
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time

from core.config import get_settings
from core.logging import get_logger, LoggerMixin
from models.video_models import VideoFrame, VideoSegment, ProcessingJob, VideoMetadata

logger = get_logger(__name__)

class ProcessingStage(str, Enum):
    """Video processing stages"""
    VALIDATION = "validation"
    EXTRACTION = "extraction"
    DETECTION = "detection"
    TRACKING = "tracking"
    ANALYSIS = "analysis"
    COMPLETE = "complete"
    ERROR = "error"

@dataclass
class FrameData:
    """Represents a single frame with metadata"""
    frame_number: int
    timestamp: float
    image_data: np.ndarray
    width: int
    height: int
    processed: bool = False
    detections: List[Any] = None

class VideoProcessor(LoggerMixin):
    """
    Advanced video processor implementing VisionCraft best practices:
    - Async frame processing with concurrent analysis
    - Memory-efficient streaming
    - Real-time progress tracking
    - Background task processing
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.processing_jobs: Dict[str, ProcessingJob] = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "ttball4_video"
        self.temp_dir.mkdir(exist_ok=True)
        
        # VisionCraft: Configure processing parameters
        self.max_concurrent_frames = self.settings.num_workers or 4
        self.frame_queue_size = 100
        self.chunk_size = 30  # Process 30 frames at a time
        
        self.log_info("VideoProcessor initialized", extra={
            "max_concurrent_frames": self.max_concurrent_frames,
            "frame_queue_size": self.frame_queue_size,
            "chunk_size": self.chunk_size
        })

    async def process_video_async(
        self, 
        video_id: str, 
        video_path: str,
        enable_detection: bool = True,
        enable_tracking: bool = True,
        enable_multimodal: bool = False
    ) -> ProcessingJob:
        """
        Process video asynchronously with real-time progress updates
        
        Args:
            video_id: Unique identifier for the video
            video_path: Path to the video file
            enable_detection: Enable object detection
            enable_tracking: Enable multi-object tracking
            enable_multimodal: Enable multimodal analysis
            
        Returns:
            ProcessingJob with job details
        """
        try:
            self.log_info("Starting async video processing", extra={
                "video_id": video_id,
                "video_path": video_path,
                "enable_detection": enable_detection,
                "enable_tracking": enable_tracking,
                "enable_multimodal": enable_multimodal
            })
            
            # Create processing job
            job = ProcessingJob(
                job_id=video_id,
                video_path=video_path,
                status="processing",
                progress=0.0,
                current_stage=ProcessingStage.VALIDATION,
                enable_detection=enable_detection,
                enable_tracking=enable_tracking,
                enable_multimodal=enable_multimodal,
                created_at=time.time()
            )
            
            self.processing_jobs[video_id] = job
            
            # Start background processing task
            asyncio.create_task(self._process_video_background(job))
            
            return job
            
        except Exception as e:
            self.log_error("Failed to start video processing", extra={
                "error": str(e),
                "video_id": video_id
            })
            raise

    async def get_processing_status(self, video_id: str) -> Optional[ProcessingJob]:
        """Get current processing status for a video"""
        return self.processing_jobs.get(video_id)

    async def extract_frames_stream(
        self, 
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        Stream video frames asynchronously (VisionCraft best practice)
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (None for entire video)
            max_frames: Maximum number of frames to extract
            
        Yields:
            VideoFrame objects with frame data
        """
        try:
            self.log_info("Starting frame extraction stream", extra={
                "video_path": video_path,
                "start_time": start_time,
                "end_time": end_time,
                "max_frames": max_frames
            })
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate frame range
            start_frame = int(start_time * fps) if start_time > 0 else 0
            end_frame = int(end_time * fps) if end_time else total_frames
            
            if max_frames:
                end_frame = min(end_frame, start_frame + max_frames)
            
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = 0
            current_frame = start_frame
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = current_frame / fps
                
                # Create VideoFrame object
                video_frame = VideoFrame(
                    frame_number=current_frame,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                    format="BGR",
                    data=frame.tobytes() if self.settings.include_frame_data else None
                )
                
                yield video_frame
                
                frame_count += 1
                current_frame += 1
                
                # VisionCraft: Yield control for async processing
                if frame_count % 10 == 0:
                    await asyncio.sleep(0.001)  # Allow other tasks to run
            
            cap.release()
            
            self.log_info("Frame extraction completed", extra={
                "frames_extracted": frame_count,
                "video_path": video_path
            })
            
        except Exception as e:
            self.log_error("Frame extraction failed", extra={
                "error": str(e),
                "video_path": video_path
            })
            raise

    async def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract comprehensive video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoMetadata with video properties
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Extract metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + \
                   chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
            
            cap.release()
            
            # Get file size
            file_size = Path(video_path).stat().st_size
            
            metadata = VideoMetadata(
                filename=Path(video_path).name,
                file_size_bytes=file_size,
                duration_seconds=duration,
                fps=fps,
                total_frames=total_frames,
                width=width,
                height=height,
                codec=codec,
                format="video"
            )
            
            self.log_info("Video metadata extracted", extra={
                "video_path": video_path,
                "duration": duration,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "total_frames": total_frames
            })
            
            return metadata
            
        except Exception as e:
            self.log_error("Failed to extract video metadata", extra={
                "error": str(e),
                "video_path": video_path
            })
            raise

    async def process_frames_concurrent(
        self,
        frames: List[FrameData],
        processor_func,
        max_workers: Optional[int] = None
    ) -> List[FrameData]:
        """
        Process multiple frames concurrently (VisionCraft best practice)
        
        Args:
            frames: List of frames to process
            processor_func: Function to process each frame
            max_workers: Maximum concurrent workers
            
        Returns:
            List of processed frames
        """
        try:
            max_workers = max_workers or self.max_concurrent_frames
            
            self.log_info("Starting concurrent frame processing", extra={
                "frame_count": len(frames),
                "max_workers": max_workers
            })
            
            # Create semaphore to limit concurrent processing
            semaphore = asyncio.Semaphore(max_workers)
            
            async def process_single_frame(frame_data: FrameData) -> FrameData:
                async with semaphore:
                    try:
                        processed_frame = await processor_func(frame_data)
                        processed_frame.processed = True
                        return processed_frame
                    except Exception as e:
                        self.log_error("Frame processing failed", extra={
                            "error": str(e),
                            "frame_number": frame_data.frame_number
                        })
                        return frame_data
            
            # Process all frames concurrently
            tasks = [process_single_frame(frame) for frame in frames]
            processed_frames = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log errors
            valid_frames = []
            for i, result in enumerate(processed_frames):
                if isinstance(result, Exception):
                    self.log_error("Frame processing exception", extra={
                        "error": str(result),
                        "frame_number": frames[i].frame_number
                    })
                    valid_frames.append(frames[i])  # Return original frame
                else:
                    valid_frames.append(result)
            
            self.log_info("Concurrent frame processing completed", extra={
                "processed_count": len(valid_frames),
                "successful_count": sum(1 for f in valid_frames if f.processed)
            })
            
            return valid_frames
            
        except Exception as e:
            self.log_error("Concurrent frame processing failed", extra={
                "error": str(e),
                "frame_count": len(frames)
            })
            raise

    async def create_video_segments(
        self,
        video_path: str,
        segment_duration: float = 10.0,
        overlap: float = 1.0
    ) -> List[VideoSegment]:
        """
        Create video segments for batch processing
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Returns:
            List of VideoSegment objects
        """
        try:
            metadata = await self.get_video_metadata(video_path)
            
            segments = []
            current_time = 0.0
            segment_id = 0
            
            while current_time < metadata.duration_seconds:
                end_time = min(current_time + segment_duration, metadata.duration_seconds)
                
                segment = VideoSegment(
                    segment_id=segment_id,
                    start_time=current_time,
                    end_time=end_time,
                    duration=end_time - current_time,
                    frame_count=int((end_time - current_time) * metadata.fps)
                )
                
                segments.append(segment)
                
                # Move to next segment with overlap
                current_time += segment_duration - overlap
                segment_id += 1
            
            self.log_info("Video segments created", extra={
                "video_path": video_path,
                "segment_count": len(segments),
                "segment_duration": segment_duration,
                "overlap": overlap
            })
            
            return segments
            
        except Exception as e:
            self.log_error("Failed to create video segments", extra={
                "error": str(e),
                "video_path": video_path
            })
            raise

    async def _process_video_background(self, job: ProcessingJob):
        """Background task for video processing with progress updates"""
        try:
            self.log_info("Starting background video processing", extra={
                "job_id": job.job_id,
                "video_path": job.video_path
            })
            
            # Stage 1: Validation
            job.current_stage = ProcessingStage.VALIDATION
            job.progress = 0.1
            await self._validate_video(job)
            
            # Stage 2: Metadata extraction
            job.current_stage = ProcessingStage.EXTRACTION
            job.progress = 0.2
            metadata = await self.get_video_metadata(job.video_path)
            job.metadata = metadata
            
            # Stage 3: Frame processing
            job.current_stage = ProcessingStage.DETECTION
            job.progress = 0.3
            
            if job.enable_detection or job.enable_tracking:
                await self._process_frames_with_ai(job)
            
            # Stage 4: Multimodal analysis
            if job.enable_multimodal:
                job.current_stage = ProcessingStage.ANALYSIS
                job.progress = 0.8
                await self._process_multimodal_analysis(job)
            
            # Stage 5: Complete
            job.current_stage = ProcessingStage.COMPLETE
            job.progress = 1.0
            job.status = "completed"
            job.completed_at = time.time()
            
            self.log_info("Background video processing completed", extra={
                "job_id": job.job_id,
                "processing_time": job.completed_at - job.created_at
            })
            
        except Exception as e:
            job.current_stage = ProcessingStage.ERROR
            job.status = "error"
            job.error_message = str(e)
            job.completed_at = time.time()
            
            self.log_error("Background video processing failed", extra={
                "error": str(e),
                "job_id": job.job_id
            })

    async def _validate_video(self, job: ProcessingJob):
        """Validate video file and properties"""
        try:
            if not Path(job.video_path).exists():
                raise ValueError(f"Video file not found: {job.video_path}")
            
            # Test if video can be opened
            cap = cv2.VideoCapture(job.video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {job.video_path}")
            
            # Check basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            # Validate video properties
            if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
                raise ValueError("Invalid video properties detected")
            
            # Check minimum requirements
            if width < 320 or height < 240:
                raise ValueError("Video resolution too low (minimum 320x240)")
            
            if total_frames < 10:
                raise ValueError("Video too short (minimum 10 frames)")
            
            self.log_info("Video validation passed", extra={
                "job_id": job.job_id,
                "fps": fps,
                "total_frames": total_frames,
                "resolution": f"{width}x{height}"
            })
            
        except Exception as e:
            self.log_error("Video validation failed", extra={
                "error": str(e),
                "job_id": job.job_id
            })
            raise

    async def _process_frames_with_ai(self, job: ProcessingJob):
        """Process frames with AI models (placeholder for model integration)"""
        try:
            self.log_info("Processing frames with AI", extra={
                "job_id": job.job_id,
                "enable_detection": job.enable_detection,
                "enable_tracking": job.enable_tracking
            })
            
            # This is where we'll integrate with ModelManager
            # For now, simulate processing
            metadata = job.metadata
            total_frames = metadata.total_frames
            
            processed_frames = 0
            chunk_size = min(self.chunk_size, total_frames)
            
            while processed_frames < total_frames:
                # Simulate processing a chunk of frames
                await asyncio.sleep(0.1)  # Simulate processing time
                
                processed_frames += chunk_size
                job.progress = 0.3 + (0.5 * processed_frames / total_frames)
                
                if processed_frames % 100 == 0:
                    self.log_info("Frame processing progress", extra={
                        "job_id": job.job_id,
                        "processed_frames": processed_frames,
                        "total_frames": total_frames,
                        "progress": job.progress
                    })
            
            self.log_info("AI frame processing completed", extra={
                "job_id": job.job_id,
                "total_frames": total_frames
            })
            
        except Exception as e:
            self.log_error("AI frame processing failed", extra={
                "error": str(e),
                "job_id": job.job_id
            })
            raise

    async def _process_multimodal_analysis(self, job: ProcessingJob):
        """Process video with multimodal AI (placeholder for Gemma 3n integration)"""
        try:
            self.log_info("Processing multimodal analysis", extra={
                "job_id": job.job_id
            })
            
            # This is where we'll integrate with Gemma 3n
            # For now, simulate multimodal processing
            await asyncio.sleep(2.0)  # Simulate multimodal processing time
            
            self.log_info("Multimodal analysis completed", extra={
                "job_id": job.job_id
            })
            
        except Exception as e:
            self.log_error("Multimodal analysis failed", extra={
                "error": str(e),
                "job_id": job.job_id
            })
            raise

    async def cleanup_job(self, video_id: str) -> bool:
        """Clean up processing job and temporary files"""
        try:
            if video_id in self.processing_jobs:
                del self.processing_jobs[video_id]
            
            # Clean up temporary files
            temp_files = list(self.temp_dir.glob(f"*{video_id}*"))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            
            self.log_info("Job cleanup completed", extra={
                "video_id": video_id,
                "cleaned_files": len(temp_files)
            })
            
            return True
            
        except Exception as e:
            self.log_error("Job cleanup failed", extra={
                "error": str(e),
                "video_id": video_id
            })
            return False

    async def get_active_jobs(self) -> List[ProcessingJob]:
        """Get all currently active processing jobs"""
        return [job for job in self.processing_jobs.values() if job.status == "processing"]

    async def cancel_job(self, video_id: str) -> bool:
        """Cancel a processing job"""
        try:
            if video_id in self.processing_jobs:
                job = self.processing_jobs[video_id]
                job.status = "cancelled"
                job.completed_at = time.time()
                
                self.log_info("Job cancelled", extra={"video_id": video_id})
                return True
            
            return False
            
        except Exception as e:
            self.log_error("Failed to cancel job", extra={
                "error": str(e),
                "video_id": video_id
            })
            return False

# VideoProcessor Service Placeholder 