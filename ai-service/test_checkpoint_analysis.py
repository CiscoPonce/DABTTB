"""
Test script for checkpoint-based video analysis
"""

import os
import sys
import cv2
import numpy as np
import tempfile
import logging
from pathlib import Path

# Add the ai-service directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_main import (
    extract_video_metadata,
    extract_checkpoint_frames,
    analyze_frame_content,
    analyze_video_with_checkpoints
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_video(duration_seconds: int = 10, fps: int = 30) -> str:
    """Create a simple test video for analysis"""
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Video properties
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create video writer
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        total_frames = duration_seconds * fps
        
        logger.info(f"🎬 Creating test video: {duration_seconds}s at {fps} FPS ({total_frames} frames)")
        
        for frame_num in range(total_frames):
            # Create a frame with moving ball simulation
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Background color (green table)
            frame[:, :] = [50, 150, 50]
            
            # Simulate ball movement (circular motion)
            t = frame_num / fps  # Time in seconds
            center_x = int(width // 2 + 100 * np.cos(t * 2))
            center_y = int(height // 2 + 50 * np.sin(t * 2))
            
            # Draw ball (white circle)
            cv2.circle(frame, (center_x, center_y), 10, (255, 255, 255), -1)
            
            # Add some noise/texture
            noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            out.write(frame)
        
        out.release()
        
        # Verify the video was created
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            logger.info(f"✅ Test video created: {temp_path} ({os.path.getsize(temp_path)} bytes)")
            return temp_path
        else:
            raise Exception("Failed to create test video")
            
    except Exception as e:
        logger.error(f"❌ Failed to create test video: {e}")
        raise


def test_video_metadata_extraction():
    """Test video metadata extraction"""
    logger.info("\n🔬 Testing video metadata extraction...")
    
    try:
        # Create test video
        test_video_path = create_test_video(duration_seconds=5, fps=25)
        
        # Extract metadata
        metadata = extract_video_metadata(test_video_path)
        
        logger.info(f"📹 Extracted metadata: {metadata}")
        
        # Verify metadata
        assert metadata["fps"] == 25, f"Expected FPS 25, got {metadata['fps']}"
        assert abs(metadata["duration_seconds"] - 5.0) < 0.5, f"Expected ~5s duration, got {metadata['duration_seconds']}"
        assert metadata["width"] == 640, f"Expected width 640, got {metadata['width']}"
        assert metadata["height"] == 480, f"Expected height 480, got {metadata['height']}"
        
        logger.info("✅ Video metadata extraction test passed")
        
        # Cleanup
        os.unlink(test_video_path)
        
    except Exception as e:
        logger.error(f"❌ Video metadata extraction test failed: {e}")
        raise


def test_checkpoint_frame_extraction():
    """Test checkpoint frame extraction"""
    logger.info("\n🔬 Testing checkpoint frame extraction...")
    
    try:
        # Create test video
        test_video_path = create_test_video(duration_seconds=6, fps=30)
        
        # Extract checkpoint frames (every 1 second)
        checkpoint_frames = extract_checkpoint_frames(test_video_path, checkpoint_interval=1.0)
        
        logger.info(f"📸 Extracted {len(checkpoint_frames)} checkpoint frames")
        
        # Verify we got the expected number of frames
        expected_frames = 6  # 6 seconds with 1-second intervals
        assert len(checkpoint_frames) >= expected_frames - 1, f"Expected ~{expected_frames} frames, got {len(checkpoint_frames)}"
        
        # Verify frame data
        for i, (timestamp, frame) in enumerate(checkpoint_frames[:3]):
            logger.info(f"   Frame {i}: timestamp={timestamp:.1f}s, shape={frame.shape}")
            assert frame.shape == (480, 640, 3), f"Unexpected frame shape: {frame.shape}"
            assert abs(timestamp - i) < 0.1, f"Unexpected timestamp: {timestamp}"
        
        logger.info("✅ Checkpoint frame extraction test passed")
        
        # Cleanup
        os.unlink(test_video_path)
        
    except Exception as e:
        logger.error(f"❌ Checkpoint frame extraction test failed: {e}")
        raise


def test_frame_analysis():
    """Test individual frame analysis"""
    logger.info("\n🔬 Testing frame analysis...")
    
    try:
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:, :] = [50, 150, 50]  # Green background
        
        # Add a white ball
        cv2.circle(test_frame, (320, 240), 15, (255, 255, 255), -1)
        
        # Analyze the frame
        analysis = analyze_frame_content(2.5, test_frame)
        
        logger.info(f"🔍 Frame analysis result: {analysis}")
        
        # Verify analysis results
        assert analysis["timestamp"] == 2.5, f"Unexpected timestamp: {analysis['timestamp']}"
        assert analysis["resolution"] == "640x480", f"Unexpected resolution: {analysis['resolution']}"
        assert "brightness" in analysis, "Missing brightness analysis"
        assert "edge_density" in analysis, "Missing edge density analysis"
        assert "ball_detected" in analysis, "Missing ball detection"
        assert "confidence" in analysis, "Missing confidence score"
        
        logger.info("✅ Frame analysis test passed")
        
    except Exception as e:
        logger.error(f"❌ Frame analysis test failed: {e}")
        raise


def test_full_checkpoint_analysis():
    """Test complete checkpoint-based video analysis"""
    logger.info("\n🔬 Testing full checkpoint analysis...")
    
    try:
        # Create test video
        test_video_path = create_test_video(duration_seconds=8, fps=30)
        file_size = os.path.getsize(test_video_path)
        
        # Perform full analysis
        results = analyze_video_with_checkpoints(
            filename="test_video.mp4",
            file_size=file_size,
            analysis_type="full",
            video_path=test_video_path
        )
        
        logger.info(f"📊 Analysis results: {results}")
        
        # Verify results structure
        required_fields = [
            "ball_detections", "trajectory_points", "analysis_duration",
            "video_duration_seconds", "confidence", "detection_rate",
            "checkpoint_analysis", "frame_analyses", "multimodal_insights",
            "technical_stats"
        ]
        
        for field in required_fields:
            assert field in results, f"Missing field in results: {field}"
        
        # Verify checkpoint analysis details
        checkpoint_data = results["checkpoint_analysis"]
        assert checkpoint_data["total_checkpoints"] >= 7, f"Expected ~8 checkpoints, got {checkpoint_data['total_checkpoints']}"
        assert checkpoint_data["checkpoint_interval"] == 1.0, f"Unexpected interval: {checkpoint_data['checkpoint_interval']}"
        
        # Verify technical stats indicate checkpoint method
        assert results["technical_stats"]["analysis_method"] == "checkpoint_based_frame_analysis"
        
        logger.info("✅ Full checkpoint analysis test passed")
        
        # Cleanup
        os.unlink(test_video_path)
        
    except Exception as e:
        logger.error(f"❌ Full checkpoint analysis test failed: {e}")
        raise


def main():
    """Run all checkpoint analysis tests"""
    logger.info("🚀 Starting checkpoint analysis tests...")
    
    try:
        test_video_metadata_extraction()
        test_checkpoint_frame_extraction()
        test_frame_analysis()
        test_full_checkpoint_analysis()
        
        logger.info("\n🎉 All checkpoint analysis tests passed!")
        
    except Exception as e:
        logger.error(f"\n💥 Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 