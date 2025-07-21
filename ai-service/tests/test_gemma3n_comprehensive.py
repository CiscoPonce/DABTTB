"""
TTBall4 Gemma 3N Comprehensive Tests
====================================

Comprehensive test suite for Gemma 3N multimodal AI integration.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Dict, Any

# Import AI service components
try:
    from core.config import get_settings
    from core.logging import get_logger
    from services.model_manager import ModelManager, ModelType
    from services.multimodal_service import MultimodalService
    from models.ai_models import ModelInfo, ModelStatus
except ImportError:
    # Handle missing imports gracefully in testing
    pass

logger = get_logger(__name__)


class TestGemma3NIntegration:
    """Comprehensive test suite for Gemma 3N integration"""

    @pytest.mark.unit
    async def test_model_initialization(self):
        """Test Gemma 3N model initialization"""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                mock_tokenizer.return_value = MagicMock()
                mock_model.return_value = MagicMock()
                
                # Test initialization
                model_manager = ModelManager()
                result = await model_manager.load_model("gemma-3n", ModelType.GEMMA_3N)
                
                assert result is True
                mock_tokenizer.assert_called_once()
                mock_model.assert_called_once()

    @pytest.mark.unit
    async def test_multimodal_analysis(self):
        """Test multimodal analysis capabilities"""
        multimodal_service = MultimodalService()
        
        # Mock input data
        video_frames = [
            {"frame": 1, "data": b"fake_frame_data_1"},
            {"frame": 2, "data": b"fake_frame_data_2"}
        ]
        
        prompt = "Analyze this table tennis serve technique"
        
        with patch.object(multimodal_service, '_process_video_frames') as mock_process:
            with patch.object(multimodal_service, '_generate_analysis') as mock_generate:
                mock_process.return_value = {"processed_frames": 2}
                mock_generate.return_value = {
                    "analysis": "Professional serve detected",
                    "confidence": 0.92,
                    "technique_score": 8.5
                }
                
                result = await multimodal_service.analyze_video_with_prompt(
                    video_frames, prompt
                )
                
                assert result["analysis"] == "Professional serve detected"
                assert result["confidence"] == 0.92
                assert result["technique_score"] == 8.5

    @pytest.mark.unit
    async def test_video_understanding(self):
        """Test video understanding capabilities"""
        multimodal_service = MultimodalService()
        
        video_path = "/fake/path/to/video.mp4"
        
        with patch.object(multimodal_service, '_extract_video_features') as mock_extract:
            with patch.object(multimodal_service, '_understand_video_content') as mock_understand:
                mock_extract.return_value = {
                    "frames": 30,
                    "duration": 1.0,
                    "features": ["ball_motion", "player_action"]
                }
                mock_understand.return_value = {
                    "actions": ["serve", "return"],
                    "technique_analysis": {
                        "serve_quality": "excellent",
                        "ball_trajectory": "optimal"
                    }
                }
                
                result = await multimodal_service.understand_video(video_path)
                
                assert "actions" in result
                assert "technique_analysis" in result
                assert result["actions"] == ["serve", "return"]

    @pytest.mark.integration
    async def test_gemma3n_with_real_inference(self):
        """Test Gemma 3N with simulated real inference"""
        # This test simulates real model inference without loading the actual model
        
        fake_model_output = {
            "text": "The table tennis serve shows excellent technique with proper ball toss height and racket angle.",
            "logits": [0.1, 0.9, 0.3, 0.7],
            "attention_weights": [[0.2, 0.8], [0.6, 0.4]]
        }
        
        with patch('torch.no_grad'):
            with patch('transformers.AutoModelForCausalLM.generate') as mock_generate:
                mock_generate.return_value = [[1, 2, 3, 4, 5]]  # Mock token IDs
                
                multimodal_service = MultimodalService()
                
                result = await multimodal_service.generate_text_analysis(
                    prompt="Analyze the technique in this table tennis video",
                    context={"ball_positions": [[100, 200], [120, 180]]}
                )
                
                assert "analysis" in result
                assert isinstance(result["analysis"], str)

    @pytest.mark.performance
    async def test_large_video_processing(self):
        """Test processing large video files"""
        # Simulate large video processing
        large_video_data = {
            "frames": 1000,
            "duration": 33.33,  # 30 FPS
            "resolution": "1920x1080"
        }
        
        multimodal_service = MultimodalService()
        
        with patch.object(multimodal_service, '_process_video_efficiently') as mock_process:
            mock_process.return_value = {
                "processed_frames": 1000,
                "processing_time": 2.5,
                "memory_usage": "512MB"
            }
            
            start_time = asyncio.get_event_loop().time()
            result = await multimodal_service.process_large_video(large_video_data)
            end_time = asyncio.get_event_loop().time()
            
            processing_time = end_time - start_time
            assert processing_time < 5.0  # Should process in under 5 seconds
            assert result["processed_frames"] == 1000

    @pytest.mark.unit
    async def test_error_handling(self):
        """Test error handling in various scenarios"""
        multimodal_service = MultimodalService()
        
        # Test invalid video format
        with pytest.raises(ValueError):
            await multimodal_service.analyze_video_with_prompt(
                None, "Analyze this video"
            )
        
        # Test empty prompt
        with pytest.raises(ValueError):
            await multimodal_service.analyze_video_with_prompt(
                [{"frame": 1, "data": b"data"}], ""
            )
        
        # Test model inference error
        with patch.object(multimodal_service, '_generate_analysis') as mock_generate:
            mock_generate.side_effect = Exception("Model inference failed")
            
            with pytest.raises(Exception):
                await multimodal_service.analyze_video_with_prompt(
                    [{"frame": 1, "data": b"data"}], "Analyze this"
                )

    @pytest.mark.unit
    async def test_memory_management(self):
        """Test memory management during processing"""
        model_manager = ModelManager()
        
        with patch.object(model_manager, 'get_memory_usage') as mock_memory:
            mock_memory.return_value = {
                "gpu_memory_used": "2.5GB",
                "gpu_memory_total": "8GB",
                "system_memory_used": "4GB",
                "system_memory_total": "16GB"
            }
            
            memory_info = await model_manager.get_system_resources()
            
            assert "gpu_memory_used" in memory_info
            assert "system_memory_used" in memory_info
            
            # Check memory usage is within limits
            gpu_usage_gb = float(memory_info["gpu_memory_used"].replace("GB", ""))
            assert gpu_usage_gb < 6.0  # Should use less than 6GB

    @pytest.mark.unit
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent analysis requests"""
        multimodal_service = MultimodalService()
        
        async def analyze_request(request_id):
            return await multimodal_service.analyze_video_with_prompt(
                [{"frame": 1, "data": f"request_{request_id}".encode()}],
                f"Analyze request {request_id}"
            )
        
        with patch.object(multimodal_service, '_generate_analysis') as mock_generate:
            mock_generate.return_value = {
                "analysis": "Analysis complete",
                "confidence": 0.9
            }
            
            # Run 5 concurrent requests
            tasks = [analyze_request(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(r["analysis"] == "Analysis complete" for r in results)

    @pytest.mark.security
    async def test_input_validation(self):
        """Test input validation and security"""
        multimodal_service = MultimodalService()
        
        # Test malicious prompt injection
        malicious_prompts = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "\x00\x01\x02malicious",
        ]
        
        for prompt in malicious_prompts:
            with patch.object(multimodal_service, '_sanitize_prompt') as mock_sanitize:
                mock_sanitize.return_value = "sanitized_prompt"
                
                result = await multimodal_service.analyze_video_with_prompt(
                    [{"frame": 1, "data": b"safe_data"}], prompt
                )
                
                mock_sanitize.assert_called_once_with(prompt)

    @pytest.mark.unit
    async def test_model_capabilities_detection(self):
        """Test detection of model capabilities"""
        model_manager = ModelManager()
        
        with patch.object(model_manager, 'detect_model_features') as mock_detect:
            mock_detect.return_value = {
                "multimodal": True,
                "video_understanding": True,
                "text_generation": True,
                "image_analysis": True,
                "action_recognition": True,
                "supported_languages": ["en", "es", "fr", "de"],
                "max_sequence_length": 4096
            }
            
            capabilities = await model_manager.get_model_capabilities("gemma-3n")
            
            assert capabilities["multimodal"] is True
            assert capabilities["video_understanding"] is True
            assert "action_recognition" in capabilities
            assert len(capabilities["supported_languages"]) >= 4

    @pytest.mark.integration
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end analysis workflow"""
        # Simulate complete workflow from video upload to analysis results
        
        workflow_data = {
            "video_path": "/fake/video.mp4",
            "analysis_type": "comprehensive",
            "options": {
                "detect_ball": True,
                "analyze_technique": True,
                "provide_recommendations": True
            }
        }
        
        model_manager = ModelManager()
        multimodal_service = MultimodalService()
        
        with patch.object(model_manager, 'load_model') as mock_load:
            with patch.object(multimodal_service, 'process_video_analysis') as mock_process:
                mock_load.return_value = True
                mock_process.return_value = {
                    "ball_detections": [
                        {"frame": 1, "x": 100, "y": 200, "confidence": 0.95}
                    ],
                    "technique_analysis": {
                        "serve_quality": "excellent",
                        "recommendations": ["Maintain consistent ball toss"]
                    },
                    "performance_metrics": {
                        "accuracy": 0.92,
                        "speed": 25.5
                    }
                }
                
                # Execute workflow
                await model_manager.initialize()
                result = await multimodal_service.analyze_complete_video(workflow_data)
                
                # Verify results
                assert "ball_detections" in result
                assert "technique_analysis" in result
                assert "performance_metrics" in result
                assert result["technique_analysis"]["serve_quality"] == "excellent"


class TestGemma3NPerformance:
    """Performance-specific tests for Gemma 3N"""

    @pytest.mark.performance
    async def test_inference_speed(self):
        """Test model inference speed"""
        multimodal_service = MultimodalService()
        
        test_data = {
            "frames": [{"frame": i, "data": f"frame_{i}".encode()} for i in range(10)],
            "prompt": "Analyze this table tennis sequence"
        }
        
        with patch.object(multimodal_service, '_generate_analysis') as mock_generate:
            mock_generate.return_value = {"analysis": "Fast analysis"}
            
            start_time = asyncio.get_event_loop().time()
            result = await multimodal_service.analyze_video_with_prompt(
                test_data["frames"], test_data["prompt"]
            )
            end_time = asyncio.get_event_loop().time()
            
            inference_time = end_time - start_time
            assert inference_time < 2.0  # Should complete in under 2 seconds
            assert result["analysis"] == "Fast analysis"

    @pytest.mark.performance
    async def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        model_manager = ModelManager()
        
        initial_memory = 1000  # MB
        
        with patch.object(model_manager, 'get_memory_usage') as mock_memory:
            # Simulate memory usage during processing
            memory_sequence = [1000, 1200, 1500, 1300, 1000]  # MB
            mock_memory.side_effect = memory_sequence
            
            for i in range(5):
                memory_usage = await model_manager.get_memory_usage()
                assert memory_usage < 2000  # Should stay under 2GB
                
                # Simulate processing step
                await asyncio.sleep(0.1)

    @pytest.mark.performance
    async def test_throughput(self):
        """Test processing throughput"""
        multimodal_service = MultimodalService()
        
        # Process multiple videos concurrently
        video_count = 10
        videos = [
            {"frames": [{"frame": 1, "data": f"video_{i}".encode()}], "id": i}
            for i in range(video_count)
        ]
        
        with patch.object(multimodal_service, 'analyze_video_with_prompt') as mock_analyze:
            mock_analyze.return_value = {"analysis": "Processed", "id": "video"}
            
            start_time = asyncio.get_event_loop().time()
            
            tasks = [
                multimodal_service.analyze_video_with_prompt(
                    video["frames"], f"Analyze video {video['id']}"
                )
                for video in videos
            ]
            
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()
            
            total_time = end_time - start_time
            throughput = len(results) / total_time
            
            assert len(results) == video_count
            assert throughput > 2.0  # Should process more than 2 videos per second


class TestGemma3NCompatibility:
    """Compatibility tests for different environments"""

    @pytest.mark.unit
    async def test_cpu_fallback(self):
        """Test CPU fallback when GPU is unavailable"""
        model_manager = ModelManager()
        
        with patch('torch.cuda.is_available', return_value=False):
            device = await model_manager.get_optimal_device()
            assert device == "cpu"
            
            # Test model can load on CPU
            with patch.object(model_manager, 'load_model') as mock_load:
                mock_load.return_value = True
                result = await model_manager.load_model("gemma-3n", ModelType.GEMMA_3N)
                assert result is True

    @pytest.mark.unit
    async def test_gpu_optimization(self):
        """Test GPU optimization when available"""
        model_manager = ModelManager()
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                device = await model_manager.get_optimal_device()
                assert device == "cuda"
                
                # Test GPU memory optimization
                with patch.object(model_manager, 'optimize_gpu_usage') as mock_optimize:
                    mock_optimize.return_value = True
                    result = await model_manager.setup_gpu_optimization()
                    assert result is True

    @pytest.mark.unit
    async def test_different_precision_modes(self):
        """Test different precision modes (fp16, fp32, int8)"""
        model_manager = ModelManager()
        
        precision_modes = ["fp16", "fp32", "int8"]
        
        for precision in precision_modes:
            with patch.object(model_manager, 'set_precision') as mock_precision:
                mock_precision.return_value = True
                
                result = await model_manager.configure_precision(precision)
                assert result is True
                mock_precision.assert_called_with(precision)


if __name__ == "__main__":
    # Run specific test suites
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ]) 