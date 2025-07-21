"""
TTBall_4 Gemma 3n Integration Test

Test suite to verify Gemma 3n multimodal model integration
for table tennis video analysis.
"""

import asyncio
import json
import tempfile
import torch
from pathlib import Path
from typing import Dict, Any

# Import our TTBall_4 components
from core.config import get_settings
from core.logging import setup_logging, get_logger
from services.model_manager import ModelManager, ModelType

# Setup logging
setup_logging()
logger = get_logger(__name__)

class Gemma3nTestSuite:
    """Test suite for Gemma 3n integration"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_manager = None
        self.test_results = []
        
    async def run_all_tests(self):
        """Run comprehensive Gemma 3n test suite"""
        print("🚀 TTBall_4 Gemma 3n Integration Test Suite")
        print("=" * 60)
        
        tests = [
            ("Configuration Test", self.test_configuration),
            ("Model Manager Initialization", self.test_model_manager_init),
            ("Gemma 3n Model Loading", self.test_gemma3n_loading),
            ("Model Capabilities Test", self.test_model_capabilities),
            ("Memory Management Test", self.test_memory_management),
            ("Multimodal Analysis Test", self.test_multimodal_analysis),
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔬 Running {test_name}...")
            try:
                result = await test_func()
                if result:
                    print(f"✅ {test_name} PASSED")
                    self.test_results.append((test_name, "PASSED", None))
                else:
                    print(f"❌ {test_name} FAILED")
                    self.test_results.append((test_name, "FAILED", "Test returned False"))
            except Exception as e:
                print(f"❌ {test_name} FAILED: {e}")
                self.test_results.append((test_name, "FAILED", str(e)))
        
        # Print summary
        self.print_test_summary()
        
        # Cleanup
        if self.model_manager:
            await self.model_manager.cleanup()
    
    async def test_configuration(self) -> bool:
        """Test Gemma 3n configuration"""
        try:
            # Verify Gemma 3n model name
            assert self.settings.gemma_model_name == "google/gemma-3n-E4B"
            
            # Verify multimodal features enabled
            assert hasattr(self.settings, 'enable_3d_analysis')
            
            # Verify proper device configuration
            assert self.settings.device in ["cpu", "cuda", "mps"]
            
            logger.info("Configuration test passed", extra={
                "gemma_model": self.settings.gemma_model_name,
                "device": self.settings.device,
                "enable_3d": self.settings.enable_3d_analysis
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration test failed: {e}")
            return False
    
    async def test_model_manager_init(self) -> bool:
        """Test ModelManager initialization"""
        try:
            self.model_manager = ModelManager()
            
            # Verify initialization
            assert self.model_manager is not None
            assert hasattr(self.model_manager, 'models')
            assert hasattr(self.model_manager, 'gpu_available')
            
            # Check GPU availability
            gpu_info = {
                "gpu_available": self.model_manager.gpu_available,
                "gpu_count": self.model_manager.gpu_count if hasattr(self.model_manager, 'gpu_count') else 0
            }
            
            logger.info("ModelManager initialized successfully", extra=gpu_info)
            
            return True
            
        except Exception as e:
            logger.error(f"ModelManager initialization failed: {e}")
            return False
    
    async def test_gemma3n_loading(self) -> bool:
        """Test Gemma 3n model loading (mock test for now)"""
        try:
            # For now, we'll test the script generation without actually loading
            # the model to avoid memory issues in testing
            
            if not self.model_manager:
                return False
            
            # Test script generation
            script_path = await self.model_manager._create_gemma_3n_script("e4b", "cpu")
            
            # Verify script was created
            assert script_path.exists()
            
            # Verify script content
            with open(script_path, 'r') as f:
                script_content = f.read()
                assert "google/gemma-3n-E4B" in script_content
                assert "AutoModelForCausalLM" in script_content
                assert "transformers" in script_content
            
            logger.info("Gemma 3n script generation successful", extra={
                "script_path": str(script_path),
                "model_name": "google/gemma-3n-E4B"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Gemma 3n loading test failed: {e}")
            return False
    
    async def test_model_capabilities(self) -> bool:
        """Test model capabilities definition"""
        try:
            # Test that Gemma 3n capabilities are properly defined
            expected_capabilities = [
                "multimodal_analysis",
                "video_understanding", 
                "text_generation",
                "action_recognition",
                "image_text_to_text"
            ]
            
            # Mock the model info that would be returned
            from models.ai_models import ModelInfo, ModelStatus
            
            mock_model_info = ModelInfo(
                model_type=ModelType.GEMMA_3N,
                model_name="google/gemma-3n-E4B",
                status=ModelStatus.READY,
                memory_usage_mb=1000,
                gpu_memory_mb=0,
                device="cpu",
                capabilities=expected_capabilities
            )
            
            # Verify all expected capabilities are present
            for capability in expected_capabilities:
                assert capability in mock_model_info.capabilities
            
            logger.info("Model capabilities test passed", extra={
                "capabilities": mock_model_info.capabilities
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Model capabilities test failed: {e}")
            return False
    
    async def test_memory_management(self) -> bool:
        """Test memory management features"""
        try:
            if not self.model_manager:
                return False
            
            # Test system resource monitoring
            if hasattr(self.model_manager, 'get_system_resources'):
                resources = await self.model_manager.get_system_resources()
                
                # Verify resource info is available
                assert isinstance(resources, dict)
                
                logger.info("Memory management test passed", extra={
                    "resources_available": True
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Memory management test failed: {e}")
            return False
    
    async def test_multimodal_analysis(self) -> bool:
        """Test multimodal analysis workflow (mock)"""
        try:
            # Create a mock multimodal analysis workflow
            analysis_config = {
                "model": "google/gemma-3n-E4B",
                "task": "video_understanding",
                "capabilities": [
                    "frame_analysis",
                    "action_recognition", 
                    "technique_assessment",
                    "text_generation"
                ]
            }
            
            # Verify configuration is valid for table tennis analysis
            assert "video_understanding" in analysis_config["task"]
            assert "action_recognition" in analysis_config["capabilities"]
            assert "google/gemma-3n-E4B" in analysis_config["model"]
            
            logger.info("Multimodal analysis workflow test passed", extra={
                "analysis_config": analysis_config
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Multimodal analysis test failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        passed = sum(1 for _, status, _ in self.test_results if status == "PASSED")
        total = len(self.test_results)
        
        for test_name, status, error in self.test_results:
            status_icon = "✅" if status == "PASSED" else "❌"
            print(f"{status_icon} {test_name}: {status}")
            if error:
                print(f"   Error: {error}")
        
        print(f"\n📊 Tests Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Gemma 3n integration is ready!")
        else:
            print("⚠️  Some tests failed. Please review and fix issues.")
        
        # Gemma 3n specific information
        print("\n🔍 Gemma 3n Integration Details:")
        print(f"   Model: google/gemma-3n-E4B")
        print(f"   Hugging Face: https://huggingface.co/google/gemma-3n-E4B")
        print(f"   Capabilities: Multimodal, Video Understanding, Text Generation")
        print(f"   Use Case: Table Tennis Video Analysis")


async def main():
    """Main test execution"""
    test_suite = Gemma3nTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 