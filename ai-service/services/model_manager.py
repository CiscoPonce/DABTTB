"""
TTBall_4 Model Manager Service
Implements VisionCraft 2024-2025 best practices for multimodal AI model management
"""

import asyncio
import json
import os
import psutil
import tempfile
import torch
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from core.config import get_settings
from core.logging import get_logger, LoggerMixin
from models.ai_models import ModelStatus, ModelInfo

logger = get_logger(__name__)

class ModelType(str, Enum):
    """AI model types supported by TTBall_4"""
    YOLO11 = "yolo11"
    GEMMA_3N = "gemma_3n"
    TRAJECTORY = "trajectory"

@dataclass
class ModelProcess:
    """Represents a model running in an isolated subprocess"""
    process: asyncio.subprocess.Process
    model_type: ModelType
    pid: int
    memory_usage: float
    gpu_memory: float
    status: ModelStatus
    startup_time: float

class ModelManager(LoggerMixin):
    """
    Advanced model manager implementing VisionCraft best practices:
    - Subprocess isolation for VRAM efficiency
    - Dynamic memory allocation
    - GPU/CPU fallback handling
    - Real-time monitoring
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.models: Dict[ModelType, ModelProcess] = {}
        self.model_scripts_dir = Path(__file__).parent / "model_scripts"
        self.temp_dir = Path(tempfile.gettempdir()) / "ttball4_models"
        self.temp_dir.mkdir(exist_ok=True)
        
        # VisionCraft: Track system resources
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.cpu_count = psutil.cpu_count()
        
        self.log_info("ModelManager initialized", extra={
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "cpu_count": self.cpu_count
        })

    async def load_yolo11(self, model_size: str = "n") -> ModelInfo:
        """
        Load YOLO11 model in isolated subprocess (VisionCraft best practice)
        
        Args:
            model_size: Model size variant (n, s, m, l, x)
            
        Returns:
            ModelInfo with loaded model details
        """
        try:
            self.log_info("Loading YOLO11 model", extra={"model_size": model_size})
            
            # Create model script for subprocess isolation
            script_path = await self._create_yolo11_script(model_size)
            
            # VisionCraft: Use asyncio.create_subprocess_exec for isolation
            process = await asyncio.create_subprocess_exec(
                "python", str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.model_scripts_dir)
            )
            
            # Monitor memory usage
            memory_info = await self._get_process_memory(process.pid)
            
            model_process = ModelProcess(
                process=process,
                model_type=ModelType.YOLO11,
                pid=process.pid,
                memory_usage=memory_info.get("memory_mb", 0),
                gpu_memory=memory_info.get("gpu_memory_mb", 0),
                status=ModelStatus.LOADING,
                startup_time=asyncio.get_event_loop().time()
            )
            
            self.models[ModelType.YOLO11] = model_process
            
            # Wait for model to be ready
            await self._wait_for_model_ready(ModelType.YOLO11)
            
            model_info = ModelInfo(
                model_type=ModelType.YOLO11,
                model_name=f"yolo11{model_size}",
                status=ModelStatus.READY,
                memory_usage_mb=model_process.memory_usage,
                gpu_memory_mb=model_process.gpu_memory,
                device="cuda" if self.gpu_available else "cpu",
                capabilities=["object_detection", "tracking", "segmentation"]
            )
            
            self.log_info("YOLO11 model loaded successfully", extra={
                "model_size": model_size,
                "memory_mb": model_process.memory_usage,
                "gpu_memory_mb": model_process.gpu_memory,
                "pid": process.pid
            })
            
            return model_info
            
        except Exception as e:
            self.log_error("Failed to load YOLO11 model", extra={
                "error": str(e),
                "model_size": model_size
            })
            raise

    async def load_gemma_3n(self, variant: str = "e4b") -> ModelInfo:
        """
        Load Gemma 3n model with VisionCraft optimizations
        
        Args:
            variant: Model variant (e2b, e4b)
            
        Returns:
            ModelInfo with loaded model details
        """
        try:
            self.log_info("Loading Gemma 3n model", extra={"variant": variant})
            
            # Check memory requirements (VisionCraft: 3GB for E4B)
            required_memory = 3000 if variant == "e4b" else 2000  # MB
            if self.gpu_available:
                available_memory = await self._get_gpu_memory()
                if available_memory < required_memory:
                    self.log_warning("Insufficient GPU memory, falling back to CPU", extra={
                        "required_mb": required_memory,
                        "available_mb": available_memory
                    })
                    device = "cpu"
                else:
                    device = "cuda"
            else:
                device = "cpu"
            
            # Create Gemma 3n script for subprocess isolation
            script_path = await self._create_gemma_3n_script(variant, device)
            
            # VisionCraft: Subprocess execution for memory efficiency
            process = await asyncio.create_subprocess_exec(
                "python", str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.model_scripts_dir)
            )
            
            memory_info = await self._get_process_memory(process.pid)
            
            model_process = ModelProcess(
                process=process,
                model_type=ModelType.GEMMA_3N,
                pid=process.pid,
                memory_usage=memory_info.get("memory_mb", 0),
                gpu_memory=memory_info.get("gpu_memory_mb", 0),
                status=ModelStatus.LOADING,
                startup_time=asyncio.get_event_loop().time()
            )
            
            self.models[ModelType.GEMMA_3N] = model_process
            
            await self._wait_for_model_ready(ModelType.GEMMA_3N)
            
            model_info = ModelInfo(
                model_type=ModelType.GEMMA_3N,
                model_name=f"google/gemma-3n-{variant.upper()}",
                status=ModelStatus.READY,
                memory_usage_mb=model_process.memory_usage,
                gpu_memory_mb=model_process.gpu_memory,
                device=device,
                capabilities=["multimodal_analysis", "video_understanding", "text_generation", "action_recognition", "image_text_to_text"]
            )
            
            self.log_info("Gemma 3n model loaded successfully", extra={
                "variant": variant,
                "device": device,
                "memory_mb": model_process.memory_usage,
                "gpu_memory_mb": model_process.gpu_memory,
                "pid": process.pid
            })
            
            return model_info
            
        except Exception as e:
            self.log_error("Failed to load Gemma 3n model", extra={
                "error": str(e),
                "variant": variant
            })
            raise

    async def get_model_status(self, model_type: ModelType) -> ModelStatus:
        """Get current status of a model"""
        if model_type not in self.models:
            return ModelStatus.NOT_LOADED
            
        model_process = self.models[model_type]
        
        # Check if process is still alive
        if model_process.process.returncode is not None:
            return ModelStatus.ERROR
            
        return model_process.status

    async def unload_model(self, model_type: ModelType) -> bool:
        """
        Safely unload a model and free resources
        
        Args:
            model_type: Type of model to unload
            
        Returns:
            True if successfully unloaded
        """
        try:
            if model_type not in self.models:
                return True
                
            model_process = self.models[model_type]
            
            # Graceful shutdown
            model_process.process.terminate()
            
            try:
                await asyncio.wait_for(model_process.process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                # Force kill if graceful shutdown fails
                model_process.process.kill()
                await model_process.process.wait()
            
            del self.models[model_type]
            
            self.log_info("Model unloaded successfully", extra={
                "model_type": model_type,
                "pid": model_process.pid
            })
            
            return True
            
        except Exception as e:
            self.log_error("Failed to unload model", extra={
                "error": str(e),
                "model_type": model_type
            })
            return False

    async def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # GPU usage
            gpu_info = {}
            if self.gpu_available:
                gpu_info = await self._get_gpu_info()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
                "gpu_info": gpu_info,
                "models_loaded": list(self.models.keys()),
                "model_count": len(self.models)
            }
            
        except Exception as e:
            self.log_error("Failed to get system resources", extra={"error": str(e)})
            return {}

    async def _create_yolo11_script(self, model_size: str) -> Path:
        """Create YOLO11 model script for subprocess execution"""
        script_content = f'''
import sys
import torch
from ultralytics import YOLO
import json
import time

def main():
    try:
        # Load YOLO11 model
        model = YOLO("yolo11{model_size}.pt")
        
        # Warm up model
        if torch.cuda.is_available():
            model.to("cuda")
            # Warm up with dummy data
            import numpy as np
            dummy_input = np.random.rand(640, 640, 3).astype(np.uint8)
            _ = model(dummy_input, verbose=False)
        
        print("MODEL_READY")
        
        # Keep process alive for inference
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"MODEL_ERROR: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        script_path = self.model_scripts_dir / f"yolo11_{model_size}_script.py"
        self.model_scripts_dir.mkdir(exist_ok=True)
        
        with open(script_path, "w") as f:
            f.write(script_content)
            
        return script_path

    async def _create_gemma_3n_script(self, variant: str, device: str) -> Path:
        """Create Gemma 3n model script for subprocess execution"""
        script_content = f'''
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time

def main():
    try:
        model_name = "google/gemma-3n-{variant.upper()}"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None
        )
        
        if device == "cpu":
            model = model.to("cpu")
        
        print("MODEL_READY")
        
        # Keep process alive for inference
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"MODEL_ERROR: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        script_path = self.model_scripts_dir / f"gemma_3n_{variant}_script.py"
        self.model_scripts_dir.mkdir(exist_ok=True)
        
        with open(script_path, "w") as f:
            f.write(script_content)
            
        return script_path

    async def _wait_for_model_ready(self, model_type: ModelType, timeout: float = 300.0):
        """Wait for model to be ready with timeout"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if model_type not in self.models:
                raise RuntimeError(f"Model {model_type} not found")
                
            process = self.models[model_type].process
            
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                if line:
                    message = line.decode().strip()
                    if "MODEL_READY" in message:
                        self.models[model_type].status = ModelStatus.READY
                        return
                    elif "MODEL_ERROR" in message:
                        error = message.split(":", 1)[1] if ":" in message else "Unknown error"
                        raise RuntimeError(f"Model loading failed: {error}")
                        
            except asyncio.TimeoutError:
                continue
                
            await asyncio.sleep(0.1)
            
        raise TimeoutError(f"Model {model_type} failed to load within {timeout} seconds")

    async def _get_process_memory(self, pid: int) -> Dict[str, float]:
        """Get memory usage for a process"""
        try:
            process = psutil.Process(pid)
            memory_info = process.memory_info()
            
            result = {
                "memory_mb": memory_info.rss / (1024 * 1024)
            }
            
            # GPU memory if available
            if self.gpu_available:
                gpu_memory = await self._get_process_gpu_memory(pid)
                result["gpu_memory_mb"] = gpu_memory
                
            return result
            
        except Exception:
            return {"memory_mb": 0, "gpu_memory_mb": 0}

    async def _get_gpu_memory(self) -> float:
        """Get available GPU memory in MB"""
        if not self.gpu_available:
            return 0.0
            
        try:
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return info.free / (1024 * 1024)  # Convert to MB
            
        except Exception:
            # Fallback using torch
            torch.cuda.empty_cache()
            return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    async def _get_process_gpu_memory(self, pid: int) -> float:
        """Get GPU memory usage for a specific process"""
        # Simplified version - in production would use nvidia-ml-py
        return 0.0

    async def _get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information"""
        if not self.gpu_available:
            return {}
            
        try:
            gpu_info = {}
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                memory_cached = torch.cuda.memory_reserved(i) / (1024 * 1024)
                
                gpu_info[f"gpu_{i}"] = {
                    "name": props.name,
                    "total_memory_mb": props.total_memory / (1024 * 1024),
                    "allocated_memory_mb": memory_allocated,
                    "cached_memory_mb": memory_cached,
                    "compute_capability": f"{props.major}.{props.minor}"
                }
                
            return gpu_info
            
        except Exception as e:
            self.log_error("Failed to get GPU info", extra={"error": str(e)})
            return {}

    async def cleanup(self):
        """Cleanup all models and resources"""
        try:
            for model_type in list(self.models.keys()):
                await self.unload_model(model_type)
                
            self.log_info("ModelManager cleanup completed")
            
        except Exception as e:
            self.log_error("Failed to cleanup models", extra={"error": str(e)}) 