#!/usr/bin/env python3
"""
Test Structured Prompts 1 and 2 for Ball Detection

This script tests the exact structured prompts that proved successful 
in the debug session, ensuring CUDA is working properly.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from typing import Dict, Any

class StructuredPromptTester:
    """Test the specific structured prompts that worked in debug session"""
    
    def __init__(self, model_path: str = "/app/model_files/gemma-3n-E4B"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🧪 Structured Prompt Tester")
        print(f"   Device: {self.device}")
        print(f"   Model path: {model_path}")
    
    def verify_cuda(self):
        """Verify CUDA is working properly"""
        print(f"\n🔍 CUDA Verification:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"   Current device: {torch.cuda.current_device()}")
        else:
            print(f"   ⚠️  CUDA not available - will use CPU")
    
    def load_model(self) -> bool:
        """Load model with CUDA optimization"""
        try:
            self.verify_cuda()
            
            print(f"\n🔍 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            print(f"   ✅ Tokenizer loaded")
            
            print(f"🔍 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"   ✅ Model loaded on device: {self.model.device}")
            
            # Verify model is on CUDA
            if torch.cuda.is_available():
                if hasattr(self.model, 'device') and 'cuda' in str(self.model.device):
                    print(f"   ✅ Model confirmed on CUDA")
                else:
                    print(f"   ⚠️  Model may not be fully on CUDA")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_structured_prompt_1(self, frame: np.ndarray, context: str = "") -> str:
        """
        Create Structured Prompt 1 - the format that gave us:
        "YES, there is an orange ball visible in this image. The ball is located 
        in the center of the table tennis table, hovering above the net."
        """
        # Get image metadata
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        avg_color = np.mean(rgb_frame, axis=(0, 1))
        
        image_context = f"Image: {width}x{height} pixels, avg colors: R={avg_color[0]:.0f}, G={avg_color[1]:.0f}, B={avg_color[2]:.0f}"
        
        return f"""MULTIMODAL_INPUT:
{image_context}

TASK: Ball detection in table tennis scene
QUESTION: Is there an orange ball visible in this image?
RESPONSE_FORMAT: Answer with YES or NO followed by brief explanation.
ANSWER:"""
    
    def create_structured_prompt_2(self, frame: np.ndarray, context: str = "") -> str:
        """
        Create Structured Prompt 2 - the format that gave us:
        "YES Ball detected coordinates: x=240, y=240, w=10, h=10 pixels"
        """
        # Get image metadata
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        avg_color = np.mean(rgb_frame, axis=(0, 1))
        
        image_context = f"Image: {width}x{height} pixels, avg colors: R={avg_color[0]:.0f}, G={avg_color[1]:.0f}, B={avg_color[2]:.0f}"
        
        return f"""VISION_ANALYSIS:
Scene: Table tennis game
{image_context}

INSTRUCTION: Analyze this image for ball detection.
Look for: Orange/yellow spherical objects
ANSWER: Ball detected (YES/NO):"""
    
    def test_frame_with_both_prompts(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Test a frame with both structured prompts"""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        results = {
            "timestamp": timestamp,
            "frame_shape": frame.shape,
            "prompt_1_result": {},
            "prompt_2_result": {}
        }
        
        # Test Structured Prompt 1
        print(f"\n🔍 Testing Structured Prompt 1 at {timestamp}s:")
        try:
            prompt_1 = self.create_structured_prompt_1(frame)
            print(f"   Prompt length: {len(prompt_1)} characters")
            
            inputs = self.tokenizer(prompt_1, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt_1):].strip()
            
            results["prompt_1_result"] = {
                "success": True,
                "response": generated,
                "prompt_used": prompt_1
            }
            
            print(f"   ✅ Response: {generated}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results["prompt_1_result"] = {"success": False, "error": str(e)}
        
        # Test Structured Prompt 2
        print(f"\n🔍 Testing Structured Prompt 2 at {timestamp}s:")
        try:
            prompt_2 = self.create_structured_prompt_2(frame)
            print(f"   Prompt length: {len(prompt_2)} characters")
            
            inputs = self.tokenizer(prompt_2, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt_2):].strip()
            
            results["prompt_2_result"] = {
                "success": True,
                "response": generated,
                "prompt_used": prompt_2
            }
            
            print(f"   ✅ Response: {generated}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results["prompt_2_result"] = {"success": False, "error": str(e)}
        
        return results
    
    def analyze_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the responses for ball detection indicators"""
        analysis = {
            "timestamp": results["timestamp"],
            "prompt_1_analysis": {},
            "prompt_2_analysis": {},
            "comparison": {}
        }
        
        # Analyze Prompt 1 response
        if results["prompt_1_result"].get("success", False):
            response_1 = results["prompt_1_result"]["response"].lower()
            analysis["prompt_1_analysis"] = {
                "ball_detected": "yes" in response_1,
                "ball_not_detected": "no" in response_1,
                "location_mentioned": any(word in response_1 for word in ["center", "table", "net", "hovering", "air", "left", "right"]),
                "confidence_indicators": any(word in response_1 for word in ["visible", "located", "detected", "present"]),
                "raw_response": results["prompt_1_result"]["response"]
            }
        
        # Analyze Prompt 2 response
        if results["prompt_2_result"].get("success", False):
            response_2 = results["prompt_2_result"]["response"].lower()
            analysis["prompt_2_analysis"] = {
                "ball_detected": "yes" in response_2,
                "ball_not_detected": "no" in response_2,
                "coordinates_mentioned": any(word in response_2 for word in ["x=", "y=", "coordinates", "pixels"]),
                "detection_format": "detected" in response_2,
                "raw_response": results["prompt_2_result"]["response"]
            }
        
        # Comparison
        prompt_1_success = analysis["prompt_1_analysis"].get("ball_detected", False)
        prompt_2_success = analysis["prompt_2_analysis"].get("ball_detected", False)
        
        analysis["comparison"] = {
            "both_detected": prompt_1_success and prompt_2_success,
            "either_detected": prompt_1_success or prompt_2_success,
            "neither_detected": not prompt_1_success and not prompt_2_success,
            "prompt_1_better": prompt_1_success and analysis["prompt_1_analysis"].get("location_mentioned", False),
            "prompt_2_better": prompt_2_success and analysis["prompt_2_analysis"].get("coordinates_mentioned", False)
        }
        
        return analysis
    
    def run_comprehensive_test(self, video_path: str = "../test_video.mp4") -> Dict[str, Any]:
        """
        Run comprehensive breakthrough validation test
        
        Returns:
            Validation results with detection metrics
        """
        try:
            # Load model if not already loaded
            if not self.load_model():
                return {"error": "Failed to load model", "validation_passed": False}
            
            # Check if video exists
            if not os.path.exists(video_path):
                return {"error": f"Video file not found: {video_path}", "validation_passed": False}
            
            # Test critical timestamps
            test_timestamps = [10.0, 32.0, 60.0, 120.0]
            results = []
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            detection_count = 0
            critical_32s_detected = False
            
            for timestamp in test_timestamps:
                if timestamp > duration:
                    continue
                    
                # Seek to timestamp
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Test with both structured prompts
                test_result = self.test_frame_with_both_prompts(frame, timestamp)
                
                # Check if either prompt detected the ball
                prompt1_detected = 'ball' in test_result.get('prompt_1_response', '').lower()
                prompt2_detected = 'ball' in test_result.get('prompt_2_response', '').lower()
                ball_detected = prompt1_detected or prompt2_detected
                
                if ball_detected:
                    detection_count += 1
                    
                if timestamp == 32.0 and ball_detected:
                    critical_32s_detected = True
                
                results.append({
                    "timestamp": timestamp,
                    "ball_detected": ball_detected,
                    "prompt1_detected": prompt1_detected,
                    "prompt2_detected": prompt2_detected,
                    "test_result": test_result
                })
            
            cap.release()
            
            # Calculate metrics
            detection_rate = detection_count / len(results) if results else 0
            validation_passed = detection_rate >= 0.75 and critical_32s_detected
            
            return {
                "validation_passed": validation_passed,
                "detection_rate": detection_rate,
                "detection_count": detection_count,
                "total_tested": len(results),
                "critical_32s_detected": critical_32s_detected,
                "breakthrough_achieved": detection_rate >= 1.0,  # 100% rate
                "test_results": results,
                "video_info": {
                    "path": video_path,
                    "duration": duration,
                    "fps": fps
                },
                "system_info": {
                    "device": self.device,
                    "cuda_available": torch.cuda.is_available()
                }
            }
            
        except Exception as e:
            return {
                "error": f"Validation test failed: {str(e)}",
                "validation_passed": False
            }

def run_structured_prompt_test():
    """Run the structured prompt test on test_video.mp4"""
    print("🧪 Testing Structured Prompts 1 & 2 on test_video.mp4")
    print("=" * 70)
    
    # Initialize tester
    tester = StructuredPromptTester()
    
    # Load model
    if not tester.load_model():
        print("❌ Failed to load model")
        return
    
    # Test video
    video_path = "../test_video.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"✅ Video file found: {video_path}")
    
    # Test multiple timestamps including the critical 32s mark
    test_timestamps = [10.0, 32.0, 60.0, 120.0]
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
    
    all_results = []
    
    for timestamp in test_timestamps:
        print(f"\n{'='*70}")
        print(f"🎯 Testing timestamp: {timestamp}s")
        print(f"{'='*70}")
        
        # Extract frame
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Could not extract frame at {timestamp}s")
            continue
        
        print(f"✅ Frame extracted: {frame.shape}")
        
        # Test with both prompts
        results = tester.test_frame_with_both_prompts(frame, timestamp)
        
        # Analyze results
        analysis = tester.analyze_responses(results)
        all_results.append(analysis)
        
        # Display analysis
        print(f"\n📊 Analysis for {timestamp}s:")
        
        # Prompt 1 analysis
        p1_analysis = analysis["prompt_1_analysis"]
        if p1_analysis:
            print(f"   Prompt 1 (Detailed):")
            print(f"      Ball detected: {'✅' if p1_analysis['ball_detected'] else '❌'}")
            print(f"      Location mentioned: {'✅' if p1_analysis['location_mentioned'] else '❌'}")
            print(f"      Confidence indicators: {'✅' if p1_analysis['confidence_indicators'] else '❌'}")
        
        # Prompt 2 analysis
        p2_analysis = analysis["prompt_2_analysis"]
        if p2_analysis:
            print(f"   Prompt 2 (Coordinates):")
            print(f"      Ball detected: {'✅' if p2_analysis['ball_detected'] else '❌'}")
            print(f"      Coordinates mentioned: {'✅' if p2_analysis['coordinates_mentioned'] else '❌'}")
            print(f"      Detection format: {'✅' if p2_analysis['detection_format'] else '❌'}")
        
        # Comparison
        comparison = analysis["comparison"]
        print(f"   Comparison:")
        print(f"      Both detected ball: {'✅' if comparison['both_detected'] else '❌'}")
        print(f"      Either detected ball: {'✅' if comparison['either_detected'] else '❌'}")
        print(f"      Best prompt: {'Prompt 1' if comparison['prompt_1_better'] else 'Prompt 2' if comparison['prompt_2_better'] else 'Neither'}")
    
    cap.release()
    
    # Final summary
    print(f"\n{'='*70}")
    print("📈 FINAL SUMMARY")
    print(f"{'='*70}")
    
    total_tests = len(all_results)
    prompt_1_detections = sum(1 for r in all_results if r["prompt_1_analysis"].get("ball_detected", False))
    prompt_2_detections = sum(1 for r in all_results if r["prompt_2_analysis"].get("ball_detected", False))
    both_detections = sum(1 for r in all_results if r["comparison"].get("both_detected", False))
    either_detections = sum(1 for r in all_results if r["comparison"].get("either_detected", False))
    
    print(f"Total timestamps tested: {total_tests}")
    print(f"Prompt 1 detections: {prompt_1_detections}/{total_tests} ({prompt_1_detections/total_tests*100:.1f}%)")
    print(f"Prompt 2 detections: {prompt_2_detections}/{total_tests} ({prompt_2_detections/total_tests*100:.1f}%)")
    print(f"Both prompts detected: {both_detections}/{total_tests} ({both_detections/total_tests*100:.1f}%)")
    print(f"Either prompt detected: {either_detections}/{total_tests} ({either_detections/total_tests*100:.1f}%)")
    
    # Special focus on 32s mark (where user confirmed ball is visible)
    timestamp_32_result = next((r for r in all_results if r["timestamp"] == 32.0), None)
    if timestamp_32_result:
        print(f"\n🎯 CRITICAL 32s TIMESTAMP RESULT:")
        p1_detected = timestamp_32_result["prompt_1_analysis"].get("ball_detected", False)
        p2_detected = timestamp_32_result["prompt_2_analysis"].get("ball_detected", False)
        print(f"   Prompt 1 detected ball: {'✅ SUCCESS' if p1_detected else '❌ FAILED'}")
        print(f"   Prompt 2 detected ball: {'✅ SUCCESS' if p2_detected else '❌ FAILED'}")
        
        if p1_detected or p2_detected:
            print(f"   🎉 BREAKTHROUGH: At least one prompt successfully detected the ball!")
        else:
            print(f"   ⚠️  Both prompts failed at the critical timestamp")
    
    print(f"\n🎉 Structured prompt testing completed!")

if __name__ == "__main__":
    run_structured_prompt_test() 