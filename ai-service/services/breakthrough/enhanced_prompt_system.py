#!/usr/bin/env python3
"""
Enhanced Prompt System for Gemma 3N Multimodal Analysis

This module provides sophisticated prompt engineering for table tennis
video analysis, with structured prompts designed to guide Gemma 3N
towards accurate ball detection and gameplay understanding.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class AnalysisType(Enum):
    """Types of analysis that can be performed"""
    BALL_DETECTION = "ball_detection"
    BALL_TRACKING = "ball_tracking"
    GAMEPLAY_ANALYSIS = "gameplay_analysis"
    PLAYER_ANALYSIS = "player_analysis"
    SHOT_CLASSIFICATION = "shot_classification"
    GAME_STATE = "game_state"
    MOVEMENT_ANALYSIS = "movement_analysis"
    TACTICAL_ANALYSIS = "tactical_analysis"

class PromptComplexity(Enum):
    """Complexity levels for prompts"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    EXPERT = "expert"

@dataclass
class PromptTemplate:
    """Template for structured prompts"""
    system_context: str
    task_description: str
    instructions: List[str]
    response_format: Dict[str, str]
    examples: Optional[List[str]] = None
    constraints: Optional[List[str]] = None

class EnhancedPromptSystem:
    """
    Enhanced prompt engineering system for Gemma 3N multimodal analysis
    
    Features:
    - Structured prompt templates for different analysis types
    - Adaptive complexity based on analysis requirements
    - Context-aware prompt generation
    - Response format specification
    - Error handling and fallback prompts
    """
    
    def __init__(self):
        """Initialize the enhanced prompt system"""
        self.prompt_templates = self._initialize_prompt_templates()
        print("✅ Enhanced Prompt System initialized")
        print(f"   Available analysis types: {len(self.prompt_templates)}")
    
    def _initialize_prompt_templates(self) -> Dict[AnalysisType, Dict[PromptComplexity, PromptTemplate]]:
        """Initialize all prompt templates"""
        templates = {}
        
        # Ball Detection Templates
        templates[AnalysisType.BALL_DETECTION] = {
            PromptComplexity.SIMPLE: PromptTemplate(
                system_context="You are analyzing a table tennis video frame to detect the ball.",
                task_description="Find the orange table tennis ball in this image.",
                instructions=[
                    "Look for a small orange/yellow spherical object",
                    "Check all areas of the image carefully",
                    "Consider motion blur effects"
                ],
                response_format={
                    "BALL_VISIBLE": "YES or NO",
                    "CONFIDENCE": "HIGH, MEDIUM, or LOW",
                    "LOCATION": "Brief description if visible"
                }
            ),
            
            PromptComplexity.DETAILED: PromptTemplate(
                system_context="You are an expert table tennis analyst examining video frames for precise ball detection.",
                task_description="Perform comprehensive ball detection and localization in this table tennis scene.",
                instructions=[
                    "Examine the entire frame systematically for the orange table tennis ball",
                    "Look for spherical objects with orange/yellow coloration",
                    "Consider partial visibility, motion blur, or lighting effects",
                    "Check areas: in flight, on table surface, near paddles, at frame edges",
                    "Distinguish the ball from other orange objects (clothing, equipment)"
                ],
                response_format={
                    "BALL_DETECTED": "YES or NO",
                    "DETECTION_CONFIDENCE": "HIGH (90-100%), MEDIUM (70-89%), LOW (50-69%)",
                    "BALL_LOCATION": "Specific position description",
                    "BALL_STATE": "In flight, on table, being hit, or stationary",
                    "VISUAL_CHARACTERISTICS": "Size, color, clarity, motion blur",
                    "ALTERNATIVE_OBJECTS": "Other orange objects that might be confused with ball"
                },
                constraints=[
                    "Focus only on the actual table tennis ball",
                    "Provide specific spatial descriptions",
                    "Acknowledge uncertainty when appropriate"
                ]
            ),
            
            PromptComplexity.EXPERT: PromptTemplate(
                system_context="You are a professional table tennis video analyst with expertise in ball tracking and detection systems.",
                task_description="Conduct expert-level ball detection analysis with detailed spatial and temporal assessment.",
                instructions=[
                    "Perform systematic visual search across all image regions",
                    "Apply advanced ball detection criteria: spherical geometry, orange/yellow spectrum, appropriate size relative to scene",
                    "Evaluate motion indicators: blur patterns, trajectory implications, speed estimation",
                    "Consider lighting conditions, shadows, and environmental factors affecting visibility",
                    "Assess occlusion possibilities: player bodies, paddles, table elements",
                    "Distinguish between ball and similar objects using contextual analysis",
                    "Estimate ball trajectory and movement direction if visible"
                ],
                response_format={
                    "DETECTION_STATUS": "CONFIRMED, PROBABLE, UNCERTAIN, or NOT_DETECTED",
                    "CONFIDENCE_SCORE": "Numerical confidence (0-100%)",
                    "SPATIAL_COORDINATES": "Relative position in frame (quadrant/region)",
                    "BALL_CHARACTERISTICS": "Size estimation, color accuracy, motion blur assessment",
                    "TRAJECTORY_ANALYSIS": "Movement direction, speed indication, arc estimation",
                    "OCCLUSION_STATUS": "Fully visible, partially occluded, or heavily obscured",
                    "CONTEXTUAL_FACTORS": "Lighting, background, competing objects",
                    "DETECTION_REASONING": "Detailed explanation of detection decision"
                },
                examples=[
                    "CONFIRMED detection with HIGH confidence in upper-right quadrant",
                    "PROBABLE detection with motion blur indicating fast movement",
                    "UNCERTAIN due to partial occlusion by player paddle"
                ]
            )
        }
        
        # Gameplay Analysis Templates
        templates[AnalysisType.GAMEPLAY_ANALYSIS] = {
            PromptComplexity.SIMPLE: PromptTemplate(
                system_context="You are analyzing table tennis gameplay.",
                task_description="Describe what's happening in this table tennis scene.",
                instructions=[
                    "Identify the current game phase",
                    "Describe player positions",
                    "Note any action taking place"
                ],
                response_format={
                    "GAME_PHASE": "Serve, rally, or pause",
                    "PLAYERS": "Position and action description",
                    "BALL_STATUS": "Ball state if visible"
                }
            ),
            
            PromptComplexity.DETAILED: PromptTemplate(
                system_context="You are analyzing table tennis gameplay with focus on match dynamics and player interactions.",
                task_description="Provide comprehensive gameplay analysis of this table tennis scene.",
                instructions=[
                    "Identify current game phase: serve preparation, service, rally, point conclusion, or break",
                    "Analyze player positions, stances, and readiness states",
                    "Assess shot type being executed or prepared",
                    "Evaluate court positioning and tactical setup",
                    "Note equipment positions (paddles, ball if visible)"
                ],
                response_format={
                    "GAME_PHASE": "Specific phase with details",
                    "PLAYER_ANALYSIS": "Position, stance, action for each player",
                    "SHOT_CONTEXT": "Type of shot being played or prepared",
                    "TACTICAL_SITUATION": "Court positioning and strategy",
                    "BALL_INVOLVEMENT": "Ball state and its role in current action",
                    "MATCH_INTENSITY": "Competitive level and engagement"
                }
            )
        }
        
        # Movement Analysis Templates
        templates[AnalysisType.MOVEMENT_ANALYSIS] = {
            PromptComplexity.SIMPLE: PromptTemplate(
                system_context="You are analyzing movement in table tennis.",
                task_description="Identify movement and motion in this scene.",
                instructions=[
                    "Look for motion blur or movement indicators",
                    "Identify player movements",
                    "Note ball movement if visible"
                ],
                response_format={
                    "MOTION_DETECTED": "YES or NO",
                    "MOVEMENT_TYPE": "Player movement, ball movement, or both",
                    "INTENSITY": "Slow, moderate, or fast movement"
                }
            ),
            
            PromptComplexity.DETAILED: PromptTemplate(
                system_context="You are analyzing movement dynamics in table tennis with focus on motion patterns and biomechanics.",
                task_description="Conduct detailed movement analysis of this table tennis scene.",
                instructions=[
                    "Identify all forms of motion: player movements, ball trajectory, equipment motion",
                    "Analyze motion blur patterns and their implications",
                    "Assess player biomechanics: stance, stroke mechanics, footwork",
                    "Evaluate movement speed and acceleration indicators",
                    "Consider movement coordination between players"
                ],
                response_format={
                    "MOTION_INVENTORY": "Complete list of detected movements",
                    "PLAYER_KINEMATICS": "Body position, stroke phase, footwork analysis",
                    "BALL_DYNAMICS": "Trajectory, speed indicators, spin implications",
                    "MOVEMENT_QUALITY": "Technique assessment and efficiency",
                    "TEMPORAL_PHASE": "Movement timing and coordination",
                    "BIOMECHANICAL_NOTES": "Technical movement observations"
                }
            )
        }
        
        # Game State Templates
        templates[AnalysisType.GAME_STATE] = {
            PromptComplexity.SIMPLE: PromptTemplate(
                system_context="You are determining the current state of a table tennis match.",
                task_description="Identify the current game state.",
                instructions=[
                    "Determine if the game is active or paused",
                    "Check if players are in playing positions",
                    "Note if a point is being played"
                ],
                response_format={
                    "GAME_ACTIVE": "YES or NO",
                    "POINT_STATUS": "In play, between points, or break",
                    "PLAYER_READINESS": "Ready to play or not ready"
                }
            )
        }
        
        return templates
    
    def generate_prompt(self, 
                       analysis_type: AnalysisType,
                       complexity: PromptComplexity = PromptComplexity.DETAILED,
                       additional_context: str = "",
                       frame_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a structured prompt for the specified analysis type
        
        Args:
            analysis_type: Type of analysis to perform
            complexity: Complexity level of the prompt
            additional_context: Additional context information
            frame_context: Context about the frame (timestamp, sequence, etc.)
            
        Returns:
            Formatted prompt string
        """
        if analysis_type not in self.prompt_templates:
            analysis_type = AnalysisType.BALL_DETECTION  # Default fallback
        
        if complexity not in self.prompt_templates[analysis_type]:
            complexity = PromptComplexity.SIMPLE  # Fallback to simpler version
        
        template = self.prompt_templates[analysis_type][complexity]
        
        # Build the prompt
        prompt_parts = []
        
        # System context
        prompt_parts.append(f"SYSTEM: {template.system_context}")
        
        # Frame context if provided
        if frame_context:
            context_info = []
            if "timestamp" in frame_context:
                context_info.append(f"Timestamp: {frame_context['timestamp']:.1f}s")
            if "frame_number" in frame_context:
                context_info.append(f"Frame: {frame_context['frame_number']}")
            if "video_info" in frame_context:
                context_info.append(f"Video: {frame_context['video_info']}")
            
            if context_info:
                prompt_parts.append(f"CONTEXT: {', '.join(context_info)}")
        
        # Task description
        prompt_parts.append(f"TASK: {template.task_description}")
        
        # Instructions
        prompt_parts.append("INSTRUCTIONS:")
        for i, instruction in enumerate(template.instructions, 1):
            prompt_parts.append(f"{i}. {instruction}")
        
        # Additional context
        if additional_context:
            prompt_parts.append(f"ADDITIONAL CONTEXT: {additional_context}")
        
        # Response format
        prompt_parts.append("RESPONSE FORMAT:")
        for key, description in template.response_format.items():
            prompt_parts.append(f"{key}: {description}")
        
        # Constraints if provided
        if template.constraints:
            prompt_parts.append("CONSTRAINTS:")
            for constraint in template.constraints:
                prompt_parts.append(f"- {constraint}")
        
        # Examples if provided
        if template.examples:
            prompt_parts.append("EXAMPLES:")
            for example in template.examples:
                prompt_parts.append(f"- {example}")
        
        return "\n".join(prompt_parts)
    
    def get_adaptive_prompt(self, 
                          analysis_type: AnalysisType,
                          detection_history: List[Dict[str, Any]] = None,
                          frame_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate adaptive prompt based on detection history and context
        
        Args:
            analysis_type: Type of analysis to perform
            detection_history: Previous detection results for context
            frame_context: Current frame context
            
        Returns:
            Adaptive prompt string
        """
        # Determine complexity based on history
        complexity = PromptComplexity.DETAILED  # Default
        
        if detection_history:
            recent_failures = sum(1 for result in detection_history[-5:] 
                                if not result.get("success", False))
            
            if recent_failures >= 3:
                complexity = PromptComplexity.EXPERT  # Use expert prompts for difficult cases
            elif recent_failures <= 1:
                complexity = PromptComplexity.SIMPLE  # Use simple prompts for easy cases
        
        # Generate adaptive context
        adaptive_context = ""
        if detection_history:
            if recent_failures > 0:
                adaptive_context += "Previous detection attempts have been challenging. "
                adaptive_context += "Pay extra attention to subtle visual cues and partial visibility. "
            
            # Add context about ball behavior patterns
            ball_detections = [r for r in detection_history if r.get("ball_detected", False)]
            if ball_detections:
                adaptive_context += f"Ball has been detected in {len(ball_detections)} recent frames. "
        
        return self.generate_prompt(
            analysis_type=analysis_type,
            complexity=complexity,
            additional_context=adaptive_context,
            frame_context=frame_context
        )
    
    def get_available_analysis_types(self) -> List[str]:
        """Get list of available analysis types"""
        return [analysis_type.value for analysis_type in AnalysisType]
    
    def get_prompt_info(self, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Get information about available prompts for an analysis type"""
        if analysis_type not in self.prompt_templates:
            return {}
        
        templates = self.prompt_templates[analysis_type]
        return {
            "analysis_type": analysis_type.value,
            "available_complexities": [c.value for c in templates.keys()],
            "template_count": len(templates)
        }

def test_enhanced_prompts():
    """Test the enhanced prompt system"""
    print("🧪 Testing Enhanced Prompt System")
    print("=" * 50)
    
    prompt_system = EnhancedPromptSystem()
    
    # Test different analysis types and complexities
    test_cases = [
        (AnalysisType.BALL_DETECTION, PromptComplexity.SIMPLE),
        (AnalysisType.BALL_DETECTION, PromptComplexity.DETAILED),
        (AnalysisType.BALL_DETECTION, PromptComplexity.EXPERT),
        (AnalysisType.GAMEPLAY_ANALYSIS, PromptComplexity.DETAILED),
        (AnalysisType.MOVEMENT_ANALYSIS, PromptComplexity.SIMPLE),
    ]
    
    for analysis_type, complexity in test_cases:
        print(f"\n{'='*40}")
        print(f"Testing: {analysis_type.value} - {complexity.value}")
        print(f"{'='*40}")
        
        frame_context = {
            "timestamp": 32.5,
            "frame_number": 975,
            "video_info": "test_video.mp4"
        }
        
        prompt = prompt_system.generate_prompt(
            analysis_type=analysis_type,
            complexity=complexity,
            additional_context="Orange ball visible in previous frame",
            frame_context=frame_context
        )
        
        print(prompt)
        print(f"\nPrompt length: {len(prompt)} characters")
    
    # Test adaptive prompts
    print(f"\n{'='*50}")
    print("Testing Adaptive Prompts")
    print(f"{'='*50}")
    
    # Simulate detection history with failures
    detection_history = [
        {"success": False, "ball_detected": False},
        {"success": False, "ball_detected": False},
        {"success": False, "ball_detected": False},
        {"success": True, "ball_detected": True},
    ]
    
    adaptive_prompt = prompt_system.get_adaptive_prompt(
        analysis_type=AnalysisType.BALL_DETECTION,
        detection_history=detection_history,
        frame_context={"timestamp": 45.0}
    )
    
    print(adaptive_prompt)
    
    print(f"\n🎉 Enhanced prompt testing completed!")

if __name__ == "__main__":
    test_enhanced_prompts() 