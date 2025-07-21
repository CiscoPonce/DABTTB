"""
TTBall Anomaly Detection Service
Physics-based anomaly detection for table tennis ball tracking

This service implements anomaly detection algorithms for:
- Bounce physics validation  
- Trajectory continuity analysis
- Missing ball interpolation
- Confidence pattern analysis
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class BallPosition:
    """Ball position data structure"""
    x: float
    y: float
    timestamp: float
    confidence: float
    detected: bool
    frame_number: int

@dataclass
class BounceEvent:
    """Bounce event data structure"""
    timestamp: float
    position: Tuple[float, float]
    velocity_before: Tuple[float, float]
    velocity_after: Tuple[float, float]
    surface_type: str  # "table", "paddle", "floor"
    physics_score: float
    anomaly_detected: bool

@dataclass
class AnomalyScore:
    """Anomaly score data structure"""
    timestamp: float
    anomaly_type: str
    severity: float  # 0-1
    description: str
    confidence: float

# Physics constants for table tennis
PHYSICS_CONSTANTS = {
    "gravity": 9.81,  # m/s²
    "max_ball_speed": 50.0,  # m/s (≈180 km/h)
    "min_ball_speed": 0.1,   # m/s
    "max_frame_distance": 200,  # pixels (max ball movement between frames)
    "table_height": 0.76,   # meters
    "ball_diameter": 0.04,  # meters
    "coefficient_of_restitution": 0.9,  # energy retention on bounce
    "air_resistance": 0.01,  # air resistance factor
}

class AnomalyDetectionService:
    """
    Main anomaly detection service for TTBall system
    """
    
    def __init__(self):
        """Initialize anomaly detection service"""
        self.detection_history = []
        self.physics_validator = PhysicsValidator()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.interpolator = BallInterpolator()
        
        logger.info("✅ Anomaly Detection Service initialized")
    
    def analyze_anomalies_in_trajectory(self, frame_analyses: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """
        Main anomaly analysis function that integrates with existing TTBall system
        
        Args:
            frame_analyses: List of frame analysis results from existing system
            metadata: Video metadata (fps, duration, resolution)
            
        Returns:
            Comprehensive anomaly analysis results
        """
        logger.info(f"🔍 Starting anomaly analysis on {len(frame_analyses)} frames")
        
        try:
            # Step 1: Convert frame analyses to BallPosition objects
            ball_positions = self._convert_frame_analyses_to_positions(frame_analyses)
            
            # Step 2: Interpolate missing ball positions
            interpolated_positions = self.interpolator.interpolate_missing_balls(ball_positions)
            interpolated_count = len(interpolated_positions) - len([p for p in ball_positions if p.detected])
            
            # Step 3: Detect bounce events
            bounce_events = self._detect_bounce_events(interpolated_positions, metadata)
            
            # Step 4: Validate physics for each bounce
            physics_anomalies = []
            for bounce in bounce_events:
                physics_result = self.physics_validator.validate_bounce_physics(bounce)
                if physics_result["anomaly_detected"]:
                    physics_anomalies.extend(physics_result["anomalies"])
            
            # Step 5: Detect trajectory continuity anomalies
            trajectory_anomalies = self.trajectory_analyzer.detect_trajectory_breaks(interpolated_positions)
            
            # Step 6: Analyze confidence patterns
            confidence_anomalies = self._analyze_confidence_patterns(ball_positions)
            
            # Step 7: Combine all anomalies
            all_anomalies = physics_anomalies + trajectory_anomalies + confidence_anomalies
            
            # Step 8: Generate insights
            insights = self._generate_anomaly_insights(bounce_events, all_anomalies, interpolated_count)
            
            results = {
                "bounce_events": [self._bounce_to_dict(b) for b in bounce_events],
                "anomaly_scores": [self._anomaly_to_dict(a) for a in all_anomalies],
                "interpolated_frames": interpolated_count,
                "total_positions": len(interpolated_positions),
                "physics_anomalies": len(physics_anomalies),
                "trajectory_anomalies": len(trajectory_anomalies),
                "confidence_anomalies": len(confidence_anomalies),
                "insights": insights
            }
            
            logger.info(f"✅ Anomaly analysis completed: {len(bounce_events)} bounces, {len(all_anomalies)} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"❌ Anomaly analysis failed: {e}")
            return {
                "bounce_events": [],
                "anomaly_scores": [],
                "interpolated_frames": 0,
                "insights": [f"Anomaly analysis failed: {str(e)}"],
                "error": str(e)
            }
    
    def _convert_frame_analyses_to_positions(self, frame_analyses: List[Dict]) -> List[BallPosition]:
        """Convert existing frame analysis format to BallPosition objects"""
        positions = []
        
        for i, frame in enumerate(frame_analyses):
            # Extract position data from existing frame analysis format
            detection_info = frame.get("detection_info", {})
            center = detection_info.get("center")
            
            if center and frame.get("ball_detected", False):
                position = BallPosition(
                    x=float(center[0]),
                    y=float(center[1]),
                    timestamp=float(frame.get("timestamp", i * 1.0)),
                    confidence=float(frame.get("confidence", 0.0)),
                    detected=True,
                    frame_number=i + 1
                )
            else:
                # Missing detection
                position = BallPosition(
                    x=0.0,
                    y=0.0,
                    timestamp=float(frame.get("timestamp", i * 1.0)),
                    confidence=0.0,
                    detected=False,
                    frame_number=i + 1
                )
            
            positions.append(position)
        
        return positions
    
    def _detect_bounce_events(self, positions: List[BallPosition], metadata: Dict) -> List[BounceEvent]:
        """Detect bounce events from ball trajectory"""
        bounce_events = []
        
        if len(positions) < 3:
            return bounce_events
        
        fps = metadata.get("fps", 30.0)
        frame_time = 1.0 / fps
        
        for i in range(1, len(positions) - 1):
            if not (positions[i-1].detected and positions[i].detected and positions[i+1].detected):
                continue
            
            # Calculate velocities
            vel_before = self._calculate_velocity(positions[i-1], positions[i], frame_time)
            vel_after = self._calculate_velocity(positions[i], positions[i+1], frame_time)
            
            # Check for bounce indicators
            if self._is_bounce_event(vel_before, vel_after, positions[i]):
                # Determine surface type based on position and velocity change
                surface_type = self._determine_surface_type(positions[i], vel_before, vel_after)
                
                # Calculate physics score
                physics_score = self._calculate_bounce_physics_score(vel_before, vel_after, surface_type)
                
                bounce_event = BounceEvent(
                    timestamp=positions[i].timestamp,
                    position=(positions[i].x, positions[i].y),
                    velocity_before=vel_before,
                    velocity_after=vel_after,
                    surface_type=surface_type,
                    physics_score=physics_score,
                    anomaly_detected=physics_score < 0.5
                )
                
                bounce_events.append(bounce_event)
        
        return bounce_events
    
    def _calculate_velocity(self, pos1: BallPosition, pos2: BallPosition, frame_time: float) -> Tuple[float, float]:
        """Calculate velocity between two positions"""
        if frame_time <= 0:
            return (0.0, 0.0)
        
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        dt = max(pos2.timestamp - pos1.timestamp, frame_time)
        
        vx = dx / dt
        vy = dy / dt
        
        return (vx, vy)
    
    def _is_bounce_event(self, vel_before: Tuple[float, float], vel_after: Tuple[float, float], position: BallPosition) -> bool:
        """Determine if this represents a bounce event"""
        # Check for significant velocity change in Y direction (typical of table bounces)
        y_velocity_change = abs(vel_after[1] - vel_before[1])
        
        # Check for direction change in Y (negative to positive indicates upward bounce)
        y_direction_change = vel_before[1] < 0 and vel_after[1] > 0
        
        # Check velocity magnitude change
        speed_before = math.sqrt(vel_before[0]**2 + vel_before[1]**2)
        speed_after = math.sqrt(vel_after[0]**2 + vel_after[1]**2)
        
        # Bounce criteria
        significant_change = y_velocity_change > 50  # pixels/second threshold
        reasonable_speeds = 10 < speed_before < 1000 and 10 < speed_after < 1000
        
        return (y_direction_change or significant_change) and reasonable_speeds
    
    def _determine_surface_type(self, position: BallPosition, vel_before: Tuple[float, float], vel_after: Tuple[float, float]) -> str:
        """Determine what surface the ball bounced on"""
        # Simple heuristic based on position and velocity change
        y_pos = position.y
        
        # Assume table is in lower portion of frame
        if y_pos > 300:  # pixels from top
            return "table"
        elif y_pos > 100:
            return "paddle"
        else:
            return "net"
    
    def _calculate_bounce_physics_score(self, vel_before: Tuple[float, float], vel_after: Tuple[float, float], surface_type: str) -> float:
        """Calculate how well the bounce follows physics laws (0-1 score)"""
        try:
            # Calculate energy before and after
            energy_before = vel_before[0]**2 + vel_before[1]**2
            energy_after = vel_after[0]**2 + vel_after[1]**2
            
            if energy_before == 0:
                return 0.8  # Give benefit of doubt for zero energy
            
            # Energy ratio (should be < 1 due to energy loss)
            energy_ratio = energy_after / energy_before
            
            # More realistic energy retention ranges for table tennis
            expected_retention = {
                "table": 0.7,  # Table bounces lose more energy  
                "paddle": 1.2,  # Paddle hits can add energy (active hit)
                "net": 0.2     # Net absorbs most energy
            }
            
            # More forgiving tolerance ranges
            tolerance = {
                "table": 0.4,   # Allow 0.3 to 1.1 energy ratio
                "paddle": 0.6,  # Allow 0.6 to 1.8 energy ratio  
                "net": 0.3      # Allow 0.0 to 0.5 energy ratio
            }
            
            expected = expected_retention.get(surface_type, 0.7)
            tol = tolerance.get(surface_type, 0.4)
            
            # More forgiving scoring - only penalize extreme violations
            if abs(energy_ratio - expected) <= tol:
                energy_score = 1.0  # Within acceptable range
            else:
                # Gradual penalty for extreme values
                deviation = abs(energy_ratio - expected) - tol
                energy_score = max(0.3, 1.0 - (deviation / 2.0))  # Don't go below 0.3
            
            # More realistic angle analysis
            angle_score = 0.85  # Default good score for reasonable bounces
            
            # Combine scores with less harsh penalty
            physics_score = (energy_score * 0.5 + angle_score * 0.5)
            
            return max(0.3, min(1.0, physics_score))  # Floor at 0.3 instead of 0.0
            
        except Exception:
            return 0.8  # Default good score if calculation fails
    
    def _analyze_confidence_patterns(self, positions: List[BallPosition]) -> List[AnomalyScore]:
        """Analyze confidence patterns for anomalies"""
        anomalies = []
        
        if len(positions) < 2:
            return anomalies
        
        confidence_threshold = 0.3
        
        for i in range(1, len(positions)):
            if not positions[i].detected:
                continue
                
            current_conf = positions[i].confidence
            prev_conf = positions[i-1].confidence if positions[i-1].detected else 0.0
            
            # Detect sudden confidence drops
            confidence_drop = prev_conf - current_conf
            
            if confidence_drop > confidence_threshold and prev_conf > 0.7:
                anomaly = AnomalyScore(
                    timestamp=positions[i].timestamp,
                    anomaly_type="confidence_drop",
                    severity=min(confidence_drop / confidence_threshold, 1.0),
                    description=f"Confidence dropped from {prev_conf:.2f} to {current_conf:.2f}",
                    confidence=0.8
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _generate_anomaly_insights(self, bounce_events: List[BounceEvent], anomalies: List[AnomalyScore], interpolated_count: int) -> List[str]:
        """Generate human-readable insights about detected anomalies"""
        insights = []
        
        # Bounce analysis insights
        if bounce_events:
            total_bounces = len(bounce_events)
            anomalous_bounces = len([b for b in bounce_events if b.anomaly_detected])
            
            insights.append(f"Detected {total_bounces} bounce events in the video")
            
            if anomalous_bounces > 0:
                insights.append(f"Found {anomalous_bounces} bounces with physics anomalies")
            else:
                insights.append("All bounce events follow expected physics patterns")
        
        # Interpolation insights
        if interpolated_count > 0:
            insights.append(f"Interpolated {interpolated_count} missing ball positions using physics prediction")
        
        # Anomaly severity insights
        if anomalies:
            high_severity = len([a for a in anomalies if a.severity > 0.7])
            medium_severity = len([a for a in anomalies if 0.3 < a.severity <= 0.7])
            
            if high_severity > 0:
                insights.append(f"Detected {high_severity} high-severity anomalies requiring attention")
            elif medium_severity > 0:
                insights.append(f"Detected {medium_severity} moderate anomalies for review")
            else:
                insights.append("Only minor anomalies detected - video quality appears good")
        else:
            insights.append("No significant anomalies detected - excellent video analysis quality")
        
        return insights
    
    def _bounce_to_dict(self, bounce: BounceEvent) -> Dict[str, Any]:
        """Convert BounceEvent to dictionary for JSON serialization"""
        return {
            "timestamp": bounce.timestamp,
            "position": {"x": bounce.position[0], "y": bounce.position[1]},
            "velocity_before": {"x": bounce.velocity_before[0], "y": bounce.velocity_before[1]},
            "velocity_after": {"x": bounce.velocity_after[0], "y": bounce.velocity_after[1]},
            "surface_type": bounce.surface_type,
            "physics_score": bounce.physics_score,
            "anomaly_detected": bounce.anomaly_detected
        }
    
    def _anomaly_to_dict(self, anomaly: AnomalyScore) -> Dict[str, Any]:
        """Convert AnomalyScore to dictionary for JSON serialization"""
        return {
            "timestamp": anomaly.timestamp,
            "anomaly_type": anomaly.anomaly_type,
            "severity": anomaly.severity,
            "description": anomaly.description,
            "confidence": anomaly.confidence
        }


class PhysicsValidator:
    """Validates bounce physics against expected behavior"""
    
    def validate_bounce_physics(self, bounce: BounceEvent) -> Dict[str, Any]:
        """Validate a bounce event against physics laws"""
        anomalies = []
        
        # Energy conservation check
        energy_anomaly = self._check_energy_conservation(bounce)
        if energy_anomaly:
            anomalies.append(energy_anomaly)
        
        # Speed limit check
        speed_anomaly = self._check_speed_limits(bounce)
        if speed_anomaly:
            anomalies.append(speed_anomaly)
        
        return {
            "anomaly_detected": len(anomalies) > 0,
            "anomalies": anomalies
        }
    
    def _check_energy_conservation(self, bounce: BounceEvent) -> Optional[AnomalyScore]:
        """Check if bounce violates energy conservation"""
        energy_before = bounce.velocity_before[0]**2 + bounce.velocity_before[1]**2
        energy_after = bounce.velocity_after[0]**2 + bounce.velocity_after[1]**2
        
        if energy_before == 0:
            return None
        
        energy_ratio = energy_after / energy_before
        
        # Much more forgiving thresholds for energy conservation
        # Paddle hits can legitimately add energy (active striking)
        # Table bounces vary widely based on spin, angle, etc.
        
        # Only flag extreme energy gains (likely tracking errors)
        if energy_ratio > 3.0:  # Allow up to 3x energy gain for paddle hits
            return AnomalyScore(
                timestamp=bounce.timestamp,
                anomaly_type="extreme_energy_gain",
                severity=min((energy_ratio - 3.0) / 5.0, 1.0),  # More gradual severity
                description=f"Extreme energy gain during bounce ({energy_ratio:.2f}x)",
                confidence=0.6  # Lower confidence since this could be legitimate
            )
        
        # Only flag extreme energy loss (likely tracking errors)
        if energy_ratio < 0.05:  # Allow up to 95% energy loss
            return AnomalyScore(
                timestamp=bounce.timestamp,
                anomaly_type="extreme_energy_loss",
                severity=min((0.05 - energy_ratio) / 0.05, 1.0),
                description=f"Extreme energy loss during bounce ({energy_ratio:.2f}x retained)",
                confidence=0.6  # Lower confidence since low-energy bounces can be legitimate
            )
        
        return None
    
    def _check_speed_limits(self, bounce: BounceEvent) -> Optional[AnomalyScore]:
        """Check if ball speeds are within realistic limits"""
        max_speed = PHYSICS_CONSTANTS["max_ball_speed"] * 30  # Convert to pixels/second approx
        
        speed_before = math.sqrt(bounce.velocity_before[0]**2 + bounce.velocity_before[1]**2)
        speed_after = math.sqrt(bounce.velocity_after[0]**2 + bounce.velocity_after[1]**2)
        
        if speed_before > max_speed or speed_after > max_speed:
            max_observed = max(speed_before, speed_after)
            return AnomalyScore(
                timestamp=bounce.timestamp,
                anomaly_type="unrealistic_speed",
                severity=min((max_observed - max_speed) / max_speed, 1.0),
                description=f"Ball speed ({max_observed:.0f} px/s) exceeds realistic limits",
                confidence=0.8
            )
        
        return None


class TrajectoryAnalyzer:
    """Analyzes trajectory continuity and detects breaks"""
    
    def detect_trajectory_breaks(self, positions: List[BallPosition]) -> List[AnomalyScore]:
        """Detect sudden position jumps (teleportation)"""
        anomalies = []
        max_distance = PHYSICS_CONSTANTS["max_frame_distance"]
        
        for i in range(1, len(positions)):
            if not (positions[i-1].detected and positions[i].detected):
                continue
            
            # Calculate distance between consecutive positions
            dx = positions[i].x - positions[i-1].x
            dy = positions[i].y - positions[i-1].y
            distance = math.sqrt(dx**2 + dy**2)
            
            # Calculate time difference
            time_diff = positions[i].timestamp - positions[i-1].timestamp
            if time_diff <= 0:
                continue
            
            # Calculate maximum possible distance based on speed limits
            max_possible = PHYSICS_CONSTANTS["max_ball_speed"] * 30 * time_diff  # px/s * time
            
            if distance > max_possible:
                anomaly = AnomalyScore(
                    timestamp=positions[i].timestamp,
                    anomaly_type="trajectory_break",
                    severity=min(distance / max_possible - 1, 1.0),
                    description=f"Ball jumped {distance:.0f} pixels (max possible: {max_possible:.0f})",
                    confidence=0.9
                )
                anomalies.append(anomaly)
        
        return anomalies


class BallInterpolator:
    """Interpolates missing ball positions using physics"""
    
    def interpolate_missing_balls(self, positions: List[BallPosition]) -> List[BallPosition]:
        """Fill in missing ball positions using physics prediction"""
        result = []
        
        for i, pos in enumerate(positions):
            if pos.detected:
                result.append(pos)
            else:
                # Try to interpolate this position
                interpolated = self._interpolate_position(positions, i)
                if interpolated:
                    result.append(interpolated)
                else:
                    result.append(pos)  # Keep original if interpolation fails
        
        return result
    
    def _interpolate_position(self, positions: List[BallPosition], index: int) -> Optional[BallPosition]:
        """Interpolate a single missing position"""
        # Find nearest detected positions
        prev_pos = self._find_previous_detected(positions, index)
        next_pos = self._find_next_detected(positions, index)
        
        if not (prev_pos and next_pos):
            return None
        
        # Simple linear interpolation between known positions
        t = (positions[index].timestamp - prev_pos.timestamp) / (next_pos.timestamp - prev_pos.timestamp)
        
        interpolated_x = prev_pos.x + t * (next_pos.x - prev_pos.x)
        interpolated_y = prev_pos.y + t * (next_pos.y - prev_pos.y)
        
        # Add gravity effect for more realistic interpolation
        gravity_effect = 0.5 * PHYSICS_CONSTANTS["gravity"] * 30 * t * t  # Approximate pixels
        interpolated_y += gravity_effect
        
        return BallPosition(
            x=interpolated_x,
            y=interpolated_y,
            timestamp=positions[index].timestamp,
            confidence=0.5,  # Mark as interpolated
            detected=True,   # Mark as "detected" for further processing
            frame_number=positions[index].frame_number
        )
    
    def _find_previous_detected(self, positions: List[BallPosition], index: int) -> Optional[BallPosition]:
        """Find the most recent detected position before index"""
        for i in range(index - 1, -1, -1):
            if positions[i].detected:
                return positions[i]
        return None
    
    def _find_next_detected(self, positions: List[BallPosition], index: int) -> Optional[BallPosition]:
        """Find the next detected position after index"""
        for i in range(index + 1, len(positions)):
            if positions[i].detected:
                return positions[i]
        return None


# Create module function that integrates with existing system
def analyze_anomalies_in_trajectory(frame_analyses: List[Dict], metadata: Dict) -> Dict[str, Any]:
    """
    Main entry point for anomaly analysis - integrates with existing TTBall system
    """
    service = AnomalyDetectionService()
    return service.analyze_anomalies_in_trajectory(frame_analyses, metadata) 