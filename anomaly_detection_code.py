#!/usr/bin/env python3
"""
Anomaly Detection Algorithm Implementation
Key code snippets for Appendix C.2 of the dissertation report
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
import logging

@dataclass
class BounceEvent:
    """Data class representing a bounce event"""
    frame_number: int
    position: Tuple[float, float, float]  # x, y, z coordinates
    velocity_before: Tuple[float, float, float]
    velocity_after: Tuple[float, float, float]
    timestamp: float
    confidence: float

@dataclass
class AnomalyResult:
    """Data class for anomaly detection results"""
    is_anomaly: bool
    anomaly_type: str
    confidence: float
    energy_ratio: float
    expected_range: Tuple[float, float]
    description: str

class PhysicsBasedAnomalyDetector:
    """
    Physics-based anomaly detection for table tennis ball bounces
    
    This class implements the core anomaly detection algorithm used in the
    table tennis ball tracking system. It analyzes bounce events for
    physical inconsistencies.
    """
    
    def __init__(self, 
                 energy_threshold: float = 0.15,
                 spin_threshold: float = 0.3,
                 surface_threshold: float = 0.2):
        """
        Initialize the anomaly detector with physics thresholds
        
        Args:
            energy_threshold: Maximum allowed energy loss deviation
            spin_threshold: Maximum allowed spin effect deviation  
            surface_threshold: Maximum allowed surface interaction deviation
        """
        self.energy_threshold = energy_threshold
        self.spin_threshold = spin_threshold
        self.surface_threshold = surface_threshold
        
        # Physics constants
        self.GRAVITY = 9.81  # m/s²
        self.TABLE_RESTITUTION = 0.85  # Typical table tennis table
        self.BALL_MASS = 0.0027  # kg (2.7g)
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_bounce_physics(self, bounce: BounceEvent) -> AnomalyResult:
        """
        Main anomaly detection function - analyzes a single bounce event
        
        This is the core algorithm that determines if a bounce exhibits
        anomalous behavior based on physics principles.
        
        Args:
            bounce: BounceEvent object containing bounce data
            
        Returns:
            AnomalyResult object with detection results
        """
        try:
            # Calculate energy conservation
            energy_result = self._check_energy_conservation(bounce)
            
            # Check for unexpected spin effects
            spin_result = self._check_spin_effects(bounce)
            
            # Analyze surface interaction
            surface_result = self._check_surface_interaction(bounce)
            
            # Determine overall anomaly status
            anomaly_detected = (energy_result['is_anomaly'] or 
                              spin_result['is_anomaly'] or 
                              surface_result['is_anomaly'])
            
            if anomaly_detected:
                # Determine primary anomaly type
                anomaly_type = self._determine_primary_anomaly(
                    energy_result, spin_result, surface_result
                )
                
                confidence = max(
                    energy_result['confidence'],
                    spin_result['confidence'], 
                    surface_result['confidence']
                )
                
                return AnomalyResult(
                    is_anomaly=True,
                    anomaly_type=anomaly_type['type'],
                    confidence=confidence,
                    energy_ratio=energy_result['ratio'],
                    expected_range=anomaly_type['expected_range'],
                    description=anomaly_type['description']
                )
            
            else:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type="Normal",
                    confidence=0.95,
                    energy_ratio=energy_result['ratio'],
                    expected_range=(0.80, 0.95),
                    description="Bounce follows expected physics"
                )
                
        except Exception as e:
            self.logger.error(f"Error in bounce analysis: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type="Error",
                confidence=0.0,
                energy_ratio=0.0,
                expected_range=(0.0, 0.0),
                description=f"Analysis failed: {str(e)}"
            )
    
    def _check_energy_conservation(self, bounce: BounceEvent) -> Dict:
        """
        Check if bounce violates energy conservation principles
        
        Args:
            bounce: BounceEvent to analyze
            
        Returns:
            Dictionary with energy analysis results
        """
        # Calculate kinetic energy before and after bounce
        v_before = np.array(bounce.velocity_before)
        v_after = np.array(bounce.velocity_after)
        
        ke_before = 0.5 * self.BALL_MASS * np.dot(v_before, v_before)
        ke_after = 0.5 * self.BALL_MASS * np.dot(v_after, v_after)
        
        # Calculate energy ratio (should be ~0.85 for table tennis)
        energy_ratio = ke_after / ke_before if ke_before > 0 else 0
        
        # Expected range for normal bounces
        expected_min = self.TABLE_RESTITUTION - self.energy_threshold
        expected_max = self.TABLE_RESTITUTION + 0.05  # Allow slight increase
        
        is_anomaly = not (expected_min <= energy_ratio <= expected_max)
        
        # Calculate confidence based on deviation
        if is_anomaly:
            deviation = min(
                abs(energy_ratio - expected_min),
                abs(energy_ratio - expected_max)
            )
            confidence = min(0.99, deviation / self.energy_threshold)
        else:
            confidence = 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'ratio': energy_ratio,
            'expected_range': (expected_min, expected_max),
            'confidence': confidence,
            'type': 'Energy Loss Violation'
        }
    
    def _check_spin_effects(self, bounce: BounceEvent) -> Dict:
        """
        Analyze bounce for unexpected spin effects
        
        Args:
            bounce: BounceEvent to analyze
            
        Returns:
            Dictionary with spin analysis results
        """
        v_before = np.array(bounce.velocity_before)
        v_after = np.array(bounce.velocity_after)
        
        # Calculate angle change (simplified spin effect detection)
        angle_before = np.arctan2(v_before[1], v_before[0])
        angle_after = np.arctan2(v_after[1], v_after[0])
        angle_change = abs(angle_after - angle_before)
        
        # Normalize angle change
        if angle_change > np.pi:
            angle_change = 2 * np.pi - angle_change
        
        # Expected angle change for normal bounce (should be small)
        expected_max = self.spin_threshold
        is_anomaly = angle_change > expected_max
        
        confidence = min(0.99, angle_change / expected_max) if is_anomaly else 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'angle_change': angle_change,
            'expected_max': expected_max,
            'confidence': confidence,
            'type': 'Unexpected Spin Effect'
        }
    
    def _check_surface_interaction(self, bounce: BounceEvent) -> Dict:
        """
        Check for unusual surface interaction patterns
        
        Args:
            bounce: BounceEvent to analyze
            
        Returns:
            Dictionary with surface interaction results
        """
        v_before = np.array(bounce.velocity_before)
        v_after = np.array(bounce.velocity_after)
        
        # Check vertical component behavior (z-axis)
        vz_before = abs(v_before[2])
        vz_after = abs(v_after[2])
        
        # Normal bounce should reverse and reduce vertical velocity
        expected_ratio = self.TABLE_RESTITUTION
        actual_ratio = vz_after / vz_before if vz_before > 0 else 0
        
        deviation = abs(actual_ratio - expected_ratio)
        is_anomaly = deviation > self.surface_threshold
        
        confidence = min(0.99, deviation / self.surface_threshold) if is_anomaly else 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'vertical_ratio': actual_ratio,
            'expected_ratio': expected_ratio,
            'deviation': deviation,
            'confidence': confidence,
            'type': 'Surface Interaction'
        }
    
    def _determine_primary_anomaly(self, energy_result: Dict, 
                                 spin_result: Dict, 
                                 surface_result: Dict) -> Dict:
        """
        Determine the primary anomaly type based on confidence scores
        
        Args:
            energy_result: Energy conservation analysis
            spin_result: Spin effect analysis  
            surface_result: Surface interaction analysis
            
        Returns:
            Dictionary with primary anomaly information
        """
        results = [
            (energy_result, "Energy Loss Violation", (0.70, 0.90)),
            (spin_result, "Unexpected Spin Effect", (0.0, 0.3)),
            (surface_result, "Surface Interaction", (0.75, 0.95))
        ]
        
        # Find result with highest confidence
        primary = max(results, key=lambda x: x[0]['confidence'])
        
        return {
            'type': primary[1],
            'expected_range': primary[2],
            'description': f"Detected {primary[1].lower()} with {primary[0]['confidence']:.1%} confidence"
        }
    
    def batch_analyze_bounces(self, bounces: List[BounceEvent]) -> List[AnomalyResult]:
        """
        Analyze multiple bounce events for anomalies
        
        Args:
            bounces: List of BounceEvent objects
            
        Returns:
            List of AnomalyResult objects
        """
        results = []
        
        for i, bounce in enumerate(bounces):
            self.logger.info(f"Analyzing bounce {i+1}/{len(bounces)}")
            result = self.analyze_bounce_physics(bounce)
            results.append(result)
        
        return results
    
    def generate_summary_report(self, results: List[AnomalyResult]) -> Dict:
        """
        Generate summary statistics from anomaly detection results
        
        Args:
            results: List of AnomalyResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        total_bounces = len(results)
        anomalies = [r for r in results if r.is_anomaly]
        
        # Count anomaly types
        anomaly_counts = {}
        for anomaly in anomalies:
            anomaly_counts[anomaly.anomaly_type] = anomaly_counts.get(anomaly.anomaly_type, 0) + 1
        
        # Calculate statistics
        detection_rate = len(anomalies) / total_bounces if total_bounces > 0 else 0
        avg_confidence = np.mean([a.confidence for a in anomalies]) if anomalies else 0
        
        return {
            'total_bounces': total_bounces,
            'anomalies_detected': len(anomalies),
            'detection_rate': detection_rate,
            'average_confidence': avg_confidence,
            'anomaly_types': anomaly_counts,
            'normal_bounces': total_bounces - len(anomalies)
        }

# Example usage and testing functions
def create_test_bounce(frame: int, pos: Tuple[float, float, float],
                      v_before: Tuple[float, float, float],
                      v_after: Tuple[float, float, float]) -> BounceEvent:
    """Create a test bounce event"""
    return BounceEvent(
        frame_number=frame,
        position=pos,
        velocity_before=v_before,
        velocity_after=v_after,
        timestamp=frame / 30.0,  # Assuming 30 FPS
        confidence=0.95
    )

def main():
    """Example usage of the anomaly detection system"""
    
    # Initialize detector
    detector = PhysicsBasedAnomalyDetector()
    
    # Create test bounces
    test_bounces = [
        # Normal bounce
        create_test_bounce(100, (1.0, 0.5, 0.0), (2.0, 0.1, -3.0), (2.2, 0.1, 2.5)),
        
        # Energy loss violation (too much energy lost)
        create_test_bounce(200, (1.5, 0.3, 0.0), (2.5, 0.2, -3.5), (1.8, 0.2, 2.0)),
        
        # Spin effect anomaly
        create_test_bounce(300, (2.0, 0.8, 0.0), (1.8, 0.5, -2.8), (-1.2, 2.1, 2.3)),
    ]
    
    # Analyze bounces
    results = detector.batch_analyze_bounces(test_bounces)
    
    # Generate report
    summary = detector.generate_summary_report(results)
    
    print("=== ANOMALY DETECTION RESULTS ===")
    print(f"Total bounces analyzed: {summary['total_bounces']}")
    print(f"Anomalies detected: {summary['anomalies_detected']}")
    print(f"Detection rate: {summary['detection_rate']:.1%}")
    print(f"Average confidence: {summary['average_confidence']:.1%}")
    print("\nAnomaly types found:")
    for anomaly_type, count in summary['anomaly_types'].items():
        print(f"  - {anomaly_type}: {count}")
    
    return results, summary

if __name__ == "__main__":
    main()
