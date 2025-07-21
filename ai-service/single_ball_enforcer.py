"""
TTBall_5 Single Ball Enforcement System

This module enforces the fundamental rule of table tennis: 
ONLY ONE BALL EXISTS AT ANY GIVEN TIME

Critical Issue: After initial cleaning, multiple ball detections still exist
in the same time intervals, which violates basic physics.

Author: TTBall_5 AI Service
Date: 2025-07-21
"""

import duckdb
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BallDetection:
    """Single ball detection with priority scoring"""
    frame_number: int
    timestamp: float
    x: float
    y: float
    confidence: float
    video_id: int
    priority_score: float = 0.0

class SingleBallEnforcer:
    """
    Enforces the ONE BALL ONLY rule by removing duplicate detections
    in the same time intervals and ensuring temporal consistency
    """
    
    def __init__(self, db_path: str = "/app/results/ttball_new.duckdb"):
        self.db_path = db_path
        self.logger = logger
        
        # Temporal deduplication settings
        self.TIME_WINDOW = 1.0  # seconds - no two detections within this window
        self.MIN_DISTANCE_BETWEEN_DETECTIONS = 50.0  # pixels
        self.CONFIDENCE_WEIGHT = 0.4
        self.MOVEMENT_CONSISTENCY_WEIGHT = 0.6
        
    def enforce_single_ball_rule(self, video_id: int, create_backup: bool = True) -> Dict:
        """
        Enforce single ball rule by removing duplicate detections
        
        Args:
            video_id: Video ID to process
            create_backup: Whether to create backup
            
        Returns:
            Dict: Enforcement results
        """
        try:
            con = duckdb.connect(self.db_path, read_only=False)
            
            # Create backup if requested
            if create_backup:
                backup_table = f"single_ball_backup_{video_id}_{int(datetime.now().timestamp())}"
                con.execute(f"""
                    CREATE TABLE {backup_table} AS 
                    SELECT * FROM frame_analysis WHERE video_id = ?
                """, [video_id])
                self.logger.info(f"Created backup table: {backup_table}")
            
            # Get all current detections
            detections = con.execute("""
                SELECT frame_number, timestamp_seconds, ball_x, ball_y, ball_confidence
                FROM frame_analysis 
                WHERE video_id = ? AND ball_detected = true
                ORDER BY timestamp_seconds, frame_number
            """, [video_id]).fetchall()
            
            if not detections:
                self.logger.warning(f"No detections found for video_id={video_id}")
                return {"status": "no_detections", "removed": 0}
            
            self.logger.info(f"Processing {len(detections)} detections for single ball enforcement")
            
            # Convert to BallDetection objects
            ball_detections = [
                BallDetection(d[0], d[1], d[2], d[3], d[4], video_id)
                for d in detections
            ]
            
            # Calculate priority scores for each detection
            self._calculate_priority_scores(ball_detections)
            
            # Apply temporal deduplication
            kept_detections, removed_detections = self._apply_temporal_deduplication(ball_detections)
            
            # Remove duplicate detections from database
            if removed_detections:
                removed_frames = [d.frame_number for d in removed_detections]
                placeholders = ','.join(['?' for _ in removed_frames])
                
                con.execute(f"""
                    UPDATE frame_analysis 
                    SET ball_detected = false, 
                        ball_confidence = 0.0,
                        ball_x = NULL,
                        ball_y = NULL
                    WHERE video_id = ? AND frame_number IN ({placeholders})
                """, [video_id] + removed_frames)
                
                self.logger.info(f"Removed {len(removed_detections)} duplicate ball detections")
            
            # Verify no multiple detections remain
            verification = self._verify_single_ball_rule(con, video_id)
            
            con.close()
            
            return {
                "status": "success",
                "original_detections": len(ball_detections),
                "kept_detections": len(kept_detections),
                "removed_detections": len(removed_detections),
                "verification": verification
            }
            
        except Exception as e:
            self.logger.error(f"Error enforcing single ball rule: {e}")
            raise
    
    def _calculate_priority_scores(self, detections: List[BallDetection]) -> None:
        """
        Calculate priority scores for each detection to determine which to keep
        
        Higher score = higher priority = more likely to be kept
        """
        for i, detection in enumerate(detections):
            score = 0.0
            
            # Confidence component (0.4 weight)
            confidence_score = detection.confidence * self.CONFIDENCE_WEIGHT
            score += confidence_score
            
            # Movement consistency component (0.6 weight)
            movement_score = self._calculate_movement_consistency_score(detections, i)
            score += movement_score * self.MOVEMENT_CONSISTENCY_WEIGHT
            
            detection.priority_score = score
    
    def _calculate_movement_consistency_score(self, detections: List[BallDetection], index: int) -> float:
        """
        Calculate movement consistency score based on neighboring detections
        """
        if len(detections) < 3:
            return 0.5  # Neutral score for insufficient data
        
        current = detections[index]
        score = 0.0
        
        # Check consistency with previous detection
        if index > 0:
            prev = detections[index - 1]
            distance = np.sqrt((current.x - prev.x)**2 + (current.y - prev.y)**2)
            time_diff = current.timestamp - prev.timestamp
            
            if time_diff > 0:
                velocity = distance / time_diff
                # Reasonable velocity gets higher score
                if 10 <= velocity <= 500:  # pixels per second
                    score += 0.3
                elif velocity < 10:  # Too slow (static)
                    score += 0.1
                else:  # Too fast (unrealistic)
                    score += 0.05
        
        # Check consistency with next detection
        if index < len(detections) - 1:
            next_det = detections[index + 1]
            distance = np.sqrt((current.x - next_det.x)**2 + (current.y - next_det.y)**2)
            time_diff = next_det.timestamp - current.timestamp
            
            if time_diff > 0:
                velocity = distance / time_diff
                # Reasonable velocity gets higher score
                if 10 <= velocity <= 500:  # pixels per second
                    score += 0.3
                elif velocity < 10:  # Too slow (static)
                    score += 0.1
                else:  # Too fast (unrealistic)
                    score += 0.05
        
        # Check for position uniqueness (avoid clustering)
        cluster_penalty = 0
        for other in detections:
            if other != current:
                distance = np.sqrt((current.x - other.x)**2 + (current.y - other.y)**2)
                if distance < 20:  # Too close to another detection
                    cluster_penalty += 0.1
        
        score = max(0.0, score - cluster_penalty)
        return min(1.0, score)
    
    def _apply_temporal_deduplication(self, detections: List[BallDetection]) -> Tuple[List[BallDetection], List[BallDetection]]:
        """
        Apply temporal deduplication to ensure only one ball per time window
        """
        kept_detections = []
        removed_detections = []
        
        # Sort by timestamp
        detections.sort(key=lambda d: d.timestamp)
        
        i = 0
        while i < len(detections):
            current = detections[i]
            
            # Find all detections within the time window
            window_detections = [current]
            j = i + 1
            
            while j < len(detections) and (detections[j].timestamp - current.timestamp) <= self.TIME_WINDOW:
                window_detections.append(detections[j])
                j += 1
            
            if len(window_detections) == 1:
                # Only one detection in window, keep it
                kept_detections.append(current)
            else:
                # Multiple detections in window, keep the one with highest priority
                window_detections.sort(key=lambda d: d.priority_score, reverse=True)
                
                best_detection = window_detections[0]
                kept_detections.append(best_detection)
                
                # Remove the rest
                for det in window_detections[1:]:
                    removed_detections.append(det)
                
                self.logger.info(
                    f"Time window {current.timestamp:.1f}s: kept detection at "
                    f"({best_detection.x}, {best_detection.y}) with score {best_detection.priority_score:.3f}, "
                    f"removed {len(window_detections)-1} duplicates"
                )
            
            i = j  # Move to next group
        
        return kept_detections, removed_detections
    
    def _verify_single_ball_rule(self, con, video_id: int) -> Dict:
        """
        Verify that single ball rule is now enforced
        """
        # Check for any remaining multiple detections
        multiple_detections = con.execute("""
            SELECT timestamp_seconds, COUNT(*) as count
            FROM frame_analysis 
            WHERE video_id = ? AND ball_detected = true
            GROUP BY timestamp_seconds
            HAVING COUNT(*) > 1
        """, [video_id]).fetchall()
        
        # Check detection intervals
        interval_detections = con.execute("""
            SELECT CAST(timestamp_seconds/5 AS INTEGER) * 5 as interval_start, 
                   COUNT(*) as detections
            FROM frame_analysis 
            WHERE video_id = ? AND ball_detected = true
            GROUP BY CAST(timestamp_seconds/5 AS INTEGER)
            HAVING COUNT(*) > 1
            ORDER BY interval_start
        """, [video_id]).fetchall()
        
        return {
            "simultaneous_detections": len(multiple_detections),
            "multiple_interval_detections": len(interval_detections),
            "rule_enforced": len(multiple_detections) == 0,
            "intervals_with_multiple": [int(row[0]) for row in interval_detections]
        }
    
    def generate_enforcement_report(self, video_id: int) -> Dict:
        """
        Generate comprehensive enforcement report
        """
        try:
            con = duckdb.connect(self.db_path, read_only=True)
            
            # Current detection statistics
            total_detections = con.execute(
                "SELECT COUNT(*) FROM frame_analysis WHERE video_id = ? AND ball_detected = true", 
                [video_id]
            ).fetchone()[0]
            
            total_frames = con.execute(
                "SELECT COUNT(*) FROM frame_analysis WHERE video_id = ?", 
                [video_id]
            ).fetchone()[0]
            
            # Check for violations
            verification = self._verify_single_ball_rule(con, video_id)
            
            # Detection intervals
            intervals = con.execute("""
                SELECT CAST(timestamp_seconds/5 AS INTEGER) * 5 as interval_start, 
                       COUNT(*) as detections
                FROM frame_analysis 
                WHERE video_id = ? AND ball_detected = true
                GROUP BY CAST(timestamp_seconds/5 AS INTEGER)
                ORDER BY interval_start
            """, [video_id]).fetchall()
            
            con.close()
            
            return {
                "video_id": video_id,
                "timestamp": datetime.now().isoformat(),
                "detection_stats": {
                    "total_detections": total_detections,
                    "total_frames": total_frames,
                    "detection_rate": f"{(total_detections/total_frames)*100:.1f}%"
                },
                "single_ball_verification": verification,
                "detection_intervals": [
                    {"interval": f"{row[0]}-{row[0]+5}s", "detections": row[1]}
                    for row in intervals
                ],
                "status": "COMPLIANT" if verification["rule_enforced"] else "VIOLATION",
                "recommendations": self._generate_enforcement_recommendations(verification, total_detections, total_frames)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating enforcement report: {e}")
            raise
    
    def _generate_enforcement_recommendations(self, verification: Dict, total_detections: int, total_frames: int) -> List[str]:
        """Generate recommendations based on enforcement results"""
        recommendations = []
        
        if not verification["rule_enforced"]:
            recommendations.append("CRITICAL: Single ball rule still violated - additional cleaning required")
        
        if verification["simultaneous_detections"] > 0:
            recommendations.append(f"Remove {verification['simultaneous_detections']} simultaneous detections")
        
        if len(verification["intervals_with_multiple"]) > 0:
            recommendations.append(f"Fix {len(verification['intervals_with_multiple'])} intervals with multiple detections")
        
        detection_rate = (total_detections / total_frames) * 100
        if detection_rate > 30:
            recommendations.append("Detection rate still high - consider additional temporal spacing")
        
        if not recommendations:
            recommendations.append("Single ball rule successfully enforced - system compliant")
        
        return recommendations

def main():
    """Main function for testing the single ball enforcer"""
    enforcer = SingleBallEnforcer()
    
    video_id = 4
    
    print(f"=== SINGLE BALL RULE ENFORCEMENT FOR VIDEO_ID={video_id} ===")
    
    # Generate report before enforcement
    print("\n1. BEFORE ENFORCEMENT:")
    report_before = enforcer.generate_enforcement_report(video_id)
    print(json.dumps(report_before, indent=2))
    
    # Enforce single ball rule
    print(f"\n2. ENFORCEMENT PROCESS:")
    results = enforcer.enforce_single_ball_rule(video_id)
    print(json.dumps(results, indent=2))
    
    # Generate report after enforcement
    print("\n3. AFTER ENFORCEMENT:")
    report_after = enforcer.generate_enforcement_report(video_id)
    print(json.dumps(report_after, indent=2))

if __name__ == "__main__":
    main()
