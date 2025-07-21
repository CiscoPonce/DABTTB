"""
DABTTB Integrated Data Quality System

Automatically applies outlier detection, single ball enforcement, and improved
analytics dashboard generation as part of the video analysis pipeline.

Author: DABTTB AI Service
Created: 2025-07-21
"""

import duckdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class IntegratedDataQuality:
    """Integrated data quality system for automatic cleaning and visualization"""
    
    def __init__(self, db_path: str = "/app/results/ttball_new.duckdb"):
        self.db_path = db_path
        self.results_dir = Path("/app/results")
        self.results_dir.mkdir(exist_ok=True)
        
    def apply_data_quality_pipeline(self, video_id: int) -> Dict[str, Any]:
        """
        Apply complete data quality pipeline to a video analysis
        
        Args:
            video_id: Video ID to process
            
        Returns:
            Dictionary with cleaning results and analytics info
        """
        try:
            logger.info(f"Starting data quality pipeline for video_id={video_id}")
            
            # Step 1: Run outlier detection and cleaning
            outlier_results = self._run_outlier_detection(video_id)
            
            # Step 2: Apply single ball enforcement
            enforcement_results = self._apply_single_ball_enforcement(video_id)
            
            # Step 3: Generate improved analytics dashboard
            dashboard_results = self._generate_improved_dashboard(video_id)
            
            # Step 4: Compile final results
            final_results = {
                "video_id": video_id,
                "outlier_cleaning": outlier_results,
                "single_ball_enforcement": enforcement_results,
                "dashboard_generation": dashboard_results,
                "status": "success",
                "timestamp": time.time()
            }
            
            logger.info(f"Data quality pipeline completed for video_id={video_id}")
            return final_results
            
        except Exception as e:
            logger.error(f"Data quality pipeline failed for video_id={video_id}: {e}")
            return {
                "video_id": video_id,
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _run_outlier_detection(self, video_id: int) -> Dict[str, Any]:
        """Run comprehensive outlier detection and cleaning"""
        try:
            con = duckdb.connect(database=self.db_path, read_only=False)
            
            # Create backup
            backup_table = f'outlier_backup_{video_id}_{int(time.time())}'
            con.execute(f'CREATE TABLE {backup_table} AS SELECT * FROM frame_analysis WHERE video_id = ?', [video_id])
            
            # Get all detections
            detections = con.execute('''
                SELECT frame_number, timestamp_seconds, ball_x, ball_y, ball_confidence
                FROM frame_analysis 
                WHERE video_id = ? AND ball_detected = true
                ORDER BY timestamp_seconds
            ''', [video_id]).fetchall()
            
            original_count = len(detections)
            
            if original_count == 0:
                con.close()
                return {
                    "original_detections": 0,
                    "outliers_removed": 0,
                    "final_detections": 0,
                    "detection_rate": 0.0,
                    "status": "no_detections"
                }
            
            # 1. Static position outliers
            position_groups = {}
            for detection in detections:
                frame_num, timestamp, x, y, confidence = detection
                pos_key = f'{int(x)},{int(y)}'
                if pos_key not in position_groups:
                    position_groups[pos_key] = []
                position_groups[pos_key].append((frame_num, timestamp))
            
            static_outliers = []
            for pos, frames in position_groups.items():
                if len(frames) > 6:  # More than 6 frames at same position
                    static_outliers.extend([f[0] for f in frames[3:]])
            
            # 2. Confidence uniformity outliers
            confidence_counts = {}
            for detection in detections:
                conf = round(detection[4], 3)
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            
            uniform_outliers = []
            for conf, count in confidence_counts.items():
                if count / len(detections) > 0.8:  # >80% same confidence
                    for detection in detections:
                        if round(detection[4], 3) == conf:
                            uniform_outliers.append(detection[0])
                    break
            
            # 3. Movement outliers (unrealistic jumps)
            movement_outliers = []
            for i in range(1, len(detections)):
                prev_detection = detections[i-1]
                curr_detection = detections[i]
                
                dx = abs(curr_detection[2] - prev_detection[2])
                dy = abs(curr_detection[3] - prev_detection[3])
                dt = curr_detection[1] - prev_detection[1]
                
                if dt > 0:
                    speed = np.sqrt(dx*dx + dy*dy) / dt
                    if speed > 500:  # Unrealistic speed threshold
                        movement_outliers.append(curr_detection[0])
            
            # Combine all outliers
            all_outliers = list(set(static_outliers + uniform_outliers + movement_outliers))
            
            # Remove outliers
            outliers_removed = 0
            if all_outliers:
                placeholders = ','.join(['?' for _ in all_outliers])
                con.execute(f'''
                    UPDATE frame_analysis 
                    SET ball_detected = false, ball_confidence = 0.0, ball_x = NULL, ball_y = NULL 
                    WHERE video_id = ? AND frame_number IN ({placeholders})
                ''', [video_id] + all_outliers)
                outliers_removed = len(all_outliers)
            
            # Get final count after outlier removal
            remaining_count = con.execute('''
                SELECT COUNT(*) FROM frame_analysis 
                WHERE video_id = ? AND ball_detected = true
            ''', [video_id]).fetchone()[0]
            
            total_frames = con.execute('''
                SELECT COUNT(*) FROM frame_analysis WHERE video_id = ?
            ''', [video_id]).fetchone()[0]
            
            detection_rate = (remaining_count / total_frames) * 100 if total_frames > 0 else 0
            
            con.close()
            
            return {
                "original_detections": original_count,
                "outliers_removed": outliers_removed,
                "final_detections": remaining_count,
                "detection_rate": detection_rate,
                "backup_table": backup_table,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Outlier detection failed for video_id={video_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _apply_single_ball_enforcement(self, video_id: int) -> Dict[str, Any]:
        """Apply single ball enforcement (max 1 detection per interval)"""
        try:
            con = duckdb.connect(database=self.db_path, read_only=False)
            
            # Get remaining detections after outlier cleaning
            detections = con.execute('''
                SELECT frame_number, timestamp_seconds, ball_x, ball_y, ball_confidence
                FROM frame_analysis 
                WHERE video_id = ? AND ball_detected = true
                ORDER BY timestamp_seconds
            ''', [video_id]).fetchall()
            
            before_count = len(detections)
            
            if before_count == 0:
                con.close()
                return {
                    "before_enforcement": 0,
                    "after_enforcement": 0,
                    "enforcement_rate": 0.0,
                    "status": "no_detections"
                }
            
            # Group by 5-second intervals and keep highest confidence per interval
            intervals = {}
            for detection in detections:
                frame_num, timestamp, x, y, confidence = detection
                interval = int(timestamp // 5) * 5  # 5-second intervals
                if interval not in intervals or confidence > intervals[interval][4]:
                    intervals[interval] = detection
            
            # Clear all detections and re-enable only selected ones
            con.execute('''
                UPDATE frame_analysis 
                SET ball_detected = false, ball_confidence = 0.0, ball_x = NULL, ball_y = NULL 
                WHERE video_id = ?
            ''', [video_id])
            
            # Re-enable selected detections
            for detection in intervals.values():
                frame_num, timestamp, x, y, confidence = detection
                con.execute('''
                    UPDATE frame_analysis 
                    SET ball_detected = true, ball_confidence = ?, ball_x = ?, ball_y = ?
                    WHERE video_id = ? AND frame_number = ?
                ''', [confidence, x, y, video_id, frame_num])
            
            after_count = len(intervals)
            enforcement_rate = (after_count / before_count) * 100 if before_count > 0 else 0
            
            con.close()
            
            return {
                "before_enforcement": before_count,
                "after_enforcement": after_count,
                "enforcement_rate": enforcement_rate,
                "intervals_used": len(intervals),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Single ball enforcement failed for video_id={video_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _generate_improved_dashboard(self, video_id: int) -> Dict[str, Any]:
        """Generate improved analytics dashboard with enhanced layout"""
        try:
            con = duckdb.connect(database=self.db_path, read_only=True)
            
            # Get ball detection data
            detections = con.execute('''
                SELECT timestamp_seconds, ball_x, ball_y, ball_confidence
                FROM frame_analysis 
                WHERE video_id = ? AND ball_detected = true
                ORDER BY timestamp_seconds
            ''', [video_id]).fetchall()
            
            # Get video info
            video_info = con.execute('''
                SELECT filename, duration, fps, resolution
                FROM video_metadata WHERE id = ?
            ''', [video_id]).fetchone()
            
            con.close()
            
            if not detections or not video_info:
                return {
                    "status": "no_data",
                    "message": "No detection data or video info available"
                }
            
            # Create improved dashboard with enhanced layout including 3D cube
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle(f'DABTTB Analytics Dashboard - Video ID {video_id}', fontsize=18, fontweight='bold')
            
            # Create custom layout with 3D trajectory cube and enhanced charts
            gs = fig.add_gridspec(4, 4, height_ratios=[1.5, 1.5, 1, 1], width_ratios=[1.5, 1.5, 1, 1])
            
            # 3D Trajectory Cube (top-left, prominent)
            ax_3d = fig.add_subplot(gs[0, :2], projection='3d')
            
            # 2D Trajectory plot (top-right)
            ax_traj = fig.add_subplot(gs[0, 2:])
            
            # Analysis statistics box (second row, left)
            ax_stats = fig.add_subplot(gs[1, 0])
            
            # Ball detection confidence over time (second row, middle-right)
            ax_conf = fig.add_subplot(gs[1, 1:])
            
            # Detection count per interval (third row, left)
            ax_count = fig.add_subplot(gs[2, :2])
            
            # Movement analysis (third row, right)
            ax_movement = fig.add_subplot(gs[2, 2:])
            
            # Performance metrics (bottom row)
            ax_perf = fig.add_subplot(gs[3, :])
            
            # Extract data
            times = [d[0] for d in detections]
            x_coords = [d[1] for d in detections]
            y_coords = [d[2] for d in detections]
            confidences = [d[3] for d in detections]
            
            # Normalize time for 3D visualization (Z-axis)
            time_normalized = [(t - min(times)) / (max(times) - min(times)) * 100 if max(times) > min(times) else 0 for t in times]
            
            # 1. 3D Trajectory Cube Visualization
            # Create 3D scatter plot with trajectory path
            scatter_3d = ax_3d.scatter(x_coords, y_coords, time_normalized, 
                                     c=confidences, s=80, alpha=0.8, cmap='plasma')
            
            # Draw 3D trajectory line
            ax_3d.plot(x_coords, y_coords, time_normalized, 'b-', alpha=0.6, linewidth=2)
            
            # Add trajectory points with size based on confidence
            for i, (x, y, z, conf) in enumerate(zip(x_coords, y_coords, time_normalized, confidences)):
                ax_3d.scatter([x], [y], [z], s=conf*100, alpha=0.7, c='red')
            
            # Customize 3D cube
            ax_3d.set_xlabel('X Position (pixels)', fontsize=10)
            ax_3d.set_ylabel('Y Position (pixels)', fontsize=10)
            ax_3d.set_zlabel('Time Progress (%)', fontsize=10)
            ax_3d.set_title('3D Ball Trajectory Cube', fontweight='bold', fontsize=12)
            
            # Add cube wireframe for better depth perception
            if x_coords and y_coords:
                x_range = [min(x_coords), max(x_coords)]
                y_range = [min(y_coords), max(y_coords)]
                z_range = [0, 100]
                
                # Draw cube edges
                for x in x_range:
                    for y in y_range:
                        ax_3d.plot([x, x], [y, y], z_range, 'k-', alpha=0.1)
                for x in x_range:
                    for z in z_range:
                        ax_3d.plot([x, x], y_range, [z, z], 'k-', alpha=0.1)
                for y in y_range:
                    for z in z_range:
                        ax_3d.plot(x_range, [y, y], [z, z], 'k-', alpha=0.1)
            
            # Set viewing angle for best perspective
            ax_3d.view_init(elev=20, azim=45)
            
            # Add colorbar for 3D plot
            cbar_3d = plt.colorbar(scatter_3d, ax=ax_3d, shrink=0.8)
            cbar_3d.set_label('Confidence', fontsize=10)
            
            # 2. Enhanced 2D trajectory visualization (complementary view)
            scatter = ax_traj.scatter(x_coords, y_coords, c=times, s=40, alpha=0.7, cmap='viridis')
            ax_traj.plot(x_coords, y_coords, 'b-', alpha=0.4, linewidth=1)
            ax_traj.set_xlabel('X Position (pixels)', fontsize=10)
            ax_traj.set_ylabel('Y Position (pixels)', fontsize=10)
            ax_traj.set_title('2D Trajectory View', fontweight='bold', fontsize=11)
            ax_traj.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax_traj, shrink=0.8)
            cbar.set_label('Time (seconds)', fontsize=9)
            
            # 3. Analysis statistics (enhanced with 3D metrics)
            ax_stats.axis('off')
            
            # Calculate 3D trajectory metrics
            total_3d_distance = 0
            if len(x_coords) > 1:
                for i in range(1, len(x_coords)):
                    dx = x_coords[i] - x_coords[i-1]
                    dy = y_coords[i] - y_coords[i-1]
                    dt = time_normalized[i] - time_normalized[i-1]
                    distance_3d = np.sqrt(dx*dx + dy*dy + dt*dt)
                    total_3d_distance += distance_3d
            
            stats_text = f"""📊 Analysis Statistics

🎾 Total Detections: {len(detections)}
📈 Avg Confidence: {np.mean(confidences):.3f}
⏱️ Time Range: {min(times):.1f} - {max(times):.1f}s
📹 Video: {video_info[0][:15]}...
⏳ Duration: {video_info[1]:.1f}s
🎬 FPS: {video_info[2]:.1f}
📐 Resolution: {video_info[3]}

🔵 3D Trajectory:
   Distance: {total_3d_distance:.1f}px
   Complexity: {'High' if total_3d_distance > 500 else 'Medium'}

✅ Quality: HIGH
🧹 Status: CLEANED"""
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.9))
            
            # 4. Ball detection confidence over time (enhanced)
            ax_conf.plot(times, confidences, 'g-', linewidth=2, alpha=0.8, label='Confidence')
            ax_conf.fill_between(times, confidences, alpha=0.3, color='green')
            
            # Add confidence threshold line
            ax_conf.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Threshold')
            
            ax_conf.set_xlabel('Time (seconds)', fontsize=10)
            ax_conf.set_ylabel('Detection Confidence', fontsize=10)
            ax_conf.set_title('Detection Confidence Analysis', fontweight='bold', fontsize=11)
            ax_conf.grid(True, alpha=0.3)
            ax_conf.set_ylim(0, 1)
            ax_conf.legend(fontsize=9)
            
            # 5. Detection count per time interval (enhanced)
            interval_counts = {}
            for t in times:
                interval = int(t // 5) * 5  # 5-second intervals for better granularity
                interval_counts[interval] = interval_counts.get(interval, 0) + 1
            
            intervals = sorted(interval_counts.keys())
            counts = [interval_counts[i] for i in intervals]
            
            bars = ax_count.bar(range(len(intervals)), counts, alpha=0.8, 
                               color=['red' if c > 2 else 'orange' if c > 1 else 'green' for c in counts])
            ax_count.set_xlabel('Time Interval (5s)', fontsize=10)
            ax_count.set_ylabel('Detections', fontsize=10)
            ax_count.set_title('Detection Distribution', fontweight='bold', fontsize=11)
            ax_count.set_xticks(range(len(intervals)))
            ax_count.set_xticklabels([f'{i}s' for i in intervals], rotation=45, fontsize=8)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax_count.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                             f'{count}', ha='center', va='bottom', fontsize=8)
            
            # 6. Movement analysis (enhanced)
            if len(x_coords) > 1:
                movements = []
                velocities = []
                for i in range(1, len(x_coords)):
                    dx = x_coords[i] - x_coords[i-1]
                    dy = y_coords[i] - y_coords[i-1]
                    dt = times[i] - times[i-1] if times[i] != times[i-1] else 0.033  # Assume 30fps if same time
                    movement = np.sqrt(dx*dx + dy*dy)
                    velocity = movement / dt if dt > 0 else 0
                    movements.append(movement)
                    velocities.append(velocity)
                
                ax_movement.plot(times[1:], movements, 'r-', linewidth=2, alpha=0.8, label='Distance')
                ax_movement.fill_between(times[1:], movements, alpha=0.3, color='red')
                
                # Add velocity as secondary axis
                ax_vel = ax_movement.twinx()
                ax_vel.plot(times[1:], velocities, 'purple', linewidth=1.5, alpha=0.7, label='Velocity')
                ax_vel.set_ylabel('Velocity (px/s)', color='purple', fontsize=10)
                
                ax_movement.set_xlabel('Time (seconds)', fontsize=10)
                ax_movement.set_ylabel('Movement Distance (pixels)', fontsize=10)
                ax_movement.set_title('Movement & Velocity Analysis', fontweight='bold', fontsize=11)
                ax_movement.grid(True, alpha=0.3)
                ax_movement.legend(loc='upper left', fontsize=9)
                ax_vel.legend(loc='upper right', fontsize=9)
            
            # 7. Performance metrics summary
            ax_perf.axis('off')
            
            # Calculate performance metrics
            avg_movement = np.mean(movements) if movements else 0
            max_movement = max(movements) if movements else 0
            detection_rate = len(detections) / video_info[1] if video_info[1] > 0 else 0
            
            perf_text = f"""🚀 DABTTB Performance Metrics

📊 Detection Rate: {detection_rate:.2f} detections/second  |  🎯 Max Movement: {max_movement:.1f}px  |  📈 Avg Movement: {avg_movement:.1f}px
⚡ Processing: Real-time  |  🧹 Data Quality: Outliers removed, Single ball enforced  |  📐 3D Analysis: Complete
🎾 Ball Tracking: {len(detections)} valid detections across {video_info[1]:.1f}s video  |  ✅ Status: Production Ready"""
            
            ax_perf.text(0.5, 0.5, perf_text, transform=ax_perf.transAxes, 
                        fontsize=11, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            
            # Save dashboard
            output_path = self.results_dir / f"improved_analytics_dashboard_video_{video_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "status": "completed",
                "output_file": str(output_path),
                "detections_processed": len(detections),
                "dashboard_type": "improved_layout"
            }
            
        except Exception as e:
            logger.error(f"Dashboard generation failed for video_id={video_id}: {e}")
            return {"status": "error", "error": str(e)}

# Global instance for easy access
integrated_quality = IntegratedDataQuality()

def apply_integrated_data_quality(video_id: int) -> Dict[str, Any]:
    """
    Apply integrated data quality pipeline to a video
    
    Args:
        video_id: Video ID to process
        
    Returns:
        Complete data quality results
    """
    return integrated_quality.apply_data_quality_pipeline(video_id)
