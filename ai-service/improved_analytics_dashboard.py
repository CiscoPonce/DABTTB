#!/usr/bin/env python3
"""
DABTTB Improved Analytics Dashboard - Enhanced Layout

Improved analytics dashboard with better layout positioning for trajectory visualization
and analysis statistics. The grey "Analysis Statistics" info box is now positioned
next to the "Trajectory Generated Successfully" section for better visual organization.

Computer Science Project - London BSc Computer Systems Engineering
London South Bank University
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

class ImprovedDABTTBAnalyticsDashboard:
    """Improved analytics dashboard with enhanced layout for DABTTB analysis results"""
    
    def __init__(self, db_path: str = "/app/results/ttball_new.duckdb"):
        """Initialize dashboard with database connection"""
        self.db_path = db_path
        self.conn = None
        self.connect_database()
    
    def connect_database(self):
        """Connect to DuckDB database"""
        try:
            self.conn = duckdb.connect(database=self.db_path, read_only=True)
            print(f"✅ Connected to database: {self.db_path}")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get summary statistics from database"""
        summary = {}
        
        try:
            # Video metadata summary
            videos_query = """
            SELECT 
                COUNT(*) as total_videos,
                AVG(duration) as avg_duration,
                AVG(fps) as avg_fps,
                resolution
            FROM video_metadata
            """
            videos_result = self.conn.execute(videos_query).fetchone()
            
            summary['videos'] = {
                'total': videos_result[0] if videos_result[0] else 0,
                'avg_duration': round(videos_result[1], 2) if videos_result[1] else 0,
                'avg_fps': round(videos_result[2], 2) if videos_result[2] else 0,
                'avg_resolution': videos_result[3] if videos_result[3] else 'Unknown'
            }
            
            # Frame analysis summary
            frames_query = """
            SELECT 
                COUNT(*) as total_frames,
                COUNT(CASE WHEN ball_detected = true THEN 1 END) as frames_with_ball,
                AVG(CASE WHEN ball_detected = true THEN ball_confidence END) as avg_confidence,
                MIN(timestamp_seconds) as min_time,
                MAX(timestamp_seconds) as max_time
            FROM frame_analysis
            """
            frames_result = self.conn.execute(frames_query).fetchone()
            
            total_frames = frames_result[0] if frames_result[0] else 0
            frames_with_ball = frames_result[1] if frames_result[1] else 0
            detection_rate = (frames_with_ball / total_frames * 100) if total_frames > 0 else 0
            
            summary['frames'] = {
                'total': total_frames,
                'with_ball': frames_with_ball,
                'detection_rate': round(detection_rate, 1),
                'avg_confidence': round(frames_result[2], 3) if frames_result[2] else 0,
                'time_range': f"{frames_result[3]:.1f}s - {frames_result[4]:.1f}s" if frames_result[3] and frames_result[4] else 'Unknown'
            }
            
        except Exception as e:
            print(f"❌ Error getting database summary: {e}")
            summary = {
                'videos': {'total': 0, 'avg_duration': 0, 'avg_fps': 0, 'avg_resolution': 'Unknown'},
                'frames': {'total': 0, 'with_ball': 0, 'detection_rate': 0, 'avg_confidence': 0, 'time_range': 'Unknown'}
            }
        
        return summary
    
    def get_ball_detections(self, video_id: Optional[int] = None) -> pd.DataFrame:
        """Get ball detection data for visualization"""
        try:
            if video_id:
                query = """
                SELECT fa.*, vm.filename, vm.resolution
                FROM frame_analysis fa
                JOIN video_metadata vm ON fa.video_id = vm.id
                WHERE fa.video_id = ?
                ORDER BY fa.timestamp_seconds
                """
                return self.conn.execute(query, [video_id]).df()
            else:
                query = """
                SELECT fa.*, vm.filename, vm.resolution
                FROM frame_analysis fa
                JOIN video_metadata vm ON fa.video_id = vm.id
                ORDER BY fa.video_id, fa.timestamp_seconds
                """
                return self.conn.execute(query).df()
        except Exception as e:
            print(f"❌ Error getting ball detections: {e}")
            return pd.DataFrame()
    
    def plot_improved_trajectory(self, video_id: Optional[int] = None, save_path: Optional[str] = None):
        """Create improved trajectory visualization with better layout"""
        detections = self.get_ball_detections(video_id)
        
        if detections.empty:
            print("❌ No ball detection data found for visualization")
            return
        
        # Filter only detected balls
        ball_detections = detections[detections['ball_detected'] == True]
        
        if ball_detections.empty:
            print("❌ No ball detections found for trajectory")
            return
        
        # Create figure with improved layout
        fig = plt.figure(figsize=(16, 10))
        
        # Get video dimensions for proper scaling
        if not detections.empty and 'resolution' in detections.columns:
            resolution_str = detections['resolution'].iloc[0]
            video_width, video_height = map(int, resolution_str.split('x'))
        else:
            video_width, video_height = 1920, 1080
            print("⚠️ Warning: Video resolution not found, using default 1920x1080")
        
        # Main trajectory plot (larger, positioned on the left)
        plt.subplot(2, 3, (1, 4))  # Spans 2 rows, 1 column
        plt.scatter(ball_detections['ball_x'], ball_detections['ball_y'], 
                   c=ball_detections['timestamp_seconds'], cmap='viridis', 
                   s=ball_detections['ball_confidence']*100, alpha=0.7)
        plt.colorbar(label='Time (seconds)')
        plt.xlim(0, video_width)
        plt.ylim(video_height, 0)  # Invert Y axis for image coordinates
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Ball Trajectory Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trajectory success indicator
        plt.text(0.02, 0.98, '✅ Trajectory Generated Successfully', 
                transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', color='green',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # Analysis Statistics (positioned next to trajectory)
        plt.subplot(2, 3, 2)
        stats_text = f"""Analysis Statistics
        
Total Frames: {len(detections)}
Frames with Ball: {len(ball_detections)}
Detection Rate: {len(ball_detections)/len(detections)*100:.1f}%
Avg Confidence: {ball_detections['ball_confidence'].mean():.3f}
Time Range: {detections['timestamp_seconds'].min():.1f}s - {detections['timestamp_seconds'].max():.1f}s
Video: {detections['filename'].iloc[0]}"""
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontweight='normal',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        plt.axis('off')
        plt.title('📊 Analysis Statistics', fontsize=12, fontweight='bold', pad=20)
        
        # Detection confidence over time (top right)
        plt.subplot(2, 3, 3)
        plt.plot(ball_detections['timestamp_seconds'], ball_detections['ball_confidence'], 
                marker='o', linewidth=2, markersize=4, color='blue')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Detection Confidence')
        plt.title('Ball Detection Confidence Over Time', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        # Detection count per time interval (bottom middle)
        plt.subplot(2, 3, 5)
        time_bins = np.arange(0, detections['timestamp_seconds'].max() + 5, 5)
        detection_counts = []
        for i in range(len(time_bins)-1):
            count = len(ball_detections[
                (ball_detections['timestamp_seconds'] >= time_bins[i]) & 
                (ball_detections['timestamp_seconds'] < time_bins[i+1])
            ])
            detection_counts.append(count)
        
        plt.bar(time_bins[:-1], detection_counts, width=4, alpha=0.7, color='orange')
        plt.xlabel('Time Interval (seconds)')
        plt.ylabel('Ball Detections')
        plt.title('Ball Detections per 5-Second Interval', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Movement analysis (bottom right)
        plt.subplot(2, 3, 6)
        if len(ball_detections) > 1:
            # Calculate movement distances
            x_diff = ball_detections['ball_x'].diff().fillna(0)
            y_diff = ball_detections['ball_y'].diff().fillna(0)
            movement_distances = np.sqrt(x_diff**2 + y_diff**2)
            
            plt.plot(ball_detections['timestamp_seconds'], movement_distances, 
                    marker='s', linewidth=2, markersize=3, color='red')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Movement Distance (pixels)')
            plt.title('Ball Movement Analysis', fontsize=12)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Insufficient data\nfor movement analysis', 
                    transform=plt.gca().transAxes, ha='center', va='center',
                    fontsize=12, style='italic')
            plt.axis('off')
        
        plt.tight_layout(pad=3.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Improved trajectory plot saved to: {save_path}")
        else:
            plt.show()
    
    def display_summary(self):
        """Display database summary statistics"""
        summary = self.get_database_summary()
        
        print("\n" + "="*60)
        print("🏓 DABTTB Analytics Dashboard - Summary")
        print("="*60)
        
        # Video statistics
        print(f"\n📹 Video Analysis:")
        print(f"   Total Videos: {summary['videos']['total']}")
        print(f"   Average Duration: {summary['videos']['avg_duration']}s")
        print(f"   Average FPS: {summary['videos']['avg_fps']}")
        print(f"   Average Resolution: {summary['videos']['avg_resolution']}")
        
        # Frame analysis statistics
        print(f"\n🎯 Frame Analysis:")
        print(f"   Total Frames: {summary['frames']['total']}")
        print(f"   Frames with Ball: {summary['frames']['with_ball']}")
        print(f"   Detection Rate: {summary['frames']['detection_rate']}%")
        print(f"   Average Confidence: {summary['frames']['avg_confidence']}")
        print(f"   Time Range: {summary['frames']['time_range']}")
        
        print("\n" + "="*60)
    
    def run_improved_dashboard(self, video_id: int = 4):
        """Run improved dashboard for specific video"""
        print("\n" + "="*60)
        print("🏓 DABTTB Improved Analytics Dashboard")
        print("="*60)
        
        # Display summary
        self.display_summary()
        
        # Generate improved trajectory plot
        print(f"\n📊 Generating improved trajectory visualization for video_id={video_id}...")
        save_path = f"/app/results/improved_trajectory_video_{video_id}.png"
        self.plot_improved_trajectory(video_id=video_id, save_path=save_path)
        
        print(f"\n✅ Improved dashboard completed!")
        print(f"   - Grey 'Analysis Statistics' box now positioned next to 'Trajectory Generated Successfully'")
        print(f"   - Enhanced layout with better visual organization")
        print(f"   - Larger trajectory plot for better analysis")
        print(f"   - Additional movement analysis included")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")

def main():
    """Main function to run the improved dashboard"""
    print("🚀 Starting DABTTB Improved Analytics Dashboard...")
    
    try:
        dashboard = ImprovedDABTTBAnalyticsDashboard()
        dashboard.run_improved_dashboard(video_id=4)
        dashboard.close()
    except Exception as e:
        print(f"❌ Dashboard failed: {e}")

if __name__ == "__main__":
    main()
