#!/usr/bin/env python3
"""
Check database for anomalies2 video analysis data
"""

import duckdb
import json
from pathlib import Path

def check_anomalies_database():
    """Check database for anomalies2 video data"""
    
    # Connect to database
    db_path = "/app/results/ttball_new.duckdb"
    if not Path(db_path).exists():
        print(f"Database not found at: {db_path}")
        return
        
    conn = duckdb.connect(db_path)
    
    print('=== CHECKING DATABASE FOR video anomalies2 ===\n')

    # 1. Check analysis results
    print('1. VIDEO ANALYSIS RESULTS:')
    try:
        results = conn.execute("""
            SELECT video_id, analysis_type, timestamp, status, summary
            FROM analysis_results 
            WHERE video_id LIKE '%anomalies2%' 
            ORDER BY timestamp DESC
        """).fetchall()
        
        if results:
            for i, row in enumerate(results):
                print(f"  Analysis #{i+1}:")
                print(f"    Video: {row[0]}")
                print(f"    Type: {row[1]}")
                print(f"    Status: {row[3]}")
                print(f"    Summary: {row[4][:200] if row[4] else 'No summary'}...")
                print()
        else:
            print("  ❌ No analysis results found for anomalies2")
    except Exception as e:
        print(f"  ❌ Error querying analysis_results: {e}")

    # 2. Check anomaly analysis table
    print('\n2. ANOMALY ANALYSIS SUMMARY:')
    try:
        anomaly_results = conn.execute("""
            SELECT video_id, analysis_timestamp, total_bounces, total_anomalies, interpolated_frames
            FROM anomaly_analysis 
            WHERE video_id LIKE '%anomalies2%'
            ORDER BY analysis_timestamp DESC
        """).fetchall()
        
        if anomaly_results:
            for i, row in enumerate(anomaly_results):
                print(f"  Summary #{i+1}:")
                print(f"    Video: {row[0]}")
                print(f"    Total Bounces: {row[2]}")
                print(f"    Total Anomalies: {row[3]}")
                print(f"    Interpolated Frames: {row[4]}")
                print(f"    Timestamp: {row[1]}")
                print()
        else:
            print("  ❌ No anomaly analysis found for anomalies2")
    except Exception as e:
        print(f"  ❌ Error querying anomaly_analysis: {e}")

    # 3. Check detailed anomaly scores
    print('\n3. DETAILED ANOMALY SCORES:')
    try:
        scores = conn.execute("""
            SELECT video_id, frame_number, anomaly_type, severity, confidence, description
            FROM anomaly_scores 
            WHERE video_id LIKE '%anomalies2%'
            ORDER BY severity DESC, frame_number
            LIMIT 10
        """).fetchall()
        
        if scores:
            print(f"  Found {len(scores)} anomaly scores (showing first 10 by severity):")
            for i, row in enumerate(scores):
                print(f"    #{i+1}: Frame {row[1]}, Type: {row[2]}, Severity: {row[3]}")
                print(f"         Confidence: {row[4]:.2f}, Description: {row[5]}")
                print()
        else:
            print("  ❌ No detailed anomaly scores found")
    except Exception as e:
        print(f"  ❌ Error querying anomaly_scores: {e}")

    # 4. Check bounce events
    print('\n4. BOUNCE EVENTS:')
    try:
        bounces = conn.execute("""
            SELECT video_id, frame_number, surface_type, physics_score
            FROM bounce_events 
            WHERE video_id LIKE '%anomalies2%'
            ORDER BY frame_number
            LIMIT 5
        """).fetchall()
        
        if bounces:
            print(f"  Found bounce events (showing first 5):")
            for i, row in enumerate(bounces):
                print(f"    #{i+1}: Frame {row[1]}, Surface: {row[2]}, Physics Score: {row[3]:.2f}")
        else:
            print("  ❌ No bounce events found")
    except Exception as e:
        print(f"  ❌ Error querying bounce_events: {e}")

    # 5. List all videos in database
    print('\n5. ALL VIDEOS IN DATABASE:')
    try:
        all_videos = conn.execute("""
            SELECT DISTINCT video_id, COUNT(*) as analysis_count
            FROM analysis_results 
            GROUP BY video_id
            ORDER BY MAX(timestamp) DESC
            LIMIT 10
        """).fetchall()
        
        if all_videos:
            print("  Recent videos:")
            for video_id, count in all_videos:
                print(f"    - {video_id} ({count} analyses)")
        else:
            print("  ❌ No videos found in database")
    except Exception as e:
        print(f"  ❌ Error listing videos: {e}")

    conn.close()
    print('\n=== DATABASE CHECK COMPLETE ===')

if __name__ == "__main__":
    check_anomalies_database() 