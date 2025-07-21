#!/usr/bin/env python3

import duckdb
import time
import random
from datetime import datetime

def generate_conversational_response(user_message: str, video_context: str = "") -> str:
    """Debug version of the conversational response function"""
    
    print(f"DEBUG: Message = '{user_message}'")
    print(f"DEBUG: Video context = '{video_context}'")
    
    # Analyze the user's message to determine response type
    message_lower = user_message.lower()
    print(f"DEBUG: Message lower = '{message_lower}'")
    
    # Parse video context for specific data
    video_data = {}
    if video_context:
        lines = video_context.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                video_data[key.strip()] = value.strip()
    
    print(f"DEBUG: Video data parsed = {video_data}")
    
    # Anomaly-focused responses
    anomaly_keywords = ['anomaly', 'anomalies', 'bounce', 'physics', 'unusual', 'strange']
    has_anomaly_keyword = any(word in message_lower for word in anomaly_keywords)
    print(f"DEBUG: Has anomaly keyword = {has_anomaly_keyword}")
    
    if has_anomaly_keyword:
        if video_context:
            print("DEBUG: Taking anomaly path with video context")
            # Extract specific anomaly data
            detection_info = video_data.get('Ball Detection', 'analysis available')
            duration = video_data.get('Duration', 'unknown duration')
            anomaly_count = video_data.get('Anomalies Found', 'unknown')
            bounce_count = video_data.get('Bounce Events', 'unknown')
            
            response = f"Looking at your Video-anomalies2.mp4 analysis, I found significant anomaly patterns. "
            response += f"Specifically: {anomaly_count} anomalies and {bounce_count} bounce events detected. "
            response += f"The system detected physics violations in your ball bounces - this typically indicates very aggressive play or unusual surface interactions. "
            response += f"With {detection_info.lower()}, I can see we have solid tracking data to analyze these bounce patterns. "
            response += f"The anomalies suggest either high-energy paddle strikes or potential measurement variations. Would you like me to explain specific bounce physics that were flagged?"
            return response
        else:
            print("DEBUG: Taking anomaly path without video context")
            return "To analyze anomalies, I need video context. Please select a specific video from the dropdown and I can provide detailed anomaly analysis including bounce physics, energy patterns, and unusual ball behaviors detected in your gameplay."
    
    print("DEBUG: Taking default path")
    return "I love discussing table tennis strategy and technique! The AI analysis provides rich data about your gameplay that we can explore together. What would you like to know more about?"

def test_chat():
    """Test the chat functionality with database context"""
    
    # Connect to database
    conn = duckdb.connect('/app/results/ttball_new.duckdb')
    
    # Get video information for Video-anomalies2.mp4
    video_data = conn.execute(
        'SELECT id, filename, duration, fps, resolution FROM video_metadata WHERE filename = ?', 
        ('Video-anomalies2.mp4',)
    ).fetchone()
    
    if video_data:
        vid_id = video_data[0]
        filename = video_data[1]
        duration = video_data[2]
        fps = video_data[3]
        resolution = video_data[4]
        
        print(f"Found video: ID={vid_id}, filename={filename}")
        
        # Get analysis stats
        frame_count = conn.execute(
            'SELECT COUNT(*) FROM frame_analysis WHERE video_id = ?', 
            (vid_id,)
        ).fetchone()[0]
        
        ball_detected_count = conn.execute(
            'SELECT COUNT(*) FROM frame_analysis WHERE video_id = ? AND ball_detected = true', 
            (vid_id,)
        ).fetchone()[0]
        
        avg_confidence = conn.execute(
            'SELECT AVG(ball_confidence) FROM frame_analysis WHERE video_id = ? AND ball_detected = true', 
            (vid_id,)
        ).fetchone()[0] or 0
        
        # Get anomaly data
        anomaly_data = conn.execute(
            'SELECT total_anomalies, physics_anomalies FROM anomaly_analysis WHERE video_id = ?', 
            (vid_id,)
        ).fetchone()
        
        if anomaly_data:
            anomaly_count = anomaly_data[0] or 0
            physics_anomalies = anomaly_data[1] or 0
        else:
            anomaly_count = 0
            physics_anomalies = 0
        
        # Get bounce events
        bounce_count = conn.execute(
            'SELECT COUNT(*) FROM bounce_events WHERE video_id = ?', 
            (vid_id,)
        ).fetchone()[0] or 0
        
        context_info = f"""
Video Context:
- Filename: {filename}
- Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)
- Resolution: {resolution} at {fps} FPS
- Analysis: {frame_count} frames analyzed
- Ball Detection: {ball_detected_count}/{frame_count} frames ({ball_detected_count/frame_count*100:.1f}%)
- Average Confidence: {avg_confidence:.2f}
- Bounce Events: {bounce_count} detected
- Anomalies Found: {anomaly_count} total ({physics_anomalies} physics)
"""
        
        print("Video context created:")
        print(context_info)
        
        # Test the response function
        response = generate_conversational_response("tell me more about the anomalies", context_info)
        print(f"\nFinal response: {response}")
        
    else:
        print("Video not found in database!")
    
    conn.close()

if __name__ == "__main__":
    test_chat() 