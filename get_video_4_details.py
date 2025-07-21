import duckdb
import json

def get_video_details():
    db_path = r'C:\Users\Gaming PC\Desktop\TTBall_5\ai-service\results\ttball_new.duckdb'
    video_id_to_find = 4

    try:
        con = duckdb.connect(database=db_path, read_only=True)

        print(f'--- Detailed Information for video_id = {video_id_to_find} ---\n')

        # 1. Get video information
        print('1. VIDEO INFORMATION:')
        video_info = con.execute(f"""
            SELECT id, filename, duration, fps, resolution, created_at
            FROM video_metadata 
            WHERE id = {video_id_to_find}
        """).fetchone()
        
        if video_info:
            print(f"  ID: {video_info[0]}")
            print(f"  Filename: {video_info[1]}")
            print(f"  Duration: {video_info[2]}s")
            print(f"  FPS: {video_info[3]}")
            print(f"  Resolution: {video_info[4]}")
            print(f"  Created At: {video_info[5]}")
        else:
            print(f"  No information found for video_id = {video_id_to_find}")

        # 2. Get analysis results
        print('\n2. ANALYSIS RESULTS:')
        analysis_results = con.execute(f"""
            SELECT analysis_text, timestamp_seconds, confidence
            FROM gemma_analysis 
            WHERE video_id = {video_id_to_find} 
            ORDER BY timestamp_seconds DESC
        """).fetchall()
        
        if analysis_results:
            for i, row in enumerate(analysis_results):
                print(f"  Analysis #{i+1}:")
                print(f"    Analysis Text: {row[0]}")
                print(f"    Timestamp: {row[1]}")
                print(f"    Confidence: {row[2]}")
        else:
            print("  No analysis results found")

        # 3. Get anomaly analysis
        print('\n3. ANOMALY ANALYSIS:')
        anomaly_analysis = con.execute(f"""
            SELECT total_bounces, total_anomalies, interpolated_frames, physics_anomalies, trajectory_anomalies, confidence_anomalies
            FROM anomaly_analysis 
            WHERE video_id = {video_id_to_find}
        """).fetchone()
        
        if anomaly_analysis:
            print(f"  Total Bounces: {anomaly_analysis[0]}")
            print(f"  Total Anomalies: {anomaly_analysis[1]}")
            print(f"  Interpolated Frames: {anomaly_analysis[2]}")
            print(f"  Physics Anomalies: {anomaly_analysis[3]}")
            print(f"  Trajectory Anomalies: {anomaly_analysis[4]}")
            print(f"  Confidence Anomalies: {anomaly_analysis[5]}")
        else:
            print("  No anomaly analysis found")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'con' in locals() and con:
            con.close()

if __name__ == "__main__":
    get_video_details()
