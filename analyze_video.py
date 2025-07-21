import sys
import os
import duckdb
from datetime import datetime

# Add the directory containing simple_main.py to the Python path
# This path is relative to the container's root, where simple_main.py is located
sys.path.insert(0, '/app')

from simple_main import analyze_video_with_checkpoints, store_analysis_in_database, initialize_database

# Initialize the database (important for fresh runs)
initialize_database()

video_filename = "Video-anomalies2.mp4"
# Path to the video file INSIDE the Docker container
container_video_path = "/app/uploads/Video-anomalies2.mp4"

# Get file size from the container path
file_size = os.path.getsize(container_video_path)
analysis_type = "anomaly"

print(f"Starting analysis for {video_filename}...")

# Perform analysis using the container_video_path
analysis_results = analyze_video_with_checkpoints(video_filename, file_size, analysis_type, container_video_path)

# Store results in analytics database
store_success = store_analysis_in_database(video_filename, analysis_results, analysis_results.get("all_frame_analyses", []))

if store_success:
    print(f"Analysis results for {video_filename} stored successfully.")
else:
    print(f"Failed to store analysis results for {video_filename}.")

# Verify data directly from DuckDB
# Connect to the database inside the container
conn = duckdb.connect('/app/results/ttball_new.duckdb')
video_id_query = conn.execute(f"SELECT id FROM video_metadata WHERE filename = '{video_filename}'").fetchone()

if video_id_query:
    video_id = video_id_query[0]
    print(f"Found video_id: {video_id} for {video_filename}")

    anomaly_data = conn.execute(f"SELECT * FROM anomaly_analysis WHERE video_id = {video_id}").fetchall()
    print(f"Anomaly analysis data for video_id {video_id}: {anomaly_data}")

    bounce_data = conn.execute(f"SELECT * FROM bounce_events WHERE video_id = {video_id}").fetchall()
    print(f"Bounce events data for video_id {video_id}: {bounce_data}")

    anomaly_scores_data = conn.execute(f"SELECT * FROM anomaly_scores WHERE video_id = {video_id}").fetchall()
    print(f"Anomaly scores data for video_id {video_id}: {anomaly_scores_data}")
else:
    print(f"Video {video_filename} not found in video_metadata after analysis.")

conn.close()