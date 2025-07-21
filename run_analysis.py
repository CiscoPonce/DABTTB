import os
import sys
from simple_main import analyze_video_with_checkpoints, store_analysis_in_database

# Ensure /app is in the Python path
sys.path.insert(0, '/app')

video_path = '/app/uploads/Video-anomalies2.mp4'
filename = os.path.basename(video_path)
file_size = os.path.getsize(video_path)

print(f'Starting analysis for {filename}...')

try:
    analysis_results = analyze_video_with_checkpoints(filename, file_size, 'anomaly', video_path)
    frame_analyses = analysis_results.get('all_frame_analyses', [])
    store_success = store_analysis_in_database(filename, analysis_results, frame_analyses)
    print(f'Analysis completed. Data stored: {store_success}')
except Exception as e:
    print(f'Error during analysis: {e}')
    sys.exit(1)