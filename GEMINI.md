# Gemini Conversation Context: TTBall_5 Project

This document summarizes the interaction with the Gemini CLI agent regarding the TTBall_5 project, providing context for future interactions or handover to another model.

## 1. Project Overview

TTBall_5 is a sophisticated video analysis platform designed for table tennis. It leverages AI and machine learning to process and analyze game footage, including ball detection, trajectory generation, anomaly identification, and AI-driven summaries.

## 2. System Architecture

The project utilizes a microservice architecture containerized with Docker.

### 2.1. Microservices & Containers

- **`ttball5-ai-service`**: Python-based backend (FastAPI) for AI/ML processing, database interactions, and API exposure.
- **`ttball5-frontend`**: Web-based frontend (HTML, CSS, JavaScript) for user interaction, video uploads, and analysis visualization.
- **`ttball5-nginx`**: Nginx reverse proxy for routing traffic to `ai-service` and `frontend`, serving static files.

## 3. Key Python Files in `ai-service`

| File                               | Description                                                                                                                                                           |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`main.py`**                      | The main entry point for the service, defining FastAPI endpoints.                                                                                                     |
| **`core/config.py`**               | Manages application configuration using Pydantic.                                                                                                                     |
| **`services/analysis_service.py`** | Orchestrates the video analysis pipeline.                                                                                                                             |
| **`services/video_processor.py`**  | Handles low-level video processing (e.g., frame extraction).                                                                                                          |
| **`services/model_manager.py`**    | Manages loading and unloading of AI/ML models (e.g., Gemma).                                                                                                          |
| **`services/multimodal_service.py`**| Interacts with the multimodal Gemma model for advanced analysis.                                                                                                      |
| **`services/anomaly_service.py`**  | Contains logic for detecting anomalies in gameplay and trajectories.                                                                                                  |
| **`models/ai_models.py`**          | Defines Pydantic data models for API requests and responses.                                                                                                          |
| **`check_anomalies_db.py`**        | A utility script for directly querying the DuckDB database.                                                                                                           |
| **`simple_main.py`**               | A simplified version of `main.py`, used for development and testing, and where static files were mounted.                                                               |
| **`local_development/basic_dashboard/analytics_dashboard.py`** | Contains the `TTBallAnalyticsDashboard` class, responsible for database interactions, summary generation, and plotting (including trajectory). |

## 4. Case Studies: Video Analysis

### 4.1. Analysis of `video_id=3` (Expected Behavior)

- **Purpose**: To validate the system's functionality on "clean" data.
- **Results**:
    - Video Info: `Video_ 20250410_134638.mp4`, 69.6s, 30 FPS, 480x640.
    - Gemma Summary: "High ball detection rate... excellent tracking conditions; High average confidence... clear video quality; Analyzed 70 checkpoints at 1.0s intervals..."
    - Confidence: 0.948.
    - Anomalies: No anomaly analysis found.
- **Conclusion**: This case represents the system's "happy path," demonstrating correct ball detection, checkpoint strategy, and accurate AI summarization. The absence of anomalies validates the anomaly detection system's ability to identify clean data.

### 4.2. Analysis of `video_id=4` (Anomaly Detection & Identified Bug)

- **Purpose**: To test anomaly detection capabilities using `Video-anomalies2.mp4`.
- **Results**:
    - Video Info: `Video-anomalies2.mp4`, 144.0s, 30 FPS, 1920x1080.
    - Gemma Summary: "...Detected 107 bounce events... Found 72 bounces with physics anomalies; Detected 47 high-severity anomalies requiring attention."
    - Confidence: 0.943.
    - Structured Anomaly Data (`anomaly_analysis` table): **0** for all anomaly types.
- **Conclusion**: This revealed a discrepancy: Gemma correctly identified anomalies, but the structured database table (`anomaly_analysis`) showed no anomalies. This indicates a **potential bug in the data persistence layer** where results from the `anomaly_service` are not being correctly saved to the database.

## 5. Trajectory Display Issue & Resolution

The user reported that the frontend was not displaying the trajectory plots.

### 5.1. Initial Diagnosis

- **Frontend (`app.js`)**: The `generateTrajectory()` function was hardcoded to `video_id=1` and was not dynamically selecting a video.
- **Backend (`ai-service`)**: The `/analytics/trajectory/{video_id}` endpoint was returning the correct path, but the image was not being rendered.

### 5.2. Steps Taken to Resolve

1.  **Frontend Enhancement (User Interaction)**:
    - Added a video selection dropdown (`trajectoryVideoSelect`) to `frontend/public/index.html`.
    - Modified `frontend/public/app.js` to:
        - Populate the dropdown with available video IDs and filenames from the backend's `/analytics/summary` endpoint.
        - Update `generateTrajectory()` to use the selected `video_id` from the dropdown.
        - Updated the `<img>` tag in `generateTrajectory()` to correctly display the image.
    - **Deployment**: Rebuilt `ttball5-frontend` Docker image and restarted the service.

2.  **Backend Static File Serving**:
    - Modified `ai-service/simple_main.py` to import `StaticFiles` and mount the `/app/results` directory as a static file endpoint at `/results`. This allows the frontend to directly access the generated image files.
    - **Deployment**: Rebuilt `ttball5-ai-service` Docker image and restarted the service.

3.  **Backend Plotting Robustness**:
    - Modified `local_development/basic_dashboard/analytics_dashboard.py` to ensure `video_width` and `video_height` are correctly extracted from the `detections` DataFrame (which contains all frames for the video) and used for plotting, making the resolution extraction more robust.
    - **Deployment**: Rebuilt `ttball5-ai-service` Docker image and restarted the service.

## 6. Current State

- The project is running in Docker.
- The frontend is accessible at `localhost:3005`.
- The `ai-service` is accessible at `localhost:8005`.
- The trajectory display issue should now be resolved, with images correctly rendered on the frontend.
- A potential bug in the anomaly data persistence has been identified (Gemma reports anomalies, but the `anomaly_analysis` table is empty for `video_id=4`).

---
**Next Steps for a New Model:**

- Verify the trajectory display on `localhost:3005`.
- Investigate the anomaly data persistence bug (e.g., by examining `ai-service/services/analysis_service.py` and `ai-service/services/anomaly_service.py` and their interaction with the database).
- Continue with further analysis or development tasks as requested by the user.
