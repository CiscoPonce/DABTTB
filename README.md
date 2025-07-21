# DABTTB - Advanced Table Tennis Ball Tracking

**Computer Science Project**  
**London BSc Computer Systems Engineering**  
**London South Bank University**

---

## 🏓 Project Overview

DABTTB (Dynamic Analysis of Ball Trajectory in Table Tennis) is an advanced AI-powered system for real-time table tennis ball tracking and analysis. This project combines computer vision, machine learning, and physics-based validation to provide accurate ball detection, trajectory analysis, and anomaly detection in table tennis videos.

## 🎯 Key Features

### 🎨 **3D Visualization Engine** (NEW!)
- **3D Trajectory Cube**: Revolutionary interactive 3D ball path visualization
- **Multi-Dimensional Analysis**: X, Y, and Time dimensions with depth perception
- **Color-Coded Confidence**: Plasma colormap showing detection reliability
- **Professional Dashboard**: 7-panel enhanced analytics layout
- **Academic Presentation**: University-branded professional output

### 🤖 AI-Powered Detection
- **YOLO Object Detection**: State-of-the-art ball detection with high accuracy
- **Gemma 3N Multimodal AI**: Advanced multimodal analysis capabilities
- **Real-time Processing**: Efficient video analysis with GPU acceleration
- **CUDA Acceleration**: NVIDIA GPU support for real-time performance

### 📊 Advanced Analytics
- **Interactive 3D Trajectory**: Revolutionary cube visualization with trajectory paths
- **Physics-Based Validation**: Realistic ball movement analysis
- **Anomaly Detection**: Identification of unrealistic or outlier detections
- **Performance Metrics**: Comprehensive analysis statistics
- **Data Quality Pipeline**: Automated cleaning and validation

### 🔧 Data Quality Assurance
- **Integrated Quality Pipeline**: Automated outlier detection and cleaning
- **Single Ball Enforcement**: Physics-compliant detection (one ball at a time)
- **Confidence Analysis**: Detection confidence pattern validation
- **Movement Validation**: Realistic ball movement constraints
- **Quality Reporting**: Before/after statistics and metrics

### 📈 Enhanced Dashboard
- **7-Panel Analytics Layout**: Professional comprehensive dashboard
- **Real-time 3D Visualization**: Interactive trajectory cube
- **Export Capabilities**: Data export in multiple formats
- **Visual Reports**: Enhanced trajectory and statistics visualization
- **University Branding**: Academic presentation ready

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   AI Service    │    │   Database      │
│   (React/HTML)  │◄──►│   (FastAPI)     │◄──►│   (DuckDB)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌────▼────┐              ┌───▼───┐              ┌────▼────┐
    │ Upload  │              │ YOLO  │              │ Video   │
    │ Videos  │              │ Model │              │ Metadata│
    └─────────┘              └───────┘              └─────────┘
                                 │
                            ┌────▼────┐
                            │ Gemma   │
                            │ 3N AI   │
                            └─────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- Python 3.8+ (for local development)
- Git LFS (for large model files)

### Model Setup

**Important**: This project uses large AI models that need to be downloaded separately:

#### 1. Gemma 3N Multimodal Model
The Gemma 3N model is not included in the repository due to size constraints. Download it from HuggingFace:

```bash
# Create model directory
mkdir -p ai-service/model_files/gemma-3n-E4B

# Download from HuggingFace (requires huggingface-hub)
pip install huggingface-hub
huggingface-cli download google/gemma-3n-2b-it --local-dir ai-service/model_files/gemma-3n-E4B
```

**Alternative**: Visit [HuggingFace Gemma 3N](https://huggingface.co/google/gemma-3n-2b-it) and download manually to `ai-service/model_files/gemma-3n-E4B/`

#### 2. YOLO Model
YOLO models are automatically downloaded by Ultralytics on first run. No manual setup required.

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DABTTB
   ```

2. **Download required models** (see Model Setup above)

3. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - Frontend: http://localhost:3005
   - API Documentation: http://localhost:8005/docs

### Usage

1. **Upload Video**: Use the web interface to upload a table tennis video
2. **Analysis**: The system automatically runs Gemma-Enhanced Detection with advanced AI analysis
3. **View Results**: Access comprehensive analytics including 3D trajectory visualization
4. **Export Data**: Download analysis results and visualizations

## 📁 Project Structure

```
DABTTB/
├── ai-service/                 # Core AI service
│   ├── core/                   # Configuration and utilities
│   ├── services/               # AI services (models, analysis)
│   ├── breakthrough/           # Advanced detection system
│   └── main.py                 # FastAPI application
├── frontend/                   # Web interface
│   ├── public/                 # Static assets
│   └── src/                    # React components (if applicable)
├── docs/                       # Documentation
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # Container definition
└── README.md                   # This file
```

## 🔬 Technical Implementation

### AI Models
- **YOLO11**: Object detection for ball identification
- **Gemma 3N**: Multimodal AI for advanced analysis
- **Custom Physics Engine**: Ball trajectory validation

### Data Processing Pipeline
1. **Video Upload**: Secure file upload and validation
2. **Frame Extraction**: Efficient video processing
3. **Ball Detection**: AI-powered object detection
4. **Trajectory Analysis**: Physics-based movement validation
5. **Anomaly Detection**: Outlier identification and cleaning
6. **Results Storage**: Persistent data storage in DuckDB

### Quality Assurance Features
- **Static Position Detection**: Identifies unrealistic stationary balls
- **Confidence Uniformity Analysis**: Detects artificial confidence patterns
- **Movement Validation**: Ensures realistic ball physics
- **Single Ball Enforcement**: Maintains physical constraints

## 📊 Performance Metrics

### Detection Accuracy
- **Before Optimization**: 100% detection rate (unrealistic)
- **After Optimization**: 21.4% detection rate (realistic)
- **Outlier Reduction**: 78.6% improvement in data quality

### System Performance
- **GPU Acceleration**: NVIDIA CUDA support
- **Memory Optimization**: Dynamic VRAM allocation
- **Real-time Processing**: Efficient video analysis
- **Database Performance**: Optimized DuckDB queries

## 🛠️ API Endpoints

### Core Analysis
- `POST /analyze` - Video analysis
- `POST /detect` - Object detection
- `GET /health` - Health check
- `GET /metrics` - Performance metrics

### Analytics
- `GET /analytics/summary` - Analysis summary
- `GET /analytics/detections` - Detection data
- `POST /analytics/export` - Data export

### Advanced Features
- `POST /analyze/breakthrough` - Advanced detection
- `POST /breakthrough/validate` - Validation system

## 🧪 Testing

### Automated Tests
```bash
# Run test suite
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_detection.py
python -m pytest tests/test_analytics.py
```

### Manual Testing
- Upload test videos through the web interface
- Verify detection accuracy and trajectory analysis
- Test anomaly detection and data cleaning features

## 📈 Analytics Dashboard

The interactive analytics dashboard provides:

- **Real-time Visualization**: Live trajectory mapping
- **Statistical Analysis**: Comprehensive detection metrics
- **Data Export**: CSV and JSON export capabilities
- **Performance Monitoring**: System health and metrics

## 🔧 Configuration

### Environment Variables
```bash
# AI Service Configuration
AI_SERVICE_PORT=8005
MODEL_PATH=/app/models
DEVICE=cuda

# Database Configuration
DB_PATH=/app/results/dabttb.duckdb

# Frontend Configuration
FRONTEND_PORT=3000
API_URL=http://localhost:8005
```

### Docker Configuration
- **GPU Support**: NVIDIA runtime enabled
- **Volume Mounts**: Persistent data storage
- **Network Configuration**: Service communication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is part of a Computer Science degree program at London South Bank University. All rights reserved.

## 🎓 Academic Context

**Institution**: London South Bank University  
**Program**: BSc Computer Systems Engineering  
**Project Type**: Computer Science Final Year Project  
**Focus Areas**: Computer Vision, Machine Learning, Data Analysis

## 🙏 Acknowledgments

- London South Bank University Computer Science Department
- YOLO11 and Ultralytics team for object detection models
- Google for Gemma 3N multimodal AI capabilities
- FastAPI and modern web development communities

---

**© 2025 DABTTB AI Service - Computer Science Project**  
**London BSc Computer Systems Engineering**  
**London South Bank University**
