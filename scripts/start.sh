#!/bin/bash
# TTBall_4 AI Service - Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}  TTBall_4 AI Service Setup${NC}"
    echo -e "${PURPLE}  Gemma 3N Multimodal Interface${NC}"
    echo -e "${PURPLE}================================${NC}"
    echo
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed!"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running!"
        exit 1
    fi
    
    print_success "All prerequisites met!"
}

# Check model files
check_models() {
    print_status "Checking Gemma 3N model files..."
    
    MODEL_DIR="models/gemma-3n-E4B"
    
    if [ ! -d "$MODEL_DIR" ]; then
        print_error "Model directory not found: $MODEL_DIR"
        exit 1
    fi
    
    # Check for essential files
    REQUIRED_FILES=("config.json" "tokenizer.json" "tokenizer.model")
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$MODEL_DIR/$file" ]; then
            print_error "Required model file not found: $MODEL_DIR/$file"
            exit 1
        fi
    done
    
    # Check for model weight files
    MODEL_FILES=$(find "$MODEL_DIR" -name "model-*.safetensors" | wc -l)
    if [ "$MODEL_FILES" -eq 0 ]; then
        print_error "No model weight files found in $MODEL_DIR"
        exit 1
    fi
    
    # Calculate total size
    TOTAL_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
    print_success "Gemma 3N model found! Size: $TOTAL_SIZE"
}

# Check available resources
check_resources() {
    print_status "Checking system resources..."
    
    # Check available memory (require at least 4GB)
    AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7/1024 }')
    if [ "$AVAILABLE_MEM" -lt 4 ]; then
        print_warning "Low available memory: ${AVAILABLE_MEM}GB (recommended: 8GB+)"
    else
        print_success "Available memory: ${AVAILABLE_MEM}GB"
    fi
    
    # Check disk space (require at least 20GB)
    AVAILABLE_DISK=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_DISK" -lt 20 ]; then
        print_warning "Low disk space: ${AVAILABLE_DISK}GB (recommended: 50GB+)"
    else
        print_success "Available disk space: ${AVAILABLE_DISK}GB"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p ai-service/uploads
    mkdir -p ai-service/results  
    mkdir -p ai-service/logs
    
    print_success "Directories created!"
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Pull base images first
    print_status "Pulling base images..."
    docker-compose pull nginx || true
    
    # Build services
    print_status "Building AI service..."
    docker-compose build ai-service
    
    print_status "Building frontend..."
    docker-compose build frontend
    
    # Start services
    print_status "Starting all services..."
    docker-compose up -d
    
    print_success "Services started!"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for AI service
    for i in {1..30}; do
        if curl -f http://localhost:8001/health &> /dev/null; then
            print_success "AI service is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "AI service failed to start!"
            docker-compose logs ai-service
            exit 1
        fi
        sleep 5
    done
    
    # Wait for frontend
    for i in {1..15}; do
        if curl -f http://localhost:3000 &> /dev/null; then
            print_success "Frontend is ready!"
            break
        fi
        if [ $i -eq 15 ]; then
            print_error "Frontend failed to start!"
            docker-compose logs frontend
            exit 1
        fi
        sleep 2
    done
    
    # Wait for nginx
    for i in {1..10}; do
        if curl -f http://localhost/health &> /dev/null; then
            print_success "Nginx proxy is ready!"
            break
        fi
        if [ $i -eq 10 ]; then
            print_error "Nginx failed to start!"
            docker-compose logs nginx
            exit 1
        fi
        sleep 2
    done
}

# Show service status
show_status() {
    echo
    print_status "Service Status:"
    docker-compose ps
    
    echo
    print_status "Access URLs:"
    echo "  🌐 Web Interface:     http://localhost"
    echo "  🤖 AI Service:       http://localhost:8001"
    echo "  📱 Frontend:         http://localhost:3000"
    echo "  📊 API Docs:         http://localhost:8001/docs"
    echo "  ⚡ Health Check:     http://localhost:8001/health"
    
    echo
    print_status "Management Commands:"
    echo "  View logs:           docker-compose logs -f"
    echo "  Stop services:       docker-compose down"
    echo "  Restart:             docker-compose restart"
    echo "  Update:              docker-compose pull && docker-compose up -d"
}

# Main execution
main() {
    print_header
    
    check_prerequisites
    check_models
    check_resources
    create_directories
    start_services
    wait_for_services
    show_status
    
    echo
    print_success "🎉 TTBall_4 AI Service with Gemma 3N is now running!"
    print_status "Visit http://localhost to start analyzing table tennis videos!"
}

# Handle script arguments
case "${1:-start}" in
    "start"|"")
        main
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose down
        print_success "Services stopped!"
        ;;
    "restart")
        print_status "Restarting services..."
        docker-compose restart
        print_success "Services restarted!"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        print_status "Cleaning up..."
        docker-compose down -v
        docker system prune -f
        print_success "Cleanup complete!"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|clean}"
        echo
        echo "Commands:"
        echo "  start    - Start all services (default)"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - Show logs"
        echo "  status   - Show service status"
        echo "  clean    - Stop and clean up everything"
        exit 1
        ;;
esac 