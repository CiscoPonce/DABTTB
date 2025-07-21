#!/bin/bash
# GPU Support Check Script for TTBall_4 AI Service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

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
    echo -e "${PURPLE}  GPU Support Check${NC}"
    echo -e "${PURPLE}  TTBall_4 AI Service${NC}"
    echo -e "${PURPLE}================================${NC}"
    echo
}

check_nvidia_driver() {
    print_status "Checking NVIDIA drivers..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA drivers found!"
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
        echo
        return 0
    else
        print_error "NVIDIA drivers not found or nvidia-smi not available!"
        return 1
    fi
}

check_nvidia_docker() {
    print_status "Checking NVIDIA Docker support..."
    
    # Check if nvidia-container-runtime is available
    if docker info 2>/dev/null | grep -q "nvidia"; then
        print_success "NVIDIA Container Runtime detected!"
        return 0
    fi
    
    # Try to run a simple CUDA test
    if docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi &>/dev/null; then
        print_success "NVIDIA Docker support is working!"
        return 0
    else
        print_error "NVIDIA Docker support not available!"
        return 1
    fi
}

check_docker_compose_gpu() {
    print_status "Checking Docker Compose GPU configuration..."
    
    if grep -q "capabilities: \[gpu\]" docker-compose.yml; then
        print_success "Docker Compose is configured for GPU support!"
        return 0
    else
        print_warning "Docker Compose GPU configuration not found!"
        return 1
    fi
}

provide_instructions() {
    echo
    print_status "GPU Setup Instructions:"
    echo
    echo "1. Install NVIDIA Container Toolkit:"
    echo "   For Ubuntu/Debian:"
    echo "   curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -"
    echo "   distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    echo "   curl -s -L https://nvidia.github.io/nvidia-container-runtime/\$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list"
    echo "   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo
    echo "2. Restart Docker:"
    echo "   sudo systemctl restart docker"
    echo
    echo "3. Test GPU access:"
    echo "   docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi"
    echo
    echo "4. For Windows with WSL2:"
    echo "   - Install NVIDIA drivers for Windows"
    echo "   - Ensure WSL2 is using the latest kernel"
    echo "   - Install Docker Desktop with WSL2 backend"
    echo "   - Enable GPU support in Docker Desktop settings"
    echo
}

main() {
    print_header
    
    gpu_available=true
    
    if ! check_nvidia_driver; then
        gpu_available=false
    fi
    
    if ! check_nvidia_docker; then
        gpu_available=false
    fi
    
    check_docker_compose_gpu
    
    if [ "$gpu_available" = true ]; then
        echo
        print_success "🎉 GPU support is ready!"
        print_status "You can start the services with GPU acceleration:"
        echo "   docker-compose up --build"
        echo
        print_status "Expected performance improvements:"
        echo "   - Faster model loading"
        echo "   - Accelerated inference"
        echo "   - Better memory management"
        echo "   - Reduced processing time"
    else
        echo
        print_warning "⚠️  GPU support is not available!"
        print_status "The service will run on CPU mode."
        echo
        provide_instructions
        echo
        print_status "To run without GPU (current setup):"
        echo "   # Edit docker-compose.yml and change DEVICE=cuda to DEVICE=cpu"
        echo "   docker-compose up --build"
    fi
}

main "$@" 