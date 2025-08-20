#!/bin/bash
# Inference Systems Lab - Development Environment Setup Script
# 
# This script sets up the complete development environment for ML inference systems.
# It handles Docker setup, GPU drivers, and development tools configuration.
#
# Usage:
#   chmod +x scripts/setup_dev_environment.sh
#   ./scripts/setup_dev_environment.sh [options]
#
# Options:
#   --gpu-only     Setup only GPU drivers and Docker (skip development tools)
#   --no-gpu       Setup without GPU support (CPU-only development)
#   --jupyter      Include Jupyter Lab in the setup
#   --tensorboard  Include TensorBoard in the setup
#   --help         Show this help message

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_IMAGE="inference-lab:dev"
COMPOSE_FILE="docker-compose.dev.yml"

# Default options
GPU_SUPPORT=true
INCLUDE_JUPYTER=false
INCLUDE_TENSORBOARD=false
GPU_ONLY=false

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

show_help() {
    cat << EOF
Inference Systems Lab - Development Environment Setup

This script sets up a complete CUDA development environment for ML inference systems.

Usage:
    $0 [options]

Options:
    --gpu-only      Setup only GPU drivers and Docker (skip development tools)
    --no-gpu        Setup without GPU support (CPU-only development)
    --jupyter       Include Jupyter Lab in the setup
    --tensorboard   Include TensorBoard in the setup
    --help          Show this help message

Examples:
    $0                          # Full setup with GPU support
    $0 --no-gpu                 # CPU-only development setup
    $0 --jupyter --tensorboard  # Full setup with additional services

Requirements:
    - Ubuntu 20.04+ or similar Linux distribution
    - NVIDIA GPU (for GPU support)
    - Docker and Docker Compose
    - At least 8GB RAM, 20GB disk space

EOF
}

check_system_requirements() {
    log_step "Checking System Requirements"
    
    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot determine OS version"
        exit 1
    fi
    
    local os_name=$(grep ^NAME /etc/os-release | cut -d'"' -f2)
    log_info "Operating System: $os_name"
    
    # Check available memory
    local mem_gb=$(free -g | awk 'NR==2{print $2}')
    if [[ $mem_gb -lt 8 ]]; then
        log_warning "Only ${mem_gb}GB RAM available. Recommended: 8GB+"
    else
        log_info "Memory: ${mem_gb}GB (OK)"
    fi
    
    # Check available disk space
    local disk_gb=$(df -BG . | awk 'NR==2{print $4}' | tr -d 'G')
    if [[ $disk_gb -lt 20 ]]; then
        log_warning "Only ${disk_gb}GB disk space available. Recommended: 20GB+"
    else
        log_info "Disk space: ${disk_gb}GB (OK)"
    fi
    
    # Check for NVIDIA GPU
    if [[ $GPU_SUPPORT == true ]]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            local gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
            log_info "GPU detected: $gpu_info"
        else
            log_warning "NVIDIA GPU not detected or drivers not installed"
            log_info "Continuing with GPU setup - drivers will be installed"
        fi
    fi
}

install_docker() {
    log_step "Installing Docker and Docker Compose"
    
    if command -v docker >/dev/null 2>&1; then
        log_info "Docker is already installed: $(docker --version)"
    else
        log_info "Installing Docker..."
        
        # Install Docker's official GPG key
        sudo apt-get update
        sudo apt-get install ca-certificates curl gnupg lsb-release -y
        sudo mkdir -p /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        
        # Add Docker repository
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        # Install Docker Engine
        sudo apt-get update
        sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y
        
        # Add current user to docker group
        sudo usermod -aG docker $USER
        
        log_success "Docker installed successfully"
    fi
    
    # Check Docker Compose
    if command -v docker compose >/dev/null 2>&1; then
        log_info "Docker Compose is available: $(docker compose version)"
    else
        log_error "Docker Compose not available. Please install Docker Compose v2+"
        exit 1
    fi
}

install_nvidia_docker() {
    log_step "Installing NVIDIA Container Toolkit"
    
    if [[ $GPU_SUPPORT != true ]]; then
        log_info "Skipping NVIDIA Docker setup (--no-gpu specified)"
        return
    fi
    
    # Check if nvidia-docker2 is already installed
    if docker info 2>/dev/null | grep -q "nvidia"; then
        log_info "NVIDIA Docker runtime is already configured"
        return
    fi
    
    log_info "Installing NVIDIA Container Toolkit..."
    
    # Add NVIDIA package repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Install NVIDIA Container Toolkit
    sudo apt-get update
    sudo apt-get install nvidia-container-toolkit -y
    
    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    log_success "NVIDIA Container Toolkit installed and configured"
}

build_development_image() {
    log_step "Building Development Docker Image"
    
    cd "$PROJECT_ROOT"
    
    log_info "Building $DOCKER_IMAGE..."
    log_info "This may take 15-30 minutes for the first build..."
    
    # Build the Docker image
    if docker build -f Dockerfile.dev -t "$DOCKER_IMAGE" .; then
        log_success "Development image built successfully"
    else
        log_error "Failed to build development image"
        exit 1
    fi
    
    # Show image information
    local image_size=$(docker images "$DOCKER_IMAGE" --format "table {{.Size}}" | tail -n1)
    log_info "Image size: $image_size"
}

test_gpu_support() {
    log_step "Testing GPU Support in Container"
    
    if [[ $GPU_SUPPORT != true ]]; then
        log_info "Skipping GPU test (--no-gpu specified)"
        return
    fi
    
    log_info "Running GPU test container..."
    
    # Test CUDA availability
    if docker run --rm --gpus all "$DOCKER_IMAGE" nvidia-smi; then
        log_success "GPU support is working correctly"
    else
        log_error "GPU support test failed"
        log_error "Please check NVIDIA drivers and Docker configuration"
        exit 1
    fi
}

create_development_aliases() {
    log_step "Creating Development Aliases and Scripts"
    
    cd "$PROJECT_ROOT"
    
    # Create quick-start script
    cat > dev_start.sh << 'EOF'
#!/bin/bash
# Quick start script for Inference Systems Lab development

echo "🚀 Starting Inference Systems Lab development environment..."

# Start main development container
docker-compose -f docker-compose.dev.yml up -d dev

echo "✅ Development environment is starting!"
echo ""
echo "To connect to the development container:"
echo "  docker-compose -f docker-compose.dev.yml exec dev bash"
echo ""
echo "Or use the alias:"
echo "  alias dev-shell='docker-compose -f docker-compose.dev.yml exec dev bash'"
echo ""
echo "Jupyter Lab (if enabled): http://localhost:8889"
echo "TensorBoard (if enabled): http://localhost:6007"
EOF
    
    chmod +x dev_start.sh
    
    # Create stop script
    cat > dev_stop.sh << 'EOF'
#!/bin/bash
# Stop development environment

echo "🛑 Stopping Inference Systems Lab development environment..."
docker-compose -f docker-compose.dev.yml down

echo "✅ Development environment stopped"
EOF
    
    chmod +x dev_stop.sh
    
    # Create shell script
    cat > dev_shell.sh << 'EOF'
#!/bin/bash
# Connect to development environment shell

if ! docker-compose -f docker-compose.dev.yml ps dev | grep -q "Up"; then
    echo "🚀 Development environment not running. Starting..."
    docker-compose -f docker-compose.dev.yml up -d dev
    sleep 3
fi

echo "🔗 Connecting to development shell..."
docker-compose -f docker-compose.dev.yml exec dev bash
EOF
    
    chmod +x dev_shell.sh
    
    log_success "Development scripts created:"
    log_info "  ./dev_start.sh   - Start development environment"
    log_info "  ./dev_stop.sh    - Stop development environment"
    log_info "  ./dev_shell.sh   - Connect to development shell"
}

show_next_steps() {
    log_step "Setup Complete! Next Steps"
    
    echo -e "${GREEN}✅ Development environment setup completed successfully!${NC}\n"
    
    echo -e "${CYAN}Quick Start:${NC}"
    echo "  1. Start development environment:"
    echo "     ${YELLOW}./dev_start.sh${NC}"
    echo ""
    echo "  2. Connect to development shell:"
    echo "     ${YELLOW}./dev_shell.sh${NC}"
    echo ""
    echo "  3. In the container, configure and build:"
    echo "     ${YELLOW}source cmake_config.sh${NC}"
    echo "     ${YELLOW}cmake_debug${NC}"
    echo "     ${YELLOW}cmake --build build/debug${NC}"
    echo ""
    
    if [[ $INCLUDE_JUPYTER == true ]]; then
        echo -e "${CYAN}Jupyter Lab:${NC}"
        echo "  Start: ${YELLOW}docker-compose -f docker-compose.dev.yml --profile jupyter up -d${NC}"
        echo "  URL: ${YELLOW}http://localhost:8889${NC} (token: inference-lab-dev)"
        echo ""
    fi
    
    if [[ $INCLUDE_TENSORBOARD == true ]]; then
        echo -e "${CYAN}TensorBoard:${NC}"
        echo "  Start: ${YELLOW}docker-compose -f docker-compose.dev.yml --profile tensorboard up -d${NC}"
        echo "  URL: ${YELLOW}http://localhost:6007${NC}"
        echo ""
    fi
    
    echo -e "${CYAN}Additional Commands:${NC}"
    echo "  - View logs: ${YELLOW}docker-compose -f docker-compose.dev.yml logs -f dev${NC}"
    echo "  - Stop environment: ${YELLOW}./dev_stop.sh${NC}"
    echo "  - Rebuild image: ${YELLOW}docker-compose -f docker-compose.dev.yml build${NC}"
    echo ""
    
    if [[ $GPU_SUPPORT == true ]]; then
        echo -e "${CYAN}GPU Development:${NC}"
        echo "  - Test GPU: ${YELLOW}docker run --rm --gpus all $DOCKER_IMAGE nvidia-smi${NC}"
        echo "  - CUDA samples available in container at ${YELLOW}/usr/local/cuda/samples${NC}"
        echo ""
    fi
    
    echo -e "${PURPLE}Happy Coding! 🎉${NC}"
}

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-only)
            GPU_ONLY=true
            shift
            ;;
        --no-gpu)
            GPU_SUPPORT=false
            shift
            ;;
        --jupyter)
            INCLUDE_JUPYTER=true
            shift
            ;;
        --tensorboard)
            INCLUDE_TENSORBOARD=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Setup Process
# =============================================================================

main() {
    echo -e "${CYAN}"
    cat << 'EOF'
    ___        __                              ____            
   /   |____  / /__  ________  ________  _____/ __/___  _______
  / /| |  / |/ / _ \/ ___/ _ \/ ___/ _ \/ ___/ __/ __ \/_____/ 
 / ___ |\__   | |_/ (_  |  __/ /  |  __/ | | /_//  __/       
/_/  |_|  |_/_/|_/| .__ /___/\___/_|   \___/_/   \___/        
                  |/                                           
    ____        __                         __    __        __  
   / __ \__  __/ /_  ___  _____  ____ ___/ /_  / /  ____ _/ /_ 
  / /_/ / / / / __ \/ _ \/ ___/ / __ `/ _ \/ __/ / /  / __ `/ _ \
 / ____/ /_/ / /_/ |  __/ /    / / / |  __/ /__ / /___/ /_/ |  __/
/_/    \  __/_.___/\___/_/     \  / / \___/\___ /____/ \__,_/ \___/
        \/                      \/             \/                 

      Machine Learning Inference Systems Development Environment
EOF
    echo -e "${NC}"
    
    log_info "Starting development environment setup..."
    log_info "GPU Support: $GPU_SUPPORT"
    log_info "Include Jupyter: $INCLUDE_JUPYTER"
    log_info "Include TensorBoard: $INCLUDE_TENSORBOARD"
    
    # System checks
    check_system_requirements
    
    # Docker setup
    if [[ $GPU_ONLY != true ]]; then
        install_docker
        
        if [[ $GPU_SUPPORT == true ]]; then
            install_nvidia_docker
        fi
        
        # Note: User might need to log out/in for docker group changes
        if ! groups | grep -q docker; then
            log_warning "You need to log out and log back in for Docker group changes to take effect"
            log_warning "Or run: newgrp docker"
        fi
        
        build_development_image
        test_gpu_support
        create_development_aliases
        show_next_steps
    else
        if [[ $GPU_SUPPORT == true ]]; then
            install_nvidia_docker
        fi
        log_success "GPU-only setup completed"
    fi
}

# Run main function
main "$@"
