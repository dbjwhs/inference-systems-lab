# Docker Development Environment Guide

This document provides comprehensive guidance for using the Docker-based development environment for the Inference Systems Lab project.

## Overview

The Docker development environment provides a complete, reproducible setup for ML inference systems development with:

- **NVIDIA CUDA 12.3** with cuDNN support
- **TensorRT 8.6** for GPU inference optimization
- **ONNX Runtime 1.16** with GPU providers
- **Modern C++17 toolchain** (GCC 11, CMake 3.24+)
- **Python 3.10** with comprehensive ML libraries
- **Development tools** (clang-format, clang-tidy, debuggers)

## Quick Start

### 1. Automated Setup

The easiest way to get started is using the automated setup script:

```bash
# Make setup script executable
chmod +x scripts/setup_dev_environment.sh

# Full setup with GPU support
./scripts/setup_dev_environment.sh

# CPU-only development (no GPU required)
./scripts/setup_dev_environment.sh --no-gpu

# Include additional services
./scripts/setup_dev_environment.sh --jupyter --tensorboard
```

### 2. Manual Setup

If you prefer manual setup or need custom configuration:

```bash
# Build the development image
docker build -f Dockerfile.dev -t inference-lab:dev .

# Start the development environment
docker-compose -f docker-compose.dev.yml up -d dev

# Connect to the development shell
docker-compose -f docker-compose.dev.yml exec dev bash
```

### 3. Quick Development Commands

After setup, use these convenient scripts:

```bash
./dev_start.sh    # Start development environment
./dev_shell.sh    # Connect to development shell
./dev_stop.sh     # Stop development environment
```

## Development Workflow

### Inside the Container

Once connected to the development shell:

```bash
# Source CMake helper functions
source cmake_config.sh

# Configure project (debug build with CUDA/TensorRT)
cmake_debug

# Build the project
cmake --build build/debug --parallel

# Run tests
cd build/debug && ctest --output-on-failure

# Run benchmarks
./common/container_benchmarks

# Format code
clang-format -i common/src/*.hpp
python3 tools/check_format.py --fix

# Static analysis
python3 tools/check_static_analysis.py --check
```

### Available Build Configurations

The development environment provides several pre-configured build types:

#### Debug Build with GPU Support
```bash
cmake_debug
# Equivalent to:
cmake -B build/debug -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DENABLE_TENSORRT=ON \
    -DENABLE_ONNX_RUNTIME=ON \
    -DENABLE_CUDA=ON
```

#### Release Build (Optimized)
```bash
cmake_release
# Optimized for performance with GPU support
```

#### Sanitizer Build (Development/Testing)
```bash
cmake_sanitizers
# Debug build with AddressSanitizer and UBSan
# GPU disabled for compatibility with sanitizers
```

## Container Architecture

### Multi-Stage Build

The Docker image uses a multi-stage build process:

1. **cuda_base**: CUDA 12.3 base with system dependencies
2. **python_ml**: Python ML libraries (PyTorch, ONNX, etc.)
3. **capnproto_build**: Cap'n Proto and Google libraries
4. **tensorrt_install**: TensorRT 8.6 installation
5. **dev_environment**: Development tools and user setup
6. **final**: Complete development environment

### Container Features

- **Non-root user**: `developer` user for security
- **GPU access**: Full CUDA and TensorRT support
- **Volume persistence**: Code, data, and build cache persistence
- **Port forwarding**: Jupyter (8888), TensorBoard (6006), dev servers
- **Development tools**: Complete C++ and Python toolchain

## Volume Management

The development environment uses several Docker volumes for persistence:

| Volume | Purpose | Location |
|--------|---------|----------|
| `inference_lab_data` | Persistent data storage | `/data` |
| `inference_lab_models` | ML model files | `/models` |
| `inference_lab_build` | Build cache | `/workspace/build` |
| `jupyter_notebooks` | Jupyter notebooks | `/workspace/notebooks` |
| `tensorboard_logs` | TensorBoard logs | `/workspace/logs` |

### Volume Commands

```bash
# List volumes
docker volume ls | grep inference-lab

# Inspect volume
docker volume inspect inference-lab-data

# Backup volume
docker run --rm -v inference-lab-data:/data -v $(pwd):/backup ubuntu tar czf /backup/data-backup.tar.gz /data

# Restore volume
docker run --rm -v inference-lab-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/data-backup.tar.gz -C /
```

## GPU Development

### GPU Requirements

- **NVIDIA GPU**: Compute Capability 7.0+ (RTX 20-series or newer)
- **CUDA Driver**: 525.60.13+ for CUDA 12.3
- **Docker**: NVIDIA Container Toolkit installed
- **Memory**: Sufficient GPU memory for models and batch processing

### Testing GPU Support

```bash
# Test CUDA availability
docker run --rm --gpus all inference-lab:dev nvidia-smi

# Test PyTorch CUDA
docker run --rm --gpus all inference-lab:dev python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Test TensorRT
docker run --rm --gpus all inference-lab:dev python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

### GPU Memory Management

Monitor GPU memory usage during development:

```bash
# In container - monitor GPU usage
watch -n 1 nvidia-smi

# Get detailed GPU info
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
```

## Service Configuration

### Jupyter Lab

Start Jupyter Lab for interactive development:

```bash
# Start Jupyter service
docker-compose -f docker-compose.dev.yml --profile jupyter up -d

# Access at: http://localhost:8889
# Token: inference-lab-dev
```

Features:
- **GPU support**: Full CUDA and PyTorch GPU access
- **Persistence**: Notebooks saved in `jupyter_notebooks` volume
- **Extensions**: Pre-installed extensions for ML development

### TensorBoard

Monitor training and visualize model performance:

```bash
# Start TensorBoard service
docker-compose -f docker-compose.dev.yml --profile tensorboard up -d

# Access at: http://localhost:6007
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Accessible

**Problem**: `nvidia-smi` not found or GPU not visible in container

**Solutions**:
```bash
# Check NVIDIA drivers on host
nvidia-smi

# Install/update NVIDIA Container Toolkit
sudo apt update && sudo apt install nvidia-container-toolkit
sudo systemctl restart docker

# Verify Docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.3-base-ubuntu22.04 nvidia-smi
```

#### 2. Permission Issues

**Problem**: Permission denied accessing files or Docker socket

**Solutions**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker  # Or logout/login

# Fix file permissions
sudo chown -R $USER:$USER .
```

#### 3. Build Failures

**Problem**: Docker build fails with dependency errors

**Solutions**:
```bash
# Clean build with no cache
docker build --no-cache -f Dockerfile.dev -t inference-lab:dev .

# Check available disk space
df -h

# Clean Docker system
docker system prune -a
```

#### 4. Port Conflicts

**Problem**: Port already in use errors

**Solutions**:
```bash
# Check what's using the port
sudo netstat -tulpn | grep :8888

# Kill process using port
sudo kill -9 $(sudo lsof -t -i:8888)

# Or use different ports in docker-compose.dev.yml
```

### Performance Optimization

#### 1. Build Performance

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build with more parallel jobs
docker build --build-arg MAKEFLAGS=-j$(nproc) -f Dockerfile.dev -t inference-lab:dev .
```

#### 2. Container Performance

```bash
# Increase shared memory for large models
docker run --shm-size=2g --gpus all inference-lab:dev

# Or add to docker-compose.dev.yml:
services:
  dev:
    shm_size: '2gb'
```

#### 3. GPU Memory Optimization

```bash
# Monitor GPU memory
nvidia-smi dmon -s m -i 0

# Set memory growth for TensorFlow/PyTorch
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Advanced Configuration

### Custom CUDA Versions

To use a different CUDA version, modify the Dockerfile.dev base image:

```dockerfile
# Change base image
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as cuda_base

# Update TensorRT version accordingly
# TensorRT compatibility matrix: https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/
```

### Additional ML Libraries

Add custom libraries to the Python ML stage:

```dockerfile
# In python_ml stage, add:
RUN pip3 install \
    transformers>=4.21.0 \
    accelerate>=0.20.0 \
    your-custom-library>=1.0.0
```

### Development Tools

Customize development tools in the dev_environment stage:

```dockerfile
# Add additional tools
RUN apt-get install -y \
    your-favorite-editor \
    custom-debugging-tools \
    profiling-utilities
```

## Integration with IDEs

### Visual Studio Code

Use VS Code with the Remote-Containers extension:

1. Install the Remote-Containers extension
2. Open project folder in VS Code
3. Command Palette: "Remote-Containers: Reopen in Container"
4. Select the Dockerfile.dev configuration

### CLion/IntelliJ

Configure CLion for remote development:

1. File → Settings → Build, Execution, Deployment → Toolchains
2. Add Docker toolchain pointing to the development container
3. Configure CMake to use the Docker toolchain

### Vim/Neovim

The container includes Vim with basic configuration. For advanced setup:

```bash
# In container, install your preferred Vim configuration
git clone https://github.com/your-username/vim-config.git ~/.vim
```

## Security Considerations

### Container Security

- **Non-root user**: Development runs as `developer` user
- **Isolated networking**: Custom Docker network
- **Volume restrictions**: Limited host filesystem access
- **Capability restrictions**: No unnecessary privileges

### Host Security

```bash
# Run with restricted capabilities
docker run --rm --gpus all --cap-drop=ALL --cap-add=SYS_NICE inference-lab:dev

# Use read-only root filesystem
docker run --rm --gpus all --read-only --tmpfs /tmp inference-lab:dev
```

## Production Deployment

The development environment can be adapted for production:

### Multi-stage Production Build

```dockerfile
FROM inference-lab:dev as production

# Remove development tools
RUN apt-get purge -y \
    gdb valgrind strace \
    vim nano \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Add production user
RUN useradd -m -s /bin/bash -u 1001 production
USER production

# Set production environment
ENV ENVIRONMENT=production
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-lab-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference-lab-api
  template:
    metadata:
      labels:
        app: inference-lab-api
    spec:
      containers:
      - name: api
        image: inference-lab:production
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
```

This comprehensive Docker development environment provides everything needed for advanced ML inference systems development while maintaining reproducibility and ease of use.