# Docker Development Environment for Inference Systems Lab

## Purpose

This directory contains Docker configuration files that provide a comprehensive CUDA-enabled development environment for ML inference systems. These files enable developers to work with GPU-accelerated inference engines without needing to install complex dependencies locally.

## Why Docker for This Project?

The Inference Systems Lab is evolving from a pure C++ rule-based inference system into a hybrid platform that integrates modern ML inference engines. This requires:

1. **GPU Infrastructure**: CUDA 12.3, cuDNN, and GPU drivers for acceleration
2. **ML Inference Runtimes**: TensorRT 8.6 for NVIDIA GPU optimization, ONNX Runtime 1.16 for cross-platform support
3. **Python ML Stack**: PyTorch, TensorFlow, and other ML libraries for model development
4. **Development Tools**: Jupyter Lab for experimentation, TensorBoard for visualization
5. **Consistent Environment**: Reproducible builds across different developer machines

## Files in This Directory

### `Dockerfile.dev`
Multi-stage Docker image that builds a complete ML development environment:
- **Base**: NVIDIA CUDA 12.3 development image with cuDNN
- **Toolchain**: GCC 11, CMake 3.24+, Ninja, clang-format, clang-tidy
- **ML Libraries**: PyTorch, ONNX, TensorRT, ONNX Runtime with GPU support
- **Development Tools**: Jupyter, pytest, debugging tools (gdb, valgrind)
- **C++ Dependencies**: Cap'n Proto, Google Test, Google Benchmark

### `docker-compose.dev.yml`
Orchestrates multiple services for the development workflow:
- **dev**: Main development container with full ML stack and GPU access
- **jupyter**: Optional Jupyter Lab service for interactive ML development
- **tensorboard**: Optional TensorBoard service for training visualization

### `.dockerignore`
Prevents unnecessary files from being included in the Docker build context:
- Build artifacts (build/, cmake-build-*/)
- Version control (.git/)
- IDE files (.vscode/, .idea/)
- Large data files (data/, models/, datasets/)

## Usage

### Quick Start

1. **Build and start the development environment:**
```bash
docker-compose -f docker-dev/docker-compose.dev.yml up -d
```

2. **Enter the development container:**
```bash
docker-compose -f docker-dev/docker-compose.dev.yml exec dev bash
```

3. **Build the project inside the container:**
```bash
source cmake_config.sh
cmake_debug
cmake --build build/debug
```

### GPU Development

The containers are configured for GPU passthrough, enabling CUDA development:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Optional Services

**Start Jupyter Lab for ML experimentation:**
```bash
docker-compose -f docker-dev/docker-compose.dev.yml --profile jupyter up -d jupyter
# Access at http://localhost:8889 with token: inference-lab-dev
```

**Start TensorBoard for visualization:**
```bash
docker-compose -f docker-dev/docker-compose.dev.yml --profile tensorboard up -d tensorboard
# Access at http://localhost:6007
```

## Integration with Project Roadmap

These Docker files support the ML integration phases outlined in the main project:

### Phase 1: TensorRT Foundation
- CUDA 12.3 and TensorRT 8.6 pre-installed
- GPU acceleration immediately available
- Example: `engines/src/tensorrt/` development

### Phase 2: ONNX Runtime Cross-Platform
- ONNX Runtime 1.16 with GPU providers
- Python bindings for prototyping
- Example: `engines/src/onnx/` implementation

### Phase 3: Unified Inference Architecture
- Complete environment for hybrid neural-symbolic systems
- All dependencies for both rule-based and ML inference
- Integrated testing and benchmarking tools

## Volume Mappings

The Docker environment uses several persistent volumes:

- **Source Code**: Project root mounted at `/workspace`
- **Data Storage**: `inference_lab_data` volume at `/data`
- **Model Storage**: `inference_lab_models` volume at `/models`
- **Build Cache**: `inference_lab_build` for faster rebuilds

## Development Workflow

1. **Local Development**: Edit code on your host machine
2. **Container Building**: Compile and test inside the container
3. **GPU Testing**: Run CUDA/TensorRT code with GPU access
4. **Jupyter Experimentation**: Prototype ML models interactively
5. **Performance Analysis**: Use TensorBoard for profiling

## Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA 12.3 support (optional but recommended)

## Troubleshooting

### GPU Not Detected
Ensure NVIDIA Docker runtime is installed:
```bash
docker run --rm --gpus all nvidia/cuda:12.3-base nvidia-smi
```

### Permission Issues
The container runs as user `developer` (UID 1000). Adjust if needed:
```yaml
RUN useradd -m -s /bin/bash -u $(id -u) developer
```

### Build Context Too Large
Check `.dockerignore` is properly excluding large directories like `build/`, `data/`, and `models/`.

## Future Enhancements

- Add distributed training support (Horovod, DeepSpeed)
- Include model serving frameworks (Triton Inference Server)
- Add MLOps tools (MLflow, Weights & Biases)
- Support for additional accelerators (AMD ROCm, Intel oneAPI)