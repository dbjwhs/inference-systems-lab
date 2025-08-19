# Dependencies Guide - Inference Systems Laboratory

This document provides comprehensive setup instructions for all dependencies required by the Inference Systems Laboratory, including traditional C++ development tools and optional ML inference frameworks.

## Core Dependencies (Required)

### C++ Development Environment

#### Compiler Requirements
- **GCC 10+** or **Clang 12+** or **MSVC 2019+** with C++17 support
- **CMake 3.20+** for build system management
- **Git** for version control

#### Development Tools
- **clang-format** - Code formatting (automatically detected by CMake)
- **clang-tidy** - Static analysis (automatically detected by CMake)
- **Python 3.8+** - Required for development tooling scripts

#### Testing & Quality Assurance
- **GoogleTest** - Unit testing framework (automatically fetched by CMake)
- **Google Benchmark** - Performance benchmarking (automatically fetched by CMake)
- **AddressSanitizer & UBSan** - Memory safety validation (built into compilers)

### Core Libraries
- **Cap'n Proto** - Serialization framework with schema evolution support
- **Standard C++17 libraries** - No additional runtime dependencies for core functionality

---

## ML Integration Dependencies (Optional)

The inference laboratory supports optional integration with modern machine learning inference frameworks. These dependencies are only required when building with ML features enabled.

### TensorRT (NVIDIA GPU Acceleration)

#### System Requirements
- **NVIDIA GPU** with Compute Capability 7.0+ (RTX 20-series or newer recommended)
- **CUDA Toolkit 11.8+** with cuDNN support
- **TensorRT 8.5+** inference runtime

#### Installation Guide

**Step 1: CUDA Toolkit Setup**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Verify installation
nvcc --version
nvidia-smi
```

**Step 2: TensorRT Installation**
```bash
# Download TensorRT from NVIDIA Developer Portal
# Extract to /opt/tensorrt or preferred location
export TENSORRT_ROOT=/opt/tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH

# Add to ~/.bashrc for persistence
echo 'export TENSORRT_ROOT=/opt/tensorrt' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Step 3: Verification**
```bash
# Test TensorRT installation
trtexec --help
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

### ONNX Runtime (Cross-Platform ML)

#### System Requirements
- **Cross-platform support**: Linux, macOS, Windows
- **Optional accelerators**: CUDA, DirectML, OpenVINO, CoreML
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large models)

#### Installation Options

**Option 1: Package Manager (Recommended)**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libonnxruntime-dev

# macOS with Homebrew
brew install onnxruntime

# Verify installation
pkg-config --modversion onnxruntime
```

**Option 2: Build from Source (Advanced)**
```bash
# Clone and build ONNX Runtime
git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime

# CPU-only build
./build.sh --config Release --build_shared_lib --parallel

# GPU-enabled build (requires CUDA)
./build.sh --config Release --build_shared_lib --parallel --use_cuda --cuda_home /usr/local/cuda

# Install built libraries
sudo cp build/Linux/Release/libonnxruntime.so.* /usr/local/lib/
sudo cp -r include/onnxruntime /usr/local/include/
sudo ldconfig
```

**Step 3: Verification**
```bash
# Test ONNX Runtime installation
python3 -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')"
python3 -c "import onnxruntime; print(f'Available providers: {onnxruntime.get_available_providers()}')"
```

---

## System Configuration Recommendations

### Minimal System Configuration

**Development Machine (Basic ML Support)**
- **CPU**: Intel i5-8400 / AMD Ryzen 5 3600 (6+ cores, 3.0+ GHz)
- **RAM**: 16GB DDR4 (minimum for ONNX Runtime + development)
- **Storage**: 256GB SSD (fast builds and model storage)
- **GPU**: NVIDIA RTX 3060 (8GB VRAM) for basic TensorRT testing
- **OS**: Ubuntu 20.04+ / macOS 12+ / Windows 10 Pro

**Build Performance**: ~5-10 minutes full builds, handles small-medium models

**Suitable for**:
- Learning TensorRT/ONNX integration
- Development and testing with lightweight models
- Single-threaded inference workloads
- Basic benchmarking and validation

### Sweet Spot System Configuration (RECOMMENDED)

**Optimized ML Development Build (~$1,250)**
- **CPU**: AMD Ryzen 5 7600 (~$230) - 6 cores/12 threads, excellent performance
- **GPU**: RTX 4060 Ti 16GB (~$500) - 16GB VRAM crucial for larger models  
- **RAM**: 32GB DDR5-5600 (~$150) - ML workloads are memory hungry
- **Storage**: 1TB NVMe SSD (~$80) - Fast model loading and compilation
- **Motherboard**: B650 chipset (~$130) - AM5 future upgrade path
- **PSU**: 750W 80+ Gold (~$100) - Headroom for sustained ML workloads
- **Case**: Mid-tower with good airflow (~$60)

**Build Performance**: ~2-3 minutes full builds, handles 7B+ parameter models

**Why These Choices**:
- **16GB VRAM**: Handles larger models that 12GB cards cannot load
- **6-core CPU**: Perfect for ML development without overpaying for unused cores
- **32GB RAM**: Prevents bottlenecks when loading large datasets and development tools
- **AM5 Platform**: Future upgrade path to next-generation CPUs
- **750W PSU**: Sustained TensorRT workloads + future GPU upgrade headroom

**💰 Cost-Benefit Analysis**:
- **vs Cloud**: At $1,250 total cost, pays for itself in 3-4 months vs sustained cloud usage
- No "oops I left it running" bills or surprise monthly charges
- Full control over development environment and model storage
- Dedicated hardware available 24/7 for experimentation

**Suitable for**:
- Serious TensorRT/ONNX development with larger models
- Production model optimization and testing
- Local neural-symbolic reasoning research
- Cost-effective alternative to cloud-based development

### Top-of-the-Line System Configuration

**Research/Production Machine (Enterprise ML Support)**
- **CPU**: Intel i9-13900K / AMD Ryzen 9 7950X (16+ cores, 4.0+ GHz base)
- **RAM**: 64GB DDR5-5600 (large model loading + parallel development)
- **Storage**: 2TB NVMe SSD (PCIe 4.0, models + fast compilation)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or RTX 6000 Ada (48GB VRAM)
- **Alternative**: Multiple RTX 4080 for distributed inference testing
- **OS**: Ubuntu 22.04 LTS with latest CUDA drivers

**Build Performance**: ~1-2 minutes full builds, handles largest production models

**Suitable for**:
- Production model optimization and deployment
- Large language model inference (7B+ parameters)
- Multi-GPU distributed inference development
- High-throughput benchmarking and performance analysis
- Neural-symbolic hybrid system research

### Cloud Development Alternative

**AWS/GCP/Azure Instance Recommendations**:
- **Minimal**: `g4dn.xlarge` (4 vCPU, 16GB RAM, T4 GPU) - $0.50-1.00/hour
- **Optimal**: `p3.2xlarge` (8 vCPU, 61GB RAM, V100 GPU) - $3.00-4.00/hour
- **Maximum**: `p4d.24xlarge` (96 vCPU, 1.1TB RAM, 8x A100 GPUs) - $30.00+/hour

**Benefits**: No hardware investment, latest drivers, scalable for large experiments

---

## Platform-Specific Setup

### Ubuntu/Debian Linux

```bash
# Core development tools
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    clang-format \
    clang-tidy \
    pkg-config

# Optional: ML dependencies
sudo apt install -y \
    libonnxruntime-dev \
    python3-onnx \
    python3-numpy

# Verify C++17 support
g++ --version  # Should be 10+
cmake --version  # Should be 3.20+
```

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Core development tools
brew install cmake git python@3.9 llvm

# Optional: ML dependencies
brew install onnxruntime

# Verify installation
clang++ --version  # Should support C++17
cmake --version    # Should be 3.20+
```

### Windows

```powershell
# Install Visual Studio 2019+ with C++ workload
# Install CMake from cmake.org
# Install Git from git-scm.com
# Install Python 3.8+ from python.org

# Verify C++17 support
cl  # Should be MSVC 2019+
cmake --version  # Should be 3.20+

# Optional: Install CUDA and TensorRT for GPU acceleration
# Download from NVIDIA Developer Portal
```

---

## Build Configuration

### CMake Configuration Options

The build system automatically detects available dependencies and enables features accordingly:

```bash
# Basic build (core functionality only)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Development build with all quality tools
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DSANITIZER_TYPE=address+undefined \
         -DENABLE_STATIC_ANALYSIS=ON

# ML-enabled build (when dependencies are available)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_TENSORRT=ON \
         -DENABLE_ONNX_RUNTIME=ON \
         -DTENSORRT_ROOT=/opt/tensorrt

# Cross-compilation support
cmake .. -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake
```

### Dependency Detection

The CMake system provides detailed feedback about dependency availability:

```bash
# Check which features will be built
cmake .. -DCMAKE_BUILD_TYPE=Release
# Output shows:
# -- TensorRT: FOUND (version 8.5.1)
# -- ONNX Runtime: FOUND (version 1.15.0)
# -- CUDA: FOUND (version 11.8)
# -- Features enabled: TensorRT_GPU, ONNX_RUNTIME, BENCHMARKING
```

---

## Troubleshooting

### Common Issues

**TensorRT linking errors:**
```bash
# Ensure library path is correctly set
export LD_LIBRARY_PATH=/opt/tensorrt/lib:$LD_LIBRARY_PATH
sudo ldconfig

# Verify shared libraries can be found
ldd build/engines/tensorrt_engine_test
```

**ONNX Runtime version conflicts:**
```bash
# Remove conflicting versions
sudo apt remove libonnxruntime-dev
pip3 uninstall onnxruntime

# Clean reinstall
sudo apt install libonnxruntime-dev
```

**CUDA driver/toolkit mismatch:**
```bash
# Check driver version
nvidia-smi

# Check CUDA toolkit version
nvcc --version

# Ensure compatibility (driver >= toolkit version)
```

### Performance Validation

After successful installation, validate performance:

```bash
# Build and run ML benchmarks
cd build
make -j$(nproc)
./engines/tensorrt_benchmarks  # Should show GPU acceleration
./engines/onnx_benchmarks      # Should show cross-platform execution

# Compare performance across backends
python3 ../tools/run_benchmarks.py --compare-ml-backends
```

---

## Version Compatibility Matrix

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|-------------------|-------|
| C++ Compiler | GCC 10, Clang 12, MSVC 2019 | Latest stable | C++17 support required |
| CMake | 3.20 | 3.25+ | Modern CMake features |
| CUDA Toolkit | 11.8 | 12.0+ | TensorRT compatibility |
| TensorRT | 8.5 | 8.6+ | GPU inference optimization |
| ONNX Runtime | 1.15 | 1.16+ | Cross-platform ML support |
| Python | 3.8 | 3.10+ | Development tooling |

---

## Next Steps

After dependency setup:

1. **Verify Installation**: Run `cmake .. && make -j$(nproc) && ctest`
2. **Review Documentation**: Study [DEVELOPMENT.md](DEVELOPMENT.md) for ML workflow
3. **Explore Examples**: Check [README.md](../README.md) for learning path
4. **Development Setup**: Install pre-commit hooks with `python3 tools/install_hooks.py --install`

For questions or issues, refer to the [troubleshooting guide](../README.md#troubleshooting) or check the project's documentation.