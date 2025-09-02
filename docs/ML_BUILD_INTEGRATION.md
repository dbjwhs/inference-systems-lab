# ML Framework Build Integration Guide

This document describes the comprehensive ML framework integration system that provides optional support for TensorRT and ONNX Runtime with graceful fallbacks.

## Overview

The build system now supports optional ML framework integration with automatic detection, explicit control, and graceful fallbacks when frameworks are unavailable. This ensures the project builds successfully in any environment while maximizing performance when ML frameworks are available.

## Configuration Options

### CMake Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `ENABLE_TENSORRT` | `ON`/`OFF`/`AUTO` | `AUTO` | Enable NVIDIA TensorRT GPU acceleration |
| `ENABLE_ONNX_RUNTIME` | `ON`/`OFF`/`AUTO` | `AUTO` | Enable ONNX Runtime cross-platform inference |

### Option Behavior

- **`AUTO`**: Automatically detect and enable if available, gracefully disable if not found
- **`ON`**: Explicitly require the framework - build fails if not found
- **`OFF`**: Explicitly disable the framework even if available

## Usage Examples

### Basic Configuration (Recommended)
```bash
# Auto-detect available frameworks
mkdir build && cd build
cmake .. -DENABLE_TENSORRT=AUTO -DENABLE_ONNX_RUNTIME=AUTO
make -j$(nproc)
```

### Explicit Framework Control
```bash
# Require both frameworks (build fails if either missing)
cmake .. -DENABLE_TENSORRT=ON -DENABLE_ONNX_RUNTIME=ON

# Disable all ML frameworks (CPU-only mode)
cmake .. -DENABLE_TENSORRT=OFF -DENABLE_ONNX_RUNTIME=OFF

# Mixed configuration
cmake .. -DENABLE_TENSORRT=OFF -DENABLE_ONNX_RUNTIME=AUTO
```

## Framework Detection

### TensorRT Detection

The build system searches for TensorRT in standard locations:

1. `$TENSORRT_ROOT` environment variable
2. `/usr/local/tensorrt`, `/opt/tensorrt`
3. System-wide installation paths

**Required components:**
- `NvInfer.h` header
- `libnvinfer`, `libnvinfer_plugin`, `libnvonnxparser` libraries
- CUDA Toolkit 11.8+

### ONNX Runtime Detection

The build system searches for ONNX Runtime in:

1. `$ONNXRUNTIME_ROOT` environment variable
2. `/usr/local/onnxruntime`, `/opt/onnxruntime` (Linux/macOS)
3. `C:\Program Files\onnxruntime` (Windows)

**Required components:**
- `onnxruntime_cxx_api.h` header
- `libonnxruntime` library

## Code Integration

### Compile-Time Detection

```cpp
#include "ml_config.hpp"

// Check framework availability at compile time
if constexpr (inference_lab::engines::ml::has_tensorrt) {
    // TensorRT-specific code
}

if constexpr (inference_lab::engines::ml::has_onnx_runtime) {
    // ONNX Runtime-specific code
}

if constexpr (inference_lab::engines::ml::has_ml_frameworks) {
    // General ML framework code
}
```

### Runtime Backend Selection

```cpp
#include "ml_config.hpp"

using namespace inference_lab::engines::ml;

// Get runtime capabilities
const auto& caps = capabilities;
std::cout << caps.to_string() << std::endl;

// Get available backends
auto backends = get_available_backends();

// Get optimal backend for current configuration
auto optimal = detect_optimal_backend();

// Check specific backend availability
bool can_use_gpu = caps.is_backend_available(MLBackend::TENSORRT_GPU);
```

### CMake Target Integration

```cmake
# Link ML frameworks to your target
target_link_libraries(your_target PRIVATE engines)

# The engines library automatically provides ML frameworks if available

# For explicit framework usage:
if(TARGET ML::TensorRT)
    target_link_libraries(your_target PRIVATE ML::TensorRT)
endif()

if(TARGET ML::ONNXRuntime) 
    target_link_libraries(your_target PRIVATE ML::ONNXRuntime)
endif()

# Or link all available frameworks:
if(TARGET ML::Frameworks)
    target_link_libraries(your_target PRIVATE ML::Frameworks)
endif()
```

## Build System Architecture

### Module Structure

```
cmake/
├── MLIntegration.cmake       # Main ML integration logic
├── TensorRT.cmake           # TensorRT detection and configuration
├── ONNXRuntime.cmake        # ONNX Runtime detection and configuration
└── PackageConfig.cmake      # Base package configuration

engines/src/
├── ml_config.hpp            # ML configuration API
├── ml_config.cpp            # Runtime capabilities
└── ...
```

### Integration Flow

1. **Validation**: Check option values are valid (`ON`/`OFF`/`AUTO`)
2. **Detection**: Attempt to find each requested framework
3. **Resolution**: Apply AUTO logic (enable if found, disable if not)
4. **Configuration**: Create CMake targets and compile definitions
5. **Summary**: Display configuration status

### Compile Definitions

| Definition | Value | Description |
|------------|-------|-------------|
| `ENABLE_TENSORRT` | `1`/undefined | TensorRT is available |
| `ENABLE_ONNX_RUNTIME` | `1`/undefined | ONNX Runtime is available |
| `ENABLE_ML_FRAMEWORKS` | `1`/`0` | Any ML framework available |
| `ML_FRAMEWORKS_COUNT` | `0-2` | Number of enabled frameworks |

## Installation Requirements

### TensorRT Installation

1. Download TensorRT 8.5+ from [NVIDIA Developer](https://developer.nvidia.com/tensorrt)
2. Install CUDA Toolkit 11.8+
3. Set environment variable: `export TENSORRT_ROOT=/path/to/tensorrt`

### ONNX Runtime Installation

1. Download from [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
2. Extract to standard location or set `ONNXRUNTIME_ROOT`
3. For GPU support, ensure CUDA/DirectML/CoreML is available

## Testing and Verification

### Demo Application

Run the ML framework detection demo to verify integration:

```bash
./engines/ml_framework_detection_demo
```

This will show:
- Compile-time detection results
- Runtime capabilities
- Available backends
- Configuration guidance
- Integration test results

### Build Verification

The configuration summary shows ML framework status:

```
=== Inference Systems Lab Configuration ===
...
Optional features:
  ML Frameworks (1):
    TensorRT 8.6.1
  TensorRT: ON
  ONNX Runtime: OFF
==========================================
```

## Troubleshooting

### Common Issues

**"TensorRT not found"**
- Install CUDA Toolkit first
- Set `TENSORRT_ROOT` environment variable
- Check library paths in `/etc/ld.so.conf.d/`

**"ONNX Runtime not found"**  
- Verify installation path
- Set `ONNXRUNTIME_ROOT` if needed
- Ensure correct platform binaries (x64, ARM64)

**"Explicit enable but not found"**
- Switch to `AUTO` for development
- Install missing dependencies
- Check system paths and permissions

### Debug Information

Enable verbose output to debug detection issues:

```bash
cmake .. -DCMAKE_FIND_DEBUG_MODE=ON
```

This will show detailed package search information.

## Performance Impact

### Framework Comparison

| Backend | Best For | Performance | Portability |
|---------|----------|-------------|-------------|
| CPU-only | Development, Testing | Baseline | High |
| ONNX Runtime | Cross-platform Production | 2-5x | High |
| TensorRT GPU | NVIDIA Production | 5-20x | Low |
| Hybrid | Enterprise Deployment | Optimal | Medium |

### Memory Usage

- **CPU-only**: Minimal overhead
- **ONNX Runtime**: +50-100MB runtime
- **TensorRT**: +200-500MB GPU memory
- **Hybrid**: Combined overhead

## Future Enhancements

Planned improvements for the ML build integration:

1. **Additional Backends**: OpenVINO, DirectML, Metal Performance Shaders
2. **Model Format Support**: Native PyTorch, JAX, TensorFlow Lite
3. **Hardware Detection**: Automatic GPU capability detection
4. **Performance Profiling**: Integrated benchmark comparison
5. **Cloud Integration**: Support for cloud ML services

This build system provides a solid foundation for these future enhancements while maintaining backward compatibility and graceful degradation.
