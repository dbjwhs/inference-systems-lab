# ONNX Runtime Integration Examples

This document describes the comprehensive ONNX Runtime demonstration applications that showcase cross-platform ML inference capabilities integrated into the Inference Systems Laboratory.

## Overview

The ONNX Runtime integration provides universal model format support with dynamic backend switching, enabling high-performance ML inference across different hardware platforms. The examples demonstrate real-world usage patterns from simple inference to production-grade model serving.

## Example Applications

### 1. ONNX Inference Demo (`onnx_inference_demo`)

**Purpose**: Demonstrates basic ONNX Runtime integration with model loading, inference execution, and performance analysis.

**Features**:
- 🔹 Automatic ML framework detection and capability reporting
- 🔹 Dynamic execution provider selection (CPU, CUDA, TensorRT, etc.)
- 🔹 Model inspection (input/output tensors, shapes, data types)
- 🔹 Synthetic data generation for testing
- 🔹 Performance benchmarking with statistical analysis
- 🔹 Provider comparison and switching

**Usage**:
```bash
# Framework detection mode (no model required)
./onnx_inference_demo

# Model inference mode
./onnx_inference_demo path/to/model.onnx
```

**Example Output**:
```
=== ONNX Runtime Inference Demo ===

ML Framework Status:
  ML Capabilities: 1 framework(s) - ONNX Runtime
✅ ONNX Runtime is available!

Available ML Backends:
  - CPU-only
  - ONNX Runtime
Optimal backend: ONNX Runtime

Model Information:
  Input tensors:
    [0] input: [1, 3, 224, 224] float32
  Output tensors:
    [0] output: [1, 1000] float32

✅ Inference completed successfully
Output tensors: 1
  Output 0 (output): shape [1, 1000]
    First 5 values: [0.1234, -0.5678, 0.9012, -0.3456, 0.7890, ...]

Average inference time: 15.32 µs
Throughput: 65,261.78 inferences/sec
```

### 2. ONNX Model Server Demo (`onnx_model_server_demo`)

**Purpose**: Production-ready multi-threaded model serving architecture with concurrent request processing and real-time monitoring.

**Features**:
- 🔹 Multi-threaded inference engine (configurable worker threads)
- 🔹 Concurrent request queue management
- 🔹 Multiple model support with automatic load balancing
- 🔹 Real-time performance monitoring and statistics
- 🔹 Load generation and stress testing capabilities
- 🔹 Production-grade error handling and recovery
- 🔹 Comprehensive metrics collection

**Architecture**:
```
Client Requests → Request Queue → Worker Threads → ONNX Engines
                                        ↓
Statistics Monitor ← Performance Metrics ← Inference Results
```

**Usage**:
```bash
# Simulation mode (demonstrates architecture without models)
./onnx_model_server_demo

# Production mode with actual models
./onnx_model_server_demo model1.onnx model2.onnx model3.onnx
```

**Key Components**:

1. **ONNXModelServer**: Thread-safe model serving infrastructure
   - Manages multiple ONNX Runtime engines
   - Handles concurrent request processing
   - Provides load balancing across worker threads

2. **LoadGenerator**: Synthetic load generation for testing
   - Configurable request rate (QPS)
   - Realistic request patterns
   - Stress testing capabilities

3. **StatisticsMonitor**: Real-time performance monitoring
   - Success/failure rates
   - Latency statistics (mean, p95, p99)
   - Throughput monitoring
   - Memory usage tracking

**Sample Statistics Output**:
```
=== Server Statistics ===
Total requests: 2847
Successful: 2847 (100.0%)
Failed: 0 (0.0%)
Average processing time: 1,234.56 µs
Success rate: 100.0%
========================
```

### 3. ML Framework Benchmark (`ml_framework_benchmark`)

**Purpose**: Comprehensive performance comparison tool for different ML frameworks, execution providers, and model configurations.

**Features**:
- 🔹 Multi-provider benchmarking (CPU, CUDA, TensorRT, etc.)
- 🔹 Statistical analysis with percentiles (P95, P99)
- 🔹 Memory usage monitoring
- 🔹 CSV export for analysis and reporting
- 🔹 Model comparison across different architectures
- 🔹 Throughput and latency measurement
- 🔹 Success rate tracking and error analysis

**Metrics Collected**:
- **Latency**: Mean, median, min, max, standard deviation
- **Percentiles**: P95, P99 for tail latency analysis
- **Throughput**: Queries per second (QPS)
- **Memory**: Usage and peak memory consumption
- **Reliability**: Success rates and error statistics

**Usage**:
```bash
# Demonstration mode
./ml_framework_benchmark --help

# Single model benchmark
./ml_framework_benchmark model.onnx --iterations 1000

# Multiple model comparison
./ml_framework_benchmark model1.onnx model2.onnx --verbose

# Custom configuration
./ml_framework_benchmark resnet50.onnx efficientnet.onnx --iterations 500 --verbose
```

**Sample Output**:
```
=== Benchmark Summary ===

Model: Model_1
  🏆 Best: CPU (1234.56 µs, 810.23 QPS)
  All results:
    CPU         : 1234.56 µs ( 810.23 QPS) [1.0x]
    CUDA        : 2345.67 µs ( 426.31 QPS) [1.9x]

Overall Statistics:
  Total configurations tested: 4
  Average latency: 1834.32 µs
  Combined throughput: 1236.54 QPS
```

**CSV Export Format**:
```csv
Framework,Provider,Model,Iterations,MeanUS,MedianUS,MinUS,MaxUS,StdDevUS,P95US,P99US,ThroughputQPS,MemoryMB,PeakMemoryMB,SuccessfulRuns,FailedRuns,SuccessRate
ONNX Runtime,CPU,Model_1,1000,1234.56,1200.34,890.12,2345.67,234.56,1890.23,2100.45,810.23,128,256,1000,0,100.0
```

## Integration with Existing Systems

### ML Configuration Integration

All examples automatically detect available ML frameworks and backends using the integrated `ml_config` system:

```cpp
#include "../src/ml_config.hpp"

// Check framework availability
const auto& caps = capabilities;
if (caps.onnx_runtime_available) {
    // ONNX Runtime specific code
}

// Get optimal backend
auto optimal = detect_optimal_backend();
```

### Build System Integration

The examples are fully integrated with the project's CMake build system and ML framework detection:

```cmake
# Examples are built conditionally based on available frameworks
if(TARGET ML::Frameworks)
    # ONNX Runtime examples available
endif()
```

### Error Handling Patterns

All examples follow the project's `Result<T,E>` error handling pattern:

```cpp
auto engine_result = create_onnx_engine_from_model(model_path);
if (!engine_result) {
    // Handle error case
    return;
}

auto& engine = *engine_result.value();
auto inference_result = engine.run_inference(inputs);
```

## Performance Characteristics

### Expected Performance Ranges

| Provider | Typical Latency | Throughput | Memory Usage | Use Case |
|----------|----------------|------------|--------------|----------|
| CPU | 1-10ms | 100-1K QPS | 100-500MB | Development, Testing |
| CUDA | 0.1-5ms | 200-5K QPS | 500MB-2GB | GPU-accelerated production |
| TensorRT | 0.05-2ms | 500-10K QPS | 300MB-1GB | Optimized NVIDIA deployment |

### Optimization Recommendations

1. **Model Optimization**: Use ONNX Runtime's graph optimization levels
2. **Provider Selection**: Choose providers based on available hardware
3. **Batch Processing**: Increase batch size for higher throughput
4. **Memory Management**: Enable memory arena for better allocation patterns
5. **Threading**: Configure intra/inter-op thread counts for optimal CPU usage

## Production Deployment Patterns

### Model Server Architecture

```
Load Balancer
    ↓
Multiple Model Server Instances
    ↓
ONNX Runtime Engines (CPU/GPU)
    ↓
Hardware Resources
```

### Monitoring and Observability

- **Metrics**: Latency, throughput, error rates, resource utilization
- **Logging**: Request/response logging, error tracking, performance logs  
- **Alerting**: SLA violations, error rate thresholds, resource limits
- **Profiling**: ONNX Runtime profiling integration for optimization

### Scaling Strategies

1. **Vertical Scaling**: Increase worker threads, memory, CPU cores
2. **Horizontal Scaling**: Multiple server instances with load balancing
3. **Hardware Acceleration**: GPU deployment for compute-intensive models
4. **Model Optimization**: TensorRT conversion for NVIDIA hardware

## Development Workflow

### Testing New Models

1. Use `onnx_inference_demo` to verify model compatibility
2. Run `ml_framework_benchmark` to compare providers
3. Test with `onnx_model_server_demo` for production readiness
4. Analyze CSV results for optimization opportunities

### Performance Optimization

1. **Profile**: Use ONNX Runtime profiling to identify bottlenecks
2. **Compare**: Benchmark different execution providers
3. **Optimize**: Apply model-specific optimizations
4. **Validate**: Verify improvements with statistical significance

### Integration Testing

1. **Unit Tests**: Individual component functionality
2. **Load Tests**: Concurrent request handling
3. **Stress Tests**: Resource limit testing
4. **End-to-End**: Complete workflow validation

## Future Enhancements

### Planned Features

1. **Additional Providers**: OpenVINO, DirectML, Metal Performance Shaders
2. **Model Formats**: PyTorch, TensorFlow, JAX model support
3. **Dynamic Batching**: Automatic batch size optimization
4. **Model Caching**: Intelligent model loading and caching
5. **Distributed Inference**: Multi-node model serving
6. **A/B Testing**: Model version comparison and rollout

### Research Directions

1. **Adaptive Optimization**: Runtime performance tuning
2. **Resource Scheduling**: Intelligent provider selection
3. **Model Compression**: Runtime quantization and pruning
4. **Edge Deployment**: Optimized mobile and embedded deployment

This ONNX Runtime integration provides a comprehensive foundation for production ML inference systems with enterprise-grade reliability, performance, and observability.
