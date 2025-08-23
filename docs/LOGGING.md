# Logger Class - Technical Reference

**Inference Systems Laboratory**  
**Version**: 1.0.0  
**API Level**: Enterprise  
**Thread Safety**: Full  
**C++ Standard**: C++17+  

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Design](#architecture--design)
3. [Core Logging API](#core-logging-api)
4. [ML-Specific Logging Extensions](#ml-specific-logging-extensions)
5. [Performance & Threading](#performance--threading)
6. [Configuration & Control](#configuration--control)
7. [Integration Patterns](#integration-patterns)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)
10. [Examples](#examples)

---

## Overview

The `Logger` class provides a high-performance, thread-safe logging infrastructure specifically designed for ML inference systems and enterprise C++ applications. It combines traditional structured logging with ML-specific telemetry, performance metrics tracking, and model lifecycle management.

### Key Features

- **Thread-Safe Singleton Architecture**: Lock-free reads, minimal contention writes
- **Structured ML Logging**: Model context tracking, inference metrics, error correlation  
- **Performance Optimized**: Atomic operations, dedicated mutexes, zero-cost abstractions
- **Enterprise Grade**: Comprehensive error handling, configurable output, production monitoring
- **Modern C++17**: Template-based formatting, RAII patterns, type-safe APIs

### Design Principles

1. **Zero Runtime Cost When Disabled**: Log level checks are atomic and branch-predictable
2. **Minimal Lock Contention**: Separate mutexes for file I/O and ML context operations
3. **Structured Over Freeform**: Typed data structures over string concatenation
4. **ML-First Design**: Native support for model versioning, metrics aggregation, inference telemetry

---

## Architecture & Design

### Class Hierarchy

```cpp
namespace inference_lab::common {
    class Logger {
        // Singleton with thread-safe initialization
        // Separate concerns: file I/O, ML context, configuration
        // Lock-free atomic operations for hot paths
    };
}
```

### Memory Layout & Threading Model

```
┌─────────────────────────────────────────────────────────────────┐
│                           Logger Instance                        │
├─────────────────────────────────────────────────────────────────┤
│  Core State (Atomics)         │  File I/O (Mutex Protected)     │
│  - m_enabled_levels_[6]       │  - m_log_file_                  │
│  - m_stderr_enabled_          │  - m_file_mutex_                │
│  - m_file_output_enabled_     │                                 │
│  - m_ml_logging_enabled_      │                                 │
├─────────────────────────────────────────────────────────────────┤
│  ML Context (Mutex Protected) │  Static Singleton Management    │
│  - m_model_contexts_          │  - m_instance (shared_ptr)      │
│  - m_metrics_buffer_          │  - m_instance_mutex             │
│  - m_ml_context_mutex_        │                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Threading Model:**
- **Lock-Free Hot Path**: Level checks use atomic loads with acquire semantics
- **Dedicated Mutexes**: File I/O and ML operations have separate synchronization
- **Singleton Safety**: Thread-safe lazy initialization with double-checked locking

---

## Core Logging API

### Log Levels & Semantics

```cpp
enum class LogLevel : std::uint8_t {
    DEBUG = 0,      // Detailed diagnostic information
    INFO = 1,       // General informational messages  
    NORMAL = 2,     // Standard operational messages
    WARNING = 3,    // Warning conditions (recoverable)
    ERROR = 4,      // Error conditions (operation failed)
    CRITICAL = 5    // Critical conditions (system unstable)
};
```

### Basic Logging Interface

```cpp
#include "common/src/logging.hpp"

// Simple message logging
LOG_INFO_PRINT("Application started successfully");
LOG_ERROR_PRINT("Failed to load configuration");

// Formatted logging with type safety
LOG_INFO_PRINT("Processing batch: {} items in {:.2f}ms", batch_size, elapsed_ms);
LOG_WARNING_PRINT("Memory usage: {}/{} MB ({:.1f}%)", used_mb, total_mb, percentage);

// Structured logging with depth (for nested operations)
auto& logger = Logger::get_instance();
logger.print_log_with_depth(LogLevel::INFO, 2, "  Validating model: {}", model_name);
```

**Output Format:**
```
2025-01-15 14:32:18.456 UTC [INFO] [Thread:0x1a2b3c4d5] Processing batch: 1024 items in 45.67ms
2025-01-15 14:32:18.501 UTC [WARNING] [Thread:0x1a2b3c4d5] Memory usage: 2048/4096 MB (50.0%)
2025-01-15 14:32:18.502 UTC [INFO] [Thread:0x1a2b3c4d5]     Validating model: resnet50_v2
```

### Advanced Configuration

```cpp
// Runtime log level control
Logger::set_level_enabled(LogLevel::DEBUG, false);    // Disable debug logs
Logger::set_level_enabled(LogLevel::ERROR, true);     // Enable error logs

// Output stream control
auto& logger = Logger::get_instance();
logger.set_file_output_enabled(false);  // Disable file output
logger.disable_stderr();                // Suppress stderr for errors
logger.flush();                         // Force immediate file write

// RAII stderr suppression for sensitive operations
{
    Logger::StderrSuppressionGuard guard;
    // No stderr output in this scope
    sensitive_operation();
}   // stderr restored automatically
```

---

## ML-Specific Logging Extensions

### Model Context & Lifecycle Management

The Logger provides comprehensive model lifecycle tracking with version management and stage progression monitoring.

#### Model Registration

```cpp
#include "common/src/logging.hpp"

// Define model context with full metadata
ModelContext context{
    .name = "resnet50_classifier",
    .version = "2.1.0", 
    .framework = "TensorRT",
    .stage = ModelStage::PRODUCTION,
    .path = "/models/resnet50_v2.1.0.trt",
    .size_mb = 512,
    .checksum = "sha256:a1b2c3d4e5f6...",
    .loaded_at = std::chrono::system_clock::now()
};

auto& logger = Logger::get_instance();
logger.register_model(context);
```

**Output:**
```
2025-01-15 14:35:22.123 UTC [INFO] [Thread:0x1a2b3c4d5] Registered ML model: name=resnet50_classifier version=2.1.0 framework=TensorRT stage=PRODUCTION size=512MB
```

#### Model Stage Transitions

```cpp
// Track model progression through deployment pipeline
logger.update_model_stage("resnet50_classifier", ModelStage::STAGING);
logger.update_model_stage("resnet50_classifier", ModelStage::PRODUCTION);
logger.update_model_stage("resnet50_classifier", ModelStage::ARCHIVED);
```

**Output:**
```
2025-01-15 14:36:15.789 UTC [INFO] [Thread:0x1a2b3c4d5] Updated model stage: name=resnet50_classifier STAGING -> PRODUCTION
2025-01-15 14:45:30.456 UTC [INFO] [Thread:0x1a2b3c4d5] Updated model stage: name=resnet50_classifier PRODUCTION -> ARCHIVED
```

### ML Operation Logging

#### Structured Operation Logging

```cpp
// Log specific ML operations with automatic context injection
LOG_MODEL_LOAD("resnet50_classifier", "Loaded from checkpoint: {}", checkpoint_path);
LOG_INFERENCE_START("resnet50_classifier", "Processing batch: {} images", batch_size);
LOG_INFERENCE_COMPLETE("resnet50_classifier", "Classified {} images in {:.2f}ms", 
                      batch_size, inference_time);
LOG_BATCH_PROCESS("resnet50_classifier", "Throughput: {:.1f} images/sec", throughput);

// Generic ML operation logging
LOG_ML_OPERATION(MLOperation::MODEL_VALIDATE, "resnet50_classifier", 
                "Validation accuracy: {:.3f}", accuracy);
```

**Output:**
```
2025-01-15 14:37:45.123 UTC [INFO] [Thread:0x1a2b3c4d5] [ML:MODEL_LOAD] model=resnet50_classifier version=2.1.0 stage=PRODUCTION framework=TensorRT Loaded from checkpoint: /models/checkpoint_epoch_100.pth
2025-01-15 14:37:45.234 UTC [INFO] [Thread:0x1a2b3c4d5] [ML:INFERENCE_START] model=resnet50_classifier version=2.1.0 stage=PRODUCTION framework=TensorRT Processing batch: 32 images
2025-01-15 14:37:45.267 UTC [INFO] [Thread:0x1a2b3c4d5] [ML:INFERENCE_COMPLETE] model=resnet50_classifier version=2.1.0 stage=PRODUCTION framework=TensorRT Classified 32 images in 33.45ms
```

### Inference Metrics Tracking

#### Real-Time Metrics Logging

```cpp
// Comprehensive inference metrics
InferenceMetrics metrics{
    .latency_ms = 33.45,
    .preprocessing_ms = 5.12,
    .inference_ms = 25.33,
    .postprocessing_ms = 3.00,
    .memory_mb = 1024,
    .batch_size = 32,
    .throughput = 956.8,
    .confidence = 0.94,
    .device = "CUDA:0"
};

LOG_ML_METRICS("resnet50_classifier", metrics);
```

**Output:**
```
2025-01-15 14:37:45.268 UTC [INFO] [Thread:0x1a2b3c4d5] [ML:METRICS] model=resnet50_classifier latency=33.45ms preprocessing=5.12ms inference=25.33ms postprocessing=3.00ms memory=1024MB batch_size=32 throughput=956.80/s confidence=0.940 device=CUDA:0
```

#### Metrics Buffering & Aggregation

```cpp
auto& logger = Logger::get_instance();

// Configure metrics buffering
logger.set_max_metrics_buffer_size(1000);

// Buffer metrics for batch analysis
for (const auto& batch : inference_batches) {
    InferenceMetrics metrics = process_batch(batch);
    logger.buffer_metrics(metrics);
}

// Flush and log aggregate statistics
logger.flush_metrics_buffer();

// Retrieve aggregate metrics for analysis
auto aggregate = logger.get_aggregate_metrics("resnet50_classifier", 
                                             std::chrono::minutes(10));
if (aggregate) {
    std::cout << "Average latency: " << aggregate->latency_ms << "ms\n";
    std::cout << "Average throughput: " << aggregate->throughput << " samples/sec\n";
}
```

**Output:**
```
2025-01-15 14:40:15.789 UTC [INFO] [Thread:0x1a2b3c4d5] [ML:AGGREGATE] buffered_samples=100 avg_latency=31.25ms avg_throughput=985.30 avg_confidence=0.928
```

### Enhanced Error Context

#### Structured ML Error Logging

```cpp
// Rich error context for debugging and monitoring
MLErrorContext error_context{
    .error_code = "CUDA_OUT_OF_MEMORY",
    .component = "TensorRTEngine", 
    .operation = "execute_inference",
    .metadata = {
        {"requested_memory_mb", "2048"},
        {"available_memory_mb", "1536"},
        {"batch_size", "64"},
        {"model_precision", "FP16"}
    }
};

LOG_ML_ERROR("resnet50_classifier", error_context, 
            "Insufficient GPU memory for batch size 64 with FP16 precision");
```

**Output:**
```
2025-01-15 14:42:33.456 UTC [ERROR] [Thread:0x1a2b3c4d5] [ML:ERROR] model=resnet50_classifier component=TensorRTEngine operation=execute_inference error_code=CUDA_OUT_OF_MEMORY metadata={requested_memory_mb=2048, available_memory_mb=1536, batch_size=64, model_precision=FP16} message=Insufficient GPU memory for batch size 64 with FP16 precision
```

---

## Performance & Threading

### Lock-Free Hot Path Design

The Logger is optimized for high-throughput logging with minimal performance impact:

```cpp
// Hot path: atomic check with early exit (typically 1-2 CPU cycles)
if (!Logger::is_level_enabled(LogLevel::INFO)) {
    return;  // Zero cost when logging disabled
}

// Cold path: actual logging with appropriate synchronization
Logger::get_instance().print_log(LogLevel::INFO, "Message: {}", value);
```

### Thread Safety Guarantees

```cpp
// Thread-safe concurrent logging from multiple threads
void worker_thread(int thread_id) {
    // All operations are thread-safe
    LOG_INFO_PRINT("Worker {} started", thread_id);
    
    // ML operations are fully thread-safe
    ModelContext context = create_model_context(thread_id);
    Logger::get_instance().register_model(context);
    
    // Metrics can be logged concurrently
    InferenceMetrics metrics = run_inference();
    LOG_ML_METRICS("worker_model_" + std::to_string(thread_id), metrics);
}

// Launch multiple worker threads safely
std::vector<std::thread> workers;
for (int i = 0; i < 8; ++i) {
    workers.emplace_back(worker_thread, i);
}
```

### Performance Characteristics

| Operation | Typical Latency | Thread Contention | Memory Allocation |
|-----------|----------------|-------------------|-------------------|
| `is_level_enabled()` | 1-2 CPU cycles | Lock-free | None |
| `LOG_INFO_PRINT()` | 50-100 μs | Minimal (file mutex) | Stack-based formatting |
| `register_model()` | 10-20 μs | Low (ML context mutex) | Map insertion |
| `buffer_metrics()` | 5-10 μs | Low (ML context mutex) | Vector append |
| `flush_metrics_buffer()` | 100-500 μs | Medium (calculation) | Temporary aggregation |

---

## Configuration & Control

### Runtime Configuration

```cpp
auto& logger = Logger::get_instance();

// Selective log level control
Logger::set_level_enabled(LogLevel::DEBUG, false);    // Disable debug
Logger::set_level_enabled(LogLevel::WARNING, true);   // Enable warnings

// Output control
logger.set_file_output_enabled(true);   // Enable file logging
logger.set_file_output_enabled(false);  // Disable file logging
logger.enable_stderr();                 // Enable error output to stderr
logger.disable_stderr();                // Suppress stderr output

// ML-specific configuration
logger.set_ml_logging_enabled(true);           // Enable ML logging
logger.set_max_metrics_buffer_size(5000);     // Set buffer limit
```

### Production Configuration Patterns

```cpp
// Production setup with performance optimization
void configure_production_logging() {
    auto& logger = Logger::get_instance("/var/log/inference/application.log");
    
    // Optimize for production performance
    Logger::set_level_enabled(LogLevel::DEBUG, false);    // Disable debug
    Logger::set_level_enabled(LogLevel::INFO, true);      // Keep info
    Logger::set_level_enabled(LogLevel::WARNING, true);   // Keep warnings
    Logger::set_level_enabled(LogLevel::ERROR, true);     // Keep errors
    
    // Configure ML logging for monitoring
    logger.set_ml_logging_enabled(true);
    logger.set_max_metrics_buffer_size(10000);  // Large buffer for aggregation
    
    // Enable file output, suppress console noise
    logger.set_file_output_enabled(true);
    logger.disable_stderr();  // Logs go to file only
}

// Development setup with full debugging
void configure_development_logging() {
    auto& logger = Logger::get_instance("/tmp/dev_inference.log");
    
    // Enable all levels for debugging
    for (int level = 0; level <= static_cast<int>(LogLevel::CRITICAL); ++level) {
        Logger::set_level_enabled(static_cast<LogLevel>(level), true);
    }
    
    // Enable console output for immediate feedback
    logger.enable_stderr();
    logger.set_file_output_enabled(true);
    logger.set_ml_logging_enabled(true);
}
```

---

## Integration Patterns

### Error Handling Integration

```cpp
#include "common/src/result.hpp"  // Assumes Result<T,E> pattern

// Integration with Result<T,E> error handling
template<typename T, typename E>
auto log_result_error(const Result<T, E>& result, const std::string& operation) -> void {
    if (result.is_err()) {
        MLErrorContext error_context{
            .error_code = "OPERATION_FAILED",
            .component = "InferenceEngine",
            .operation = operation,
            .metadata = {}
        };
        
        LOG_ML_ERROR("current_model", error_context, 
                    "Operation failed: {}", result.err_value());
    }
}

// Usage pattern
auto load_result = load_model("resnet50.trt");
log_result_error(load_result, "load_model");

if (load_result.is_ok()) {
    LOG_MODEL_LOAD("resnet50", "Model loaded successfully");
    auto inference_result = run_inference(batch);
    log_result_error(inference_result, "run_inference");
}
```

### MLOps Pipeline Integration

```cpp
// Complete MLOps lifecycle with structured logging
class MLModelManager {
private:
    Logger& logger_ = Logger::get_instance();
    
public:
    auto deploy_model(const std::string& model_path, const std::string& version) -> Result<void, std::string> {
        // Register model in development stage
        ModelContext context{
            .name = extract_model_name(model_path),
            .version = version,
            .framework = detect_framework(model_path),
            .stage = ModelStage::DEVELOPMENT,
            .path = model_path,
            .size_mb = get_file_size_mb(model_path),
            .checksum = calculate_checksum(model_path),
            .loaded_at = std::chrono::system_clock::now()
        };
        
        logger_.register_model(context);
        LOG_MODEL_LOAD(context.name, "Loaded for validation");
        
        // Validation stage
        if (auto validation_result = validate_model(context); validation_result.is_err()) {
            MLErrorContext error{
                .error_code = "VALIDATION_FAILED",
                .component = "ModelValidator",
                .operation = "validate_accuracy",
                .metadata = {{"expected_accuracy", "0.95"}, {"actual_accuracy", "0.89"}}
            };
            LOG_ML_ERROR(context.name, error, "Model validation failed");
            return Err(validation_result.err_value());
        }
        
        // Promote to staging
        logger_.update_model_stage(context.name, ModelStage::STAGING);
        LOG_MODEL_VALIDATE(context.name, "Validation passed, promoting to staging");
        
        // Performance benchmarking
        if (auto benchmark_result = benchmark_model(context); benchmark_result.is_ok()) {
            InferenceMetrics metrics = benchmark_result.ok_value();
            LOG_ML_METRICS(context.name, metrics);
            LOG_PERFORMANCE_BENCHMARK(context.name, "Staging benchmarks completed");
            
            // Promote to production if benchmarks pass
            if (metrics.latency_ms < 50.0 && metrics.throughput > 1000.0) {
                logger_.update_model_stage(context.name, ModelStage::PRODUCTION);
                LOG_MODEL_LOAD(context.name, "Promoted to production");
            }
        }
        
        return Ok();
    }
    
    auto run_production_inference(const std::string& model_name, const InputBatch& batch) -> Result<OutputBatch, std::string> {
        LOG_INFERENCE_START(model_name, "Processing batch size: {}", batch.size());
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = execute_inference(model_name, batch);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (result.is_ok()) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            InferenceMetrics metrics{
                .latency_ms = duration.count() / 1000.0,
                .batch_size = batch.size(),
                .throughput = (batch.size() * 1000000.0) / duration.count(),
                .confidence = calculate_confidence(result.ok_value()),
                .device = get_device_name()
            };
            
            LOG_ML_METRICS(model_name, metrics);
            logger_.buffer_metrics(metrics);  // For aggregate analysis
            
            LOG_INFERENCE_COMPLETE(model_name, "Processed {} samples in {:.2f}ms", 
                                 batch.size(), metrics.latency_ms);
        }
        
        return result;
    }
};
```

### Monitoring & Alerting Integration

```cpp
// Integration with monitoring systems
class ProductionMonitor {
private:
    Logger& logger_ = Logger::get_instance();
    std::chrono::steady_clock::time_point last_flush_;
    
public:
    void periodic_metrics_flush() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_flush_ > std::chrono::minutes(5)) {
            // Flush aggregated metrics for monitoring dashboard
            logger_.flush_metrics_buffer();
            
            // Get recent performance data
            auto recent_metrics = logger_.get_aggregate_metrics("production_model", 
                                                              std::chrono::minutes(5));
            if (recent_metrics) {
                // Check for performance degradation
                if (recent_metrics->latency_ms > 100.0) {
                    MLErrorContext alert{
                        .error_code = "PERFORMANCE_DEGRADATION",
                        .component = "PerformanceMonitor",
                        .operation = "latency_check",
                        .metadata = {
                            {"current_latency_ms", std::to_string(recent_metrics->latency_ms)},
                            {"threshold_ms", "100.0"},
                            {"sample_count", std::to_string(logger_.get_metrics_buffer_size())}
                        }
                    };
                    LOG_ML_ERROR("production_model", alert, 
                               "Latency exceeded threshold: {:.2f}ms > 100.0ms", 
                               recent_metrics->latency_ms);
                }
                
                if (recent_metrics->confidence < 0.90) {
                    LOG_ML_ERROR("production_model", 
                               {"CONFIDENCE_DROP", "ConfidenceMonitor", "confidence_check", {}},
                               "Model confidence dropped below threshold: {:.3f} < 0.90", 
                               recent_metrics->confidence);
                }
            }
            
            last_flush_ = now;
        }
    }
};
```

---

## Best Practices

### 1. Log Level Selection

```cpp
// ✅ Good: Use appropriate log levels
LOG_DEBUG_PRINT("Tensor shape: [{}, {}, {}]", h, w, c);           // Detailed diagnostics
LOG_INFO_PRINT("Model {} loaded successfully", model_name);        // Important events
LOG_WARNING_PRINT("Model accuracy below threshold: {:.3f}", acc);  // Recoverable issues
LOG_ERROR_PRINT("Failed to allocate GPU memory: {} MB", size);     // Operation failures
LOG_CRITICAL_PRINT("CUDA driver initialization failed");          // System instability

// ❌ Avoid: Wrong log levels
LOG_CRITICAL_PRINT("Processing batch 42");  // Not critical
LOG_DEBUG_PRINT("System out of memory");    // Should be ERROR or CRITICAL
```

### 2. Structured vs. Freeform Logging

```cpp
// ✅ Good: Use structured ML logging for metrics and operations
InferenceMetrics metrics{.latency_ms = 45.2, .batch_size = 32};
LOG_ML_METRICS("resnet50", metrics);

LOG_MODEL_LOAD("resnet50", "Checkpoint restored from epoch: {}", epoch);

// ✅ Good: Use regular logging for general application events
LOG_INFO_PRINT("Server started on port: {}", port);
LOG_WARNING_PRINT("Configuration file not found, using defaults");

// ❌ Avoid: Freeform logging for ML metrics
LOG_INFO_PRINT("Inference took 45.2ms for batch size 32");  // Hard to parse
```

### 3. Error Context Best Practices

```cpp
// ✅ Good: Rich error context with actionable metadata
MLErrorContext context{
    .error_code = "CUDA_INSUFFICIENT_MEMORY",
    .component = "TensorRTEngine",
    .operation = "allocate_workspace",
    .metadata = {
        {"requested_bytes", std::to_string(requested_size)},
        {"available_bytes", std::to_string(available_size)},
        {"gpu_id", std::to_string(gpu_id)},
        {"model_name", model_name}
    }
};
LOG_ML_ERROR(model_name, context, "Workspace allocation failed");

// ❌ Avoid: Generic error messages without context
LOG_ERROR_PRINT("CUDA error occurred");
```

### 4. Performance Optimization

```cpp
// ✅ Good: Check log levels before expensive operations
if (Logger::is_level_enabled(LogLevel::DEBUG)) {
    std::string expensive_debug_info = generate_tensor_summary(tensor);
    LOG_DEBUG_PRINT("Tensor analysis: {}", expensive_debug_info);
}

// ✅ Good: Use metrics buffering for high-frequency logging
for (const auto& batch : batches) {
    InferenceMetrics metrics = process_batch(batch);
    logger.buffer_metrics(metrics);  // Buffer instead of immediate log
}
logger.flush_metrics_buffer();  // Periodic flush with aggregation

// ❌ Avoid: Expensive operations in disabled log calls
LOG_DEBUG_PRINT("Debug: {}", expensive_calculation());  // Always calculates
```

### 5. Thread Safety Patterns

```cpp
// ✅ Good: Logger is thread-safe, use freely across threads
void inference_worker_thread() {
    LOG_INFO_PRINT("Worker thread started");
    
    // All ML logging operations are thread-safe
    ModelContext ctx = load_model();
    Logger::get_instance().register_model(ctx);
    
    while (running) {
        auto metrics = process_batch();
        LOG_ML_METRICS(ctx.name, metrics);
    }
}

// ✅ Good: Use RAII for temporary configuration
{
    Logger::StderrSuppressionGuard guard;  // Suppress stderr in this scope
    run_noisy_operation();
}  // stderr automatically restored
```

---

## API Reference

### Core Types

#### LogLevel
```cpp
enum class LogLevel : std::uint8_t {
    DEBUG = 0,      // Detailed diagnostic information  
    INFO = 1,       // General informational messages
    NORMAL = 2,     // Standard operational messages
    WARNING = 3,    // Warning conditions (recoverable)
    ERROR = 4,      // Error conditions (operation failed)
    CRITICAL = 5    // Critical conditions (system unstable)
};
```

#### ModelStage
```cpp
enum class ModelStage : std::uint8_t {
    DEVELOPMENT = 0,  // Model under development
    STAGING = 1,      // Model in staging environment
    PRODUCTION = 2,   // Model deployed in production
    ARCHIVED = 3,     // Model archived (still available)
    DEPRECATED = 4    // Model deprecated (removal pending)
};
```

#### MLOperation
```cpp
enum class MLOperation : std::uint8_t {
    MODEL_LOAD = 0,           // Model loading operation
    MODEL_UNLOAD = 1,         // Model unloading operation
    INFERENCE_START = 2,      // Inference operation started
    INFERENCE_COMPLETE = 3,   // Inference operation completed
    BATCH_PROCESS = 4,        // Batch processing operation
    MODEL_VALIDATE = 5,       // Model validation operation
    PERFORMANCE_BENCHMARK = 6, // Performance benchmarking
    ERROR_OCCURRED = 7        // Error occurred during operation
};
```

### Core Structures

#### InferenceMetrics
```cpp
struct InferenceMetrics {
    double latency_ms = 0.0;           ///< End-to-end inference latency
    double preprocessing_ms = 0.0;     ///< Input preprocessing time
    double inference_ms = 0.0;         ///< Core model inference time  
    double postprocessing_ms = 0.0;    ///< Output postprocessing time
    std::size_t memory_mb = 0;         ///< Memory usage in MB
    std::size_t batch_size = 1;        ///< Batch size processed
    double throughput = 0.0;           ///< Samples per second
    double confidence = 0.0;           ///< Average prediction confidence
    std::string device = "CPU";        ///< Execution device (CPU/GPU/etc.)
};
```

#### ModelContext
```cpp
struct ModelContext {
    std::string name;                                    ///< Model name/identifier
    std::string version = "1.0.0";                      ///< Semantic version
    std::string framework = "ONNX";                     ///< ML framework
    ModelStage stage = ModelStage::DEVELOPMENT;         ///< Deployment stage
    std::string path;                                    ///< Model file path
    std::size_t size_mb = 0;                           ///< Model size in MB
    std::string checksum;                               ///< Model file checksum
    std::chrono::system_clock::time_point loaded_at;    ///< Load timestamp
};
```

#### MLErrorContext
```cpp
struct MLErrorContext {
    std::string error_code;                                        ///< Structured error code
    std::string component;                                         ///< Component where error occurred
    std::string operation;                                         ///< Operation that failed
    std::unordered_map<std::string, std::string> metadata;       ///< Additional context
};
```

### Logger Class Methods

#### Singleton Management
```cpp
// Get logger instance (default log file)
static auto get_instance() -> Logger&;

// Get logger instance with custom log file  
static auto get_instance(const std::string& custom_path, bool append = true) -> Logger&;

// Get shared_ptr to logger instance
static auto get_instance_ptr() -> std::shared_ptr<Logger>;
static auto get_instance_ptr(const std::string& custom_path, bool append = true) -> std::shared_ptr<Logger>;
```

#### Core Logging
```cpp
// Template-based formatted logging
template<typename... FormatArgs>
void print_log(LogLevel level, const std::string& format, const FormatArgs&... args);

// Logging with indentation for nested operations  
template<typename... FormatArgs>
void print_log_with_depth(LogLevel level, int depth, const std::string& format, const FormatArgs&... args);
```

#### Configuration Control
```cpp
// Log level control
static void set_level_enabled(LogLevel level, bool enabled);
static auto is_level_enabled(LogLevel level) -> bool;

// Output stream control
void disable_stderr();
void enable_stderr();
auto is_stderr_enabled() const -> bool;

void set_file_output_enabled(bool enabled);
auto is_file_output_enabled() const -> bool;

// Force immediate file flush
void flush();
```

#### ML-Specific Methods
```cpp
// Model lifecycle management
void register_model(const ModelContext& context);
void unregister_model(const std::string& model_name);
auto get_model_context(const std::string& model_name) const -> std::optional<ModelContext>;
void update_model_stage(const std::string& model_name, ModelStage stage);

// ML operation logging
template<typename... FormatArgs>
void log_ml_operation(MLOperation operation, const std::string& model_name, 
                     const std::string& format = "", const FormatArgs&... args);

// Metrics tracking
void log_inference_metrics(const std::string& model_name, const InferenceMetrics& metrics);
void buffer_metrics(const InferenceMetrics& metrics);
void flush_metrics_buffer();
auto get_metrics_buffer_size() const -> std::size_t;
void set_max_metrics_buffer_size(std::size_t size);

// Error logging
void log_ml_error(const std::string& model_name, const MLErrorContext& error_context, 
                 const std::string& message = "");

// Configuration
void set_ml_logging_enabled(bool enabled);
auto is_ml_logging_enabled() const -> bool;

// Analytics
auto get_aggregate_metrics(const std::string& model_name, std::chrono::minutes duration) const 
    -> std::optional<InferenceMetrics>;
```

### Logging Macros

#### Standard Logging Macros
```cpp
#define LOG_DEBUG_PRINT(...)    LOG_BASE_PRINT(LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO_PRINT(...)     LOG_BASE_PRINT(LogLevel::INFO, __VA_ARGS__)  
#define LOG_NORMAL_PRINT(...)   LOG_BASE_PRINT(LogLevel::NORMAL, __VA_ARGS__)
#define LOG_WARNING_PRINT(...)  LOG_BASE_PRINT(LogLevel::WARNING, __VA_ARGS__)
#define LOG_ERROR_PRINT(...)    LOG_BASE_PRINT(LogLevel::ERROR, __VA_ARGS__)
#define LOG_CRITICAL_PRINT(...) LOG_BASE_PRINT(LogLevel::CRITICAL, __VA_ARGS__)
```

#### ML-Specific Logging Macros
```cpp
// Generic ML operation logging
#define LOG_ML_OPERATION(operation, model_name, ...) \
    Logger::get_instance().log_ml_operation(operation, model_name, __VA_ARGS__)

// Convenience macros for specific operations
#define LOG_MODEL_LOAD(model_name, ...)        LOG_ML_OPERATION(MLOperation::MODEL_LOAD, model_name, __VA_ARGS__)
#define LOG_MODEL_UNLOAD(model_name, ...)      LOG_ML_OPERATION(MLOperation::MODEL_UNLOAD, model_name, __VA_ARGS__)
#define LOG_INFERENCE_START(model_name, ...)   LOG_ML_OPERATION(MLOperation::INFERENCE_START, model_name, __VA_ARGS__)
#define LOG_INFERENCE_COMPLETE(model_name, ...) LOG_ML_OPERATION(MLOperation::INFERENCE_COMPLETE, model_name, __VA_ARGS__)
#define LOG_BATCH_PROCESS(model_name, ...)     LOG_ML_OPERATION(MLOperation::BATCH_PROCESS, model_name, __VA_ARGS__)
#define LOG_MODEL_VALIDATE(model_name, ...)    LOG_ML_OPERATION(MLOperation::MODEL_VALIDATE, model_name, __VA_ARGS__)
#define LOG_PERFORMANCE_BENCHMARK(model_name, ...) LOG_ML_OPERATION(MLOperation::PERFORMANCE_BENCHMARK, model_name, __VA_ARGS__)

// Metrics and error logging
#define LOG_ML_METRICS(model_name, metrics) \
    Logger::get_instance().log_inference_metrics(model_name, metrics)

#define LOG_ML_ERROR(model_name, error_context, message) \
    Logger::get_instance().log_ml_error(model_name, error_context, message)
```

---

## Examples

### Example 1: Complete ML Pipeline Logging

```cpp
#include "common/src/logging.hpp"
#include <chrono>
#include <vector>

class ImageClassificationPipeline {
private:
    Logger& logger_ = Logger::get_instance("/var/log/ml/classification.log");
    std::string model_name_ = "resnet50_classifier";
    
public:
    auto initialize() -> bool {
        // Configure logging for production
        logger_.set_ml_logging_enabled(true);
        logger_.set_max_metrics_buffer_size(1000);
        Logger::set_level_enabled(LogLevel::DEBUG, false);  // Disable debug in production
        
        // Register model
        ModelContext context{
            .name = model_name_,
            .version = "2.1.0",
            .framework = "TensorRT", 
            .stage = ModelStage::PRODUCTION,
            .path = "/models/resnet50_v2.1.0.trt",
            .size_mb = 512,
            .checksum = "sha256:abc123def456...",
            .loaded_at = std::chrono::system_clock::now()
        };
        
        logger_.register_model(context);
        LOG_MODEL_LOAD(model_name_, "Model initialized for production inference");
        
        return true;
    }
    
    auto classify_batch(const std::vector<Image>& images) -> Result<std::vector<Classification>, std::string> {
        LOG_INFERENCE_START(model_name_, "Processing batch of {} images", images.size());
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Preprocessing
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        auto preprocessed = preprocess_images(images);
        if (preprocessed.is_err()) {
            MLErrorContext error{
                .error_code = "PREPROCESSING_FAILED",
                .component = "ImagePreprocessor",
                .operation = "normalize_and_resize", 
                .metadata = {{"batch_size", std::to_string(images.size())}}
            };
            LOG_ML_ERROR(model_name_, error, "Image preprocessing failed: {}", preprocessed.err_value());
            return Err(preprocessed.err_value());
        }
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        
        // Inference
        auto inference_start = std::chrono::high_resolution_clock::now();
        auto inference_result = run_tensorrt_inference(preprocessed.ok_value());
        if (inference_result.is_err()) {
            MLErrorContext error{
                .error_code = "INFERENCE_EXECUTION_FAILED",
                .component = "TensorRTEngine",
                .operation = "execute_inference",
                .metadata = {
                    {"batch_size", std::to_string(images.size())},
                    {"input_shape", get_input_shape_str()},
                    {"gpu_memory_mb", std::to_string(get_gpu_memory_usage())}
                }
            };
            LOG_ML_ERROR(model_name_, error, "TensorRT inference failed: {}", inference_result.err_value());
            return Err(inference_result.err_value());
        }
        auto inference_end = std::chrono::high_resolution_clock::now();
        
        // Postprocessing  
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        auto classifications = postprocess_outputs(inference_result.ok_value());
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        
        auto total_end = std::chrono::high_resolution_clock::now();
        
        // Calculate metrics
        auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start);
        auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
        auto postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - postprocess_start);
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - start_time);
        
        InferenceMetrics metrics{
            .latency_ms = total_time.count() / 1000.0,
            .preprocessing_ms = preprocess_time.count() / 1000.0,
            .inference_ms = inference_time.count() / 1000.0,
            .postprocessing_ms = postprocess_time.count() / 1000.0,
            .memory_mb = get_gpu_memory_usage(),
            .batch_size = images.size(),
            .throughput = (images.size() * 1000000.0) / total_time.count(),
            .confidence = calculate_average_confidence(classifications),
            .device = "CUDA:0"
        };
        
        // Log metrics and buffer for aggregation
        LOG_ML_METRICS(model_name_, metrics);
        logger_.buffer_metrics(metrics);
        
        LOG_INFERENCE_COMPLETE(model_name_, "Classified {} images in {:.2f}ms (throughput: {:.1f} images/sec)", 
                             images.size(), metrics.latency_ms, metrics.throughput);
        
        return Ok(classifications);
    }
    
    auto get_performance_summary() -> void {
        // Flush buffered metrics and get aggregates
        logger_.flush_metrics_buffer();
        
        auto recent_performance = logger_.get_aggregate_metrics(model_name_, std::chrono::minutes(10));
        if (recent_performance) {
            LOG_INFO_PRINT("Performance Summary (10min): avg_latency={:.2f}ms avg_throughput={:.1f}/s avg_confidence={:.3f}",
                          recent_performance->latency_ms, recent_performance->throughput, recent_performance->confidence);
            
            // Check for performance issues
            if (recent_performance->latency_ms > 100.0) {
                LOG_WARNING_PRINT("Performance degradation detected: latency {:.2f}ms exceeds 100ms threshold",
                                recent_performance->latency_ms);
            }
            
            if (recent_performance->confidence < 0.90) {
                LOG_WARNING_PRINT("Model confidence degradation: {:.3f} below 0.90 threshold", 
                                recent_performance->confidence);
            }
        }
    }
};
```

### Example 2: Multi-Model A/B Testing with Logging

```cpp
#include "common/src/logging.hpp"

class ABTestingFramework {
private:
    Logger& logger_ = Logger::get_instance();
    std::string model_a_name_ = "resnet50_v1";
    std::string model_b_name_ = "resnet50_v2";
    
public:
    auto setup_models() -> void {
        // Register both models for A/B testing
        ModelContext model_a{
            .name = model_a_name_,
            .version = "1.5.0",
            .framework = "ONNX",
            .stage = ModelStage::PRODUCTION,
            .path = "/models/resnet50_v1.5.0.onnx",
            .size_mb = 256,
            .checksum = "sha256:old_model_hash",
            .loaded_at = std::chrono::system_clock::now()
        };
        
        ModelContext model_b{
            .name = model_b_name_,
            .version = "2.0.0", 
            .framework = "TensorRT",
            .stage = ModelStage::STAGING,  // New model in staging
            .path = "/models/resnet50_v2.0.0.trt",
            .size_mb = 384,
            .checksum = "sha256:new_model_hash",
            .loaded_at = std::chrono::system_clock::now()
        };
        
        logger_.register_model(model_a);
        logger_.register_model(model_b);
        
        LOG_INFO_PRINT("A/B testing setup complete: {} vs {}", model_a_name_, model_b_name_);
    }
    
    auto run_ab_test(const std::vector<TestBatch>& test_batches) -> void {
        LOG_INFO_PRINT("Starting A/B test with {} batches", test_batches.size());
        
        for (size_t i = 0; i < test_batches.size(); ++i) {
            const auto& batch = test_batches[i];
            
            // Route 50% traffic to each model
            std::string selected_model = (i % 2 == 0) ? model_a_name_ : model_b_name_;
            
            LOG_BATCH_PROCESS(selected_model, "Processing A/B test batch {} with {} samples", 
                            i, batch.size());
            
            auto start_time = std::chrono::high_resolution_clock::now();
            auto result = process_batch_with_model(selected_model, batch);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            if (result.is_ok()) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                
                InferenceMetrics metrics{
                    .latency_ms = duration.count() / 1000.0,
                    .batch_size = batch.size(),
                    .throughput = (batch.size() * 1000000.0) / duration.count(),
                    .confidence = calculate_batch_confidence(result.ok_value()),
                    .device = selected_model == model_b_name_ ? "CUDA:0" : "CPU"
                };
                
                LOG_ML_METRICS(selected_model, metrics);
                logger_.buffer_metrics(metrics);
            } else {
                MLErrorContext error{
                    .error_code = "BATCH_PROCESSING_FAILED",
                    .component = "ABTestFramework",
                    .operation = "process_batch",
                    .metadata = {
                        {"model_name", selected_model},
                        {"batch_index", std::to_string(i)},
                        {"batch_size", std::to_string(batch.size())}
                    }
                };
                LOG_ML_ERROR(selected_model, error, "Batch processing failed: {}", result.err_value());
            }
        }
        
        // Generate A/B test comparison report
        generate_ab_comparison_report();
    }
    
private:
    auto generate_ab_comparison_report() -> void {
        logger_.flush_metrics_buffer();
        
        auto model_a_metrics = logger_.get_aggregate_metrics(model_a_name_, std::chrono::minutes(30));
        auto model_b_metrics = logger_.get_aggregate_metrics(model_b_name_, std::chrono::minutes(30));
        
        if (model_a_metrics && model_b_metrics) {
            LOG_INFO_PRINT("=== A/B Test Results ===");
            LOG_INFO_PRINT("Model A ({}): latency={:.2f}ms throughput={:.1f}/s confidence={:.3f}",
                          model_a_name_, model_a_metrics->latency_ms, 
                          model_a_metrics->throughput, model_a_metrics->confidence);
            LOG_INFO_PRINT("Model B ({}): latency={:.2f}ms throughput={:.1f}/s confidence={:.3f}",
                          model_b_name_, model_b_metrics->latency_ms, 
                          model_b_metrics->throughput, model_b_metrics->confidence);
            
            // Determine winner
            double latency_improvement = ((model_a_metrics->latency_ms - model_b_metrics->latency_ms) / model_a_metrics->latency_ms) * 100.0;
            double throughput_improvement = ((model_b_metrics->throughput - model_a_metrics->throughput) / model_a_metrics->throughput) * 100.0;
            double confidence_improvement = ((model_b_metrics->confidence - model_a_metrics->confidence) / model_a_metrics->confidence) * 100.0;
            
            LOG_INFO_PRINT("Performance delta: latency={:.1f}% throughput={:.1f}% confidence={:.1f}%",
                          latency_improvement, throughput_improvement, confidence_improvement);
            
            if (latency_improvement > 5.0 && confidence_improvement > 0.0) {
                LOG_INFO_PRINT("Model B shows significant improvement, recommend promotion to production");
                logger_.update_model_stage(model_b_name_, ModelStage::PRODUCTION);
                logger_.update_model_stage(model_a_name_, ModelStage::ARCHIVED);
            } else {
                LOG_INFO_PRINT("Model A remains the better choice, keeping current deployment");
            }
        }
    }
};
```

### Example 3: Production Monitoring with Health Checks

```cpp
#include "common/src/logging.hpp"
#include <thread>

class ProductionHealthMonitor {
private:
    Logger& logger_ = Logger::get_instance();
    std::atomic<bool> monitoring_active_{true};
    std::vector<std::string> production_models_;
    
public:
    auto start_monitoring(const std::vector<std::string>& model_names) -> void {
        production_models_ = model_names;
        
        LOG_INFO_PRINT("Starting production health monitoring for {} models", model_names.size());
        
        // Start monitoring thread
        std::thread monitor_thread([this]() {
            while (monitoring_active_.load()) {
                check_model_health();
                check_performance_metrics();
                std::this_thread::sleep_for(std::chrono::seconds(30));
            }
        });
        
        monitor_thread.detach();
    }
    
    auto stop_monitoring() -> void {
        monitoring_active_.store(false);
        LOG_INFO_PRINT("Production health monitoring stopped");
    }
    
private:
    auto check_model_health() -> void {
        for (const auto& model_name : production_models_) {
            auto context = logger_.get_model_context(model_name);
            if (!context) {
                MLErrorContext error{
                    .error_code = "MODEL_NOT_REGISTERED",
                    .component = "HealthMonitor",
                    .operation = "check_registration",
                    .metadata = {{"model_name", model_name}}
                };
                LOG_ML_ERROR(model_name, error, "Model not found in registry");
                continue;
            }
            
            // Check if model is in production stage
            if (context->stage != ModelStage::PRODUCTION) {
                LOG_WARNING_PRINT("Model {} is not in PRODUCTION stage (current: {})", 
                                model_name, static_cast<int>(context->stage));
            }
            
            // Check model file integrity
            if (!verify_model_checksum(context->path, context->checksum)) {
                MLErrorContext error{
                    .error_code = "MODEL_INTEGRITY_FAILURE",
                    .component = "HealthMonitor",
                    .operation = "verify_checksum",
                    .metadata = {
                        {"model_path", context->path},
                        {"expected_checksum", context->checksum}
                    }
                };
                LOG_ML_ERROR(model_name, error, "Model file integrity check failed");
            }
        }
    }
    
    auto check_performance_metrics() -> void {
        // Flush current metrics for analysis
        logger_.flush_metrics_buffer();
        
        for (const auto& model_name : production_models_) {
            auto recent_metrics = logger_.get_aggregate_metrics(model_name, std::chrono::minutes(5));
            
            if (!recent_metrics) {
                LOG_WARNING_PRINT("No recent metrics available for model: {}", model_name);
                continue;
            }
            
            // Check latency SLA (< 100ms)
            if (recent_metrics->latency_ms > 100.0) {
                MLErrorContext error{
                    .error_code = "SLA_VIOLATION_LATENCY",
                    .component = "PerformanceMonitor",
                    .operation = "check_latency_sla",
                    .metadata = {
                        {"current_latency_ms", std::to_string(recent_metrics->latency_ms)},
                        {"sla_threshold_ms", "100.0"},
                        {"model_stage", "PRODUCTION"}
                    }
                };
                LOG_ML_ERROR(model_name, error, 
                           "Latency SLA violation: {:.2f}ms > 100.0ms", recent_metrics->latency_ms);
            }
            
            // Check throughput SLA (> 1000 samples/sec)
            if (recent_metrics->throughput < 1000.0) {
                MLErrorContext error{
                    .error_code = "SLA_VIOLATION_THROUGHPUT",
                    .component = "PerformanceMonitor", 
                    .operation = "check_throughput_sla",
                    .metadata = {
                        {"current_throughput", std::to_string(recent_metrics->throughput)},
                        {"sla_threshold", "1000.0"},
                        {"model_stage", "PRODUCTION"}
                    }
                };
                LOG_ML_ERROR(model_name, error,
                           "Throughput SLA violation: {:.1f}/s < 1000.0/s", recent_metrics->throughput);
            }
            
            // Check model confidence (> 0.90)
            if (recent_metrics->confidence < 0.90) {
                LOG_WARNING_PRINT("Model confidence below threshold: {} at {:.3f} (threshold: 0.90)",
                                model_name, recent_metrics->confidence);
            }
            
            // Log healthy status for successful checks
            if (recent_metrics->latency_ms <= 100.0 && recent_metrics->throughput >= 1000.0 && recent_metrics->confidence >= 0.90) {
                LOG_INFO_PRINT("Health check passed: {} (latency: {:.1f}ms, throughput: {:.1f}/s, confidence: {:.3f})",
                              model_name, recent_metrics->latency_ms, recent_metrics->throughput, recent_metrics->confidence);
            }
        }
    }
    
    auto verify_model_checksum(const std::string& path, const std::string& expected_checksum) -> bool {
        // Implementation would calculate actual checksum and compare
        // Returning true for example purposes
        return true;
    }
};

// Usage example
int main() {
    // Configure production logging
    auto& logger = Logger::get_instance("/var/log/ml/production.log");
    logger.set_ml_logging_enabled(true);
    logger.set_max_metrics_buffer_size(10000);
    Logger::set_level_enabled(LogLevel::DEBUG, false);
    
    // Register production models
    std::vector<std::string> production_models = {
        "image_classifier_v2", 
        "object_detector_v1", 
        "text_embedder_v3"
    };
    
    for (const auto& model_name : production_models) {
        ModelContext context{
            .name = model_name,
            .version = "1.0.0",
            .framework = "TensorRT",
            .stage = ModelStage::PRODUCTION,
            .path = "/models/" + model_name + ".trt",
            .size_mb = 512,
            .loaded_at = std::chrono::system_clock::now()
        };
        logger.register_model(context);
    }
    
    // Start health monitoring
    ProductionHealthMonitor monitor;
    monitor.start_monitoring(production_models);
    
    LOG_INFO_PRINT("Production monitoring system started");
    
    // Keep running...
    std::this_thread::sleep_for(std::chrono::hours(24));
    
    monitor.stop_monitoring();
    return 0;
}
```

---

*This documentation represents the complete technical reference for the Logger class in the Inference Systems Laboratory. For additional examples, implementation details, or integration support, please refer to the test suites in `common/tests/test_ml_logging.cpp` and example applications in `common/examples/`.*

**Document Version**: 1.0.0  
**Last Updated**: January 2025  
**Maintainer**: Inference Systems Laboratory Team  
