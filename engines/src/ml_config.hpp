// MIT License
// Copyright (c) 2025 Inference Systems Lab

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Machine Learning Framework Configuration
// This header provides compile-time and runtime detection of available ML frameworks

namespace inference_lab {
namespace engines {
namespace ml {

// Compile-time ML framework availability
#ifdef ENABLE_TENSORRT
constexpr bool has_tensorrt = true;
#else
constexpr bool has_tensorrt = false;
#endif

#ifdef ENABLE_ONNX_RUNTIME
constexpr bool has_onnx_runtime = true;
#else
constexpr bool has_onnx_runtime = false;
#endif

#ifdef ENABLE_ML_FRAMEWORKS
constexpr bool has_ml_frameworks = (ENABLE_ML_FRAMEWORKS > 0);
constexpr size_t ml_framework_count = ML_FRAMEWORKS_COUNT;
#else
constexpr bool has_ml_frameworks = false;
constexpr size_t ml_framework_count = 0;
#endif

// ML Backend enumeration
enum class MLBackend : uint8_t { CPU_ONLY, TENSORRT_GPU, ONNX_RUNTIME, HYBRID_NEURAL_SYMBOLIC };

// ML Framework capabilities
struct MLCapabilities {
    bool tensorrt_available = has_tensorrt;
    bool onnx_runtime_available = has_onnx_runtime;
    bool gpu_acceleration = has_tensorrt;
    bool cross_platform_inference = has_onnx_runtime;
    size_t framework_count = ml_framework_count;

    // Get string representation of capabilities
    auto to_string() const -> std::string;

    // Check if specific backend is available
    auto is_backend_available(MLBackend backend) const -> bool;

    // Get recommended backend based on available frameworks
    auto get_recommended_backend() const -> MLBackend;
};

// Global capabilities instance
extern const MLCapabilities capabilities;

// Utility functions
auto get_available_backends() -> std::vector<MLBackend>;
auto backend_to_string(MLBackend backend) -> std::string;
auto detect_optimal_backend() -> MLBackend;

}  // namespace ml
}  // namespace engines
}  // namespace inference_lab
