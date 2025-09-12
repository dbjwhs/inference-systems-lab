// MIT License
// Copyright (c) 2025 dbjwhs
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../common/src/logging.hpp"
#include "../../common/src/ml_types.hpp"
#include "../../common/src/result.hpp"
#include "../inference_engine.hpp"

#ifdef ENABLE_ONNX_RUNTIME
    #include <onnxruntime_cxx_api.h>
#endif

namespace inference_lab {
namespace engines {
namespace onnx {

using common::Result;
using common::ml::DataType;
using common::ml::FloatTensor;
using common::ml::Shape;

// ONNX Runtime execution providers
enum class ExecutionProvider : uint8_t {
    CPU,       // Default CPU execution
    CUDA,      // NVIDIA GPU (if available)
    DIRECTML,  // DirectML (Windows)
    COREML,    // Apple CoreML (macOS/iOS)
    TENSORRT,  // TensorRT (if available)
    OPENVINO,  // Intel OpenVINO
    ROCM,      // AMD ROCm
    AUTO       // Automatic selection
};

// ONNX model optimization levels
enum class GraphOptimizationLevel : uint8_t {
    ORT_DISABLE_ALL,      // Disable all optimizations
    ORT_ENABLE_BASIC,     // Enable basic optimizations
    ORT_ENABLE_EXTENDED,  // Enable extended optimizations
    ORT_ENABLE_ALL        // Enable all available optimizations
};

// Model input/output information
struct TensorInfo {
    std::string name;
    DataType data_type;
    Shape shape;
    bool is_dynamic = false;

    TensorInfo() = default;
    TensorInfo(std::string name, DataType type, Shape shape);
};

// ONNX Runtime configuration
struct ONNXRuntimeConfig {
    // Execution configuration
    ExecutionProvider provider = ExecutionProvider::AUTO;
    GraphOptimizationLevel optimization_level = GraphOptimizationLevel::ORT_ENABLE_ALL;

    // Performance tuning
    int intra_op_num_threads = 0;  // 0 = use default (number of cores)
    int inter_op_num_threads = 0;  // 0 = use default

    // Memory management
    bool enable_cpu_mem_arena = true;
    bool enable_mem_pattern = true;

    // Profiling and debugging
    bool enable_profiling = false;
    std::string profile_file_prefix = "onnx_profile";

    // Provider-specific options
    std::unordered_map<std::string, std::string> provider_options;
};

// Performance metrics for ONNX inference
struct ONNXMetrics {
    // Model information
    std::string model_path;
    size_t model_size_bytes = 0;

    // Execution metrics
    std::chrono::microseconds inference_time_us{0};
    std::chrono::microseconds preprocessing_time_us{0};
    std::chrono::microseconds postprocessing_time_us{0};

    // Provider information
    ExecutionProvider active_provider = ExecutionProvider::CPU;
    std::vector<std::string> available_providers;

    // Memory usage
    size_t memory_usage_bytes = 0;
    size_t peak_memory_bytes = 0;

    // Throughput metrics
    double inferences_per_second = 0.0;
    size_t total_inferences = 0;
};

// Error types for ONNX Runtime
enum class ONNXError : uint8_t {
    MODEL_LOAD_FAILED,
    INVALID_MODEL_PATH,
    UNSUPPORTED_MODEL_FORMAT,
    EXECUTION_PROVIDER_ERROR,
    INPUT_SHAPE_MISMATCH,
    OUTPUT_SHAPE_MISMATCH,
    INFERENCE_EXECUTION_FAILED,
    MEMORY_ALLOCATION_FAILED,
    UNSUPPORTED_DATA_TYPE,
    CONFIGURATION_ERROR,
    UNKNOWN_ERROR
};

auto to_string(ONNXError error) -> std::string;
auto to_string(ExecutionProvider provider) -> std::string;
auto to_string(GraphOptimizationLevel level) -> std::string;

// Forward declarations for PIMPL pattern
namespace detail {
#ifdef ENABLE_ONNX_RUNTIME
class ONNXRuntimeImpl;
#else
class ONNXRuntimeStub;
#endif
}  // namespace detail

// Main ONNX Runtime inference engine
class ONNXRuntimeEngine : public InferenceEngine {
  public:
    explicit ONNXRuntimeEngine(const ONNXRuntimeConfig& config = ONNXRuntimeConfig{});
    ~ONNXRuntimeEngine() override;

    // Disable copy, enable move
    ONNXRuntimeEngine(const ONNXRuntimeEngine&) = delete;
    ONNXRuntimeEngine& operator=(const ONNXRuntimeEngine&) = delete;
    // Move operations deleted due to base class InferenceEngine
    ONNXRuntimeEngine(ONNXRuntimeEngine&&) = delete;
    ONNXRuntimeEngine& operator=(ONNXRuntimeEngine&&) = delete;

    // Model management
    auto load_model(const std::string& model_path) -> Result<bool, ONNXError>;
    auto load_model_from_buffer(const std::vector<uint8_t>& buffer,
                                const std::string& model_name = "buffer_model")
        -> Result<bool, ONNXError>;

    // Model inspection
    auto get_input_info() const -> Result<std::vector<TensorInfo>, ONNXError>;
    auto get_output_info() const -> Result<std::vector<TensorInfo>, ONNXError>;
    auto get_model_metadata() const
        -> Result<std::unordered_map<std::string, std::string>, ONNXError>;

    // Inference operations
    auto run_inference(const std::vector<FloatTensor>& inputs)
        -> Result<std::vector<FloatTensor>, ONNXError>;
    auto run_inference(const std::unordered_map<std::string, FloatTensor>& named_inputs)
        -> Result<std::unordered_map<std::string, FloatTensor>, ONNXError>;

    // Configuration management
    void update_config(const ONNXRuntimeConfig& new_config);
    auto get_config() const -> const ONNXRuntimeConfig&;

    // Provider management
    auto get_available_providers() const -> std::vector<ExecutionProvider>;
    auto set_execution_provider(ExecutionProvider provider) -> Result<bool, ONNXError>;
    auto get_current_provider() const -> ExecutionProvider;

    // Performance monitoring
    auto get_metrics() const -> const ONNXMetrics&;
    void reset_metrics();

    // Utility methods
    auto is_model_loaded() const -> bool;
    void warm_up(size_t iterations = 5);

    // InferenceEngine interface implementation
    auto run_inference(const InferenceRequest& request)
        -> Result<InferenceResponse, InferenceError> override;
    auto get_backend_info() const -> std::string override;
    auto get_performance_stats() const -> std::string override;
    bool is_ready() const override;

  private:
    // PIMPL pattern to hide ONNX Runtime dependencies when not available
#ifdef ENABLE_ONNX_RUNTIME
    std::unique_ptr<detail::ONNXRuntimeImpl> impl_;
#else
    std::unique_ptr<detail::ONNXRuntimeStub> impl_;
#endif

    ONNXRuntimeConfig config_;
    ONNXMetrics metrics_;

    static constexpr const char* LOGGER_NAME = "ONNXEngine";
};

// Factory functions
auto create_onnx_engine(const ONNXRuntimeConfig& config = ONNXRuntimeConfig{})
    -> Result<std::unique_ptr<ONNXRuntimeEngine>, ONNXError>;

auto create_onnx_engine_from_model(const std::string& model_path,
                                   const ONNXRuntimeConfig& config = ONNXRuntimeConfig{})
    -> Result<std::unique_ptr<ONNXRuntimeEngine>, ONNXError>;

// Provider utility functions
auto detect_optimal_provider() -> ExecutionProvider;
auto get_system_providers() -> std::vector<ExecutionProvider>;
auto is_provider_available(ExecutionProvider provider) -> bool;

// Model format validation
auto validate_onnx_model(const std::string& model_path) -> Result<bool, ONNXError>;
auto get_model_info(const std::string& model_path)
    -> Result<std::pair<std::vector<TensorInfo>, std::vector<TensorInfo>>, ONNXError>;

// Utility functions for testing and benchmarking
namespace testing {
auto create_test_model(const std::vector<TensorInfo>& inputs,
                       const std::vector<TensorInfo>& outputs)
    -> Result<std::vector<uint8_t>, ONNXError>;

auto benchmark_providers(const std::string& model_path,
                         const std::vector<ExecutionProvider>& providers,
                         size_t iterations = 100)
    -> std::vector<std::pair<ExecutionProvider, ONNXMetrics>>;

auto generate_test_inputs(const std::vector<TensorInfo>& input_info) -> std::vector<FloatTensor>;
}  // namespace testing

}  // namespace onnx
}  // namespace engines
}  // namespace inference_lab
