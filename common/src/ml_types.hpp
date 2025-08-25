// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file ml_types.hpp
 * @brief ML-specific type definitions and utilities for inference systems
 *
 * This header provides comprehensive type definitions and utilities specifically
 * designed for machine learning inference workloads. It bridges traditional
 * symbolic AI concepts with modern ML/neural network types, enabling hybrid
 * inference systems.
 *
 * Key Features:
 * - Unified tensor types with automatic memory management
 * - Neural network layer and model abstractions
 * - Probability distributions and uncertainty quantification
 * - Type-safe model metadata and configuration
 * - Performance-optimized batch processing types
 * - Integration with existing Result<T,E> error handling
 *
 * Design Principles:
 * - Zero-cost abstractions where possible
 * - RAII resource management for GPU/CPU memory
 * - Template-based generic programming for performance
 * - Clear separation between compile-time and runtime types
 * - Seamless integration with existing container types
 *
 * Usage Example:
 * @code
 * // Create a neural network input tensor
 * auto input = FloatTensor::zeros({1, 3, 224, 224});  // NCHW format
 *
 * // Define model configuration
 * ModelConfig config{
 *     .name = "resnet50",
 *     .input_shapes = {{1, 3, 224, 224}},
 *     .output_shapes = {{1, 1000}},
 *     .precision = Precision::FP16,
 *     .max_batch_size = 4
 * };
 *
 * // Run inference with uncertainty quantification
 * auto result = model.predict_with_uncertainty(input);
 * if (result.is_ok()) {
 *     auto [predictions, confidence] = result.unwrap();
 *     // Process predictions...
 * }
 * @endcode
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "containers.hpp"
#include "result.hpp"

namespace inference_lab::common::ml {

//=============================================================================
// Core Type Definitions
//=============================================================================

/**
 * @brief Supported data types for ML tensors
 */
enum class DataType : std::uint8_t {
    FLOAT32 = 0,    ///< 32-bit floating point (IEEE 754)
    FLOAT16 = 1,    ///< 16-bit floating point (half precision)
    INT32 = 2,      ///< 32-bit signed integer
    INT16 = 3,      ///< 16-bit signed integer
    INT8 = 4,       ///< 8-bit signed integer
    UINT8 = 5,      ///< 8-bit unsigned integer
    BOOL = 6,       ///< Boolean type
    FLOAT64 = 7,    ///< 64-bit floating point (double precision)
    COMPLEX64 = 8,  ///< 64-bit complex number (2x float32)
    COMPLEX128 = 9  ///< 128-bit complex number (2x float64)
};

/**
 * @brief Get size in bytes for a data type
 */
constexpr std::size_t get_dtype_size(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::COMPLEX64:
            return 4;
        case DataType::FLOAT16:
        case DataType::INT16:
            return 2;
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
            return 1;
        case DataType::FLOAT64:
        case DataType::COMPLEX128:
            return 8;
        default:
            return 0;
    }
}

/**
 * @brief Get string representation of data type
 */
constexpr const char* dtype_to_string(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::FLOAT32:
            return "float32";
        case DataType::FLOAT16:
            return "float16";
        case DataType::INT32:
            return "int32";
        case DataType::INT16:
            return "int16";
        case DataType::INT8:
            return "int8";
        case DataType::UINT8:
            return "uint8";
        case DataType::BOOL:
            return "bool";
        case DataType::FLOAT64:
            return "float64";
        case DataType::COMPLEX64:
            return "complex64";
        case DataType::COMPLEX128:
            return "complex128";
        default:
            return "unknown";
    }
}

/**
 * @brief Inference precision settings
 */
enum class Precision : std::uint8_t {
    FP32 = 0,  ///< Full 32-bit precision
    FP16 = 1,  ///< Half precision (faster on modern GPUs)
    INT8 = 2,  ///< Quantized 8-bit inference
    MIXED = 3  ///< Mixed precision (FP16 compute, FP32 accumulate)
};

// InferenceBackend enum moved to engines/src/inference_engine.hpp to avoid conflicts

/**
 * @brief Tensor shape type (dimensions)
 */
using Shape = std::vector<std::size_t>;

/**
 * @brief Batch size type
 */
using BatchSize = std::uint32_t;

/**
 * @brief Confidence/probability type
 */
using Confidence = float;

//=============================================================================
// Tensor Types and Utilities
//=============================================================================

/**
 * @brief Generic ML tensor with automatic memory management
 *
 * This is a specialized version of TensorContainer optimized for ML workloads.
 * It provides additional functionality for common ML operations while maintaining
 * compatibility with the existing container framework.
 *
 * @tparam T Element type (float, int, etc.)
 * @tparam Allocator Memory allocator type
 */
template <typename T, typename Allocator = MemoryPool<T>>
class MLTensor : public TensorContainer<T, Allocator> {
  public:
    using ElementType = T;
    using AllocatorType = Allocator;
    using BaseType = TensorContainer<T, Allocator>;

    // Inherit constructors from base class
    using BaseType::BaseType;

    /**
     * @brief Create tensor with specific data type
     */
    static auto create(const Shape& shape,
                       DataType dtype,
                       AllocatorType allocator = AllocatorType(1024))
        -> Result<MLTensor, std::string> {
        if (!is_compatible_dtype<T>(dtype)) {
            return Err(std::string("Incompatible data type for tensor element type"));
        }
        return Ok(MLTensor(shape, std::move(allocator)));
    }

    /**
     * @brief Get data type of tensor
     */
    auto dtype() const -> DataType { return get_dtype<T>(); }

    /**
     * @brief Get total memory usage including metadata
     */
    auto total_memory_usage() const -> std::size_t { return this->memory_usage() + sizeof(*this); }

    /**
     * @brief Reshape tensor with validation
     */
    auto reshape_safe(const Shape& new_shape) -> Result<bool, std::string> {
        std::size_t new_total = 1;
        for (auto dim : new_shape) {
            if (dim == 0) {
                return Err(std::string("Invalid dimension: zero size not allowed"));
            }
            new_total *= dim;
        }

        if (new_total != this->size()) {
            return Err(std::string("New shape total size doesn't match current size"));
        }

        if (this->reshape(new_shape)) {
            return Ok(true);
        } else {
            return Err(std::string("Reshape failed"));
        }
    }

    /**
     * @brief Extract a batch from the tensor
     */
    auto extract_batch(std::size_t batch_idx, std::size_t batch_size)
        -> Result<MLTensor, std::string> {
        if (this->ndim() == 0) {
            return Err(std::string("Cannot extract batch from scalar tensor"));
        }

        auto shape = this->shape();
        if (batch_idx >= shape[0]) {
            return Err(std::string("Batch index out of range"));
        }

        if (batch_idx + batch_size > shape[0]) {
            return Err(std::string("Batch size exceeds tensor bounds"));
        }

        // Create new shape for batch
        Shape batch_shape = shape;
        batch_shape[0] = batch_size;

        MLTensor batch_tensor(batch_shape, AllocatorType(1024));

        // Copy data for the specified batch
        std::size_t elements_per_batch = this->size() / shape[0];
        std::size_t start_idx = batch_idx * elements_per_batch;
        std::size_t copy_elements = batch_size * elements_per_batch;

        std::copy(this->data() + start_idx,
                  this->data() + start_idx + copy_elements,
                  batch_tensor.data());

        return Ok(std::move(batch_tensor));
    }

  private:
    /**
     * @brief Check if data type is compatible with element type
     */
    template <typename U>
    static constexpr bool is_compatible_dtype(DataType dtype) {
        if constexpr (std::is_same_v<U, float>) {
            return dtype == DataType::FLOAT32;
        } else if constexpr (std::is_same_v<U, double>) {
            return dtype == DataType::FLOAT64;
        } else if constexpr (std::is_same_v<U, std::int32_t>) {
            return dtype == DataType::INT32;
        } else if constexpr (std::is_same_v<U, std::int16_t>) {
            return dtype == DataType::INT16;
        } else if constexpr (std::is_same_v<U, std::int8_t>) {
            return dtype == DataType::INT8;
        } else if constexpr (std::is_same_v<U, std::uint8_t>) {
            return dtype == DataType::UINT8;
        } else if constexpr (std::is_same_v<U, bool>) {
            return dtype == DataType::BOOL;
        } else {
            return false;
        }
    }

    /**
     * @brief Get data type for element type
     */
    template <typename U>
    static constexpr DataType get_dtype() {
        if constexpr (std::is_same_v<U, float>) {
            return DataType::FLOAT32;
        } else if constexpr (std::is_same_v<U, double>) {
            return DataType::FLOAT64;
        } else if constexpr (std::is_same_v<U, std::int32_t>) {
            return DataType::INT32;
        } else if constexpr (std::is_same_v<U, std::int16_t>) {
            return DataType::INT16;
        } else if constexpr (std::is_same_v<U, std::int8_t>) {
            return DataType::INT8;
        } else if constexpr (std::is_same_v<U, std::uint8_t>) {
            return DataType::UINT8;
        } else if constexpr (std::is_same_v<U, bool>) {
            return DataType::BOOL;
        } else {
            static_assert(std::is_same_v<U, float>, "Unsupported data type");
            return DataType::FLOAT32;  // Fallback
        }
    }
};

//=============================================================================
// Common Tensor Type Aliases
//=============================================================================

using FloatTensor = MLTensor<float>;
using DoubleTensor = MLTensor<double>;
using IntTensor = MLTensor<std::int32_t>;
using ByteTensor = MLTensor<std::uint8_t>;
using BoolTensor = MLTensor<bool>;

//=============================================================================
// Model Configuration and Metadata
//=============================================================================

/**
 * @brief Model input/output specification
 */
struct TensorSpec {
    std::string name;  ///< Tensor name (e.g., "input", "logits")
    Shape shape;       ///< Tensor dimensions
    DataType dtype;    ///< Data type
    bool is_dynamic;   ///< Whether shape can change at runtime
    Shape min_shape;   ///< Minimum shape (for dynamic tensors)
    Shape max_shape;   ///< Maximum shape (for dynamic tensors)

    /**
     * @brief Get total number of elements
     */
    auto num_elements() const -> std::size_t {
        std::size_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * @brief Get memory size in bytes
     */
    auto memory_size() const -> std::size_t { return num_elements() * get_dtype_size(dtype); }

    /**
     * @brief Check if tensor spec is valid
     */
    auto is_valid() const -> bool {
        if (name.empty() || shape.empty()) {
            return false;
        }

        for (auto dim : shape) {
            if (dim == 0) {
                return false;
            }
        }

        if (is_dynamic) {
            return min_shape.size() == shape.size() && max_shape.size() == shape.size();
        }

        return true;
    }
};

/**
 * @brief Model performance requirements
 */
struct PerformanceRequirements {
    std::optional<float> max_latency_ms;       ///< Maximum acceptable latency
    std::optional<float> min_throughput_fps;   ///< Minimum throughput requirement
    std::optional<std::size_t> max_memory_mb;  ///< Maximum memory usage
    std::optional<float> min_accuracy;         ///< Minimum accuracy requirement
    bool prefer_low_latency = true;            ///< Optimize for latency vs throughput
};

/**
 * @brief Comprehensive model configuration
 */
struct ModelConfig {
    // Basic model information
    std::string name;                      ///< Model name
    std::string version = "1.0.0";         ///< Model version
    std::string description;               ///< Human-readable description
    std::vector<TensorSpec> input_specs;   ///< Input tensor specifications
    std::vector<TensorSpec> output_specs;  ///< Output tensor specifications

    // Runtime configuration
    Precision precision = Precision::FP32;  ///< Computation precision
    BatchSize max_batch_size = 1;           ///< Maximum batch size
    std::uint32_t max_sequence_length = 0;  ///< For sequence models (0 = not applicable)

    // Model file paths
    std::string model_path;    ///< Path to model file (.onnx, .trt, etc.)
    std::string config_path;   ///< Path to config file (optional)
    std::string weights_path;  ///< Path to weights file (optional)

    // Performance and optimization
    PerformanceRequirements performance;  ///< Performance requirements
    bool enable_optimization = true;      ///< Enable backend-specific optimizations
    bool enable_profiling = false;        ///< Enable performance profiling
    std::uint32_t gpu_device_id = 0;      ///< GPU device ID (if applicable)

    // Model-specific metadata
    std::unordered_map<std::string, std::string> metadata;  ///< Additional metadata

    /**
     * @brief Validate model configuration
     */
    auto validate() const -> Result<bool, std::string> {
        if (name.empty()) {
            return Err(std::string("Model name cannot be empty"));
        }

        if (model_path.empty()) {
            return Err(std::string("Model path cannot be empty"));
        }

        if (input_specs.empty()) {
            return Err(std::string("At least one input specification required"));
        }

        if (output_specs.empty()) {
            return Err(std::string("At least one output specification required"));
        }

        for (const auto& spec : input_specs) {
            if (!spec.is_valid()) {
                return Err(std::string("Invalid input specification: " + spec.name));
            }
        }

        for (const auto& spec : output_specs) {
            if (!spec.is_valid()) {
                return Err(std::string("Invalid output specification: " + spec.name));
            }
        }

        if (max_batch_size == 0) {
            return Err(std::string("Max batch size must be greater than 0"));
        }

        return Ok(true);
    }

    /**
     * @brief Get total input memory requirement
     */
    auto total_input_memory() const -> std::size_t {
        std::size_t total = 0;
        for (const auto& spec : input_specs) {
            total += spec.memory_size() * max_batch_size;
        }
        return total;
    }

    /**
     * @brief Get total output memory requirement
     */
    auto total_output_memory() const -> std::size_t {
        std::size_t total = 0;
        for (const auto& spec : output_specs) {
            total += spec.memory_size() * max_batch_size;
        }
        return total;
    }
};

//=============================================================================
// Inference Request and Response Types
//=============================================================================

/**
 * @brief Single tensor input for inference
 */
struct TensorInput {
    std::string name;                                                     ///< Input tensor name
    std::variant<FloatTensor, IntTensor, ByteTensor, BoolTensor> tensor;  ///< Actual tensor data
    std::unordered_map<std::string, std::string> metadata;  ///< Input-specific metadata

    // Make it move-only to match tensor semantics
    TensorInput() = default;
    TensorInput(const TensorInput&) = delete;
    TensorInput& operator=(const TensorInput&) = delete;
    TensorInput(TensorInput&&) = default;
    TensorInput& operator=(TensorInput&&) = default;

    /**
     * @brief Get tensor shape
     */
    auto get_shape() const -> Shape {
        return std::visit([](const auto& t) { return t.shape(); }, tensor);
    }

    /**
     * @brief Get tensor data type
     */
    auto get_dtype() const -> DataType {
        return std::visit([](const auto& t) { return t.dtype(); }, tensor);
    }

    /**
     * @brief Get tensor size
     */
    auto get_size() const -> std::size_t {
        return std::visit([](const auto& t) { return t.size(); }, tensor);
    }
};

/**
 * @brief Inference request containing multiple inputs
 */
struct InferenceRequest {
    std::vector<TensorInput> inputs;                        ///< Input tensors
    BatchSize batch_size = 1;                               ///< Actual batch size
    std::unordered_map<std::string, std::string> metadata;  ///< Request metadata
    std::optional<std::uint64_t> request_id;                ///< Unique request identifier

    // Make it move-only to match TensorInput semantics
    InferenceRequest() = default;
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
    InferenceRequest(InferenceRequest&&) = default;
    InferenceRequest& operator=(InferenceRequest&&) = default;

    /**
     * @brief Validate inference request
     */
    auto validate(const ModelConfig& config) const -> Result<bool, std::string> {
        if (inputs.empty()) {
            return Err(std::string("No inputs provided"));
        }

        if (batch_size == 0) {
            return Err(std::string("Batch size must be greater than 0"));
        }

        if (batch_size > config.max_batch_size) {
            return Err(std::string("Batch size exceeds model maximum"));
        }

        if (inputs.size() != config.input_specs.size()) {
            return Err(std::string("Input count mismatch"));
        }

        // Validate each input against corresponding spec
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            const auto& spec = config.input_specs[i];

            if (input.name != spec.name) {
                return Err(std::string("Input name mismatch: expected " + spec.name + ", got " +
                                       input.name));
            }

            auto input_shape = input.get_shape();
            if (!spec.is_dynamic && input_shape != spec.shape) {
                return Err(std::string("Input shape mismatch for " + spec.name));
            }

            if (input.get_dtype() != spec.dtype) {
                return Err(std::string("Input data type mismatch for " + spec.name));
            }
        }

        return Ok(true);
    }
};

/**
 * @brief Single tensor output from inference
 */
struct TensorOutput {
    std::string name;                                                     ///< Output tensor name
    std::variant<FloatTensor, IntTensor, ByteTensor, BoolTensor> tensor;  ///< Result tensor data
    Confidence confidence = 1.0f;  ///< Confidence in the output (0.0 to 1.0)
    std::unordered_map<std::string, std::string> metadata;  ///< Output-specific metadata

    // Make it move-only to match tensor semantics
    TensorOutput() = default;
    TensorOutput(const TensorOutput&) = delete;
    TensorOutput& operator=(const TensorOutput&) = delete;
    TensorOutput(TensorOutput&&) = default;
    TensorOutput& operator=(TensorOutput&&) = default;

    /**
     * @brief Get tensor shape
     */
    auto get_shape() const -> Shape {
        return std::visit([](const auto& t) { return t.shape(); }, tensor);
    }

    /**
     * @brief Get tensor data type
     */
    auto get_dtype() const -> DataType {
        return std::visit([](const auto& t) { return t.dtype(); }, tensor);
    }
};

/**
 * @brief Inference response containing outputs and metadata
 */
struct InferenceResponse {
    std::vector<TensorOutput> outputs;                      ///< Output tensors
    std::chrono::milliseconds inference_time;               ///< Time taken for inference
    Confidence overall_confidence = 1.0f;                   ///< Overall confidence in results
    std::unordered_map<std::string, std::string> metadata;  ///< Response metadata
    std::optional<std::uint64_t> request_id;                ///< Corresponding request ID

    // Make it move-only to match TensorOutput semantics
    InferenceResponse() = default;
    InferenceResponse(const InferenceResponse&) = delete;
    InferenceResponse& operator=(const InferenceResponse&) = delete;
    InferenceResponse(InferenceResponse&&) = default;
    InferenceResponse& operator=(InferenceResponse&&) = default;

    /**
     * @brief Get output by name
     */
    auto get_output(const std::string& name) const
        -> std::optional<std::reference_wrapper<const TensorOutput>> {
        for (const auto& output : outputs) {
            if (output.name == name) {
                return std::cref(output);
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Get total memory usage of outputs
     */
    auto total_output_memory() const -> std::size_t {
        std::size_t total = 0;
        for (const auto& output : outputs) {
            total += std::visit([](const auto& t) { return t.memory_usage(); }, output.tensor);
        }
        return total;
    }
};

//=============================================================================
// Probability and Uncertainty Types
//=============================================================================

/**
 * @brief Probability distribution over discrete classes
 */
struct ClassificationResult {
    std::vector<float> probabilities;  ///< Probability for each class
    std::vector<std::string> labels;   ///< Class labels (optional)
    std::size_t predicted_class;       ///< Index of most likely class
    float max_probability;             ///< Probability of predicted class

    /**
     * @brief Get top-k predictions
     */
    auto top_k(std::size_t k) const -> std::vector<std::pair<std::size_t, float>> {
        std::vector<std::pair<std::size_t, float>> indexed_probs;
        indexed_probs.reserve(probabilities.size());

        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            indexed_probs.emplace_back(i, probabilities[i]);
        }

        std::size_t sort_k = std::min(k, indexed_probs.size());
        std::partial_sort(indexed_probs.begin(),
                          indexed_probs.begin() + sort_k,
                          indexed_probs.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });

        indexed_probs.resize(std::min(k, indexed_probs.size()));
        return indexed_probs;
    }
};

/**
 * @brief Uncertainty quantification for model predictions
 */
struct UncertaintyEstimate {
    float epistemic_uncertainty;   ///< Model uncertainty (reducible with more data)
    float aleatoric_uncertainty;   ///< Data uncertainty (irreducible noise)
    float total_uncertainty;       ///< Combined uncertainty
    float confidence_interval_95;  ///< 95% confidence interval width

    /**
     * @brief Check if prediction is reliable
     */
    auto is_reliable(float threshold = 0.1f) const -> bool { return total_uncertainty < threshold; }
};

/**
 * @brief Batch processing result with per-sample statistics
 */
struct BatchResult {
    std::vector<TensorOutput> batch_outputs;         ///< Outputs for each sample in batch
    std::vector<UncertaintyEstimate> uncertainties;  ///< Uncertainty for each sample
    std::chrono::milliseconds total_time;            ///< Total batch processing time
    std::chrono::milliseconds avg_per_sample_time;   ///< Average time per sample
    float batch_efficiency;                          ///< Batch efficiency (0.0 to 1.0)

    /**
     * @brief Get throughput in samples per second
     */
    auto get_throughput() const -> float {
        if (total_time.count() == 0) {
            return 0.0f;
        }
        return (batch_outputs.size() * 1000.0f) / total_time.count();
    }
};

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Create tensor factory functions
 */
namespace tensor_factory {

/**
 * @brief Create a zero-filled tensor
 */
template <typename T = float>
auto zeros(const Shape& shape) -> MLTensor<T> {
    MLTensor<T> tensor(shape);
    tensor.zero();
    return tensor;
}

/**
 * @brief Create a one-filled tensor
 */
template <typename T = float>
auto ones(const Shape& shape) -> MLTensor<T> {
    MLTensor<T> tensor(shape);
    tensor.fill(static_cast<T>(1));
    return tensor;
}

/**
 * @brief Create a random tensor with uniform distribution
 */
template <typename T = float>
auto random_uniform(const Shape& shape, T min_val = T(0), T max_val = T(1)) -> MLTensor<T> {
    MLTensor<T> tensor(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(min_val, max_val);

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor.data()[i] = dist(gen);
    }

    return tensor;
}

/**
 * @brief Create a tensor from existing data
 */
template <typename T>
auto from_data(const Shape& shape, const std::vector<T>& data) -> Result<MLTensor<T>, std::string> {
    std::size_t expected_size = 1;
    for (auto dim : shape) {
        expected_size *= dim;
    }

    if (data.size() != expected_size) {
        return Err(std::string("Data size doesn't match shape"));
    }

    MLTensor<T> tensor(shape);
    std::copy(data.begin(), data.end(), tensor.data());
    return Ok(std::move(tensor));
}

}  // namespace tensor_factory

/**
 * @brief Convert between data types
 */
template <typename FromT, typename ToT>
auto convert_tensor(const MLTensor<FromT>& input) -> MLTensor<ToT> {
    MLTensor<ToT> output(input.shape());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output.data()[i] = static_cast<ToT>(input.data()[i]);
    }
    return output;
}

/**
 * @brief Calculate tensor statistics
 */
struct TensorStats {
    float mean;
    float std_dev;
    float min_val;
    float max_val;
    std::size_t non_zero_count;
};

template <typename T>
auto calculate_stats(const MLTensor<T>& tensor) -> TensorStats {
    if (tensor.empty()) {
        return {0.0f, 0.0f, 0.0f, 0.0f, 0};
    }

    T min_val = tensor.data()[0];
    T max_val = tensor.data()[0];
    double sum = 0.0;
    std::size_t non_zero = 0;

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        T val = tensor.data()[i];
        sum += static_cast<double>(val);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        if (val != T(0)) {
            ++non_zero;
        }
    }

    float mean = static_cast<float>(sum / tensor.size());

    // Calculate standard deviation
    double variance_sum = 0.0;
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        double diff = static_cast<double>(tensor.data()[i]) - mean;
        variance_sum += diff * diff;
    }

    float std_dev = static_cast<float>(std::sqrt(variance_sum / tensor.size()));

    return {mean, std_dev, static_cast<float>(min_val), static_cast<float>(max_val), non_zero};
}

}  // namespace inference_lab::common::ml
