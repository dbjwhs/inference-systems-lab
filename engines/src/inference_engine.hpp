// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file inference_engine.hpp
 * @brief Unified inference interface for rule-based and ML inference systems
 *
 * This header defines the core abstraction layer that enables seamless integration
 * between traditional rule-based inference engines and modern machine learning
 * inference systems (TensorRT, ONNX Runtime). The interface maintains consistency
 * with the project's Result<T,E> error handling patterns and provides a foundation
 * for neural-symbolic reasoning systems.
 *
 * Key Design Principles:
 * - Unified API across all inference backends (rule-based, TensorRT, ONNX)
 * - Integration with existing Result<T,E> error handling patterns
 * - RAII resource management for GPU memory and model lifecycle
 * - Performance-oriented with zero-cost abstractions where possible
 * - Extensible plugin architecture for custom inference backends
 *
 * Usage Example:
 * @code
 * auto engine = create_inference_engine(InferenceBackend::TENSORRT_GPU, config)
 *     .map_err([](InferenceError err) {
 *         LOG_ERROR_PRINT("Engine creation failed: {}", to_string(err));
 *         return err;
 *     });
 *
 * if (engine.is_ok()) {
 *     auto result = engine.unwrap()->run_inference(request);
 *     // Handle inference result...
 * }
 * @endcode
 *
 * Architecture Flow:
 * @code
 *   ┌─────────────┐   create_engine()   ┌─────────────────┐
 *   │ User Code   │ ─────────────────▶ │ Factory Method  │
 *   │             │                    │ Backend         │
 *   └─────────────┘                    │ Selection       │
 *          │                           └─────────────────┘
 *          │ run_inference()                     │
 *          ▼                                     ▼
 *   ┌─────────────┐                    ┌─────────────────┐
 *   │ Inference   │◄───────────────────│ Concrete Engine │
 *   │ Response    │    Result<T,E>     │ • TensorRT      │
 *   └─────────────┘                    │ • ONNX Runtime  │
 *                                      │ • Rule-Based    │
 *                                      └─────────────────┘
 * @endcode
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../common/src/result.hpp"

namespace inference_lab::engines {

/**
 * @brief Supported inference backend types
 *
 * Defines the available inference execution backends, from traditional rule-based
 * systems to modern GPU-accelerated ML frameworks. The enum is designed to be
 * extensible for future backend additions while maintaining backward compatibility.
 */
enum class InferenceBackend : std::uint8_t {
    RULE_BASED,             ///< Traditional forward/backward chaining inference
    TENSORRT_GPU,           ///< NVIDIA TensorRT GPU-accelerated inference
    ONNX_RUNTIME,           ///< Microsoft ONNX Runtime cross-platform inference
    HYBRID_NEURAL_SYMBOLIC  ///< Combined rule-based and ML inference (future)
};

/**
 * @brief Comprehensive error types for ML inference operations
 *
 * Extends the project's error handling patterns to cover ML-specific failure modes.
 * Each error type provides semantic meaning for different failure scenarios,
 * enabling proper error recovery and user feedback.
 *
 * Error Handling Flow with Result<T,E> Integration:
 * @code
 *   ┌─────────────────┐   create_engine()   ┌─────────────────┐
 *   │ User Request    │ ─────────────────▶ │ Factory Method  │
 *   │ Backend + Config│                    │ Validation      │
 *   └─────────────────┘                    └─────────────────┘
 *            │                                      │
 *            │ Configuration Error                  │ Success
 *            ▼                                      ▼
 *   ┌─────────────────┐   INVALID_BACKEND_CONFIG   ┌─────────────────┐
 *   │ Result<T,E>     │ ◄─────────────────────────  │ Engine Creation │
 *   │ Err(error_type) │                            │ GPU/Model Init  │
 *   └─────────────────┘                            └─────────────────┘
 *            │                                              │
 *            │                          GPU/Model Error    │ Success
 *            │                                   ┌─────────┼─────────┐
 *            │                                   ▼         ▼         ▼
 *            │                          ┌─────────────┐ ┌─────────────┐
 *            │                          │GPU_MEMORY_  │ │MODEL_LOAD_  │
 *            │                          │EXHAUSTED    │ │FAILED       │
 *            │                          └─────────────┘ └─────────────┘
 *            │                                   │         │
 *            │ ◄─────────────────────────────────┼─────────┘
 *            │                                   │
 *            ▼                                   ▼
 *   ┌─────────────────┐  error.map_err()       ┌─────────────────┐
 *   │ Error           │ ──────────────────────▶│ Logging &       │
 *   │ Propagation     │                        │ User Feedback   │
 *   │ & Recovery      │                        │ with Context    │
 *   └─────────────────┘                        └─────────────────┘
 * @endcode
 */
enum class InferenceError : std::uint8_t {
    // Model Loading Errors
    MODEL_LOAD_FAILED,         ///< Failed to load model file or parse format
    UNSUPPORTED_MODEL_FORMAT,  ///< Model format not supported by backend
    MODEL_VERSION_MISMATCH,    ///< Model version incompatible with runtime

    // Runtime Errors
    BACKEND_NOT_AVAILABLE,       ///< Requested backend not available (e.g., no CUDA)
    GPU_MEMORY_EXHAUSTED,        ///< Insufficient GPU memory for model/batch
    INFERENCE_EXECUTION_FAILED,  ///< Runtime execution error during inference

    // Input/Output Errors
    INVALID_INPUT_SHAPE,       ///< Input tensor shape mismatch with model expectations
    INVALID_INPUT_TYPE,        ///< Input data type incompatible with model
    OUTPUT_PROCESSING_FAILED,  ///< Error processing inference results

    // Configuration Errors
    INVALID_BACKEND_CONFIG,  ///< Backend configuration parameters invalid
    OPTIMIZATION_FAILED,     ///< Model optimization/compilation failed

    // System Errors
    INSUFFICIENT_SYSTEM_MEMORY,  ///< Insufficient system RAM for operation
    DRIVER_COMPATIBILITY_ERROR,  ///< GPU driver incompatible with runtime
    UNKNOWN_ERROR                ///< Unexpected error condition
};

/**
 * @brief Convert InferenceError to human-readable string
 * @param error The error code to convert
 * @return String description of the error
 */
std::string to_string(InferenceError error);

/**
 * @brief Configuration parameters for inference engine creation
 *
 * Provides backend-agnostic configuration interface with extensible design.
 * Specific backends can extend this with their own configuration parameters
 * while maintaining compatibility with the unified factory interface.
 */
struct ModelConfig {
    std::string model_path;           ///< Path to model file (.onnx, .trt, etc.)
    std::uint32_t max_batch_size{1};  ///< Maximum batch size for inference
    bool enable_optimization{true};   ///< Enable backend-specific optimizations
    bool enable_profiling{false};     ///< Enable performance profiling

    // GPU-specific settings (ignored by CPU backends)
    std::uint32_t gpu_device_id{0};                ///< GPU device index for multi-GPU systems
    std::uint64_t max_workspace_size{1ULL << 30};  ///< Max GPU workspace (1GB default)
};

/**
 * @brief Input data for inference operations
 *
 * Represents input tensors/data for inference execution. Design supports both
 * traditional symbolic inputs (facts, rules) and numerical tensor data for
 * ML models. Future versions will support mixed symbolic-numeric inputs for
 * neural-symbolic reasoning.
 */
struct InferenceRequest {
    // For ML models: tensor data
    std::vector<std::vector<float>> input_tensors;  ///< Input tensor data (simplified)
    std::vector<std::string> input_names;           ///< Named inputs for model

    // For rule-based systems: symbolic data (future extension)
    // std::vector<Fact> facts;
    // std::vector<Rule> rules;

    std::uint32_t batch_size{1};  ///< Batch size for this request
};

/**
 * @brief Results from inference operations
 *
 * Contains inference outputs and metadata. Supports both numerical ML outputs
 * and symbolic reasoning results, providing foundation for hybrid systems.
 */
struct InferenceResponse {
    // ML model outputs
    std::vector<std::vector<float>> output_tensors;  ///< Output tensor data
    std::vector<std::string> output_names;           ///< Named outputs from model

    // Rule-based outputs (future extension)
    // std::vector<Fact> derived_facts;
    // std::vector<std::string> reasoning_trace;

    // Performance metadata
    double inference_time_ms{0.0};       ///< Execution time in milliseconds
    std::uint64_t memory_used_bytes{0};  ///< Peak memory usage during inference
};

/**
 * @brief Abstract base class for all inference engines
 *
 * Defines the common interface that all inference backends must implement.
 * Provides virtual destructor for proper cleanup of derived classes and
 * pure virtual methods that enforce consistent API across all backends.
 *
 * The interface is designed for performance with minimal virtual call overhead
 * in hot paths while maintaining type safety and error handling consistency.
 */
class InferenceEngine {
  public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~InferenceEngine() = default;

    /**
     * @brief Execute inference with the loaded model
     * @param request Input data for inference
     * @return Result containing inference output or error
     */
    virtual auto run_inference(const InferenceRequest& request)
        -> common::Result<InferenceResponse, InferenceError> = 0;

    /**
     * @brief Get backend-specific information and capabilities
     * @return String describing the backend and its current state
     */
    virtual auto get_backend_info() const -> std::string = 0;

    /**
     * @brief Check if the backend is ready for inference
     * @return True if ready, false if initialization or loading failed
     */
    virtual auto is_ready() const -> bool = 0;

    /**
     * @brief Get performance statistics from the engine
     * @return String containing performance metrics and statistics
     */
    virtual auto get_performance_stats() const -> std::string = 0;

  protected:
    /**
     * @brief Protected constructor - only derived classes can create instances
     */
    InferenceEngine() = default;

    /**
     * @brief Non-copyable - engines manage unique resources
     */
    InferenceEngine(const InferenceEngine&) = delete;
    auto operator=(const InferenceEngine&) -> InferenceEngine& = delete;

    /**
     * @brief Non-movable for now - can be revisited if needed
     */
    InferenceEngine(InferenceEngine&&) = delete;
    auto operator=(InferenceEngine&&) -> InferenceEngine& = delete;
};

/**
 * @brief Factory function to create inference engines
 *
 * Provides unified creation interface for all inference backends. Handles
 * backend selection, configuration validation, and resource initialization.
 * Returns Result<T,E> for consistent error handling with project patterns.
 *
 * @param backend The inference backend to create
 * @param config Configuration parameters for the engine
 * @return Result containing unique_ptr to engine or InferenceError
 *
 * @note This function performs backend availability checking and will return
 *       appropriate errors if hardware requirements are not met (e.g., no CUDA
 *       for TensorRT backend).
 */
auto create_inference_engine(InferenceBackend backend, const ModelConfig& config)
    -> common::Result<std::unique_ptr<InferenceEngine>, InferenceError>;

}  // namespace inference_lab::engines
