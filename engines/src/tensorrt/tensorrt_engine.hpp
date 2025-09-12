// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file tensorrt_engine.hpp
 * @brief TensorRT GPU-accelerated inference engine implementation
 *
 * This header provides a RAII-compliant wrapper around NVIDIA TensorRT for
 * GPU-accelerated inference. The implementation follows the project's patterns
 * for error handling, logging, and resource management while providing optimal
 * performance for GPU inference workloads.
 *
 * Key Features:
 * - RAII resource management for GPU memory, CUDA contexts, and TensorRT engines
 * - Integration with existing Result<T,E> error handling patterns
 * - Support for both .onnx model loading and pre-optimized .trt engines
 * - Automatic batch processing and GPU memory optimization
 * - Thread-safe inference execution with proper CUDA context management
 * - Comprehensive error handling for GPU-specific failure modes
 *
 * Hardware Requirements:
 * - NVIDIA GPU with Compute Capability 7.0+ (RTX 20-series or newer)
 * - CUDA Toolkit 11.8+ with cuDNN support
 * - TensorRT 8.5+ runtime libraries
 * - Sufficient GPU memory for model + batch processing
 *
 * Usage Example:
 * @code
 * ModelConfig config{
 *     .model_path = "model.onnx",
 *     .max_batch_size = 8,
 *     .enable_optimization = true,
 *     .gpu_device_id = 0
 * };
 *
 * auto engine = TensorRTEngine::create(config);
 * if (engine.is_ok()) {
 *     auto result = engine.unwrap()->run_inference(request);
 *     // Handle inference result...
 * }
 * @endcode
 *
 * TensorRT Workflow:
 * @code
 *   ┌─────────────┐  load_model()   ┌─────────────────┐     ONNX Parser    ┌─────────────────┐
 *   │ ONNX Model  │ ──────────────▶ │ TensorRT Engine │ ─────────────────▶ │ Optimized       │
 *   │ (.onnx)     │                 │ Builder         │                    │ GPU Engine      │
 *   └─────────────┘                 └─────────────────┘                    │ (.trt)          │
 *        │                                     │                           └─────────────────┘
 *        │                                     │                                     │
 *        └──────── Model Loading ──────────────┴────── Engine Building ──────────────┘
 *                                                                                    |
 *                                                                                    ▼
 *                                                                           ┌─────────────────┐
 *                                                                           |                 ▼
 *                                                                           ▼       ┌─────────────────┐
 *   allocate_buffers()  ┌─────────────────┐      enqueue()      ┌─────────────────┐ │ GPU Memory      │
 *   ◄────────────────── │ CUDA Context    │ ──────────────────▶ │ Inference       │ │ Input/Output    │
 *          ▲            │ & Execution     │                     │ Execution       │ │ Buffers         │
 *          │            │ Context         │                     │ (GPU Kernels)   │ └─────────────────┘
 *          │            └─────────────────┘                     └─────────────────┘
 *          │                                                                             │
 *          │ copy_from_host()                                             copy_to_host() │
 *          │                                                                             ▼
 *   ┌─────────────┐  InferenceRequest   ┌─────────────────┐  InferenceResponse  ┌─────────────────┐
 *   │ Host Input  │ ─────────────────▶  │ TensorRT Engine │ ─────────────────▶  │ Host Output     │
 *   │ Tensors     │                     │ run_inference() │                     │ Tensors         │
 *   └─────────────┘                     └─────────────────┘                     └─────────────────┘
 * @endcode
 */

#pragma once

#ifdef ENABLE_TENSORRT  // Only compile when TensorRT is available

    #include <memory>
    #include <mutex>
    #include <string>
    #include <vector>

// TensorRT headers (forward declarations to minimize header dependencies)
namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
class IRuntime;
class ILogger;
}  // namespace nvinfer1

    #include "../../common/src/logging.hpp"
    #include "../../common/src/result.hpp"
    #include "../inference_engine.hpp"

namespace inference_lab::engines::tensorrt {

/**
 * @brief RAII wrapper for CUDA device memory
 *
 * Manages GPU memory allocation and deallocation using RAII principles.
 * Ensures proper cleanup even in exception scenarios and provides
 * type-safe access to GPU memory buffers.
 */
class CudaBuffer {
  public:
    /**
     * @brief Allocate GPU memory buffer
     * @param size_bytes Size of buffer to allocate in bytes
     * @return Result containing CudaBuffer or InferenceError
     */
    static auto allocate(std::size_t size_bytes) -> common::Result<CudaBuffer, InferenceError>;

    /**
     * @brief Destructor - automatically frees GPU memory
     */
    ~CudaBuffer();

    /**
     * @brief Copy data from host to device
     * @param host_data Pointer to host memory
     * @param size_bytes Number of bytes to copy
     * @return Result indicating success or failure
     */
    auto copy_from_host(const void* host_data, std::size_t size_bytes)
        -> common::Result<void, InferenceError>;

    /**
     * @brief Copy data from device to host
     * @param host_data Pointer to host memory destination
     * @param size_bytes Number of bytes to copy
     * @return Result indicating success or failure
     */
    auto copy_to_host(void* host_data, std::size_t size_bytes) const
        -> common::Result<void, InferenceError>;

    /**
     * @brief Get raw device pointer (for TensorRT binding)
     * @return Raw CUDA device pointer
     */
    auto device_ptr() const -> void* { return device_ptr_; }

    /**
     * @brief Get buffer size in bytes
     * @return Size of allocated buffer
     */
    auto size_bytes() const -> std::size_t { return size_bytes_; }

    // Non-copyable but movable
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

  private:
    /**
     * @brief Private constructor - use allocate() factory method
     */
    explicit CudaBuffer(void* device_ptr, std::size_t size_bytes);

    void* device_ptr_{nullptr};  ///< CUDA device memory pointer
    std::size_t size_bytes_{0};  ///< Size of allocated buffer
};

/**
 * @brief TensorRT inference engine implementation
 *
 * Provides GPU-accelerated inference using NVIDIA TensorRT runtime.
 * Implements the unified InferenceEngine interface while providing
 * TensorRT-specific optimizations and error handling.
 *
 * The implementation is thread-safe for inference operations but
 * model loading should be done from a single thread.
 */
class TensorRTEngine final : public InferenceEngine {
  public:
    /**
     * @brief Factory method to create TensorRT engine
     * @param config Model configuration including paths and GPU settings
     * @return Result containing engine instance or InferenceError
     *
     * @note This method handles all initialization including:
     *       - CUDA device selection and context creation
     *       - Model loading and optimization (if needed)
     *       - GPU memory allocation for inference buffers
     *       - TensorRT engine compilation (for .onnx models)
     */
    static auto create(const ModelConfig& config)
        -> common::Result<std::unique_ptr<TensorRTEngine>, InferenceError>;

    /**
     * @brief Destructor - ensures proper cleanup of GPU resources
     */
    ~TensorRTEngine() override;

    /**
     * @brief Execute inference with GPU acceleration
     * @param request Input data for inference
     * @return Result containing inference output or error
     *
     * @note This method is thread-safe and handles:
     *       - Input data validation and preprocessing
     *       - Host-to-device memory transfers
     *       - GPU inference execution
     *       - Device-to-host result transfers
     *       - Performance timing and logging
     */
    auto run_inference(const InferenceRequest& request)
        -> common::Result<InferenceResponse, InferenceError> override;

    /**
     * @brief Get TensorRT engine information
     * @return String containing engine details and GPU info
     */
    auto get_backend_info() const -> std::string override;

    /**
     * @brief Check if TensorRT engine is ready for inference
     * @return True if engine is loaded and GPU resources allocated
     */
    auto is_ready() const -> bool override;

    /**
     * @brief Get performance statistics from TensorRT profiling
     * @return String containing timing and memory usage statistics
     */
    auto get_performance_stats() const -> std::string override;

    // Non-copyable and non-movable (manages unique GPU resources)
    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;
    TensorRTEngine(TensorRTEngine&&) = delete;
    TensorRTEngine& operator=(TensorRTEngine&&) = delete;

  private:
    /**
     * @brief Private constructor - use create() factory method
     */
    explicit TensorRTEngine(const ModelConfig& config);

    /**
     * @brief Initialize CUDA device and context
     * @return Result indicating success or failure
     */
    auto initialize_cuda_device() -> common::Result<void, InferenceError>;

    /**
     * @brief Load model from file (.onnx or .trt)
     * @return Result indicating success or failure
     */
    auto load_model() -> common::Result<void, InferenceError>;

    /**
     * @brief Allocate GPU buffers for inference
     * @return Result indicating success or failure
     */
    auto allocate_gpu_buffers() -> common::Result<void, InferenceError>;

    /**
     * @brief Build TensorRT engine from ONNX model
     * @param onnx_file_path Path to ONNX model file
     * @return Result indicating success or failure
     */
    auto build_engine_from_onnx(const std::string& onnx_file_path)
        -> common::Result<void, InferenceError>;

    /**
     * @brief Load pre-built TensorRT engine
     * @param trt_file_path Path to serialized TensorRT engine
     * @return Result indicating success or failure
     */
    auto load_serialized_engine(const std::string& trt_file_path)
        -> common::Result<void, InferenceError>;

    /**
     * @brief Validate input shapes and types
     * @param request Input request to validate
     * @return Result indicating success or failure
     */
    auto validate_input(const InferenceRequest& request) const
        -> common::Result<void, InferenceError>;

    // Configuration
    ModelConfig config_;  ///< Engine configuration parameters

    // TensorRT objects (using smart pointers for RAII)
    std::unique_ptr<nvinfer1::IRuntime> runtime_;           ///< TensorRT runtime
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;         ///< TensorRT engine
    std::unique_ptr<nvinfer1::IExecutionContext> context_;  ///< Execution context
    std::unique_ptr<nvinfer1::ILogger> logger_;             ///< TensorRT logger

    // GPU memory management
    std::vector<CudaBuffer> input_buffers_;   ///< GPU input buffers
    std::vector<CudaBuffer> output_buffers_;  ///< GPU output buffers
    std::vector<void*> bindings_;             ///< TensorRT binding pointers

    // Model metadata
    std::vector<std::string> input_names_;         ///< Input tensor names
    std::vector<std::string> output_names_;        ///< Output tensor names
    std::vector<std::vector<int>> input_shapes_;   ///< Input tensor shapes
    std::vector<std::vector<int>> output_shapes_;  ///< Output tensor shapes

    // Thread safety and performance
    mutable std::mutex inference_mutex_;  ///< Protects inference operations
    bool is_ready_{false};                ///< Indicates if engine is ready for inference

    // Performance tracking
    mutable std::uint64_t total_inference_count_{0};    ///< Total inferences executed
    mutable double total_inference_time_ms_{0.0};       ///< Cumulative inference time
    mutable std::uint64_t peak_memory_usage_bytes_{0};  ///< Peak GPU memory usage
};

/**
 * @brief TensorRT-specific utility functions
 */
namespace utils {

/**
 * @brief Check if CUDA is available and compatible
 * @return Result indicating CUDA availability or specific error
 */
auto check_cuda_availability() -> common::Result<void, InferenceError>;

/**
 * @brief Get CUDA device information
 * @param device_id GPU device index to query
 * @return Result containing device info string or error
 */
auto get_cuda_device_info(std::uint32_t device_id) -> common::Result<std::string, InferenceError>;

/**
 * @brief Estimate GPU memory requirements for model
 * @param model_path Path to model file
 * @param batch_size Maximum batch size
 * @return Result containing estimated memory in bytes or error
 */
auto estimate_gpu_memory_requirements(const std::string& model_path, std::uint32_t batch_size)
    -> common::Result<std::uint64_t, InferenceError>;

}  // namespace utils

}  // namespace inference_lab::engines::tensorrt

#endif  // ENABLE_TENSORRT
