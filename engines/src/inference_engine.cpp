// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file inference_engine.cpp
 * @brief Implementation of unified inference interface
 *
 * Provides implementations for common inference functionality including
 * error handling, factory methods, and utility functions that are shared
 * across all inference backends.
 */

#include "inference_engine.hpp"

#ifdef ENABLE_TENSORRT
    #include "tensorrt/tensorrt_engine.hpp"
#endif

#ifdef ENABLE_ONNX_RUNTIME
    #include "onnx/onnx_engine.hpp"
#endif

namespace inference_lab::engines {

std::string to_string(InferenceError error) {
    switch (error) {
        // Model Loading Errors
        case InferenceError::MODEL_LOAD_FAILED:
            return "Failed to load model file or parse format";
        case InferenceError::UNSUPPORTED_MODEL_FORMAT:
            return "Model format not supported by backend";
        case InferenceError::MODEL_VERSION_MISMATCH:
            return "Model version incompatible with runtime";

        // Runtime Errors
        case InferenceError::BACKEND_NOT_AVAILABLE:
            return "Requested backend not available";
        case InferenceError::GPU_MEMORY_EXHAUSTED:
            return "Insufficient GPU memory for model/batch";
        case InferenceError::INFERENCE_EXECUTION_FAILED:
            return "Runtime execution error during inference";

        // Input/Output Errors
        case InferenceError::INVALID_INPUT_SHAPE:
            return "Input tensor shape mismatch with model expectations";
        case InferenceError::INVALID_INPUT_TYPE:
            return "Input data type incompatible with model";
        case InferenceError::OUTPUT_PROCESSING_FAILED:
            return "Error processing inference results";

        // Configuration Errors
        case InferenceError::INVALID_BACKEND_CONFIG:
            return "Backend configuration parameters invalid";
        case InferenceError::OPTIMIZATION_FAILED:
            return "Model optimization/compilation failed";

        // System Errors
        case InferenceError::INSUFFICIENT_SYSTEM_MEMORY:
            return "Insufficient system RAM for operation";
        case InferenceError::DRIVER_COMPATIBILITY_ERROR:
            return "GPU driver incompatible with runtime";
        case InferenceError::UNKNOWN_ERROR:
            return "Unexpected error condition";
    }
    return "Unknown inference error";
}

auto create_inference_engine(InferenceBackend backend, const ModelConfig& config)
    -> common::Result<std::unique_ptr<InferenceEngine>, InferenceError> {
    using common::Err;
    using common::Ok;
    using common::Result;

    // Validate common configuration parameters
    if (config.model_path.empty()) {
        return Err(InferenceError::INVALID_BACKEND_CONFIG);
    }

    if (config.max_batch_size == 0) {
        return Err(InferenceError::INVALID_BACKEND_CONFIG);
    }

    // Create backend-specific engine
    switch (backend) {
        case InferenceBackend::RULE_BASED:
            // TODO: Implement rule-based engine factory
            return Err(InferenceError::BACKEND_NOT_AVAILABLE);

        case InferenceBackend::TENSORRT_GPU:
#ifdef ENABLE_TENSORRT
            return tensorrt::TensorRTEngine::create(config);
#else
            return Err(InferenceError::BACKEND_NOT_AVAILABLE);
#endif

        case InferenceBackend::ONNX_RUNTIME:
#ifdef ENABLE_ONNX_RUNTIME
            return onnx::ONNXEngine::create(config);
#else
            return Err(InferenceError::BACKEND_NOT_AVAILABLE);
#endif

        case InferenceBackend::HYBRID_NEURAL_SYMBOLIC:
            // TODO: Implement hybrid neural-symbolic engine
            return Err(InferenceError::BACKEND_NOT_AVAILABLE);

        default:
            return Err(InferenceError::BACKEND_NOT_AVAILABLE);
    }
}

}  // namespace inference_lab::engines
