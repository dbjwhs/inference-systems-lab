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

/**
 * @file inference_engine.cpp
 * @brief Implementation of unified inference interface
 *
 * Provides implementations for common inference functionality including
 * error handling, factory methods, and utility functions that are shared
 * across all inference backends.
 */

#include "inference_engine.hpp"

#include "forward_chaining.hpp"

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
        case InferenceBackend::RULE_BASED: {
            // Create forward chaining engine for rule-based inference
            auto result =
                create_forward_chaining_engine(ConflictResolutionStrategy::PRIORITY_ORDER,
                                               1000,  // max iterations
                                               false  // tracing disabled by default for production
                );

            if (result.is_err()) {
                return Err(InferenceError::INVALID_BACKEND_CONFIG);
            }

            // Move the result and cast to base class pointer
            auto engine_ptr = std::move(result).unwrap();
            std::unique_ptr<InferenceEngine> engine = std::move(engine_ptr);
            return Ok(std::move(engine));
        }

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
