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

#include "ml_config.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace inference_lab {
namespace engines {
namespace ml {

// Global capabilities instance
const MLCapabilities capabilities{};

auto MLCapabilities::to_string() const -> std::string {
    std::stringstream ss;
    ss << "ML Capabilities: ";

    if (framework_count == 0) {
        ss << "CPU-only mode (no ML frameworks)";
        return ss.str();
    }

    ss << framework_count << " framework(s) - ";

    std::vector<std::string> frameworks;
    if (tensorrt_available)
        frameworks.push_back("TensorRT");
    if (onnx_runtime_available)
        frameworks.push_back("ONNX Runtime");

    for (size_t i = 0; i < frameworks.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << frameworks[i];
    }

    return ss.str();
}

auto MLCapabilities::is_backend_available(MLBackend backend) const -> bool {
    switch (backend) {
        case MLBackend::CPU_ONLY:
            return true;  // Always available
        case MLBackend::TENSORRT_GPU:
            return tensorrt_available;
        case MLBackend::ONNX_RUNTIME:
            return onnx_runtime_available;
        case MLBackend::HYBRID_NEURAL_SYMBOLIC:
            return tensorrt_available && onnx_runtime_available;
        default:
            return false;
    }
}

auto MLCapabilities::get_recommended_backend() const -> MLBackend {
    // Preference order: Hybrid > TensorRT > ONNX Runtime > CPU-only
    if (tensorrt_available && onnx_runtime_available) {
        return MLBackend::HYBRID_NEURAL_SYMBOLIC;
    }
    if (tensorrt_available) {
        return MLBackend::TENSORRT_GPU;
    }
    if (onnx_runtime_available) {
        return MLBackend::ONNX_RUNTIME;
    }
    return MLBackend::CPU_ONLY;
}

auto get_available_backends() -> std::vector<MLBackend> {
    std::vector<MLBackend> backends;

    // CPU-only is always available
    backends.push_back(MLBackend::CPU_ONLY);

    if (capabilities.tensorrt_available) {
        backends.push_back(MLBackend::TENSORRT_GPU);
    }

    if (capabilities.onnx_runtime_available) {
        backends.push_back(MLBackend::ONNX_RUNTIME);
    }

    if (capabilities.tensorrt_available && capabilities.onnx_runtime_available) {
        backends.push_back(MLBackend::HYBRID_NEURAL_SYMBOLIC);
    }

    return backends;
}

auto backend_to_string(MLBackend backend) -> std::string {
    switch (backend) {
        case MLBackend::CPU_ONLY:
            return "CPU-only";
        case MLBackend::TENSORRT_GPU:
            return "TensorRT GPU";
        case MLBackend::ONNX_RUNTIME:
            return "ONNX Runtime";
        case MLBackend::HYBRID_NEURAL_SYMBOLIC:
            return "Hybrid Neural-Symbolic";
        default:
            return "Unknown";
    }
}

auto detect_optimal_backend() -> MLBackend {
    return capabilities.get_recommended_backend();
}

}  // namespace ml
}  // namespace engines
}  // namespace inference_lab
