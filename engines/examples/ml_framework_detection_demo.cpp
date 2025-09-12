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

#include <iomanip>
#include <iostream>

#include "ml_config.hpp"

using namespace inference_lab::engines::ml;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << " " << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

int main() {
    print_separator("ML Framework Detection Demo");

    std::cout << "This demo showcases the build system's ML framework integration\n";
    std::cout << "and demonstrates graceful handling of optional dependencies.\n\n";

    // Display compile-time capabilities
    print_separator("Compile-Time Detection");

    std::cout << std::left;
    std::cout << std::setw(25) << "TensorRT available:" << (has_tensorrt ? "YES" : "NO") << "\n";
    std::cout << std::setw(25) << "ONNX Runtime available:" << (has_onnx_runtime ? "YES" : "NO")
              << "\n";
    std::cout << std::setw(25) << "ML frameworks enabled:" << (has_ml_frameworks ? "YES" : "NO")
              << "\n";
    std::cout << std::setw(25) << "Framework count:" << ml_framework_count << "\n";

    // Display runtime capabilities
    print_separator("Runtime Capabilities");

    std::cout << capabilities.to_string() << "\n\n";

    std::cout << "Framework Features:\n";
    std::cout << "  GPU Acceleration: "
              << (capabilities.gpu_acceleration ? "Available" : "Not available") << "\n";
    std::cout << "  Cross-platform Inference: "
              << (capabilities.cross_platform_inference ? "Available" : "Not available") << "\n";

    // Display available backends
    print_separator("Available Backends");

    auto backends = get_available_backends();
    std::cout << "Found " << backends.size() << " available backend(s):\n\n";

    for (size_t i = 0; i < backends.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << backend_to_string(backends[i]) << "\n";
    }

    // Show recommended backend
    print_separator("Optimal Configuration");

    auto optimal = detect_optimal_backend();
    std::cout << "Recommended backend: " << backend_to_string(optimal) << "\n\n";

    std::cout << "Backend availability check:\n";
    for (const auto& backend : backends) {
        bool available = capabilities.is_backend_available(backend);
        std::cout << "  " << std::setw(25) << backend_to_string(backend) << ": "
                  << (available ? "✓ Available" : "✗ Not available") << "\n";
    }

    // Configuration guidance
    print_separator("Configuration Guidance");

    if (ml_framework_count == 0) {
        std::cout << "⚠️  No ML frameworks detected!\n\n";
        std::cout << "To enable ML framework support:\n";
        std::cout << "  1. Install TensorRT and/or ONNX Runtime\n";
        std::cout
            << "  2. Reconfigure with: cmake -DENABLE_TENSORRT=ON -DENABLE_ONNX_RUNTIME=ON ..\n";
        std::cout << "  3. Rebuild the project\n\n";
        std::cout << "Current configuration will use CPU-only implementations.\n";
    } else {
        std::cout << "✅ ML frameworks successfully integrated!\n\n";
        std::cout << "Available acceleration:\n";

        if (has_tensorrt) {
            std::cout << "  • TensorRT: NVIDIA GPU acceleration for production inference\n";
        }
        if (has_onnx_runtime) {
            std::cout << "  • ONNX Runtime: Cross-platform inference with multiple backends\n";
        }
        if (has_tensorrt && has_onnx_runtime) {
            std::cout << "  • Hybrid: Best of both frameworks for optimal performance\n";
        }
    }

    print_separator("Build System Integration Test");

    std::cout << "Build system integration: ";

    // Test compile-time constants
    static_assert(ml_framework_count >= 0, "Framework count must be non-negative");

    // Test runtime functionality
    bool integration_ok = true;

    // Verify compile-time and runtime consistency
    if (capabilities.framework_count != ml_framework_count) {
        std::cout << "❌ FAILED - Framework count mismatch\n";
        integration_ok = false;
    }

    if (capabilities.tensorrt_available != has_tensorrt) {
        std::cout << "❌ FAILED - TensorRT availability mismatch\n";
        integration_ok = false;
    }

    if (capabilities.onnx_runtime_available != has_onnx_runtime) {
        std::cout << "❌ FAILED - ONNX Runtime availability mismatch\n";
        integration_ok = false;
    }

    if (integration_ok) {
        std::cout << "✅ PASSED - All integration checks successful\n";
        std::cout << "\nThe build system correctly detected and configured ML frameworks.\n";
        std::cout << "Applications can reliably use the ML configuration API.\n";
    }

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Demo completed successfully!\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
