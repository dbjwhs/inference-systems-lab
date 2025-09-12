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

#include <set>

#include <gtest/gtest.h>

#include "../src/ml_config.hpp"

namespace inference_lab {
namespace engines {
namespace ml {
namespace test {

class MLConfigTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Reset any global state if needed
    }

    void TearDown() override {
        // Clean up after each test
    }
};

// Test compile-time detection constants
TEST_F(MLConfigTest, CompileTimeDetection) {
    // These should be constexpr and deterministic at compile time
    static_assert(has_tensorrt == true || has_tensorrt == false,
                  "has_tensorrt must be boolean constexpr");
    static_assert(has_onnx_runtime == true || has_onnx_runtime == false,
                  "has_onnx_runtime must be boolean constexpr");
    static_assert(has_ml_frameworks == true || has_ml_frameworks == false,
                  "has_ml_frameworks must be boolean constexpr");

    // Verify consistency
    bool expected_has_ml = has_tensorrt || has_onnx_runtime;
    EXPECT_EQ(has_ml_frameworks, expected_has_ml);

    // Log current configuration for debugging
    EXPECT_TRUE(true) << "TensorRT available: " << has_tensorrt;
    EXPECT_TRUE(true) << "ONNX Runtime available: " << has_onnx_runtime;
    EXPECT_TRUE(true) << "ML Frameworks available: " << has_ml_frameworks;
}

// Test MLBackend enum values
TEST_F(MLConfigTest, MLBackendEnumValues) {
    // Verify enum values are distinct
    EXPECT_NE(static_cast<uint8_t>(MLBackend::CPU_ONLY),
              static_cast<uint8_t>(MLBackend::TENSORRT_GPU));
    EXPECT_NE(static_cast<uint8_t>(MLBackend::CPU_ONLY),
              static_cast<uint8_t>(MLBackend::ONNX_RUNTIME));
    EXPECT_NE(static_cast<uint8_t>(MLBackend::TENSORRT_GPU),
              static_cast<uint8_t>(MLBackend::ONNX_RUNTIME));

    // Verify all enum values are within expected range (0-3)
    EXPECT_LE(static_cast<uint8_t>(MLBackend::CPU_ONLY), 3);
    EXPECT_LE(static_cast<uint8_t>(MLBackend::TENSORRT_GPU), 3);
    EXPECT_LE(static_cast<uint8_t>(MLBackend::ONNX_RUNTIME), 3);
    EXPECT_LE(static_cast<uint8_t>(MLBackend::HYBRID_NEURAL_SYMBOLIC), 3);
}

// Test MLCapabilities structure and methods
TEST_F(MLConfigTest, MLCapabilitiesStructure) {
    const auto& caps = capabilities;

    // Verify basic structure
    EXPECT_EQ(caps.tensorrt_available, has_tensorrt);
    EXPECT_EQ(caps.onnx_runtime_available, has_onnx_runtime);
    EXPECT_EQ(caps.gpu_acceleration, has_tensorrt);
    EXPECT_EQ(caps.cross_platform_inference, has_onnx_runtime);

    // Test string conversion
    std::string caps_str = caps.to_string();
    EXPECT_FALSE(caps_str.empty());
    EXPECT_NE(caps_str.find("ML Capabilities:"), std::string::npos);

    // Verify string contains expected values
    if (has_ml_frameworks) {
        if (has_tensorrt) {
            EXPECT_NE(caps_str.find("TensorRT"), std::string::npos);
        }
        if (has_onnx_runtime) {
            EXPECT_NE(caps_str.find("ONNX Runtime"), std::string::npos);
        }
    } else {
        EXPECT_NE(caps_str.find("CPU-only mode"), std::string::npos);
    }
}

// Test backend availability checking
TEST_F(MLConfigTest, BackendAvailability) {
    const auto& caps = capabilities;

    // CPU_ONLY should always be available
    EXPECT_TRUE(caps.is_backend_available(MLBackend::CPU_ONLY));

    // TensorRT availability should match compile-time detection
    EXPECT_EQ(caps.is_backend_available(MLBackend::TENSORRT_GPU), has_tensorrt);

    // ONNX Runtime availability should match compile-time detection
    EXPECT_EQ(caps.is_backend_available(MLBackend::ONNX_RUNTIME), has_onnx_runtime);

    // Hybrid should be available if both TensorRT and ONNX Runtime are available
    EXPECT_EQ(caps.is_backend_available(MLBackend::HYBRID_NEURAL_SYMBOLIC),
              has_tensorrt && has_onnx_runtime);
}

// Test get_available_backends function
TEST_F(MLConfigTest, GetAvailableBackends) {
    auto backends = get_available_backends();

    // Should always include CPU_ONLY
    EXPECT_NE(std::find(backends.begin(), backends.end(), MLBackend::CPU_ONLY), backends.end());

    // Should include TensorRT if available
    if (has_tensorrt) {
        EXPECT_NE(std::find(backends.begin(), backends.end(), MLBackend::TENSORRT_GPU),
                  backends.end());
    } else {
        EXPECT_EQ(std::find(backends.begin(), backends.end(), MLBackend::TENSORRT_GPU),
                  backends.end());
    }

    // Should include ONNX Runtime if available
    if (has_onnx_runtime) {
        EXPECT_NE(std::find(backends.begin(), backends.end(), MLBackend::ONNX_RUNTIME),
                  backends.end());
    } else {
        EXPECT_EQ(std::find(backends.begin(), backends.end(), MLBackend::ONNX_RUNTIME),
                  backends.end());
    }

    // Should include hybrid if both frameworks are available
    if (has_tensorrt && has_onnx_runtime) {
        EXPECT_NE(std::find(backends.begin(), backends.end(), MLBackend::HYBRID_NEURAL_SYMBOLIC),
                  backends.end());
    } else {
        EXPECT_EQ(std::find(backends.begin(), backends.end(), MLBackend::HYBRID_NEURAL_SYMBOLIC),
                  backends.end());
    }

    // Verify no duplicates
    std::set<MLBackend> unique_backends(backends.begin(), backends.end());
    EXPECT_EQ(unique_backends.size(), backends.size());

    // Verify reasonable size (1-4 backends)
    EXPECT_GE(backends.size(), 1);  // At least CPU_ONLY
    EXPECT_LE(backends.size(), 4);  // At most all 4 backends
}

// Test detect_optimal_backend function
TEST_F(MLConfigTest, DetectOptimalBackend) {
    auto optimal = detect_optimal_backend();

    // Should return a valid backend
    auto available = get_available_backends();
    EXPECT_NE(std::find(available.begin(), available.end(), optimal), available.end());

    // Test priority order logic
    if (has_tensorrt) {
        // TensorRT should be preferred for GPU acceleration
        EXPECT_EQ(optimal, MLBackend::TENSORRT_GPU);
    } else if (has_onnx_runtime) {
        // ONNX Runtime should be preferred for cross-platform
        EXPECT_EQ(optimal, MLBackend::ONNX_RUNTIME);
    } else {
        // CPU_ONLY should be fallback
        EXPECT_EQ(optimal, MLBackend::CPU_ONLY);
    }
}

// Test backend string conversions
TEST_F(MLConfigTest, BackendStringConversion) {
    // Test all enum values have string representations
    EXPECT_FALSE(backend_to_string(MLBackend::CPU_ONLY).empty());
    EXPECT_FALSE(backend_to_string(MLBackend::TENSORRT_GPU).empty());
    EXPECT_FALSE(backend_to_string(MLBackend::ONNX_RUNTIME).empty());
    EXPECT_FALSE(backend_to_string(MLBackend::HYBRID_NEURAL_SYMBOLIC).empty());

    // Test expected string values
    EXPECT_EQ(backend_to_string(MLBackend::CPU_ONLY), "CPU-only");
    EXPECT_EQ(backend_to_string(MLBackend::TENSORRT_GPU), "TensorRT GPU");
    EXPECT_EQ(backend_to_string(MLBackend::ONNX_RUNTIME), "ONNX Runtime");
    EXPECT_EQ(backend_to_string(MLBackend::HYBRID_NEURAL_SYMBOLIC), "Hybrid Neural-Symbolic");
}

// Test framework configuration consistency
TEST_F(MLConfigTest, ConfigurationConsistency) {
    const auto& caps = capabilities;

    // If no ML frameworks available, only CPU backend should be available
    if (!has_ml_frameworks) {
        auto backends = get_available_backends();
        EXPECT_EQ(backends.size(), 1);
        EXPECT_EQ(backends[0], MLBackend::CPU_ONLY);
        EXPECT_EQ(detect_optimal_backend(), MLBackend::CPU_ONLY);
    }

    // If ML frameworks available, should have more than just CPU
    if (has_ml_frameworks) {
        auto backends = get_available_backends();
        EXPECT_GT(backends.size(), 1);
        EXPECT_NE(detect_optimal_backend(), MLBackend::CPU_ONLY);
    }

    // Capabilities should be consistent with compile-time detection
    EXPECT_EQ(caps.tensorrt_available, has_tensorrt);
    EXPECT_EQ(caps.onnx_runtime_available, has_onnx_runtime);
    EXPECT_EQ(caps.gpu_acceleration, has_tensorrt);
    EXPECT_EQ(caps.cross_platform_inference, has_onnx_runtime);

    // Framework count should match enabled frameworks
    size_t expected_count = 0;
    if (has_tensorrt)
        expected_count++;
    if (has_onnx_runtime)
        expected_count++;
    EXPECT_EQ(caps.framework_count, expected_count);
}

// Test edge cases and error conditions
TEST_F(MLConfigTest, EdgeCases) {
    // Test that functions are robust and don't crash
    EXPECT_NO_THROW(get_available_backends());
    EXPECT_NO_THROW(detect_optimal_backend());
    EXPECT_NO_THROW(capabilities.to_string());

    // Test all backend availability checks
    for (int i = 0; i <= 3; ++i) {
        auto backend = static_cast<MLBackend>(i);
        EXPECT_NO_THROW(capabilities.is_backend_available(backend));
        EXPECT_NO_THROW(backend_to_string(backend));
    }
}

// Integration test: Verify the API works as expected in realistic scenarios
TEST_F(MLConfigTest, IntegrationScenario) {
    // Simulate a typical usage pattern
    const auto& caps = capabilities;

    // 1. Check what's available
    auto available_backends = get_available_backends();
    EXPECT_FALSE(available_backends.empty());

    // 2. Get optimal backend
    auto optimal = detect_optimal_backend();
    EXPECT_TRUE(caps.is_backend_available(optimal));

    // 3. Verify optimal is in available list
    EXPECT_NE(std::find(available_backends.begin(), available_backends.end(), optimal),
              available_backends.end());

    // 4. Log configuration for debugging
    std::string config_summary = caps.to_string();
    EXPECT_FALSE(config_summary.empty());

    // 5. Verify consistency
    if (has_ml_frameworks) {
        EXPECT_GT(available_backends.size(), 1);
        EXPECT_NE(optimal, MLBackend::CPU_ONLY);
    } else {
        EXPECT_EQ(available_backends.size(), 1);
        EXPECT_EQ(optimal, MLBackend::CPU_ONLY);
    }
}

}  // namespace test
}  // namespace ml
}  // namespace engines
}  // namespace inference_lab
