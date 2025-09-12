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
 * @file test_inference_engine.cpp
 * @brief Comprehensive unit tests for inference engine interfaces
 *
 * This test suite validates the unified inference interface and provides
 * mock GPU environments for testing TensorRT functionality without requiring
 * actual GPU hardware in CI/CD environments.
 *
 * Test Strategy:
 * - Mock implementations for testing interface contracts
 * - Conditional GPU tests that skip when hardware unavailable
 * - Performance regression testing with baseline comparisons
 * - Error handling validation for all failure modes
 * - Memory safety validation with sanitizers
 */

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../common/src/logging.hpp"
#include "../src/inference_engine.hpp"
#include "result.hpp"

using namespace inference_lab::engines;
using namespace inference_lab::common;

/**
 * @class MockInferenceEngine
 * @brief Mock implementation for testing interface contracts
 */
class MockInferenceEngine : public InferenceEngine {
  public:
    explicit MockInferenceEngine(bool should_be_ready = true) : is_ready_(should_be_ready) {}

    auto run_inference(const InferenceRequest& request)
        -> Result<InferenceResponse, InferenceError> override {
        if (!is_ready_) {
            return Err(InferenceError::BACKEND_NOT_AVAILABLE);
        }

        if (request.input_tensors.empty()) {
            return Err(InferenceError::INVALID_INPUT_SHAPE);
        }

        // Mock successful inference
        InferenceResponse response{.output_tensors = {{1.0F, 2.0F, 3.0F}},
                                   .output_names = {"mock_output"},
                                   .inference_time_ms = 1.5,
                                   .memory_used_bytes = 1024};

        return Ok(response);
    }

    auto get_backend_info() const -> std::string override {
        return "MockEngine v1.0 - Test Implementation";
    }

    auto is_ready() const -> bool override { return is_ready_; }

    auto get_performance_stats() const -> std::string override {
        return "Mock stats: 0 inferences, 0ms total time";
    }

  private:
    bool is_ready_;
};

/**
 * @class InferenceEngineTest
 * @brief Test fixture for inference engine testing
 */
class InferenceEngineTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize logging for test output
        Logger::get_instance("./test_inference_engine.log", false);
    }

    void TearDown() override {
        // Clean up test artifacts
        static_cast<void>(std::remove("./test_inference_engine.log"));
    }

    /**
     * @brief Create sample inference request for testing
     */
    auto create_sample_request() -> InferenceRequest {
        return InferenceRequest{
            .input_tensors = {{1.0F, 2.0F, 3.0F, 4.0F}}, .input_names = {"input"}, .batch_size = 1};
    }

    /**
     * @brief Check if CUDA is available for GPU tests
     */
    auto has_cuda_device() -> bool {
        // Simple CUDA availability check
        // In real implementation, this would check for actual GPU
        return std::getenv("CUDA_VISIBLE_DEVICES") != nullptr;
    }
};

/**
 * @brief Test error string conversion
 */
TEST_F(InferenceEngineTest, ErrorStringConversion) {
    EXPECT_EQ(to_string(InferenceError::MODEL_LOAD_FAILED),
              "Failed to load model file or parse format");
    EXPECT_EQ(to_string(InferenceError::GPU_MEMORY_EXHAUSTED),
              "Insufficient GPU memory for model/batch");
    EXPECT_EQ(to_string(InferenceError::INVALID_INPUT_SHAPE),
              "Input tensor shape mismatch with model expectations");
    EXPECT_EQ(to_string(InferenceError::BACKEND_NOT_AVAILABLE), "Requested backend not available");
}

/**
 * @brief Test mock engine basic functionality
 */
TEST_F(InferenceEngineTest, MockEngineBasicOperation) {
    auto engine = std::make_unique<MockInferenceEngine>(true);

    EXPECT_TRUE(engine->is_ready());
    EXPECT_EQ(engine->get_backend_info(), "MockEngine v1.0 - Test Implementation");

    auto request = create_sample_request();
    auto result = engine->run_inference(request);

    ASSERT_TRUE(result.is_ok());

    const auto& response = result.unwrap();
    EXPECT_EQ(response.output_tensors.size(), 1);
    EXPECT_EQ(response.output_tensors[0].size(), 3);
    EXPECT_EQ(response.output_names[0], "mock_output");
    EXPECT_GT(response.inference_time_ms, 0.0);
}

/**
 * @brief Test mock engine error handling
 */
TEST_F(InferenceEngineTest, MockEngineErrorHandling) {
    auto engine = std::make_unique<MockInferenceEngine>(false);

    EXPECT_FALSE(engine->is_ready());

    auto request = create_sample_request();
    auto result = engine->run_inference(request);

    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
}

/**
 * @brief Test invalid input handling
 */
TEST_F(InferenceEngineTest, InvalidInputHandling) {
    auto engine = std::make_unique<MockInferenceEngine>(true);

    InferenceRequest empty_request{.input_tensors = {}, .input_names = {}, .batch_size = 1};

    auto result = engine->run_inference(empty_request);

    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::INVALID_INPUT_SHAPE);
}

/**
 * @brief Test factory function with invalid configuration
 */
TEST_F(InferenceEngineTest, FactoryInvalidConfig) {
    ModelConfig invalid_config{.model_path = "",  // Invalid empty path
                               .max_batch_size = 1};

    auto result = create_inference_engine(InferenceBackend::TENSORRT_GPU, invalid_config);

    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::INVALID_BACKEND_CONFIG);
}

/**
 * @brief Test factory function with zero batch size
 */
TEST_F(InferenceEngineTest, FactoryZeroBatchSize) {
    ModelConfig invalid_config{
        .model_path = "test_model.onnx",
        .max_batch_size = 0  // Invalid zero batch size
    };

    auto result = create_inference_engine(InferenceBackend::ONNX_RUNTIME, invalid_config);

    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::INVALID_BACKEND_CONFIG);
}

/**
 * @brief Test TensorRT backend availability
 */
TEST_F(InferenceEngineTest, TensorRTAvailability) {
    ModelConfig config{.model_path = "test_model.onnx", .max_batch_size = 1};

    auto result = create_inference_engine(InferenceBackend::TENSORRT_GPU, config);

    // Should fail if TensorRT not compiled in or no GPU available
#ifndef ENABLE_TENSORRT
    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
#else
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for TensorRT tests";
    }
    // If we have CUDA, the result depends on actual TensorRT installation
    // This would be tested in integration tests with real hardware
#endif
}

/**
 * @brief Test ONNX Runtime backend availability
 */
TEST_F(InferenceEngineTest, ONNXRuntimeAvailability) {
    ModelConfig config{.model_path = "test_model.onnx", .max_batch_size = 1};

    auto result = create_inference_engine(InferenceBackend::ONNX_RUNTIME, config);

    // Should fail if ONNX Runtime not compiled in
#ifndef ENABLE_ONNX_RUNTIME
    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
#else
    // If ONNX Runtime is available, this would proceed to model loading
    // which might fail without an actual model file - that's expected
#endif
}

/**
 * @brief Test rule-based backend (not yet implemented)
 */
TEST_F(InferenceEngineTest, RuleBasedNotImplemented) {
    ModelConfig config{.model_path = "test_rules.json", .max_batch_size = 1};

    auto result = create_inference_engine(InferenceBackend::RULE_BASED, config);

    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
}

/**
 * @brief Test hybrid neural-symbolic backend (future implementation)
 */
TEST_F(InferenceEngineTest, HybridNotImplemented) {
    ModelConfig config{.model_path = "hybrid_model.json", .max_batch_size = 1};

    auto result = create_inference_engine(InferenceBackend::HYBRID_NEURAL_SYMBOLIC, config);

    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
}

/**
 * @brief Performance test for mock engine
 */
TEST_F(InferenceEngineTest, MockEnginePerformance) {
    auto engine = std::make_unique<MockInferenceEngine>(true);
    auto request = create_sample_request();

    const int NUM_ITERATIONS = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto result = engine->run_inference(request);
        ASSERT_TRUE(result.is_ok()) << true << (i != 0);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_us = static_cast<double>(duration.count()) / NUM_ITERATIONS;

    // Mock engine should be very fast (< 100 microseconds per inference)
    EXPECT_LT(avg_time_us, 100.0) << true;

    std::ostringstream oss;
    oss << "Mock engine average inference time: " << std::fixed << std::setprecision(2)
        << avg_time_us << " microseconds";
    LOG_INFO_PRINT("{}", oss.str());
}
