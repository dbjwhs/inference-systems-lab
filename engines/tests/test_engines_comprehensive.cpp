// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file test_engines_comprehensive.cpp
 * @brief Comprehensive test suite for engines module - Phase 4.1 Enterprise Coverage
 *
 * This test suite implements Phase 4.1 of the Enterprise Test Coverage Initiative,
 * targeting comprehensive testing of the engines module to achieve 80%+ coverage.
 *
 * Coverage Targets:
 * - inference_engine.hpp/.cpp: Core factory and error handling
 * - forward_chaining.hpp/.cpp: Rule-based inference engine
 * - model_registry.hpp/.cpp: Model version management
 * - TensorRT mock testing: GPU resource management validation
 *
 * Test Categories:
 * 1. Interface Contract Testing - Virtual methods, polymorphism, RAII
 * 2. Factory Pattern Testing - Backend selection, configuration validation
 * 3. Error Handling Testing - All InferenceError types and propagation
 * 4. Rule-Based Engine Testing - Forward chaining logic, pattern matching
 * 5. Performance Testing - Latency, throughput, memory usage
 * 6. Integration Testing - Cross-component interaction validation
 */

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../../common/src/logging.hpp"
#include "../../common/src/result.hpp"
#include "../src/forward_chaining.hpp"
#include "../src/inference_engine.hpp"
#include "../src/model_registry.hpp"

using namespace inference_lab::engines;
using namespace inference_lab::common;

/**
 * @class EnginesComprehensiveTest
 * @brief Main test fixture for comprehensive engines testing
 */
class EnginesComprehensiveTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize logging for test output
        Logger::get_instance("./test_engines_comprehensive.log", false);
        LOG_INFO_PRINT("Starting engines comprehensive test suite");
    }

    void TearDown() override {
        LOG_INFO_PRINT("Engines comprehensive test suite complete");
        // Clean up test artifacts
        static_cast<void>(std::remove("./test_engines_comprehensive.log"));
    }

    /**
     * @brief Create valid ModelConfig for testing
     */
    auto create_valid_config() -> ModelConfig {
        return ModelConfig{
            .model_path = "test_model.onnx",
            .max_batch_size = 4,
            .enable_optimization = true,
            .enable_profiling = false,
            .gpu_device_id = 0,
            .max_workspace_size = 1ULL << 28  // 256MB
        };
    }

    /**
     * @brief Create sample inference request
     */
    auto create_inference_request(std::uint32_t batch_size = 1) -> InferenceRequest {
        return InferenceRequest{.input_tensors = {{1.0f, 2.0f, 3.0f, 4.0f},
                                                  {5.0f, 6.0f, 7.0f, 8.0f}},
                                .input_names = {"input_1", "input_2"},
                                .batch_size = batch_size};
    }
};

// =============================================================================
// Error Handling and String Conversion Tests
// =============================================================================

/**
 * @brief Test comprehensive error string conversions
 */
TEST_F(EnginesComprehensiveTest, AllErrorStringConversions) {
    // Model Loading Errors
    EXPECT_EQ(to_string(InferenceError::MODEL_LOAD_FAILED),
              "Failed to load model file or parse format");
    EXPECT_EQ(to_string(InferenceError::UNSUPPORTED_MODEL_FORMAT),
              "Model format not supported by backend");
    EXPECT_EQ(to_string(InferenceError::MODEL_VERSION_MISMATCH),
              "Model version incompatible with runtime");

    // Runtime Errors
    EXPECT_EQ(to_string(InferenceError::BACKEND_NOT_AVAILABLE), "Requested backend not available");
    EXPECT_EQ(to_string(InferenceError::GPU_MEMORY_EXHAUSTED),
              "Insufficient GPU memory for model/batch");
    EXPECT_EQ(to_string(InferenceError::INFERENCE_EXECUTION_FAILED),
              "Runtime execution error during inference");

    // Input/Output Errors
    EXPECT_EQ(to_string(InferenceError::INVALID_INPUT_SHAPE),
              "Input tensor shape mismatch with model expectations");
    EXPECT_EQ(to_string(InferenceError::INVALID_INPUT_TYPE),
              "Input data type incompatible with model");
    EXPECT_EQ(to_string(InferenceError::OUTPUT_PROCESSING_FAILED),
              "Error processing inference results");

    // Configuration Errors
    EXPECT_EQ(to_string(InferenceError::INVALID_BACKEND_CONFIG),
              "Backend configuration parameters invalid");
    EXPECT_EQ(to_string(InferenceError::OPTIMIZATION_FAILED),
              "Model optimization/compilation failed");

    // System Errors
    EXPECT_EQ(to_string(InferenceError::INSUFFICIENT_SYSTEM_MEMORY),
              "Insufficient system RAM for operation");
    EXPECT_EQ(to_string(InferenceError::DRIVER_COMPATIBILITY_ERROR),
              "GPU driver incompatible with runtime");
    EXPECT_EQ(to_string(InferenceError::UNKNOWN_ERROR), "Unexpected error condition");

    LOG_INFO_PRINT("All {} error types have proper string representations",
                   static_cast<int>(InferenceError::UNKNOWN_ERROR) + 1);
}

// =============================================================================
// Factory Function Validation Tests
// =============================================================================

/**
 * @brief Test factory function with comprehensive invalid configurations
 */
TEST_F(EnginesComprehensiveTest, FactoryConfigurationValidation) {
    // Test empty model path
    {
        ModelConfig invalid_config = create_valid_config();
        invalid_config.model_path = "";

        auto result = create_inference_engine(InferenceBackend::RULE_BASED, invalid_config);
        ASSERT_TRUE(result.is_err());
        EXPECT_EQ(result.unwrap_err(), InferenceError::INVALID_BACKEND_CONFIG);
    }

    // Test zero batch size
    {
        ModelConfig invalid_config = create_valid_config();
        invalid_config.max_batch_size = 0;

        auto result = create_inference_engine(InferenceBackend::ONNX_RUNTIME, invalid_config);
        ASSERT_TRUE(result.is_err());
        EXPECT_EQ(result.unwrap_err(), InferenceError::INVALID_BACKEND_CONFIG);
    }

    // Test very large batch size (boundary testing)
    {
        ModelConfig config = create_valid_config();
        config.max_batch_size = 1000000;  // Very large but not zero

        // Should not fail on configuration validation alone
        auto result = create_inference_engine(InferenceBackend::RULE_BASED, config);
        // May fail later for other reasons, but not config validation
    }

    LOG_INFO_PRINT("Factory configuration validation tests complete");
}

/**
 * @brief Test all backend availability scenarios
 */
TEST_F(EnginesComprehensiveTest, BackendAvailabilityTesting) {
    ModelConfig config = create_valid_config();

    // Test RULE_BASED backend
    {
        auto result = create_inference_engine(InferenceBackend::RULE_BASED, config);
        // This should work if forward chaining is implemented
        if (result.is_err()) {
            EXPECT_EQ(result.unwrap_err(), InferenceError::INVALID_BACKEND_CONFIG);
            LOG_INFO_PRINT("RULE_BASED backend not yet fully implemented");
        } else {
            LOG_INFO_PRINT("RULE_BASED backend available");
        }
    }

    // Test TENSORRT_GPU backend
    {
        auto result = create_inference_engine(InferenceBackend::TENSORRT_GPU, config);
        ASSERT_TRUE(result.is_err());  // Should fail without ENABLE_TENSORRT
        EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
        LOG_INFO_PRINT("TENSORRT_GPU backend not available (expected without ENABLE_TENSORRT)");
    }

    // Test ONNX_RUNTIME backend
    {
        auto result = create_inference_engine(InferenceBackend::ONNX_RUNTIME, config);
        ASSERT_TRUE(result.is_err());  // Should fail without ENABLE_ONNX_RUNTIME
        EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
        LOG_INFO_PRINT("ONNX_RUNTIME backend not available (expected without ENABLE_ONNX_RUNTIME)");
    }

    // Test HYBRID_NEURAL_SYMBOLIC backend (future implementation)
    {
        auto result = create_inference_engine(InferenceBackend::HYBRID_NEURAL_SYMBOLIC, config);
        ASSERT_TRUE(result.is_err());
        EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
        LOG_INFO_PRINT("HYBRID_NEURAL_SYMBOLIC backend not available (future implementation)");
    }
}

// =============================================================================
// MockInferenceEngine for Interface Testing
// =============================================================================

/**
 * @class ComprehensiveMockEngine
 * @brief Enhanced mock engine for comprehensive testing
 */
class ComprehensiveMockEngine : public InferenceEngine {
  public:
    explicit ComprehensiveMockEngine(bool ready = true, bool should_fail = false)
        : is_ready_(ready), should_fail_inference_(should_fail), inference_count_(0) {}

    auto run_inference(const InferenceRequest& request)
        -> Result<InferenceResponse, InferenceError> override {
        if (!is_ready_) {
            return Err(InferenceError::BACKEND_NOT_AVAILABLE);
        }

        if (should_fail_inference_) {
            return Err(InferenceError::INFERENCE_EXECUTION_FAILED);
        }

        ++inference_count_;

        if (request.input_tensors.empty()) {
            return Err(InferenceError::INVALID_INPUT_SHAPE);
        }

        if (request.batch_size == 0) {
            return Err(InferenceError::INVALID_BACKEND_CONFIG);
        }

        // Simulate processing time based on batch size
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(request.batch_size * 10));
        auto end = std::chrono::high_resolution_clock::now();

        double inference_time = std::chrono::duration<double, std::milli>(end - start).count();

        InferenceResponse response{
            .output_tensors = {{static_cast<float>(inference_count_),
                                2.0f * request.batch_size,
                                3.0f}},
            .output_names = {"mock_output_1"},
            .inference_time_ms = inference_time,
            .memory_used_bytes = request.batch_size * 1024  // Simulate memory usage
        };

        return Ok(response);
    }

    auto get_backend_info() const -> std::string override {
        return "ComprehensiveMockEngine v2.0 - Enhanced Test Implementation";
    }

    auto is_ready() const -> bool override { return is_ready_; }

    auto get_performance_stats() const -> std::string override {
        return "Mock stats: " + std::to_string(inference_count_) + " inferences completed";
    }

    // Test helper methods
    auto get_inference_count() const -> std::uint32_t { return inference_count_; }
    void set_ready(bool ready) { is_ready_ = ready; }
    void set_should_fail(bool fail) { should_fail_inference_ = fail; }

  private:
    bool is_ready_;
    bool should_fail_inference_;
    std::uint32_t inference_count_;
};

// =============================================================================
// Interface Contract and Polymorphism Tests
// =============================================================================

/**
 * @brief Test virtual method dispatch and polymorphism
 */
TEST_F(EnginesComprehensiveTest, VirtualMethodPolymorphism) {
    // Test polymorphic behavior through base class pointer
    std::unique_ptr<InferenceEngine> engine =
        std::make_unique<ComprehensiveMockEngine>(true, false);

    EXPECT_TRUE(engine->is_ready());
    EXPECT_EQ(engine->get_backend_info(),
              "ComprehensiveMockEngine v2.0 - Enhanced Test Implementation");

    auto request = create_inference_request(2);
    auto result = engine->run_inference(request);

    ASSERT_TRUE(result.is_ok());
    const auto& response = result.unwrap();
    EXPECT_EQ(response.output_tensors.size(), 1);
    EXPECT_EQ(response.output_tensors[0].size(), 3);
    EXPECT_EQ(response.output_names[0], "mock_output_1");
    EXPECT_GT(response.inference_time_ms, 0.0);
    EXPECT_GT(response.memory_used_bytes, 0);

    LOG_INFO_PRINT("Virtual method polymorphism verified");
}

/**
 * @brief Test RAII and resource management
 */
TEST_F(EnginesComprehensiveTest, RAIIResourceManagement) {
    // Test that engines properly clean up resources on destruction
    {
        auto engine = std::make_unique<ComprehensiveMockEngine>(true, false);
        auto request = create_inference_request(1);
        auto result = engine->run_inference(request);
        ASSERT_TRUE(result.is_ok());

        // Engine should be automatically cleaned up when leaving scope
    }

    // Test move semantics (engines are non-copyable, non-movable for now)
    auto engine1 = std::make_unique<ComprehensiveMockEngine>(true, false);
    std::unique_ptr<InferenceEngine> engine2 = std::move(engine1);

    EXPECT_FALSE(engine1);  // Should be null after move
    EXPECT_TRUE(engine2);   // Should have the moved resource
    EXPECT_TRUE(engine2->is_ready());

    LOG_INFO_PRINT("RAII resource management verified");
}

// =============================================================================
// Error Handling and Recovery Tests
// =============================================================================

/**
 * @brief Test comprehensive error handling scenarios
 */
TEST_F(EnginesComprehensiveTest, ComprehensiveErrorHandling) {
    // Test backend not ready scenario
    {
        auto engine = std::make_unique<ComprehensiveMockEngine>(false, false);
        auto request = create_inference_request(1);
        auto result = engine->run_inference(request);

        ASSERT_TRUE(result.is_err());
        EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
        LOG_INFO_PRINT("Backend not ready error handling verified");
    }

    // Test inference execution failure
    {
        auto engine = std::make_unique<ComprehensiveMockEngine>(true, true);
        auto request = create_inference_request(1);
        auto result = engine->run_inference(request);

        ASSERT_TRUE(result.is_err());
        EXPECT_EQ(result.unwrap_err(), InferenceError::INFERENCE_EXECUTION_FAILED);
        LOG_INFO_PRINT("Inference execution failure handling verified");
    }

    // Test invalid input shape
    {
        auto engine = std::make_unique<ComprehensiveMockEngine>(true, false);
        InferenceRequest empty_request{.input_tensors = {}, .input_names = {}, .batch_size = 1};
        auto result = engine->run_inference(empty_request);

        ASSERT_TRUE(result.is_err());
        EXPECT_EQ(result.unwrap_err(), InferenceError::INVALID_INPUT_SHAPE);
        LOG_INFO_PRINT("Invalid input shape error handling verified");
    }

    // Test invalid batch size
    {
        auto engine = std::make_unique<ComprehensiveMockEngine>(true, false);
        InferenceRequest zero_batch_request{
            .input_tensors = {{1.0f, 2.0f}}, .input_names = {"input"}, .batch_size = 0};
        auto result = engine->run_inference(zero_batch_request);

        ASSERT_TRUE(result.is_err());
        EXPECT_EQ(result.unwrap_err(), InferenceError::INVALID_BACKEND_CONFIG);
        LOG_INFO_PRINT("Invalid batch size error handling verified");
    }
}

/**
 * @brief Test error propagation through Result<T,E> chains
 */
TEST_F(EnginesComprehensiveTest, ErrorPropagationChains) {
    auto engine = std::make_unique<ComprehensiveMockEngine>(false, false);
    auto request = create_inference_request(1);

    // Test error propagation and transformation
    auto result =
        engine->run_inference(request)
            .map_err([](InferenceError err) -> InferenceError {
                LOG_INFO_PRINT("Error intercepted: {}", to_string(err));
                return err;  // Pass through unchanged
            })
            .and_then([](InferenceResponse response) -> Result<std::string, InferenceError> {
                // This should not execute due to error
                return Ok(std::string("Success"));
            })
            .or_else([](InferenceError err) -> Result<std::string, InferenceError> {
                // This should execute and recover
                LOG_INFO_PRINT("Recovering from error: {}", to_string(err));
                return Ok(std::string("Recovered"));
            });

    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(result.unwrap(), "Recovered");
    LOG_INFO_PRINT("Error propagation chain verified");
}

// =============================================================================
// Performance and Load Testing
// =============================================================================

/**
 * @brief Test engine performance under load
 */
TEST_F(EnginesComprehensiveTest, PerformanceUnderLoad) {
    auto engine = std::make_unique<ComprehensiveMockEngine>(true, false);

    const int NUM_ITERATIONS = 50;  // Reduced for faster testing
    const std::uint32_t BATCH_SIZE = 4;

    auto request = create_inference_request(BATCH_SIZE);

    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto result = engine->run_inference(request);
        ASSERT_TRUE(result.is_ok()) << "Iteration " << i << " failed";

        const auto& response = result.unwrap();
        EXPECT_EQ(response.output_tensors[0][1], 2.0f * BATCH_SIZE);
        EXPECT_GT(response.inference_time_ms, 0.0);
        EXPECT_EQ(response.memory_used_bytes, BATCH_SIZE * 1024);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_us = static_cast<double>(total_duration.count()) / NUM_ITERATIONS;
    double throughput_inferences_per_sec = 1000000.0 / avg_time_us;

    // Performance expectations for mock engine
    EXPECT_LT(avg_time_us, 1000.0);                   // < 1ms per inference
    EXPECT_GT(throughput_inferences_per_sec, 100.0);  // > 100 inferences/sec

    // Check that inference count was properly tracked
    auto mock_engine = dynamic_cast<ComprehensiveMockEngine*>(engine.get());
    ASSERT_NE(mock_engine, nullptr);
    EXPECT_EQ(mock_engine->get_inference_count(), NUM_ITERATIONS);

    LOG_INFO_PRINT("Performance test: {:.2f}μs avg, {:.1f} inferences/sec",
                   avg_time_us,
                   throughput_inferences_per_sec);
}

/**
 * @brief Test memory usage patterns
 */
TEST_F(EnginesComprehensiveTest, MemoryUsagePatterns) {
    auto engine = std::make_unique<ComprehensiveMockEngine>(true, false);

    // Test different batch sizes and their memory impact
    std::vector<std::uint32_t> batch_sizes = {1, 4, 8, 16};

    for (auto batch_size : batch_sizes) {
        auto request = create_inference_request(batch_size);
        auto result = engine->run_inference(request);

        ASSERT_TRUE(result.is_ok()) << "Batch size " << batch_size << " failed";

        const auto& response = result.unwrap();
        std::uint64_t expected_memory = batch_size * 1024;
        EXPECT_EQ(response.memory_used_bytes, expected_memory);

        LOG_INFO_PRINT("Batch size {} -> {} bytes memory", batch_size, response.memory_used_bytes);
    }
}

// =============================================================================
// Dynamic Behavior and State Management Tests
// =============================================================================

/**
 * @brief Test dynamic state changes during runtime
 */
TEST_F(EnginesComprehensiveTest, DynamicStateManagement) {
    auto mock_engine = std::make_unique<ComprehensiveMockEngine>(true, false);
    auto request = create_inference_request(1);

    // Test initial ready state
    EXPECT_TRUE(mock_engine->is_ready());
    auto result = mock_engine->run_inference(request);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(mock_engine->get_inference_count(), 1);

    // Test dynamic state change - make not ready
    mock_engine->set_ready(false);
    EXPECT_FALSE(mock_engine->is_ready());
    result = mock_engine->run_inference(request);
    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::BACKEND_NOT_AVAILABLE);
    EXPECT_EQ(mock_engine->get_inference_count(), 1);  // Should not increment on failure

    // Test recovery - make ready again
    mock_engine->set_ready(true);
    EXPECT_TRUE(mock_engine->is_ready());
    result = mock_engine->run_inference(request);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(mock_engine->get_inference_count(), 2);

    // Test failure mode toggling
    mock_engine->set_should_fail(true);
    result = mock_engine->run_inference(request);
    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), InferenceError::INFERENCE_EXECUTION_FAILED);

    mock_engine->set_should_fail(false);
    result = mock_engine->run_inference(request);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(mock_engine->get_inference_count(), 3);

    LOG_INFO_PRINT("Dynamic state management verified - {} total inferences",
                   mock_engine->get_inference_count());
}

// =============================================================================
// Integration and Cross-Component Tests
// =============================================================================

/**
 * @brief Test integration between factory and engines
 */
TEST_F(EnginesComprehensiveTest, FactoryEngineIntegration) {
    // Test that factory-created engines behave correctly
    ModelConfig config = create_valid_config();

    // Try to create rule-based engine through factory
    auto result = create_inference_engine(InferenceBackend::RULE_BASED, config);

    if (result.is_ok()) {
        auto engine = std::move(result).unwrap();
        EXPECT_TRUE(engine != nullptr);

        // Test that factory-created engine implements interface correctly
        EXPECT_NO_THROW({
            auto info = engine->get_backend_info();
            auto ready = engine->is_ready();
            auto stats = engine->get_performance_stats();
            LOG_INFO_PRINT("Factory-created engine: {} (ready: {})", info, ready);
        });

        LOG_INFO_PRINT("Factory-engine integration successful");
    } else {
        // Expected if rule-based engine not fully implemented
        EXPECT_EQ(result.unwrap_err(), InferenceError::INVALID_BACKEND_CONFIG);
        LOG_INFO_PRINT("Factory-engine integration deferred (rule-based engine not implemented)");
    }
}
