// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file test_ml_integration.cpp
 * @brief Comprehensive integration tests for ML inference systems
 *
 * This file contains a complete suite of integration tests that validate the
 * ML inference framework across different backends, testing scenarios, and
 * error conditions. The tests are designed to run in CI/CD environments with
 * mock backends when hardware is not available, while also supporting full
 * hardware testing when GPUs and specialized inference hardware are present.
 *
 * Test Categories:
 * - Single backend validation tests
 * - Multi-backend comparison tests
 * - Performance and latency tests
 * - Memory management and leak detection tests
 * - Error handling and recovery tests
 * - Concurrent execution tests
 * - Resource exhaustion tests
 * - End-to-end pipeline tests
 *
 * The tests demonstrate proper usage of the integration framework and serve
 * as both validation and documentation for the testing patterns.
 */

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../../common/src/logging.hpp"
#include "../../common/src/ml_types.hpp"
#include "../src/integration_test_utils.hpp"
#include "../src/ml_integration_framework.hpp"
#include "../src/mock_engines.hpp"

using namespace inference_lab::integration;
using namespace inference_lab::integration::utils;
using namespace inference_lab::integration::mocks;
using namespace inference_lab::common;
using namespace inference_lab::common::ml;
using namespace inference_lab::engines;

//=============================================================================
// Test Fixtures and Setup
//=============================================================================

/**
 * @brief Base test fixture for ML integration tests
 */
class MLIntegrationTestBase : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize logging for tests
        setup_test_environment("DEBUG");

        // Create integration framework with mock factory
        auto mock_factory = std::make_unique<TestBackendFactory>();
        test_factory_ = mock_factory.get();
        framework_ = std::make_unique<MLIntegrationFramework>(std::move(mock_factory));

        // Set up common mock engines
        setup_mock_engines();

        // Create common test fixtures
        classification_fixture_ =
            std::make_shared<ClassificationTestFixture>(ClassificationTestFixture::create()
                                                            .with_model("resnet50")
                                                            .with_input_shape({1, 3, 224, 224})
                                                            .with_num_classes(1000)
                                                            .with_precision(Precision::FP32));

        object_detection_fixture_ =
            std::make_shared<ObjectDetectionTestFixture>(ObjectDetectionTestFixture::create());

        nlp_fixture_ = std::make_shared<NLPTestFixture>(NLPTestFixture::create());

        // Initialize performance analyzer
        performance_analyzer_ = std::make_unique<PerformanceAnalyzer>(
            PerformanceAnalyzer::Config{.confidence_level = 0.95f,
                                        .min_iterations = 5,
                                        .max_iterations = 50,
                                        .enable_statistical_tests = true});

        // Initialize memory tracker
        memory_tracker_ = std::make_unique<MemoryTracker>();

        LOG_INFO_PRINT("ML Integration test setup completed");
    }

    void TearDown() override {
        // Clean up test environment
        cleanup_test_environment();
        LOG_INFO_PRINT("ML Integration test cleanup completed");
    }

    /**
     * @brief Set up mock engines for testing
     */
    void setup_mock_engines() {
        // TensorRT mock with realistic GPU behavior
        auto tensorrt_config = MockTensorRTEngine::create_tensorrt_config();
        tensorrt_config.performance.base_latency_ms = 5.0f;
        tensorrt_config.simulate_gpu_memory = true;
        tensorrt_config.max_gpu_memory_mb = 8192;

        test_factory_->inject_mock_engine(InferenceBackend::TENSORRT_GPU, [tensorrt_config]() {
            return std::make_unique<MockTensorRTEngine>(tensorrt_config);
        });

        // ONNX Runtime mock with cross-platform behavior
        auto onnx_config = MockONNXRuntimeEngine::create_onnx_config();
        onnx_config.performance.base_latency_ms = 15.0f;
        onnx_config.simulate_hardware = false;  // CPU-based

        test_factory_->inject_mock_engine(InferenceBackend::ONNX_RUNTIME, [onnx_config]() {
            return std::make_unique<MockONNXRuntimeEngine>(onnx_config);
        });

        // Rule-based mock for hybrid testing
        auto rule_config = MockRuleBasedEngine::create_rule_based_config();
        rule_config.performance.base_latency_ms = 2.0f;

        test_factory_->inject_mock_engine(InferenceBackend::RULE_BASED, [rule_config]() {
            return std::make_unique<MockRuleBasedEngine>(rule_config);
        });
    }

    // Test infrastructure
    std::unique_ptr<MLIntegrationFramework> framework_;
    TestBackendFactory* test_factory_;  // Non-owning pointer
    std::unique_ptr<PerformanceAnalyzer> performance_analyzer_;
    std::unique_ptr<MemoryTracker> memory_tracker_;

    // Test fixtures
    std::shared_ptr<ClassificationTestFixture> classification_fixture_;
    std::shared_ptr<ObjectDetectionTestFixture> object_detection_fixture_;
    std::shared_ptr<NLPTestFixture> nlp_fixture_;
};

//=============================================================================
// Single Backend Tests
//=============================================================================

/**
 * @brief Test single backend functionality in isolation
 */
class SingleBackendTest : public MLIntegrationTestBase {};

TEST_F(SingleBackendTest, TensorRTBasicInference) {
    LOG_INFO_PRINT("Testing TensorRT basic inference functionality");

    auto model_config = classification_fixture_->get_model_config();
    auto test_inputs = classification_fixture_->generate_test_data(5);

    auto result =
        framework_->test_single_backend(InferenceBackend::TENSORRT_GPU, model_config, test_inputs);

    ASSERT_TRUE(result.is_ok()) << "TensorRT test failed: " << result.unwrap_err();

    auto backend_result = result.unwrap();
    EXPECT_TRUE(backend_result.success) << "TensorRT inference should succeed";
    EXPECT_EQ(backend_result.backend, InferenceBackend::TENSORRT_GPU);
    EXPECT_EQ(backend_result.outputs.size(), test_inputs.size());
    EXPECT_GT(backend_result.performance.throughput_fps, 0.0f);

    // Validate each output
    for (const auto& output : backend_result.outputs) {
        auto validation_result = classification_fixture_->validate_output(output);
        EXPECT_TRUE(validation_result.is_ok()) << validation_result.unwrap_err();
    }

    LOG_INFO_PRINT("TensorRT basic inference test completed successfully");
}

TEST_F(SingleBackendTest, ONNXRuntimeBasicInference) {
    LOG_INFO_PRINT("Testing ONNX Runtime basic inference functionality");

    auto model_config = classification_fixture_->get_model_config();
    model_config.backend = InferenceBackend::ONNX_RUNTIME;
    auto test_inputs = classification_fixture_->generate_test_data(5);

    auto result =
        framework_->test_single_backend(InferenceBackend::ONNX_RUNTIME, model_config, test_inputs);

    ASSERT_TRUE(result.is_ok()) << "ONNX Runtime test failed: " << result.unwrap_err();

    auto backend_result = result.unwrap();
    EXPECT_TRUE(backend_result.success);
    EXPECT_EQ(backend_result.backend, InferenceBackend::ONNX_RUNTIME);
    EXPECT_EQ(backend_result.outputs.size(), test_inputs.size());

    LOG_INFO_PRINT("ONNX Runtime basic inference test completed successfully");
}

TEST_F(SingleBackendTest, RuleBasedInference) {
    LOG_INFO_PRINT("Testing Rule-Based inference functionality");

    auto model_config = classification_fixture_->get_model_config();
    model_config.backend = InferenceBackend::RULE_BASED;
    auto test_inputs = classification_fixture_->generate_test_data(3);

    auto result =
        framework_->test_single_backend(InferenceBackend::RULE_BASED, model_config, test_inputs);

    ASSERT_TRUE(result.is_ok()) << "Rule-based test failed: " << result.unwrap_err();

    auto backend_result = result.unwrap();
    EXPECT_TRUE(backend_result.success);
    EXPECT_EQ(backend_result.backend, InferenceBackend::RULE_BASED);

    LOG_INFO_PRINT("Rule-Based inference test completed successfully");
}

//=============================================================================
// Multi-Backend Comparison Tests
//=============================================================================

/**
 * @brief Test multiple backends with the same inputs for consistency
 */
class MultiBackendTest : public MLIntegrationTestBase {};

TEST_F(MultiBackendTest, CrossBackendConsistency) {
    LOG_INFO_PRINT("Testing cross-backend consistency");

    auto model_config = classification_fixture_->get_model_config();
    auto test_inputs = classification_fixture_->generate_test_data(3);

    std::vector<InferenceBackend> backends = {InferenceBackend::TENSORRT_GPU,
                                              InferenceBackend::ONNX_RUNTIME};

    auto result = framework_->compare_backends(
        backends, model_config, test_inputs, ValidationStrategy::STATISTICAL_COMPARISON);

    ASSERT_TRUE(result.is_ok()) << "Multi-backend comparison failed: " << result.unwrap_err();

    auto integration_result = result.unwrap();
    EXPECT_TRUE(integration_result.overall_success);
    EXPECT_EQ(integration_result.backend_results.size(), backends.size());

    // Check that both backends produced results
    for (const auto& backend_result : integration_result.backend_results) {
        EXPECT_TRUE(backend_result.success)
            << "Backend " << to_string(backend_result.backend) << " should succeed";
        EXPECT_EQ(backend_result.outputs.size(), test_inputs.size());
    }

    // Verify similarity between backends
    if (integration_result.cross_backend_similarity.has_value()) {
        EXPECT_GT(*integration_result.cross_backend_similarity, 0.8f)
            << "Backends should produce similar results";
    }

    LOG_INFO_PRINT("Cross-backend consistency test completed successfully");
}

TEST_F(MultiBackendTest, PerformanceComparison) {
    LOG_INFO_PRINT("Testing performance comparison between backends");

    auto model_config = classification_fixture_->get_model_config();
    auto test_inputs = classification_fixture_->generate_test_data(10);

    std::vector<InferenceBackend> backends = {InferenceBackend::TENSORRT_GPU,
                                              InferenceBackend::ONNX_RUNTIME,
                                              InferenceBackend::RULE_BASED};

    // Use performance analyzer to get detailed metrics
    auto perf_result = performance_analyzer_->compare_backend_performance(
        backends, model_config, test_inputs, test_factory_);

    ASSERT_TRUE(perf_result.is_ok())
        << "Performance comparison failed: " << perf_result.unwrap_err();

    auto performance_metrics = perf_result.unwrap();
    EXPECT_EQ(performance_metrics.size(), backends.size());

    // Verify expected performance characteristics
    auto tensorrt_metrics = performance_metrics.find(InferenceBackend::TENSORRT_GPU);
    auto onnx_metrics = performance_metrics.find(InferenceBackend::ONNX_RUNTIME);
    auto rule_metrics = performance_metrics.find(InferenceBackend::RULE_BASED);

    ASSERT_NE(tensorrt_metrics, performance_metrics.end());
    ASSERT_NE(onnx_metrics, performance_metrics.end());
    ASSERT_NE(rule_metrics, performance_metrics.end());

    // TensorRT should be faster than ONNX for GPU-optimized models
    EXPECT_LT(tensorrt_metrics->second.avg_latency, onnx_metrics->second.avg_latency)
        << "TensorRT should have lower latency than ONNX Runtime";

    // Rule-based should have very low latency for simple operations
    EXPECT_LT(rule_metrics->second.avg_latency, std::chrono::milliseconds(10))
        << "Rule-based inference should be very fast";

    LOG_INFO_PRINT("Performance comparison test completed successfully");
}

//=============================================================================
// Error Handling and Recovery Tests
//=============================================================================

/**
 * @brief Test error handling and recovery mechanisms
 */
class ErrorHandlingTest : public MLIntegrationTestBase {};

TEST_F(ErrorHandlingTest, GPUMemoryExhaustion) {
    LOG_INFO_PRINT("Testing GPU memory exhaustion handling");

    // Configure TensorRT mock to simulate memory exhaustion
    auto tensorrt_config = MockTensorRTEngine::create_tensorrt_config();
    tensorrt_config.error_injection.error_rates["GPU_MEMORY_EXHAUSTED"] = 0.5f;  // 50% error rate

    test_factory_->inject_mock_engine(InferenceBackend::TENSORRT_GPU, [tensorrt_config]() {
        return std::make_unique<MockTensorRTEngine>(tensorrt_config);
    });

    auto model_config = classification_fixture_->get_model_config();
    auto test_inputs = classification_fixture_->generate_test_data(10);

    auto result =
        framework_->test_single_backend(InferenceBackend::TENSORRT_GPU, model_config, test_inputs);

    ASSERT_TRUE(result.is_ok()) << "Error handling test setup failed: " << result.unwrap_err();

    auto backend_result = result.unwrap();

    // Should have some failures due to memory exhaustion
    EXPECT_GT(backend_result.failed_iterations, 0)
        << "Should have some failures due to error injection";
    EXPECT_GT(backend_result.errors.size(), 0) << "Should have recorded error information";

    // Check that errors were properly categorized
    bool found_memory_error = false;
    for (const auto& error : backend_result.errors) {
        if (error.error_code.find("MEMORY") != std::string::npos) {
            found_memory_error = true;
            break;
        }
    }
    EXPECT_TRUE(found_memory_error) << "Should have detected memory-related errors";

    LOG_INFO_PRINT("GPU memory exhaustion test completed successfully");
}

TEST_F(ErrorHandlingTest, ModelLoadingFailure) {
    LOG_INFO_PRINT("Testing model loading failure handling");

    // Configure ONNX mock to simulate model loading failure
    auto onnx_config = MockONNXRuntimeEngine::create_onnx_config();
    onnx_config.error_injection.error_rates["MODEL_LOAD_FAILED"] = 1.0f;  // Always fail

    test_factory_->inject_mock_engine(InferenceBackend::ONNX_RUNTIME, [onnx_config]() {
        return std::make_unique<MockONNXRuntimeEngine>(onnx_config);
    });

    auto model_config = classification_fixture_->get_model_config();
    model_config.backend = InferenceBackend::ONNX_RUNTIME;
    auto test_inputs = classification_fixture_->generate_test_data(1);

    auto result =
        framework_->test_single_backend(InferenceBackend::ONNX_RUNTIME, model_config, test_inputs);

    // Test should complete but with failures
    ASSERT_TRUE(result.is_ok()) << "Error handling test should complete: " << result.unwrap_err();

    auto backend_result = result.unwrap();
    EXPECT_FALSE(backend_result.success) << "Should fail due to model loading error";
    EXPECT_GT(backend_result.errors.size(), 0) << "Should have error information";

    LOG_INFO_PRINT("Model loading failure test completed successfully");
}

//=============================================================================
// Memory Management Tests
//=============================================================================

/**
 * @brief Test memory management and leak detection
 */
class MemoryManagementTest : public MLIntegrationTestBase {};

TEST_F(MemoryManagementTest, MemoryLeakDetection) {
    LOG_INFO_PRINT("Testing memory leak detection");

    auto model_config = classification_fixture_->get_model_config();

    auto result = framework_->test_memory_management(InferenceBackend::TENSORRT_GPU,
                                                     model_config,
                                                     100  // stress iterations
    );

    ASSERT_TRUE(result.is_ok()) << "Memory management test failed: " << result.unwrap_err();

    auto backend_result = result.unwrap();
    EXPECT_TRUE(backend_result.success) << "Memory management test should succeed";
    EXPECT_FALSE(backend_result.performance.memory_leaks_detected)
        << "No memory leaks should be detected";

    // Check memory allocation/deallocation balance
    EXPECT_EQ(backend_result.performance.memory_allocations,
              backend_result.performance.memory_deallocations)
        << "Allocations and deallocations should be balanced";

    LOG_INFO_PRINT("Memory leak detection test completed successfully");
}

TEST_F(MemoryManagementTest, ResourceExhaustionHandling) {
    LOG_INFO_PRINT("Testing resource exhaustion handling");

    auto model_config = classification_fixture_->get_model_config();

    auto result =
        framework_->test_resource_exhaustion(InferenceBackend::TENSORRT_GPU, model_config);

    ASSERT_TRUE(result.is_ok()) << "Resource exhaustion test failed: " << result.unwrap_err();

    auto backend_result = result.unwrap();

    // Test should handle resource exhaustion gracefully
    EXPECT_GT(backend_result.errors.size(), 0) << "Should have some resource-related errors";

    // Check for proper error handling
    bool found_resource_error = false;
    for (const auto& error : backend_result.errors) {
        if (error.error_code.find("MEMORY") != std::string::npos ||
            error.error_code.find("RESOURCE") != std::string::npos) {
            found_resource_error = true;
            break;
        }
    }
    EXPECT_TRUE(found_resource_error) << "Should have detected resource-related errors";

    LOG_INFO_PRINT("Resource exhaustion handling test completed successfully");
}

//=============================================================================
// Concurrent Execution Tests
//=============================================================================

/**
 * @brief Test concurrent inference execution
 */
class ConcurrentExecutionTest : public MLIntegrationTestBase {};

TEST_F(ConcurrentExecutionTest, MultiThreadedInference) {
    LOG_INFO_PRINT("Testing multi-threaded inference execution");

    auto model_config = classification_fixture_->get_model_config();

    auto result = framework_->test_concurrent_inference(InferenceBackend::TENSORRT_GPU,
                                                        model_config,
                                                        4,  // num_threads
                                                        10  // requests_per_thread
    );

    ASSERT_TRUE(result.is_ok()) << "Concurrent inference test failed: " << result.unwrap_err();

    auto backend_result = result.unwrap();
    EXPECT_TRUE(backend_result.success) << "Concurrent inference should succeed";
    EXPECT_EQ(backend_result.successful_iterations, 40) << "Should complete all 40 requests";

    // Verify thread safety - no crashes or data corruption
    EXPECT_FALSE(backend_result.performance.memory_leaks_detected)
        << "No memory leaks in concurrent execution";

    LOG_INFO_PRINT("Multi-threaded inference test completed successfully");
}

//=============================================================================
// Performance and Latency Tests
//=============================================================================

/**
 * @brief Test performance characteristics and latency requirements
 */
class PerformanceTest : public MLIntegrationTestBase {};

TEST_F(PerformanceTest, LatencyRequirements) {
    LOG_INFO_PRINT("Testing latency requirements compliance");

    auto scenario = TestScenarioBuilder()
                        .with_name("latency_test")
                        .with_single_backend(InferenceBackend::TENSORRT_GPU)
                        .with_model_config(classification_fixture_->get_model_config())
                        .with_test_fixture(classification_fixture_)
                        .with_max_latency(std::chrono::milliseconds(50))  // 50ms max latency
                        .with_iterations(20)
                        .with_warmup_iterations(5)
                        .build();

    ASSERT_TRUE(scenario.is_ok()) << "Scenario creation failed: " << scenario.unwrap_err();

    auto result = framework_->run_integration_test(scenario.unwrap());
    ASSERT_TRUE(result.is_ok()) << "Latency test failed: " << result.unwrap_err();

    auto test_result = result.unwrap();
    EXPECT_TRUE(test_result.overall_success) << "Latency requirements should be met";

    // Check specific latency metrics
    for (const auto& backend_result : test_result.backend_results) {
        EXPECT_LT(backend_result.performance.avg_latency, std::chrono::milliseconds(50))
            << "Average latency should be under 50ms";
        EXPECT_LT(backend_result.performance.max_latency, std::chrono::milliseconds(100))
            << "Maximum latency should be reasonable";
    }

    LOG_INFO_PRINT("Latency requirements test completed successfully");
}

TEST_F(PerformanceTest, ThroughputBenchmark) {
    LOG_INFO_PRINT("Testing throughput benchmarking");

    auto scenario =
        TestScenarioBuilder()
            .with_name("throughput_test")
            .with_backends({InferenceBackend::TENSORRT_GPU, InferenceBackend::ONNX_RUNTIME})
            .with_model_config(classification_fixture_->get_model_config())
            .with_test_fixture(classification_fixture_)
            .with_min_throughput(10.0f)  // 10 FPS minimum
            .with_iterations(50)
            .with_warmup_iterations(10)
            .build();

    ASSERT_TRUE(scenario.is_ok()) << "Scenario creation failed: " << scenario.unwrap_err();

    auto result = framework_->run_integration_test(scenario.unwrap());
    ASSERT_TRUE(result.is_ok()) << "Throughput test failed: " << result.unwrap_err();

    auto test_result = result.unwrap();
    EXPECT_TRUE(test_result.overall_success) << "Throughput requirements should be met";

    // Verify throughput for each backend
    for (const auto& backend_result : test_result.backend_results) {
        EXPECT_GT(backend_result.performance.throughput_fps, 10.0f)
            << "Throughput should exceed 10 FPS for backend: " << to_string(backend_result.backend);
    }

    LOG_INFO_PRINT("Throughput benchmark test completed successfully");
}

//=============================================================================
// End-to-End Pipeline Tests
//=============================================================================

/**
 * @brief Test complete end-to-end inference pipelines
 */
class EndToEndTest : public MLIntegrationTestBase {};

TEST_F(EndToEndTest, ClassificationPipeline) {
    LOG_INFO_PRINT("Testing end-to-end classification pipeline");

    auto scenarios = std::vector<TestScenario>{
        TestScenarioBuilder::create_correctness_test(
            "classification_correctness", InferenceBackend::TENSORRT_GPU, classification_fixture_)
            .build()
            .unwrap(),

        TestScenarioBuilder::create_performance_test("classification_performance",
                                                     {InferenceBackend::TENSORRT_GPU,
                                                      InferenceBackend::ONNX_RUNTIME},
                                                     classification_fixture_)
            .build()
            .unwrap()};

    auto result = framework_->run_test_suite(scenarios);
    ASSERT_TRUE(result.is_ok()) << "Test suite failed: " << result.unwrap_err();

    auto test_results = result.unwrap();
    EXPECT_EQ(test_results.size(), scenarios.size());

    for (const auto& test_result : test_results) {
        EXPECT_TRUE(test_result.overall_success)
            << "Test scenario " << test_result.scenario_name << " should succeed";
    }

    LOG_INFO_PRINT("End-to-end classification pipeline test completed successfully");
}

TEST_F(EndToEndTest, ObjectDetectionPipeline) {
    LOG_INFO_PRINT("Testing end-to-end object detection pipeline");

    auto scenario = TestScenarioBuilder()
                        .with_name("object_detection_test")
                        .with_single_backend(InferenceBackend::TENSORRT_GPU)
                        .with_model_config(object_detection_fixture_->get_model_config())
                        .with_test_fixture(object_detection_fixture_)
                        .with_iterations(10)
                        .build();

    ASSERT_TRUE(scenario.is_ok()) << "Scenario creation failed: " << scenario.unwrap_err();

    auto result = framework_->run_integration_test(scenario.unwrap());
    ASSERT_TRUE(result.is_ok()) << "Object detection test failed: " << result.unwrap_err();

    auto test_result = result.unwrap();
    EXPECT_TRUE(test_result.overall_success) << "Object detection pipeline should succeed";

    LOG_INFO_PRINT("End-to-end object detection pipeline test completed successfully");
}

//=============================================================================
// Test Suite Entry Point
//=============================================================================

/**
 * @brief Integration test suite that can run with different configurations
 */
class IntegrationTestSuite : public ::testing::Test {
  public:
    static void SetUpTestSuite() { LOG_INFO_PRINT("Starting ML Integration Test Suite"); }

    static void TearDownTestSuite() { LOG_INFO_PRINT("Completed ML Integration Test Suite"); }
};

/**
 * @brief Comprehensive integration test that exercises all major components
 */
TEST_F(IntegrationTestSuite, ComprehensiveIntegrationTest) {
    LOG_INFO_PRINT("Running comprehensive integration test");

    // Set up framework with mocks
    auto framework = create_mock_integration_framework();

    // Create a comprehensive test scenario
    auto classification_fixture =
        std::make_shared<ClassificationTestFixture>(ClassificationTestFixture::create()
                                                        .with_model("comprehensive_test_model")
                                                        .with_input_shape({1, 3, 224, 224})
                                                        .with_num_classes(1000));

    auto scenario = TestScenarioBuilder()
                        .with_name("comprehensive_integration_test")
                        .with_backends({InferenceBackend::TENSORRT_GPU,
                                        InferenceBackend::ONNX_RUNTIME,
                                        InferenceBackend::RULE_BASED})
                        .with_model_config(classification_fixture->get_model_config())
                        .with_test_fixture(classification_fixture)
                        .with_validation_strategy(ValidationStrategy::STATISTICAL_COMPARISON)
                        .with_iterations(10)
                        .with_warmup_iterations(2)
                        .build();

    ASSERT_TRUE(scenario.is_ok()) << "Comprehensive scenario creation failed: "
                                  << scenario.unwrap_err();

    auto result = framework->run_integration_test(scenario.unwrap());
    ASSERT_TRUE(result.is_ok()) << "Comprehensive integration test failed: " << result.unwrap_err();

    auto test_result = result.unwrap();
    EXPECT_TRUE(test_result.overall_success) << "Comprehensive test should succeed";
    EXPECT_EQ(test_result.backend_results.size(), 3) << "Should test all three backends";

    // Verify each backend was tested
    std::vector<InferenceBackend> tested_backends;
    for (const auto& backend_result : test_result.backend_results) {
        tested_backends.push_back(backend_result.backend);
        EXPECT_TRUE(backend_result.success)
            << "Backend " << to_string(backend_result.backend) << " should succeed";
    }

    EXPECT_NE(
        std::find(tested_backends.begin(), tested_backends.end(), InferenceBackend::TENSORRT_GPU),
        tested_backends.end());
    EXPECT_NE(
        std::find(tested_backends.begin(), tested_backends.end(), InferenceBackend::ONNX_RUNTIME),
        tested_backends.end());
    EXPECT_NE(
        std::find(tested_backends.begin(), tested_backends.end(), InferenceBackend::RULE_BASED),
        tested_backends.end());

    LOG_INFO_PRINT("Comprehensive integration test completed successfully");
}

//=============================================================================
// Main Test Function
//=============================================================================

/**
 * @brief Main function for running integration tests
 *
 * This can be used to run the tests independently or as part of a larger
 * test suite. The function demonstrates how to set up and execute the
 * integration testing framework.
 */
int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Set up logging for the test run
    setup_test_environment("INFO");

    LOG_INFO_PRINT("Starting ML Integration Tests");
    LOG_INFO_PRINT("Testing framework version: 1.0.0");
    LOG_INFO_PRINT("Backend support: TensorRT (mock), ONNX Runtime (mock), Rule-based (mock)");

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Clean up
    cleanup_test_environment();

    LOG_INFO_PRINT("ML Integration Tests completed with result: {}", result);

    return result;
}
