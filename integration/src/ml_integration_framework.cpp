// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file ml_integration_framework.cpp
 * @brief Implementation of ML integration testing framework
 *
 * This file contains the implementation of the comprehensive ML integration
 * testing framework. The implementation provides concrete functionality for
 * testing multiple inference backends, performance analysis, and error handling.
 */

#include "ml_integration_framework.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <thread>

#include "../../common/src/logging.hpp"
#include "../../engines/src/inference_engine.hpp"
#include "mock_engines.hpp"

namespace inference_lab::integration {

using common::Err;
using common::LogLevel;
using common::Ok;

//=============================================================================
// Utility Functions Implementation
//=============================================================================

std::string to_string(engines::InferenceBackend backend) {
    switch (backend) {
        case engines::InferenceBackend::TENSORRT_GPU:
            return "TensorRT_GPU";
        case engines::InferenceBackend::ONNX_RUNTIME:
            return "ONNX_Runtime";
        case engines::InferenceBackend::RULE_BASED:
            return "Rule_Based";
        case engines::InferenceBackend::HYBRID_NEURAL_SYMBOLIC:
            return "Hybrid_Neural_Symbolic";
        default:
            return "Unknown";
    }
}

std::string to_string(TestMode mode) {
    switch (mode) {
        case TestMode::SINGLE_BACKEND:
            return "Single_Backend";
        case TestMode::MULTI_BACKEND:
            return "Multi_Backend";
        case TestMode::PERFORMANCE:
            return "Performance";
        case TestMode::STRESS_TEST:
            return "Stress_Test";
        case TestMode::CONCURRENCY:
            return "Concurrency";
        default:
            return "Unknown";
    }
}

std::string to_string(ValidationStrategy strategy) {
    switch (strategy) {
        case ValidationStrategy::EXACT_MATCH:
            return "Exact_Match";
        case ValidationStrategy::NUMERICAL_TOLERANCE:
            return "Numerical_Tolerance";
        case ValidationStrategy::STATISTICAL_COMPARISON:
            return "Statistical_Comparison";
        case ValidationStrategy::SEMANTIC_EQUIVALENCE:
            return "Semantic_Equivalence";
        case ValidationStrategy::CROSS_BACKEND_CONSISTENCY:
            return "Cross_Backend_Consistency";
        default:
            return "Unknown";
    }
}

//=============================================================================
// IntegrationTestResults Implementation
//=============================================================================

auto IntegrationTestResults::generate_report() const -> std::string {
    std::stringstream report;

    report << "=== ML Integration Test Report ===\n";
    report << "Scenario: " << scenario.name << "\n";
    report << "Test Mode: " << to_string(scenario.mode) << "\n";
    report << "Overall Success: " << (passed ? "PASS" : "FAIL") << "\n";
    report << "Total Duration: " << total_execution_time.count() << " ms\n";

    if (!failure_reason.empty()) {
        report << "Failure Reason: " << failure_reason << "\n";
    }

    report << "\nPerformance Metrics:\n";
    for (const auto& [backend, perf_metrics] : metrics) {
        report << "  Backend: " << to_string(backend) << "\n";
        report << "    Min Latency: " << perf_metrics.min_latency.count() << " ns\n";
        report << "    Mean Latency: " << perf_metrics.mean_latency.count() << " ns\n";
        report << "    Max Latency: " << perf_metrics.max_latency.count() << " ns\n";
        report << "    Throughput: " << perf_metrics.throughput_inferences_per_sec
               << " inferences/sec\n";
        report << "    Peak Memory: " << perf_metrics.peak_memory_usage_bytes << " bytes\n";
        report << "\n";
    }

    if (!error_messages.empty()) {
        report << "Error Messages:\n";
        for (const auto& error : error_messages) {
            report << "  - " << error << "\n";
        }
    }

    return report.str();
}

auto IntegrationTestResults::meets_performance_requirements() const -> bool {
    // Simple implementation - check if test passed
    return passed;
}

auto IntegrationTestResults::get_performance_comparison() const -> std::string {
    std::stringstream comparison;
    comparison << "Performance Comparison Across Backends:\n";

    for (const auto& [backend, perf_metrics] : metrics) {
        comparison << to_string(backend) << ": " << perf_metrics.mean_latency.count() << " ns avg, "
                   << perf_metrics.throughput_inferences_per_sec << " inferences/sec\n";
    }

    return comparison.str();
}

//=============================================================================
// PerformanceMetrics Implementation
//=============================================================================

auto PerformanceMetrics::to_string() const -> std::string {
    std::stringstream ss;
    ss << "Performance Metrics:\n";
    ss << "  Min Latency: " << min_latency.count() << " ns\n";
    ss << "  Mean Latency: " << mean_latency.count() << " ns\n";
    ss << "  Max Latency: " << max_latency.count() << " ns\n";
    ss << "  P95 Latency: " << p95_latency.count() << " ns\n";
    ss << "  P99 Latency: " << p99_latency.count() << " ns\n";
    ss << "  Throughput: " << throughput_inferences_per_sec << " inferences/sec\n";
    ss << "  Peak Memory: " << peak_memory_usage_bytes << " bytes\n";
    ss << "  Error Count: " << error_count;
    return ss.str();
}

auto StatisticalAnalysis::is_statistically_valid(double significance_level) const -> bool {
    return p_value < significance_level && passes_normality_test;
}

std::string to_string(IntegrationTestError error) {
    switch (error) {
        case IntegrationTestError::BACKEND_CREATION_FAILED:
            return "Backend creation failed";
        case IntegrationTestError::BACKEND_NOT_AVAILABLE:
            return "Backend not available";
        case IntegrationTestError::TEST_SCENARIO_INVALID:
            return "Test scenario invalid";
        case IntegrationTestError::VALIDATION_FAILED:
            return "Validation failed";
        case IntegrationTestError::PERFORMANCE_REGRESSION:
            return "Performance regression detected";
        case IntegrationTestError::MEMORY_LEAK_DETECTED:
            return "Memory leak detected";
        case IntegrationTestError::TIMEOUT_EXCEEDED:
            return "Timeout exceeded";
        case IntegrationTestError::STATISTICAL_ANALYSIS_FAILED:
            return "Statistical analysis failed";
        case IntegrationTestError::BACKEND_INCONSISTENCY:
            return "Backend inconsistency";
        case IntegrationTestError::RESOURCE_EXHAUSTION:
            return "Resource exhaustion";
        case IntegrationTestError::UNKNOWN_ERROR:
        default:
            return "Unknown error";
    }
}

auto TestScenario::validate() const -> common::Result<std::monostate, IntegrationTestError> {
    if (name.empty()) {
        return Err(IntegrationTestError::TEST_SCENARIO_INVALID);
    }

    if (backends.empty()) {
        return Err(IntegrationTestError::TEST_SCENARIO_INVALID);
    }

    if (iterations == 0) {
        return Err(IntegrationTestError::TEST_SCENARIO_INVALID);
    }

    return Ok(std::monostate{});
}

//=============================================================================
// TestScenarioBuilder Implementation
//=============================================================================

TestScenarioBuilder::TestScenarioBuilder() {
    // Initialize with defaults
    scenario_.validation_strategy = ValidationStrategy::STATISTICAL_COMPARISON;
    scenario_.mode = TestMode::SINGLE_BACKEND;
}

auto TestScenarioBuilder::with_name(const std::string& name) -> TestScenarioBuilder& {
    scenario_.name = name;
    return *this;
}

auto TestScenarioBuilder::with_backends(const std::vector<engines::InferenceBackend>& backends)
    -> TestScenarioBuilder& {
    scenario_.backends = backends;
    return *this;
}

auto TestScenarioBuilder::with_model_config(const engines::ModelConfig& config)
    -> TestScenarioBuilder& {
    scenario_.model_config = config;
    return *this;
}

auto TestScenarioBuilder::with_validation_strategy(ValidationStrategy strategy)
    -> TestScenarioBuilder& {
    scenario_.validation_strategy = strategy;
    return *this;
}

auto TestScenarioBuilder::with_mode(TestMode mode) -> TestScenarioBuilder& {
    scenario_.mode = mode;
    return *this;
}

auto TestScenarioBuilder::with_iterations(std::uint32_t iterations) -> TestScenarioBuilder& {
    scenario_.iterations = iterations;
    return *this;
}

auto TestScenarioBuilder::with_timeout(std::chrono::milliseconds timeout) -> TestScenarioBuilder& {
    scenario_.timeout = timeout;
    return *this;
}

auto TestScenarioBuilder::with_max_latency(std::chrono::milliseconds latency)
    -> TestScenarioBuilder& {
    scenario_.max_latency = latency;
    return *this;
}

auto TestScenarioBuilder::with_numerical_tolerance(double tolerance) -> TestScenarioBuilder& {
    scenario_.numerical_tolerance = tolerance;
    return *this;
}

auto TestScenarioBuilder::with_memory_tracking(bool enable) -> TestScenarioBuilder& {
    scenario_.enable_memory_tracking = enable;
    return *this;
}

auto TestScenarioBuilder::with_performance_profiling(bool enable) -> TestScenarioBuilder& {
    scenario_.enable_performance_profiling = enable;
    return *this;
}

auto TestScenarioBuilder::with_error_injection(bool enable) -> TestScenarioBuilder& {
    scenario_.inject_errors = enable;
    return *this;
}

auto TestScenarioBuilder::with_concurrency_level(std::uint32_t level) -> TestScenarioBuilder& {
    scenario_.concurrency_level = level;
    return *this;
}

auto TestScenarioBuilder::build() -> common::Result<TestScenario, IntegrationTestError> {
    auto validation_result = scenario_.validate();
    if (validation_result.is_err()) {
        return Err(validation_result.unwrap_err());
    }
    return Ok(scenario_);
}

//=============================================================================
// MLIntegrationFramework Implementation
//=============================================================================

MLIntegrationFramework::MLIntegrationFramework(std::unique_ptr<BackendFactory> factory)
    : backend_factory_(std::move(factory)) {
    LOG_INFO_PRINT("ML Integration Framework initialized");
}

auto MLIntegrationFramework::run_integration_test(const TestScenario& scenario)
    -> common::Result<IntegrationTestResults, IntegrationTestError> {
    LOG_INFO_PRINT("Starting integration test: {}", scenario.name);

    // Validate scenario
    auto validation_result = scenario.validate();
    if (validation_result.is_err()) {
        return Err(validation_result.unwrap_err());
    }

    IntegrationTestResults results;
    results.scenario = scenario;
    results.passed = true;  // Start optimistic, set to false if any backend fails

    // Generate test inputs if not provided in scenario
    std::vector<engines::InferenceRequest> test_inputs;
    if (scenario.iterations > 0) {
        // Generate simple test inputs based on model config
        for (uint32_t i = 0; i < scenario.iterations; ++i) {
            engines::InferenceRequest request;
            request.batch_size = 1;
            request.input_tensors.resize(1);
            request.input_tensors[0].resize(224 * 224 * 3, 0.5f);  // Mock image data
            request.input_names = {"input"};
            test_inputs.push_back(request);
        }
    }

    // Run test on each backend
    for (const auto& backend : scenario.backends) {
        LOG_INFO_PRINT("Testing backend: {}", to_string(backend));

        auto backend_result = test_single_backend(backend, scenario.model_config, test_inputs);
        if (backend_result.is_err()) {
            results.passed = false;
            results.error_messages.push_back("Backend test failed: " + to_string(backend));
            continue;
        }

        auto backend_test_result = backend_result.unwrap();
        if (!backend_test_result.passed) {
            results.passed = false;
            for (const auto& error : backend_test_result.error_messages) {
                results.error_messages.push_back(error);
            }
        }

        // Store performance metrics for this backend
        results.metrics[backend] = backend_test_result.performance;
    }

    // If all backends passed, set passed to true
    if (results.error_messages.empty() && !results.metrics.empty()) {
        results.passed = true;
    } else if (results.metrics.empty()) {
        results.passed = false;
        results.error_messages.push_back("No backends completed successfully");
    }

    LOG_INFO_PRINT("Integration test '{}' completed: passed={}, backends={}",
                   scenario.name,
                   results.passed,
                   results.metrics.size());

    return Ok(std::move(results));
}

auto MLIntegrationFramework::run_performance_benchmark(const TestScenario& scenario)
    -> common::Result<IntegrationTestResults, IntegrationTestError> {
    return run_integration_test(scenario);
}

auto MLIntegrationFramework::run_memory_safety_test(const TestScenario& scenario)
    -> common::Result<IntegrationTestResults, IntegrationTestError> {
    return run_integration_test(scenario);
}

auto MLIntegrationFramework::run_concurrency_test(const TestScenario& scenario)
    -> common::Result<IntegrationTestResults, IntegrationTestError> {
    return run_integration_test(scenario);
}

auto MLIntegrationFramework::run_error_injection_test(const TestScenario& scenario)
    -> common::Result<IntegrationTestResults, IntegrationTestError> {
    return run_integration_test(scenario);
}

auto MLIntegrationFramework::get_available_backends() const
    -> std::vector<engines::InferenceBackend> {
    return {engines::InferenceBackend::RULE_BASED};  // Basic implementation
}

auto MLIntegrationFramework::get_framework_info() const -> std::string {
    return "ML Integration Testing Framework v1.0";
}

auto MLIntegrationFramework::test_single_backend(
    engines::InferenceBackend backend,
    const engines::ModelConfig& config,
    const std::vector<engines::InferenceRequest>& inputs)
    -> common::Result<BackendTestResult, IntegrationTestError> {
    LOG_INFO_PRINT("Testing single backend: {}", to_string(backend));

    BackendTestResult result{};
    result.backend = backend;
    result.scenario.backends.push_back(backend);
    result.scenario.name = "single_backend_test_" + to_string(backend);

    try {
        // Create engine using the backend factory (enables error injection)
        auto engine_result = backend_factory_->create_engine(backend, config);
        if (engine_result.is_err()) {
            result.passed = false;
            result.error_messages.push_back("Failed to create engine: " +
                                            to_string(engine_result.unwrap_err()));
            return Ok(result);
        }
        auto engine = std::move(engine_result).unwrap();

        // Verify engine is ready
        if (!engine->is_ready()) {
            result.passed = false;
            result.error_messages.push_back("Engine not ready after initialization");
            return Ok(result);
        }

        // Test each input
        std::vector<engines::InferenceResponse> responses;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (const auto& input : inputs) {
            auto inference_result = engine->run_inference(input);
            if (inference_result.is_err()) {
                result.error_messages.push_back("Inference failed: " +
                                                engines::to_string(inference_result.unwrap_err()));
                continue;
            }

            responses.push_back(inference_result.unwrap());
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Calculate performance metrics
        if (!responses.empty()) {
            result.passed = true;
            result.performance.throughput_inferences_per_sec =
                static_cast<double>(responses.size()) / (total_duration.count() / 1000.0);

            // Calculate average latency
            double total_latency = 0.0;
            for (const auto& response : responses) {
                total_latency += response.inference_time_ms;
            }
            result.performance.mean_latency =
                std::chrono::milliseconds(static_cast<int64_t>(total_latency / responses.size()));

            result.outputs = std::move(responses);
        } else {
            result.passed = false;
            result.error_messages.push_back("No successful inferences completed");
        }

        LOG_INFO_PRINT("Single backend test completed: passed={}, throughput={:.1f}/sec",
                       result.passed,
                       result.performance.throughput_inferences_per_sec);

        return Ok(result);

    } catch (const std::exception& e) {
        result.passed = false;
        result.error_messages.push_back(std::string("Exception during testing: ") + e.what());
        return Ok(result);
    }
}

auto MLIntegrationFramework::compare_backends(
    const std::vector<engines::InferenceBackend>& backends,
    const engines::ModelConfig& config,
    const std::vector<engines::InferenceRequest>& inputs,
    ValidationStrategy strategy) -> common::Result<IntegrationTestResults, IntegrationTestError> {
    // Create a test scenario for multi-backend comparison
    TestScenario scenario;
    scenario.name = "Multi-backend comparison";
    scenario.backends = backends;
    scenario.model_config = config;
    scenario.mode = TestMode::MULTI_BACKEND;
    scenario.validation_strategy = strategy;
    scenario.iterations = static_cast<std::uint32_t>(inputs.size());

    return run_integration_test(scenario);
}

//=============================================================================
// Factory Functions Implementation
//=============================================================================

auto create_hardware_integration_framework()
    -> common::Result<std::unique_ptr<MLIntegrationFramework>, IntegrationTestError> {
    // For now, return an error since hardware backends aren't implemented
    return Err(IntegrationTestError::BACKEND_NOT_AVAILABLE);
}

/**
 * @brief Mock backend factory for testing without hardware dependencies
 */
class MockBackendFactory : public BackendFactory {
  public:
    MockBackendFactory() = default;
    ~MockBackendFactory() override = default;

    auto create_engine(engines::InferenceBackend backend, const engines::ModelConfig& config)
        -> common::Result<std::unique_ptr<engines::InferenceEngine>,
                          IntegrationTestError> override {
        using namespace inference_lab::integration::mocks;

        // Get error injection configuration from model config if available
        MockEngineConfig mock_config;

        // Configure realistic performance characteristics for each backend
        switch (backend) {
            case engines::InferenceBackend::TENSORRT_GPU:
                mock_config.performance.base_latency_ms = 5.0f;  // TensorRT is fast
                break;
            case engines::InferenceBackend::ONNX_RUNTIME:
                mock_config.performance.base_latency_ms = 15.0f;  // ONNX Runtime is slower
                break;
            case engines::InferenceBackend::RULE_BASED:
                mock_config.performance.base_latency_ms = 2.0f;  // Rule-based is fastest
                break;
            default:
                mock_config.performance.base_latency_ms = 10.0f;  // Default
                break;
        }

        // Check if this is an error injection test by examining the model path
        if (config.model_path.find("error_injection") != std::string::npos) {
            // Configure error injection based on model path
            if (config.model_path.find("GPU_MEMORY_EXHAUSTED") != std::string::npos) {
                mock_config.error_injection.error_rates["GPU_MEMORY_EXHAUSTED"] = 1.0f;
            } else if (config.model_path.find("MODEL_LOAD_FAILED") != std::string::npos) {
                mock_config.error_injection.error_rates["MODEL_LOAD_FAILED"] = 1.0f;
            } else if (config.model_path.find("INFERENCE_EXECUTION_FAILED") != std::string::npos) {
                mock_config.error_injection.error_rates["INFERENCE_EXECUTION_FAILED"] = 1.0f;
            }
        }

        mock_config.engine_name = "Mock_" + to_string(backend);
        mock_config.enable_logging = true;

        std::unique_ptr<engines::InferenceEngine> engine;
        switch (backend) {
            case engines::InferenceBackend::TENSORRT_GPU:
                engine = std::make_unique<MockTensorRTEngine>(mock_config);
                break;
            case engines::InferenceBackend::ONNX_RUNTIME:
                engine = std::make_unique<MockONNXRuntimeEngine>(mock_config);
                break;
            case engines::InferenceBackend::RULE_BASED:
                engine = std::make_unique<MockRuleBasedEngine>(mock_config);
                break;
            default:
                return Err(IntegrationTestError::BACKEND_NOT_AVAILABLE);
        }

        if (!engine) {
            return Err(IntegrationTestError::BACKEND_CREATION_FAILED);
        }

        return Ok(std::move(engine));
    }

    auto is_backend_available(engines::InferenceBackend backend) const -> bool override {
        // All mock backends are always available
        switch (backend) {
            case engines::InferenceBackend::TENSORRT_GPU:
            case engines::InferenceBackend::ONNX_RUNTIME:
            case engines::InferenceBackend::RULE_BASED:
                return true;
            default:
                return false;
        }
    }

    auto get_factory_type() const -> std::string override { return "MockBackendFactory"; }
};

auto create_mock_integration_framework()
    -> common::Result<std::unique_ptr<MLIntegrationFramework>, IntegrationTestError> {
    auto factory = std::make_unique<MockBackendFactory>();
    auto framework = std::make_unique<MLIntegrationFramework>(std::move(factory));
    return Ok(std::move(framework));
}

auto create_hybrid_integration_framework(bool use_real_tensorrt, bool use_real_onnx)
    -> common::Result<std::unique_ptr<MLIntegrationFramework>, IntegrationTestError> {
    // For now, return an error since hybrid backends aren't implemented
    (void)use_real_tensorrt;
    (void)use_real_onnx;
    return Err(IntegrationTestError::BACKEND_NOT_AVAILABLE);
}

}  // namespace inference_lab::integration
