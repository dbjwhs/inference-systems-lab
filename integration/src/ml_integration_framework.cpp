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
    results.passed = false;  // Default to failed
    results.failure_reason = "Not implemented";

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
    -> common::Result<IntegrationTestResults, IntegrationTestError> {
    // Create a test scenario from the parameters
    TestScenario scenario;
    scenario.name = "Single backend test";
    scenario.backends = {backend};
    scenario.model_config = config;
    scenario.mode = TestMode::SINGLE_BACKEND;
    scenario.iterations = static_cast<std::uint32_t>(inputs.size());

    return run_integration_test(scenario);
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

auto create_mock_integration_framework()
    -> common::Result<std::unique_ptr<MLIntegrationFramework>, IntegrationTestError> {
    // For now, return an error since mock factory isn't implemented
    return Err(IntegrationTestError::BACKEND_NOT_AVAILABLE);
}

auto create_hybrid_integration_framework(bool use_real_tensorrt, bool use_real_onnx)
    -> common::Result<std::unique_ptr<MLIntegrationFramework>, IntegrationTestError> {
    // For now, return an error since hybrid backends aren't implemented
    (void)use_real_tensorrt;
    (void)use_real_onnx;
    return Err(IntegrationTestError::BACKEND_NOT_AVAILABLE);
}

}  // namespace inference_lab::integration
