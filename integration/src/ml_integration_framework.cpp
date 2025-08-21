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
// IntegrationTestResult Implementation
//=============================================================================

std::string IntegrationTestResult::generate_report() const {
    std::stringstream report;

    report << "=== ML Integration Test Report ===\n";
    report << "Scenario: " << scenario_name << "\n";
    report << "Test Mode: " << to_string(mode) << "\n";
    report << "Overall Success: " << (overall_success ? "PASS" : "FAIL") << "\n";
    report << "Total Duration: " << total_test_duration.count() << " ms\n";
    report << "Memory Allocated: " << total_memory_allocated << " bytes\n";
    report << "Memory Freed: " << total_memory_freed << " bytes\n\n";

    if (cross_backend_similarity.has_value()) {
        report << "Cross-Backend Similarity: " << *cross_backend_similarity << "\n\n";
    }

    report << "Backend Results:\n";
    for (const auto& result : backend_results) {
        report << "  Backend: " << to_string(result.backend) << "\n";
        report << "    Success: " << (result.success ? "PASS" : "FAIL") << "\n";
        report << "    Iterations: " << result.successful_iterations << " successful, "
               << result.failed_iterations << " failed\n";
        report << "    Avg Latency: " << result.performance.avg_latency.count() << " ms\n";
        report << "    Throughput: " << result.performance.throughput_fps << " FPS\n";
        report << "    Peak Memory: " << result.performance.peak_memory_usage_mb << " MB\n";

        if (!result.errors.empty()) {
            report << "    Errors: " << result.errors.size() << " total\n";
        }
        report << "\n";
    }

    if (!warnings.empty()) {
        report << "Warnings:\n";
        for (const auto& warning : warnings) {
            report << "  - " << warning << "\n";
        }
    }

    return report.str();
}

//=============================================================================
// ProductionBackendFactory Implementation
//=============================================================================

auto ProductionBackendFactory::create_engine(InferenceBackend backend, const ModelConfig& config)
    -> Result<std::unique_ptr<InferenceEngine>, InferenceError> {
    // This would normally call the real factory function
    // For now, return an error since we're focusing on the testing framework
    return Err(InferenceError::BACKEND_NOT_AVAILABLE);
}

auto ProductionBackendFactory::is_backend_available(InferenceBackend backend) const -> bool {
    // In a real implementation, this would check for hardware availability
    switch (backend) {
        case InferenceBackend::RULE_BASED:
            return true;  // Always available
        case InferenceBackend::TENSORRT_GPU:
            return false;  // Would check for CUDA/TensorRT
        case InferenceBackend::ONNX_RUNTIME:
            return false;  // Would check for ONNX Runtime installation
        default:
            return false;
    }
}

auto ProductionBackendFactory::get_backend_info(InferenceBackend backend) const -> std::string {
    return "Production backend: " + to_string(backend);
}

//=============================================================================
// TestBackendFactory Implementation
//=============================================================================

void TestBackendFactory::inject_mock_engine(
    InferenceBackend backend, std::function<std::unique_ptr<InferenceEngine>()> factory) {
    mock_factories_[backend] = std::move(factory);
}

void TestBackendFactory::remove_mock_injection(InferenceBackend backend) {
    mock_factories_.erase(backend);
}

void TestBackendFactory::clear_mock_injections() {
    mock_factories_.clear();
}

auto TestBackendFactory::create_engine(InferenceBackend backend, const ModelConfig& config)
    -> Result<std::unique_ptr<InferenceEngine>, InferenceError> {
    // Check for mock injection first
    auto mock_it = mock_factories_.find(backend);
    if (mock_it != mock_factories_.end()) {
        auto engine = mock_it->second();
        if (engine) {
            return Ok(std::move(engine));
        } else {
            return Err(InferenceError::BACKEND_NOT_AVAILABLE);
        }
    }

    // Fall back to production factory
    if (!production_factory_) {
        production_factory_ = std::make_unique<ProductionBackendFactory>();
    }

    return production_factory_->create_engine(backend, config);
}

auto TestBackendFactory::is_backend_available(InferenceBackend backend) const -> bool {
    // If we have a mock injection, the backend is available
    if (mock_factories_.find(backend) != mock_factories_.end()) {
        return true;
    }

    // Otherwise check production factory
    if (!production_factory_) {
        const_cast<TestBackendFactory*>(this)->production_factory_ =
            std::make_unique<ProductionBackendFactory>();
    }

    return production_factory_->is_backend_available(backend);
}

auto TestBackendFactory::get_backend_info(InferenceBackend backend) const -> std::string {
    if (mock_factories_.find(backend) != mock_factories_.end()) {
        return "Mock backend: " + to_string(backend);
    }

    if (!production_factory_) {
        const_cast<TestBackendFactory*>(this)->production_factory_ =
            std::make_unique<ProductionBackendFactory>();
    }

    return production_factory_->get_backend_info(backend);
}

//=============================================================================
// MLIntegrationFramework Implementation
//=============================================================================

MLIntegrationFramework::MLIntegrationFramework()
    : backend_factory_(std::make_unique<ProductionBackendFactory>()), verbose_logging_(false) {
    LOG_INFO_PRINT("ML Integration Framework initialized with production factory");
}

MLIntegrationFramework::MLIntegrationFramework(std::unique_ptr<BackendFactory> factory)
    : backend_factory_(std::move(factory)), verbose_logging_(false) {
    LOG_INFO_PRINT("ML Integration Framework initialized with custom factory");
}

auto MLIntegrationFramework::run_integration_test(const TestScenario& scenario)
    -> Result<IntegrationTestResult, std::string> {
    if (verbose_logging_) {
        LOG_INFO_PRINT("Starting integration test: {}", scenario.name);
    }

    // Validate scenario
    auto validation_result = scenario.validate();
    if (validation_result.is_err()) {
        return Err(std::string("Scenario validation failed: ") + validation_result.unwrap_err());
    }

    auto start_time = std::chrono::steady_clock::now();
    IntegrationTestResult result;
    result.scenario_name = scenario.name;
    result.mode = scenario.mode;

    try {
        switch (scenario.mode) {
            case TestMode::SINGLE_BACKEND:
                if (scenario.backends.size() != 1) {
                    return Err(std::string("Single backend mode requires exactly one backend"));
                }
                return run_single_backend_test(scenario, scenario.backends[0])
                    .map([&](BackendTestResult backend_result) {
                        result.backend_results.push_back(std::move(backend_result));
                        result.overall_success = result.backend_results[0].success;
                        auto end_time = std::chrono::steady_clock::now();
                        result.total_test_duration =
                            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                                  start_time);
                        return result;
                    });

            case TestMode::MULTI_BACKEND:
                return run_multi_backend_test(scenario);

            default:
                return Err(std::string("Test mode not yet implemented: ") +
                           to_string(scenario.mode));
        }
    } catch (const std::exception& e) {
        return Err(std::string("Exception during test execution: ") + e.what());
    }
}

auto MLIntegrationFramework::run_test_suite(const std::vector<TestScenario>& scenarios)
    -> Result<std::vector<IntegrationTestResult>, std::string> {
    std::vector<IntegrationTestResult> results;
    results.reserve(scenarios.size());

    for (const auto& scenario : scenarios) {
        auto result = run_integration_test(scenario);
        if (result.is_err()) {
            return Err(std::string("Test suite failed at scenario '") + scenario.name +
                       "': " + result.unwrap_err());
        }
        results.push_back(result.unwrap());
    }

    return Ok(std::move(results));
}

auto MLIntegrationFramework::test_single_backend(InferenceBackend backend,
                                                 const ModelConfig& config,
                                                 const std::vector<InferenceRequest>& inputs,
                                                 const PerformanceThresholds& thresholds)
    -> Result<BackendTestResult, std::string> {
    if (verbose_logging_) {
        LOG_INFO_PRINT("Testing single backend: {}", to_string(backend));
    }

    BackendTestResult result;
    result.backend = backend;
    auto start_time = std::chrono::steady_clock::now();

    // Create engine
    auto engine_result = backend_factory_->create_engine(backend, config);
    if (engine_result.is_err()) {
        result.success = false;
        result.errors.push_back(
            TestError{.error_code = "ENGINE_CREATION_FAILED",
                      .error_message = "Failed to create engine for backend: " + to_string(backend),
                      .backend_name = to_string(backend)});
        return Ok(std::move(result));
    }

    auto engine = engine_result.unwrap();

    // Run inference for each input
    for (const auto& input : inputs) {
        auto inference_start = std::chrono::steady_clock::now();
        auto inference_result = engine->run_inference(input);
        auto inference_end = std::chrono::steady_clock::now();

        auto latency =
            std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start);

        if (inference_result.is_ok()) {
            result.outputs.push_back(inference_result.unwrap());
            result.successful_iterations++;

            // Update performance metrics
            result.performance.total_test_time += latency;
            if (result.performance.min_latency == std::chrono::milliseconds(0) ||
                latency < result.performance.min_latency) {
                result.performance.min_latency = latency;
            }
            if (latency > result.performance.max_latency) {
                result.performance.max_latency = latency;
            }
        } else {
            result.failed_iterations++;
            result.errors.push_back(TestError{.error_code = "INFERENCE_FAILED",
                                              .error_message = "Inference failed",
                                              .backend_name = to_string(backend)});
        }
    }

    // Calculate final metrics
    if (result.successful_iterations > 0) {
        result.performance.avg_latency =
            result.performance.total_test_time / result.successful_iterations;
        auto total_time_seconds = result.performance.total_test_time.count() / 1000.0f;
        if (total_time_seconds > 0) {
            result.performance.throughput_fps = result.successful_iterations / total_time_seconds;
        }
    }

    result.success = (result.failed_iterations == 0) &&
                     result.performance.meets_thresholds(thresholds);

    auto end_time = std::chrono::steady_clock::now();
    result.total_test_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (verbose_logging_) {
        LOG_INFO_PRINT("Backend test completed: {} iterations successful, {} failed",
                       result.successful_iterations,
                       result.failed_iterations);
    }

    return Ok(std::move(result));
}

void MLIntegrationFramework::inject_mock_backend(
    InferenceBackend backend, std::function<std::unique_ptr<InferenceEngine>()> factory) {
    auto test_factory = dynamic_cast<TestBackendFactory*>(backend_factory_.get());
    if (test_factory) {
        test_factory->inject_mock_engine(backend, std::move(factory));
        if (verbose_logging_) {
            LOG_INFO_PRINT("Mock backend injected: {}", to_string(backend));
        }
    } else {
        LOG_WARN_PRINT("Cannot inject mock - not using TestBackendFactory");
    }
}

void MLIntegrationFramework::set_backend_factory(std::unique_ptr<BackendFactory> factory) {
    backend_factory_ = std::move(factory);
    if (verbose_logging_) {
        LOG_INFO_PRINT("Backend factory updated");
    }
}

void MLIntegrationFramework::set_verbose_logging(bool enable) {
    verbose_logging_ = enable;
    if (enable) {
        LOG_INFO_PRINT("Verbose logging enabled");
    }
}

void MLIntegrationFramework::set_default_performance_thresholds(
    const PerformanceThresholds& thresholds) {
    default_thresholds_ = thresholds;
    if (verbose_logging_) {
        LOG_INFO_PRINT("Default performance thresholds updated");
    }
}

// Placeholder implementations for other methods
auto MLIntegrationFramework::run_single_backend_test(const TestScenario& scenario,
                                                     InferenceBackend backend)
    -> Result<BackendTestResult, std::string> {
    return test_single_backend(
        backend, scenario.model_config, scenario.test_inputs, scenario.performance);
}

auto MLIntegrationFramework::run_multi_backend_test(const TestScenario& scenario)
    -> Result<IntegrationTestResult, std::string> {
    IntegrationTestResult result;
    result.scenario_name = scenario.name;
    result.mode = scenario.mode;

    // Run each backend individually
    for (auto backend : scenario.backends) {
        auto backend_result = run_single_backend_test(scenario, backend);
        if (backend_result.is_ok()) {
            result.backend_results.push_back(backend_result.unwrap());
        } else {
            return Err(backend_result.unwrap_err());
        }
    }

    // Determine overall success
    result.overall_success = std::all_of(result.backend_results.begin(),
                                         result.backend_results.end(),
                                         [](const BackendTestResult& r) { return r.success; });

    return Ok(std::move(result));
}

//=============================================================================
// Factory Functions Implementation
//=============================================================================

auto create_default_test_scenario(const std::string& name,
                                  InferenceBackend backend,
                                  const ModelConfig& config) -> TestScenario {
    TestScenario scenario;
    scenario.name = name;
    scenario.mode = TestMode::SINGLE_BACKEND;
    scenario.backends = {backend};
    scenario.model_config = config;
    scenario.validation = ValidationStrategy::STATISTICAL_COMPARISON;
    scenario.num_iterations = 1;
    scenario.tolerance = 1e-5f;

    return scenario;
}

}  // namespace inference_lab::integration
