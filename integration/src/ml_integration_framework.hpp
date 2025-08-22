// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file ml_integration_framework.hpp
 * @brief Comprehensive ML integration testing framework for inference systems
 *
 * This framework provides a unified testing interface for validating ML inference
 * integration across different backends, scenarios, and error conditions. It supports
 * both hardware-based testing with real GPU acceleration and mock-based testing for
 * CI/CD environments without specialized hardware.
 *
 * Key Features:
 * - Multi-backend testing (TensorRT, ONNX Runtime, Rule-Based, Hybrid)
 * - Mock engine implementations for hardware-free testing
 * - Statistical validation with configurable tolerance levels
 * - Performance benchmarking with regression detection
 * - Memory safety testing with leak detection
 * - Comprehensive error injection and recovery testing
 * - Thread safety validation for concurrent inference
 *
 * Design Principles:
 * - Follows project's Result<T,E> error handling patterns
 * - RAII resource management for GPU/CPU memory
 * - Zero-cost abstractions where possible
 * - Dependency injection for backend selection
 * - Statistical analysis for result validation
 * - Modern C++17 patterns and template metaprogramming
 *
 * Usage Example:
 * @code
 * // Create testing framework with hardware backends
 * auto framework = create_hardware_integration_framework();
 *
 * // Or use mock backends for CI/CD
 * auto mock_framework = create_mock_integration_framework();
 *
 * // Create test scenario
 * auto scenario = TestScenarioBuilder()
 *     .with_name("classification_accuracy_test")
 *     .with_backends({InferenceBackend::TENSORRT_GPU, InferenceBackend::ONNX_RUNTIME})
 *     .with_model_config(resnet50_config)
 *     .with_validation_strategy(ValidationStrategy::STATISTICAL_COMPARISON)
 *     .with_iterations(1000)
 *     .build();
 *
 * // Run integration test
 * auto results = framework->run_integration_test(scenario.unwrap());
 * if (results.is_ok()) {
 *     auto report = results.unwrap().generate_report();
 *     LOG_INFO_PRINT("Test Results:\n{}", report);
 * }
 * @endcode
 *
 * Architecture Flow:
 * @code
 *   ┌─────────────────┐   create_test()   ┌─────────────────┐   inject_backends()
 * ┌─────────────────┐ │ Test Scenario   │ ──────────────▶   │ ML Integration  │ ─────────────────▶
 * │ Backend Factory │ │ Configuration   │                   │ Framework       │ │ (Real/Mock)     │
 *   └─────────────────┘                   └─────────────────┘ └─────────────────┘ │ │ │ │
 * validation_rules                      │ execute_test()                         │ create_engines()
 *            ▼                                      ▼                                         ▼
 *   ┌─────────────────┐                   ┌─────────────────┐   parallel_execution
 * ┌─────────────────┐ │ Statistical     │ ◄─────────────────│ Test Executor   │ ─────────────────▶
 * │ Inference       │ │ Validator       │   validate_output │ & Coordinator   │ │ Engines         │
 *   └─────────────────┘                   └─────────────────┘                       │
 * (Multi-Backend) │ │                                      │ └─────────────────┘ │ generate_metrics
 * │ collect_results                         │ ▼                                      ▼ │
 *   ┌─────────────────┐                   ┌─────────────────┐   memory_tracking ┌─────────────────┐
 *   │ Performance     │ ◄─────────────────│ Results         │ ◄─────────────────   │ Resource │ │
 * Report & Stats  │                   │ Aggregator      │                       │ Monitor         │
 *   └─────────────────┘                   └─────────────────┘ └─────────────────┘
 * @endcode
 */

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../common/src/logging.hpp"
#include "../../common/src/result.hpp"
#include "../../engines/src/inference_engine.hpp"

namespace inference_lab::integration {

/**
 * @brief Integration testing specific error types
 */
enum class IntegrationTestError : std::uint8_t {
    BACKEND_CREATION_FAILED,      ///< Failed to create inference backend
    BACKEND_NOT_AVAILABLE,        ///< Requested backend not available
    TEST_SCENARIO_INVALID,        ///< Invalid test scenario configuration
    VALIDATION_FAILED,            ///< Output validation failed
    PERFORMANCE_REGRESSION,       ///< Performance below expected thresholds
    MEMORY_LEAK_DETECTED,         ///< Memory leak detected during testing
    TIMEOUT_EXCEEDED,             ///< Test execution timeout exceeded
    STATISTICAL_ANALYSIS_FAILED,  ///< Statistical validation failed
    BACKEND_INCONSISTENCY,        ///< Results inconsistent across backends
    RESOURCE_EXHAUSTION,          ///< Insufficient system resources
    UNKNOWN_ERROR                 ///< Unexpected error condition
};

/**
 * @brief Convert IntegrationTestError to human-readable string
 */
std::string to_string(IntegrationTestError error);

/**
 * @brief Validation strategies for test result verification
 */
enum class ValidationStrategy : std::uint8_t {
    EXACT_MATCH,               ///< Bit-for-bit exact output matching
    NUMERICAL_TOLERANCE,       ///< Floating point comparison with tolerance
    STATISTICAL_COMPARISON,    ///< Statistical significance testing
    SEMANTIC_EQUIVALENCE,      ///< Domain-specific semantic validation
    CROSS_BACKEND_CONSISTENCY  ///< Consistency across multiple backends
};

/**
 * @brief Test execution modes for different scenarios
 */
enum class TestMode : std::uint8_t {
    SINGLE_BACKEND,  ///< Test single backend in isolation
    MULTI_BACKEND,   ///< Test multiple backends for consistency
    PERFORMANCE,     ///< Focus on performance characteristics
    STRESS_TEST,     ///< Resource exhaustion and error injection
    CONCURRENCY      ///< Thread safety and parallel execution
};

/**
 * @brief Performance metrics collected during testing
 */
struct PerformanceMetrics {
    std::chrono::nanoseconds min_latency{0};    ///< Minimum inference latency
    std::chrono::nanoseconds max_latency{0};    ///< Maximum inference latency
    std::chrono::nanoseconds mean_latency{0};   ///< Average inference latency
    std::chrono::nanoseconds p95_latency{0};    ///< 95th percentile latency
    std::chrono::nanoseconds p99_latency{0};    ///< 99th percentile latency
    double throughput_inferences_per_sec{0.0};  ///< Sustained throughput
    std::uint64_t peak_memory_usage_bytes{0};   ///< Peak memory consumption
    std::uint64_t total_allocations{0};         ///< Total memory allocations
    double cpu_utilization_percent{0.0};        ///< CPU usage during test
    double gpu_utilization_percent{0.0};        ///< GPU usage (if applicable)
    std::uint32_t error_count{0};               ///< Number of inference errors

    /**
     * @brief Generate human-readable performance summary
     */
    auto to_string() const -> std::string;
};

/**
 * @brief Statistical analysis results for validation
 */
struct StatisticalAnalysis {
    double mean_difference{0.0};         ///< Mean difference between outputs
    double standard_deviation{0.0};      ///< Standard deviation of differences
    double confidence_interval_95{0.0};  ///< 95% confidence interval
    double p_value{0.0};                 ///< Statistical significance p-value
    bool passes_normality_test{false};   ///< Whether data passes normality test
    std::uint32_t sample_count{0};       ///< Number of samples analyzed

    /**
     * @brief Check if statistical validation passes
     */
    auto is_statistically_valid(double significance_level = 0.05) const -> bool;
};

/**
 * @brief Test scenario configuration and parameters
 */
struct TestScenario {
    std::string name;                                 ///< Human-readable test name
    std::vector<engines::InferenceBackend> backends;  ///< Backends to test
    engines::ModelConfig model_config;                ///< Model configuration
    ValidationStrategy validation_strategy;           ///< How to validate results
    TestMode mode;                                    ///< Test execution mode
    std::uint32_t iterations{100};                    ///< Number of test iterations
    std::chrono::milliseconds timeout{30000};         ///< Per-test timeout
    std::chrono::milliseconds max_latency{1000};      ///< Maximum acceptable latency
    double numerical_tolerance{1e-6};                 ///< Tolerance for numerical comparison
    bool enable_memory_tracking{true};                ///< Whether to track memory usage
    bool enable_performance_profiling{true};          ///< Whether to profile performance
    bool inject_errors{false};                        ///< Whether to inject error conditions
    std::uint32_t concurrency_level{1};               ///< Number of parallel threads

    /**
     * @brief Validate scenario configuration
     */
    auto validate() const -> common::Result<std::monostate, IntegrationTestError>;
};

/**
 * @brief Builder pattern for creating test scenarios
 */
class TestScenarioBuilder {
  public:
    TestScenarioBuilder();

    auto with_name(const std::string& name) -> TestScenarioBuilder&;
    auto with_backends(const std::vector<engines::InferenceBackend>& backends)
        -> TestScenarioBuilder&;
    auto with_model_config(const engines::ModelConfig& config) -> TestScenarioBuilder&;
    auto with_validation_strategy(ValidationStrategy strategy) -> TestScenarioBuilder&;
    auto with_mode(TestMode mode) -> TestScenarioBuilder&;
    auto with_iterations(std::uint32_t iterations) -> TestScenarioBuilder&;
    auto with_timeout(std::chrono::milliseconds timeout) -> TestScenarioBuilder&;
    auto with_max_latency(std::chrono::milliseconds latency) -> TestScenarioBuilder&;
    auto with_numerical_tolerance(double tolerance) -> TestScenarioBuilder&;
    auto with_memory_tracking(bool enable) -> TestScenarioBuilder&;
    auto with_performance_profiling(bool enable) -> TestScenarioBuilder&;
    auto with_error_injection(bool enable) -> TestScenarioBuilder&;
    auto with_concurrency_level(std::uint32_t level) -> TestScenarioBuilder&;

    auto build() -> common::Result<TestScenario, IntegrationTestError>;

  private:
    TestScenario scenario_;
};

/**
 * @brief Results from integration test execution
 */
struct IntegrationTestResults {
    TestScenario scenario;  ///< Original test scenario
    std::unordered_map<engines::InferenceBackend, PerformanceMetrics> metrics;  ///< Per-backend
                                                                                ///< metrics
    StatisticalAnalysis statistical_analysis;           ///< Statistical validation results
    std::vector<std::string> error_messages;            ///< Any error messages encountered
    std::chrono::milliseconds total_execution_time{0};  ///< Total test execution time
    bool passed{false};                                 ///< Whether test passed overall
    std::string failure_reason;                         ///< Reason for failure (if any)

    /**
     * @brief Generate comprehensive test report
     */
    auto generate_report() const -> std::string;

    /**
     * @brief Check if results meet performance requirements
     */
    auto meets_performance_requirements() const -> bool;

    /**
     * @brief Get performance comparison across backends
     */
    auto get_performance_comparison() const -> std::string;
};

/**
 * @brief Abstract interface for backend factories
 */
class BackendFactory {
  public:
    virtual ~BackendFactory() = default;

    /**
     * @brief Create inference engine for specified backend
     */
    virtual auto create_engine(engines::InferenceBackend backend,
                               const engines::ModelConfig& config)
        -> common::Result<std::unique_ptr<engines::InferenceEngine>, IntegrationTestError> = 0;

    /**
     * @brief Check if backend is available
     */
    virtual auto is_backend_available(engines::InferenceBackend backend) const -> bool = 0;

    /**
     * @brief Get factory type description
     */
    virtual auto get_factory_type() const -> std::string = 0;
};

/**
 * @brief Main ML integration testing framework
 */
class MLIntegrationFramework {
  public:
    /**
     * @brief Constructor with dependency injection
     */
    explicit MLIntegrationFramework(std::unique_ptr<BackendFactory> factory);

    /**
     * @brief Run comprehensive integration test
     */
    auto run_integration_test(const TestScenario& scenario)
        -> common::Result<IntegrationTestResults, IntegrationTestError>;

    /**
     * @brief Run performance benchmark comparison
     */
    auto run_performance_benchmark(const TestScenario& scenario)
        -> common::Result<IntegrationTestResults, IntegrationTestError>;

    /**
     * @brief Run memory safety validation
     */
    auto run_memory_safety_test(const TestScenario& scenario)
        -> common::Result<IntegrationTestResults, IntegrationTestError>;

    /**
     * @brief Run concurrency and thread safety test
     */
    auto run_concurrency_test(const TestScenario& scenario)
        -> common::Result<IntegrationTestResults, IntegrationTestError>;

    /**
     * @brief Run error injection and recovery test
     */
    auto run_error_injection_test(const TestScenario& scenario)
        -> common::Result<IntegrationTestResults, IntegrationTestError>;

    /**
     * @brief Get available backends
     */
    auto get_available_backends() const -> std::vector<engines::InferenceBackend>;

    /**
     * @brief Get framework information
     */
    auto get_framework_info() const -> std::string;

    /**
     * @brief Test single backend functionality (for test compatibility)
     */
    auto test_single_backend(engines::InferenceBackend backend,
                             const engines::ModelConfig& config,
                             const std::vector<engines::InferenceRequest>& inputs)
        -> common::Result<IntegrationTestResults, IntegrationTestError>;

    /**
     * @brief Compare multiple backends (for test compatibility)
     */
    auto compare_backends(const std::vector<engines::InferenceBackend>& backends,
                          const engines::ModelConfig& config,
                          const std::vector<engines::InferenceRequest>& inputs,
                          ValidationStrategy strategy)
        -> common::Result<IntegrationTestResults, IntegrationTestError>;

  private:
    std::unique_ptr<BackendFactory> backend_factory_;

    // Internal execution methods
    auto execute_single_backend_test(const TestScenario& scenario,
                                     engines::InferenceBackend backend)
        -> common::Result<PerformanceMetrics, IntegrationTestError>;

    auto execute_multi_backend_test(const TestScenario& scenario)
        -> common::Result<IntegrationTestResults, IntegrationTestError>;

    auto validate_outputs(const TestScenario& scenario,
                          const std::vector<engines::InferenceResponse>& outputs)
        -> common::Result<StatisticalAnalysis, IntegrationTestError>;

    auto collect_performance_metrics(const std::vector<std::chrono::nanoseconds>& latencies,
                                     std::uint64_t memory_usage,
                                     std::uint32_t error_count) -> PerformanceMetrics;
};

/**
 * @brief Factory function to create framework with hardware backends
 */
auto create_hardware_integration_framework()
    -> common::Result<std::unique_ptr<MLIntegrationFramework>, IntegrationTestError>;

/**
 * @brief Factory function to create framework with mock backends
 */
auto create_mock_integration_framework()
    -> common::Result<std::unique_ptr<MLIntegrationFramework>, IntegrationTestError>;

/**
 * @brief Factory function to create framework with mixed hardware/mock backends
 */
auto create_hybrid_integration_framework(bool use_real_tensorrt = false, bool use_real_onnx = false)
    -> common::Result<std::unique_ptr<MLIntegrationFramework>, IntegrationTestError>;

}  // namespace inference_lab::integration
