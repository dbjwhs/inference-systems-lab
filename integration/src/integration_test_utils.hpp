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
 * @file integration_test_utils.hpp
 * @brief Comprehensive test utilities and fixtures for ML integration testing
 *
 * This header provides a complete suite of testing utilities, fixtures, and
 * helper functions designed to streamline ML integration testing. It includes
 * pre-configured test scenarios, data generation utilities, validation helpers,
 * and specialized fixtures for different testing patterns.
 *
 * Key Features:
 * - Pre-built test fixtures for common ML scenarios
 * - Realistic test data generation with statistical properties
 * - Comprehensive validation utilities for correctness testing
 * - Performance testing helpers with statistical analysis
 * - Memory usage tracking and leak detection utilities
 * - Error injection patterns for robustness testing
 * - Test data serialization for reproducible testing
 *
 * Design Philosophy:
 * - Reduce boilerplate: Common test patterns should be simple to set up
 * - Realistic testing: Test data should reflect real-world characteristics
 * - Comprehensive coverage: Support all common testing scenarios
 * - Statistical rigor: Proper statistical analysis for performance testing
 * - Debugging support: Rich logging and diagnostic information
 *
 * Architecture:
 * @code
 *   ┌─────────────────┐   provides    ┌─────────────────┐
 *   │ Test Fixtures   │──────────────▶│ Common Patterns │
 *   │ • Classification│               │ • Image Recog   │
 *   │ • Object Detect │               │ • NLP Tasks     │
 *   │ • Segmentation  │               │ • Time Series   │
 *   └─────────────────┘               └─────────────────┘
 *            │                                  │
 *            │ configures                       │ generates
 *            ▼                                  ▼
 *   ┌─────────────────┐               ┌─────────────────┐
 *   │ Data Generators │               │ Test Scenarios  │
 *   │ • Synthetic     │               │ • Performance   │
 *   │ • Statistical   │               │ • Correctness   │
 *   │ • Realistic     │               │ • Stress        │
 *   └─────────────────┘               └─────────────────┘
 *            │                                  │
 *            │ feeds into                       │ validates with
 *            ▼                                  ▼
 *   ┌─────────────────┐               ┌─────────────────┐
 *   │ Validation      │◄──────────────│ Statistical     │
 *   │ Utilities       │  analyzes     │ Analysis        │
 *   │ • Correctness   │               │ • Distributions │
 *   │ • Performance   │               │ • Correlations  │
 *   │ • Memory Safety │               │ • Significance  │
 *   └─────────────────┘               └─────────────────┘
 * @endcode
 *
 * Usage Example:
 * @code
 * // Create a classification test fixture
 * auto fixture = ClassificationTestFixture::create()
 *     .with_model("resnet50")
 *     .with_input_shape({1, 3, 224, 224})
 *     .with_num_classes(1000)
 *     .with_test_samples(100)
 *     .build();
 *
 * // Generate realistic test data
 * auto test_data = fixture.generate_test_data()
 *     .with_statistical_properties()
 *     .with_label_distribution(Distribution::UNIFORM)
 *     .build();
 *
 * // Run performance analysis
 * auto perf_analyzer = PerformanceAnalyzer::create()
 *     .with_confidence_level(0.95)
 *     .with_statistical_tests(true)
 *     .build();
 *
 * auto results = perf_analyzer.analyze_backend_performance(
 *     {EngineInferenceBackend::TENSORRT_GPU, EngineInferenceBackend::ONNX_RUNTIME},
 *     test_data);
 * @endcode
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../common/src/logging.hpp"
#include "../../common/src/ml_types.hpp"
#include "../../common/src/result.hpp"
#include "ml_integration_framework.hpp"
#include "mock_engines.hpp"

namespace inference_lab::integration::utils {

// Resolve namespace conflicts by being specific about which types to use
using common::Err;
using common::Ok;
using common::Result;

// For ML integration testing, prefer the engines namespace types for the interface
using EngineInferenceRequest = engines::InferenceRequest;
using EngineInferenceResponse = engines::InferenceResponse;
using EngineModelConfig = engines::ModelConfig;
using EngineInferenceBackend = engines::InferenceBackend;

// For complex ML operations, use the common::ml types
using MLFloatTensor = common::ml::FloatTensor;
using MLModelConfig = common::ml::ModelConfig;
using Shape = common::ml::Shape;
using Precision = common::ml::Precision;
using TensorOutput = common::ml::TensorOutput;
using TensorSpec = common::ml::TensorSpec;

// Define missing types for integration testing
struct PerformanceThresholds {
    float max_latency_ms = 100.0f;
    float min_throughput = 10.0f;
    float max_memory_mb = 1000.0f;
    float max_error_rate = 0.01f;
};

// Simplified test metrics type
using TestPerformanceMetrics = PerformanceMetrics;

// Simplified result type alias
using IntegrationTestResult = IntegrationTestResults;

//=============================================================================
// Statistical Utilities and Analysis
//=============================================================================

/**
 * @brief Statistical distribution types for test data generation
 */
enum class Distribution : std::uint8_t {
    UNIFORM,      ///< Uniform distribution
    NORMAL,       ///< Gaussian/normal distribution
    EXPONENTIAL,  ///< Exponential distribution
    BETA,         ///< Beta distribution
    REALISTIC     ///< Domain-specific realistic distribution
};

/**
 * @brief Statistical properties for data generation
 */
struct StatisticalProperties {
    float mean = 0.0f;           ///< Distribution mean
    float std_deviation = 1.0f;  ///< Standard deviation
    float min_value = -1.0f;     ///< Minimum value
    float max_value = 1.0f;      ///< Maximum value
    Distribution distribution = Distribution::NORMAL;
    std::optional<std::uint32_t> seed;  ///< Random seed for reproducibility
};

/**
 * @brief Statistical analysis results
 */
struct StatisticalAnalysis {
    float mean;                       ///< Sample mean
    float median;                     ///< Sample median
    float std_deviation;              ///< Sample standard deviation
    float min_value;                  ///< Minimum value
    float max_value;                  ///< Maximum value
    float percentile_95;              ///< 95th percentile
    float percentile_99;              ///< 99th percentile
    std::size_t sample_size;          ///< Number of samples
    bool normal_distribution;         ///< Whether data appears normally distributed
    float confidence_interval_95[2];  ///< 95% confidence interval for mean
};

/**
 * @brief Comprehensive statistical analyzer
 */
class StatisticalAnalyzer {
  public:
    /**
     * @brief Analyze numerical data distribution
     */
    static auto analyze_distribution(const std::vector<float>& data) -> StatisticalAnalysis;

    /**
     * @brief Test for statistical significance between two datasets
     */
    static auto test_significance(const std::vector<float>& data1,
                                  const std::vector<float>& data2,
                                  float alpha = 0.05f) -> Result<bool, std::string>;

    /**
     * @brief Calculate correlation between two datasets
     */
    static auto calculate_correlation(const std::vector<float>& data1,
                                      const std::vector<float>& data2) -> float;

    /**
     * @brief Test for normality using Shapiro-Wilk test
     */
    static auto test_normality(const std::vector<float>& data) -> bool;

    /**
     * @brief Calculate confidence interval for mean
     */
    static auto confidence_interval(const std::vector<float>& data, float confidence_level = 0.95f)
        -> std::array<float, 2>;

  private:
    static auto calculate_percentile(const std::vector<float>& sorted_data, float percentile)
        -> float;
};

//=============================================================================
// Test Data Generation
//=============================================================================

/**
 * @brief Advanced test data generator with statistical control
 */
class TestDataGenerator {
  public:
    explicit TestDataGenerator(std::uint32_t seed = 42);

    /**
     * @brief Generate tensor with specified statistical properties
     */
    auto generate_tensor(const Shape& shape, const StatisticalProperties& properties = {})
        -> MLFloatTensor;

    /**
     * @brief Generate classification test data
     */
    auto generate_classification_data(const Shape& input_shape,
                                      std::uint32_t num_classes,
                                      std::uint32_t num_samples)
        -> std::vector<EngineInferenceRequest>;

    /**
     * @brief Generate object detection test data
     */
    auto generate_object_detection_data(const Shape& input_shape,
                                        std::uint32_t max_objects,
                                        std::uint32_t num_samples)
        -> std::vector<EngineInferenceRequest>;

    /**
     * @brief Generate time series test data
     */
    auto generate_time_series_data(std::uint32_t sequence_length,
                                   std::uint32_t num_features,
                                   std::uint32_t num_samples)
        -> std::vector<EngineInferenceRequest>;

    /**
     * @brief Generate NLP test data (token sequences)
     */
    auto generate_nlp_data(std::uint32_t sequence_length,
                           std::uint32_t vocab_size,
                           std::uint32_t num_samples) -> std::vector<EngineInferenceRequest>;

    /**
     * @brief Generate realistic image data with natural statistics
     */
    auto generate_realistic_image_data(const Shape& image_shape, std::uint32_t num_samples)
        -> std::vector<EngineInferenceRequest>;

    /**
     * @brief Generate adversarial examples for robustness testing
     */
    auto generate_adversarial_examples(const EngineInferenceRequest& base_input,
                                       float epsilon = 0.1f,
                                       std::uint32_t num_examples = 10)
        -> std::vector<EngineInferenceRequest>;

    /**
     * @brief Generate edge case inputs for stress testing
     */
    auto generate_edge_cases(const EngineModelConfig& config)
        -> std::vector<EngineInferenceRequest>;

  private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;
    std::normal_distribution<float> normal_dist_;

    auto generate_realistic_image_patch(std::uint32_t height, std::uint32_t width)
        -> std::vector<float>;
    auto add_adversarial_noise(const MLFloatTensor& input, float epsilon) -> MLFloatTensor;
};

//=============================================================================
// Test Fixtures for Common Scenarios
//=============================================================================

/**
 * @brief Base class for all test fixtures
 */
class TestFixture {
  public:
    virtual ~TestFixture() = default;

    /**
     * @brief Get model configuration for this fixture
     */
    virtual auto get_model_config() const -> EngineModelConfig = 0;

    /**
     * @brief Generate test data for this fixture
     */
    virtual auto generate_test_data(std::uint32_t num_samples = 10)
        -> std::vector<EngineInferenceRequest> = 0;

    /**
     * @brief Get expected performance characteristics
     */
    virtual auto get_performance_expectations() const -> PerformanceThresholds = 0;

    /**
     * @brief Validate output correctness
     */
    virtual auto validate_output(const EngineInferenceResponse& response) const
        -> Result<bool, std::string> = 0;

  protected:
    TestFixture() = default;
};

/**
 * @brief Image classification test fixture
 */
class ClassificationTestFixture : public TestFixture {
  public:
    struct Config {
        std::string model_name = "resnet50";
        Shape input_shape = {1, 3, 224, 224};
        std::uint32_t num_classes = 1000;
        Precision precision = Precision::FP32;
        EngineInferenceBackend preferred_backend = EngineInferenceBackend::TENSORRT_GPU;
        StatisticalProperties data_properties;
    };

    explicit ClassificationTestFixture(Config config);

    auto get_model_config() const -> EngineModelConfig override;
    auto generate_test_data(std::uint32_t num_samples = 10)
        -> std::vector<EngineInferenceRequest> override;
    auto get_performance_expectations() const -> PerformanceThresholds override;
    auto validate_output(const EngineInferenceResponse& response) const
        -> Result<bool, std::string> override;

    /**
     * @brief Create builder for easy configuration
     */
    static auto create() -> ClassificationTestFixture;

    /**
     * @brief Builder pattern methods
     */
    auto with_model(const std::string& name) -> ClassificationTestFixture&;
    auto with_input_shape(const Shape& shape) -> ClassificationTestFixture&;
    auto with_num_classes(std::uint32_t classes) -> ClassificationTestFixture&;
    auto with_precision(Precision prec) -> ClassificationTestFixture&;

  private:
    Config config_;
    TestDataGenerator data_generator_;
};

/**
 * @brief Object detection test fixture
 */
class ObjectDetectionTestFixture : public TestFixture {
  public:
    struct Config {
        std::string model_name = "yolov5";
        Shape input_shape = {1, 3, 640, 640};
        std::uint32_t num_classes = 80;
        std::uint32_t max_detections = 100;
        float confidence_threshold = 0.5f;
        float nms_threshold = 0.4f;
    };

    explicit ObjectDetectionTestFixture(Config config);

    auto get_model_config() const -> EngineModelConfig override;
    auto generate_test_data(std::uint32_t num_samples = 10)
        -> std::vector<EngineInferenceRequest> override;
    auto get_performance_expectations() const -> PerformanceThresholds override;
    auto validate_output(const EngineInferenceResponse& response) const
        -> Result<bool, std::string> override;

    static auto create() -> ObjectDetectionTestFixture;

  private:
    Config config_;
    TestDataGenerator data_generator_;
};

/**
 * @brief NLP (Natural Language Processing) test fixture
 */
class NLPTestFixture : public TestFixture {
  public:
    struct Config {
        std::string model_name = "bert_base";
        std::uint32_t max_sequence_length = 512;
        std::uint32_t vocab_size = 30522;
        std::string task_type = "classification";  // classification, ner, qa
        std::uint32_t num_labels = 2;
    };

    explicit NLPTestFixture(Config config);

    auto get_model_config() const -> EngineModelConfig override;
    auto generate_test_data(std::uint32_t num_samples = 10)
        -> std::vector<EngineInferenceRequest> override;
    auto get_performance_expectations() const -> PerformanceThresholds override;
    auto validate_output(const EngineInferenceResponse& response) const
        -> Result<bool, std::string> override;

    static auto create() -> NLPTestFixture;

  private:
    Config config_;
    TestDataGenerator data_generator_;
};

//=============================================================================
// Performance Analysis and Benchmarking
//=============================================================================

/**
 * @brief Comprehensive performance analysis framework
 */
class PerformanceAnalyzer {
  public:
    struct Config {
        float confidence_level = 0.95f;        ///< Statistical confidence level
        std::uint32_t min_iterations = 10;     ///< Minimum benchmark iterations
        std::uint32_t max_iterations = 1000;   ///< Maximum benchmark iterations
        bool enable_statistical_tests = true;  ///< Enable significance testing
        bool enable_outlier_detection = true;  ///< Remove statistical outliers
        float outlier_threshold = 2.0f;        ///< Standard deviations for outlier
    };

    explicit PerformanceAnalyzer(Config config);
    PerformanceAnalyzer() : PerformanceAnalyzer(Config{}) {}

    /**
     * @brief Benchmark single backend performance
     */
    auto benchmark_backend(engines::InferenceEngine* engine,
                           const std::vector<EngineInferenceRequest>& inputs)
        -> Result<TestPerformanceMetrics, std::string>;

    /**
     * @brief Compare performance between backends
     */
    auto compare_backend_performance(const std::vector<EngineInferenceBackend>& backends,
                                     const EngineModelConfig& model_config,
                                     const std::vector<EngineInferenceRequest>& inputs,
                                     BackendFactory* factory)
        -> Result<std::unordered_map<EngineInferenceBackend, TestPerformanceMetrics>, std::string>;

    /**
     * @brief Analyze latency distribution
     */
    auto analyze_latency_distribution(const std::vector<std::chrono::milliseconds>& latencies)
        -> StatisticalAnalysis;

    /**
     * @brief Detect performance regressions
     */
    auto detect_regression(const TestPerformanceMetrics& baseline,
                           const TestPerformanceMetrics& current,
                           float threshold = 0.05f) -> Result<bool, std::string>;

    /**
     * @brief Generate performance report
     */
    auto generate_performance_report(
        const std::unordered_map<EngineInferenceBackend, TestPerformanceMetrics>& results)
        -> std::string;

  private:
    Config config_;

    auto remove_outliers(std::vector<float>& data) -> void;
    auto calculate_required_iterations(const std::vector<float>& initial_samples) -> std::uint32_t;
};

//=============================================================================
// Memory Testing and Leak Detection
//=============================================================================

/**
 * @brief Memory usage tracker for integration testing
 */
class MemoryTracker {
  public:
    struct MemorySnapshot {
        std::uint64_t allocated_bytes;     ///< Currently allocated memory
        std::uint64_t peak_usage_bytes;    ///< Peak memory usage
        std::uint32_t allocation_count;    ///< Number of allocations
        std::uint32_t deallocation_count;  ///< Number of deallocations
        std::chrono::steady_clock::time_point timestamp;
    };

    /**
     * @brief Start memory tracking
     */
    void start_tracking();

    /**
     * @brief Stop memory tracking and return results
     */
    auto stop_tracking() -> MemorySnapshot;

    /**
     * @brief Take memory snapshot
     */
    auto take_snapshot() -> MemorySnapshot;

    /**
     * @brief Check for memory leaks
     */
    auto check_for_leaks(const MemorySnapshot& before, const MemorySnapshot& after) -> bool;

    /**
     * @brief Test memory stress scenario
     */
    auto test_memory_stress(engines::InferenceEngine* engine,
                            const EngineInferenceRequest& request,
                            std::uint32_t stress_iterations = 1000) -> Result<bool, std::string>;

    /**
     * @brief Simulate memory pressure
     */
    auto simulate_memory_pressure(std::uint64_t pressure_mb = 1024) -> Result<bool, std::string>;

  private:
    std::optional<MemorySnapshot> start_snapshot_;
    std::vector<MemorySnapshot> snapshots_;

    auto get_current_memory_usage() -> std::uint64_t;
};

//=============================================================================
// Validation Utilities
//=============================================================================

/**
 * @brief Comprehensive output validation framework
 */
class OutputValidator {
  public:
    /**
     * @brief Validate classification output
     */
    static auto validate_classification_output(const EngineInferenceResponse& response,
                                               std::uint32_t expected_classes)
        -> Result<bool, std::string>;

    /**
     * @brief Validate object detection output
     */
    static auto validate_object_detection_output(const EngineInferenceResponse& response,
                                                 std::uint32_t max_detections)
        -> Result<bool, std::string>;

    /**
     * @brief Validate tensor shapes and types
     */
    static auto validate_tensor_specs(const EngineInferenceResponse& response,
                                      const std::vector<TensorSpec>& expected_specs)
        -> Result<bool, std::string>;

    /**
     * @brief Validate numerical stability
     */
    static auto validate_numerical_stability(const EngineInferenceResponse& response)
        -> Result<bool, std::string>;

    /**
     * @brief Compare outputs for consistency
     */
    static auto compare_outputs(const EngineInferenceResponse& response1,
                                const EngineInferenceResponse& response2,
                                float tolerance = 1e-5f)
        -> Result<float, std::string>;  // Returns similarity score

    /**
     * @brief Validate confidence scores
     */
    static auto validate_confidence_scores(const EngineInferenceResponse& response)
        -> Result<bool, std::string>;

  private:
    static auto check_tensor_bounds(const TensorOutput& output) -> bool;
    static auto calculate_tensor_similarity(const TensorOutput& tensor1,
                                            const TensorOutput& tensor2,
                                            float tolerance) -> float;
};

//=============================================================================
// Test Scenario Builders
//=============================================================================

/**
 * @brief Builder for creating comprehensive test scenarios
 */
class TestScenarioBuilder {
  public:
    TestScenarioBuilder() = default;

    /**
     * @brief Set scenario name and description
     */
    auto with_name(const std::string& name) -> TestScenarioBuilder&;
    auto with_description(const std::string& description) -> TestScenarioBuilder&;

    /**
     * @brief Configure backends to test
     */
    auto with_backends(const std::vector<EngineInferenceBackend>& backends) -> TestScenarioBuilder&;
    auto with_single_backend(EngineInferenceBackend backend) -> TestScenarioBuilder&;

    /**
     * @brief Configure model and data
     */
    auto with_model_config(const EngineModelConfig& config) -> TestScenarioBuilder&;
    auto with_test_fixture(std::shared_ptr<TestFixture> fixture) -> TestScenarioBuilder&;
    auto with_test_data(const std::vector<EngineInferenceRequest>& data) -> TestScenarioBuilder&;

    /**
     * @brief Configure performance requirements
     */
    auto with_performance_thresholds(const PerformanceThresholds& thresholds)
        -> TestScenarioBuilder&;
    auto with_max_latency(std::chrono::milliseconds latency) -> TestScenarioBuilder&;
    auto with_min_throughput(float fps) -> TestScenarioBuilder&;

    /**
     * @brief Configure validation strategy
     */
    auto with_validation_strategy(ValidationStrategy strategy) -> TestScenarioBuilder&;
    auto with_tolerance(float tolerance) -> TestScenarioBuilder&;

    /**
     * @brief Configure test execution
     */
    auto with_iterations(std::uint32_t iterations) -> TestScenarioBuilder&;
    auto with_warmup_iterations(std::uint32_t warmup) -> TestScenarioBuilder&;
    auto with_concurrent_execution(std::uint32_t max_concurrent) -> TestScenarioBuilder&;

    /**
     * @brief Build final test scenario
     */
    auto build() -> Result<TestScenario, std::string>;

    /**
     * @brief Quick builders for common scenarios
     */
    static auto create_performance_test(const std::string& name,
                                        const std::vector<EngineInferenceBackend>& backends,
                                        std::shared_ptr<TestFixture> fixture)
        -> TestScenarioBuilder;

    static auto create_correctness_test(const std::string& name,
                                        EngineInferenceBackend backend,
                                        std::shared_ptr<TestFixture> fixture)
        -> TestScenarioBuilder;

    static auto create_stress_test(const std::string& name,
                                   EngineInferenceBackend backend,
                                   std::shared_ptr<TestFixture> fixture) -> TestScenarioBuilder;

  private:
    TestScenario scenario_;
    std::shared_ptr<TestFixture> fixture_;
};

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Load test configuration from JSON file
 */
auto load_test_config_from_file(const std::string& filename)
    -> Result<std::vector<TestScenario>, std::string>;

/**
 * @brief Save test results to file
 */
auto save_test_results_to_file(const std::string& filename,
                               const std::vector<IntegrationTestResult>& results)
    -> Result<bool, std::string>;

/**
 * @brief Generate test report in HTML format
 */
auto generate_html_report(const std::vector<IntegrationTestResult>& results) -> std::string;

/**
 * @brief Compare test results with baseline
 */
auto compare_with_baseline(const IntegrationTestResult& current,
                           const IntegrationTestResult& baseline,
                           float tolerance = 0.05f) -> Result<bool, std::string>;

/**
 * @brief Create mock integration framework for testing
 */
auto create_mock_integration_framework() -> std::unique_ptr<MLIntegrationFramework>;

/**
 * @brief Set up test environment with logging and configuration
 */
auto setup_test_environment(const std::string& log_level = "INFO") -> void;

/**
 * @brief Clean up test environment and temporary files
 */
auto cleanup_test_environment() -> void;

}  // namespace inference_lab::integration::utils
