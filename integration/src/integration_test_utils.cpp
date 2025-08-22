// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file integration_test_utils_minimal.cpp
 * @brief Minimal implementation to ensure clean compilation
 */

#include "integration_test_utils.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "../../common/src/logging.hpp"

namespace inference_lab::integration::utils {

using common::LogLevel;

// Minimal implementations to satisfy the interface without complex dependencies

//=============================================================================
// TestDataGenerator Implementation
//=============================================================================

TestDataGenerator::TestDataGenerator(std::uint32_t seed)
    : rng_(seed), uniform_dist_(0.0f, 1.0f), normal_dist_(0.0f, 1.0f) {}

auto TestDataGenerator::generate_classification_data(const Shape& input_shape,
                                                     std::uint32_t num_classes,
                                                     std::uint32_t num_samples)
    -> std::vector<EngineInferenceRequest> {
    std::vector<EngineInferenceRequest> requests;
    requests.reserve(num_samples);

    for (std::uint32_t i = 0; i < num_samples; ++i) {
        EngineInferenceRequest request;
        request.batch_size = 1;
        request.input_tensors = {{1.0f, 2.0f, 3.0f}};  // Simple mock data
        request.input_names = {"input"};
        requests.push_back(std::move(request));
    }

    return requests;
}

auto TestDataGenerator::generate_tensor(const Shape& shape, const StatisticalProperties& properties)
    -> MLFloatTensor {
    // Return a basic tensor - actual implementation would be more sophisticated
    return MLFloatTensor{};
}

auto TestDataGenerator::generate_object_detection_data(const Shape& input_shape,
                                                       std::uint32_t max_objects,
                                                       std::uint32_t num_samples)
    -> std::vector<EngineInferenceRequest> {
    std::vector<EngineInferenceRequest> requests;
    requests.reserve(num_samples);

    for (std::uint32_t i = 0; i < num_samples; ++i) {
        EngineInferenceRequest request;
        request.batch_size = 1;

        // Generate mock image data (simplified)
        std::vector<float> image_data;
        std::size_t total_elements = 1;
        for (auto dim : input_shape) {
            total_elements *= dim;
        }
        image_data.resize(total_elements, 0.5f);  // Mock normalized image data

        request.input_tensors = {std::move(image_data)};
        request.input_names = {"images"};
        requests.push_back(std::move(request));
    }

    return requests;
}

auto TestDataGenerator::generate_nlp_data(std::uint32_t sequence_length,
                                          std::uint32_t vocab_size,
                                          std::uint32_t num_samples)
    -> std::vector<EngineInferenceRequest> {
    std::vector<EngineInferenceRequest> requests;
    requests.reserve(num_samples);

    for (std::uint32_t i = 0; i < num_samples; ++i) {
        EngineInferenceRequest request;
        request.batch_size = 1;

        // Generate mock token sequence
        std::vector<float> token_ids;
        token_ids.reserve(sequence_length);

        for (std::uint32_t j = 0; j < sequence_length; ++j) {
            // Generate random token ID within vocab size
            token_ids.push_back(static_cast<float>(j % vocab_size));
        }

        request.input_tensors = {std::move(token_ids)};
        request.input_names = {"input_ids"};
        requests.push_back(std::move(request));
    }

    return requests;
}

//=============================================================================
// TestFixture Implementations
//=============================================================================

// ClassificationTestFixture Implementation
ClassificationTestFixture::ClassificationTestFixture(Config config) : config_(std::move(config)) {}

auto ClassificationTestFixture::create() -> ClassificationTestFixture {
    return ClassificationTestFixture{Config{}};
}

auto ClassificationTestFixture::with_model(const std::string& name) -> ClassificationTestFixture& {
    config_.model_name = name;
    return *this;
}

auto ClassificationTestFixture::with_input_shape(const Shape& shape) -> ClassificationTestFixture& {
    config_.input_shape = shape;
    return *this;
}

auto ClassificationTestFixture::with_num_classes(std::uint32_t classes)
    -> ClassificationTestFixture& {
    config_.num_classes = classes;
    return *this;
}

auto ClassificationTestFixture::with_precision(Precision prec) -> ClassificationTestFixture& {
    config_.precision = prec;
    return *this;
}

auto ClassificationTestFixture::get_model_config() const -> EngineModelConfig {
    EngineModelConfig model_config;
    model_config.model_path = config_.model_name + ".onnx";  // Convert name to path
    model_config.max_batch_size = 1;
    return model_config;
}

auto ClassificationTestFixture::generate_test_data(std::uint32_t num_samples)
    -> std::vector<EngineInferenceRequest> {
    return data_generator_.generate_classification_data(
        config_.input_shape, config_.num_classes, num_samples);
}

auto ClassificationTestFixture::get_performance_expectations() const -> PerformanceThresholds {
    PerformanceThresholds thresholds;
    thresholds.max_latency_ms = 50.0f;   // 50ms for classification
    thresholds.min_throughput = 20.0f;   // 20 requests per second
    thresholds.max_memory_mb = 2048.0f;  // 2GB memory
    return thresholds;
}

auto ClassificationTestFixture::validate_output(const EngineInferenceResponse& response) const
    -> Result<bool, std::string> {
    if (response.output_tensors.empty()) {
        return Err(std::string("No output tensors in response"));
    }

    if (response.output_tensors[0].size() != config_.num_classes) {
        return Err(std::string("Output tensor size mismatch: expected ") +
                   std::to_string(config_.num_classes) + ", got " +
                   std::to_string(response.output_tensors[0].size()));
    }

    return Ok(true);
}

// ObjectDetectionTestFixture Implementation
ObjectDetectionTestFixture::ObjectDetectionTestFixture(Config config)
    : config_(std::move(config)) {}

auto ObjectDetectionTestFixture::create() -> ObjectDetectionTestFixture {
    return ObjectDetectionTestFixture{Config{}};
}

auto ObjectDetectionTestFixture::get_model_config() const -> EngineModelConfig {
    EngineModelConfig model_config;
    model_config.model_path = config_.model_name + ".onnx";  // Convert name to path
    model_config.max_batch_size = 1;
    return model_config;
}

auto ObjectDetectionTestFixture::generate_test_data(std::uint32_t num_samples)
    -> std::vector<EngineInferenceRequest> {
    return data_generator_.generate_object_detection_data(
        config_.input_shape, config_.max_detections, num_samples);
}

auto ObjectDetectionTestFixture::get_performance_expectations() const -> PerformanceThresholds {
    PerformanceThresholds thresholds;
    thresholds.max_latency_ms = 100.0f;  // 100ms for object detection
    thresholds.min_throughput = 10.0f;   // 10 requests per second
    thresholds.max_memory_mb = 4096.0f;  // 4GB memory
    return thresholds;
}

auto ObjectDetectionTestFixture::validate_output(const EngineInferenceResponse& response) const
    -> Result<bool, std::string> {
    if (response.output_tensors.empty()) {
        return Err(std::string("No output tensors in response"));
    }

    // Basic validation - object detection should have detections output
    return Ok(true);
}

// NLPTestFixture Implementation
NLPTestFixture::NLPTestFixture(Config config) : config_(std::move(config)) {}

auto NLPTestFixture::create() -> NLPTestFixture {
    return NLPTestFixture{Config{}};
}

auto NLPTestFixture::get_model_config() const -> EngineModelConfig {
    EngineModelConfig model_config;
    model_config.model_path = config_.model_name + ".onnx";  // Convert name to path
    model_config.max_batch_size = 1;
    return model_config;
}

auto NLPTestFixture::generate_test_data(std::uint32_t num_samples)
    -> std::vector<EngineInferenceRequest> {
    return data_generator_.generate_nlp_data(
        config_.max_sequence_length, config_.vocab_size, num_samples);
}

auto NLPTestFixture::get_performance_expectations() const -> PerformanceThresholds {
    PerformanceThresholds thresholds;
    thresholds.max_latency_ms = 200.0f;  // 200ms for NLP
    thresholds.min_throughput = 5.0f;    // 5 requests per second
    thresholds.max_memory_mb = 8192.0f;  // 8GB memory for large language models
    return thresholds;
}

auto NLPTestFixture::validate_output(const EngineInferenceResponse& response) const
    -> Result<bool, std::string> {
    if (response.output_tensors.empty()) {
        return Err(std::string("No output tensors in response"));
    }

    // Basic validation for NLP - check if we have expected number of labels
    if (config_.task_type == "classification" &&
        response.output_tensors[0].size() != config_.num_labels) {
        return Err(std::string("Classification output size mismatch"));
    }

    return Ok(true);
}

//=============================================================================
// PerformanceAnalyzer Implementation
//=============================================================================

PerformanceAnalyzer::PerformanceAnalyzer(Config config) : config_(std::move(config)) {}

auto PerformanceAnalyzer::benchmark_backend(engines::InferenceEngine* engine,
                                            const std::vector<EngineInferenceRequest>& inputs)
    -> Result<TestPerformanceMetrics, std::string> {
    if (!engine) {
        return Err(std::string("Null engine provided"));
    }

    if (inputs.empty()) {
        return Err(std::string("No input data provided"));
    }

    TestPerformanceMetrics metrics;
    std::vector<float> latencies;
    latencies.reserve(config_.max_iterations);

    // Run benchmark iterations
    std::uint32_t iterations =
        std::min(config_.max_iterations,
                 std::max(config_.min_iterations, static_cast<std::uint32_t>(inputs.size())));

    for (std::uint32_t i = 0; i < iterations; ++i) {
        const auto& input = inputs[i % inputs.size()];

        auto start = std::chrono::high_resolution_clock::now();
        auto result = engine->run_inference(input);
        auto end = std::chrono::high_resolution_clock::now();

        if (result.is_err()) {
            // Count errors but continue benchmarking
            metrics.error_count++;
            continue;
        }

        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        latencies.push_back(static_cast<float>(latency_ns.count()));
    }

    if (latencies.empty()) {
        return Err(std::string("All benchmark iterations failed"));
    }

    // Remove outliers if enabled
    if (config_.enable_outlier_detection) {
        remove_outliers(latencies);
    }

    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());

    metrics.min_latency = std::chrono::nanoseconds(static_cast<std::uint64_t>(latencies.front()));
    metrics.max_latency = std::chrono::nanoseconds(static_cast<std::uint64_t>(latencies.back()));

    float sum = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
    metrics.mean_latency =
        std::chrono::nanoseconds(static_cast<std::uint64_t>(sum / latencies.size()));

    // Percentiles
    auto p95_idx = static_cast<std::size_t>(latencies.size() * 0.95);
    auto p99_idx = static_cast<std::size_t>(latencies.size() * 0.99);
    metrics.p95_latency = std::chrono::nanoseconds(static_cast<std::uint64_t>(latencies[p95_idx]));
    metrics.p99_latency = std::chrono::nanoseconds(static_cast<std::uint64_t>(latencies[p99_idx]));

    // Throughput (requests per second)
    float avg_latency_seconds = static_cast<float>(metrics.mean_latency.count()) / 1e9f;
    metrics.throughput_inferences_per_sec = 1.0f / avg_latency_seconds;

    // Memory usage (simplified)
    metrics.peak_memory_usage_bytes = 1024 * 1024 * 100;  // 100MB estimate

    return Ok(std::move(metrics));
}

auto PerformanceAnalyzer::compare_backend_performance(
    const std::vector<EngineInferenceBackend>& backends,
    const EngineModelConfig& model_config,
    const std::vector<EngineInferenceRequest>& inputs,
    BackendFactory* factory)
    -> Result<std::unordered_map<EngineInferenceBackend, TestPerformanceMetrics>, std::string> {
    if (!factory) {
        return Err(std::string("Null factory provided"));
    }

    if (backends.empty()) {
        return Err(std::string("No backends provided"));
    }

    std::unordered_map<EngineInferenceBackend, TestPerformanceMetrics> results;

    for (const auto& backend : backends) {
        // Create engine for this backend
        auto engine_result = factory->create_engine(backend, model_config);
        if (engine_result.is_err()) {
            // Skip this backend but continue with others
            continue;
        }

        auto engine = std::move(engine_result).unwrap();

        // Benchmark this backend
        auto metrics_result = benchmark_backend(engine.get(), inputs);
        if (metrics_result.is_ok()) {
            results[backend] = metrics_result.unwrap();
        }
    }

    if (results.empty()) {
        return Err(std::string("No backends could be benchmarked successfully"));
    }

    return Ok(std::move(results));
}

auto PerformanceAnalyzer::detect_regression(const TestPerformanceMetrics& baseline,
                                            const TestPerformanceMetrics& current,
                                            float threshold) -> Result<bool, std::string> {
    // Compare mean latency - regression if current is significantly slower
    float baseline_latency_ms = static_cast<float>(baseline.mean_latency.count()) / 1e6f;
    float current_latency_ms = static_cast<float>(current.mean_latency.count()) / 1e6f;

    float latency_change = (current_latency_ms - baseline_latency_ms) / baseline_latency_ms;

    // Compare throughput - regression if current is significantly lower
    float throughput_change =
        (current.throughput_inferences_per_sec - baseline.throughput_inferences_per_sec) /
        baseline.throughput_inferences_per_sec;

    // Regression detected if latency increased or throughput decreased beyond threshold
    bool regression = (latency_change > threshold) || (throughput_change < -threshold);

    return Ok(regression);
}

auto PerformanceAnalyzer::generate_performance_report(
    const std::unordered_map<EngineInferenceBackend, TestPerformanceMetrics>& results)
    -> std::string {
    std::stringstream report;
    report << "Performance Analysis Report\n";
    report << "===========================\n\n";

    for (const auto& [backend, metrics] : results) {
        report << "Backend: " << static_cast<int>(backend) << "\n";
        report << "  Mean Latency: " << (metrics.mean_latency.count() / 1e6f) << " ms\n";
        report << "  P95 Latency: " << (metrics.p95_latency.count() / 1e6f) << " ms\n";
        report << "  P99 Latency: " << (metrics.p99_latency.count() / 1e6f) << " ms\n";
        report << "  Throughput: " << metrics.throughput_inferences_per_sec << " req/sec\n";
        report << "  Memory Usage: " << (metrics.peak_memory_usage_bytes / (1024 * 1024))
               << " MB\n";
        report << "  Error Count: " << metrics.error_count << "\n\n";
    }

    return report.str();
}

void PerformanceAnalyzer::remove_outliers(std::vector<float>& data) {
    if (data.size() < 3)
        return;  // Need at least 3 points for outlier detection

    // Calculate mean and standard deviation
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    float mean = sum / data.size();

    float sq_sum = std::accumulate(data.begin(), data.end(), 0.0f, [mean](float acc, float val) {
        return acc + (val - mean) * (val - mean);
    });
    float std_dev = std::sqrt(sq_sum / data.size());

    // Remove outliers beyond threshold * standard deviations
    float lower_bound = mean - config_.outlier_threshold * std_dev;
    float upper_bound = mean + config_.outlier_threshold * std_dev;

    data.erase(std::remove_if(data.begin(),
                              data.end(),
                              [lower_bound, upper_bound](float val) {
                                  return val < lower_bound || val > upper_bound;
                              }),
               data.end());
}

auto PerformanceAnalyzer::calculate_required_iterations(const std::vector<float>& initial_samples)
    -> std::uint32_t {
    // Simple heuristic - use more iterations if there's high variance
    if (initial_samples.size() < 2) {
        return config_.min_iterations;
    }

    float sum = std::accumulate(initial_samples.begin(), initial_samples.end(), 0.0f);
    float mean = sum / initial_samples.size();

    float variance = std::accumulate(initial_samples.begin(),
                                     initial_samples.end(),
                                     0.0f,
                                     [mean](float acc, float val) {
                                         return acc + (val - mean) * (val - mean);
                                     }) /
                     initial_samples.size();

    float coefficient_of_variation = std::sqrt(variance) / mean;

    // Higher CV means more iterations needed
    std::uint32_t required =
        static_cast<std::uint32_t>(config_.min_iterations * (1.0f + coefficient_of_variation));

    return std::min(required, config_.max_iterations);
}

//=============================================================================
// TestScenarioBuilder Implementation
//=============================================================================

auto TestScenarioBuilder::with_name(const std::string& name) -> TestScenarioBuilder& {
    scenario_.name = name;
    return *this;
}

auto TestScenarioBuilder::with_description(const std::string& description) -> TestScenarioBuilder& {
    // Note: TestScenario doesn't have description field, skip for now
    return *this;
}

auto TestScenarioBuilder::with_backends(const std::vector<EngineInferenceBackend>& backends)
    -> TestScenarioBuilder& {
    scenario_.backends = backends;
    return *this;
}

auto TestScenarioBuilder::with_single_backend(EngineInferenceBackend backend)
    -> TestScenarioBuilder& {
    scenario_.backends = {backend};
    return *this;
}

auto TestScenarioBuilder::with_model_config(const EngineModelConfig& config)
    -> TestScenarioBuilder& {
    scenario_.model_config = config;
    return *this;
}

auto TestScenarioBuilder::with_test_fixture(std::shared_ptr<TestFixture> fixture)
    -> TestScenarioBuilder& {
    fixture_ = std::move(fixture);
    return *this;
}

auto TestScenarioBuilder::build() -> Result<TestScenario, std::string> {
    // Validate required fields
    if (scenario_.name.empty()) {
        return Err(std::string("Scenario name is required"));
    }

    if (scenario_.backends.empty()) {
        return Err(std::string("At least one backend is required"));
    }

    // If we have a test fixture, get model config from it
    if (fixture_) {
        scenario_.model_config = fixture_->get_model_config();
    }

    return Ok(scenario_);
}

auto TestScenarioBuilder::create_performance_test(
    const std::string& name,
    const std::vector<EngineInferenceBackend>& backends,
    std::shared_ptr<TestFixture> fixture) -> TestScenarioBuilder {
    TestScenarioBuilder builder;
    builder.scenario_.name = name;
    builder.scenario_.backends = backends;
    builder.scenario_.mode = TestMode::MULTI_BACKEND;
    builder.scenario_.validation_strategy = ValidationStrategy::STATISTICAL_COMPARISON;
    builder.scenario_.iterations = 50;  // More iterations for performance testing
    builder.fixture_ = std::move(fixture);

    return builder;
}

auto TestScenarioBuilder::create_correctness_test(const std::string& name,
                                                  EngineInferenceBackend backend,
                                                  std::shared_ptr<TestFixture> fixture)
    -> TestScenarioBuilder {
    TestScenarioBuilder builder;
    builder.scenario_.name = name;
    builder.scenario_.backends = {backend};
    builder.scenario_.mode = TestMode::SINGLE_BACKEND;
    builder.scenario_.validation_strategy = ValidationStrategy::EXACT_MATCH;
    builder.scenario_.iterations = 10;  // Fewer iterations for correctness testing
    builder.fixture_ = std::move(fixture);

    return builder;
}

//=============================================================================
// Test Environment Functions
//=============================================================================

auto setup_test_environment(const std::string& log_level) -> void {
    // Set up logging for test environment
    LOG_INFO_PRINT("Setting up ML integration test environment with log level: {}", log_level);

    // Initialize any global test state here
    // For now, just logging setup is sufficient
}

auto cleanup_test_environment() -> void {
    LOG_INFO_PRINT("Cleaning up ML integration test environment");

    // Clean up any global test state here
    // For now, just a cleanup message is sufficient
}

}  // namespace inference_lab::integration::utils
