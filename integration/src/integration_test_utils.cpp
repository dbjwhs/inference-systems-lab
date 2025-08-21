// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file integration_test_utils.cpp
 * @brief Implementation of integration test utilities and fixtures
 *
 * This file provides the implementation of comprehensive test utilities,
 * fixtures, and helper functions for ML integration testing. The implementation
 * focuses on providing realistic test data generation, statistical analysis,
 * and validation utilities.
 */

#include "integration_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>

#include "../../common/src/logging.hpp"

namespace inference_lab::integration::utils {

//=============================================================================
// StatisticalAnalyzer Implementation
//=============================================================================

auto StatisticalAnalyzer::analyze_distribution(const std::vector<float>& data)
    -> StatisticalAnalysis {
    StatisticalAnalysis analysis{};

    if (data.empty()) {
        return analysis;
    }

    analysis.sample_size = data.size();

    // Calculate basic statistics
    auto sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    analysis.min_value = sorted_data.front();
    analysis.max_value = sorted_data.back();
    analysis.median = calculate_percentile(sorted_data, 0.5f);
    analysis.percentile_95 = calculate_percentile(sorted_data, 0.95f);
    analysis.percentile_99 = calculate_percentile(sorted_data, 0.99f);

    // Calculate mean
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    analysis.mean = static_cast<float>(sum / data.size());

    // Calculate standard deviation
    double variance_sum = 0.0;
    for (float value : data) {
        double diff = value - analysis.mean;
        variance_sum += diff * diff;
    }
    analysis.std_deviation = static_cast<float>(std::sqrt(variance_sum / data.size()));

    // Test for normality (simplified Shapiro-Wilk)
    analysis.normal_distribution = test_normality(data);

    // Calculate confidence interval for mean
    auto ci = confidence_interval(data, 0.95f);
    analysis.confidence_interval_95[0] = ci[0];
    analysis.confidence_interval_95[1] = ci[1];

    return analysis;
}

auto StatisticalAnalyzer::test_significance(const std::vector<float>& data1,
                                            const std::vector<float>& data2,
                                            float alpha) -> Result<bool, std::string> {
    if (data1.size() < 2 || data2.size() < 2) {
        return Err(std::string("Insufficient data for significance test"));
    }

    // Simplified t-test
    auto analysis1 = analyze_distribution(data1);
    auto analysis2 = analyze_distribution(data2);

    // Calculate pooled standard error
    float se1 = analysis1.std_deviation / std::sqrt(static_cast<float>(data1.size()));
    float se2 = analysis2.std_deviation / std::sqrt(static_cast<float>(data2.size()));
    float pooled_se = std::sqrt(se1 * se1 + se2 * se2);

    if (pooled_se == 0.0f) {
        return Ok(false);  // No variance, no significance
    }

    // Calculate t-statistic
    float t_stat = std::abs(analysis1.mean - analysis2.mean) / pooled_se;

    // Simplified critical value (approximation for large samples)
    float critical_value = 1.96f;  // For alpha = 0.05, two-tailed
    if (alpha < 0.01f) {
        critical_value = 2.58f;
    }

    return Ok(t_stat > critical_value);
}

auto StatisticalAnalyzer::calculate_correlation(const std::vector<float>& data1,
                                                const std::vector<float>& data2) -> float {
    if (data1.size() != data2.size() || data1.empty()) {
        return 0.0f;
    }

    // Calculate means
    float mean1 = std::accumulate(data1.begin(), data1.end(), 0.0f) / data1.size();
    float mean2 = std::accumulate(data2.begin(), data2.end(), 0.0f) / data2.size();

    // Calculate correlation coefficient
    float numerator = 0.0f;
    float sum_sq1 = 0.0f;
    float sum_sq2 = 0.0f;

    for (std::size_t i = 0; i < data1.size(); ++i) {
        float diff1 = data1[i] - mean1;
        float diff2 = data2[i] - mean2;
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }

    float denominator = std::sqrt(sum_sq1 * sum_sq2);
    return (denominator > 0.0f) ? (numerator / denominator) : 0.0f;
}

auto StatisticalAnalyzer::test_normality(const std::vector<float>& data) -> bool {
    if (data.size() < 3) {
        return false;
    }

    // Simplified normality test using skewness and kurtosis
    auto analysis = analyze_distribution(data);

    // Calculate skewness
    double skewness_sum = 0.0;
    for (float value : data) {
        double normalized = (value - analysis.mean) / analysis.std_deviation;
        skewness_sum += normalized * normalized * normalized;
    }
    float skewness = static_cast<float>(skewness_sum / data.size());

    // Normal distribution should have skewness close to 0
    return std::abs(skewness) < 1.0f;
}

auto StatisticalAnalyzer::confidence_interval(const std::vector<float>& data,
                                              float confidence_level) -> std::array<float, 2> {
    if (data.empty()) {
        return {0.0f, 0.0f};
    }

    auto analysis = analyze_distribution(data);
    float se = analysis.std_deviation / std::sqrt(static_cast<float>(data.size()));

    // Critical value for 95% confidence (approximation)
    float critical_value = 1.96f;
    if (confidence_level > 0.99f) {
        critical_value = 2.58f;
    } else if (confidence_level < 0.90f) {
        critical_value = 1.64f;
    }

    float margin = critical_value * se;
    return {analysis.mean - margin, analysis.mean + margin};
}

auto StatisticalAnalyzer::calculate_percentile(const std::vector<float>& sorted_data,
                                               float percentile) -> float {
    if (sorted_data.empty()) {
        return 0.0f;
    }

    float index = percentile * (sorted_data.size() - 1);
    std::size_t lower = static_cast<std::size_t>(std::floor(index));
    std::size_t upper = static_cast<std::size_t>(std::ceil(index));

    if (lower == upper) {
        return sorted_data[lower];
    }

    float weight = index - lower;
    return sorted_data[lower] * (1.0f - weight) + sorted_data[upper] * weight;
}

//=============================================================================
// TestDataGenerator Implementation
//=============================================================================

TestDataGenerator::TestDataGenerator(std::uint32_t seed)
    : rng_(seed), uniform_dist_(0.0f, 1.0f), normal_dist_(0.0f, 1.0f) {}

auto TestDataGenerator::generate_tensor(const Shape& shape, const StatisticalProperties& properties)
    -> FloatTensor {
    FloatTensor tensor(shape);

    switch (properties.distribution) {
        case Distribution::UNIFORM: {
            std::uniform_real_distribution<float> dist(properties.min_value, properties.max_value);
            for (std::size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(rng_);
            }
            break;
        }
        case Distribution::NORMAL: {
            std::normal_distribution<float> dist(properties.mean, properties.std_deviation);
            for (std::size_t i = 0; i < tensor.size(); ++i) {
                float value = dist(rng_);
                tensor.data()[i] = std::clamp(value, properties.min_value, properties.max_value);
            }
            break;
        }
        case Distribution::REALISTIC:
        default: {
            // Use normal distribution centered around mean
            std::normal_distribution<float> dist(properties.mean, properties.std_deviation);
            for (std::size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(rng_);
            }
            break;
        }
    }

    return tensor;
}

auto TestDataGenerator::generate_classification_data(const Shape& input_shape,
                                                     std::uint32_t num_classes,
                                                     std::uint32_t num_samples)
    -> std::vector<InferenceRequest> {
    std::vector<InferenceRequest> requests;
    requests.reserve(num_samples);

    for (std::uint32_t i = 0; i < num_samples; ++i) {
        InferenceRequest request;
        request.batch_size = 1;
        request.request_id = i;

        // Generate input tensor
        TensorInput input;
        input.name = "input";
        input.tensor = generate_tensor(input_shape,
                                       StatisticalProperties{.mean = 0.0f,
                                                             .std_deviation = 1.0f,
                                                             .min_value = -3.0f,
                                                             .max_value = 3.0f,
                                                             .distribution = Distribution::NORMAL});

        request.inputs.push_back(std::move(input));
        requests.push_back(std::move(request));
    }

    return requests;
}

auto TestDataGenerator::generate_realistic_image_data(const Shape& image_shape,
                                                      std::uint32_t num_samples)
    -> std::vector<InferenceRequest> {
    std::vector<InferenceRequest> requests;
    requests.reserve(num_samples);

    for (std::uint32_t i = 0; i < num_samples; ++i) {
        InferenceRequest request;
        request.batch_size = 1;
        request.request_id = i;

        // Generate realistic image data with natural statistics
        TensorInput input;
        input.name = "image";

        // Images typically have pixel values in [0, 255] normalized to [0, 1]
        input.tensor = generate_tensor(image_shape,
                                       StatisticalProperties{.mean = 0.5f,  // Normalized pixel mean
                                                             .std_deviation =
                                                                 0.2f,  // Realistic pixel variance
                                                             .min_value = 0.0f,
                                                             .max_value = 1.0f,
                                                             .distribution = Distribution::NORMAL});

        request.inputs.push_back(std::move(input));
        requests.push_back(std::move(request));
    }

    return requests;
}

//=============================================================================
// ClassificationTestFixture Implementation
//=============================================================================

ClassificationTestFixture::ClassificationTestFixture(Config config)
    : config_(std::move(config)), data_generator_(42) {}

auto ClassificationTestFixture::get_model_config() const -> ModelConfig {
    ModelConfig model_config;
    model_config.name = config_.model_name;
    model_config.backend = config_.preferred_backend;
    model_config.precision = config_.precision;
    model_config.max_batch_size = 1;

    // Set up input specification
    TensorSpec input_spec;
    input_spec.name = "input";
    input_spec.shape = config_.input_shape;
    input_spec.dtype = DataType::FLOAT32;
    input_spec.is_dynamic = false;
    model_config.input_specs = {input_spec};

    // Set up output specification
    TensorSpec output_spec;
    output_spec.name = "output";
    output_spec.shape = {1, config_.num_classes};
    output_spec.dtype = DataType::FLOAT32;
    output_spec.is_dynamic = false;
    model_config.output_specs = {output_spec};

    return model_config;
}

auto ClassificationTestFixture::generate_test_data(std::uint32_t num_samples)
    -> std::vector<InferenceRequest> {
    return data_generator_.generate_classification_data(
        config_.input_shape, config_.num_classes, num_samples);
}

auto ClassificationTestFixture::get_performance_expectations() const -> PerformanceThresholds {
    PerformanceThresholds thresholds;
    thresholds.max_latency = std::chrono::milliseconds(100);  // 100ms max
    thresholds.min_throughput_fps = 10.0f;                    // 10 FPS minimum
    thresholds.max_memory_usage_mb = 2048;                    // 2GB max
    thresholds.fail_on_memory_leaks = true;
    return thresholds;
}

auto ClassificationTestFixture::validate_output(const InferenceResponse& response) const
    -> Result<bool, std::string> {
    return OutputValidator::validate_classification_output(response, config_.num_classes);
}

auto ClassificationTestFixture::create() -> ClassificationTestFixture {
    return ClassificationTestFixture(Config{});
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

//=============================================================================
// ObjectDetectionTestFixture Implementation
//=============================================================================

ObjectDetectionTestFixture::ObjectDetectionTestFixture(Config config)
    : config_(std::move(config)), data_generator_(42) {}

auto ObjectDetectionTestFixture::get_model_config() const -> ModelConfig {
    ModelConfig model_config;
    model_config.name = config_.model_name;
    model_config.max_batch_size = 1;

    // Input specification
    TensorSpec input_spec;
    input_spec.name = "input";
    input_spec.shape = config_.input_shape;
    input_spec.dtype = DataType::FLOAT32;
    input_spec.is_dynamic = false;
    model_config.input_specs = {input_spec};

    // Object detection typically has multiple outputs
    TensorSpec boxes_spec;
    boxes_spec.name = "boxes";
    boxes_spec.shape = {1, config_.max_detections, 4};  // [x, y, w, h]
    boxes_spec.dtype = DataType::FLOAT32;

    TensorSpec scores_spec;
    scores_spec.name = "scores";
    scores_spec.shape = {1, config_.max_detections};
    scores_spec.dtype = DataType::FLOAT32;

    TensorSpec classes_spec;
    classes_spec.name = "classes";
    classes_spec.shape = {1, config_.max_detections};
    classes_spec.dtype = DataType::INT32;

    model_config.output_specs = {boxes_spec, scores_spec, classes_spec};

    return model_config;
}

auto ObjectDetectionTestFixture::generate_test_data(std::uint32_t num_samples)
    -> std::vector<InferenceRequest> {
    return data_generator_.generate_object_detection_data(
        config_.input_shape, config_.max_detections, num_samples);
}

auto ObjectDetectionTestFixture::get_performance_expectations() const -> PerformanceThresholds {
    PerformanceThresholds thresholds;
    thresholds.max_latency = std::chrono::milliseconds(200);  // 200ms max for object detection
    thresholds.min_throughput_fps = 5.0f;                     // 5 FPS minimum
    thresholds.max_memory_usage_mb = 4096;                    // 4GB max
    return thresholds;
}

auto ObjectDetectionTestFixture::validate_output(const InferenceResponse& response) const
    -> Result<bool, std::string> {
    return OutputValidator::validate_object_detection_output(response, config_.max_detections);
}

auto ObjectDetectionTestFixture::create() -> ObjectDetectionTestFixture {
    return ObjectDetectionTestFixture(Config{});
}

//=============================================================================
// OutputValidator Implementation
//=============================================================================

auto OutputValidator::validate_classification_output(const InferenceResponse& response,
                                                     std::uint32_t expected_classes)
    -> Result<bool, std::string> {
    if (response.outputs.empty()) {
        return Err(std::string("No outputs in response"));
    }

    const auto& output = response.outputs[0];
    auto shape = output.get_shape();

    if (shape.size() != 2) {
        return Err(std::string("Expected 2D output tensor for classification"));
    }

    if (shape[1] != expected_classes) {
        return Err(std::string("Output class count mismatch"));
    }

    // Validate numerical stability
    return validate_numerical_stability(response);
}

auto OutputValidator::validate_object_detection_output(const InferenceResponse& response,
                                                       std::uint32_t max_detections)
    -> Result<bool, std::string> {
    if (response.outputs.size() < 3) {
        return Err(
            std::string("Object detection requires at least 3 outputs (boxes, scores, classes)"));
    }

    // Validate boxes output
    const auto& boxes = response.outputs[0];
    auto boxes_shape = boxes.get_shape();
    if (boxes_shape.size() != 3 || boxes_shape[2] != 4) {
        return Err(std::string("Invalid boxes tensor shape"));
    }

    return validate_numerical_stability(response);
}

auto OutputValidator::validate_numerical_stability(const InferenceResponse& response)
    -> Result<bool, std::string> {
    for (const auto& output : response.outputs) {
        if (!check_tensor_bounds(output)) {
            return Err(std::string("Tensor contains invalid values (NaN/Inf)"));
        }
    }

    return Ok(true);
}

auto OutputValidator::check_tensor_bounds(const TensorOutput& output) -> bool {
    return std::visit(
        [](const auto& tensor) {
            for (std::size_t i = 0; i < tensor.size(); ++i) {
                auto value = tensor.data()[i];
                if (std::isnan(value) || std::isinf(value)) {
                    return false;
                }
            }
            return true;
        },
        output.tensor);
}

//=============================================================================
// PerformanceAnalyzer Implementation
//=============================================================================

PerformanceAnalyzer::PerformanceAnalyzer(Config config) : config_(config) {}

auto PerformanceAnalyzer::benchmark_backend(InferenceEngine* engine,
                                            const std::vector<InferenceRequest>& inputs)
    -> Result<TestPerformanceMetrics, std::string> {
    if (!engine) {
        return Err(std::string("Null engine pointer"));
    }

    if (inputs.empty()) {
        return Err(std::string("No input data provided"));
    }

    std::vector<std::chrono::milliseconds> latencies;
    latencies.reserve(inputs.size() * config_.max_iterations);

    TestPerformanceMetrics metrics;
    auto start_time = std::chrono::steady_clock::now();

    // Run benchmark iterations
    for (std::uint32_t iter = 0; iter < config_.max_iterations; ++iter) {
        for (const auto& input : inputs) {
            auto inference_start = std::chrono::steady_clock::now();
            auto result = engine->run_inference(input);
            auto inference_end = std::chrono::steady_clock::now();

            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end -
                                                                                 inference_start);
            latencies.push_back(latency);

            if (result.is_err()) {
                // Continue benchmarking but note the error
                continue;
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (latencies.empty()) {
        return Err(std::string("No successful inference runs"));
    }

    // Calculate metrics
    std::vector<float> latency_values;
    latency_values.reserve(latencies.size());
    for (const auto& latency : latencies) {
        latency_values.push_back(static_cast<float>(latency.count()));
    }

    auto analysis = StatisticalAnalyzer::analyze_distribution(latency_values);

    metrics.avg_latency = std::chrono::milliseconds(static_cast<int>(analysis.mean));
    metrics.min_latency = std::chrono::milliseconds(static_cast<int>(analysis.min_value));
    metrics.max_latency = std::chrono::milliseconds(static_cast<int>(analysis.max_value));

    float total_seconds = total_time.count() / 1000.0f;
    if (total_seconds > 0) {
        metrics.throughput_fps = latencies.size() / total_seconds;
    }

    return Ok(metrics);
}

//=============================================================================
// Utility Functions Implementation
//=============================================================================

auto setup_test_environment(const std::string& log_level) -> void {
    // Initialize logging system
    // This would typically set up the logging configuration
    LOG_INFO_PRINT("Test environment setup with log level: {}", log_level);
}

auto cleanup_test_environment() -> void {
    // Clean up any temporary files or resources
    LOG_INFO_PRINT("Test environment cleanup completed");
}

auto create_mock_integration_framework() -> std::unique_ptr<MLIntegrationFramework> {
    auto mock_factory = std::make_unique<TestBackendFactory>();
    return std::make_unique<MLIntegrationFramework>(std::move(mock_factory));
}

}  // namespace inference_lab::integration::utils
