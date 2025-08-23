// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file test_ml_logging.cpp
 * @brief Comprehensive tests for ML-specific logging extensions
 *
 * This test suite validates all aspects of the ML logging extensions including:
 * - Model context registration and tracking
 * - Inference metrics logging and aggregation
 * - ML operation logging with structured output
 * - Error context logging with metadata
 * - Performance metrics buffering and flushing
 * - Thread safety of ML logging operations
 */

#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../src/logging.hpp"

namespace inference_lab::common::tests {

class MLLoggingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create temporary log file for testing
        test_log_file_ = std::filesystem::temp_directory_path() / "test_ml_logging.log";

        // Get fresh logger instance with test log file
        logger_ = &Logger::get_instance(test_log_file_.string(), false);

        // Enable all logging levels and ML logging
        Logger::set_level_enabled(LogLevel::DEBUG, true);
        Logger::set_level_enabled(LogLevel::INFO, true);
        Logger::set_level_enabled(LogLevel::WARNING, true);
        Logger::set_level_enabled(LogLevel::ERROR, true);
        logger_->set_ml_logging_enabled(true);

        // Ensure file output is enabled
        logger_->set_file_output_enabled(true);

        // Clear any existing models
        clear_test_models();
    }

    void TearDown() override {
        // Clean up test models
        clear_test_models();

        // Remove test log file
        if (std::filesystem::exists(test_log_file_)) {
            std::filesystem::remove(test_log_file_);
        }
    }

    void clear_test_models() {
        // Unregister any test models
        logger_->unregister_model("test_model");
        logger_->unregister_model("resnet50");
        logger_->unregister_model("bert_base");
        logger_->unregister_model("concurrent_model");
    }

    auto read_log_content() -> std::string {
        // Flush the log file to ensure all writes are complete
        logger_->flush();

        // Try to read from common test log locations
        std::vector<std::filesystem::path> possible_paths = {
            test_log_file_,
            "../custom.log",  // Default logger path
            "custom.log"      // Alternative default path
        };

        for (const auto& path : possible_paths) {
            if (std::filesystem::exists(path)) {
                std::ifstream file(path);
                if (file.is_open()) {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    auto content = buffer.str();
                    if (!content.empty()) {
                        return content;
                    }
                }
            }
        }

        return "";
    }

    auto create_test_model_context(const std::string& name = "test_model",
                                   const std::string& version = "1.0.0") -> ModelContext {
        ModelContext context;
        context.name = name;
        context.version = version;
        context.framework = "ONNX";
        context.stage = ModelStage::DEVELOPMENT;
        context.path = "/tmp/" + name + ".onnx";
        context.size_mb = 100;
        context.checksum = "abc123def456";
        context.loaded_at = std::chrono::system_clock::now();
        return context;
    }

    auto create_test_metrics(double latency = 50.0, double throughput = 100.0) -> InferenceMetrics {
        InferenceMetrics metrics;
        metrics.latency_ms = latency;
        metrics.preprocessing_ms = 10.0;
        metrics.inference_ms = 35.0;
        metrics.postprocessing_ms = 5.0;
        metrics.memory_mb = 512;
        metrics.batch_size = 8;
        metrics.throughput = throughput;
        metrics.confidence = 0.85;
        metrics.device = "CPU";
        return metrics;
    }

    std::filesystem::path test_log_file_;
    Logger* logger_;
};

//=============================================================================
// Model Context Registration Tests
//=============================================================================

TEST_F(MLLoggingTest, RegisterModelContext) {
    auto context = create_test_model_context();

    logger_->register_model(context);

    auto retrieved = logger_->get_model_context("test_model");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->name, "test_model");
    EXPECT_EQ(retrieved->version, "1.0.0");
    EXPECT_EQ(retrieved->framework, "ONNX");
    EXPECT_EQ(retrieved->stage, ModelStage::DEVELOPMENT);
    EXPECT_EQ(retrieved->size_mb, 100);

    // Check log output
    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("Registered ML model"), std::string::npos);
    EXPECT_NE(log_content.find("name=test_model"), std::string::npos);
    EXPECT_NE(log_content.find("version=1.0.0"), std::string::npos);
}

TEST_F(MLLoggingTest, UnregisterModelContext) {
    auto context = create_test_model_context();
    logger_->register_model(context);

    logger_->unregister_model("test_model");

    auto retrieved = logger_->get_model_context("test_model");
    EXPECT_FALSE(retrieved.has_value());

    // Check log output
    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("Unregistered ML model"), std::string::npos);
}

TEST_F(MLLoggingTest, UpdateModelStage) {
    auto context = create_test_model_context();
    logger_->register_model(context);

    logger_->update_model_stage("test_model", ModelStage::PRODUCTION);

    auto retrieved = logger_->get_model_context("test_model");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->stage, ModelStage::PRODUCTION);

    // Check log output
    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("Updated model stage"), std::string::npos);
    EXPECT_NE(log_content.find("DEVELOPMENT -> PRODUCTION"), std::string::npos);
}

TEST_F(MLLoggingTest, NonExistentModelOperations) {
    // Test operations on non-existent model
    auto retrieved = logger_->get_model_context("nonexistent");
    EXPECT_FALSE(retrieved.has_value());

    // Update stage on non-existent model should not crash
    logger_->update_model_stage("nonexistent", ModelStage::PRODUCTION);

    // Unregister non-existent model should not crash
    logger_->unregister_model("nonexistent");
}

//=============================================================================
// ML Operation Logging Tests
//=============================================================================

TEST_F(MLLoggingTest, LogMLOperationWithRegisteredModel) {
    auto context = create_test_model_context();
    logger_->register_model(context);

    LOG_MODEL_LOAD("test_model", "Loading from path: {}", context.path);
    LOG_INFERENCE_START("test_model", "batch_size={}", 8);
    LOG_INFERENCE_COMPLETE("test_model", "processed {} samples", 64);

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("[ML:MODEL_LOAD]"), std::string::npos);
    EXPECT_NE(log_content.find("[ML:INFERENCE_START]"), std::string::npos);
    EXPECT_NE(log_content.find("[ML:INFERENCE_COMPLETE]"), std::string::npos);
    EXPECT_NE(log_content.find("model=test_model"), std::string::npos);
    EXPECT_NE(log_content.find("version=1.0.0"), std::string::npos);
    EXPECT_NE(log_content.find("framework=ONNX"), std::string::npos);
}

TEST_F(MLLoggingTest, LogMLOperationWithUnregisteredModel) {
    LOG_MODEL_LOAD("unregistered_model", "Loading model");

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("[ML:MODEL_LOAD]"), std::string::npos);
    EXPECT_NE(log_content.find("model=unregistered_model (unregistered)"), std::string::npos);
}

TEST_F(MLLoggingTest, LogMLOperationWithoutMessage) {
    auto context = create_test_model_context();
    logger_->register_model(context);

    LOG_ML_OPERATION(MLOperation::BATCH_PROCESS, "test_model", "");

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("[ML:BATCH_PROCESS]"), std::string::npos);
    EXPECT_NE(log_content.find("model=test_model"), std::string::npos);
}

//=============================================================================
// Inference Metrics Logging Tests
//=============================================================================

TEST_F(MLLoggingTest, LogInferenceMetrics) {
    auto metrics = create_test_metrics();

    LOG_ML_METRICS("test_model", metrics);

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("[ML:METRICS]"), std::string::npos);
    EXPECT_NE(log_content.find("model=test_model"), std::string::npos);
    EXPECT_NE(log_content.find("latency=50.00ms"), std::string::npos);
    EXPECT_NE(log_content.find("preprocessing=10.00ms"), std::string::npos);
    EXPECT_NE(log_content.find("inference=35.00ms"), std::string::npos);
    EXPECT_NE(log_content.find("postprocessing=5.00ms"), std::string::npos);
    EXPECT_NE(log_content.find("memory=512MB"), std::string::npos);
    EXPECT_NE(log_content.find("batch_size=8"), std::string::npos);
    EXPECT_NE(log_content.find("throughput=100.00/s"), std::string::npos);
    EXPECT_NE(log_content.find("confidence=0.850"), std::string::npos);
    EXPECT_NE(log_content.find("device=CPU"), std::string::npos);
}

TEST_F(MLLoggingTest, LogInferenceMetricsVariousDevices) {
    auto cpu_metrics = create_test_metrics();
    cpu_metrics.device = "CPU";

    auto gpu_metrics = create_test_metrics();
    gpu_metrics.device = "CUDA:0";
    gpu_metrics.latency_ms = 25.0;  // GPU should be faster

    LOG_ML_METRICS("cpu_model", cpu_metrics);
    LOG_ML_METRICS("gpu_model", gpu_metrics);

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("device=CPU"), std::string::npos);
    EXPECT_NE(log_content.find("device=CUDA:0"), std::string::npos);
    EXPECT_NE(log_content.find("latency=25.00ms"), std::string::npos);
}

//=============================================================================
// ML Error Logging Tests
//=============================================================================

TEST_F(MLLoggingTest, LogMLError) {
    MLErrorContext error_context;
    error_context.error_code = "INFERENCE_TIMEOUT";
    error_context.component = "TensorRTEngine";
    error_context.operation = "run_inference";
    error_context.metadata = {
        {"timeout_ms", "5000"}, {"batch_size", "16"}, {"model_size", "1.2GB"}};

    LOG_ML_ERROR("test_model", error_context, "Inference timed out after 5 seconds");

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("[ML:ERROR]"), std::string::npos);
    EXPECT_NE(log_content.find("model=test_model"), std::string::npos);
    EXPECT_NE(log_content.find("component=TensorRTEngine"), std::string::npos);
    EXPECT_NE(log_content.find("operation=run_inference"), std::string::npos);
    EXPECT_NE(log_content.find("error_code=INFERENCE_TIMEOUT"), std::string::npos);
    EXPECT_NE(log_content.find("timeout_ms=5000"), std::string::npos);
    EXPECT_NE(log_content.find("batch_size=16"), std::string::npos);
    EXPECT_NE(log_content.find("message=Inference timed out"), std::string::npos);
}

TEST_F(MLLoggingTest, LogMLErrorWithoutMessage) {
    MLErrorContext error_context;
    error_context.error_code = "MODEL_LOAD_FAILED";
    error_context.component = "ONNXEngine";
    error_context.operation = "load_model";

    LOG_ML_ERROR("test_model", error_context, "");

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("[ML:ERROR]"), std::string::npos);
    EXPECT_NE(log_content.find("error_code=MODEL_LOAD_FAILED"), std::string::npos);
    // Should not contain "message=" when message is empty
    EXPECT_EQ(log_content.find("message="), std::string::npos);
}

//=============================================================================
// Metrics Buffering Tests
//=============================================================================

TEST_F(MLLoggingTest, BufferAndFlushMetrics) {
    // Buffer multiple metrics
    for (int i = 0; i < 5; ++i) {
        auto metrics = create_test_metrics(50.0 + i, 100.0 + i * 10);
        logger_->buffer_metrics(metrics);
    }

    EXPECT_EQ(logger_->get_metrics_buffer_size(), 5);

    // Flush metrics
    logger_->flush_metrics_buffer();

    EXPECT_EQ(logger_->get_metrics_buffer_size(), 0);

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("[ML:AGGREGATE]"), std::string::npos);
    EXPECT_NE(log_content.find("buffered_samples=5"), std::string::npos);
    EXPECT_NE(log_content.find("avg_latency=52.00ms"),
              std::string::npos);  // Average of 50,51,52,53,54
}

TEST_F(MLLoggingTest, MetricsBufferSizeLimit) {
    const std::size_t max_size = 3;
    logger_->set_max_metrics_buffer_size(max_size);

    // Add more metrics than the limit
    for (int i = 0; i < 5; ++i) {
        auto metrics = create_test_metrics();
        logger_->buffer_metrics(metrics);
    }

    // Should not exceed the limit
    EXPECT_EQ(logger_->get_metrics_buffer_size(), max_size);
}

TEST_F(MLLoggingTest, FlushEmptyBuffer) {
    // Flushing empty buffer should not crash
    logger_->flush_metrics_buffer();

    auto log_content = read_log_content();
    // Should not contain aggregate message for empty buffer
    EXPECT_EQ(log_content.find("[ML:AGGREGATE]"), std::string::npos);
}

//=============================================================================
// ML Logging Enable/Disable Tests
//=============================================================================

TEST_F(MLLoggingTest, DisableMLLogging) {
    logger_->set_ml_logging_enabled(false);
    EXPECT_FALSE(logger_->is_ml_logging_enabled());

    auto context = create_test_model_context();
    logger_->register_model(context);

    LOG_MODEL_LOAD("test_model", "This should not appear");

    auto metrics = create_test_metrics();
    LOG_ML_METRICS("test_model", metrics);

    auto log_content = read_log_content();
    // Should only contain the enable/disable message, not the ML operations
    EXPECT_NE(log_content.find("ML logging disabled"), std::string::npos);
    EXPECT_EQ(log_content.find("[ML:MODEL_LOAD]"), std::string::npos);
    EXPECT_EQ(log_content.find("[ML:METRICS]"), std::string::npos);
}

TEST_F(MLLoggingTest, EnableMLLogging) {
    logger_->set_ml_logging_enabled(true);
    EXPECT_TRUE(logger_->is_ml_logging_enabled());

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("ML logging enabled"), std::string::npos);
}

//=============================================================================
// Aggregate Metrics Tests
//=============================================================================

TEST_F(MLLoggingTest, GetAggregateMetrics) {
    // Buffer some metrics
    for (int i = 0; i < 3; ++i) {
        auto metrics = create_test_metrics(100.0 + i * 10, 50.0 + i * 5);
        logger_->buffer_metrics(metrics);
    }

    auto aggregate = logger_->get_aggregate_metrics("test_model", std::chrono::minutes(5));
    ASSERT_TRUE(aggregate.has_value());

    // Should be average: (100+110+120)/3 = 110
    EXPECT_NEAR(aggregate->latency_ms, 110.0, 0.1);
    // Throughput average: (50+55+60)/3 = 55
    EXPECT_NEAR(aggregate->throughput, 55.0, 0.1);
}

TEST_F(MLLoggingTest, GetAggregateMetricsEmptyBuffer) {
    auto aggregate = logger_->get_aggregate_metrics("test_model", std::chrono::minutes(5));
    EXPECT_FALSE(aggregate.has_value());
}

//=============================================================================
// Thread Safety Tests
//=============================================================================

TEST_F(MLLoggingTest, ConcurrentModelRegistration) {
    const int num_threads = 10;
    std::vector<std::thread> threads;

    // Register models concurrently
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, i]() {
            auto context = create_test_model_context("concurrent_model_" + std::to_string(i));
            logger_->register_model(context);
        });
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all models were registered
    for (int i = 0; i < num_threads; ++i) {
        auto retrieved = logger_->get_model_context("concurrent_model_" + std::to_string(i));
        EXPECT_TRUE(retrieved.has_value()) << "Model " << i << " not found";
    }

    // Clean up
    for (int i = 0; i < num_threads; ++i) {
        logger_->unregister_model("concurrent_model_" + std::to_string(i));
    }
}

TEST_F(MLLoggingTest, ConcurrentMetricsBuffering) {
    const int num_threads = 5;
    const int metrics_per_thread = 10;
    std::vector<std::thread> threads;

    // Buffer metrics concurrently
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, metrics_per_thread = metrics_per_thread]() {
            for (int j = 0; j < metrics_per_thread; ++j) {
                auto metrics = create_test_metrics();
                logger_->buffer_metrics(metrics);
            }
        });
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Should have all metrics buffered (unless buffer size limit was hit)
    auto buffer_size = logger_->get_metrics_buffer_size();
    EXPECT_GT(buffer_size, 0);
    EXPECT_LE(buffer_size, num_threads * metrics_per_thread);
}

//=============================================================================
// Integration Tests with Real-World Scenarios
//=============================================================================

TEST_F(MLLoggingTest, CompleteInferenceWorkflow) {
    // Register model
    auto context = create_test_model_context("resnet50", "2.1.0");
    context.stage = ModelStage::PRODUCTION;
    context.framework = "TensorRT";
    logger_->register_model(context);

    // Model load
    LOG_MODEL_LOAD("resnet50", "Loading from {}", context.path);

    // Inference workflow
    LOG_INFERENCE_START("resnet50", "Processing batch of {} images", 32);

    auto metrics = create_test_metrics(15.2, 2100.0);  // Fast GPU inference
    metrics.device = "CUDA:0";
    metrics.batch_size = 32;
    LOG_ML_METRICS("resnet50", metrics);

    LOG_INFERENCE_COMPLETE("resnet50", "Processed {} images successfully", 32);

    // Validation
    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("name=resnet50"), std::string::npos);
    EXPECT_NE(log_content.find("version=2.1.0"), std::string::npos);
    EXPECT_NE(log_content.find("stage=PRODUCTION"), std::string::npos);
    EXPECT_NE(log_content.find("framework=TensorRT"), std::string::npos);
    EXPECT_NE(log_content.find("latency=15.20ms"), std::string::npos);
    EXPECT_NE(log_content.find("device=CUDA:0"), std::string::npos);
    EXPECT_NE(log_content.find("batch_size=32"), std::string::npos);
}

TEST_F(MLLoggingTest, ModelLifecycleLogging) {
    auto context = create_test_model_context("bert_base", "1.0.0");

    // Development stage
    logger_->register_model(context);
    LOG_MODEL_VALIDATE("bert_base", "Validation accuracy: {:.3f}", 0.924);

    // Promote to staging
    logger_->update_model_stage("bert_base", ModelStage::STAGING);
    LOG_PERFORMANCE_BENCHMARK("bert_base", "Staging performance test complete");

    // Promote to production
    logger_->update_model_stage("bert_base", ModelStage::PRODUCTION);
    LOG_MODEL_LOAD("bert_base", "Production deployment successful");

    // Eventually archive
    logger_->update_model_stage("bert_base", ModelStage::ARCHIVED);
    LOG_MODEL_UNLOAD("bert_base", "Model archived due to new version");

    auto log_content = read_log_content();
    EXPECT_NE(log_content.find("DEVELOPMENT -> STAGING"), std::string::npos);
    EXPECT_NE(log_content.find("STAGING -> PRODUCTION"), std::string::npos);
    EXPECT_NE(log_content.find("PRODUCTION -> ARCHIVED"), std::string::npos);
}

}  // namespace inference_lab::common::tests
