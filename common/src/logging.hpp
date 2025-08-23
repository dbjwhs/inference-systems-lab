// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

// Common logging utilities for the Inference Systems Laboratory
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace inference_lab::common {

enum class LogLevel : std::uint8_t {  // NOLINT(performance-enum-size) - false positive, uint8_t is
                                      // correct
    DEBUG = 0,
    INFO = 1,
    NORMAL = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
};

//=============================================================================
// ML-Specific Logging Extensions
//=============================================================================

/**
 * @brief ML inference operation types for structured logging
 */
enum class MLOperation : std::uint8_t {
    MODEL_LOAD = 0,
    MODEL_UNLOAD = 1,
    INFERENCE_START = 2,
    INFERENCE_COMPLETE = 3,
    BATCH_PROCESS = 4,
    MODEL_VALIDATE = 5,
    PERFORMANCE_BENCHMARK = 6,
    ERROR_OCCURRED = 7
};

/**
 * @brief ML model lifecycle stages for tracking
 */
enum class ModelStage : std::uint8_t {
    DEVELOPMENT = 0,
    STAGING = 1,
    PRODUCTION = 2,
    ARCHIVED = 3,
    DEPRECATED = 4
};

/**
 * @brief Inference metrics structure for structured logging
 */
struct InferenceMetrics {
    double latency_ms = 0.0;         ///< End-to-end inference latency
    double preprocessing_ms = 0.0;   ///< Input preprocessing time
    double inference_ms = 0.0;       ///< Core model inference time
    double postprocessing_ms = 0.0;  ///< Output postprocessing time
    std::size_t memory_mb = 0;       ///< Memory usage in MB
    std::size_t batch_size = 1;      ///< Batch size processed
    double throughput = 0.0;         ///< Samples per second
    double confidence = 0.0;         ///< Average prediction confidence
    std::string device = "CPU";      ///< Execution device (CPU/GPU/etc.)
};

/**
 * @brief Model context for tracking model versions and metadata
 */
struct ModelContext {
    std::string name;                                 ///< Model name/identifier
    std::string version = "1.0.0";                    ///< Semantic version
    std::string framework = "ONNX";                   ///< ML framework (ONNX, TensorRT, etc.)
    ModelStage stage = ModelStage::DEVELOPMENT;       ///< Deployment stage
    std::string path;                                 ///< Model file path
    std::size_t size_mb = 0;                          ///< Model size in MB
    std::string checksum;                             ///< Model file checksum
    std::chrono::system_clock::time_point loaded_at;  ///< Load timestamp
};

/**
 * @brief ML error context for enhanced error logging
 */
struct MLErrorContext {
    std::string error_code;                                 ///< Structured error code
    std::string component;                                  ///< Component where error occurred
    std::string operation;                                  ///< Operation that failed
    std::unordered_map<std::string, std::string> metadata;  ///< Additional context
};

class Logger {
  public:
    Logger(const Logger&) = delete;
    auto operator=(const Logger&) -> Logger& = delete;
    Logger(Logger&&) = delete;
    auto operator=(Logger&&) -> Logger& = delete;

  private:
    // shared_ptr to maintain the singleton
    inline static std::shared_ptr<Logger> m_instance;
    inline static std::mutex m_instance_mutex;  // mutex for thread-safe initialization

    // helper method to handle the common logging logic
    void write_log_message(LogLevel level, const std::string& message);

    // helper to build the log prefix
    static auto create_log_prefix(LogLevel level) -> std::stringstream;

    // constructor is now private to control instantiation
    explicit Logger(const std::string& path, bool append = true);

  public:
    // raii class for temporarily disabling stderr output
    class StderrSuppressionGuard {
      public:
        StderrSuppressionGuard();
        ~StderrSuppressionGuard();
        StderrSuppressionGuard(const StderrSuppressionGuard&) = delete;
        auto operator=(const StderrSuppressionGuard&) -> StderrSuppressionGuard& = delete;
        StderrSuppressionGuard(StderrSuppressionGuard&&) = delete;
        auto operator=(StderrSuppressionGuard&&) -> StderrSuppressionGuard& = delete;

      private:
        bool m_was_enabled_;
    };

    // private method to get or create the instance
    static auto get_or_create_instance(const std::string& path = "../custom.log",
                                       bool append = true) -> std::shared_ptr<Logger>;

    // returns a reference for backward compatibility but uses shared_ptr internally
    static auto get_instance() -> Logger&;

    // custom path version of getinstance
    static auto get_instance(const std::string& custom_path, bool append = true) -> Logger&;

    // new method for code that explicitly wants to manage the shared_ptr
    static auto get_instance_ptr() -> std::shared_ptr<Logger>;

    // with custom path for the shared_ptr version
    static auto get_instance_ptr(const std::string& custom_path, bool append = true)
        -> std::shared_ptr<Logger>;

    // destructor
    ~Logger();

    // Template-based logging methods for C++17 compatibility
    template <typename... FormatArgs>
    void print_log(const LogLevel LEVEL, const std::string& format, const FormatArgs&... args) {
        if (!is_level_enabled(LEVEL)) {
            return;
        }
        auto prefix = create_log_prefix(LEVEL);
        auto formatted_message = format_message(format, args...);
        auto full_message = prefix.str() + formatted_message + "\n";
        write_log_message(LEVEL, full_message);
    }

    // Template-based logging with depth for C++17 compatibility
    template <typename... FormatArgs>
    void print_log_with_depth(const LogLevel LEVEL,
                              const int DEPTH,
                              const std::string& format,
                              const FormatArgs&... args) {
        if (!is_level_enabled(LEVEL)) {
            return;
        }
        auto prefix = create_log_prefix(LEVEL);
        auto formatted_message = format_message(format, args...);
        auto full_message = prefix.str() + get_indentation(DEPTH) + formatted_message + "\n";
        write_log_message(LEVEL, full_message);
    }

    // enable/disable specific log level
    static void set_level_enabled(LogLevel level, bool enabled);

    // check if a specific log level is enabled
    static auto is_level_enabled(LogLevel level) -> bool;

    // disable stderr output
    void disable_stderr();

    // enable stderr output
    void enable_stderr();

    // get current stderr output state
    auto is_stderr_enabled() const -> bool;

    // enable/disable file output
    void set_file_output_enabled(bool enabled);

    // check if file output is enabled
    auto is_file_output_enabled() const -> bool;

    // flush the log file immediately
    void flush();

    //=============================================================================
    // ML-Specific Logging Methods
    //=============================================================================

    // Register a model for context tracking
    void register_model(const ModelContext& context);

    // Unregister a model from context tracking
    void unregister_model(const std::string& model_name);

    // Get model context by name
    auto get_model_context(const std::string& model_name) const -> std::optional<ModelContext>;

    // Update model stage (dev -> staging -> production)
    void update_model_stage(const std::string& model_name, ModelStage stage);

    // Log ML operation with model context
    template <typename... FormatArgs>
    void log_ml_operation(MLOperation operation,
                          const std::string& model_name,
                          const std::string& format = "",
                          const FormatArgs&... args) {
        if (!m_ml_logging_enabled_.load() || !is_level_enabled(LogLevel::INFO)) {
            return;
        }

        std::lock_guard<std::mutex> lock(m_ml_context_mutex_);
        auto context_iter = m_model_contexts_.find(model_name);

        std::ostringstream oss;
        oss << "[ML:" << ml_operation_to_string(operation) << "]";

        if (context_iter != m_model_contexts_.end()) {
            const auto& context = context_iter->second;
            oss << " model=" << context.name << " version=" << context.version
                << " stage=" << model_stage_to_string(context.stage)
                << " framework=" << context.framework;
        } else {
            oss << " model=" << model_name << " (unregistered)";
        }

        if (!format.empty()) {
            oss << " " << format_message(format, args...);
        }

        print_log(LogLevel::INFO, oss.str());
    }

    // Log inference metrics
    void log_inference_metrics(const std::string& model_name, const InferenceMetrics& metrics);

    // Log ML error with enhanced context
    void log_ml_error(const std::string& model_name,
                      const MLErrorContext& error_context,
                      const std::string& message = "");

    // Add metrics to buffer for batch processing
    void buffer_metrics(const InferenceMetrics& metrics);

    // Flush buffered metrics (e.g., for periodic reporting)
    void flush_metrics_buffer();

    // Get current metrics buffer size
    auto get_metrics_buffer_size() const -> std::size_t;

    // Set maximum metrics buffer size
    void set_max_metrics_buffer_size(std::size_t size);

    // Enable/disable ML-specific logging
    void set_ml_logging_enabled(bool enabled);

    // Check if ML logging is enabled
    auto is_ml_logging_enabled() const -> bool;

    // Get aggregate metrics for a time period
    auto get_aggregate_metrics(const std::string& model_name, std::chrono::minutes duration) const
        -> std::optional<InferenceMetrics>;

  private:
    std::ofstream m_log_file_{};
    mutable std::mutex m_file_mutex_{};  // separate mutex only for file I/O
    std::atomic<bool> m_stderr_enabled_{true};
    std::atomic<bool> m_file_output_enabled_{true};
    std::array<std::atomic<bool>, 6> m_enabled_levels_{
        {true, true, true, true, true, true}};  // one for each log level

    // ML-specific logging state
    mutable std::mutex m_ml_context_mutex_{};  // mutex for ML context operations
    std::unordered_map<std::string, ModelContext> m_model_contexts_{};  // registered models
    std::vector<InferenceMetrics> m_metrics_buffer_{};          // metrics for batch processing
    std::atomic<bool> m_ml_logging_enabled_{true};              // enable/disable ML logging
    std::atomic<std::size_t> m_max_metrics_buffer_size_{1000};  // max buffered metrics

    // utility function for expression tree visualization
    static auto get_indentation(int depth) -> std::string;

    // convert log level to string
    static auto log_level_to_string(LogLevel level) -> std::string;

    // get current utc timestamp
    static auto get_utc_timestamp() -> std::string;

    // ML-specific helper methods
    static auto ml_operation_to_string(MLOperation operation) -> std::string;
    static auto model_stage_to_string(ModelStage stage) -> std::string;
    auto format_inference_metrics(const InferenceMetrics& metrics) -> std::string;
    auto calculate_aggregate_metrics(const std::vector<InferenceMetrics>& metrics_list) const
        -> InferenceMetrics;

    // C++17 compatible format message helper
    template <typename... FormatArgs>
    static auto format_message(const std::string& format, const FormatArgs&... args)
        -> std::string {
        // Simple string substitution approach for C++17
        // For more complex formatting, consider using fmtlib or similar
        std::ostringstream oss;
        format_message_impl(oss, format, args...);
        return oss.str();
    }

    // Helper for formatting - base case
    static void format_message_impl(std::ostringstream& oss, const std::string& format) {
        oss << format;
    }

    // Helper for formatting - recursive case
    template <typename ValueType, typename... FormatArgs>
    static void format_message_impl(std::ostringstream& oss,
                                    const std::string& format,
                                    const ValueType& value,
                                    const FormatArgs&... args) {
        // Simple approach: replace first {} with the value
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            oss << format.substr(0, pos) << value;
            format_message_impl(oss, format.substr(pos + 2), args...);
        } else {
            oss << format;
        }
    }
};

// Modern C++17 logging interface - use inline functions instead of macros to avoid variadic issues
inline void LOG_BASE_PRINT(LogLevel level, const std::string& message) {
    Logger::get_instance().print_log(level, message);
}

template <typename... Args>
inline void LOG_BASE_PRINT(LogLevel level, const std::string& format, Args&&... args) {
    Logger::get_instance().print_log(level, format, std::forward<Args>(args)...);
}

#define LOG_INFO_PRINT(...) LOG_BASE_PRINT(LogLevel::INFO, __VA_ARGS__)
#define LOG_NORMAL_PRINT(...) LOG_BASE_PRINT(LogLevel::NORMAL, __VA_ARGS__)
#define LOG_WARNING_PRINT(...) LOG_BASE_PRINT(LogLevel::WARNING, __VA_ARGS__)
#define LOG_DEBUG_PRINT(...) LOG_BASE_PRINT(LogLevel::DEBUG, __VA_ARGS__)
#define LOG_ERROR_PRINT(...) LOG_BASE_PRINT(LogLevel::ERROR, __VA_ARGS__)
#define LOG_CRITICAL_PRINT(...) LOG_BASE_PRINT(LogLevel::CRITICAL, __VA_ARGS__)

//=============================================================================
// ML-Specific Logging Macros
//=============================================================================

// Log ML operations with model context
#define LOG_ML_OPERATION(operation, model_name, ...) \
    Logger::get_instance().log_ml_operation(operation, model_name, __VA_ARGS__)

// Log inference metrics
#define LOG_ML_METRICS(model_name, metrics) \
    Logger::get_instance().log_inference_metrics(model_name, metrics)

// Log ML errors
#define LOG_ML_ERROR(model_name, error_context, message) \
    Logger::get_instance().log_ml_error(model_name, error_context, message)

// Convenience macros for common ML operations
#define LOG_MODEL_LOAD(model_name, ...) \
    LOG_ML_OPERATION(MLOperation::MODEL_LOAD, model_name, __VA_ARGS__)

#define LOG_MODEL_UNLOAD(model_name, ...) \
    LOG_ML_OPERATION(MLOperation::MODEL_UNLOAD, model_name, __VA_ARGS__)

#define LOG_INFERENCE_START(model_name, ...) \
    LOG_ML_OPERATION(MLOperation::INFERENCE_START, model_name, __VA_ARGS__)

#define LOG_INFERENCE_COMPLETE(model_name, ...) \
    LOG_ML_OPERATION(MLOperation::INFERENCE_COMPLETE, model_name, __VA_ARGS__)

#define LOG_BATCH_PROCESS(model_name, ...) \
    LOG_ML_OPERATION(MLOperation::BATCH_PROCESS, model_name, __VA_ARGS__)

#define LOG_MODEL_VALIDATE(model_name, ...) \
    LOG_ML_OPERATION(MLOperation::MODEL_VALIDATE, model_name, __VA_ARGS__)

#define LOG_PERFORMANCE_BENCHMARK(model_name, ...) \
    LOG_ML_OPERATION(MLOperation::PERFORMANCE_BENCHMARK, model_name, __VA_ARGS__)

}  // namespace inference_lab::common
