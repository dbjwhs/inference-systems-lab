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

// Common logging utilities for the Inference Systems Laboratory

#include "logging.hpp"

#include <cstdlib>  // for std::getenv

namespace inference_lab::common {

// Logger implementation

void Logger::write_log_message(const LogLevel LEVEL, const std::string& message) {
    // Fast atomic checks first - no global lock needed
    const bool level_enabled = is_level_enabled(LEVEL);
    if (!level_enabled) {
        return;  // Early exit if level disabled
    }

    // File output with dedicated mutex only when needed
    if (m_file_output_enabled_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> file_lock(m_file_mutex_);
        m_log_file_ << message;
        m_log_file_.flush();
    }

    // Console output - inherently thread-safe for single writes
    if ((LEVEL == LogLevel::CRITICAL || LEVEL == LogLevel::ERROR)) {
        if (m_stderr_enabled_.load(std::memory_order_acquire)) {
            std::cerr << message;  // stderr for errors
        }
    } else {
        std::cout << message;  // stdout for info/debug/normal
    }
}

std::stringstream Logger::create_log_prefix(LogLevel level) {
    std::stringstream message;
    message << get_utc_timestamp() << " [" << log_level_to_string(level)
            << "] [Thread:" << std::this_thread::get_id() << "] ";
    return message;
}

Logger::Logger(const std::string& path, bool append) {
    if (!std::filesystem::exists(std::filesystem::path(path).parent_path())) {
        throw std::runtime_error("Invalid path provided: " + path);
    }

    // Choose file open mode based on append parameter
    std::ios::openmode mode = append ? std::ios::app : std::ios::out;
    m_log_file_.open(path, mode);
    if (!m_log_file_.is_open()) {
        throw std::runtime_error("Failed to open log file: " + path);
    }

    // initialize enabled levels - all levels enabled by default
    for (int ndx = 0; ndx < static_cast<int>(LogLevel::CRITICAL) + 1; ++ndx) {
        m_enabled_levels_[ndx].store(true, std::memory_order_relaxed);
    }

    // Check environment variable for console output suppression (for JSON benchmarks)
    // LOG_QUIET=1 disables all console output while preserving file logging
    const char* quiet_env = std::getenv("LOG_QUIET");
    if (quiet_env && std::string(quiet_env) == "1") {
        // Disable all log levels for console output only (file logging still works)
        for (int ndx = 0; ndx < static_cast<int>(LogLevel::CRITICAL) + 1; ++ndx) {
            m_enabled_levels_[ndx].store(false, std::memory_order_relaxed);
        }
    }
}

Logger::StderrSuppressionGuard::StderrSuppressionGuard()
    : m_was_enabled_(Logger::get_instance().is_stderr_enabled()) {
    Logger::get_instance().disable_stderr();
}

Logger::StderrSuppressionGuard::~StderrSuppressionGuard() {
    if (m_was_enabled_) {
        Logger::get_instance().enable_stderr();
    }
}

std::shared_ptr<Logger> Logger::get_or_create_instance(const std::string& path, bool append) {
    std::lock_guard<std::mutex> lock(m_instance_mutex);
    if (!m_instance) {
        m_instance = std::shared_ptr<Logger>(new Logger(path, append));
    }
    return m_instance;
}

auto Logger::get_instance() -> Logger& {
    return *get_or_create_instance();
}

auto Logger::get_instance(const std::string& custom_path, bool append) -> Logger& {
    return *get_or_create_instance(custom_path, append);
}

std::shared_ptr<Logger> Logger::get_instance_ptr() {
    return get_or_create_instance();
}

std::shared_ptr<Logger> Logger::get_instance_ptr(const std::string& custom_path, bool append) {
    return get_or_create_instance(custom_path, append);
}

Logger::~Logger() {
    if (m_log_file_.is_open()) {
        m_log_file_.close();
    }
}

void Logger::set_level_enabled(LogLevel level, bool enabled) {
    int const LEVEL_INDEX = static_cast<int>(level);
    if (LEVEL_INDEX >= 0 && LEVEL_INDEX <= static_cast<int>(LogLevel::CRITICAL)) {
        get_instance().m_enabled_levels_[LEVEL_INDEX].store(enabled, std::memory_order_release);
    }
}

void Logger::disable_stderr() {
    m_stderr_enabled_.store(false, std::memory_order_release);
}

void Logger::enable_stderr() {
    m_stderr_enabled_.store(true, std::memory_order_release);
}

auto Logger::is_stderr_enabled() const -> bool {
    return m_stderr_enabled_.load(std::memory_order_acquire);
}

void Logger::set_file_output_enabled(bool enabled) {
    m_file_output_enabled_.store(enabled, std::memory_order_release);
}

auto Logger::is_file_output_enabled() const -> bool {
    return m_file_output_enabled_.load(std::memory_order_acquire);
}

void Logger::flush() {
    std::lock_guard<std::mutex> file_lock(m_file_mutex_);
    if (m_log_file_.is_open()) {
        m_log_file_.flush();
    }
}

auto Logger::is_level_enabled(LogLevel level) -> bool {
    if (const int LEVEL_INDEX = static_cast<int>(level);
        LEVEL_INDEX >= 0 && LEVEL_INDEX <= static_cast<int>(LogLevel::CRITICAL)) {
        return get_instance().m_enabled_levels_[LEVEL_INDEX].load(std::memory_order_acquire);
    }
    return false;
}

std::string Logger::get_indentation(const int DEPTH) {
    return std::string(DEPTH * 2, ' ');
}

std::string Logger::log_level_to_string(const LogLevel LEVEL) {
    switch (LEVEL) {
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::NORMAL:
            return "NORMAL";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::ERROR:
            return "ERROR";
        case LogLevel::CRITICAL:
            return "CRITICAL";
        default:
            return "UNKNOWN";
    }
}

std::string Logger::get_utc_timestamp() {
    const auto NOW = std::chrono::system_clock::now();
    const auto MS =
        std::chrono::duration_cast<std::chrono::milliseconds>(NOW.time_since_epoch()) % 1000;
    auto time = std::chrono::system_clock::to_time_t(NOW);

    struct tm tm_buf;
#ifdef _WIN32
    // ### not tested (did work a few years ago)
    gmtime_s(&tm_buf, &time);
#else
    if (gmtime_r(&time, &tm_buf) == nullptr) {
        // Fall back to a simple timestamp if gmtime_r fails
        return "TIMESTAMP_ERROR";
    }
#endif

    std::stringstream ss;
    ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << MS.count() << " UTC";
    return ss.str();
}

//=============================================================================
// ML-Specific Logging Method Implementations
//=============================================================================

void Logger::register_model(const ModelContext& context) {
    std::lock_guard<std::mutex> lock(m_ml_context_mutex_);
    m_model_contexts_[context.name] = context;

    LOG_INFO_PRINT("Registered ML model: name={} version={} framework={} stage={} size={}MB",
                   context.name,
                   context.version,
                   context.framework,
                   model_stage_to_string(context.stage),
                   context.size_mb);
}

void Logger::unregister_model(const std::string& model_name) {
    std::lock_guard<std::mutex> lock(m_ml_context_mutex_);
    auto it = m_model_contexts_.find(model_name);
    if (it != m_model_contexts_.end()) {
        LOG_INFO_PRINT("Unregistered ML model: name={}", model_name);
        m_model_contexts_.erase(it);
    }
}

auto Logger::get_model_context(const std::string& model_name) const -> std::optional<ModelContext> {
    std::lock_guard<std::mutex> lock(m_ml_context_mutex_);
    auto it = m_model_contexts_.find(model_name);
    if (it != m_model_contexts_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void Logger::update_model_stage(const std::string& model_name, ModelStage stage) {
    std::lock_guard<std::mutex> lock(m_ml_context_mutex_);
    auto it = m_model_contexts_.find(model_name);
    if (it != m_model_contexts_.end()) {
        auto old_stage = it->second.stage;
        it->second.stage = stage;
        LOG_INFO_PRINT("Updated model stage: name={} {} -> {}",
                       model_name,
                       model_stage_to_string(old_stage),
                       model_stage_to_string(stage));
    }
}

void Logger::log_inference_metrics(const std::string& model_name, const InferenceMetrics& metrics) {
    if (!m_ml_logging_enabled_.load() || !is_level_enabled(LogLevel::INFO)) {
        return;
    }

    auto formatted_metrics = format_inference_metrics(metrics);
    LOG_INFO_PRINT("[ML:METRICS] model={} {}", model_name, formatted_metrics);
}

void Logger::log_ml_error(const std::string& model_name,
                          const MLErrorContext& error_context,
                          const std::string& message) {
    if (!m_ml_logging_enabled_.load() || !is_level_enabled(LogLevel::ERROR)) {
        return;
    }

    std::ostringstream oss;
    oss << "[ML:ERROR] model=" << model_name << " component=" << error_context.component
        << " operation=" << error_context.operation << " error_code=" << error_context.error_code;

    if (!error_context.metadata.empty()) {
        oss << " metadata={";
        bool first = true;
        for (const auto& [key, value] : error_context.metadata) {
            if (!first)
                oss << ", ";
            oss << key << "=" << value;
            first = false;
        }
        oss << "}";
    }

    if (!message.empty()) {
        oss << " message=" << message;
    }

    LOG_ERROR_PRINT(oss.str());
}

void Logger::buffer_metrics(const InferenceMetrics& metrics) {
    std::lock_guard<std::mutex> lock(m_ml_context_mutex_);

    // Check buffer size limit
    if (m_metrics_buffer_.size() >= m_max_metrics_buffer_size_.load()) {
        // Remove oldest metrics (FIFO)
        m_metrics_buffer_.erase(m_metrics_buffer_.begin());
    }

    m_metrics_buffer_.push_back(metrics);
}

void Logger::flush_metrics_buffer() {
    std::lock_guard<std::mutex> lock(m_ml_context_mutex_);

    if (m_metrics_buffer_.empty()) {
        return;
    }

    // Calculate aggregate metrics
    auto aggregate = calculate_aggregate_metrics(m_metrics_buffer_);

    LOG_INFO_PRINT(
        "[ML:AGGREGATE] buffered_samples={} avg_latency={:.2f}ms avg_throughput={:.2f} "
        "avg_confidence={:.3f}",
        m_metrics_buffer_.size(),
        aggregate.latency_ms,
        aggregate.throughput,
        aggregate.confidence);

    // Clear buffer
    m_metrics_buffer_.clear();
}

auto Logger::get_metrics_buffer_size() const -> std::size_t {
    std::lock_guard<std::mutex> lock(m_ml_context_mutex_);
    return m_metrics_buffer_.size();
}

void Logger::set_max_metrics_buffer_size(std::size_t size) {
    m_max_metrics_buffer_size_.store(size);
}

void Logger::set_ml_logging_enabled(bool enabled) {
    m_ml_logging_enabled_.store(enabled);
    LOG_INFO_PRINT("ML logging {}", enabled ? "enabled" : "disabled");
}

auto Logger::is_ml_logging_enabled() const -> bool {
    return m_ml_logging_enabled_.load();
}

auto Logger::get_aggregate_metrics(const std::string& model_name,
                                   std::chrono::minutes duration) const
    -> std::optional<InferenceMetrics> {
    // For now, return aggregate of all buffered metrics
    // In a full implementation, this would filter by timestamp and model
    [[maybe_unused]] const auto& name = model_name;       // Future use for filtering
    [[maybe_unused]] const auto& time_window = duration;  // Future use for time filtering

    std::lock_guard<std::mutex> lock(m_ml_context_mutex_);

    if (m_metrics_buffer_.empty()) {
        return std::nullopt;
    }

    return calculate_aggregate_metrics(m_metrics_buffer_);
}

//=============================================================================
// ML Helper Method Implementations
//=============================================================================

auto Logger::ml_operation_to_string(MLOperation operation) -> std::string {
    switch (operation) {
        case MLOperation::MODEL_LOAD:
            return "MODEL_LOAD";
        case MLOperation::MODEL_UNLOAD:
            return "MODEL_UNLOAD";
        case MLOperation::INFERENCE_START:
            return "INFERENCE_START";
        case MLOperation::INFERENCE_COMPLETE:
            return "INFERENCE_COMPLETE";
        case MLOperation::BATCH_PROCESS:
            return "BATCH_PROCESS";
        case MLOperation::MODEL_VALIDATE:
            return "MODEL_VALIDATE";
        case MLOperation::PERFORMANCE_BENCHMARK:
            return "PERFORMANCE_BENCHMARK";
        case MLOperation::ERROR_OCCURRED:
            return "ERROR_OCCURRED";
        default:
            return "UNKNOWN";
    }
}

auto Logger::model_stage_to_string(ModelStage stage) -> std::string {
    switch (stage) {
        case ModelStage::DEVELOPMENT:
            return "DEVELOPMENT";
        case ModelStage::STAGING:
            return "STAGING";
        case ModelStage::PRODUCTION:
            return "PRODUCTION";
        case ModelStage::ARCHIVED:
            return "ARCHIVED";
        case ModelStage::DEPRECATED:
            return "DEPRECATED";
        default:
            return "UNKNOWN";
    }
}

auto Logger::format_inference_metrics(const InferenceMetrics& metrics) -> std::string {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "latency=" << metrics.latency_ms << "ms"
        << " preprocessing=" << metrics.preprocessing_ms << "ms"
        << " inference=" << metrics.inference_ms << "ms"
        << " postprocessing=" << metrics.postprocessing_ms << "ms"
        << " memory=" << metrics.memory_mb << "MB"
        << " batch_size=" << metrics.batch_size << " throughput=" << metrics.throughput << "/s"
        << " confidence=" << std::setprecision(3) << metrics.confidence
        << " device=" << metrics.device;
    return oss.str();
}

auto Logger::calculate_aggregate_metrics(const std::vector<InferenceMetrics>& metrics_list) const
    -> InferenceMetrics {
    if (metrics_list.empty()) {
        return {};
    }

    InferenceMetrics aggregate{};
    double total_throughput = 0.0;
    double total_confidence = 0.0;
    std::size_t total_memory = 0;
    std::size_t total_batch_size = 0;

    for (const auto& m : metrics_list) {
        aggregate.latency_ms += m.latency_ms;
        aggregate.preprocessing_ms += m.preprocessing_ms;
        aggregate.inference_ms += m.inference_ms;
        aggregate.postprocessing_ms += m.postprocessing_ms;
        total_throughput += m.throughput;
        total_confidence += m.confidence;
        total_memory += m.memory_mb;
        total_batch_size += m.batch_size;
    }

    const auto count = static_cast<double>(metrics_list.size());
    aggregate.latency_ms /= count;
    aggregate.preprocessing_ms /= count;
    aggregate.inference_ms /= count;
    aggregate.postprocessing_ms /= count;
    aggregate.throughput = total_throughput / count;
    aggregate.confidence = total_confidence / count;
    aggregate.memory_mb = total_memory / metrics_list.size();
    aggregate.batch_size = total_batch_size / metrics_list.size();

    // Use device from first metric (could be improved)
    if (!metrics_list.empty()) {
        aggregate.device = metrics_list[0].device;
    }

    return aggregate;
}

}  // namespace inference_lab::common
