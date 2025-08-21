// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

// Common logging utilities for the Inference Systems Laboratory

#include "logging.hpp"

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

}  // namespace inference_lab::common
