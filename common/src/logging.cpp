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
    std::lock_guard<std::mutex> lock(m_mutex);

    // write to file if file logging is enabled for this level
    if (is_level_enabled(LEVEL) && m_file_output_enabled) {
        m_log_file << message;
        m_log_file.flush();
    }

    // write to console if console logging is enabled for this level
    if (is_level_enabled(LEVEL)) {
        if ((LEVEL == LogLevel::CRITICAL || LEVEL == LogLevel::ERROR) && m_stderr_enabled) {
            std::cerr << message;
        } else {  // info, normal, debug
            std::cout << message;
        }
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
    m_log_file.open(path, mode);
    if (!m_log_file.is_open()) {
        throw std::runtime_error("Failed to open log file: " + path);
    }

    // initialize enabled levels - all levels enabled by default
    for (int ndx = 0; ndx < static_cast<int>(LogLevel::CRITICAL) + 1; ++ndx) {
        m_enabled_levels[ndx] = true;
    }
}

Logger::StderrSuppressionGuard::StderrSuppressionGuard()
    : m_was_enabled(Logger::getInstance().isStderrEnabled()) {
    Logger::getInstance().disableStderr();
}

Logger::StderrSuppressionGuard::~StderrSuppressionGuard() {
    if (m_was_enabled) {
        Logger::getInstance().enableStderr();
    }
}

std::shared_ptr<Logger> Logger::getOrCreateInstance(const std::string& path, bool append) {
    std::lock_guard<std::mutex> lock(m_instance_mutex);
    if (!m_instance) {
        m_instance = std::shared_ptr<Logger>(new Logger(path, append));
    }
    return m_instance;
}

auto Logger::getInstance() -> Logger& {
    return *getOrCreateInstance();
}

auto Logger::getInstance(const std::string& custom_path, bool append) -> Logger& {
    return *getOrCreateInstance(custom_path, append);
}

std::shared_ptr<Logger> Logger::getInstancePtr() {
    return getOrCreateInstance();
}

std::shared_ptr<Logger> Logger::getInstancePtr(const std::string& custom_path, bool append) {
    return getOrCreateInstance(custom_path, append);
}

Logger::~Logger() {
    if (m_log_file.is_open()) {
        m_log_file.close();
    }
}

void Logger::setLevelEnabled(LogLevel level, bool enabled) {
    int const level_index = static_cast<int>(level);
    if (level_index >= 0 && level_index <= static_cast<int>(LogLevel::CRITICAL)) {
        getInstance().m_enabled_levels[level_index] = enabled;
    }
}

bool Logger::isLevelEnabled(LogLevel level) {
    return is_level_enabled(level);
}

void Logger::disableStderr() {
    m_stderr_enabled = false;
}

void Logger::enableStderr() {
    m_stderr_enabled = true;
}

auto Logger::isStderrEnabled() const -> bool {
    return m_stderr_enabled;
}

void Logger::setFileOutputEnabled(bool enabled) {
    m_file_output_enabled = enabled;
}

auto Logger::isFileOutputEnabled() const -> bool {
    return m_file_output_enabled;
}

bool Logger::is_level_enabled(LogLevel level) {
    if (const int LEVEL_INDEX = static_cast<int>(level);
        LEVEL_INDEX >= 0 && LEVEL_INDEX <= static_cast<int>(LogLevel::CRITICAL)) {
        return getInstance().m_enabled_levels[LEVEL_INDEX];
    }
    return false;
}

std::string Logger::getIndentation(const int DEPTH) {
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
