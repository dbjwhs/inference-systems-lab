// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

// Common logging utilities for the Inference Systems Laboratory
#pragma once

#include <fstream>
#include <mutex>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <thread>
#include <atomic>

namespace inference_lab {
namespace common {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    NORMAL = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
};

class Logger {
private:
    // shared_ptr to maintain the singleton
    inline static std::shared_ptr<Logger> m_instance;
    inline static std::mutex m_instance_mutex; // mutex for thread-safe initialization

    // helper method to handle the common logging logic
    void write_log_message(const LogLevel level, const std::string& message);

    // helper to build the log prefix
    static std::stringstream create_log_prefix(LogLevel level);

    // constructor is now private to control instantiation
    explicit Logger(const std::string& path, bool append = true);

public:
    // raii class for temporarily disabling stderr output
    class StderrSuppressionGuard {
    public:
        StderrSuppressionGuard();
        ~StderrSuppressionGuard();

    private:
        bool m_was_enabled;
    };

    // private method to get or create the instance
    static std::shared_ptr<Logger> getOrCreateInstance(const std::string& path = "../custom.log", bool append = true);

    // returns a reference for backward compatibility but uses shared_ptr internally
    static Logger& getInstance();

    // custom path version of getinstance
    static Logger& getInstance(const std::string& custom_path, bool append = true);

    // new method for code that explicitly wants to manage the shared_ptr
    static std::shared_ptr<Logger> getInstancePtr();

    // with custom path for the shared_ptr version
    static std::shared_ptr<Logger> getInstancePtr(const std::string& custom_path, bool append = true);

    // destructor
    ~Logger();

    // Template-based logging methods for C++17 compatibility
    template<typename... Args>
    void print_log(const LogLevel level, const std::string& format, Args&&... args) {
        if (!is_level_enabled(level)) {
            return;
        }
        auto prefix = create_log_prefix(level);
        auto formatted_message = format_message(format, std::forward<Args>(args)...);
        auto full_message = prefix.str() + formatted_message + "\n";
        write_log_message(level, full_message);
    }

    // Template-based logging with depth for C++17 compatibility
    template<typename... Args>
    void print_log_with_depth(const LogLevel level, const int depth, const std::string& format, Args&&... args) {
        if (!is_level_enabled(level)) {
            return;
        }
        auto prefix = create_log_prefix(level);
        auto formatted_message = format_message(format, std::forward<Args>(args)...);
        auto full_message = prefix.str() + getIndentation(depth) + formatted_message + "\n";
        write_log_message(level, full_message);
    }

    // enable/disable specific log level
    void setLevelEnabled(LogLevel level, bool enabled);

    // check if a specific log level is enabled
    bool isLevelEnabled(LogLevel level) const;

    // disable stderr output
    void disableStderr();

    // enable stderr output
    void enableStderr();

    // get current stderr output state
    bool isStderrEnabled() const;

    // enable/disable file output
    void setFileOutputEnabled(bool enabled);

    // check if file output is enabled
    bool isFileOutputEnabled() const;

private:
    std::ofstream m_log_file;
    std::mutex m_mutex;
    std::atomic<bool> m_stderr_enabled{true};
    std::atomic<bool> m_file_output_enabled{true};
    std::atomic<bool> m_enabled_levels[6]{true, true, true, true, true, true}; // one for each log level

    // check if a level is enabled (internal helper)
    bool is_level_enabled(LogLevel level) const;

    // utility function for expression tree visualization
    static std::string getIndentation(const int depth);

    // convert log level to string
    static std::string log_level_to_string(const LogLevel level);

    // get current utc timestamp
    static std::string get_utc_timestamp();

    // C++17 compatible format message helper
    template<typename... Args>
    static std::string format_message(const std::string& format, Args&&... args) {
        // Simple string substitution approach for C++17
        // For more complex formatting, consider using fmtlib or similar
        std::ostringstream oss;
        format_message_impl(oss, format, std::forward<Args>(args)...);
        return oss.str();
    }

    // Helper for formatting - base case
    static void format_message_impl(std::ostringstream& oss, const std::string& format) {
        oss << format;
    }

    // Helper for formatting - recursive case
    template<typename T, typename... Args>
    static void format_message_impl(std::ostringstream& oss, const std::string& format, T&& value, Args&&... args) {
        // Simple approach: replace first {} with the value
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            oss << format.substr(0, pos) << std::forward<T>(value);
            format_message_impl(oss, format.substr(pos + 2), std::forward<Args>(args)...);
        } else {
            oss << format;
        }
    }
};

// C++17 compatible template-based logging macros
#define LOG_BASE_PRINT(level, message, ...) Logger::getInstance().print_log(level, message, ##__VA_ARGS__)
#define LOG_INFO_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::INFO, message, ##__VA_ARGS__)
#define LOG_NORMAL_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::NORMAL, message, ##__VA_ARGS__)
#define LOG_WARNING_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::WARNING, message, ##__VA_ARGS__)
#define LOG_DEBUG_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::DEBUG, message, ##__VA_ARGS__)
#define LOG_ERROR_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::ERROR, message, ##__VA_ARGS__)
#define LOG_CRITICAL_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::CRITICAL, message, ##__VA_ARGS__)

} // namespace common
} // namespace inference_lab