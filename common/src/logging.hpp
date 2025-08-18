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
#include <sstream>
#include <string>
#include <thread>

namespace inference_lab::common {

enum class LogLevel : std::uint8_t {
    DEBUG = 0,
    INFO = 1,
    NORMAL = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
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

  private:
    std::ofstream m_log_file_{};
    std::mutex m_mutex_{};
    std::atomic<bool> m_stderr_enabled_{true};
    std::atomic<bool> m_file_output_enabled_{true};
    std::array<std::atomic<bool>, 6> m_enabled_levels_{
        {true, true, true, true, true, true}};  // one for each log level

    // utility function for expression tree visualization
    static auto get_indentation(int depth) -> std::string;

    // convert log level to string
    static auto log_level_to_string(LogLevel level) -> std::string;

    // get current utc timestamp
    static auto get_utc_timestamp() -> std::string;

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

// C++17 compatible template-based logging macros
// Accept GNU extension for broader compatibility but suppress warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

#define LOG_BASE_PRINT(level, message, ...) \
    Logger::get_instance().print_log(level, message, ##__VA_ARGS__)
#define LOG_INFO_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::INFO, message, ##__VA_ARGS__)
#define LOG_NORMAL_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::NORMAL, message, ##__VA_ARGS__)
#define LOG_WARNING_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::WARNING, message, ##__VA_ARGS__)
#define LOG_DEBUG_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::DEBUG, message, ##__VA_ARGS__)
#define LOG_ERROR_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::ERROR, message, ##__VA_ARGS__)
#define LOG_CRITICAL_PRINT(message, ...) LOG_BASE_PRINT(LogLevel::CRITICAL, message, ##__VA_ARGS__)

#pragma GCC diagnostic pop

}  // namespace inference_lab::common
