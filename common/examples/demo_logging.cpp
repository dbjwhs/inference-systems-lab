// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

// Demo program for the logging functionality

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

#include "../src/logging.hpp"

using inference_lab::common::Logger;
using inference_lab::common::LogLevel;

namespace {

void test_basic_logging() {
    std::cout << "\n=== Testing Basic Logging ===\n";

    // Test all log levels
    LOG_DEBUG_PRINT("This is a debug message: {}", 42);
    LOG_INFO_PRINT("This is an info message: {}", "hello");
    LOG_NORMAL_PRINT("This is a normal message");
    LOG_WARNING_PRINT("This is a warning message: {:.2f}", 3.14159);
    LOG_ERROR_PRINT("This is an error message: {}", std::string("error"));
    LOG_CRITICAL_PRINT("This is a critical message");
}

void test_level_control() {
    std::cout << "\n=== Testing Level Control ===\n";

    [[maybe_unused]] auto& logger = Logger::get_instance();

    // Disable debug and info levels
    Logger::set_level_enabled(LogLevel::DEBUG, false);
    Logger::set_level_enabled(LogLevel::INFO, false);

    std::cout << "Debug and Info disabled - these should not appear:\n";
    LOG_DEBUG_PRINT("This debug message should NOT appear");
    LOG_INFO_PRINT("This info message should NOT appear");
    LOG_NORMAL_PRINT("This normal message SHOULD appear");

    // Re-enable them
    Logger::set_level_enabled(LogLevel::DEBUG, true);
    Logger::set_level_enabled(LogLevel::INFO, true);

    std::cout << "Debug and Info re-enabled - these should appear:\n";
    LOG_DEBUG_PRINT("This debug message should appear now");
    LOG_INFO_PRINT("This info message should appear now");
}

void test_stderr_suppression() {
    std::cout << "\n=== Testing Stderr Suppression ===\n";

    std::cout << "Testing stderr suppression guard:\n";
    {
        Logger::StderrSuppressionGuard const GUARD;
        LOG_ERROR_PRINT("This error should only go to stdout, not stderr");
        LOG_CRITICAL_PRINT("This critical should only go to stdout, not stderr");
    }

    std::cout << "Stderr re-enabled:\n";
    LOG_ERROR_PRINT("This error should go to stderr again");
}

void test_file_output_control() {
    std::cout << "\n=== Testing File Output Control ===\n";

    [[maybe_unused]] auto& logger = Logger::get_instance();

    // Disable file output
    logger.set_file_output_enabled(false);
    std::cout << "File output disabled - this message only goes to console:\n";
    LOG_NORMAL_PRINT("Console only message");

    // Re-enable file output
    logger.set_file_output_enabled(true);
    std::cout << "File output re-enabled - this message goes to both:\n";
    LOG_NORMAL_PRINT("Console and file message");
}

void test_depth_logging() {
    std::cout << "\n=== Testing Depth Logging ===\n";

    [[maybe_unused]] auto& logger = Logger::get_instance();

    logger.print_log_with_depth(LogLevel::INFO, 0, "Root level message");
    logger.print_log_with_depth(LogLevel::INFO, 1, "Depth 1 message");
    logger.print_log_with_depth(LogLevel::INFO, 2, "Depth 2 message");
    logger.print_log_with_depth(LogLevel::INFO, 3, "Depth 3 message");
}

void test_multithreaded_logging() {
    std::cout << "\n=== Testing Multithreaded Logging ===\n";

    std::vector<std::thread> threads;

    // Create multiple threads that log simultaneously
    for (int ndx = 0; ndx < 5; ++ndx) {
        threads.emplace_back([ndx]() {
            for (int jdx = 0; jdx < 3; ++jdx) {
                LOG_INFO_PRINT("Thread {} message {}", ndx, jdx);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thrd : threads) {
        thrd.join();
    }
}

}  // anonymous namespace

auto main() -> int {
    try {
        std::cout << "Starting logging system tests...\n";

        // Initialize logger with custom path and truncate mode for clean test start
        const std::string LOG_FILE = "../test_custom.log";
        const auto& logger =
            Logger::get_instance(LOG_FILE, false);  // false = truncate existing file

        std::cout << "Logger initialized successfully\n";
        std::cout << "Log file: " << LOG_FILE << "\n";
        std::cout << "File output enabled: " << (logger.is_file_output_enabled() ? "Yes" : "No")
                  << "\n";
        std::cout << "Stderr enabled: " << (logger.is_stderr_enabled() ? "Yes" : "No") << "\n";

        // Run tests
        test_basic_logging();
        test_level_control();
        test_stderr_suppression();
        test_file_output_control();
        test_depth_logging();
        test_multithreaded_logging();

        std::cout << "\n=== All Tests Completed ===\n";
        std::cout << "Check the log file: " << LOG_FILE << "\n";

        // Show log file contents if it exists
        if (std::filesystem::exists(LOG_FILE)) {
            std::cout << "\nLog file size: " << std::filesystem::file_size(LOG_FILE) << " bytes\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
