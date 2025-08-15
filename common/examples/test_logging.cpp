// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

// Test program for the logging functionality

#include "../src/logging.hpp"
#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <vector>

using namespace inference_lab::common;

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
    
    auto& logger = Logger::getInstance();
    
    // Disable debug and info levels
    logger.setLevelEnabled(LogLevel::DEBUG, false);
    logger.setLevelEnabled(LogLevel::INFO, false);
    
    std::cout << "Debug and Info disabled - these should not appear:\n";
    LOG_DEBUG_PRINT("This debug message should NOT appear");
    LOG_INFO_PRINT("This info message should NOT appear");
    LOG_NORMAL_PRINT("This normal message SHOULD appear");
    
    // Re-enable them
    logger.setLevelEnabled(LogLevel::DEBUG, true);
    logger.setLevelEnabled(LogLevel::INFO, true);
    
    std::cout << "Debug and Info re-enabled - these should appear:\n";
    LOG_DEBUG_PRINT("This debug message should appear now");
    LOG_INFO_PRINT("This info message should appear now");
}

void test_stderr_suppression() {
    std::cout << "\n=== Testing Stderr Suppression ===\n";
    
    std::cout << "Testing stderr suppression guard:\n";
    {
        Logger::StderrSuppressionGuard guard;
        LOG_ERROR_PRINT("This error should only go to stdout, not stderr");
        LOG_CRITICAL_PRINT("This critical should only go to stdout, not stderr");
    }
    
    std::cout << "Stderr re-enabled:\n";
    LOG_ERROR_PRINT("This error should go to stderr again");
}

void test_file_output_control() {
    std::cout << "\n=== Testing File Output Control ===\n";
    
    auto& logger = Logger::getInstance();
    
    // Disable file output
    logger.setFileOutputEnabled(false);
    std::cout << "File output disabled - this message only goes to console:\n";
    LOG_NORMAL_PRINT("Console only message");
    
    // Re-enable file output
    logger.setFileOutputEnabled(true);
    std::cout << "File output re-enabled - this message goes to both:\n";
    LOG_NORMAL_PRINT("Console and file message");
}

void test_depth_logging() {
    std::cout << "\n=== Testing Depth Logging ===\n";
    
    auto& logger = Logger::getInstance();
    
    logger.print_log_with_depth(LogLevel::INFO, 0, "Root level message");
    logger.print_log_with_depth(LogLevel::INFO, 1, "Depth 1 message");
    logger.print_log_with_depth(LogLevel::INFO, 2, "Depth 2 message");
    logger.print_log_with_depth(LogLevel::INFO, 3, "Depth 3 message");
}

void test_multithreaded_logging() {
    std::cout << "\n=== Testing Multithreaded Logging ===\n";
    
    std::vector<std::thread> threads;
    
    // Create multiple threads that log simultaneously
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([i]() {
            for (int j = 0; j < 3; ++j) {
                LOG_INFO_PRINT("Thread {} message {}", i, j);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
}

int main() {
    try {
        std::cout << "Starting logging system tests...\n";
        
        // Initialize logger with custom path and truncate mode for clean test start
        const std::string log_file = "../test_custom.log";
        auto& logger = Logger::getInstance(log_file, false); // false = truncate existing file
        
        std::cout << "Logger initialized successfully\n";
        std::cout << "Log file: " << log_file << "\n";
        std::cout << "File output enabled: " << (logger.isFileOutputEnabled() ? "Yes" : "No") << "\n";
        std::cout << "Stderr enabled: " << (logger.isStderrEnabled() ? "Yes" : "No") << "\n";
        
        // Run tests
        test_basic_logging();
        test_level_control();
        test_stderr_suppression();
        test_file_output_control();
        test_depth_logging();
        test_multithreaded_logging();
        
        std::cout << "\n=== All Tests Completed ===\n";
        std::cout << "Check the log file: " << log_file << "\n";
        
        // Show log file contents if it exists
        if (std::filesystem::exists(log_file)) {
            std::cout << "\nLog file size: " << std::filesystem::file_size(log_file) << " bytes\n";
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}