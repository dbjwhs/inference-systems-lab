// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

// Unit tests for the logging functionality

#include <gtest/gtest.h>
#include "../src/logging.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include <vector>
#include <sstream>

using namespace inference_lab::common;

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use a consistent test log file for all tests
        // Since Logger is a singleton, we need to work with the same instance
        test_log_file = "./test_unit_logging.log";
        
        // Clean up any existing test log file
        if (std::filesystem::exists(test_log_file)) {
            std::filesystem::remove(test_log_file);
        }
        
        // Initialize the singleton Logger with our test file and truncate mode
        // This creates the singleton instance that all tests will use
        Logger::getInstance(test_log_file, false);
    }

    void TearDown() override {
        // Clean up test log files after each test
        if (std::filesystem::exists(test_log_file)) {
            std::filesystem::remove(test_log_file);
        }
    }

    std::string test_log_file;
    
    std::string readLogFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return "";
        return std::string((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
    }
    
    void clearLogFile() {
        // Clear the log file by truncating it
        if (std::filesystem::exists(test_log_file)) {
            std::ofstream file(test_log_file, std::ios::trunc);
            file.close();
        }
    }
};

TEST_F(LoggerTest, SingletonBehavior) {
    // Test that getInstance returns the same instance
    auto& logger1 = Logger::getInstance();
    auto& logger2 = Logger::getInstance();
    
    EXPECT_EQ(&logger1, &logger2);
    
    // Test shared_ptr version
    const auto ptr1 = Logger::getInstancePtr();
    const auto ptr2 = Logger::getInstancePtr();
    
    EXPECT_EQ(ptr1, ptr2);
}

TEST_F(LoggerTest, BasicLoggingFunctionality) {
    // Test that logger basic functionality works
    auto& logger = Logger::getInstance();
    
    // Test that logging methods don't crash and work correctly
    EXPECT_NO_THROW(logger.print_log(LogLevel::INFO, "Test message"));
    EXPECT_NO_THROW(logger.print_log(LogLevel::WARNING, "Number: {}", 42));
    EXPECT_NO_THROW(logger.print_log_with_depth(LogLevel::DEBUG, 1, "Depth test"));
    
    // Test file output state can be controlled
    EXPECT_TRUE(logger.isFileOutputEnabled());
    logger.setFileOutputEnabled(false);
    EXPECT_FALSE(logger.isFileOutputEnabled());
    logger.setFileOutputEnabled(true);
    EXPECT_TRUE(logger.isFileOutputEnabled());
}

TEST_F(LoggerTest, LogLevelControl) {
    auto& logger = Logger::getInstance();
    
    // Test initial state - all levels should be enabled
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::DEBUG));
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::INFO));
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::NORMAL));
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::WARNING));
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::ERROR));
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::CRITICAL));
    
    // Disable specific levels
    logger.setLevelEnabled(LogLevel::DEBUG, false);
    logger.setLevelEnabled(LogLevel::INFO, false);
    
    EXPECT_FALSE(logger.isLevelEnabled(LogLevel::DEBUG));
    EXPECT_FALSE(logger.isLevelEnabled(LogLevel::INFO));
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::NORMAL));
    
    // Re-enable levels for other tests
    logger.setLevelEnabled(LogLevel::DEBUG, true);
    logger.setLevelEnabled(LogLevel::INFO, true);
    
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::DEBUG));
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::INFO));
}

TEST_F(LoggerTest, FileOutputControl) {
    auto& logger = Logger::getInstance();
    
    // Test initial state
    EXPECT_TRUE(logger.isFileOutputEnabled());
    
    // Test disable/enable
    logger.setFileOutputEnabled(false);
    EXPECT_FALSE(logger.isFileOutputEnabled());
    
    logger.setFileOutputEnabled(true);
    EXPECT_TRUE(logger.isFileOutputEnabled());
}

TEST_F(LoggerTest, StderrControl) {
    auto& logger = Logger::getInstance();
    
    // Test initial state
    EXPECT_TRUE(logger.isStderrEnabled());
    
    // Test disable/enable
    logger.disableStderr();
    EXPECT_FALSE(logger.isStderrEnabled());
    
    logger.enableStderr();
    EXPECT_TRUE(logger.isStderrEnabled());
}

TEST_F(LoggerTest, StderrSuppressionGuard) {
    auto& logger = Logger::getInstance();
    
    // Initial state should be enabled
    EXPECT_TRUE(logger.isStderrEnabled());
    
    {
        Logger::StderrSuppressionGuard guard;
        EXPECT_FALSE(logger.isStderrEnabled());
    }
    
    // Should be restored after guard destruction
    EXPECT_TRUE(logger.isStderrEnabled());
}

TEST_F(LoggerTest, ThreadSafety) {
    auto& logger = Logger::getInstance();
    
    std::vector<std::thread> threads;
    const int num_threads = 5;
    const int messages_per_thread = 10;
    std::atomic<int> successful_logs{0};
    
    // Create multiple threads that log simultaneously
    threads.reserve(num_threads);
    for (int ndx = 0; ndx < num_threads; ++ndx) {
            threads.emplace_back([&logger, &successful_logs, ndx, messages_per_thread]() {
                for (int j = 0; j < messages_per_thread; ++j) {
                    try {
                        logger.print_log(LogLevel::INFO, "Thread {} message {}", ndx, j);
                        ++successful_logs;
                    } catch (...) {
                        // If any thread crashes, the test should fail
                        FAIL() << "Thread " << ndx << " crashed during logging";
                    }
                }
            });
        }
    
    // Wait for all threads to complete
    for (auto& thrds : threads) {
        thrds.join();
    }
    
    // Verify all messages were logged successfully (no crashes)
    EXPECT_EQ(successful_logs.load(), num_threads * messages_per_thread);
}

TEST_F(LoggerTest, LogFormatting) {
    auto& logger = Logger::getInstance();
    
    // Test various formatting - verify they don't crash with complex formats
    EXPECT_NO_THROW(logger.print_log(LogLevel::INFO, "Simple message"));
    EXPECT_NO_THROW(logger.print_log(LogLevel::WARNING, "Number: {}", 42));
    EXPECT_NO_THROW(logger.print_log(LogLevel::ERROR, "Float: {:.2f}", 3.14159));
    EXPECT_NO_THROW(logger.print_log(LogLevel::DEBUG, "String: {}", std::string("test")));
    EXPECT_NO_THROW(logger.print_log(LogLevel::CRITICAL, "Multiple args: {} {} {}", 1, 2.5, "three"));
    
    // Test depth logging
    EXPECT_NO_THROW(logger.print_log_with_depth(LogLevel::INFO, 0, "Root"));
    EXPECT_NO_THROW(logger.print_log_with_depth(LogLevel::INFO, 3, "Deep"));
}

TEST_F(LoggerTest, MacroFunctionality) {
    // Test that macros work without crashing
    EXPECT_NO_THROW(LOG_DEBUG_PRINT("Debug test: {}", 1));
    EXPECT_NO_THROW(LOG_INFO_PRINT("Info test: {}", 2));
    EXPECT_NO_THROW(LOG_NORMAL_PRINT("Normal test: {}", 3));
    EXPECT_NO_THROW(LOG_WARNING_PRINT("Warning test: {}", 4));
    EXPECT_NO_THROW(LOG_ERROR_PRINT("Error test: {}", 5));
    EXPECT_NO_THROW(LOG_CRITICAL_PRINT("Critical test: {}", 6));
    
    // Test macros with various argument types
    EXPECT_NO_THROW(LOG_INFO_PRINT("String: {}, Number: {}, Float: {:.2f}", "test", 42, 3.14));
}

TEST_F(LoggerTest, LevelFiltering) {
    auto& logger = Logger::getInstance();
    
    // Test that disabled levels don't cause crashes
    logger.setLevelEnabled(LogLevel::DEBUG, false);
    EXPECT_NO_THROW(LOG_DEBUG_PRINT("This debug should be filtered"));
    EXPECT_FALSE(logger.isLevelEnabled(LogLevel::DEBUG));
    
    logger.setLevelEnabled(LogLevel::INFO, false);
    EXPECT_NO_THROW(LOG_INFO_PRINT("This info should be filtered"));
    EXPECT_FALSE(logger.isLevelEnabled(LogLevel::INFO));
    
    // Re-enable for other tests
    logger.setLevelEnabled(LogLevel::DEBUG, true);
    logger.setLevelEnabled(LogLevel::INFO, true);
    
    // Test that enabled levels work
    EXPECT_NO_THROW(LOG_DEBUG_PRINT("This debug should work"));
    EXPECT_NO_THROW(LOG_INFO_PRINT("This info should work"));
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::DEBUG));
    EXPECT_TRUE(logger.isLevelEnabled(LogLevel::INFO));
}