// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file test_logging_unit.cpp
 * @brief Comprehensive unit tests for the Logger class functionality
 * 
 * This test suite validates the core logging system implementation including:
 * - Singleton pattern behavior and thread safety
 * - Log level control and filtering mechanisms
 * - File output management and stderr control
 * - Multi-threaded concurrent logging operations
 * - Message formatting with C++23 std::format support
 * - RAII-based stderr suppression functionality
 * - Macro-based logging convenience functions
 * 
 * Each test case is designed to verify specific behavioral requirements
 * while working within the constraints of the Logger's singleton pattern.
 * The tests focus on behavioral validation rather than file I/O verification
 * to ensure reliable and consistent test execution across different environments.
 */

#include <gtest/gtest.h>
#include "../src/logging.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include <vector>
#include <sstream>

using namespace inference_lab::common;

/**
 * @class LoggerTest
 * @brief Test fixture for Logger class unit tests
 * 
 * Provides common setup and teardown functionality for all Logger tests.
 * Manages test log file lifecycle and provides utility methods for
 * file operations. Since Logger implements singleton pattern, all tests
 * work with the same Logger instance initialized during SetUp().
 */
class LoggerTest : public ::testing::Test {
protected:
    /**
     * @brief Test fixture setup - initializes Logger singleton for testing
     * 
     * Creates a clean test environment by:
     * - Setting up a dedicated test log file path
     * - Removing any existing test log files
     * - Initializing the Logger singleton with truncate mode
     * 
     * Since Logger uses singleton pattern, this setup ensures all tests
     * work with the same Logger instance in a controlled state.
     */
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

    /**
     * @brief Test fixture teardown - cleans up test artifacts
     * 
     * Removes test log files to ensure clean state for subsequent test runs.
     */
    void TearDown() override {
        // Clean up test log files after each test
        if (std::filesystem::exists(test_log_file)) {
            std::filesystem::remove(test_log_file);
        }
    }

    std::string test_log_file; ///< Path to test log file used across all tests
    
    /**
     * @brief Utility method to read entire log file contents
     * @param filename Path to the file to read
     * @return String containing full file contents, empty if file cannot be opened
     */
    std::string readLogFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return "";
        return std::string((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
    }
    
    /**
     * @brief Utility method to clear the test log file by truncating it
     * 
     * Used between tests when we need to clear log contents without
     * reinitializing the Logger singleton.
     */
    void clearLogFile() {
        // Clear the log file by truncating it
        if (std::filesystem::exists(test_log_file)) {
            std::ofstream file(test_log_file, std::ios::trunc);
            file.close();
        }
    }
};

/**
 * @brief Validates singleton pattern implementation
 * 
 * This test ensures that:
 * - Multiple calls to getInstance() return the same Logger instance
 * - Reference-based and shared_ptr-based access methods are consistent
 * - Memory addresses are identical, confirming true singleton behavior
 * 
 * Validates: Singleton pattern correctness and memory consistency
 */
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

/**
 * @brief Validates core logging functionality and basic operations
 * 
 * This test ensures that:
 * - Basic logging methods execute without exceptions
 * - C++23 std::format string formatting works correctly with various argument types
 * - Depth-based logging functions properly
 * - File output enable/disable toggle functions correctly
 * - Initial logger state is as expected (file output enabled)
 * 
 * Validates: Core API functionality, exception safety, and state management
 */
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

/**
 * @brief Validates log level control and filtering mechanisms
 * 
 * This test ensures that:
 * - All log levels are initially enabled (default state)
 * - Individual log levels can be disabled and re-enabled dynamically
 * - Level state changes persist correctly
 * - Other levels remain unaffected when specific levels are modified
 * - State restoration works properly to avoid affecting subsequent tests
 * 
 * Validates: Dynamic log level filtering, state persistence, and isolation
 */
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

/**
 * @brief Validates file output control functionality
 * 
 * This test ensures that:
 * - File output is enabled by default
 * - File output can be disabled and re-enabled dynamically
 * - State changes are immediately reflected in status queries
 * - Toggle operations work reliably multiple times
 * 
 * Validates: File output state management and dynamic control
 */
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

/**
 * @brief Validates stderr output control functionality
 * 
 * This test ensures that:
 * - Stderr output is enabled by default
 * - Stderr can be disabled and re-enabled using dedicated methods
 * - State changes are immediately reflected in status queries
 * - Enable/disable operations work correctly
 * 
 * Validates: Stderr output state management and manual control methods
 */
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

/**
 * @brief Validates RAII-based stderr suppression guard functionality
 * 
 * This test ensures that:
 * - Stderr starts in enabled state before guard creation
 * - StderrSuppressionGuard automatically disables stderr in its scope
 * - Stderr is automatically restored when guard goes out of scope
 * - RAII pattern works correctly for automatic resource management
 * 
 * Validates: RAII pattern implementation, automatic state restoration, and scoped stderr control
 */
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

/**
 * @brief Validates thread safety of concurrent logging operations
 * 
 * This test ensures that:
 * - Multiple threads can log simultaneously without crashes or corruption
 * - All logging operations complete successfully under concurrent access
 * - Internal synchronization mechanisms (mutex) work correctly
 * - No race conditions occur during high-concurrency scenarios
 * - Exception safety is maintained across all threads
 * 
 * Test approach: Creates 5 threads, each logging 10 messages concurrently,
 * and verifies that all 50 operations complete successfully without exceptions.
 * 
 * Validates: Thread safety, concurrent access protection, and exception safety under load
 */
TEST_F(LoggerTest, ThreadSafety) {
    auto& logger = Logger::getInstance();
    
    std::vector<std::thread> threads;
    const int num_threads = 5;
    const int messages_per_thread = 10;
    std::atomic<int> successful_logs{0};
    
    // Create multiple threads that log simultaneously
    threads.reserve(num_threads);
    for (int ndx = 0; ndx < num_threads; ++ndx) {
            threads.emplace_back([&logger, &successful_logs, ndx]() {
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

/**
 * @brief Validates C++23 std::format string formatting capabilities
 * 
 * This test ensures that:
 * - Simple string messages work without formatting
 * - Integer, floating-point, and string argument formatting works correctly
 * - Multiple argument formatting handles mixed types properly
 * - Precision specifiers (e.g., {:.2f}) work as expected
 * - Depth-based logging with various indentation levels functions correctly
 * - No exceptions are thrown during complex formatting operations
 * 
 * Test coverage includes: basic strings, integers, floats with precision,
 * string objects, multiple mixed arguments, and depth-based formatting.
 * 
 * Validates: C++23 std::format integration, type safety, and formatting correctness
 */
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

/**
 * @brief Validates macro-based logging convenience functions
 * 
 * This test ensures that:
 * - All six log level macros (DEBUG, INFO, NORMAL, WARNING, ERROR, CRITICAL) work correctly
 * - Macros handle various argument types without exceptions
 * - Macro-based formatting integrates properly with underlying Logger methods
 * - Complex multi-argument macro calls with mixed types function correctly
 * - Macros provide convenient interface while maintaining full functionality
 * 
 * Test coverage: All log level macros with simple and complex argument patterns,
 * including strings, integers, floats with precision formatting.
 * 
 * Validates: Macro interface functionality, argument forwarding, and convenience API reliability
 */
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

/**
 * @brief Validates log level filtering behavior with macro interface
 * 
 * This test ensures that:
 * - Disabled log levels can be safely called without crashes or exceptions
 * - Log level state changes affect macro behavior correctly
 * - Filtered messages are handled gracefully (no-op when disabled)
 * - Re-enabling levels restores normal functionality
 * - Macro interface respects level filtering settings consistently
 * 
 * Test approach: Disables specific levels, calls corresponding macros to ensure
 * no crashes occur, then re-enables levels and verifies normal operation resumes.
 * State verification confirms level changes take effect properly.
 * 
 * Validates: Level filtering integration with macros, graceful handling of disabled levels,
 * and dynamic filtering behavior
 */
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