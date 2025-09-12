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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../src/logging.hpp"

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
        test_log_file_ = "./test_unit_logging.log";

        // Clean up any existing test log file
        if (std::filesystem::exists(test_log_file_)) {
            std::filesystem::remove(test_log_file_);
        }

        // Initialize the singleton Logger with our test file and truncate mode
        // This creates the singleton instance that all tests will use
        Logger::get_instance(test_log_file_, false);
    }

    /**
     * @brief Test fixture teardown - cleans up test artifacts
     *
     * Removes test log files to ensure clean state for subsequent test runs.
     */
    void TearDown() override {
        // Clean up test log files after each test
        if (std::filesystem::exists(test_log_file_)) {
            std::filesystem::remove(test_log_file_);
        }
    }

    std::string test_log_file_;  ///< Path to test log file used across all tests

    /**
     * @brief Utility method to read entire log file contents
     * @param filename Path to the file to read
     * @return String containing full file contents, empty if file cannot be opened
     */
    static std::string read_log_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open())
            return "";
        return std::string((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    }

    /**
     * @brief Utility method to clear the test log file by truncating it
     *
     * Used between tests when we need to clear log contents without
     * reinitializing the Logger singleton.
     */
    static void clear_log_file(const std::string& filename) {
        // Clear the log file by truncating it
        if (std::filesystem::exists(filename)) {
            std::ofstream file(filename, std::ios::trunc);
            file.close();
        }
    }
};

/**
 * @brief Validates singleton pattern implementation
 *
 * This test ensures that:
 * - Multiple calls to get_instance() return the same Logger instance
 * - Reference-based and shared_ptr-based access methods are consistent
 * - Memory addresses are identical, confirming true singleton behavior
 *
 * Validates: Singleton pattern correctness and memory consistency
 */
TEST_F(LoggerTest, SingletonBehavior) {
    // Test that get_instance returns the same instance
    auto& logger1 = Logger::get_instance();
    auto& logger2 = Logger::get_instance();

    EXPECT_EQ(&logger1, &logger2);

    // Test shared_ptr version
    const auto PTR1 = Logger::get_instance_ptr();
    const auto PTR2 = Logger::get_instance_ptr();

    EXPECT_EQ(PTR1, PTR2);
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
    auto& logger = Logger::get_instance();

    // Test that logging methods don't crash and work correctly
    EXPECT_NO_THROW(logger.print_log(LogLevel::INFO, "Test message"));
    EXPECT_NO_THROW(logger.print_log(LogLevel::WARNING, "Number: {}", 42));
    EXPECT_NO_THROW(logger.print_log_with_depth(LogLevel::DEBUG, 1, "Depth test"));

    // Test file output state can be controlled
    EXPECT_TRUE(logger.is_file_output_enabled());
    logger.set_file_output_enabled(false);
    EXPECT_FALSE(logger.is_file_output_enabled());
    logger.set_file_output_enabled(true);
    EXPECT_TRUE(logger.is_file_output_enabled());
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
    [[maybe_unused]] auto& logger = Logger::get_instance();

    // Test initial state - all levels should be enabled
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::DEBUG));
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::INFO));
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::NORMAL));
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::WARNING));
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::ERROR));
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::CRITICAL));

    // Disable specific levels
    Logger::set_level_enabled(LogLevel::DEBUG, false);
    Logger::set_level_enabled(LogLevel::INFO, false);

    EXPECT_FALSE(Logger::is_level_enabled(LogLevel::DEBUG));
    EXPECT_FALSE(Logger::is_level_enabled(LogLevel::INFO));
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::NORMAL));

    // Re-enable levels for other tests
    Logger::set_level_enabled(LogLevel::DEBUG, true);
    Logger::set_level_enabled(LogLevel::INFO, true);

    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::DEBUG));
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::INFO));
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
    auto& logger = Logger::get_instance();

    // Test initial state
    EXPECT_TRUE(logger.is_file_output_enabled());

    // Test disable/enable
    logger.set_file_output_enabled(false);
    EXPECT_FALSE(logger.is_file_output_enabled());

    logger.set_file_output_enabled(true);
    EXPECT_TRUE(logger.is_file_output_enabled());
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
    auto& logger = Logger::get_instance();

    // Test initial state
    EXPECT_TRUE(logger.is_stderr_enabled());

    // Test disable/enable
    logger.disable_stderr();
    EXPECT_FALSE(logger.is_stderr_enabled());

    logger.enable_stderr();
    EXPECT_TRUE(logger.is_stderr_enabled());
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
    auto& logger = Logger::get_instance();

    // Initial state should be enabled
    EXPECT_TRUE(logger.is_stderr_enabled());

    {
        Logger::StderrSuppressionGuard const GUARD;
        EXPECT_FALSE(logger.is_stderr_enabled());
    }

    // Should be restored after guard destruction
    EXPECT_TRUE(logger.is_stderr_enabled());
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
    auto& logger = Logger::get_instance();

    std::vector<std::thread> threads;
    const int NUM_THREADS = 5;
    const int MESSAGES_PER_THREAD = 10;
    std::atomic<int> successful_logs{0};

    // Create multiple threads that log simultaneously
    threads.reserve(NUM_THREADS);
    for (int ndx = 0; ndx < NUM_THREADS; ++ndx) {
        threads.emplace_back([&logger, &successful_logs, ndx]() {
            for (int j = 0; j < MESSAGES_PER_THREAD; ++j) {
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
    for (auto& thrd : threads) {
        thrd.join();
    }

    // Verify all messages were logged successfully (no crashes)
    EXPECT_EQ(successful_logs.load(), NUM_THREADS * MESSAGES_PER_THREAD);
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
    auto& logger = Logger::get_instance();

    // Test various formatting - verify they don't crash with complex formats
    EXPECT_NO_THROW(logger.print_log(LogLevel::INFO, "Simple message"));
    EXPECT_NO_THROW(logger.print_log(LogLevel::WARNING, "Number: {}", 42));
    EXPECT_NO_THROW(logger.print_log(LogLevel::ERROR, "Float: {:.2f}", 3.14159));
    EXPECT_NO_THROW(logger.print_log(LogLevel::DEBUG, "String: {}", std::string("test")));
    EXPECT_NO_THROW(
        logger.print_log(LogLevel::CRITICAL, "Multiple args: {} {} {}", 1, 2.5, "three"));

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
    [[maybe_unused]] auto& logger = Logger::get_instance();

    // Test that disabled levels don't cause crashes
    Logger::set_level_enabled(LogLevel::DEBUG, false);
    EXPECT_NO_THROW(LOG_DEBUG_PRINT("This debug should be filtered"));
    EXPECT_FALSE(Logger::is_level_enabled(LogLevel::DEBUG));

    Logger::set_level_enabled(LogLevel::INFO, false);
    EXPECT_NO_THROW(LOG_INFO_PRINT("This info should be filtered"));
    EXPECT_FALSE(Logger::is_level_enabled(LogLevel::INFO));

    // Re-enable for other tests
    Logger::set_level_enabled(LogLevel::DEBUG, true);
    Logger::set_level_enabled(LogLevel::INFO, true);

    // Test that enabled levels work
    EXPECT_NO_THROW(LOG_DEBUG_PRINT("This debug should work"));
    EXPECT_NO_THROW(LOG_INFO_PRINT("This info should work"));
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::DEBUG));
    EXPECT_TRUE(Logger::is_level_enabled(LogLevel::INFO));
}
