// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file result_usage_examples.cpp
 * @brief Comprehensive usage examples for the Result<T, E> error handling type
 *
 * This file demonstrates practical usage patterns for the Result<T, E> type
 * in real-world scenarios typical of the inference engine. Examples cover:
 * - Basic error handling patterns and best practices
 * - Monadic composition for complex operation chains
 * - Integration with existing APIs and legacy code
 * - Performance-critical usage patterns and optimizations
 * - Error recovery and fallback strategies
 * - Type-safe error propagation across module boundaries
 *
 * Each example includes detailed comments explaining the approach and
 * rationale for using Result<T, E> over traditional error handling methods.
 */

#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../src/result.hpp"

using namespace inference_lab::common;

//=============================================================================
// Example 1: Basic File I/O with Error Handling
//=============================================================================

/**
 * @brief Error types for file operations
 */
enum class FileError : std::uint8_t {
    FILE_NOT_FOUND,
    PERMISSION_DENIED,
    INVALID_FORMAT,
    DISK_FULL,
    CORRUPTED_DATA
};

/**
 * @brief Convert FileError to human-readable string
 */
static std::string to_string(FileError error) {
    switch (error) {
        case FileError::FILE_NOT_FOUND:
            return "File not found";
        case FileError::PERMISSION_DENIED:
            return "Permission denied";
        case FileError::INVALID_FORMAT:
            return "Invalid file format";
        case FileError::DISK_FULL:
            return "Disk full";
        case FileError::CORRUPTED_DATA:
            return "Corrupted data";
    }
    return "Unknown error";
}

/**
 * @brief Read file contents with proper error handling
 *
 * This example shows how to wrap file I/O operations in Results,
 * providing type-safe error handling without exceptions.
 */
static auto read_file(const std::string& filename) -> Result<std::string, FileError> {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return Err(FileError::FILE_NOT_FOUND);
    }

    if (!file.good()) {
        return Err(FileError::PERMISSION_DENIED);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    if (file.bad()) {
        return Err(FileError::CORRUPTED_DATA);
    }

    return Ok(buffer.str());
}

/**
 * @brief Parse configuration from file content
 */
static auto parse_config(const std::string& content) -> Result<std::vector<std::string>, FileError> {
    if (content.empty()) {
        return Err(FileError::INVALID_FORMAT);
    }

    std::vector<std::string> lines;
    std::stringstream ss(content);
    std::string line;

    while (std::getline(ss, line)) {
        if (!line.empty() && line[0] != '#') {  // Skip comments
            lines.push_back(line);
        }
    }

    if (lines.empty()) {
        return Err(FileError::INVALID_FORMAT);
    }

    return Ok(lines);
}

/**
 * @brief Example of chained file operations using monadic composition
 */
static void example_file_operations() {
    std::cout << "=== File Operations Example ===" << std::endl;

    // Chain file reading and parsing operations
    auto result = read_file("config.txt")
                      .and_then([](const std::string& content) { return parse_config(content); })
                      .map([](const std::vector<std::string>& lines) { return lines.size(); });

    // Handle the result
    if (result.is_ok()) {
        std::cout << "Successfully parsed " << result.unwrap() << " configuration lines."
                  << std::endl;
    } else {
        std::cout << "Configuration loading failed: " << to_string(result.unwrap_err())
                  << std::endl;

        // Demonstrate error recovery
        std::cout << "Using default configuration..." << std::endl;
    }

    // Alternative approach with unwrap_or_else for fallback
    auto line_count = read_file("config.txt")
                          .and_then(parse_config)
                          .map([](const auto& lines) { return lines.size(); })
                          .unwrap_or_else([](FileError error) -> size_t {
                              std::cout << "Using fallback due to: " << to_string(error)
                                        << std::endl;
                              return 0;  // Default configuration
                          });

    std::cout << "Final line count: " << line_count << std::endl;
}

//=============================================================================
// Example 2: Mathematical Operations with Domain Errors
//=============================================================================

/**
 * @brief Mathematical operation errors
 */
enum class MathError : std::uint8_t {
    DIVISION_BY_ZERO,
    NEGATIVE_SQUARE_ROOT,
    DOMAIN_ERROR,
    NUMERIC_OVERFLOW,
    NUMERIC_UNDERFLOW
};

static std::string to_string(MathError error) {
    switch (error) {
        case MathError::DIVISION_BY_ZERO:
            return "Division by zero";
        case MathError::NEGATIVE_SQUARE_ROOT:
            return "Square root of negative number";
        case MathError::DOMAIN_ERROR:
            return "Value outside valid domain";
        case MathError::NUMERIC_OVERFLOW:
            return "Numeric overflow";
        case MathError::NUMERIC_UNDERFLOW:
            return "Numeric underflow";
    }
    return "Unknown math error";
}

/**
 * @brief Safe division operation
 */
static auto safe_divide(double a, double b) -> Result<double, MathError> {
    if (b == 0.0) {
        return Err(MathError::DIVISION_BY_ZERO);
    }

    double const result = a / b;

    if (std::isinf(result)) {
        return Err(MathError::NUMERIC_OVERFLOW);
    }

    return Ok(result);
}

/**
 * @brief Safe square root operation
 */
static auto safe_sqrt(double x) -> Result<double, MathError> {
    if (x < 0.0) {
        return Err(MathError::NEGATIVE_SQUARE_ROOT);
    }

    return Ok(std::sqrt(x));
}

/**
 * @brief Safe logarithm operation
 */
static auto safe_log(double x) -> Result<double, MathError> {
    if (x <= 0.0) {
        return Err(MathError::DOMAIN_ERROR);
    }

    double result = std::log(x);

    if (std::isinf(result) && result < 0) {
        return Err(MathError::NUMERIC_UNDERFLOW);
    }

    return Ok(result);
}

/**
 * @brief Complex mathematical computation using Result chaining
 *
 * Computes: sqrt(log(a / b)) with comprehensive error handling
 */
static auto complex_math_operation(double a, double b) -> Result<double, MathError> {
    return safe_divide(a, b).and_then(safe_log).and_then(safe_sqrt);
}

/**
 * @brief Example of mathematical operations with error handling
 */
static void example_math_operations() {
    std::cout << "\n=== Mathematical Operations Example ===" << std::endl;

    // Test cases with different outcomes
    std::vector<std::pair<double, double>> test_cases = {
        {10.0, 2.0},  // Should succeed: sqrt(log(5.0)) ≈ 1.61
        {1.0, 1.0},   // Should succeed: sqrt(log(1.0)) = sqrt(0) = 0
        {0.5, 1.0},   // Should fail: log(0.5) < 0, sqrt of negative
        {10.0, 0.0},  // Should fail: division by zero
        {-5.0, 2.0}   // Should fail: log of negative number
    };

    for (const auto& [a, b] : test_cases) {
        auto result = complex_math_operation(a, b);

        std::cout << "f(" << a << ", " << b << ") = ";

        if (result.is_ok()) {
            std::cout << result.unwrap() << std::endl;
        } else {
            std::cout << "Error: " << to_string(result.unwrap_err()) << std::endl;
        }
    }

    // Demonstrate error recovery with alternative computations
    auto with_fallback =
        complex_math_operation(-1.0, 2.0).or_else([](MathError error) -> Result<double, MathError> {
            std::cout << "Primary computation failed (" << to_string(error)
                      << "), trying alternative..." << std::endl;
            return Ok(0.0);  // Fallback value
        });

    std::cout << "Result with fallback: " << with_fallback.unwrap() << std::endl;
}

//=============================================================================
// Example 3: Network/API Operations Simulation
//=============================================================================

/**
 * @brief Network operation errors
 */
enum class NetworkError : std::uint8_t {
    CONNECTION_TIMEOUT,
    SERVER_ERROR,
    INVALID_RESPONSE,
    AUTHENTICATION_FAILED,
    RATE_LIMITED
};

[[maybe_unused]] static std::string to_string(NetworkError error) {
    switch (error) {
        case NetworkError::CONNECTION_TIMEOUT:
            return "Connection timeout";
        case NetworkError::SERVER_ERROR:
            return "Server error";
        case NetworkError::INVALID_RESPONSE:
            return "Invalid response format";
        case NetworkError::AUTHENTICATION_FAILED:
            return "Authentication failed";
        case NetworkError::RATE_LIMITED:
            return "Rate limited";
    }
    return "Unknown network error";
}

/**
 * @brief Simulated API response
 */
struct ApiResponse {
    int status_code_;
    std::string body_{};

    ApiResponse(int code, std::string content) : status_code_(code), body_(std::move(content)) {}
};

/**
 * @brief Simulated network request (normally would use actual HTTP library)
 */
static auto make_request(const std::string& url) -> Result<ApiResponse, NetworkError> {
    // Simulate different outcomes based on URL
    if (url.find("timeout") != std::string::npos) {
        return Err(NetworkError::CONNECTION_TIMEOUT);
    }

    if (url.find("auth") != std::string::npos) {
        return Err(NetworkError::AUTHENTICATION_FAILED);
    }

    if (url.find("rate") != std::string::npos) {
        return Err(NetworkError::RATE_LIMITED);
    }

    if (url.find("error") != std::string::npos) {
        return Ok(ApiResponse(500, "Internal Server Error"));
    }

    // Successful response
    return Ok(ApiResponse(200, R"({"status": "success", "data": [1, 2, 3]})"));
}

/**
 * @brief Parse API response and extract data
 */
static auto parse_response(const ApiResponse& response) -> Result<std::vector<int>, NetworkError> {
    if (response.status_code_ != 200) {
        return Err(NetworkError::SERVER_ERROR);
    }

    // Simplified JSON parsing (normally would use proper JSON library)
    if (response.body_.find("success") == std::string::npos) {
        return Err(NetworkError::INVALID_RESPONSE);
    }

    // Extract numbers (simplified)
    std::vector<int> data = {1, 2, 3};  // Hardcoded for example
    return Ok(data);
}

/**
 * @brief Retry logic with exponential backoff
 */
template <typename OperationType>
static auto retry_with_backoff(OperationType&& operation, int max_retries = 3)
    -> decltype(operation()) {
    for (size_t attempt = 0; attempt < static_cast<size_t>(max_retries); ++attempt) {
        auto result = operation();

        if (result.is_ok()) {
            return result;
        }

        // Check if error is retryable
        auto error = result.unwrap_err();
        if (error == NetworkError::AUTHENTICATION_FAILED) {
            // Don't retry auth failures
            return result;
        }

        if (attempt < static_cast<size_t>(max_retries) - 1) {
            std::cout << "Attempt " << (attempt + 1) << " failed: " << to_string(error)
                      << ". Retrying..." << std::endl;

            // Simulate delay (normally would use proper sleep)
            // std::this_thread::sleep_for(std::chrono::milliseconds(100 * (1 << attempt)));
        }
    }

    // All retries exhausted, return last error
    return operation();
}

/**
 * @brief Example of network operations with retry logic
 */
static void example_network_operations() {
    std::cout << "\n=== Network Operations Example ===" << std::endl;

    std::vector<std::string> test_urls = {"https://api.example.com/data",
                                          "https://api.example.com/timeout",
                                          "https://api.example.com/auth",
                                          "https://api.example.com/error"};

    for (const auto& url : test_urls) {
        std::cout << "Requesting: " << url << std::endl;

        auto result =
            retry_with_backoff([&url]() { return make_request(url).and_then(parse_response); });

        if (result.is_ok()) {
            const auto& data = result.unwrap();
            std::cout << "Success! Received " << data.size() << " items: ";
            for (int item : data) {
                std::cout << item << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Failed: " << to_string(result.unwrap_err()) << std::endl;
        }

        std::cout << std::endl;
    }
}

//=============================================================================
// Example 4: Database Operations Pattern
//=============================================================================

/**
 * @brief Database operation errors
 */
enum class DbError : std::uint8_t {
    CONNECTION_FAILED,
    QUERY_SYNTAX_ERROR,
    CONSTRAINT_VIOLATION,
    RECORD_NOT_FOUND,
    TRANSACTION_FAILED
};

static std::string to_string(DbError error) {
    switch (error) {
        case DbError::CONNECTION_FAILED:
            return "Database connection failed";
        case DbError::QUERY_SYNTAX_ERROR:
            return "SQL syntax error";
        case DbError::CONSTRAINT_VIOLATION:
            return "Database constraint violation";
        case DbError::RECORD_NOT_FOUND:
            return "Record not found";
        case DbError::TRANSACTION_FAILED:
            return "Transaction failed";
    }
    return "Unknown database error";
}

/**
 * @brief Simulated database record
 */
struct UserRecord {
    int id_;
    std::string name_{};
    std::string email_{};

    UserRecord(int user_id, std::string user_name, std::string user_email)
        : id_(user_id), name_(std::move(user_name)), email_(std::move(user_email)) {}
};

/**
 * @brief Simulated database connection
 */
class DatabaseConnection {
  public:
    auto find_user(int user_id) -> Result<UserRecord, DbError> {
        if (user_id <= 0) {
            return Err(DbError::QUERY_SYNTAX_ERROR);
        }

        if (user_id == 404) {
            return Err(DbError::RECORD_NOT_FOUND);
        }

        // Simulate successful lookup
        return Ok(UserRecord(user_id,
                             "User " + std::to_string(user_id),
                             "user" + std::to_string(user_id) + "@example.com"));
    }

    auto update_user(const UserRecord& user) -> Result<bool, DbError> {
        if (user.name_.empty()) {
            return Err(DbError::CONSTRAINT_VIOLATION);
        }

        if (user.email_.find("@") == std::string::npos) {
            return Err(DbError::CONSTRAINT_VIOLATION);
        }

        return Ok(true);
    }
};

/**
 * @brief Business logic operation using database
 */
static auto update_user_email(DatabaseConnection& db, int user_id, const std::string& new_email)
    -> Result<UserRecord, DbError> {
    return db.find_user(user_id).and_then(
        [&db, &new_email](UserRecord user) -> Result<UserRecord, DbError> {
            user.email_ = new_email;
            return db.update_user(user).map([user = std::move(user)](bool) { return user; });
        });
}

/**
 * @brief Example of database operations with error handling
 */
static void example_database_operations() {
    std::cout << "\n=== Database Operations Example ===" << std::endl;

    DatabaseConnection db;

    std::vector<std::pair<int, std::string>> test_cases = {
        {1, "newemail@example.com"},   // Should succeed
        {404, "another@example.com"},  // User not found
        {2, "invalid_email"},          // Invalid email format
        {-1, "test@example.com"}       // Invalid user ID
    };

    for (const auto& [user_id, new_email] : test_cases) {
        std::cout << "Updating user " << user_id << " email to: " << new_email << std::endl;

        auto result = update_user_email(db, user_id, new_email);

        if (result.is_ok()) {
            const auto& user = result.unwrap();
            std::cout << "Success! Updated user: " << user.name_ << " (" << user.email_ << ")"
                      << std::endl;
        } else {
            std::cout << "Failed: " << to_string(result.unwrap_err()) << std::endl;
        }

        std::cout << std::endl;
    }
}

//=============================================================================
// Example 5: Performance-Critical Path with Result
//=============================================================================

/**
 * @brief Performance-critical computation errors
 */
enum class ComputeError : std::uint8_t { INVALID_INPUT, COMPUTATION_OVERFLOW, MEMORY_EXHAUSTED };

/**
 * @brief Hot path computation that must be fast
 *
 * This example shows how Result can be used in performance-critical code
 * without significant overhead compared to raw error codes or exceptions.
 */
static auto fast_computation(const std::vector<double>& data) -> Result<double, ComputeError> {
    if (data.empty()) {
        return Err(ComputeError::INVALID_INPUT);
    }

    double sum = 0.0;
    for (double value : data) {
        sum += value * value;  // Sum of squares

        if (std::isinf(sum)) {
            return Err(ComputeError::COMPUTATION_OVERFLOW);
        }
    }

    return Ok(sum / data.size());  // Mean of squares
}

/**
 * @brief Batch processing with early termination on error
 */
static auto process_batch(std::vector<std::vector<double>>& batches)
    -> Result<std::vector<double>, ComputeError> {
    std::vector<double> results;
    results.reserve(batches.size());

    for (const auto& batch : batches) {
        auto result = fast_computation(batch);
        if (result.is_err()) {
            return Err(result.unwrap_err());
        }
        results.push_back(result.unwrap());
    }

    return Ok(results);
}

/**
 * @brief Example of performance-critical usage
 */
static void example_performance_critical() {
    std::cout << "\n=== Performance-Critical Example ===" << std::endl;

    // Generate test data
    std::vector<std::vector<double>> test_batches;
    for (int i = 0; i < 1000; ++i) {
        std::vector<double> batch;
        for (int j = 0; j < 100; ++j) {
            batch.push_back(i * 0.1 + j * 0.01);
        }
        test_batches.push_back(batch);
    }

    // Add a problematic batch
    test_batches.push_back({});  // Empty batch should cause error

    auto start = std::chrono::high_resolution_clock::now();

    auto result = process_batch(test_batches);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (result.is_ok()) {
        std::cout << "Processed " << result.unwrap().size() << " batches successfully" << std::endl;
    } else {
        std::cout << "Batch processing failed at batch with error: "
                  << static_cast<int>(result.unwrap_err()) << std::endl;
    }

    std::cout << "Processing time: " << duration.count() << " microseconds" << std::endl;
}

//=============================================================================
// Example 6: Integration with Legacy Code
//=============================================================================

/**
 * @brief Legacy function that returns error codes
 */
static auto legacy_parse_int(const char* str, int* result) -> int {
    if ((str == nullptr) || (result == nullptr))
        return -1;  // Null pointer
    if (*str == '\0')
        return -2;  // Empty string

    char* endptr = nullptr;
    long val = std::strtol(str, &endptr, 10);

    if (*endptr != '\0')
        return -3;  // Invalid characters
    if (val > INT_MAX || val < INT_MIN)
        return -4;  // Overflow

    *result = static_cast<int>(val);
    return 0;  // Success
}

/**
 * @brief Error type for legacy integration
 */
enum class ParseError : std::uint8_t {
    NULL_POINTER,
    EMPTY_STRING,
    INVALID_CHARACTERS,
    NUMERIC_OVERFLOW
};

/**
 * @brief Wrapper to convert legacy error codes to Result
 */
static auto safe_parse_int(const std::string& str) -> Result<int, ParseError> {
    int result = 0;
    int error_code = legacy_parse_int(str.c_str(), &result);

    switch (error_code) {
        case 0:
            return Ok(result);
        case -1:
            return Err(ParseError::NULL_POINTER);
        case -2:
            return Err(ParseError::EMPTY_STRING);
        case -3:
            return Err(ParseError::INVALID_CHARACTERS);
        case -4:
            return Err(ParseError::NUMERIC_OVERFLOW);
        default:
            return Err(ParseError::INVALID_CHARACTERS);
    }
}

/**
 * @brief Example of legacy code integration
 */
static void example_legacy_integration() {
    std::cout << "\n=== Legacy Code Integration Example ===" << std::endl;

    std::vector<std::string> test_inputs = {
        "123",           // Valid
        "456789",        // Valid
        "",              // Empty
        "abc",           // Invalid characters
        "999999999999",  // Overflow
        "12.34"          // Decimal point
    };

    for (const auto& input : test_inputs) {
        std::cout << "Parsing: '" << input << "' -> ";

        auto result = safe_parse_int(input);

        if (result.is_ok()) {
            std::cout << result.unwrap() << std::endl;
        } else {
            std::cout << "Error: " << static_cast<int>(result.unwrap_err()) << std::endl;
        }
    }
}

//=============================================================================
// Main Function - Run All Examples
//=============================================================================

auto main() -> int {
    std::cout << "Result<T, E> Usage Examples" << std::endl;
    std::cout << "===========================" << std::endl;

    try {
        example_file_operations();
        example_math_operations();
        example_network_operations();
        example_database_operations();
        example_performance_critical();
        example_legacy_integration();

        std::cout << "\nAll examples completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception in examples: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
