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

#include "../src/result.hpp"
#include "../src/logging.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <memory>
#include <cmath>

using namespace inference_lab::common;

//=============================================================================
// Example 1: Basic File I/O with Error Handling
//=============================================================================

/**
 * @brief Error types for file operations
 */
enum class FileError {
    FileNotFound,
    PermissionDenied,
    InvalidFormat,
    DiskFull,
    CorruptedData
};

/**
 * @brief Convert FileError to human-readable string
 */
std::string to_string(FileError error) {
    switch (error) {
        case FileError::FileNotFound: return "File not found";
        case FileError::PermissionDenied: return "Permission denied";
        case FileError::InvalidFormat: return "Invalid file format";
        case FileError::DiskFull: return "Disk full";
        case FileError::CorruptedData: return "Corrupted data";
    }
    return "Unknown error";
}

/**
 * @brief Read file contents with proper error handling
 * 
 * This example shows how to wrap file I/O operations in Results,
 * providing type-safe error handling without exceptions.
 */
auto read_file(const std::string& filename) -> Result<std::string, FileError> {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return Err(FileError::FileNotFound);
    }
    
    if (!file.good()) {
        return Err(FileError::PermissionDenied);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    if (file.bad()) {
        return Err(FileError::CorruptedData);
    }
    
    return Ok(buffer.str());
}

/**
 * @brief Parse configuration from file content
 */
auto parse_config(const std::string& content) -> Result<std::vector<std::string>, FileError> {
    if (content.empty()) {
        return Err(FileError::InvalidFormat);
    }
    
    std::vector<std::string> lines;
    std::stringstream ss(content);
    std::string line;
    
    while (std::getline(ss, line)) {
        if (!line.empty() && line[0] != '#') { // Skip comments
            lines.push_back(line);
        }
    }
    
    if (lines.empty()) {
        return Err(FileError::InvalidFormat);
    }
    
    return Ok(lines);
}

/**
 * @brief Example of chained file operations using monadic composition
 */
void example_file_operations() {
    std::cout << "=== File Operations Example ===" << std::endl;
    
    // Chain file reading and parsing operations
    auto result = read_file("config.txt")
        .and_then([](const std::string& content) {
            return parse_config(content);
        })
        .map([](const std::vector<std::string>& lines) {
            return lines.size();
        });
    
    // Handle the result
    if (result.is_ok()) {
        std::cout << "Successfully parsed " << result.unwrap() << " configuration lines." << std::endl;
    } else {
        std::cout << "Configuration loading failed: " << to_string(result.unwrap_err()) << std::endl;
        
        // Demonstrate error recovery
        std::cout << "Using default configuration..." << std::endl;
    }
    
    // Alternative approach with unwrap_or_else for fallback
    auto line_count = read_file("config.txt")
        .and_then(parse_config)
        .map([](const auto& lines) { return lines.size(); })
        .unwrap_or_else([](FileError error) -> size_t {
            std::cout << "Using fallback due to: " << to_string(error) << std::endl;
            return 0; // Default configuration
        });
    
    std::cout << "Final line count: " << line_count << std::endl;
}

//=============================================================================
// Example 2: Mathematical Operations with Domain Errors
//=============================================================================

/**
 * @brief Mathematical operation errors
 */
enum class MathError {
    DivisionByZero,
    NegativeSquareRoot,
    DomainError,
    Overflow,
    Underflow
};

std::string to_string(MathError error) {
    switch (error) {
        case MathError::DivisionByZero: return "Division by zero";
        case MathError::NegativeSquareRoot: return "Square root of negative number";
        case MathError::DomainError: return "Value outside valid domain";
        case MathError::Overflow: return "Numeric overflow";
        case MathError::Underflow: return "Numeric underflow";
    }
    return "Unknown math error";
}

/**
 * @brief Safe division operation
 */
auto safe_divide(double a, double b) -> Result<double, MathError> {
    if (b == 0.0) {
        return Err(MathError::DivisionByZero);
    }
    
    double result = a / b;
    
    if (std::isinf(result)) {
        return Err(MathError::Overflow);
    }
    
    return Ok(result);
}

/**
 * @brief Safe square root operation
 */
auto safe_sqrt(double x) -> Result<double, MathError> {
    if (x < 0.0) {
        return Err(MathError::NegativeSquareRoot);
    }
    
    return Ok(std::sqrt(x));
}

/**
 * @brief Safe logarithm operation
 */
auto safe_log(double x) -> Result<double, MathError> {
    if (x <= 0.0) {
        return Err(MathError::DomainError);
    }
    
    double result = std::log(x);
    
    if (std::isinf(result) && result < 0) {
        return Err(MathError::Underflow);
    }
    
    return Ok(result);
}

/**
 * @brief Complex mathematical computation using Result chaining
 * 
 * Computes: sqrt(log(a / b)) with comprehensive error handling
 */
auto complex_math_operation(double a, double b) -> Result<double, MathError> {
    return safe_divide(a, b)
        .and_then(safe_log)
        .and_then(safe_sqrt);
}

/**
 * @brief Example of mathematical operations with error handling
 */
void example_math_operations() {
    std::cout << "\n=== Mathematical Operations Example ===" << std::endl;
    
    // Test cases with different outcomes
    std::vector<std::pair<double, double>> test_cases = {
        {10.0, 2.0},   // Should succeed: sqrt(log(5.0)) ≈ 1.61
        {1.0, 1.0},    // Should succeed: sqrt(log(1.0)) = sqrt(0) = 0
        {0.5, 1.0},    // Should fail: log(0.5) < 0, sqrt of negative
        {10.0, 0.0},   // Should fail: division by zero
        {-5.0, 2.0}    // Should fail: log of negative number
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
    auto with_fallback = complex_math_operation(-1.0, 2.0)
        .or_else([](MathError error) -> Result<double, MathError> {
            std::cout << "Primary computation failed (" << to_string(error) 
                      << "), trying alternative..." << std::endl;
            return Ok(0.0); // Fallback value
        });
    
    std::cout << "Result with fallback: " << with_fallback.unwrap() << std::endl;
}

//=============================================================================
// Example 3: Network/API Operations Simulation
//=============================================================================

/**
 * @brief Network operation errors
 */
enum class NetworkError {
    ConnectionTimeout,
    ServerError,
    InvalidResponse,
    AuthenticationFailed,
    RateLimited
};

std::string to_string(NetworkError error) {
    switch (error) {
        case NetworkError::ConnectionTimeout: return "Connection timeout";
        case NetworkError::ServerError: return "Server error";
        case NetworkError::InvalidResponse: return "Invalid response format";
        case NetworkError::AuthenticationFailed: return "Authentication failed";
        case NetworkError::RateLimited: return "Rate limited";
    }
    return "Unknown network error";
}

/**
 * @brief Simulated API response
 */
struct ApiResponse {
    int status_code;
    std::string body;
    
    ApiResponse(int code, std::string content) 
        : status_code(code), body(std::move(content)) {}
};

/**
 * @brief Simulated network request (normally would use actual HTTP library)
 */
auto make_request(const std::string& url) -> Result<ApiResponse, NetworkError> {
    // Simulate different outcomes based on URL
    if (url.find("timeout") != std::string::npos) {
        return Err(NetworkError::ConnectionTimeout);
    }
    
    if (url.find("auth") != std::string::npos) {
        return Err(NetworkError::AuthenticationFailed);
    }
    
    if (url.find("rate") != std::string::npos) {
        return Err(NetworkError::RateLimited);
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
auto parse_response(const ApiResponse& response) -> Result<std::vector<int>, NetworkError> {
    if (response.status_code != 200) {
        return Err(NetworkError::ServerError);
    }
    
    // Simplified JSON parsing (normally would use proper JSON library)
    if (response.body.find("success") == std::string::npos) {
        return Err(NetworkError::InvalidResponse);
    }
    
    // Extract numbers (simplified)
    std::vector<int> data = {1, 2, 3}; // Hardcoded for example
    return Ok(data);
}

/**
 * @brief Retry logic with exponential backoff
 */
template<typename F>
auto retry_with_backoff(F&& operation, int max_retries = 3) -> decltype(operation()) {
    for (int attempt = 0; attempt < max_retries; ++attempt) {
        auto result = operation();
        
        if (result.is_ok()) {
            return result;
        }
        
        // Check if error is retryable
        auto error = result.unwrap_err();
        if (error == NetworkError::AuthenticationFailed) {
            // Don't retry auth failures
            return result;
        }
        
        if (attempt < max_retries - 1) {
            std::cout << "Attempt " << (attempt + 1) << " failed: " 
                      << to_string(error) << ". Retrying..." << std::endl;
            
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
void example_network_operations() {
    std::cout << "\n=== Network Operations Example ===" << std::endl;
    
    std::vector<std::string> test_urls = {
        "https://api.example.com/data",
        "https://api.example.com/timeout",
        "https://api.example.com/auth",
        "https://api.example.com/error"
    };
    
    for (const auto& url : test_urls) {
        std::cout << "Requesting: " << url << std::endl;
        
        auto result = retry_with_backoff([&url]() {
            return make_request(url)
                .and_then(parse_response);
        });
        
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
enum class DbError {
    ConnectionFailed,
    QuerySyntaxError,
    ConstraintViolation,
    RecordNotFound,
    TransactionFailed
};

std::string to_string(DbError error) {
    switch (error) {
        case DbError::ConnectionFailed: return "Database connection failed";
        case DbError::QuerySyntaxError: return "SQL syntax error";
        case DbError::ConstraintViolation: return "Database constraint violation";
        case DbError::RecordNotFound: return "Record not found";
        case DbError::TransactionFailed: return "Transaction failed";
    }
    return "Unknown database error";
}

/**
 * @brief Simulated database record
 */
struct UserRecord {
    int id;
    std::string name;
    std::string email;
    
    UserRecord(int user_id, std::string user_name, std::string user_email)
        : id(user_id), name(std::move(user_name)), email(std::move(user_email)) {}
};

/**
 * @brief Simulated database connection
 */
class DatabaseConnection {
public:
    auto find_user(int user_id) -> Result<UserRecord, DbError> {
        if (user_id <= 0) {
            return Err(DbError::QuerySyntaxError);
        }
        
        if (user_id == 404) {
            return Err(DbError::RecordNotFound);
        }
        
        // Simulate successful lookup
        return Ok(UserRecord(user_id, "User " + std::to_string(user_id), 
                           "user" + std::to_string(user_id) + "@example.com"));
    }
    
    auto update_user(const UserRecord& user) -> Result<bool, DbError> {
        if (user.name.empty()) {
            return Err(DbError::ConstraintViolation);
        }
        
        if (user.email.find("@") == std::string::npos) {
            return Err(DbError::ConstraintViolation);
        }
        
        return Ok(true);
    }
};

/**
 * @brief Business logic operation using database
 */
auto update_user_email(DatabaseConnection& db, int user_id, const std::string& new_email) 
    -> Result<UserRecord, DbError> {
    
    return db.find_user(user_id)
        .and_then([&db, &new_email](UserRecord user) -> Result<UserRecord, DbError> {
            user.email = new_email;
            return db.update_user(user)
                .map([user = std::move(user)](bool) { return user; });
        });
}

/**
 * @brief Example of database operations with error handling
 */
void example_database_operations() {
    std::cout << "\n=== Database Operations Example ===" << std::endl;
    
    DatabaseConnection db;
    
    std::vector<std::pair<int, std::string>> test_cases = {
        {1, "newemail@example.com"},      // Should succeed
        {404, "another@example.com"},     // User not found
        {2, "invalid_email"},             // Invalid email format
        {-1, "test@example.com"}          // Invalid user ID
    };
    
    for (const auto& [user_id, new_email] : test_cases) {
        std::cout << "Updating user " << user_id << " email to: " << new_email << std::endl;
        
        auto result = update_user_email(db, user_id, new_email);
        
        if (result.is_ok()) {
            const auto& user = result.unwrap();
            std::cout << "Success! Updated user: " << user.name 
                      << " (" << user.email << ")" << std::endl;
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
enum class ComputeError {
    InvalidInput,
    ComputationOverflow,
    MemoryExhausted
};

/**
 * @brief Hot path computation that must be fast
 * 
 * This example shows how Result can be used in performance-critical code
 * without significant overhead compared to raw error codes or exceptions.
 */
auto fast_computation(const std::vector<double>& data) -> Result<double, ComputeError> {
    if (data.empty()) {
        return Err(ComputeError::InvalidInput);
    }
    
    double sum = 0.0;
    for (double value : data) {
        sum += value * value; // Sum of squares
        
        if (std::isinf(sum)) {
            return Err(ComputeError::ComputationOverflow);
        }
    }
    
    return Ok(sum / data.size()); // Mean of squares
}

/**
 * @brief Batch processing with early termination on error
 */
auto process_batch(const std::vector<std::vector<double>>& batches) 
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
void example_performance_critical() {
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
    test_batches.push_back({}); // Empty batch should cause error
    
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
int legacy_parse_int(const char* str, int* result) {
    if (!str || !result) return -1;  // Null pointer
    if (*str == '\0') return -2;     // Empty string
    
    char* endptr;
    long val = std::strtol(str, &endptr, 10);
    
    if (*endptr != '\0') return -3;  // Invalid characters
    if (val > INT_MAX || val < INT_MIN) return -4; // Overflow
    
    *result = static_cast<int>(val);
    return 0; // Success
}

/**
 * @brief Error type for legacy integration
 */
enum class ParseError {
    NullPointer,
    EmptyString,
    InvalidCharacters,
    Overflow
};

/**
 * @brief Wrapper to convert legacy error codes to Result
 */
auto safe_parse_int(const std::string& str) -> Result<int, ParseError> {
    int result;
    int error_code = legacy_parse_int(str.c_str(), &result);
    
    switch (error_code) {
        case 0: return Ok(result);
        case -1: return Err(ParseError::NullPointer);
        case -2: return Err(ParseError::EmptyString);
        case -3: return Err(ParseError::InvalidCharacters);
        case -4: return Err(ParseError::Overflow);
        default: return Err(ParseError::InvalidCharacters);
    }
}

/**
 * @brief Example of legacy code integration
 */
void example_legacy_integration() {
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

int main() {
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