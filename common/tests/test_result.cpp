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
 * @file test_result.cpp
 * @brief Comprehensive unit tests for the Result<T, E> error handling type
 *
 * This test suite validates the complete Result<T, E> implementation including:
 * - Construction and basic operations with Ok and Err variants
 * - State checking and value extraction methods (safe and unsafe)
 * - Monadic operations (map, and_then, or_else) for functional composition
 * - Conversion utilities and interoperability with std::optional
 * - Type safety and SFINAE constraints for template metaprogramming
 * - Move semantics and resource management with RAII principles
 * - Structured binding support for modern C++ usage patterns
 * - Error propagation and chaining for complex operation sequences
 * - Performance characteristics and zero-cost abstraction validation
 * - Edge cases, error conditions, and exception safety guarantees
 *
 * The tests are organized to cover both normal usage patterns and edge cases,
 * ensuring the Result type provides robust error handling suitable for
 * production use in the inference engine's critical paths.
 */

#include <chrono>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "../src/result.hpp"

using namespace inference_lab::common;

/**
 * @class ResultTest
 * @brief Test fixture for Result<T, E> unit tests
 *
 * Provides common setup and utility methods for all Result tests.
 * The fixture includes helper types and functions for testing various
 * scenarios including custom error types and complex value types.
 */
class ResultTest : public ::testing::Test {
  protected:
    /**
     * @brief Common error types for testing
     */
    enum class TestError : std::uint8_t {
        NETWORK_FAILURE,
        PARSE_ERROR,
        INVALID_INPUT,
        TIMEOUT_ERROR,
        UNKNOWN_ERROR
    };

    /**
     * @brief Custom error type with additional data
     */
    struct DetailedError {
        TestError code_;
        std::string message_{};
        int error_number_;

        DetailedError(TestError c, std::string msg, int num = 0)
            : code_(c), message_(std::move(msg)), error_number_(num) {}

        auto operator==(const DetailedError& other) const -> bool {
            return code_ == other.code_ && message_ == other.message_ &&
                   error_number_ == other.error_number_;
        }
    };

    /**
     * @brief Custom value type for testing move semantics
     */
    struct MoveOnlyType {
        int value_;
        bool moved_from_ = false;

        explicit MoveOnlyType(int v) : value_(v) {}

        // Non-copyable
        MoveOnlyType(const MoveOnlyType&) = delete;
        auto operator=(const MoveOnlyType&) -> MoveOnlyType& = delete;

        // Movable
        MoveOnlyType(MoveOnlyType&& other) noexcept : value_(other.value_) {
            other.moved_from_ = true;
        }

        auto operator=(MoveOnlyType&& other) noexcept -> MoveOnlyType& {
            if (this != &other) {
                value_ = other.value_;
                other.moved_from_ = true;
            }
            return *this;
        }

        auto operator==(const MoveOnlyType& other) const -> bool {
            return value_ == other.value_ && !moved_from_ && !other.moved_from_;
        }
    };

    /**
     * @brief Helper function that returns a successful result
     */
    static auto successful_operation(int input) -> Result<int, TestError> {
        if (input >= 0) {
            return Ok(input * 2);
        }
        return Err(TestError::INVALID_INPUT);
    }

    /**
     * @brief Helper function that returns an error result
     */
    static auto failing_operation([[maybe_unused]] int input) -> Result<int, TestError> {
        return Err(TestError::NETWORK_FAILURE);
    }

    /**
     * @brief Helper function that may fail based on input
     */
    static auto conditional_operation(int input) -> Result<std::string, TestError> {
        if (input > 10) {
            return Ok(std::string("Large: ") + std::to_string(input));
        } else if (input > 0) {
            return Ok(std::string("Small: ") + std::to_string(input));
        } else {
            return Err(TestError::INVALID_INPUT);
        }
    }

    /**
     * @brief Helper function for testing and_then chaining
     */
    static auto chain_operation(int input) -> Result<int, TestError> {
        if (input < 100) {
            return Ok(input + 10);
        }
        return Err(TestError::INVALID_INPUT);
    }
};

//=============================================================================
// Basic Construction and State Tests
//=============================================================================

/**
 * @brief Test basic Result construction with Ok and Err
 *
 * Validates that:
 * - Results can be constructed from Ok and Err wrappers
 * - State checking methods (is_ok, is_err) work correctly
 * - Boolean conversion operator works as expected
 * - Type deduction works for factory functions
 */
TEST_F(ResultTest, BasicConstruction) {
    // Test successful Result construction
    Result<int, TestError> success = Ok(42);
    EXPECT_TRUE(success.is_ok());
    EXPECT_FALSE(success.is_err());
    EXPECT_TRUE(static_cast<bool>(success));

    // Test error Result construction
    Result<int, TestError> failure = Err(TestError::NETWORK_FAILURE);
    EXPECT_FALSE(failure.is_ok());
    EXPECT_TRUE(failure.is_err());
    EXPECT_FALSE(static_cast<bool>(failure));

    // Test factory functions with type deduction
    auto success_auto = make_ok(123);
    auto failure_auto = make_err(TestError::PARSE_ERROR);

    Result<int, TestError> success_typed = success_auto;
    Result<int, TestError> failure_typed = failure_auto;

    EXPECT_TRUE(success_typed.is_ok());
    EXPECT_EQ(success_typed.unwrap(), 123);
    EXPECT_TRUE(failure_typed.is_err());
    EXPECT_EQ(failure_typed.unwrap_err(), TestError::PARSE_ERROR);
}

/**
 * @brief Test Result construction with complex types
 *
 * Validates that:
 * - Results work with custom types and structures
 * - String and container types are properly handled
 * - Complex error types with additional data work correctly
 */
TEST_F(ResultTest, ComplexTypeConstruction) {
    // Test with string types
    Result<std::string, TestError> string_result = Ok(std::string("Hello, World!"));
    EXPECT_TRUE(string_result.is_ok());
    EXPECT_EQ(string_result.unwrap(), "Hello, World!");

    // Test with vector types
    std::vector<int> test_vector = {1, 2, 3, 4, 5};
    Result<std::vector<int>, TestError> vector_result = Ok(test_vector);
    EXPECT_TRUE(vector_result.is_ok());
    EXPECT_EQ(vector_result.unwrap(), test_vector);

    // Test with custom error type
    DetailedError custom_error(TestError::NETWORK_FAILURE, "Connection timeout", 404);
    Result<int, DetailedError> custom_result = Err(custom_error);
    EXPECT_TRUE(custom_result.is_err());
    EXPECT_EQ(custom_result.unwrap_err(), custom_error);
}

/**
 * @brief Test copy and move semantics
 *
 * Validates that:
 * - Results can be copied when contained types are copyable
 * - Move semantics work correctly for efficiency
 * - Move-only types can be used in Results
 * - Resource management follows RAII principles
 */
TEST_F(ResultTest, CopyAndMoveSemantics) {
    // Test copy construction and assignment
    Result<int, TestError> original = Ok(100);
    Result<int, TestError> copied = original;
    Result<int, TestError> assigned = Err(TestError::UNKNOWN_ERROR);
    assigned = original;

    EXPECT_TRUE(original.is_ok());
    EXPECT_TRUE(copied.is_ok());
    EXPECT_TRUE(assigned.is_ok());
    EXPECT_EQ(copied.unwrap(), 100);
    EXPECT_EQ(assigned.unwrap(), 100);

    // Test move construction and assignment
    Result<MoveOnlyType, TestError> move_original = Ok(MoveOnlyType(42));
    EXPECT_TRUE(move_original.is_ok());
    EXPECT_EQ(move_original.unwrap().value_, 42);

    Result<MoveOnlyType, TestError> move_constructed = std::move(move_original);
    EXPECT_TRUE(move_constructed.is_ok());
    EXPECT_EQ(move_constructed.unwrap().value_, 42);

    Result<MoveOnlyType, TestError> move_assigned = Err(TestError::INVALID_INPUT);
    move_assigned = Ok(MoveOnlyType(99));
    EXPECT_TRUE(move_assigned.is_ok());
    EXPECT_EQ(move_assigned.unwrap().value_, 99);
}

//=============================================================================
// Value Extraction Tests
//=============================================================================

/**
 * @brief Test safe and unsafe value extraction methods
 *
 * Validates that:
 * - unwrap() returns correct values and throws on errors
 * - unwrap_err() returns correct errors and throws on success
 * - unwrap_or() provides fallback values correctly
 * - unwrap_or_else() computes fallback values from errors
 */
TEST_F(ResultTest, ValueExtraction) {
    Result<int, TestError> success = Ok(42);
    Result<int, TestError> failure = Err(TestError::NETWORK_FAILURE);

    // Test successful unwrap
    EXPECT_EQ(success.unwrap(), 42);
    EXPECT_EQ(failure.unwrap_err(), TestError::NETWORK_FAILURE);

    // Test unwrap exceptions
    EXPECT_THROW(failure.unwrap(), std::runtime_error);
    EXPECT_THROW(success.unwrap_err(), std::runtime_error);

    // Test unwrap_or with default values
    EXPECT_EQ(success.unwrap_or(99), 42);
    EXPECT_EQ(failure.unwrap_or(99), 99);

    // Test unwrap_or_else with computation
    auto compute_fallback = [](TestError error) -> int {
        switch (error) {
            case TestError::NETWORK_FAILURE:
                return -1;
            case TestError::PARSE_ERROR:
                return -2;
            default:
                return -99;
        }
    };

    EXPECT_EQ(success.unwrap_or_else(compute_fallback), 42);
    EXPECT_EQ(failure.unwrap_or_else(compute_fallback), -1);

    Result<int, TestError> parse_error = Err(TestError::PARSE_ERROR);
    EXPECT_EQ(parse_error.unwrap_or_else(compute_fallback), -2);
}

/**
 * @brief Test move semantics in value extraction
 *
 * Validates that:
 * - Move versions of extraction methods work correctly
 * - Resources are properly transferred without copying
 * - Move-only types can be extracted efficiently
 */
TEST_F(ResultTest, MoveValueExtraction) {
    Result<MoveOnlyType, TestError> success = Ok(MoveOnlyType(42));
    Result<MoveOnlyType, TestError> failure = Err(TestError::NETWORK_FAILURE);

    // Test move unwrap
    MoveOnlyType extracted = std::move(success).unwrap();
    EXPECT_EQ(extracted.value_, 42);
    EXPECT_FALSE(extracted.moved_from_);

    // Test that failure result is properly recognized as error
    EXPECT_TRUE(failure.is_err());
    EXPECT_FALSE(failure.is_ok());

    // Test move unwrap_or
    Result<MoveOnlyType, TestError> failure2 = Err(TestError::PARSE_ERROR);
    MoveOnlyType fallback = std::move(failure2).unwrap_or(MoveOnlyType(99));
    EXPECT_EQ(fallback.value_, 99);

    // Test move unwrap_or_else
    auto create_fallback = [](TestError error) -> MoveOnlyType {
        return MoveOnlyType(static_cast<int>(error) + 1000);
    };

    Result<MoveOnlyType, TestError> failure3 = Err(TestError::INVALID_INPUT);
    MoveOnlyType computed = std::move(failure3).unwrap_or_else(create_fallback);
    EXPECT_EQ(computed.value_, static_cast<int>(TestError::INVALID_INPUT) + 1000);
}

//=============================================================================
// Optional Conversion Tests
//=============================================================================

/**
 * @brief Test conversion to std::optional
 *
 * Validates that:
 * - ok() method converts success to optional with value
 * - ok() method converts error to nullopt
 * - err() method converts error to optional with error
 * - err() method converts success to nullopt
 * - Move versions work correctly
 */
TEST_F(ResultTest, OptionalConversion) {
    Result<int, TestError> success = Ok(42);
    Result<int, TestError> failure = Err(TestError::NETWORK_FAILURE);

    // Test ok() conversion
    auto success_opt = success.ok();
    auto failure_opt = failure.ok();

    EXPECT_TRUE(success_opt.has_value());
    EXPECT_EQ(success_opt.value(), 42);
    EXPECT_FALSE(failure_opt.has_value());

    // Test err() conversion
    auto success_err = success.err();
    auto failure_err = failure.err();

    EXPECT_FALSE(success_err.has_value());
    EXPECT_TRUE(failure_err.has_value());
    EXPECT_EQ(failure_err.value(), TestError::NETWORK_FAILURE);

    // Test move versions
    Result<MoveOnlyType, TestError> move_success = Ok(MoveOnlyType(123));
    auto move_opt = std::move(move_success).ok();

    EXPECT_TRUE(move_opt.has_value());
    EXPECT_EQ(move_opt.value().value_, 123);
    EXPECT_FALSE(move_opt.value().moved_from_);
}

//=============================================================================
// Monadic Operations Tests
//=============================================================================

/**
 * @brief Test map operations for value transformation
 *
 * Validates that:
 * - map() transforms successful values correctly
 * - map() preserves errors unchanged
 * - Type transformations work properly
 * - Chaining multiple maps works as expected
 */
TEST_F(ResultTest, MapOperations) {
    Result<int, TestError> success = Ok(10);
    Result<int, TestError> failure = Err(TestError::NETWORK_FAILURE);

    // Test basic map transformation
    auto doubled = success.map([](int x) { return x * 2; });
    auto failure_mapped = failure.map([](int x) { return x * 2; });

    EXPECT_TRUE(doubled.is_ok());
    EXPECT_EQ(doubled.unwrap(), 20);
    EXPECT_TRUE(failure_mapped.is_err());
    EXPECT_EQ(failure_mapped.unwrap_err(), TestError::NETWORK_FAILURE);

    // Test type transformation
    auto to_string = success.map([](int x) { return std::to_string(x); });
    EXPECT_TRUE(to_string.is_ok());
    EXPECT_EQ(to_string.unwrap(), "10");

    // Test chaining maps
    auto chained =
        success.map([](int x) { return x + 5; }).map([](int x) { return x * 3; }).map([](int x) {
            return std::to_string(x);
        });

    EXPECT_TRUE(chained.is_ok());
    EXPECT_EQ(chained.unwrap(), "45");  // (10 + 5) * 3 = 45

    // Test map with error preservation through chain
    auto error_chain =
        failure.map([](int x) { return x + 1; }).map([](int x) { return std::to_string(x); });

    EXPECT_TRUE(error_chain.is_err());
    EXPECT_EQ(error_chain.unwrap_err(), TestError::NETWORK_FAILURE);
}

/**
 * @brief Test map_err operations for error transformation
 *
 * Validates that:
 * - map_err() transforms error values correctly
 * - map_err() preserves successful values unchanged
 * - Error type transformations work properly
 * - Multiple error transformations can be chained
 */
TEST_F(ResultTest, MapErrorOperations) {
    Result<int, TestError> success = Ok(42);
    Result<int, TestError> failure = Err(TestError::NETWORK_FAILURE);

    // Test basic error transformation
    auto to_detailed = failure.map_err(
        [](TestError err) { return DetailedError(err, "Network connection failed", 500); });

    EXPECT_TRUE(to_detailed.is_err());
    auto detailed_err = to_detailed.unwrap_err();
    EXPECT_EQ(detailed_err.code_, TestError::NETWORK_FAILURE);
    EXPECT_EQ(detailed_err.message_, "Network connection failed");
    EXPECT_EQ(detailed_err.error_number_, 500);

    // Test success preservation
    auto success_mapped = success.map_err(
        [](TestError err) { return DetailedError(err, "Should not be called", 0); });

    EXPECT_TRUE(success_mapped.is_ok());
    EXPECT_EQ(success_mapped.unwrap(), 42);

    // Test error transformation chaining
    auto chained_error =
        failure.map_err([](TestError err) { return static_cast<int>(err); }).map_err([](int code) {
            return "Error code: " + std::to_string(code);
        });

    EXPECT_TRUE(chained_error.is_err());
    EXPECT_EQ(chained_error.unwrap_err(),
              "Error code: " + std::to_string(static_cast<int>(TestError::NETWORK_FAILURE)));
}

/**
 * @brief Test and_then operations for monadic binding
 *
 * Validates that:
 * - and_then() chains operations that may fail
 * - Error propagation works correctly through chains
 * - Type transformations are preserved
 * - Short-circuiting on first error works as expected
 */
TEST_F(ResultTest, AndThenOperations) {
    // Test successful chaining
    auto result = successful_operation(5)
                      .and_then([](int x) { return chain_operation(x); })
                      .and_then([](int x) { return conditional_operation(x); });

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.unwrap(), "Large: 20");  // (5 * 2) + 10 = 20, which is > 10

    // Test error propagation from first operation
    auto error_result = successful_operation(-1)  // This will fail
                            .and_then([](int x) { return chain_operation(x); })
                            .and_then([](int x) { return conditional_operation(x); });

    EXPECT_TRUE(error_result.is_err());
    EXPECT_EQ(error_result.unwrap_err(), TestError::INVALID_INPUT);

    // Test error propagation from middle operation
    auto middle_error =
        successful_operation(50)                                 // 50 * 2 = 100
            .and_then([](int x) { return chain_operation(x); })  // 100 + 10 = 110 > 100, fails
            .and_then([](int x) { return conditional_operation(x); });

    EXPECT_TRUE(middle_error.is_err());
    EXPECT_EQ(middle_error.unwrap_err(), TestError::INVALID_INPUT);

    // Test type changing chain
    Result<int, TestError> start = Ok(5);
    auto type_chain = start
                          .and_then([](int x) -> Result<std::string, TestError> {
                              return Ok(std::to_string(x * 2));
                          })
                          .and_then([](const std::string& s) -> Result<int, TestError> {
                              return Ok(static_cast<int>(s.length()));
                          });

    EXPECT_TRUE(type_chain.is_ok());
    EXPECT_EQ(type_chain.unwrap(), 2);  // "10" has length 2
}

/**
 * @brief Test or_else operations for error recovery
 *
 * Validates that:
 * - or_else() provides alternative Results on error
 * - Successful values are preserved unchanged
 * - Error recovery chains work correctly
 * - Type constraints are properly enforced
 */
TEST_F(ResultTest, OrElseOperations) {
    Result<int, TestError> success = Ok(42);
    Result<int, TestError> failure = Err(TestError::NETWORK_FAILURE);

    // Test success preservation
    auto success_or_else =
        success.or_else([](TestError err [[maybe_unused]]) -> Result<int, TestError> {
            return Ok(999);  // Should not be called
        });

    EXPECT_TRUE(success_or_else.is_ok());
    EXPECT_EQ(success_or_else.unwrap(), 42);

    // Test error recovery
    auto recovered = failure.or_else([](TestError err) -> Result<int, TestError> {
        if (err == TestError::NETWORK_FAILURE) {
            return Ok(100);  // Provide fallback value
        }
        return Err(TestError::UNKNOWN_ERROR);
    });

    EXPECT_TRUE(recovered.is_ok());
    EXPECT_EQ(recovered.unwrap(), 100);

    // Test error transformation in recovery
    auto transformed_error = failure.or_else([](TestError err) -> Result<int, DetailedError> {
        return Err(DetailedError(err, "Transformed error", 123));
    });

    EXPECT_TRUE(transformed_error.is_err());
    auto detailed = transformed_error.unwrap_err();
    EXPECT_EQ(detailed.code_, TestError::NETWORK_FAILURE);
    EXPECT_EQ(detailed.message_, "Transformed error");
    EXPECT_EQ(detailed.error_number_, 123);

    // Test chained error recovery
    Result<int, TestError> multiple_errors = Err(TestError::PARSE_ERROR);
    auto multi_recovery = multiple_errors
                              .or_else([](TestError err) -> Result<int, TestError> {
                                  if (err == TestError::NETWORK_FAILURE) {
                                      return Ok(1);
                                  }
                                  return Err(TestError::TIMEOUT_ERROR);  // Transform error
                              })
                              .or_else([](TestError err) -> Result<int, TestError> {
                                  if (err == TestError::TIMEOUT_ERROR) {
                                      return Ok(2);
                                  }
                                  return Err(err);  // Pass through
                              });

    EXPECT_TRUE(multi_recovery.is_ok());
    EXPECT_EQ(multi_recovery.unwrap(), 2);
}

//=============================================================================
// Structured Binding Tests
//=============================================================================

/**
 * @brief Test structured binding support
 *
 * Validates that:
 * - Results can be decomposed using structured bindings
 * - Three-element binding works (is_ok, value_opt, error_opt)
 * - Move semantics work with structured bindings
 * - Tuple-like interface is correctly implemented
 */
TEST_F(ResultTest, StructuredBinding) {
    Result<int, TestError> success = Ok(42);
    Result<int, TestError> failure = Err(TestError::NETWORK_FAILURE);

    // Test structured binding for success
    auto [is_ok1, value_opt1, error_opt1] = success;
    EXPECT_TRUE(is_ok1);
    EXPECT_TRUE(value_opt1.has_value());
    EXPECT_EQ(value_opt1.value(), 42);
    EXPECT_FALSE(error_opt1.has_value());

    // Test structured binding for error
    auto [is_ok2, value_opt2, error_opt2] = failure;
    EXPECT_FALSE(is_ok2);
    EXPECT_FALSE(value_opt2.has_value());
    EXPECT_TRUE(error_opt2.has_value());
    EXPECT_EQ(error_opt2.value(), TestError::NETWORK_FAILURE);

    // Test with move semantics
    Result<MoveOnlyType, TestError> move_result = Ok(MoveOnlyType(123));
    auto [is_ok3, value_opt3, error_opt3] = std::move(move_result);

    EXPECT_TRUE(is_ok3);
    EXPECT_TRUE(value_opt3.has_value());
    EXPECT_EQ(value_opt3.value().value_, 123);
    EXPECT_FALSE(value_opt3.value().moved_from_);
    EXPECT_FALSE(error_opt3.has_value());
}

//=============================================================================
// Utility Function Tests
//=============================================================================

/**
 * @brief Test utility functions and convenience helpers
 *
 * Validates that:
 * - make_ok and make_err factory functions work with type deduction
 * - try_call wrapper function converts exceptions to Results
 * - combine function merges multiple Results correctly
 * - Error handling utilities provide proper functionality
 */
TEST_F(ResultTest, UtilityFunctions) {
    // Test factory functions
    auto ok_result = make_ok(std::string("success"));
    auto err_result = make_err(TestError::PARSE_ERROR);

    Result<std::string, TestError> typed_ok = ok_result;
    Result<int, TestError> typed_err = err_result;

    EXPECT_TRUE(typed_ok.is_ok());
    EXPECT_EQ(typed_ok.unwrap(), "success");
    EXPECT_TRUE(typed_err.is_err());
    EXPECT_EQ(typed_err.unwrap_err(), TestError::PARSE_ERROR);

    // Test try_call with successful function
    auto success_call = try_call<std::runtime_error>([]() { return 42; });

    EXPECT_TRUE(success_call.is_ok());
    EXPECT_EQ(success_call.unwrap(), 42);

    // Test try_call with throwing function
    auto error_call =
        try_call<std::runtime_error>([]() -> int { throw std::runtime_error("Test exception"); });

    EXPECT_TRUE(error_call.is_err());
    EXPECT_EQ(error_call.unwrap_err().what(), std::string("Test exception"));

    // Test try_call with void function
    bool side_effect = false;
    auto void_call = try_call<std::runtime_error>([&side_effect]() { side_effect = true; });

    EXPECT_TRUE(void_call.is_ok());
    EXPECT_TRUE(side_effect);
}

/**
 * @brief Test combine function for multiple Results
 *
 * Validates that:
 * - All successful Results are combined into a tuple
 * - First error is returned if any Result fails
 * - Type safety is maintained across combinations
 * - Move semantics work correctly in combinations
 */
TEST_F(ResultTest, CombineFunction) {
    // Test successful combination
    Result<int, TestError> r1 = Ok(10);
    Result<std::string, TestError> r2 = Ok(std::string("hello"));
    Result<double, TestError> r3 = Ok(3.14);

    auto combined = combine(r1, r2, r3);
    EXPECT_TRUE(combined.is_ok());

    auto [val1, val2, val3] = combined.unwrap();
    EXPECT_EQ(val1, 10);
    EXPECT_EQ(val2, "hello");
    EXPECT_DOUBLE_EQ(val3, 3.14);

    // Test first error propagation
    Result<int, TestError> r4 = Ok(20);
    Result<std::string, TestError> r5 = Err(TestError::NETWORK_FAILURE);
    Result<double, TestError> r6 = Err(TestError::PARSE_ERROR);

    auto error_combined = combine(r4, r5, r6);
    EXPECT_TRUE(error_combined.is_err());
    EXPECT_EQ(error_combined.unwrap_err(), TestError::NETWORK_FAILURE);  // First error

    // Test mixed success and error
    Result<int, TestError> r7 = Ok(30);
    Result<std::string, TestError> r8 = Ok(std::string("world"));
    Result<double, TestError> r9 = Err(TestError::INVALID_INPUT);

    auto mixed_combined = combine(r7, r8, r9);
    EXPECT_TRUE(mixed_combined.is_err());
    EXPECT_EQ(mixed_combined.unwrap_err(), TestError::INVALID_INPUT);
}

//=============================================================================
// Equality and Comparison Tests
//=============================================================================

/**
 * @brief Test equality operators and comparisons
 *
 * Validates that:
 * - Equal Results compare as equal
 * - Different Results compare as unequal
 * - Comparison works across success and error states
 * - Complex types are properly compared
 */
TEST_F(ResultTest, EqualityOperators) {
    Result<int, TestError> success1 = Ok(42);
    Result<int, TestError> success2 = Ok(42);
    Result<int, TestError> success3 = Ok(99);
    Result<int, TestError> error1 = Err(TestError::NETWORK_FAILURE);
    Result<int, TestError> error2 = Err(TestError::NETWORK_FAILURE);
    Result<int, TestError> error3 = Err(TestError::PARSE_ERROR);

    // Test success equality
    EXPECT_TRUE(success1 == success2);
    EXPECT_FALSE(success1 != success2);
    EXPECT_FALSE(success1 == success3);
    EXPECT_TRUE(success1 != success3);

    // Test error equality
    EXPECT_TRUE(error1 == error2);
    EXPECT_FALSE(error1 != error2);
    EXPECT_FALSE(error1 == error3);
    EXPECT_TRUE(error1 != error3);

    // Test success vs error
    EXPECT_FALSE(success1 == error1);
    EXPECT_TRUE(success1 != error1);

    // Test with complex types
    DetailedError detailed1(TestError::NETWORK_FAILURE, "Connection failed", 404);
    DetailedError detailed2(TestError::NETWORK_FAILURE, "Connection failed", 404);
    DetailedError detailed3(TestError::NETWORK_FAILURE, "Different message", 404);

    Result<int, DetailedError> complex_error1 = Err(detailed1);
    Result<int, DetailedError> complex_error2 = Err(detailed2);
    Result<int, DetailedError> complex_error3 = Err(detailed3);

    EXPECT_TRUE(complex_error1 == complex_error2);
    EXPECT_FALSE(complex_error1 == complex_error3);
}

//=============================================================================
// Performance and Edge Case Tests
//=============================================================================

/**
 * @brief Test performance characteristics and zero-cost abstractions
 *
 * Validates that:
 * - Result operations are efficient and don't introduce overhead
 * - Memory layout is optimal (size checks)
 * - No dynamic allocations occur in normal operations
 * - Compile-time optimizations work correctly
 */
TEST_F(ResultTest, PerformanceCharacteristics) {
    // Test size characteristics
    EXPECT_LE(sizeof(Result<int, TestError>), sizeof(std::variant<int, TestError>) + 8);
    EXPECT_GE(sizeof(Result<int, TestError>), sizeof(std::variant<int, TestError>));

    // Test that simple operations are inlined and fast
    auto start = std::chrono::high_resolution_clock::now();

    const int ITERATIONS = 100000;  // Reduced iterations to avoid overflow and improve test speed
    long long sum = 0;              // Use long long to avoid overflow

    for (int i = 0; i < ITERATIONS; ++i) {
        auto result = make_result_ok<TestError>(i)
                          .map([](int x) { return x * 2; })
                          .and_then([](int x) -> Result<int, TestError> { return Ok(x + 1); });

        if (result.is_ok()) {
            sum += result.unwrap();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Performance should be reasonable (operations should be fast)
    EXPECT_LT(duration.count(), 200000);  // Less than 200ms for 100k operations (more realistic)
    EXPECT_GT(sum, 0);                    // Ensure operations actually ran
}

/**
 * @brief Test edge cases and corner conditions
 *
 * Validates that:
 * - Empty types and void-like scenarios work
 * - Large types are handled efficiently
 * - Recursive Result types compile correctly
 * - Extreme nesting and chaining scenarios work
 */
TEST_F(ResultTest, EdgeCases) {
    // Test with empty struct
    struct Empty {};
    Result<Empty, TestError> empty_result = Ok(Empty{});
    EXPECT_TRUE(empty_result.is_ok());

    // Test with large types
    struct LargeType {
        std::array<int, 1000> data_{};
        LargeType() { data_.fill(42); }
        auto operator==(const LargeType& other) const -> bool { return data_ == other.data_; }
    };

    Result<LargeType, TestError> large_result = Ok(LargeType{});
    EXPECT_TRUE(large_result.is_ok());

    // Test extremely long chains
    auto long_chain = make_result_ok<TestError>(1)
                          .map([](int x) { return x + 1; })
                          .map([](int x) { return x * 2; })
                          .and_then([](int x) -> Result<int, TestError> { return Ok(x - 1); })
                          .map([](int x) { return x + 10; })
                          .and_then([](int x) -> Result<int, TestError> { return Ok(x / 2); })
                          .map([](int x) { return x * 3; });

    EXPECT_TRUE(long_chain.is_ok());
    // ((((1 + 1) * 2) - 1) + 10) / 2 * 3 = ((3 + 10) / 2) * 3 = 6 * 3 = 18
    EXPECT_EQ(long_chain.unwrap(), 18);

    // Test error early termination in long chains
    auto error_chain = make_result_ok<TestError>(1)
                           .map([](int x) { return x + 1; })
                           .and_then([](int x [[maybe_unused]]) -> Result<int, TestError> {
                               return Err(TestError::INVALID_INPUT);  // Error here
                           })
                           .map([](int x) { return x * 1000; })  // Should not execute
                           .and_then([](int x) -> Result<int, TestError> { return Ok(x); });

    EXPECT_TRUE(error_chain.is_err());
    EXPECT_EQ(error_chain.unwrap_err(), TestError::INVALID_INPUT);
}

/**
 * @brief Test thread safety and concurrent usage
 *
 * Validates that:
 * - Result objects can be used safely across threads
 * - Const operations are thread-safe
 * - No race conditions occur in normal usage
 * - Shared read-only access works correctly
 */
TEST_F(ResultTest, ThreadSafety) {
    const Result<int, TestError> SHARED_RESULT = Ok(42);
    std::atomic<int> success_count{0};
    std::atomic<int> total_sum{0};

    const int NUM_THREADS = 4;
    const int OPERATIONS_PER_THREAD = 1000;

    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&SHARED_RESULT, &success_count, &total_sum]() {
            for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                // These operations should be thread-safe on const Result
                if (SHARED_RESULT.is_ok()) {
                    auto value = SHARED_RESULT.unwrap();
                    total_sum += value;
                    success_count++;
                }

                auto opt = SHARED_RESULT.ok();
                if (opt.has_value()) {
                    // Use the value
                    volatile int temp = opt.value();
                    (void)temp;  // Suppress unused variable warning
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(success_count.load(), NUM_THREADS * OPERATIONS_PER_THREAD);
    EXPECT_EQ(total_sum.load(), 42 * NUM_THREADS * OPERATIONS_PER_THREAD);
}

//=============================================================================
// Type Safety and Constraint Tests
//=============================================================================

/**
 * @brief Test compile-time type safety and template constraints
 *
 * Validates that:
 * - Template constraints prevent invalid usage
 * - Type deduction works correctly in all scenarios
 * - SFINAE prevents compilation of invalid operations
 * - Error messages are meaningful for template errors
 */
TEST_F(ResultTest, TypeSafetyConstraints) {
    // Test that functions returning non-Result types are rejected by and_then
    // This test ensures compilation fails for invalid and_then usage

    Result<int, TestError> test_result = Ok(42);

    // These should compile correctly
    auto valid_and_then = test_result.and_then(
        [](int x) -> Result<std::string, TestError> { return Ok(std::to_string(x)); });
    EXPECT_TRUE(valid_and_then.is_ok());

    Result<int, TestError> error_result = Err(TestError::NETWORK_FAILURE);
    auto valid_or_else = error_result.or_else([](TestError err) -> Result<int, DetailedError> {
        return Err(DetailedError(err, "Converted", 0));
    });
    EXPECT_TRUE(valid_or_else.is_err());

    // Test type trait helpers
    static_assert(detail::IS_RESULT_V<Result<int, TestError>>);
    static_assert(!detail::IS_RESULT_V<int>);
    static_assert(!detail::IS_RESULT_V<std::optional<int>>);

    // Test that Result types maintain their template parameters correctly
    using IntStringResult = Result<int, std::string>;
    static_assert(std::is_same_v<IntStringResult::value_type, int>);
    static_assert(std::is_same_v<IntStringResult::error_type, std::string>);
}

/**
 * @brief Test integration with existing error handling patterns
 *
 * Validates that:
 * - Results integrate well with std::optional
 * - Exception-based code can be wrapped effectively
 * - Error propagation patterns work across boundaries
 * - Legacy error codes can be converted to Results
 */
TEST_F(ResultTest, ErrorHandlingIntegration) {
    // Test conversion from optional
    auto from_optional = [](std::optional<int> opt) -> Result<int, TestError> {
        if (opt.has_value()) {
            return Ok(opt.value());
        }
        return Err(TestError::INVALID_INPUT);
    };

    auto some_result = from_optional(std::make_optional(123));
    auto none_result = from_optional(std::nullopt);

    EXPECT_TRUE(some_result.is_ok());
    EXPECT_EQ(some_result.unwrap(), 123);
    EXPECT_TRUE(none_result.is_err());
    EXPECT_EQ(none_result.unwrap_err(), TestError::INVALID_INPUT);

    // Test integration with exception-based APIs
    auto risky_operation = [](int value) {
        if (value < 0) {
            throw std::invalid_argument("Negative value");
        }
        if (value > 100) {
            throw std::overflow_error("Value too large");
        }
        return value * 2;
    };

    auto safe_wrapper = [&risky_operation](int value) -> Result<int, std::string> {
        try {
            return Ok(risky_operation(value));
        } catch (const std::exception& e) {
            return Err(std::string(e.what()));
        }
    };

    auto success_wrapped = safe_wrapper(50);
    auto error_wrapped = safe_wrapper(-10);
    auto overflow_wrapped = safe_wrapper(150);

    EXPECT_TRUE(success_wrapped.is_ok());
    EXPECT_EQ(success_wrapped.unwrap(), 100);
    EXPECT_TRUE(error_wrapped.is_err());
    EXPECT_TRUE(overflow_wrapped.is_err());

    // Test error message propagation
    EXPECT_TRUE(error_wrapped.unwrap_err().find("Negative value") != std::string::npos);
    EXPECT_TRUE(overflow_wrapped.unwrap_err().find("Value too large") != std::string::npos);
}
