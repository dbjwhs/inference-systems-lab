// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file result.hpp
 * @brief Modern C++17 Result<T, E> type for robust error handling without exceptions
 * 
 * This file provides a comprehensive Result type that serves as an alternative to exception-based
 * error handling. The Result<T, E> type can hold either a successful value of type T or an error
 * of type E, providing a type-safe way to handle operations that may fail.
 * 
 * Key features:
 * - Type-safe error handling without exceptions
 * - Monadic operations (map, and_then, or_else) for functional composition
 * - RAII-compliant resource management with proper move semantics
 * - Zero-cost abstractions with optimized storage using std::variant
 * - Comprehensive conversion utilities for interoperability
 * - Thread-safe operations (const methods are thread-safe)
 * - Structured binding support for modern C++ usage patterns
 * 
 * Design Philosophy:
 * This implementation follows the Rust Result<T, E> model adapted for C++17.
 * It encourages explicit error handling and makes error propagation visible
 * in function signatures, leading to more robust and maintainable code.
 * 
 * Performance Characteristics:
 * - Size: sizeof(Result<T, E>) = sizeof(std::variant<T, E>) + alignment
 * - Time: O(1) for all operations, no dynamic allocation
 * - Memory: Stack-allocated, no heap overhead
 * 
 * Example Usage:
 * @code
 * enum class MathError { DivisionByZero, Overflow };
 * 
 * auto divide(int a, int b) -> Result<int, MathError> {
 *     if (b == 0) return Err(MathError::DivisionByZero);
 *     return Ok(a / b);
 * }
 * 
 * auto result = divide(10, 2)
 *     .map([](int x) { return x * 2; })
 *     .and_then([](int x) { return divide(x, 5); });
 * 
 * if (result.is_ok()) {
 *     std::cout << "Result: " << result.unwrap() << std::endl;
 * }
 * @endcode
 */

#pragma once

#include <variant>
#include <functional>
#include <type_traits>
#include <utility>
#include <optional>
#include <string>
#include <stdexcept>

namespace inference_lab::common {

// Forward declarations
template<typename T, typename E>
class Result;

template<typename T>
class Ok;

template<typename E>
class Err;

// Helper concepts for SFINAE and better error messages
namespace detail {
    
    /**
     * @brief Concept to check if a type is a Result type
     */
    template<typename T>
    struct is_result : std::false_type {};
    
    template<typename T, typename E>
    struct is_result<Result<T, E>> : std::true_type {};
    
    template<typename T>
    inline constexpr bool is_result_v = is_result<T>::value;
    
    /**
     * @brief Concept to check if a callable returns a Result type
     */
    template<typename F, typename T>
    using invoke_result_t = std::invoke_result_t<F, T>;
    
    template<typename F, typename T>
    inline constexpr bool returns_result_v = is_result_v<invoke_result_t<F, T>>;
    
    /**
     * @brief Helper to extract the value type from a Result return type
     */
    template<typename R>
    struct result_value_type;
    
    template<typename T, typename E>
    struct result_value_type<Result<T, E>> {
        using type = T;
    };
    
    template<typename R>
    using result_value_type_t = typename result_value_type<R>::type;
    
    /**
     * @brief Helper to extract the error type from a Result return type
     */
    template<typename R>
    struct result_error_type;
    
    template<typename T, typename E>
    struct result_error_type<Result<T, E>> {
        using type = E;
    };
    
    template<typename R>
    using result_error_type_t = typename result_error_type<R>::type;

} // namespace detail

/**
 * @class Ok
 * @brief Wrapper class for successful Result values
 * 
 * This class is used to explicitly construct successful Result instances.
 * It provides type deduction and perfect forwarding for efficient construction.
 * 
 * @tparam T The type of the successful value
 */
template<typename T>
class Ok {
public:
    /**
     * @brief Construct an Ok wrapper with the given value
     * @param value The successful value to wrap
     */
    explicit constexpr Ok(T value) : value_(std::move(value)) {}
    
    /**
     * @brief Get the wrapped value
     * @return Reference to the wrapped value
     */
    constexpr const T& value() const& { return value_; }
    
    /**
     * @brief Get the wrapped value (move version)
     * @return Moved wrapped value
     */
    constexpr T&& value() && { return std::move(value_); }

private:
    T value_;
    
    template<typename U, typename V>
    friend class Result;
};

/**
 * @class Err
 * @brief Wrapper class for error Result values
 * 
 * This class is used to explicitly construct error Result instances.
 * It provides type deduction and perfect forwarding for efficient construction.
 * 
 * @tparam E The type of the error value
 */
template<typename E>
class Err {
public:
    /**
     * @brief Construct an Err wrapper with the given error
     * @param error The error value to wrap
     */
    explicit constexpr Err(E error) : error_(std::move(error)) {}
    
    /**
     * @brief Get the wrapped error
     * @return Reference to the wrapped error
     */
    constexpr const E& error() const& { return error_; }
    
    /**
     * @brief Get the wrapped error (move version)
     * @return Moved wrapped error
     */
    constexpr E&& error() && { return std::move(error_); }

private:
    E error_;
    
    template<typename U, typename V>
    friend class Result;
};

/**
 * @class Result
 * @brief A type-safe union that represents either a successful value or an error
 * 
 * Result<T, E> is a type that can hold either a value of type T (representing success)
 * or a value of type E (representing an error). This provides a robust alternative
 * to exception-based error handling with compile-time guarantees.
 * 
 * The Result type supports monadic operations that allow for functional composition
 * and chaining of operations that may fail, making error handling both explicit
 * and composable.
 * 
 * @tparam T The type of the successful value
 * @tparam E The type of the error value
 */
// Internal discriminated wrappers to avoid std::variant ambiguity when T == E
namespace detail {
    template<typename T>
    struct ValueWrapper {
        T value;
        template<typename U>
        constexpr ValueWrapper(U&& v) : value(std::forward<U>(v)) {}
        
        constexpr bool operator==(const ValueWrapper& other) const {
            return value == other.value;
        }
        
        constexpr bool operator!=(const ValueWrapper& other) const {
            return !(*this == other);
        }
    };
    
    template<typename E>
    struct ErrorWrapper {
        E error;
        template<typename U>
        constexpr ErrorWrapper(U&& e) : error(std::forward<U>(e)) {}
        
        constexpr bool operator==(const ErrorWrapper& other) const {
            return error == other.error;
        }
        
        constexpr bool operator!=(const ErrorWrapper& other) const {
            return !(*this == other);
        }
    };
}

template<typename T, typename E>
class Result {
public:
    using value_type = T;  ///< Type of the successful value
    using error_type = E;  ///< Type of the error value
    
    // Constructors
    
    /**
     * @brief Construct a successful Result from an Ok wrapper
     * @param ok The Ok wrapper containing the successful value
     */
    constexpr Result(Ok<T> ok) : data_(detail::ValueWrapper<T>{std::move(ok.value_)}) {}
    
    /**
     * @brief Construct an error Result from an Err wrapper
     * @param err The Err wrapper containing the error value
     */
    constexpr Result(Err<E> err) : data_(detail::ErrorWrapper<E>{std::move(err.error_)}) {}
    
    /**
     * @brief Copy constructor
     * @param other The Result to copy from
     */
    constexpr Result(const Result& other) = default;
    
    /**
     * @brief Move constructor
     * @param other The Result to move from
     */
    constexpr Result(Result&& other) noexcept = default;
    
    /**
     * @brief Copy assignment operator
     * @param other The Result to copy from
     * @return Reference to this Result
     */
    constexpr Result& operator=(const Result& other) = default;
    
    /**
     * @brief Move assignment operator
     * @param other The Result to move from
     * @return Reference to this Result
     */
    constexpr Result& operator=(Result&& other) noexcept = default;
    
    /**
     * @brief Destructor
     */
    ~Result() = default;
    
    // State checking methods
    
    /**
     * @brief Check if this Result contains a successful value
     * @return true if this Result contains a value, false if it contains an error
     */
    constexpr bool is_ok() const noexcept {
        return std::holds_alternative<detail::ValueWrapper<T>>(data_);
    }
    
    /**
     * @brief Check if this Result contains an error
     * @return true if this Result contains an error, false if it contains a value
     */
    constexpr bool is_err() const noexcept {
        return std::holds_alternative<detail::ErrorWrapper<E>>(data_);
    }
    
    /**
     * @brief Boolean conversion operator (same as is_ok())
     * @return true if this Result contains a value
     */
    constexpr explicit operator bool() const noexcept {
        return is_ok();
    }
    
    // Value access methods
    
    /**
     * @brief Get the successful value, throwing if this Result contains an error
     * @return Reference to the successful value
     * @throws std::runtime_error if this Result contains an error
     */
    constexpr const T& unwrap() const& {
        if (is_err()) {
            throw std::runtime_error("Called unwrap() on an error Result");
        }
        return std::get<detail::ValueWrapper<T>>(data_).value;
    }
    
    /**
     * @brief Get the successful value (move version), throwing if this Result contains an error
     * @return Moved successful value
     * @throws std::runtime_error if this Result contains an error
     */
    constexpr T&& unwrap() && {
        if (is_err()) {
            throw std::runtime_error("Called unwrap() on an error Result");
        }
        return std::move(std::get<detail::ValueWrapper<T>>(std::move(data_)).value);
    }
    
    /**
     * @brief Get the error value, throwing if this Result contains a successful value
     * @return Reference to the error value
     * @throws std::runtime_error if this Result contains a successful value
     */
    constexpr const E& unwrap_err() const& {
        if (is_ok()) {
            throw std::runtime_error("Called unwrap_err() on a successful Result");
        }
        return std::get<detail::ErrorWrapper<E>>(data_).error;
    }
    
    /**
     * @brief Get the error value (move version), throwing if this Result contains a successful value
     * @return Moved error value
     * @throws std::runtime_error if this Result contains a successful value
     */
    constexpr E&& unwrap_err() && {
        if (is_ok()) {
            throw std::runtime_error("Called unwrap_err() on a successful Result");
        }
        return std::move(std::get<detail::ErrorWrapper<E>>(std::move(data_)).error);
    }
    
    /**
     * @brief Get the successful value or return a default value
     * @param default_value The value to return if this Result contains an error
     * @return The successful value or the default value
     */
    constexpr T unwrap_or(const T& default_value) const& {
        return is_ok() ? std::get<detail::ValueWrapper<T>>(data_).value : default_value;
    }
    
    /**
     * @brief Get the successful value or return a default value (move version)
     * @param default_value The value to return if this Result contains an error
     * @return The successful value or the default value
     */
    constexpr T unwrap_or(T&& default_value) && {
        return is_ok() ? std::move(std::get<detail::ValueWrapper<T>>(std::move(data_)).value) : std::move(default_value);
    }
    
    /**
     * @brief Get the successful value or compute one from the error
     * @param f Function that takes the error and returns a value of type T
     * @return The successful value or the computed value
     */
    template<typename F>
    constexpr T unwrap_or_else(F&& f) const& {
        static_assert(std::is_invocable_r_v<T, F, const E&>, 
                     "Function must be callable with const E& and return T");
        return is_ok() ? std::get<detail::ValueWrapper<T>>(data_).value : std::invoke(std::forward<F>(f), std::get<detail::ErrorWrapper<E>>(data_).error);
    }
    
    /**
     * @brief Get the successful value or compute one from the error (move version)
     * @param f Function that takes the error and returns a value of type T
     * @return The successful value or the computed value
     */
    template<typename F>
    constexpr T unwrap_or_else(F&& f) && {
        static_assert(std::is_invocable_r_v<T, F, E&&>, 
                     "Function must be callable with E&& and return T");
        return is_ok() ? std::move(std::get<detail::ValueWrapper<T>>(std::move(data_)).value) : std::invoke(std::forward<F>(f), std::move(std::get<detail::ErrorWrapper<E>>(std::move(data_)).error));
    }
    
    // Optional conversion methods
    
    /**
     * @brief Convert this Result to an optional, discarding error information
     * @return std::optional containing the value if successful, std::nullopt if error
     */
    constexpr std::optional<T> ok() const& {
        return is_ok() ? std::optional<T>(std::get<detail::ValueWrapper<T>>(data_).value) : std::nullopt;
    }
    
    /**
     * @brief Convert this Result to an optional (move version), discarding error information
     * @return std::optional containing the value if successful, std::nullopt if error
     */
    constexpr std::optional<T> ok() && {
        return is_ok() ? std::optional<T>(std::move(std::get<detail::ValueWrapper<T>>(std::move(data_)).value)) : std::nullopt;
    }
    
    /**
     * @brief Convert this Result to an optional error, discarding success information
     * @return std::optional containing the error if failed, std::nullopt if successful
     */
    constexpr std::optional<E> err() const& {
        return is_err() ? std::optional<E>(std::get<detail::ErrorWrapper<E>>(data_).error) : std::nullopt;
    }
    
    /**
     * @brief Convert this Result to an optional error (move version), discarding success information
     * @return std::optional containing the error if failed, std::nullopt if successful
     */
    constexpr std::optional<E> err() && {
        return is_err() ? std::optional<E>(std::move(std::get<detail::ErrorWrapper<E>>(std::move(data_)).error)) : std::nullopt;
    }
    
    // Monadic operations
    
    /**
     * @brief Transform the successful value using the given function
     * 
     * If this Result contains a successful value, apply the function to it and wrap
     * the result in a new Result. If this Result contains an error, return a new
     * Result with the same error.
     * 
     * @tparam F Function type that takes T and returns U
     * @param f Function to apply to the successful value
     * @return Result<U, E> with the transformed value or the original error
     */
    template<typename F>
    constexpr auto map(F&& f) const& -> Result<std::invoke_result_t<F, const T&>, E> {
        using U = std::invoke_result_t<F, const T&>;
        static_assert(std::is_invocable_v<F, const T&>, 
                     "Function must be callable with const T&");
        
        if (is_ok()) {
            return Result<U, E>(Ok<U>(std::invoke(std::forward<F>(f), std::get<detail::ValueWrapper<T>>(data_).value)));
        } else {
            return Result<U, E>(Err<E>(std::get<detail::ErrorWrapper<E>>(data_).error));
        }
    }
    
    /**
     * @brief Transform the successful value using the given function (move version)
     * 
     * @tparam F Function type that takes T&& and returns U
     * @param f Function to apply to the successful value
     * @return Result<U, E> with the transformed value or the original error
     */
    template<typename F>
    constexpr auto map(F&& f) && -> Result<std::invoke_result_t<F, T&&>, E> {
        using U = std::invoke_result_t<F, T&&>;
        static_assert(std::is_invocable_v<F, T&&>, 
                     "Function must be callable with T&&");
        
        if (is_ok()) {
            return Result<U, E>(Ok<U>(std::invoke(std::forward<F>(f), std::move(std::get<detail::ValueWrapper<T>>(std::move(data_)).value))));
        } else {
            return Result<U, E>(Err<E>(std::move(std::get<detail::ErrorWrapper<E>>(std::move(data_)).error)));
        }
    }
    
    /**
     * @brief Transform the error using the given function
     * 
     * If this Result contains an error, apply the function to it and wrap the result
     * in a new Result. If this Result contains a successful value, return a new Result
     * with the same value.
     * 
     * @tparam F Function type that takes E and returns U
     * @param f Function to apply to the error
     * @return Result<T, U> with the original value or the transformed error
     */
    template<typename F>
    constexpr auto map_err(F&& f) const& -> Result<T, std::invoke_result_t<F, const E&>> {
        using U = std::invoke_result_t<F, const E&>;
        static_assert(std::is_invocable_v<F, const E&>, 
                     "Function must be callable with const E&");
        
        if (is_ok()) {
            return Result<T, U>(Ok<T>(std::get<detail::ValueWrapper<T>>(data_).value));
        } else {
            return Result<T, U>(Err<U>(std::invoke(std::forward<F>(f), std::get<detail::ErrorWrapper<E>>(data_).error)));
        }
    }
    
    /**
     * @brief Transform the error using the given function (move version)
     * 
     * @tparam F Function type that takes E&& and returns U
     * @param f Function to apply to the error
     * @return Result<T, U> with the original value or the transformed error
     */
    template<typename F>
    constexpr auto map_err(F&& f) && -> Result<T, std::invoke_result_t<F, E&&>> {
        using U = std::invoke_result_t<F, E&&>;
        static_assert(std::is_invocable_v<F, E&&>, 
                     "Function must be callable with E&&");
        
        if (is_ok()) {
            return Result<T, U>(Ok<T>(std::move(std::get<detail::ValueWrapper<T>>(std::move(data_)).value)));
        } else {
            return Result<T, U>(Err<U>(std::invoke(std::forward<F>(f), std::move(std::get<detail::ErrorWrapper<E>>(std::move(data_)).error))));
        }
    }
    
    /**
     * @brief Chain this Result with another operation that returns a Result
     * 
     * If this Result contains a successful value, apply the function to it.
     * The function must return a Result<U, E>. If this Result contains an error,
     * return a new Result with the same error.
     * 
     * This is the monadic bind operation for Result types.
     * 
     * @tparam F Function type that takes T and returns Result<U, E>
     * @param f Function to apply to the successful value
     * @return Result<U, E> from the function or the original error
     */
    template<typename F>
    constexpr auto and_then(F&& f) const& -> std::invoke_result_t<F, const T&> {
        using result_type = std::invoke_result_t<F, const T&>;
        static_assert(detail::is_result_v<result_type>, 
                     "Function must return a Result type");
        static_assert(std::is_same_v<E, detail::result_error_type_t<result_type>>, 
                     "Function must return a Result with the same error type");
        static_assert(std::is_invocable_v<F, const T&>, 
                     "Function must be callable with const T&");
        
        if (is_ok()) {
            return std::invoke(std::forward<F>(f), std::get<detail::ValueWrapper<T>>(data_).value);
        } else {
            using U = detail::result_value_type_t<result_type>;
            return result_type(Err<E>(std::get<detail::ErrorWrapper<E>>(data_).error));
        }
    }
    
    /**
     * @brief Chain this Result with another operation that returns a Result (move version)
     * 
     * @tparam F Function type that takes T&& and returns Result<U, E>
     * @param f Function to apply to the successful value
     * @return Result<U, E> from the function or the original error
     */
    template<typename F>
    constexpr auto and_then(F&& f) && -> std::invoke_result_t<F, T&&> {
        using result_type = std::invoke_result_t<F, T&&>;
        static_assert(detail::is_result_v<result_type>, 
                     "Function must return a Result type");
        static_assert(std::is_same_v<E, detail::result_error_type_t<result_type>>, 
                     "Function must return a Result with the same error type");
        static_assert(std::is_invocable_v<F, T&&>, 
                     "Function must be callable with T&&");
        
        if (is_ok()) {
            return std::invoke(std::forward<F>(f), std::move(std::get<detail::ValueWrapper<T>>(std::move(data_)).value));
        } else {
            using U = detail::result_value_type_t<result_type>;
            return result_type(Err<E>(std::move(std::get<detail::ErrorWrapper<E>>(std::move(data_)).error)));
        }
    }
    
    /**
     * @brief Provide an alternative Result if this one contains an error
     * 
     * If this Result contains an error, apply the function to it. The function must
     * return a Result<T, F>. If this Result contains a successful value, return
     * a new Result with the same value.
     * 
     * @tparam F Function type that takes E and returns Result<T, F>
     * @param f Function to apply to the error
     * @return This Result if successful, or the Result from the function
     */
    template<typename F>
    constexpr auto or_else(F&& f) const& -> std::invoke_result_t<F, const E&> {
        using result_type = std::invoke_result_t<F, const E&>;
        static_assert(detail::is_result_v<result_type>, 
                     "Function must return a Result type");
        static_assert(std::is_same_v<T, detail::result_value_type_t<result_type>>, 
                     "Function must return a Result with the same value type");
        static_assert(std::is_invocable_v<F, const E&>, 
                     "Function must be callable with const E&");
        
        if (is_ok()) {
            using F_error = detail::result_error_type_t<result_type>;
            return result_type(Ok<T>(std::get<detail::ValueWrapper<T>>(data_).value));
        } else {
            return std::invoke(std::forward<F>(f), std::get<detail::ErrorWrapper<E>>(data_).error);
        }
    }
    
    /**
     * @brief Provide an alternative Result if this one contains an error (move version)
     * 
     * @tparam F Function type that takes E&& and returns Result<T, F>
     * @param f Function to apply to the error
     * @return This Result if successful, or the Result from the function
     */
    template<typename F>
    constexpr auto or_else(F&& f) && -> std::invoke_result_t<F, E&&> {
        using result_type = std::invoke_result_t<F, E&&>;
        static_assert(detail::is_result_v<result_type>, 
                     "Function must return a Result type");
        static_assert(std::is_same_v<T, detail::result_value_type_t<result_type>>, 
                     "Function must return a Result with the same value type");
        static_assert(std::is_invocable_v<F, E&&>, 
                     "Function must be callable with E&&");
        
        if (is_ok()) {
            using F_error = detail::result_error_type_t<result_type>;
            return result_type(Ok<T>(std::move(std::get<detail::ValueWrapper<T>>(std::move(data_)).value)));
        } else {
            return std::invoke(std::forward<F>(f), std::move(std::get<detail::ErrorWrapper<E>>(std::move(data_)).error));
        }
    }
    
    // Structured binding support
    
    /**
     * @brief Support for structured bindings
     * @return Tuple-like decomposition for structured binding
     */
    template<std::size_t N>
    constexpr decltype(auto) get() const& {
        if constexpr (N == 0) {
            return is_ok();
        } else if constexpr (N == 1) {
            return ok();
        } else if constexpr (N == 2) {
            return err();
        } else {
            static_assert(N < 3, "Result only supports 3-element structured binding");
        }
    }
    
    /**
     * @brief Support for structured bindings (move version)
     * @return Tuple-like decomposition for structured binding
     */
    template<std::size_t N>
    constexpr decltype(auto) get() && {
        if constexpr (N == 0) {
            return is_ok();
        } else if constexpr (N == 1) {
            return std::move(*this).ok();
        } else if constexpr (N == 2) {
            return std::move(*this).err();
        } else {
            static_assert(N < 3, "Result only supports 3-element structured binding");
        }
    }
    
    // Equality operators
    
    /**
     * @brief Equality comparison with another Result
     * @param other The Result to compare with
     * @return true if both Results have the same state and equal values/errors
     */
    constexpr bool operator==(const Result& other) const {
        return data_ == other.data_;
    }
    
    /**
     * @brief Inequality comparison with another Result
     * @param other The Result to compare with
     * @return true if Results differ in state or values/errors
     */
    constexpr bool operator!=(const Result& other) const {
        return !(*this == other);
    }

private:
    std::variant<detail::ValueWrapper<T>, detail::ErrorWrapper<E>> data_;  ///< Internal storage using discriminated wrappers
};

// Convenience factory functions

/**
 * @brief Create a successful Result with type deduction
 * @tparam T The type of the value (deduced)
 * @param value The successful value
 * @return Ok wrapper for constructing Result
 */
template<typename T>
constexpr auto make_ok(T&& value) {
    return Ok<std::decay_t<T>>(std::forward<T>(value));
}

/**
 * @brief Create an error Result with type deduction
 * @tparam E The type of the error (deduced)
 * @param error The error value
 * @return Err wrapper for constructing Result
 */
template<typename E>
constexpr auto make_err(E&& error) {
    return Err<std::decay_t<E>>(std::forward<E>(error));
}

/**
 * @brief Create a successful Result with explicit error type
 * @tparam E The error type to use
 * @tparam T The type of the value (deduced)
 * @param value The successful value
 * @return Result<T, E> containing the value
 */
template<typename E, typename T>
constexpr auto make_result_ok(T&& value) -> Result<std::decay_t<T>, E> {
    return Result<std::decay_t<T>, E>(Ok<std::decay_t<T>>(std::forward<T>(value)));
}

/**
 * @brief Create an error Result with explicit value type
 * @tparam T The value type to use
 * @tparam E The type of the error (deduced)
 * @param error The error value
 * @return Result<T, E> containing the error
 */
template<typename T, typename E>
constexpr auto make_result_err(E&& error) -> Result<T, std::decay_t<E>> {
    return Result<T, std::decay_t<E>>(Err<std::decay_t<E>>(std::forward<E>(error)));
}

/**
 * @brief Try to execute a function and wrap exceptions in Result
 * 
 * This utility function allows integration with exception-throwing code
 * by catching exceptions and converting them to Result errors.
 * 
 * @tparam F Function type
 * @tparam E Error type (defaults to std::exception)
 * @param f Function to execute
 * @return Result containing the function result or caught exception
 */
template<typename E = std::exception, typename F>
auto try_call(F&& f) -> Result<std::conditional_t<std::is_void_v<std::invoke_result_t<F>>, std::monostate, std::invoke_result_t<F>>, E> {
    using T = std::invoke_result_t<F>;
    using ReturnType = std::conditional_t<std::is_void_v<T>, std::monostate, T>;
    
    try {
        if constexpr (std::is_void_v<T>) {
            std::invoke(std::forward<F>(f));
            return Result<ReturnType, E>(Ok<ReturnType>(std::monostate{}));
        } else {
            return Result<ReturnType, E>(Ok<ReturnType>(std::invoke(std::forward<F>(f))));
        }
    } catch (const E& e) {
        return Result<ReturnType, E>(Err<E>(e));
    }
}

/**
 * @brief Combine multiple Results into a single Result containing a tuple
 * 
 * If all input Results are successful, returns a Result containing a tuple
 * of all the successful values. If any Result contains an error, returns
 * the first error encountered.
 * 
 * @tparam Results Parameter pack of Result types
 * @param results The Results to combine
 * @return Result containing tuple of values or first error
 */
template<typename... Results>
constexpr auto combine(Results&&... results) {
    // Extract value and error types
    using value_tuple = std::tuple<typename std::decay_t<Results>::value_type...>;
    using first_error_type = typename std::tuple_element_t<0, std::tuple<std::decay_t<Results>...>>::error_type;
    
    // Check if all Results have the same error type
    static_assert(((std::is_same_v<first_error_type, typename std::decay_t<Results>::error_type>) && ...),
                 "All Results must have the same error type");
    
    // Check if any Result contains an error
    if (!(results.is_ok() && ...)) {
        // Find and return the first error using a helper function
        auto get_first_error = [](auto&&... args) -> Result<value_tuple, first_error_type> {
            Result<value_tuple, first_error_type> result{Err<first_error_type>(first_error_type{})};
            ((args.is_err() ? (result = Result<value_tuple, first_error_type>(Err<first_error_type>(args.unwrap_err())), true) : false) || ...);
            return result;
        };
        return get_first_error(results...);
    }
    
    // All Results are successful, extract values into tuple
    return Result<value_tuple, first_error_type>(Ok<value_tuple>(std::make_tuple(std::forward<Results>(results).unwrap()...)));
}

} // namespace inference_lab::common

// std::tuple_size and std::tuple_element specializations for structured binding support
namespace std {
    template<typename T, typename E>
    struct tuple_size<inference_lab::common::Result<T, E>> : std::integral_constant<std::size_t, 3> {};
    
    template<typename T, typename E>
    struct tuple_element<0, inference_lab::common::Result<T, E>> {
        using type = bool;
    };
    
    template<typename T, typename E>
    struct tuple_element<1, inference_lab::common::Result<T, E>> {
        using type = std::optional<T>;
    };
    
    template<typename T, typename E>
    struct tuple_element<2, inference_lab::common::Result<T, E>> {
        using type = std::optional<E>;
    };
}