// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file result_benchmarks.cpp
 * @brief Performance benchmarks for the Result<T, E> error handling type
 *
 * This file contains comprehensive performance benchmarks for the Result<T, E>
 * implementation to validate zero-cost abstraction claims and compare against
 * alternative error handling approaches. Benchmarks include:
 * - Construction and destruction overhead
 * - Value access and extraction performance
 * - Monadic operations (map, and_then, or_else) efficiency
 * - Memory layout and cache performance characteristics
 * - Comparison with exception-based and raw error code approaches
 * - Real-world usage pattern performance simulation
 *
 * The benchmarks use Google Benchmark framework for accurate measurement
 * and statistical analysis of performance characteristics.
 */

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "../src/result.hpp"

using namespace inference_lab::common;

//=============================================================================
// Benchmark Configuration and Setup
//=============================================================================

// Common error type for benchmarks
enum class BenchError : std::uint8_t {
    INVALID_INPUT,
    PROCESSING_FAILED,
    RESOURCE_EXHAUSTED,
    NETWORK_TIMEOUT,
    UNKNOWN
};

// Large data structure for testing copy/move performance
struct LargeData {
    std::vector<double> values_{};
    std::string metadata_{};

    explicit LargeData(size_t size = 1000)
        : values(size, 3.14159), metadata("benchmark_data_" + std::to_string(size)) {}

    // Make it expensive to copy
    LargeData(const LargeData& other) : values(other.values), metadata(other.metadata) {}
    LargeData(LargeData&& other) noexcept
        : values(std::move(other.values)), metadata(std::move(other.metadata)) {}

    auto operator=(const LargeData& other) -> LargeData& {
        values = other.values;
        metadata = other.metadata;
        return *this;
    }

    auto operator=(LargeData&& other) noexcept -> LargeData& {
        values = std::move(other.values);
        metadata = std::move(other.metadata);
        return *this;
    }
};

// Global random number generator for consistent benchmarks
static thread_local std::mt19937 g_rng{std::random_device{}()};

//=============================================================================
// Basic Construction and Destruction Benchmarks
//=============================================================================

/**
 * @brief Benchmark Result construction with small types
 */
static void bm_result_construction_small_success(benchmark::State& state) {
    for (auto _ : state) {
        Result<int, BenchError> result = Ok(42);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Result_Construction_Small_Success);

static void bm_result_construction_small_error(benchmark::State& state) {
    for (auto _ : state) {
        Result<int, BenchError> result = Err(BenchError::INVALID_INPUT);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Result_Construction_Small_Error);

/**
 * @brief Benchmark Result construction with large types
 */
static void bm_result_construction_large_success(benchmark::State& state) {
    for (auto _ : state) {
        Result<LargeData, BenchError> result = Ok(LargeData(state.range(0)));
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Result_Construction_Large_Success)->Range(100, 10000);

static void bm_result_construction_large_error(benchmark::State& state) {
    for (auto _ : state) {
        Result<LargeData, BenchError> result = Err(BenchError::PROCESSING_FAILED);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Result_Construction_Large_Error);

/**
 * @brief Benchmark move construction performance
 */
static void bm_result_move_construction(benchmark::State& state) {
    for (auto _ : state) {
        Result<LargeData, BenchError> source = Ok(LargeData(1000));
        Result<LargeData, BenchError> moved = std::move(source);
        benchmark::DoNotOptimize(moved);
    }
}
BENCHMARK(BM_Result_Move_Construction);

//=============================================================================
// Value Access Benchmarks
//=============================================================================

/**
 * @brief Benchmark safe value access methods
 */
static void bm_result_is_ok_check(benchmark::State& state) {
    Result<int, BenchError> result = Ok(42);
    for (auto _ : state) {
        bool ok = result.is_ok();
        benchmark::DoNotOptimize(ok);
    }
}
BENCHMARK(BM_Result_IsOk_Check);

static void bm_result_unwrap_success(benchmark::State& state) {
    Result<int, BenchError> result = Ok(42);
    for (auto _ : state) {
        int value = result.unwrap();
        benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_Result_Unwrap_Success);

static void bm_result_unwrap_or_success(benchmark::State& state) {
    Result<int, BenchError> result = Ok(42);
    for (auto _ : state) {
        int value = result.unwrap_or(99);
        benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_Result_UnwrapOr_Success);

static void bm_result_unwrap_or_error(benchmark::State& state) {
    Result<int, BenchError> result = Err(BenchError::INVALID_INPUT);
    for (auto _ : state) {
        int value = result.unwrap_or(99);
        benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_Result_UnwrapOr_Error);

static void bm_result_unwrap_or_else_success(benchmark::State& state) {
    Result<int, BenchError> result = Ok(42);
    auto fallback = [](BenchError) { return 99; };

    for (auto _ : state) {
        int value = result.unwrap_or_else(fallback);
        benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_Result_UnwrapOrElse_Success);

static void bm_result_unwrap_or_else_error(benchmark::State& state) {
    Result<int, BenchError> result = Err(BenchError::INVALID_INPUT);
    auto fallback = [](BenchError) { return 99; };

    for (auto _ : state) {
        int value = result.unwrap_or_else(fallback);
        benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_Result_UnwrapOrElse_Error);

//=============================================================================
// Monadic Operation Benchmarks
//=============================================================================

/**
 * @brief Benchmark map operation performance
 */
static void bm_result_map_success(benchmark::State& state) {
    Result<int, BenchError> result = Ok(42);
    auto mapper = [](int x) { return x * 2; };

    for (auto _ : state) {
        auto mapped = result.map(mapper);
        benchmark::DoNotOptimize(mapped);
    }
}
BENCHMARK(BM_Result_Map_Success);

static void bm_result_map_error(benchmark::State& state) {
    Result<int, BenchError> result = Err(BenchError::INVALID_INPUT);
    auto mapper = [](int x) { return x * 2; };

    for (auto _ : state) {
        auto mapped = result.map(mapper);
        benchmark::DoNotOptimize(mapped);
    }
}
BENCHMARK(BM_Result_Map_Error);

/**
 * @brief Benchmark chained map operations
 */
static void bm_result_map_chain(benchmark::State& state) {
    Result<int, BenchError> result = Ok(10);

    for (auto _ : state) {
        auto chained =
            result.map([](int x) { return x + 1; }).map([](int x) { return x * 2; }).map([](int x) {
                return x - 5;
            });
        benchmark::DoNotOptimize(chained);
    }
}
BENCHMARK(BM_Result_Map_Chain);

/**
 * @brief Benchmark and_then operation performance
 */
static void bm_result_and_then_success(benchmark::State& state) {
    Result<int, BenchError> result = Ok(42);
    auto operation = [](int x) -> Result<int, BenchError> { return Ok(x * 2); };

    for (auto _ : state) {
        auto chained = result.and_then(operation);
        benchmark::DoNotOptimize(chained);
    }
}
BENCHMARK(BM_Result_AndThen_Success);

static void bm_result_and_then_error(benchmark::State& state) {
    Result<int, BenchError> result = Err(BenchError::INVALID_INPUT);
    auto operation = [](int x) -> Result<int, BenchError> { return Ok(x * 2); };

    for (auto _ : state) {
        auto chained = result.and_then(operation);
        benchmark::DoNotOptimize(chained);
    }
}
BENCHMARK(BM_Result_AndThen_Error);

/**
 * @brief Benchmark complex monadic chains
 */
static void bm_result_complex_chain(benchmark::State& state) {
    Result<int, BenchError> result = Ok(5);

    for (auto _ : state) {
        auto complex = result.map([](int x) { return x + 10; })
                           .and_then([](int x) -> Result<int, BenchError> {
                               if (x > 20) {
                                   return Err(BenchError::PROCESSING_FAILED);
                               }
                               return Ok(x);
                           })
                           .map([](int x) { return x * 3; })
                           .or_else([](BenchError) -> Result<int, BenchError> { return Ok(100); });
        benchmark::DoNotOptimize(complex);
    }
}
BENCHMARK(BM_Result_Complex_Chain);

//=============================================================================
// Comparison with Alternative Error Handling
//=============================================================================

/**
 * @brief Benchmark raw error code approach for comparison
 */
struct ErrorCodeResult {
    int value_;
    int error_code_;  // 0 = success, negative = error

    auto is_ok() const -> bool { return error_code_ == 0; }
    auto unwrap() const -> int { return value_; }
};

static auto error_code_operation(int input) -> ErrorCodeResult {
    if (input < 0) {
        return {0, -1};  // Error
    }
    return {input * 2, 0};  // Success
}

static void bm_error_code_operation(benchmark::State& state) {
    for (auto _ : state) {
        auto result = error_code_operation(42);
        if (result.is_ok()) {
            int const value = result.unwrap();
            benchmark::DoNotOptimize(value);
        }
    }
}
BENCHMARK(BM_ErrorCode_Operation);

/**
 * @brief Benchmark exception-based approach for comparison
 */
static auto exception_operation(int input) -> int {
    if (input < 0) {
        throw std::invalid_argument("Negative input");
    }
    return input * 2;
}

static void bm_exception_operation_success(benchmark::State& state) {
    for (auto _ : state) {
        try {
            int const value = exception_operation(42);
            benchmark::DoNotOptimize(value);
        } catch (const std::exception&) {
            // Handle error
        }
    }
}
BENCHMARK(BM_Exception_Operation_Success);

static void bm_exception_operation_error(benchmark::State& state) {
    for (auto _ : state) {
        try {
            int const value = exception_operation(-1);
            benchmark::DoNotOptimize(value);
        } catch (const std::exception&) {
            benchmark::DoNotOptimize(100);  // Default value
        }
    }
}
BENCHMARK(BM_Exception_Operation_Error);

/**
 * @brief Compare Result approach with alternatives
 */
static auto result_operation(int input) -> Result<int, BenchError> {
    if (input < 0) {
        return Err(BenchError::INVALID_INPUT);
    }
    return Ok(input * 2);
}

static void bm_result_operation_success(benchmark::State& state) {
    for (auto _ : state) {
        auto result = result_operation(42);
        if (result.is_ok()) {
            int value = result.unwrap();
            benchmark::DoNotOptimize(value);
        }
    }
}
BENCHMARK(BM_Result_Operation_Success);

static void bm_result_operation_error(benchmark::State& state) {
    for (auto _ : state) {
        auto result = result_operation(-1);
        int value = result.unwrap_or(100);
        benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_Result_Operation_Error);

//=============================================================================
// Memory and Cache Performance Benchmarks
//=============================================================================

/**
 * @brief Benchmark memory access patterns with Results
 */
static void bm_result_vector_processing(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<Result<int, BenchError>> results;
    results.reserve(size);

    // Populate with mix of success and error results
    std::uniform_int_distribution<int> dist(1, 100);
    for (size_t i = 0; i < size; ++i) {
        int value = dist(g_rng);
        if (value > 90) {
            results.emplace_back(Err(BenchError::PROCESSING_FAILED));
        } else {
            results.emplace_back(Ok(value));
        }
    }

    for (auto _ : state) {
        int const sum = 0;
        for (const auto& result : results) {
            sum += result.unwrap_or(0);
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_Result_Vector_Processing)->Range(1000, 100000);

/**
 * @brief Benchmark cache-friendly vs cache-unfriendly access patterns
 */
static void bm_result_sequential_access(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<Result<LargeData, BenchError>> results;
    results.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        results.emplace_back(Ok(LargeData(10)));  // Small data for this test
    }

    for (auto _ : state) {
        size_t total_size = 0;
        for (const auto& result : results) {
            if (result.is_ok()) {
                total_size += result.unwrap().values.size();
            }
        }
        benchmark::DoNotOptimize(total_size);
    }

    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_Result_Sequential_Access)->Range(1000, 10000);

//=============================================================================
// Real-World Usage Pattern Simulations
//=============================================================================

/**
 * @brief Simulate parsing pipeline with multiple stages
 */
static auto parse_stage1(const std::string& input) -> Result<int, BenchError> {
    if (input.empty())
        return Err(BenchError::INVALID_INPUT);
    return Ok(static_cast<int>(input.length()));
}

static auto parse_stage2(int length) -> Result<double, BenchError> {
    if (length > 1000)
        return Err(BenchError::RESOURCE_EXHAUSTED);
    return Ok(static_cast<double>(length) * 1.5);
}

static auto parse_stage3(double value) -> Result<std::string, BenchError> {
    if (value < 0)
        return Err(BenchError::PROCESSING_FAILED);
    return Ok("processed_" + std::to_string(value));
}

static void bm_result_parsing_pipeline(benchmark::State& state) {
    std::vector<std::string> inputs;
    std::uniform_int_distribution<int> size_dist(1, 50);

    // Generate test inputs
    for (int i = 0; i < 1000; ++i) {
        int size = size_dist(g_rng);
        inputs.emplace_back(size, 'a' + (i % 26));
    }

    for (auto _ : state) {
        int const successful = 0;
        for (const auto& input : inputs) {
            auto result = parse_stage1(input).and_then(parse_stage2).and_then(parse_stage3);

            if (result.is_ok()) {
                successful++;
            }
        }
        benchmark::DoNotOptimize(successful);
    }

    state.SetItemsProcessed(state.iterations() * inputs.size());
}
BENCHMARK(BM_Result_Parsing_Pipeline);

/**
 * @brief Simulate network request batching
 */
static auto simulate_network_request(int request_id) -> Result<std::string, BenchError> {
    // Simulate various outcomes based on request ID
    if (request_id % 20 == 0)
        return Err(BenchError::NETWORK_TIMEOUT);
    if (request_id % 17 == 0)
        return Err(BenchError::RESOURCE_EXHAUSTED);
    return Ok("response_" + std::to_string(request_id));
}

static void bm_result_network_batch(benchmark::State& state) {
    size_t batch_size = state.range(0);

    for (auto _ : state) {
        std::vector<Result<std::string, BenchError>> responses;
        responses.reserve(batch_size);

        // Simulate batch requests
        for (size_t i = 0; i < batch_size; ++i) {
            responses.push_back(simulate_network_request(static_cast<int>(i)));
        }

        // Process responses
        int const successful = 0;
        size_t total_data = 0;
        for (const auto& response : responses) {
            if (response.is_ok()) {
                successful++;
                total_data += response.unwrap().length();
            }
        }

        benchmark::DoNotOptimize(successful);
        benchmark::DoNotOptimize(total_data);
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_Result_Network_Batch)->Range(10, 1000);

//=============================================================================
// Memory Overhead Analysis
//=============================================================================

/**
 * @brief Benchmark to measure memory overhead
 */
static void bm_result_memory_overhead(benchmark::State& state) {
    for (auto _ : state) {
        // Measure actual memory allocation patterns
        std::vector<Result<int, BenchError>> results;
        results.reserve(10000);

        for (int i = 0; i < 10000; ++i) {
            if (i % 10 == 0) {
                results.emplace_back(Err(BenchError::PROCESSING_FAILED));
            } else {
                results.emplace_back(Ok(i));
            }
        }

        benchmark::DoNotOptimize(results);
    }
}
BENCHMARK(BM_Result_Memory_Overhead);

//=============================================================================
// Custom Main for Benchmark Output
//=============================================================================

auto main(int argc, char** argv) -> int {
    std::cout << "Result<T, E> Performance Benchmarks" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Measuring performance characteristics of Result<T, E> implementation"
              << std::endl;
    std::cout << "Comparing with alternative error handling approaches" << std::endl;
    std::cout << "All measurements include compiler optimizations (-O2/-O3)" << std::endl;
    std::cout << std::endl;

    // Display system information
    std::cout << "System Information:" << std::endl;
    std::cout << "- Result<int, BenchError> size: " << sizeof(Result<int, BenchError>) << " bytes"
              << std::endl;
    std::cout << "- std::variant<int, BenchError> size: " << sizeof(std::variant<int, BenchError>)
              << " bytes" << std::endl;
    std::cout << "- Raw int size: " << sizeof(int) << " bytes" << std::endl;
    std::cout << "- Raw pointer size: " << sizeof(void*) << " bytes" << std::endl;
    std::cout << std::endl;

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}
