# Performance optimization library

# Create performance library
add_library(performance STATIC
    src/memory_pool.cpp
    src/cache_optimizer.cpp
    src/simd_operations.cpp
    src/profiler.cpp
)

target_include_directories(performance PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(performance
    common
    Threads::Threads
)

# Performance tests
add_executable(performance_tests
    tests/test_memory_pool.cpp
    tests/test_cache_optimizer.cpp
    tests/test_simd_operations.cpp
    tests/test_profiler.cpp
)

target_link_libraries(performance_tests
    performance
    gtest_main
    gtest
)

add_test(NAME PerformanceTests COMMAND performance_tests)

# Performance benchmarks
add_executable(performance_benchmarks
    benchmarks/bench_memory_allocation.cpp
    benchmarks/bench_cache_performance.cpp
    benchmarks/bench_simd_vs_scalar.cpp
    benchmarks/bench_profiler_overhead.cpp
)

target_link_libraries(performance_benchmarks
    performance
    benchmark::benchmark
)