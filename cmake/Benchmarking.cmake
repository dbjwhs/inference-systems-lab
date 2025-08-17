# Benchmarking.cmake
# Configures Google Benchmark integration for performance testing

function(configure_benchmarking)
    # Google Benchmark
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing")
    FetchContent_MakeAvailable(benchmark)
    
    # Benchmarks target
    add_custom_target(benchmarks
        DEPENDS 
            common_benchmarks
            engines_benchmarks
            distributed_benchmarks
            performance_benchmarks
            integration_benchmarks
        COMMENT "Building all benchmarks"
    )
    
    message(STATUS "Google Benchmark configured (v1.8.3)")
endfunction()

# Helper function to display benchmarking status in configuration summary
function(display_benchmarking_status)
    message(STATUS "  Google Benchmark: v1.8.3 (FetchContent)")
endfunction()