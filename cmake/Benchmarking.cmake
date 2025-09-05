# Benchmarking.cmake
# Configures Google Benchmark integration for performance testing
#
# KNOWN ISSUE - macOS Apple Silicon Warning Suppression:
# ✅ Identified: macOS-specific Google Benchmark warnings on Apple Silicon Macs
# ✅ Root Cause: Google Benchmark trying to use Intel-style hw.cpufrequency sysctl that doesn't exist on Apple Silicon  
# ✅ Solution: Created run_benchmarks_clean.sh wrapper that filters harmless warnings
# ✅ Impact: Benchmarks work perfectly, warnings are cosmetic only
#
# The warnings that appear on macOS are:
# - "Unable to determine clock rate from sysctl: hw.cpufrequency: No such file or directory"
# - "***WARNING*** Failed to set thread affinity. Estimated CPU frequency may be incorrect."
# - "This does not affect benchmark measurements, only the metadata output."
#
# These are expected on Apple Silicon and do not affect benchmark functionality.
# Use run_benchmarks_clean.sh wrapper to filter these warnings for cleaner output.

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
