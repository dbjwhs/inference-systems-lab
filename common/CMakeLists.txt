# MIT License
# Copyright (c) 2025 dbjwhs
#
# This software is provided "as is" without warranty of any kind, express or implied.
# The authors are not liable for any damages arising from the use of this software.

cmake_minimum_required(VERSION 3.20)
project(common)

# Common utilities and abstractions library

# Create common library
add_library(common STATIC
    src/logging.cpp
    src/utils.cpp
    src/inference_types.cpp
    src/inference_builders.cpp
    src/schema_evolution.cpp
)

target_include_directories(common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>  # For generated Cap'n Proto headers
    $<INSTALL_INTERFACE:include>
)

# Set C++17 standard (consistent with main project)
target_compile_features(common PUBLIC cxx_std_17)

# Find dependencies
find_package(Threads REQUIRED)
find_package(CapnProto REQUIRED)

# Generate C++ code from Cap'n Proto schemas
capnp_generate_cpp(CAPNP_SRCS CAPNP_HDRS schemas/inference_types.capnp)

# Add generated sources to the library
target_sources(common PRIVATE ${CAPNP_SRCS})

target_link_libraries(common
    Threads::Threads
    CapnProto::capnp
)

# Demo executable
add_executable(demo_logging
    examples/demo_logging.cpp
)

target_link_libraries(demo_logging
    common
)

# Enable testing
enable_testing()

# Google Test is provided via system package or FetchContent from root CMakeLists.txt
# Use system GTest targets if available, otherwise use FetchContent targets
if(TARGET GTest::gtest_main)
    set(GTEST_MAIN_TARGET GTest::gtest_main)
    set(GTEST_TARGET GTest::gtest)
else()
    set(GTEST_MAIN_TARGET gtest_main)
    set(GTEST_TARGET gtest)
endif()

# Unit tests
add_executable(logging_tests
    tests/test_logging_unit.cpp
)

target_link_libraries(logging_tests
    common
    ${GTEST_MAIN_TARGET}
)

# Add test to CTest
add_test(NAME LoggingUnitTests COMMAND logging_tests)

# Result type tests
add_executable(result_tests
    tests/test_result.cpp
)

target_link_libraries(result_tests
    common
    ${GTEST_MAIN_TARGET}
)

# Add Result tests to CTest
add_test(NAME ResultUnitTests COMMAND result_tests)

# Serialization tests
add_executable(serialization_tests
    tests/test_serialization.cpp
)

target_link_libraries(serialization_tests
    common
    ${GTEST_MAIN_TARGET}
)

# Add Serialization tests to CTest
add_test(NAME SerializationUnitTests COMMAND serialization_tests)

# Inference types demo
add_executable(inference_types_demo
    examples/inference_types_demo.cpp
)

target_link_libraries(inference_types_demo
    common
)

# Schema evolution demo
add_executable(schema_evolution_demo
    examples/schema_evolution_demo.cpp
)

target_link_libraries(schema_evolution_demo
    common
)

# Result usage examples
add_executable(result_usage_examples
    examples/result_usage_examples.cpp
)

target_link_libraries(result_usage_examples
    common
)

# Result benchmarks (only if Google Benchmark is available)
if(TARGET benchmark::benchmark)
    add_executable(result_benchmarks
        benchmarks/result_benchmarks.cpp
    )

    target_link_libraries(result_benchmarks
        common
        benchmark::benchmark
    )

    # Create benchmark alias for the main benchmarks target
    add_custom_target(common_benchmarks DEPENDS result_benchmarks)
else()
    # If building standalone, try to find benchmark package
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        add_executable(result_benchmarks
            benchmarks/result_benchmarks.cpp
        )

        target_link_libraries(result_benchmarks
            common
            benchmark::benchmark
        )

        add_custom_target(common_benchmarks DEPENDS result_benchmarks)
    else()
        message(STATUS "Google Benchmark not found - benchmarks will not be built")
        message(STATUS "To build benchmarks, either:")
        message(STATUS "  1. Build from main project directory, or")
        message(STATUS "  2. Install Google Benchmark system-wide")
        
        # Create empty benchmark target for compatibility
        add_custom_target(common_benchmarks
            COMMAND ${CMAKE_COMMAND} -E echo "Benchmarks not available - Google Benchmark not found"
        )
    endif()
endif()