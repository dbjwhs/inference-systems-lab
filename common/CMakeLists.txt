# MIT License
# Copyright (c) 2025 dbjwhs
#
# This software is provided "as is" without warranty of any kind, express or implied.
# The authors are not liable for any damages arising from the use of this software.

cmake_minimum_required(VERSION 3.20)
project(common)

# Common utilities and abstractions library

# Find optional dependencies for configuration management
find_package(PkgConfig QUIET)
find_package(yaml-cpp QUIET)
find_package(nlohmann_json QUIET)

if(PKG_CONFIG_FOUND AND NOT TARGET yaml-cpp)
    pkg_check_modules(YAML_CPP yaml-cpp)
endif()

if(PKG_CONFIG_FOUND AND NOT TARGET nlohmann_json::nlohmann_json)
    pkg_check_modules(NLOHMANN_JSON nlohmann_json)
endif()

# Create common library
add_library(common STATIC
    src/logging.cpp
    src/utils.cpp
    src/inference_types.cpp
    src/inference_builders.cpp
    src/schema_evolution.cpp
)

# Always include a simple configuration loader
target_sources(common PRIVATE src/config_loader_simple.cpp)
message(STATUS "Simple configuration management enabled")

# Optionally add full config_loader if yaml-cpp is available (currently disabled)
# TODO: Fix namespace and compilation issues with full config_loader
# if(TARGET yaml-cpp OR YAML_CPP_FOUND OR yaml-cpp_FOUND)
#     target_sources(common PRIVATE src/config_loader.cpp)
#     
#     if(TARGET yaml-cpp)
#         target_link_libraries(common yaml-cpp)
#         message(STATUS "Using yaml-cpp target")
#     elseif(YAML_CPP_FOUND)
#         target_link_libraries(common ${YAML_CPP_LIBRARIES})
#         target_include_directories(common PRIVATE ${YAML_CPP_INCLUDE_DIRS})
#         message(STATUS "Using yaml-cpp pkg-config")
#     endif()
#     
#     if(TARGET nlohmann_json::nlohmann_json)
#         target_link_libraries(common nlohmann_json::nlohmann_json)
#         message(STATUS "Using nlohmann_json target")
#     elseif(NLOHMANN_JSON_FOUND)
#         target_link_libraries(common ${NLOHMANN_JSON_LIBRARIES})
#         target_include_directories(common PRIVATE ${NLOHMANN_JSON_INCLUDE_DIRS})
#         message(STATUS "Using nlohmann_json pkg-config")
#     endif()
#     
#     target_compile_definitions(common PRIVATE YAML_CONFIG_AVAILABLE)
#     message(STATUS "Full configuration management enabled with YAML support")
# else()
    message(STATUS "Full configuration management disabled - using simple config loader only")
    message(STATUS "To enable full YAML support: fix compilation issues in config_loader.cpp")
# endif()

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

# Apply strict warnings to our common library
apply_strict_warnings(common)

# Demo executable
add_executable(demo_logging
    examples/demo_logging.cpp
)

target_link_libraries(demo_logging
    common
)
apply_strict_warnings(demo_logging)

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
apply_strict_warnings(logging_tests)

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
apply_strict_warnings(result_tests)

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
apply_strict_warnings(serialization_tests)

# Add Serialization tests to CTest
add_test(NAME SerializationUnitTests COMMAND serialization_tests)

# Container tests
add_executable(container_tests
    tests/test_containers.cpp
)

# Link atomic library conditionally (not needed on macOS with modern Clang)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR 
   (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT APPLE))
    set(ATOMIC_LIBRARY atomic)
else()
    set(ATOMIC_LIBRARY)
endif()

target_link_libraries(container_tests
    common
    ${GTEST_MAIN_TARGET}
    ${ATOMIC_LIBRARY}
)
apply_strict_warnings(container_tests)

# Add Container tests to CTest
add_test(NAME ContainerUnitTests COMMAND container_tests)

# Advanced containers tests (Phase 2)
add_executable(advanced_containers_tests
    tests/test_advanced_containers.cpp
)

target_link_libraries(advanced_containers_tests
    common
    ${GTEST_MAIN_TARGET}
    ${ATOMIC_LIBRARY}
)
apply_strict_warnings(advanced_containers_tests)

# Add advanced containers tests to CTest
add_test(NAME AdvancedContainersTests COMMAND advanced_containers_tests)

# Type system tests (Phase 2)
add_executable(type_system_tests
    tests/test_type_system.cpp
)

target_link_libraries(type_system_tests
    common
    ${GTEST_MAIN_TARGET}
    ${ATOMIC_LIBRARY}
)
apply_strict_warnings(type_system_tests)

# Add type system tests to CTest
add_test(NAME TypeSystemTests COMMAND type_system_tests)

# ML logging tests (Phase 4)
add_executable(ml_logging_tests
    tests/test_ml_logging.cpp
)

target_link_libraries(ml_logging_tests
    common
    ${GTEST_MAIN_TARGET}
)
apply_strict_warnings(ml_logging_tests)

# Add ML logging tests to CTest
add_test(NAME MLLoggingTests COMMAND ml_logging_tests)

# Simple configuration loader tests (always available)
add_executable(config_simple_tests
    tests/test_config_simple.cpp
)

target_link_libraries(config_simple_tests
    common
    ${GTEST_MAIN_TARGET}
)
apply_strict_warnings(config_simple_tests)

# Add simple configuration tests to CTest
add_test(NAME SimpleConfigurationTests COMMAND config_simple_tests)

# Inference builders tests
add_executable(inference_builders_tests
    tests/test_inference_builders.cpp
)

target_link_libraries(inference_builders_tests
    common
    ${GTEST_MAIN_TARGET}
)
apply_strict_warnings(inference_builders_tests)

# Add inference builders tests to CTest
add_test(NAME InferenceBuildersTests COMMAND inference_builders_tests)

# Full configuration loader tests (currently disabled)
# TODO: Enable when full config_loader is fixed
# if(TARGET yaml-cpp OR YAML_CPP_FOUND OR yaml-cpp_FOUND)
#     add_executable(config_loader_tests
#         tests/test_config_loader.cpp
#     )

#     target_link_libraries(config_loader_tests
#         common
#         ${GTEST_MAIN_TARGET}
#     )
#     apply_strict_warnings(config_loader_tests)

#     # Add configuration tests to CTest
#     add_test(NAME ConfigurationLoaderTests COMMAND config_loader_tests)
# endif()

# ML types tests - temporarily disabled due to C++20 designated initializer issues
# TODO: Fix field order and aggregate type compatibility issues in test_ml_types.cpp
# add_executable(ml_types_tests
#     tests/test_ml_types.cpp
# )
# 
# target_link_libraries(ml_types_tests
#     common
#     ${GTEST_MAIN_TARGET}
# )
# apply_strict_warnings(ml_types_tests)
# 
# # Add ML types tests to CTest
# add_test(NAME MLTypesUnitTests COMMAND ml_types_tests)

# Inference types demo
add_executable(inference_types_demo
    examples/inference_types_demo.cpp
)

target_link_libraries(inference_types_demo
    common
)
apply_strict_warnings(inference_types_demo)

# Schema evolution demo
add_executable(schema_evolution_demo
    examples/schema_evolution_demo.cpp
)

target_link_libraries(schema_evolution_demo
    common
)
apply_strict_warnings(schema_evolution_demo)

# Result usage examples
add_executable(result_usage_examples
    examples/result_usage_examples.cpp
)

target_link_libraries(result_usage_examples
    common
)
apply_strict_warnings(result_usage_examples)

# Result benchmarks (only if Google Benchmark is available)
if(TARGET benchmark::benchmark)
    add_executable(result_benchmarks
        benchmarks/result_benchmarks.cpp
    )

    target_link_libraries(result_benchmarks
        common
        benchmark::benchmark
    )

    # Container benchmarks
    add_executable(container_benchmarks
        benchmarks/container_benchmarks.cpp
    )

    target_link_libraries(container_benchmarks
        common
        benchmark::benchmark
        ${ATOMIC_LIBRARY}
    )
    apply_strict_warnings(container_benchmarks)

    # Create benchmark alias for the main benchmarks target
    add_custom_target(common_benchmarks DEPENDS result_benchmarks container_benchmarks)
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
        apply_strict_warnings(result_benchmarks)

        add_executable(container_benchmarks
            benchmarks/container_benchmarks.cpp
        )

        target_link_libraries(container_benchmarks
            common
            benchmark::benchmark
            ${ATOMIC_LIBRARY}
        )
        apply_strict_warnings(container_benchmarks)

        add_custom_target(common_benchmarks DEPENDS result_benchmarks container_benchmarks)
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
